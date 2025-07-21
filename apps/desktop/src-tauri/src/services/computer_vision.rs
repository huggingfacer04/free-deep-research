use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error};
use uuid::Uuid;
use chrono::Utc;

use crate::error::{AppResult, ApiError};
use crate::services::{Service, ApiManagerService, DataPersistenceService, MonitoringService};
use crate::models::computer_vision::*;
use crate::models::api_key::ServiceProvider;
use crate::services::api_manager::service_integration::ServiceRequest;

/// Computer Vision Service for Phase 5.1
pub struct ComputerVisionService {
    api_manager: Arc<RwLock<ApiManagerService>>,
    data_persistence: Arc<RwLock<DataPersistenceService>>,
    monitoring: Arc<RwLock<MonitoringService>>,
    active_jobs: Arc<RwLock<HashMap<Uuid, VisionProcessingJob>>>,
    cost_tracker: Arc<RwLock<VisionCostTracker>>,
}

/// Cost tracking for computer vision operations
#[derive(Debug, Clone)]
pub struct VisionCostTracker {
    pub daily_cost_cents: u32,
    pub monthly_cost_cents: u32,
    pub daily_limit_cents: u32,
    pub monthly_limit_cents: u32,
    pub cost_per_request: HashMap<VisionProvider, u32>, // Cost in cents
}

impl ComputerVisionService {
    /// Create a new computer vision service
    pub async fn new(
        api_manager: Arc<RwLock<ApiManagerService>>,
        data_persistence: Arc<RwLock<DataPersistenceService>>,
        monitoring: Arc<RwLock<MonitoringService>>,
    ) -> AppResult<Self> {
        info!("Initializing Computer Vision Service");

        let cost_tracker = VisionCostTracker {
            daily_cost_cents: 0,
            monthly_cost_cents: 0,
            daily_limit_cents: 1000, // $10 daily limit
            monthly_limit_cents: 10000, // $100 monthly limit
            cost_per_request: HashMap::from([
                (VisionProvider::GoogleVision, 150), // $0.015 per request
                (VisionProvider::AWSRekognition, 100), // $0.01 per request
                (VisionProvider::AzureComputerVision, 100), // $0.01 per request
            ]),
        };

        Ok(Self {
            api_manager,
            data_persistence,
            monitoring,
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            cost_tracker: Arc::new(RwLock::new(cost_tracker)),
        })
    }

    /// Process an image with computer vision
    pub async fn process_image(
        &self,
        user_id: Uuid,
        request: VisionProcessingRequest,
    ) -> AppResult<VisionProcessingResponse> {
        info!("Processing image with computer vision for user: {}", user_id);

        // Validate request
        if request.image_url.is_none() && request.image_data.is_none() {
            return Err(ApiError::invalid_request(
                "Either image_url or image_data must be provided".to_string()
            ).into());
        }

        if request.processing_types.is_empty() {
            return Err(ApiError::invalid_request(
                "At least one processing type must be specified".to_string()
            ).into());
        }

        // Select provider based on request or use cost optimization
        let provider = if let Some(provider) = request.provider {
            provider
        } else {
            self.select_optimal_provider(&request.processing_types).await?
        };

        // Check cost limits
        self.check_cost_limits(&provider).await?;

        // Create processing job
        let job = VisionProcessingJob::new(
            user_id,
            request.image_url.clone(),
            request.image_data.clone(),
            request.processing_types[0].clone(), // Use first processing type as primary
            provider.clone(),
        );

        let job_id = job.id;

        // Store job
        {
            let mut active_jobs = self.active_jobs.write().await;
            active_jobs.insert(job_id, job);
        }

        // Process asynchronously
        let service_clone = Arc::new(self.clone());
        let request_clone = request.clone();
        tokio::spawn(async move {
            if let Err(e) = service_clone.process_image_async(job_id, request_clone).await {
                error!("Failed to process image asynchronously: {}", e);
            }
        });

        Ok(VisionProcessingResponse {
            job_id,
            status: ProcessingStatus::Processing,
            results: None,
            processing_time_ms: None,
            cost_cents: None,
            provider_used: provider,
            error_message: None,
        })
    }

    /// Get processing job status
    pub async fn get_job_status(&self, job_id: Uuid) -> AppResult<Option<VisionProcessingJob>> {
        let active_jobs = self.active_jobs.read().await;
        Ok(active_jobs.get(&job_id).cloned())
    }

    /// Get user's processing history
    pub async fn get_user_history(
        &self,
        user_id: Uuid,
        limit: Option<u32>,
    ) -> AppResult<Vec<VisionProcessingJob>> {
        // In a real implementation, this would query the database
        // For now, return jobs from active jobs that belong to the user
        let active_jobs = self.active_jobs.read().await;
        let mut user_jobs: Vec<VisionProcessingJob> = active_jobs
            .values()
            .filter(|job| job.user_id == user_id)
            .cloned()
            .collect();

        user_jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        if let Some(limit) = limit {
            user_jobs.truncate(limit as usize);
        }

        Ok(user_jobs)
    }

    /// Get cost statistics
    pub async fn get_cost_stats(&self) -> AppResult<VisionCostTracker> {
        let cost_tracker = self.cost_tracker.read().await;
        Ok(cost_tracker.clone())
    }

    /// Select optimal provider based on processing types and cost
    async fn select_optimal_provider(
        &self,
        processing_types: &[VisionProcessingType],
    ) -> AppResult<VisionProvider> {
        // Simple provider selection logic
        // In production, this would consider cost, performance, and availability
        
        // Check which providers support all requested processing types
        let providers = vec![
            VisionProvider::GoogleVision,
            VisionProvider::AWSRekognition,
            VisionProvider::AzureComputerVision,
        ];

        for provider in providers {
            let supports_all = processing_types
                .iter()
                .all(|pt| provider.supports_processing_type(pt));
            
            if supports_all {
                // Check if provider has available API keys
                let service_provider = match provider {
                    VisionProvider::GoogleVision => ServiceProvider::GoogleVision,
                    VisionProvider::AWSRekognition => ServiceProvider::AWSRekognition,
                    VisionProvider::AzureComputerVision => ServiceProvider::AzureComputerVision,
                };

                let api_manager = self.api_manager.read().await;
                if let Ok(Some(_)) = api_manager.select_best_key_for_service(service_provider).await {
                    return Ok(provider);
                }
            }
        }

        Err(ApiError::resource_not_available(
            "No suitable computer vision provider available".to_string()
        ).into())
    }

    /// Check cost limits before processing
    async fn check_cost_limits(&self, provider: &VisionProvider) -> AppResult<()> {
        let cost_tracker = self.cost_tracker.read().await;
        let request_cost = cost_tracker.cost_per_request.get(provider).unwrap_or(&100);

        if cost_tracker.daily_cost_cents + request_cost > cost_tracker.daily_limit_cents {
            return Err(ApiError::rate_limit_exceeded(
                "Daily cost limit exceeded for computer vision processing".to_string()
            ).into());
        }

        if cost_tracker.monthly_cost_cents + request_cost > cost_tracker.monthly_limit_cents {
            return Err(ApiError::rate_limit_exceeded(
                "Monthly cost limit exceeded for computer vision processing".to_string()
            ).into());
        }

        Ok(())
    }

    /// Process image asynchronously
    async fn process_image_async(
        &self,
        job_id: Uuid,
        request: VisionProcessingRequest,
    ) -> AppResult<()> {
        let start_time = std::time::Instant::now();

        // Get job
        let mut job = {
            let active_jobs = self.active_jobs.read().await;
            active_jobs.get(&job_id).cloned()
                .ok_or_else(|| ApiError::resource_not_found("Processing job not found".to_string()))?
        };

        // Update job status to processing
        job.status = ProcessingStatus::Processing;
        {
            let mut active_jobs = self.active_jobs.write().await;
            active_jobs.insert(job_id, job.clone());
        }

        // Process with the selected provider
        let result = match job.provider {
            VisionProvider::GoogleVision => {
                self.process_with_google_vision(&request).await
            }
            VisionProvider::AWSRekognition => {
                self.process_with_aws_rekognition(&request).await
            }
            VisionProvider::AzureComputerVision => {
                self.process_with_azure_computer_vision(&request).await
            }
        };

        let processing_time = start_time.elapsed().as_millis() as u32;

        // Update job with results
        match result {
            Ok(results) => {
                let cost_cents = self.calculate_cost(&job.provider, &request.processing_types).await;
                job.mark_completed(results, processing_time, cost_cents);
                
                // Update cost tracker
                self.update_cost_tracker(&job.provider, cost_cents).await?;
            }
            Err(e) => {
                job.mark_failed(e.to_string());
            }
        }

        // Store updated job
        {
            let mut active_jobs = self.active_jobs.write().await;
            active_jobs.insert(job_id, job);
        }

        Ok(())
    }

    /// Process with Google Vision API
    async fn process_with_google_vision(
        &self,
        request: &VisionProcessingRequest,
    ) -> AppResult<VisionResults> {
        debug!("Processing with Google Vision API");

        let api_manager = self.api_manager.read().await;
        let api_key = api_manager
            .select_best_key_for_service(ServiceProvider::GoogleVision)
            .await?
            .ok_or_else(|| ApiError::key_not_found("No Google Vision API key available".to_string()))?;

        // Create Google Vision request
        let vision_request = ServiceRequest {
            endpoint: "/images:annotate".to_string(),
            method: "POST".to_string(),
            headers: HashMap::new(),
            body: self.build_google_vision_request(request)?,
            timeout_ms: Some(30000),
        };

        let response = api_manager
            .make_service_request(ServiceProvider::GoogleVision, vision_request)
            .await?;

        // Parse Google Vision response
        self.parse_google_vision_response(response.body)
    }

    /// Process with AWS Rekognition
    async fn process_with_aws_rekognition(
        &self,
        request: &VisionProcessingRequest,
    ) -> AppResult<VisionResults> {
        debug!("Processing with AWS Rekognition");

        let api_manager = self.api_manager.read().await;
        let api_key = api_manager
            .select_best_key_for_service(ServiceProvider::AWSRekognition)
            .await?
            .ok_or_else(|| ApiError::key_not_found("No AWS Rekognition API key available".to_string()))?;

        // For AWS Rekognition, we need to make separate requests for different processing types
        let mut results = VisionResults {
            labels: None,
            text_detections: None,
            faces: None,
            objects: None,
            safe_search: None,
            image_properties: None,
            crop_hints: None,
            web_detection: None,
            landmarks: None,
            logos: None,
            celebrities: None,
            moderation_labels: None,
            raw_response: None,
        };

        for processing_type in &request.processing_types {
            let endpoint = match processing_type {
                VisionProcessingType::LabelDetection => "/DetectLabels",
                VisionProcessingType::TextDetection => "/DetectText",
                VisionProcessingType::FaceDetection => "/DetectFaces",
                VisionProcessingType::CelebrityRecognition => "/RecognizeCelebrities",
                VisionProcessingType::ModerationLabels => "/DetectModerationLabels",
                _ => continue, // Skip unsupported types
            };

            let rekognition_request = ServiceRequest {
                endpoint: endpoint.to_string(),
                method: "POST".to_string(),
                headers: HashMap::new(),
                body: self.build_aws_rekognition_request(request, processing_type)?,
                timeout_ms: Some(30000),
            };

            let response = api_manager
                .make_service_request(ServiceProvider::AWSRekognition, rekognition_request)
                .await?;

            // Parse and merge results
            self.merge_aws_rekognition_response(&mut results, response.body, processing_type)?;
        }

        Ok(results)
    }

    /// Process with Azure Computer Vision
    async fn process_with_azure_computer_vision(
        &self,
        request: &VisionProcessingRequest,
    ) -> AppResult<VisionResults> {
        debug!("Processing with Azure Computer Vision");

        let api_manager = self.api_manager.read().await;
        let api_key = api_manager
            .select_best_key_for_service(ServiceProvider::AzureComputerVision)
            .await?
            .ok_or_else(|| ApiError::key_not_found("No Azure Computer Vision API key available".to_string()))?;

        // Create Azure Computer Vision request
        let azure_request = ServiceRequest {
            endpoint: "/analyze".to_string(),
            method: "POST".to_string(),
            headers: HashMap::new(),
            body: self.build_azure_computer_vision_request(request)?,
            timeout_ms: Some(30000),
        };

        let response = api_manager
            .make_service_request(ServiceProvider::AzureComputerVision, azure_request)
            .await?;

        // Parse Azure Computer Vision response
        self.parse_azure_computer_vision_response(response.body)
    }

    /// Build Google Vision API request
    fn build_google_vision_request(
        &self,
        request: &VisionProcessingRequest,
    ) -> AppResult<serde_json::Value> {
        // This is a simplified implementation
        // In production, you'd use the proper Google Vision API request structures
        Ok(serde_json::json!({
            "requests": [{
                "image": {
                    "content": request.image_data,
                    "source": request.image_url.as_ref().map(|url| serde_json::json!({"imageUri": url}))
                },
                "features": request.processing_types.iter().map(|pt| {
                    serde_json::json!({
                        "type": self.map_processing_type_to_google_feature(pt),
                        "maxResults": request.options.as_ref().and_then(|o| o.max_results).unwrap_or(10)
                    })
                }).collect::<Vec<_>>()
            }]
        }))
    }

    /// Build AWS Rekognition request
    fn build_aws_rekognition_request(
        &self,
        request: &VisionProcessingRequest,
        processing_type: &VisionProcessingType,
    ) -> AppResult<serde_json::Value> {
        let image = if let Some(image_data) = &request.image_data {
            serde_json::json!({ "Bytes": image_data })
        } else if let Some(image_url) = &request.image_url {
            // For AWS Rekognition, we'd need to download the image or use S3
            // This is a simplified implementation
            serde_json::json!({ "S3Object": { "Bucket": "temp", "Name": image_url } })
        } else {
            return Err(ApiError::invalid_request("No image provided".to_string()).into());
        };

        match processing_type {
            VisionProcessingType::LabelDetection => Ok(serde_json::json!({
                "Image": image,
                "MaxLabels": request.options.as_ref().and_then(|o| o.max_results).unwrap_or(10),
                "MinConfidence": request.options.as_ref().and_then(|o| o.min_confidence).unwrap_or(50.0)
            })),
            VisionProcessingType::TextDetection => Ok(serde_json::json!({
                "Image": image
            })),
            VisionProcessingType::FaceDetection => Ok(serde_json::json!({
                "Image": image,
                "Attributes": ["ALL"]
            })),
            _ => Ok(serde_json::json!({ "Image": image })),
        }
    }

    /// Build Azure Computer Vision request
    fn build_azure_computer_vision_request(
        &self,
        request: &VisionProcessingRequest,
    ) -> AppResult<serde_json::Value> {
        let visual_features: Vec<String> = request.processing_types
            .iter()
            .filter_map(|pt| self.map_processing_type_to_azure_feature(pt))
            .collect();

        if let Some(image_url) = &request.image_url {
            Ok(serde_json::json!({
                "url": image_url,
                "visualFeatures": visual_features,
                "language": request.options.as_ref().and_then(|o| o.language.as_ref()).unwrap_or(&"en".to_string())
            }))
        } else {
            // For binary data, we'd send the image data directly
            Ok(serde_json::json!({
                "visualFeatures": visual_features,
                "language": request.options.as_ref().and_then(|o| o.language.as_ref()).unwrap_or(&"en".to_string())
            }))
        }
    }

    /// Map processing type to Google Vision feature
    fn map_processing_type_to_google_feature(&self, processing_type: &VisionProcessingType) -> String {
        match processing_type {
            VisionProcessingType::LabelDetection => "LABEL_DETECTION".to_string(),
            VisionProcessingType::TextDetection => "TEXT_DETECTION".to_string(),
            VisionProcessingType::FaceDetection => "FACE_DETECTION".to_string(),
            VisionProcessingType::ObjectDetection => "OBJECT_LOCALIZATION".to_string(),
            VisionProcessingType::SafeSearchDetection => "SAFE_SEARCH_DETECTION".to_string(),
            VisionProcessingType::ImageProperties => "IMAGE_PROPERTIES".to_string(),
            VisionProcessingType::CropHints => "CROP_HINTS".to_string(),
            VisionProcessingType::WebDetection => "WEB_DETECTION".to_string(),
            VisionProcessingType::DocumentTextDetection => "DOCUMENT_TEXT_DETECTION".to_string(),
            VisionProcessingType::LandmarkDetection => "LANDMARK_DETECTION".to_string(),
            VisionProcessingType::LogoDetection => "LOGO_DETECTION".to_string(),
            _ => "LABEL_DETECTION".to_string(), // Default fallback
        }
    }

    /// Map processing type to Azure Computer Vision feature
    fn map_processing_type_to_azure_feature(&self, processing_type: &VisionProcessingType) -> Option<String> {
        match processing_type {
            VisionProcessingType::LabelDetection => Some("Tags".to_string()),
            VisionProcessingType::FaceDetection => Some("Faces".to_string()),
            VisionProcessingType::ObjectDetection => Some("Objects".to_string()),
            VisionProcessingType::ImageAnalysis => Some("Categories".to_string()),
            VisionProcessingType::ImageProperties => Some("Color".to_string()),
            _ => None,
        }
    }

    /// Parse Google Vision response (simplified)
    fn parse_google_vision_response(&self, response: serde_json::Value) -> AppResult<VisionResults> {
        // This is a simplified parser
        // In production, you'd properly parse the Google Vision API response
        Ok(VisionResults {
            labels: None,
            text_detections: None,
            faces: None,
            objects: None,
            safe_search: None,
            image_properties: None,
            crop_hints: None,
            web_detection: None,
            landmarks: None,
            logos: None,
            celebrities: None,
            moderation_labels: None,
            raw_response: Some(response),
        })
    }

    /// Merge AWS Rekognition response (simplified)
    fn merge_aws_rekognition_response(
        &self,
        results: &mut VisionResults,
        response: serde_json::Value,
        processing_type: &VisionProcessingType,
    ) -> AppResult<()> {
        // This is a simplified merger
        // In production, you'd properly parse and merge AWS Rekognition responses
        match processing_type {
            VisionProcessingType::LabelDetection => {
                // Parse labels from response
            }
            VisionProcessingType::TextDetection => {
                // Parse text detections from response
            }
            VisionProcessingType::FaceDetection => {
                // Parse face detections from response
            }
            _ => {}
        }
        Ok(())
    }

    /// Parse Azure Computer Vision response (simplified)
    fn parse_azure_computer_vision_response(&self, response: serde_json::Value) -> AppResult<VisionResults> {
        // This is a simplified parser
        // In production, you'd properly parse the Azure Computer Vision API response
        Ok(VisionResults {
            labels: None,
            text_detections: None,
            faces: None,
            objects: None,
            safe_search: None,
            image_properties: None,
            crop_hints: None,
            web_detection: None,
            landmarks: None,
            logos: None,
            celebrities: None,
            moderation_labels: None,
            raw_response: Some(response),
        })
    }

    /// Calculate cost for processing
    async fn calculate_cost(&self, provider: &VisionProvider, processing_types: &[VisionProcessingType]) -> u32 {
        let cost_tracker = self.cost_tracker.read().await;
        let base_cost = cost_tracker.cost_per_request.get(provider).unwrap_or(&100);
        
        // Multiply by number of processing types (simplified cost model)
        base_cost * processing_types.len() as u32
    }

    /// Update cost tracker
    async fn update_cost_tracker(&self, provider: &VisionProvider, cost_cents: u32) -> AppResult<()> {
        let mut cost_tracker = self.cost_tracker.write().await;
        cost_tracker.daily_cost_cents += cost_cents;
        cost_tracker.monthly_cost_cents += cost_cents;
        Ok(())
    }
}

// Implement Clone for the service (needed for async spawning)
impl Clone for ComputerVisionService {
    fn clone(&self) -> Self {
        Self {
            api_manager: Arc::clone(&self.api_manager),
            data_persistence: Arc::clone(&self.data_persistence),
            monitoring: Arc::clone(&self.monitoring),
            active_jobs: Arc::clone(&self.active_jobs),
            cost_tracker: Arc::clone(&self.cost_tracker),
        }
    }
}

#[async_trait::async_trait]
impl Service for ComputerVisionService {
    async fn start(&mut self) -> AppResult<()> {
        info!("Starting Computer Vision Service");
        Ok(())
    }

    async fn stop(&mut self) -> AppResult<()> {
        info!("Stopping Computer Vision Service");
        Ok(())
    }

    async fn health_check(&self) -> AppResult<bool> {
        // Check if we have at least one working computer vision API key
        let api_manager = self.api_manager.read().await;
        
        let providers = vec![
            ServiceProvider::GoogleVision,
            ServiceProvider::AWSRekognition,
            ServiceProvider::AzureComputerVision,
        ];

        for provider in providers {
            if let Ok(Some(_)) = api_manager.select_best_key_for_service(provider).await {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests;

// Re-export for convenience
pub use crate::services::computer_vision::VisionCostTracker;
