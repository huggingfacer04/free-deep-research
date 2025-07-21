use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info};
use chrono::Utc;

use crate::error::{AppResult, ApiError};
use crate::models::api_key::{ApiKey, ServiceProvider};
use crate::services::api_manager::service_integration::{
    ServiceIntegration, ServiceRequest, ServiceResponse, ServiceHealth, ServiceStatus, ServiceConfig
};

/// Azure Computer Vision integration
pub struct AzureComputerVisionIntegration {
    service_provider: ServiceProvider,
    config: ServiceConfig,
}

/// Azure Computer Vision request types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeImageRequest {
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visual_features: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description_exclude: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRRequest {
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detect_orientation: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadRequest {
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_mode: Option<String>,
}

/// Azure Computer Vision response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeImageResponse {
    pub categories: Option<Vec<Category>>,
    pub adult: Option<AdultInfo>,
    pub tags: Option<Vec<Tag>>,
    pub description: Option<Description>,
    pub faces: Option<Vec<Face>>,
    pub color: Option<ColorInfo>,
    pub image_type: Option<ImageType>,
    pub objects: Option<Vec<DetectedObject>>,
    pub brands: Option<Vec<Brand>>,
    pub request_id: Option<String>,
    pub metadata: Option<ImageMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Category {
    pub name: Option<String>,
    pub score: Option<f64>,
    pub detail: Option<CategoryDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryDetail {
    pub celebrities: Option<Vec<Celebrity>>,
    pub landmarks: Option<Vec<Landmark>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Celebrity {
    pub name: Option<String>,
    pub confidence: Option<f64>,
    pub face_rectangle: Option<FaceRectangle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmark {
    pub name: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdultInfo {
    pub is_adult_content: Option<bool>,
    pub is_racy_content: Option<bool>,
    pub is_gory_content: Option<bool>,
    pub adult_score: Option<f64>,
    pub racy_score: Option<f64>,
    pub gore_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub name: Option<String>,
    pub confidence: Option<f64>,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Description {
    pub tags: Option<Vec<String>>,
    pub captions: Option<Vec<Caption>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caption {
    pub text: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    pub age: Option<u32>,
    pub gender: Option<String>,
    pub face_rectangle: Option<FaceRectangle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRectangle {
    pub left: Option<u32>,
    pub top: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorInfo {
    pub dominant_color_foreground: Option<String>,
    pub dominant_color_background: Option<String>,
    pub dominant_colors: Option<Vec<String>>,
    pub accent_color: Option<String>,
    pub is_bw_img: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageType {
    pub clip_art_type: Option<u32>,
    pub line_drawing_type: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub rectangle: Option<ObjectRectangle>,
    pub object: Option<String>,
    pub confidence: Option<f64>,
    pub parent: Option<ObjectHierarchy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectRectangle {
    pub x: Option<u32>,
    pub y: Option<u32>,
    pub w: Option<u32>,
    pub h: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectHierarchy {
    pub object: Option<String>,
    pub confidence: Option<f64>,
    pub parent: Option<Box<ObjectHierarchy>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brand {
    pub name: Option<String>,
    pub confidence: Option<f64>,
    pub rectangle: Option<ObjectRectangle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRResponse {
    pub language: Option<String>,
    pub text_angle: Option<f64>,
    pub orientation: Option<String>,
    pub regions: Option<Vec<OCRRegion>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRRegion {
    pub bounding_box: Option<String>,
    pub lines: Option<Vec<OCRLine>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRLine {
    pub bounding_box: Option<String>,
    pub words: Option<Vec<OCRWord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRWord {
    pub bounding_box: Option<String>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResponse {
    pub status: Option<String>,
    pub created_date_time: Option<String>,
    pub last_updated_date_time: Option<String>,
    pub analyze_result: Option<AnalyzeResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeResult {
    pub version: Option<String>,
    pub model_version: Option<String>,
    pub read_results: Option<Vec<ReadResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadResult {
    pub page: Option<u32>,
    pub angle: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub unit: Option<String>,
    pub lines: Option<Vec<TextLine>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLine {
    pub bounding_box: Option<Vec<f64>>,
    pub text: Option<String>,
    pub appearance: Option<TextAppearance>,
    pub words: Option<Vec<TextWord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextAppearance {
    pub style: Option<TextStyle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStyle {
    pub name: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextWord {
    pub bounding_box: Option<Vec<f64>>,
    pub text: Option<String>,
    pub confidence: Option<f64>,
}

impl AzureComputerVisionIntegration {
    pub fn new(region: Option<String>) -> Self {
        let region = region.unwrap_or_else(|| "eastus".to_string());
        let config = ServiceConfig {
            base_url: format!("https://{}.api.cognitive.microsoft.com/vision/v3.2", region),
            timeout_ms: 30000,
            max_retries: 3,
            rate_limit_per_minute: 60,
            endpoints: vec![
                "/analyze".to_string(),
                "/ocr".to_string(),
                "/read/analyze".to_string(),
                "/read/analyzeResults/{operationId}".to_string(),
                "/generateThumbnail".to_string(),
                "/models".to_string(),
                "/areaOfInterest".to_string(),
            ],
            headers: HashMap::new(),
        };

        Self {
            service_provider: ServiceProvider::AzureComputerVision,
            config,
        }
    }

    /// Create an analyze image request
    pub fn create_analyze_request(
        image_url: Option<String>,
        visual_features: Option<Vec<String>>,
        details: Option<Vec<String>>,
        language: Option<String>,
    ) -> AnalyzeImageRequest {
        AnalyzeImageRequest {
            url: image_url,
            visual_features,
            details,
            language,
            description_exclude: None,
        }
    }

    /// Create an OCR request
    pub fn create_ocr_request(
        image_url: Option<String>,
        language: Option<String>,
        detect_orientation: Option<bool>,
    ) -> OCRRequest {
        OCRRequest {
            url: image_url,
            language,
            detect_orientation,
        }
    }

    /// Create a read request
    pub fn create_read_request(
        image_url: Option<String>,
        language: Option<String>,
        pages: Option<Vec<String>>,
        read_mode: Option<String>,
    ) -> ReadRequest {
        ReadRequest {
            url: image_url,
            language,
            pages,
            read_mode,
        }
    }

    /// Get common visual features
    pub fn get_all_visual_features() -> Vec<String> {
        vec![
            "Categories".to_string(),
            "Adult".to_string(),
            "Tags".to_string(),
            "Description".to_string(),
            "Faces".to_string(),
            "Color".to_string(),
            "ImageType".to_string(),
            "Objects".to_string(),
            "Brands".to_string(),
        ]
    }

    /// Get common details
    pub fn get_all_details() -> Vec<String> {
        vec![
            "Celebrities".to_string(),
            "Landmarks".to_string(),
        ]
    }

    /// Get supported languages
    pub fn get_supported_languages() -> Vec<String> {
        vec![
            "en".to_string(),
            "es".to_string(),
            "ja".to_string(),
            "pt".to_string(),
            "zh".to_string(),
        ]
    }
}

#[async_trait]
impl ServiceIntegration for AzureComputerVisionIntegration {
    async fn make_request(&self, request: ServiceRequest, api_key: &ApiKey) -> AppResult<ServiceResponse> {
        debug!("Making Azure Computer Vision request: {:?}", request.endpoint);

        // Decrypt the API key
        let decrypted_key = &api_key.encrypted_key; // TODO: Implement proper decryption

        // Build the request URL
        let url = format!("{}{}", self.config.base_url, request.endpoint);

        // Create HTTP client
        let client = reqwest::Client::new();

        // Determine content type based on request body
        let content_type = if request.body.get("url").is_some() {
            "application/json"
        } else {
            "application/octet-stream"
        };

        // Make the request
        let mut req_builder = client
            .request(
                request.method.parse().unwrap_or(reqwest::Method::POST),
                &url,
            )
            .header("Ocp-Apim-Subscription-Key", decrypted_key)
            .header("Content-Type", content_type)
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms));

        // Add request body
        if content_type == "application/json" {
            req_builder = req_builder.json(&request.body);
        } else {
            // For binary data (direct image upload)
            if let Some(image_data) = request.body.get("image_data") {
                if let Some(data_str) = image_data.as_str() {
                    // Decode base64 image data
                    if let Ok(image_bytes) = base64::decode(data_str) {
                        req_builder = req_builder.body(image_bytes);
                    }
                }
            }
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| ApiError::request_failed(
                "azure_computer_vision".to_string(),
                format!("Request failed: {}", e)
            ))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| ApiError::request_failed(
            "azure_computer_vision".to_string(),
            format!("Failed to read response: {}", e)
        ))?;

        if status.is_success() {
            info!("Azure Computer Vision request successful");
            Ok(ServiceResponse {
                status_code: status.as_u16(),
                body: serde_json::from_str(&response_text).unwrap_or_else(|_| {
                    serde_json::json!({ "raw_response": response_text })
                }),
                headers: HashMap::new(),
                response_time_ms: 0, // TODO: Implement proper timing
            })
        } else {
            error!("Azure Computer Vision request failed: {} - {}", status, response_text);
            Err(ApiError::request_failed(
                "azure_computer_vision".to_string(),
                format!("API request failed with status {}: {}", status, response_text)
            ).into())
        }
    }

    async fn health_check(&self, api_key: &ApiKey) -> AppResult<ServiceHealth> {
        debug!("Performing Azure Computer Vision health check");

        // Create a simple test request
        let test_request = ServiceRequest {
            endpoint: "/analyze".to_string(),
            method: "POST".to_string(),
            headers: HashMap::new(),
            body: serde_json::json!({
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Broadway_and_Times_Square_by_night.jpg/450px-Broadway_and_Times_Square_by_night.jpg"
            }),
            timeout_ms: Some(10000),
        };

        let start_time = std::time::Instant::now();
        let result = self.make_request(test_request, api_key).await;
        let response_time = start_time.elapsed().as_millis() as u32;

        match result {
            Ok(_) => {
                info!("Azure Computer Vision health check passed");
                Ok(ServiceHealth {
                    service: self.service_provider,
                    status: ServiceStatus::Healthy,
                    response_time_ms: response_time,
                    last_check: Utc::now(),
                    error_message: None,
                    metadata: HashMap::new(),
                })
            }
            Err(e) => {
                error!("Azure Computer Vision health check failed: {}", e);
                Ok(ServiceHealth {
                    service: self.service_provider,
                    status: ServiceStatus::Unhealthy,
                    response_time_ms: response_time,
                    last_check: Utc::now(),
                    error_message: Some(e.to_string()),
                    metadata: HashMap::new(),
                })
            }
        }
    }

    fn get_config(&self) -> &ServiceConfig {
        &self.config
    }

    async fn update_config(&mut self, config: ServiceConfig) -> AppResult<()> {
        self.config = config;
        Ok(())
    }

    async fn validate_api_key(&self, api_key: &ApiKey) -> AppResult<bool> {
        debug!("Validating Azure Computer Vision API key");
        
        // Perform a simple health check to validate the key
        let health = self.health_check(api_key).await?;
        Ok(matches!(health.status, ServiceStatus::Healthy))
    }

    fn get_endpoints(&self) -> Vec<String> {
        self.config.endpoints.clone()
    }
}
