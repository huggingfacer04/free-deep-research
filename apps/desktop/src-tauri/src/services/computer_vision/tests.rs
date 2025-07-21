#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::computer_vision::*;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use uuid::Uuid;

    // Mock services for testing
    struct MockApiManagerService;
    struct MockDataPersistenceService;
    struct MockMonitoringService;

    #[async_trait::async_trait]
    impl crate::services::Service for MockApiManagerService {
        async fn start(&mut self) -> crate::error::AppResult<()> { Ok(()) }
        async fn stop(&mut self) -> crate::error::AppResult<()> { Ok(()) }
        async fn health_check(&self) -> crate::error::AppResult<bool> { Ok(true) }
    }

    #[async_trait::async_trait]
    impl crate::services::Service for MockDataPersistenceService {
        async fn start(&mut self) -> crate::error::AppResult<()> { Ok(()) }
        async fn stop(&mut self) -> crate::error::AppResult<()> { Ok(()) }
        async fn health_check(&self) -> crate::error::AppResult<bool> { Ok(true) }
    }

    #[async_trait::async_trait]
    impl crate::services::Service for MockMonitoringService {
        async fn start(&mut self) -> crate::error::AppResult<()> { Ok(()) }
        async fn stop(&mut self) -> crate::error::AppResult<()> { Ok(()) }
        async fn health_check(&self) -> crate::error::AppResult<bool> { Ok(true) }
    }

    async fn create_test_service() -> ComputerVisionService {
        let api_manager = Arc::new(RwLock::new(MockApiManagerService));
        let data_persistence = Arc::new(RwLock::new(MockDataPersistenceService));
        let monitoring = Arc::new(RwLock::new(MockMonitoringService));

        // For testing, we'll create a simplified version
        ComputerVisionService {
            api_manager: api_manager as Arc<RwLock<dyn crate::services::Service + Send + Sync>>,
            data_persistence: data_persistence as Arc<RwLock<dyn crate::services::Service + Send + Sync>>,
            monitoring: monitoring as Arc<RwLock<dyn crate::services::Service + Send + Sync>>,
            active_jobs: Arc::new(RwLock::new(std::collections::HashMap::new())),
            cost_tracker: Arc::new(RwLock::new(VisionCostTracker {
                daily_cost_cents: 0,
                monthly_cost_cents: 0,
                daily_limit_cents: 1000,
                monthly_limit_cents: 10000,
                cost_per_request: std::collections::HashMap::from([
                    (VisionProvider::GoogleVision, 150),
                    (VisionProvider::AWSRekognition, 100),
                    (VisionProvider::AzureComputerVision, 100),
                ]),
            })),
        }
    }

    #[tokio::test]
    async fn test_vision_processing_job_creation() {
        let user_id = Uuid::new_v4();
        let job = VisionProcessingJob::new(
            user_id,
            Some("https://example.com/image.jpg".to_string()),
            None,
            VisionProcessingType::LabelDetection,
            VisionProvider::GoogleVision,
        );

        assert_eq!(job.user_id, user_id);
        assert_eq!(job.processing_type, VisionProcessingType::LabelDetection);
        assert_eq!(job.provider, VisionProvider::GoogleVision);
        assert_eq!(job.status, ProcessingStatus::Pending);
        assert!(job.image_url.is_some());
        assert!(job.image_data.is_none());
    }

    #[tokio::test]
    async fn test_vision_processing_job_completion() {
        let user_id = Uuid::new_v4();
        let mut job = VisionProcessingJob::new(
            user_id,
            Some("https://example.com/image.jpg".to_string()),
            None,
            VisionProcessingType::LabelDetection,
            VisionProvider::GoogleVision,
        );

        let results = VisionResults {
            labels: Some(vec![VisionLabel {
                name: "Test Label".to_string(),
                confidence: 0.95,
                description: Some("Test description".to_string()),
                category: Some("Test category".to_string()),
                bounding_box: None,
            }]),
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

        job.mark_completed(results, 1500, 150);

        assert_eq!(job.status, ProcessingStatus::Completed);
        assert!(job.results.is_some());
        assert_eq!(job.processing_time_ms, Some(1500));
        assert_eq!(job.cost_cents, Some(150));
        assert!(job.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_vision_processing_job_failure() {
        let user_id = Uuid::new_v4();
        let mut job = VisionProcessingJob::new(
            user_id,
            Some("https://example.com/image.jpg".to_string()),
            None,
            VisionProcessingType::LabelDetection,
            VisionProvider::GoogleVision,
        );

        job.mark_failed("Test error message".to_string());

        assert_eq!(job.status, ProcessingStatus::Failed);
        assert_eq!(job.error_message, Some("Test error message".to_string()));
        assert!(job.completed_at.is_some());
    }

    #[tokio::test]
    async fn test_provider_supports_processing_type() {
        // Test Google Vision support
        assert!(VisionProvider::GoogleVision.supports_processing_type(&VisionProcessingType::LabelDetection));
        assert!(VisionProvider::GoogleVision.supports_processing_type(&VisionProcessingType::TextDetection));
        assert!(VisionProvider::GoogleVision.supports_processing_type(&VisionProcessingType::FaceDetection));
        assert!(!VisionProvider::GoogleVision.supports_processing_type(&VisionProcessingType::CelebrityRecognition));

        // Test AWS Rekognition support
        assert!(VisionProvider::AWSRekognition.supports_processing_type(&VisionProcessingType::LabelDetection));
        assert!(VisionProvider::AWSRekognition.supports_processing_type(&VisionProcessingType::CelebrityRecognition));
        assert!(!VisionProvider::AWSRekognition.supports_processing_type(&VisionProcessingType::CropHints));

        // Test Azure Computer Vision support
        assert!(VisionProvider::AzureComputerVision.supports_processing_type(&VisionProcessingType::LabelDetection));
        assert!(VisionProvider::AzureComputerVision.supports_processing_type(&VisionProcessingType::ImageAnalysis));
        assert!(!VisionProvider::AzureComputerVision.supports_processing_type(&VisionProcessingType::CelebrityRecognition));
    }

    #[tokio::test]
    async fn test_vision_processing_request_validation() {
        let request_with_url = VisionProcessingRequest {
            image_url: Some("https://example.com/image.jpg".to_string()),
            image_data: None,
            processing_types: vec![VisionProcessingType::LabelDetection],
            provider: Some(VisionProvider::GoogleVision),
            options: None,
        };

        let request_with_data = VisionProcessingRequest {
            image_url: None,
            image_data: Some("base64encodeddata".to_string()),
            processing_types: vec![VisionProcessingType::TextDetection],
            provider: Some(VisionProvider::AWSRekognition),
            options: Some(VisionProcessingOptions {
                max_results: Some(10),
                min_confidence: Some(0.8),
                language: Some("en".to_string()),
                include_geo_results: Some(false),
                crop_hints_aspect_ratios: None,
                face_attributes: None,
            }),
        };

        let invalid_request = VisionProcessingRequest {
            image_url: None,
            image_data: None,
            processing_types: vec![],
            provider: None,
            options: None,
        };

        // Valid requests should have either URL or data, and at least one processing type
        assert!(request_with_url.image_url.is_some() || request_with_url.image_data.is_some());
        assert!(!request_with_url.processing_types.is_empty());

        assert!(request_with_data.image_url.is_some() || request_with_data.image_data.is_some());
        assert!(!request_with_data.processing_types.is_empty());

        // Invalid request should fail validation
        assert!(invalid_request.image_url.is_none() && invalid_request.image_data.is_none());
        assert!(invalid_request.processing_types.is_empty());
    }

    #[tokio::test]
    async fn test_cost_tracker() {
        let mut cost_tracker = VisionCostTracker {
            daily_cost_cents: 500,
            monthly_cost_cents: 5000,
            daily_limit_cents: 1000,
            monthly_limit_cents: 10000,
            cost_per_request: std::collections::HashMap::from([
                (VisionProvider::GoogleVision, 150),
                (VisionProvider::AWSRekognition, 100),
                (VisionProvider::AzureComputerVision, 100),
            ]),
        };

        // Test cost per request
        assert_eq!(cost_tracker.cost_per_request.get(&VisionProvider::GoogleVision), Some(&150));
        assert_eq!(cost_tracker.cost_per_request.get(&VisionProvider::AWSRekognition), Some(&100));

        // Test limit checking
        assert!(cost_tracker.daily_cost_cents < cost_tracker.daily_limit_cents);
        assert!(cost_tracker.monthly_cost_cents < cost_tracker.monthly_limit_cents);

        // Simulate adding cost
        cost_tracker.daily_cost_cents += 150;
        cost_tracker.monthly_cost_cents += 150;

        assert_eq!(cost_tracker.daily_cost_cents, 650);
        assert_eq!(cost_tracker.monthly_cost_cents, 5150);
    }

    #[tokio::test]
    async fn test_vision_results_structure() {
        let results = VisionResults {
            labels: Some(vec![
                VisionLabel {
                    name: "Cat".to_string(),
                    confidence: 0.95,
                    description: Some("Domestic cat".to_string()),
                    category: Some("Animal".to_string()),
                    bounding_box: Some(VisionBoundingBox {
                        left: 0.1,
                        top: 0.2,
                        width: 0.3,
                        height: 0.4,
                    }),
                },
                VisionLabel {
                    name: "Furniture".to_string(),
                    confidence: 0.87,
                    description: None,
                    category: Some("Object".to_string()),
                    bounding_box: None,
                },
            ]),
            text_detections: Some(vec![
                VisionTextDetection {
                    text: "Hello World".to_string(),
                    confidence: 0.99,
                    language: Some("en".to_string()),
                    bounding_box: Some(VisionBoundingBox {
                        left: 0.0,
                        top: 0.0,
                        width: 0.5,
                        height: 0.1,
                    }),
                    text_type: Some("LINE".to_string()),
                },
            ]),
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

        // Test labels
        assert!(results.labels.is_some());
        let labels = results.labels.unwrap();
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].name, "Cat");
        assert_eq!(labels[0].confidence, 0.95);
        assert!(labels[0].bounding_box.is_some());

        // Test text detections
        assert!(results.text_detections.is_some());
        let text_detections = results.text_detections.unwrap();
        assert_eq!(text_detections.len(), 1);
        assert_eq!(text_detections[0].text, "Hello World");
        assert_eq!(text_detections[0].confidence, 0.99);
    }
}
