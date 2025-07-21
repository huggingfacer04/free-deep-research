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

/// AWS Rekognition integration
pub struct AWSRekognitionIntegration {
    service_provider: ServiceProvider,
    config: ServiceConfig,
}

/// AWS Rekognition request types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RekognitionImage {
    #[serde(rename = "Bytes")]
    pub bytes: Option<String>, // Base64 encoded image
    #[serde(rename = "S3Object")]
    pub s3_object: Option<RekognitionS3Object>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RekognitionS3Object {
    #[serde(rename = "Bucket")]
    pub bucket: String,
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Version")]
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectLabelsRequest {
    #[serde(rename = "Image")]
    pub image: RekognitionImage,
    #[serde(rename = "MaxLabels")]
    pub max_labels: Option<u32>,
    #[serde(rename = "MinConfidence")]
    pub min_confidence: Option<f32>,
    #[serde(rename = "Features")]
    pub features: Option<Vec<String>>,
    #[serde(rename = "Settings")]
    pub settings: Option<DetectLabelsSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectLabelsSettings {
    #[serde(rename = "GeneralLabels")]
    pub general_labels: Option<GeneralLabelsSettings>,
    #[serde(rename = "ImageProperties")]
    pub image_properties: Option<ImagePropertiesSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralLabelsSettings {
    #[serde(rename = "LabelInclusionFilters")]
    pub label_inclusion_filters: Option<Vec<String>>,
    #[serde(rename = "LabelExclusionFilters")]
    pub label_exclusion_filters: Option<Vec<String>>,
    #[serde(rename = "LabelCategoryInclusionFilters")]
    pub label_category_inclusion_filters: Option<Vec<String>>,
    #[serde(rename = "LabelCategoryExclusionFilters")]
    pub label_category_exclusion_filters: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePropertiesSettings {
    #[serde(rename = "MaxDominantColors")]
    pub max_dominant_colors: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectTextRequest {
    #[serde(rename = "Image")]
    pub image: RekognitionImage,
    #[serde(rename = "Filters")]
    pub filters: Option<DetectTextFilters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectTextFilters {
    #[serde(rename = "WordFilter")]
    pub word_filter: Option<DetectionFilter>,
    #[serde(rename = "RegionsOfInterest")]
    pub regions_of_interest: Option<Vec<RegionOfInterest>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionFilter {
    #[serde(rename = "MinConfidence")]
    pub min_confidence: Option<f32>,
    #[serde(rename = "MinBoundingBoxHeight")]
    pub min_bounding_box_height: Option<f32>,
    #[serde(rename = "MinBoundingBoxWidth")]
    pub min_bounding_box_width: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionOfInterest {
    #[serde(rename = "BoundingBox")]
    pub bounding_box: Option<BoundingBox>,
    #[serde(rename = "Polygon")]
    pub polygon: Option<Vec<Point>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    #[serde(rename = "Width")]
    pub width: f32,
    #[serde(rename = "Height")]
    pub height: f32,
    #[serde(rename = "Left")]
    pub left: f32,
    #[serde(rename = "Top")]
    pub top: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    #[serde(rename = "X")]
    pub x: f32,
    #[serde(rename = "Y")]
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectFacesRequest {
    #[serde(rename = "Image")]
    pub image: RekognitionImage,
    #[serde(rename = "Attributes")]
    pub attributes: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizeCelebritiesRequest {
    #[serde(rename = "Image")]
    pub image: RekognitionImage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectModerationLabelsRequest {
    #[serde(rename = "Image")]
    pub image: RekognitionImage,
    #[serde(rename = "MinConfidence")]
    pub min_confidence: Option<f32>,
    #[serde(rename = "HumanLoopConfig")]
    pub human_loop_config: Option<HumanLoopConfig>,
    #[serde(rename = "ProjectVersion")]
    pub project_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanLoopConfig {
    #[serde(rename = "HumanLoopName")]
    pub human_loop_name: String,
    #[serde(rename = "FlowDefinitionArn")]
    pub flow_definition_arn: String,
    #[serde(rename = "DataAttributes")]
    pub data_attributes: Option<HumanLoopDataAttributes>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanLoopDataAttributes {
    #[serde(rename = "ContentClassifiers")]
    pub content_classifiers: Option<Vec<String>>,
}

/// AWS Rekognition response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectLabelsResponse {
    #[serde(rename = "Labels")]
    pub labels: Option<Vec<Label>>,
    #[serde(rename = "LabelModelVersion")]
    pub label_model_version: Option<String>,
    #[serde(rename = "OrientationCorrection")]
    pub orientation_correction: Option<String>,
    #[serde(rename = "ImageProperties")]
    pub image_properties: Option<ImageProperties>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    #[serde(rename = "Name")]
    pub name: Option<String>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
    #[serde(rename = "Instances")]
    pub instances: Option<Vec<Instance>>,
    #[serde(rename = "Parents")]
    pub parents: Option<Vec<Parent>>,
    #[serde(rename = "Aliases")]
    pub aliases: Option<Vec<LabelAlias>>,
    #[serde(rename = "Categories")]
    pub categories: Option<Vec<LabelCategory>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    #[serde(rename = "BoundingBox")]
    pub bounding_box: Option<BoundingBox>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
    #[serde(rename = "DominantColors")]
    pub dominant_colors: Option<Vec<DominantColor>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parent {
    #[serde(rename = "Name")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelAlias {
    #[serde(rename = "Name")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelCategory {
    #[serde(rename = "Name")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominantColor {
    #[serde(rename = "Red")]
    pub red: Option<u8>,
    #[serde(rename = "Blue")]
    pub blue: Option<u8>,
    #[serde(rename = "Green")]
    pub green: Option<u8>,
    #[serde(rename = "HexCode")]
    pub hex_code: Option<String>,
    #[serde(rename = "CSSColor")]
    pub css_color: Option<String>,
    #[serde(rename = "SimplifiedColor")]
    pub simplified_color: Option<String>,
    #[serde(rename = "PixelPercent")]
    pub pixel_percent: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProperties {
    #[serde(rename = "Quality")]
    pub quality: Option<ImageQuality>,
    #[serde(rename = "DominantColors")]
    pub dominant_colors: Option<Vec<DominantColor>>,
    #[serde(rename = "Foreground")]
    pub foreground: Option<DetectLabelsImageForeground>,
    #[serde(rename = "Background")]
    pub background: Option<DetectLabelsImageBackground>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageQuality {
    #[serde(rename = "Brightness")]
    pub brightness: Option<f32>,
    #[serde(rename = "Sharpness")]
    pub sharpness: Option<f32>,
    #[serde(rename = "Contrast")]
    pub contrast: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectLabelsImageForeground {
    #[serde(rename = "Quality")]
    pub quality: Option<DetectLabelsImageQuality>,
    #[serde(rename = "DominantColors")]
    pub dominant_colors: Option<Vec<DominantColor>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectLabelsImageBackground {
    #[serde(rename = "Quality")]
    pub quality: Option<DetectLabelsImageQuality>,
    #[serde(rename = "DominantColors")]
    pub dominant_colors: Option<Vec<DominantColor>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectLabelsImageQuality {
    #[serde(rename = "Brightness")]
    pub brightness: Option<f32>,
    #[serde(rename = "Sharpness")]
    pub sharpness: Option<f32>,
    #[serde(rename = "Contrast")]
    pub contrast: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectTextResponse {
    #[serde(rename = "TextDetections")]
    pub text_detections: Option<Vec<TextDetection>>,
    #[serde(rename = "TextModelVersion")]
    pub text_model_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDetection {
    #[serde(rename = "DetectedText")]
    pub detected_text: Option<String>,
    #[serde(rename = "Type")]
    pub detection_type: Option<String>,
    #[serde(rename = "Id")]
    pub id: Option<u32>,
    #[serde(rename = "ParentId")]
    pub parent_id: Option<u32>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
    #[serde(rename = "Geometry")]
    pub geometry: Option<Geometry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometry {
    #[serde(rename = "BoundingBox")]
    pub bounding_box: Option<BoundingBox>,
    #[serde(rename = "Polygon")]
    pub polygon: Option<Vec<Point>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectFacesResponse {
    #[serde(rename = "FaceDetails")]
    pub face_details: Option<Vec<FaceDetail>>,
    #[serde(rename = "OrientationCorrection")]
    pub orientation_correction: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetail {
    #[serde(rename = "BoundingBox")]
    pub bounding_box: Option<BoundingBox>,
    #[serde(rename = "AgeRange")]
    pub age_range: Option<AgeRange>,
    #[serde(rename = "Smile")]
    pub smile: Option<Smile>,
    #[serde(rename = "Eyeglasses")]
    pub eyeglasses: Option<Eyeglasses>,
    #[serde(rename = "Sunglasses")]
    pub sunglasses: Option<Sunglasses>,
    #[serde(rename = "Gender")]
    pub gender: Option<Gender>,
    #[serde(rename = "Beard")]
    pub beard: Option<Beard>,
    #[serde(rename = "Mustache")]
    pub mustache: Option<Mustache>,
    #[serde(rename = "EyesOpen")]
    pub eyes_open: Option<EyeOpen>,
    #[serde(rename = "MouthOpen")]
    pub mouth_open: Option<MouthOpen>,
    #[serde(rename = "Emotions")]
    pub emotions: Option<Vec<Emotion>>,
    #[serde(rename = "Landmarks")]
    pub landmarks: Option<Vec<Landmark>>,
    #[serde(rename = "Pose")]
    pub pose: Option<Pose>,
    #[serde(rename = "Quality")]
    pub quality: Option<ImageQuality>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
    #[serde(rename = "FaceOccluded")]
    pub face_occluded: Option<FaceOccluded>,
    #[serde(rename = "EyeDirection")]
    pub eye_direction: Option<EyeDirection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeRange {
    #[serde(rename = "Low")]
    pub low: Option<u32>,
    #[serde(rename = "High")]
    pub high: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Smile {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Eyeglasses {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sunglasses {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gender {
    #[serde(rename = "Value")]
    pub value: Option<String>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beard {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mustache {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeOpen {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MouthOpen {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Emotion {
    #[serde(rename = "Type")]
    pub emotion_type: Option<String>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Landmark {
    #[serde(rename = "Type")]
    pub landmark_type: Option<String>,
    #[serde(rename = "X")]
    pub x: Option<f32>,
    #[serde(rename = "Y")]
    pub y: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    #[serde(rename = "Roll")]
    pub roll: Option<f32>,
    #[serde(rename = "Yaw")]
    pub yaw: Option<f32>,
    #[serde(rename = "Pitch")]
    pub pitch: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceOccluded {
    #[serde(rename = "Value")]
    pub value: Option<bool>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeDirection {
    #[serde(rename = "Yaw")]
    pub yaw: Option<f32>,
    #[serde(rename = "Pitch")]
    pub pitch: Option<f32>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizeCelebritiesResponse {
    #[serde(rename = "CelebrityFaces")]
    pub celebrity_faces: Option<Vec<Celebrity>>,
    #[serde(rename = "UnrecognizedFaces")]
    pub unrecognized_faces: Option<Vec<ComparedFace>>,
    #[serde(rename = "OrientationCorrection")]
    pub orientation_correction: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Celebrity {
    #[serde(rename = "Urls")]
    pub urls: Option<Vec<String>>,
    #[serde(rename = "Name")]
    pub name: Option<String>,
    #[serde(rename = "Id")]
    pub id: Option<String>,
    #[serde(rename = "Face")]
    pub face: Option<ComparedFace>,
    #[serde(rename = "MatchConfidence")]
    pub match_confidence: Option<f32>,
    #[serde(rename = "KnownGender")]
    pub known_gender: Option<KnownGender>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparedFace {
    #[serde(rename = "BoundingBox")]
    pub bounding_box: Option<BoundingBox>,
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
    #[serde(rename = "Landmarks")]
    pub landmarks: Option<Vec<Landmark>>,
    #[serde(rename = "Pose")]
    pub pose: Option<Pose>,
    #[serde(rename = "Quality")]
    pub quality: Option<ImageQuality>,
    #[serde(rename = "Emotions")]
    pub emotions: Option<Vec<Emotion>>,
    #[serde(rename = "Smile")]
    pub smile: Option<Smile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownGender {
    #[serde(rename = "Type")]
    pub gender_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectModerationLabelsResponse {
    #[serde(rename = "ModerationLabels")]
    pub moderation_labels: Option<Vec<ModerationLabel>>,
    #[serde(rename = "ModerationModelVersion")]
    pub moderation_model_version: Option<String>,
    #[serde(rename = "HumanLoopActivationOutput")]
    pub human_loop_activation_output: Option<HumanLoopActivationOutput>,
    #[serde(rename = "ProjectVersion")]
    pub project_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationLabel {
    #[serde(rename = "Confidence")]
    pub confidence: Option<f32>,
    #[serde(rename = "Name")]
    pub name: Option<String>,
    #[serde(rename = "ParentName")]
    pub parent_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanLoopActivationOutput {
    #[serde(rename = "HumanLoopArn")]
    pub human_loop_arn: Option<String>,
    #[serde(rename = "HumanLoopActivationReasons")]
    pub human_loop_activation_reasons: Option<Vec<String>>,
    #[serde(rename = "HumanLoopActivationConditionsEvaluationResults")]
    pub human_loop_activation_conditions_evaluation_results: Option<String>,
}

impl AWSRekognitionIntegration {
    pub fn new() -> Self {
        let config = ServiceConfig {
            base_url: "https://rekognition.us-east-1.amazonaws.com".to_string(),
            timeout_ms: 30000,
            max_retries: 3,
            rate_limit_per_minute: 60,
            endpoints: vec![
                "/DetectLabels".to_string(),
                "/DetectText".to_string(),
                "/DetectFaces".to_string(),
                "/RecognizeCelebrities".to_string(),
                "/DetectModerationLabels".to_string(),
                "/CompareFaces".to_string(),
                "/SearchFacesByImage".to_string(),
                "/IndexFaces".to_string(),
            ],
            headers: HashMap::new(),
        };

        Self {
            service_provider: ServiceProvider::AWSRekognition,
            config,
        }
    }

    /// Create a detect labels request
    pub fn create_detect_labels_request(
        image_bytes: String,
        max_labels: Option<u32>,
        min_confidence: Option<f32>,
    ) -> DetectLabelsRequest {
        DetectLabelsRequest {
            image: RekognitionImage {
                bytes: Some(image_bytes),
                s3_object: None,
            },
            max_labels,
            min_confidence,
            features: None,
            settings: None,
        }
    }

    /// Create a detect text request
    pub fn create_detect_text_request(image_bytes: String) -> DetectTextRequest {
        DetectTextRequest {
            image: RekognitionImage {
                bytes: Some(image_bytes),
                s3_object: None,
            },
            filters: None,
        }
    }

    /// Create a detect faces request
    pub fn create_detect_faces_request(
        image_bytes: String,
        attributes: Option<Vec<String>>,
    ) -> DetectFacesRequest {
        DetectFacesRequest {
            image: RekognitionImage {
                bytes: Some(image_bytes),
                s3_object: None,
            },
            attributes,
        }
    }

    /// Create a recognize celebrities request
    pub fn create_recognize_celebrities_request(image_bytes: String) -> RecognizeCelebritiesRequest {
        RecognizeCelebritiesRequest {
            image: RekognitionImage {
                bytes: Some(image_bytes),
                s3_object: None,
            },
        }
    }

    /// Create a detect moderation labels request
    pub fn create_detect_moderation_labels_request(
        image_bytes: String,
        min_confidence: Option<f32>,
    ) -> DetectModerationLabelsRequest {
        DetectModerationLabelsRequest {
            image: RekognitionImage {
                bytes: Some(image_bytes),
                s3_object: None,
            },
            min_confidence,
            human_loop_config: None,
            project_version: None,
        }
    }
}

#[async_trait]
impl ServiceIntegration for AWSRekognitionIntegration {
    async fn make_request(&self, request: ServiceRequest, api_key: &ApiKey) -> AppResult<ServiceResponse> {
        debug!("Making AWS Rekognition request: {:?}", request.endpoint);

        // For AWS Rekognition, we need to handle AWS signature authentication
        // This is a simplified implementation - in production, you'd use the AWS SDK
        let decrypted_key = &api_key.encrypted_key; // TODO: Implement proper decryption

        // Build the request URL
        let url = format!("{}{}", self.config.base_url, request.endpoint);

        // Create HTTP client
        let client = reqwest::Client::new();

        // AWS Rekognition uses AWS4-HMAC-SHA256 signature
        // For simplicity, we'll assume the API key contains the access key and secret
        // In production, use the AWS SDK for proper authentication
        let response = client
            .post(&url)
            .header("Authorization", format!("AWS4-HMAC-SHA256 Credential={}", decrypted_key))
            .header("Content-Type", "application/x-amz-json-1.1")
            .header("X-Amz-Target", format!("RekognitionService{}", request.endpoint.trim_start_matches('/')))
            .json(&request.body)
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .send()
            .await
            .map_err(|e| ApiError::request_failed(
                "aws_rekognition".to_string(),
                format!("Request failed: {}", e)
            ))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| ApiError::request_failed(
            "aws_rekognition".to_string(),
            format!("Failed to read response: {}", e)
        ))?;

        if status.is_success() {
            info!("AWS Rekognition request successful");
            Ok(ServiceResponse {
                status_code: status.as_u16(),
                body: serde_json::from_str(&response_text).unwrap_or_else(|_| {
                    serde_json::json!({ "raw_response": response_text })
                }),
                headers: HashMap::new(),
                response_time_ms: 0, // TODO: Implement proper timing
            })
        } else {
            error!("AWS Rekognition request failed: {} - {}", status, response_text);
            Err(ApiError::request_failed(
                "aws_rekognition".to_string(),
                format!("API request failed with status {}: {}", status, response_text)
            ).into())
        }
    }

    async fn health_check(&self, api_key: &ApiKey) -> AppResult<ServiceHealth> {
        debug!("Performing AWS Rekognition health check");

        // Create a simple test request
        let test_request = ServiceRequest {
            endpoint: "/DetectLabels".to_string(),
            method: "POST".to_string(),
            headers: HashMap::new(),
            body: serde_json::json!({
                "Image": {
                    "Bytes": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                },
                "MaxLabels": 1,
                "MinConfidence": 50.0
            }),
            timeout_ms: Some(10000),
        };

        let start_time = std::time::Instant::now();
        let result = self.make_request(test_request, api_key).await;
        let response_time = start_time.elapsed().as_millis() as u32;

        match result {
            Ok(_) => {
                info!("AWS Rekognition health check passed");
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
                error!("AWS Rekognition health check failed: {}", e);
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
        debug!("Validating AWS Rekognition API key");
        
        // Perform a simple health check to validate the key
        let health = self.health_check(api_key).await?;
        Ok(matches!(health.status, ServiceStatus::Healthy))
    }

    fn get_endpoints(&self) -> Vec<String> {
        self.config.endpoints.clone()
    }
}
