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

/// Google Vision API integration
pub struct GoogleVisionIntegration {
    service_provider: ServiceProvider,
    config: ServiceConfig,
}

/// Google Vision API request types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionRequest {
    pub image: VisionImage,
    pub features: Vec<VisionFeature>,
    pub image_context: Option<VisionImageContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionImage {
    pub content: Option<String>, // Base64 encoded image
    pub source: Option<VisionImageSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionImageSource {
    pub gcs_image_uri: Option<String>,
    pub image_uri: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionFeature {
    #[serde(rename = "type")]
    pub feature_type: VisionFeatureType,
    pub max_results: Option<u32>,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionFeatureType {
    #[serde(rename = "LABEL_DETECTION")]
    LabelDetection,
    #[serde(rename = "TEXT_DETECTION")]
    TextDetection,
    #[serde(rename = "DOCUMENT_TEXT_DETECTION")]
    DocumentTextDetection,
    #[serde(rename = "FACE_DETECTION")]
    FaceDetection,
    #[serde(rename = "LANDMARK_DETECTION")]
    LandmarkDetection,
    #[serde(rename = "LOGO_DETECTION")]
    LogoDetection,
    #[serde(rename = "SAFE_SEARCH_DETECTION")]
    SafeSearchDetection,
    #[serde(rename = "IMAGE_PROPERTIES")]
    ImageProperties,
    #[serde(rename = "CROP_HINTS")]
    CropHints,
    #[serde(rename = "WEB_DETECTION")]
    WebDetection,
    #[serde(rename = "PRODUCT_SEARCH")]
    ProductSearch,
    #[serde(rename = "OBJECT_LOCALIZATION")]
    ObjectLocalization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionImageContext {
    pub lat_long_rect: Option<VisionLatLongRect>,
    pub language_hints: Option<Vec<String>>,
    pub crop_hints_params: Option<VisionCropHintsParams>,
    pub product_search_params: Option<VisionProductSearchParams>,
    pub web_detection_params: Option<VisionWebDetectionParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLatLongRect {
    pub min_lat_lng: VisionLatLng,
    pub max_lat_lng: VisionLatLng,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLatLng {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCropHintsParams {
    pub aspect_ratios: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProductSearchParams {
    pub bounding_poly: Option<VisionBoundingPoly>,
    pub product_set: Option<String>,
    pub product_categories: Option<Vec<String>>,
    pub filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebDetectionParams {
    pub include_geo_results: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionBoundingPoly {
    pub vertices: Vec<VisionVertex>,
    pub normalized_vertices: Option<Vec<VisionNormalizedVertex>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionVertex {
    pub x: Option<i32>,
    pub y: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionNormalizedVertex {
    pub x: Option<f32>,
    pub y: Option<f32>,
}

/// Google Vision API response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionResponse {
    pub responses: Vec<VisionAnnotateImageResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionAnnotateImageResponse {
    pub label_annotations: Option<Vec<VisionEntityAnnotation>>,
    pub text_annotations: Option<Vec<VisionEntityAnnotation>>,
    pub face_annotations: Option<Vec<VisionFaceAnnotation>>,
    pub landmark_annotations: Option<Vec<VisionEntityAnnotation>>,
    pub logo_annotations: Option<Vec<VisionEntityAnnotation>>,
    pub safe_search_annotation: Option<VisionSafeSearchAnnotation>,
    pub image_properties_annotation: Option<VisionImageProperties>,
    pub crop_hints_annotation: Option<VisionCropHintsAnnotation>,
    pub web_detection: Option<VisionWebDetection>,
    pub product_search_results: Option<VisionProductSearchResults>,
    pub localized_object_annotations: Option<Vec<VisionLocalizedObjectAnnotation>>,
    pub full_text_annotation: Option<VisionTextAnnotation>,
    pub error: Option<VisionStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEntityAnnotation {
    pub mid: Option<String>,
    pub locale: Option<String>,
    pub description: Option<String>,
    pub score: Option<f32>,
    pub confidence: Option<f32>,
    pub topicality: Option<f32>,
    pub bounding_poly: Option<VisionBoundingPoly>,
    pub locations: Option<Vec<VisionLocationInfo>>,
    pub properties: Option<Vec<VisionProperty>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionFaceAnnotation {
    pub bounding_poly: Option<VisionBoundingPoly>,
    pub fd_bounding_poly: Option<VisionBoundingPoly>,
    pub landmarks: Option<Vec<VisionLandmark>>,
    pub roll_angle: Option<f32>,
    pub pan_angle: Option<f32>,
    pub tilt_angle: Option<f32>,
    pub detection_confidence: Option<f32>,
    pub landmarking_confidence: Option<f32>,
    pub joy_likelihood: Option<VisionLikelihood>,
    pub sorrow_likelihood: Option<VisionLikelihood>,
    pub anger_likelihood: Option<VisionLikelihood>,
    pub surprise_likelihood: Option<VisionLikelihood>,
    pub under_exposed_likelihood: Option<VisionLikelihood>,
    pub blurred_likelihood: Option<VisionLikelihood>,
    pub headwear_likelihood: Option<VisionLikelihood>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionLikelihood {
    #[serde(rename = "UNKNOWN")]
    Unknown,
    #[serde(rename = "VERY_UNLIKELY")]
    VeryUnlikely,
    #[serde(rename = "UNLIKELY")]
    Unlikely,
    #[serde(rename = "POSSIBLE")]
    Possible,
    #[serde(rename = "LIKELY")]
    Likely,
    #[serde(rename = "VERY_LIKELY")]
    VeryLikely,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLandmark {
    #[serde(rename = "type")]
    pub landmark_type: Option<VisionLandmarkType>,
    pub position: Option<VisionPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionLandmarkType {
    #[serde(rename = "UNKNOWN_LANDMARK")]
    UnknownLandmark,
    #[serde(rename = "LEFT_EYE")]
    LeftEye,
    #[serde(rename = "RIGHT_EYE")]
    RightEye,
    #[serde(rename = "LEFT_OF_LEFT_EYEBROW")]
    LeftOfLeftEyebrow,
    #[serde(rename = "RIGHT_OF_LEFT_EYEBROW")]
    RightOfLeftEyebrow,
    #[serde(rename = "LEFT_OF_RIGHT_EYEBROW")]
    LeftOfRightEyebrow,
    #[serde(rename = "RIGHT_OF_RIGHT_EYEBROW")]
    RightOfRightEyebrow,
    #[serde(rename = "MIDPOINT_BETWEEN_EYES")]
    MidpointBetweenEyes,
    #[serde(rename = "NOSE_TIP")]
    NoseTip,
    #[serde(rename = "UPPER_LIP")]
    UpperLip,
    #[serde(rename = "LOWER_LIP")]
    LowerLip,
    #[serde(rename = "MOUTH_LEFT")]
    MouthLeft,
    #[serde(rename = "MOUTH_RIGHT")]
    MouthRight,
    #[serde(rename = "MOUTH_CENTER")]
    MouthCenter,
    #[serde(rename = "NOSE_BOTTOM_RIGHT")]
    NoseBottomRight,
    #[serde(rename = "NOSE_BOTTOM_LEFT")]
    NoseBottomLeft,
    #[serde(rename = "NOSE_BOTTOM_CENTER")]
    NoseBottomCenter,
    #[serde(rename = "LEFT_EYE_TOP_BOUNDARY")]
    LeftEyeTopBoundary,
    #[serde(rename = "LEFT_EYE_RIGHT_CORNER")]
    LeftEyeRightCorner,
    #[serde(rename = "LEFT_EYE_BOTTOM_BOUNDARY")]
    LeftEyeBottomBoundary,
    #[serde(rename = "LEFT_EYE_LEFT_CORNER")]
    LeftEyeLeftCorner,
    #[serde(rename = "RIGHT_EYE_TOP_BOUNDARY")]
    RightEyeTopBoundary,
    #[serde(rename = "RIGHT_EYE_RIGHT_CORNER")]
    RightEyeRightCorner,
    #[serde(rename = "RIGHT_EYE_BOTTOM_BOUNDARY")]
    RightEyeBottomBoundary,
    #[serde(rename = "RIGHT_EYE_LEFT_CORNER")]
    RightEyeLeftCorner,
    #[serde(rename = "LEFT_EYEBROW_UPPER_MIDPOINT")]
    LeftEyebrowUpperMidpoint,
    #[serde(rename = "RIGHT_EYEBROW_UPPER_MIDPOINT")]
    RightEyebrowUpperMidpoint,
    #[serde(rename = "LEFT_EAR_TRAGION")]
    LeftEarTragion,
    #[serde(rename = "RIGHT_EAR_TRAGION")]
    RightEarTragion,
    #[serde(rename = "LEFT_EYE_PUPIL")]
    LeftEyePupil,
    #[serde(rename = "RIGHT_EYE_PUPIL")]
    RightEyePupil,
    #[serde(rename = "FOREHEAD_GLABELLA")]
    ForeheadGlabella,
    #[serde(rename = "CHIN_GNATHION")]
    ChinGnathion,
    #[serde(rename = "CHIN_LEFT_GONION")]
    ChinLeftGonion,
    #[serde(rename = "CHIN_RIGHT_GONION")]
    ChinRightGonion,
    #[serde(rename = "LEFT_CHEEK_CENTER")]
    LeftCheekCenter,
    #[serde(rename = "RIGHT_CHEEK_CENTER")]
    RightCheekCenter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionPosition {
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub z: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionSafeSearchAnnotation {
    pub adult: Option<VisionLikelihood>,
    pub spoof: Option<VisionLikelihood>,
    pub medical: Option<VisionLikelihood>,
    pub violence: Option<VisionLikelihood>,
    pub racy: Option<VisionLikelihood>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionImageProperties {
    pub dominant_colors: Option<VisionDominantColorsAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionDominantColorsAnnotation {
    pub colors: Vec<VisionColorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionColorInfo {
    pub color: Option<VisionColor>,
    pub score: Option<f32>,
    pub pixel_fraction: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionColor {
    pub red: Option<f32>,
    pub green: Option<f32>,
    pub blue: Option<f32>,
    pub alpha: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCropHintsAnnotation {
    pub crop_hints: Vec<VisionCropHint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCropHint {
    pub bounding_poly: Option<VisionBoundingPoly>,
    pub confidence: Option<f32>,
    pub importance_fraction: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebDetection {
    pub web_entities: Option<Vec<VisionWebEntity>>,
    pub full_matching_images: Option<Vec<VisionWebImage>>,
    pub partial_matching_images: Option<Vec<VisionWebImage>>,
    pub pages_with_matching_images: Option<Vec<VisionWebPage>>,
    pub visually_similar_images: Option<Vec<VisionWebImage>>,
    pub best_guess_labels: Option<Vec<VisionWebLabel>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebEntity {
    pub entity_id: Option<String>,
    pub score: Option<f32>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebImage {
    pub url: Option<String>,
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebPage {
    pub url: Option<String>,
    pub score: Option<f32>,
    pub page_title: Option<String>,
    pub full_matching_images: Option<Vec<VisionWebImage>>,
    pub partial_matching_images: Option<Vec<VisionWebImage>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebLabel {
    pub label: Option<String>,
    pub language_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProductSearchResults {
    pub index_time: Option<String>,
    pub results: Option<Vec<VisionResult>>,
    pub product_grouped_results: Option<Vec<VisionGroupedResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionResult {
    pub product: Option<VisionProduct>,
    pub score: Option<f32>,
    pub image: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionGroupedResult {
    pub bounding_poly: Option<VisionBoundingPoly>,
    pub results: Option<Vec<VisionResult>>,
    pub object_annotations: Option<Vec<VisionObjectAnnotation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProduct {
    pub name: Option<String>,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub product_category: Option<String>,
    pub product_labels: Option<Vec<VisionKeyValue>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionObjectAnnotation {
    pub mid: Option<String>,
    pub language_code: Option<String>,
    pub name: Option<String>,
    pub score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionKeyValue {
    pub key: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLocalizedObjectAnnotation {
    pub mid: Option<String>,
    pub language_code: Option<String>,
    pub name: Option<String>,
    pub score: Option<f32>,
    pub bounding_poly: Option<VisionBoundingPoly>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionTextAnnotation {
    pub pages: Option<Vec<VisionPage>>,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionPage {
    pub property: Option<VisionTextProperty>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub blocks: Option<Vec<VisionBlock>>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionTextProperty {
    pub detected_languages: Option<Vec<VisionDetectedLanguage>>,
    pub detected_break: Option<VisionDetectedBreak>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionDetectedLanguage {
    pub language_code: Option<String>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionDetectedBreak {
    #[serde(rename = "type")]
    pub break_type: Option<VisionBreakType>,
    pub is_prefix: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionBreakType {
    #[serde(rename = "UNKNOWN")]
    Unknown,
    #[serde(rename = "SPACE")]
    Space,
    #[serde(rename = "SURE_SPACE")]
    SureSpace,
    #[serde(rename = "EOL_SURE_SPACE")]
    EolSureSpace,
    #[serde(rename = "HYPHEN")]
    Hyphen,
    #[serde(rename = "LINE_BREAK")]
    LineBreak,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionBlock {
    pub property: Option<VisionTextProperty>,
    pub bounding_box: Option<VisionBoundingPoly>,
    pub paragraphs: Option<Vec<VisionParagraph>>,
    pub block_type: Option<VisionBlockType>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionBlockType {
    #[serde(rename = "UNKNOWN")]
    Unknown,
    #[serde(rename = "TEXT")]
    Text,
    #[serde(rename = "TABLE")]
    Table,
    #[serde(rename = "PICTURE")]
    Picture,
    #[serde(rename = "RULER")]
    Ruler,
    #[serde(rename = "BARCODE")]
    Barcode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionParagraph {
    pub property: Option<VisionTextProperty>,
    pub bounding_box: Option<VisionBoundingPoly>,
    pub words: Option<Vec<VisionWord>>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWord {
    pub property: Option<VisionTextProperty>,
    pub bounding_box: Option<VisionBoundingPoly>,
    pub symbols: Option<Vec<VisionSymbol>>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionSymbol {
    pub property: Option<VisionTextProperty>,
    pub bounding_box: Option<VisionBoundingPoly>,
    pub text: Option<String>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLocationInfo {
    pub lat_lng: Option<VisionLatLng>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProperty {
    pub name: Option<String>,
    pub value: Option<String>,
    pub uint64_value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionStatus {
    pub code: Option<i32>,
    pub message: Option<String>,
    pub details: Option<Vec<serde_json::Value>>,
}

impl GoogleVisionIntegration {
    pub fn new() -> Self {
        let config = ServiceConfig {
            base_url: "https://vision.googleapis.com/v1".to_string(),
            timeout_ms: 30000,
            max_retries: 3,
            rate_limit_per_minute: 60,
            endpoints: vec![
                "/images:annotate".to_string(),
                "/images:asyncBatchAnnotate".to_string(),
                "/files:annotate".to_string(),
                "/files:asyncBatchAnnotate".to_string(),
            ],
            headers: HashMap::new(),
        };

        Self {
            service_provider: ServiceProvider::GoogleVision,
            config,
        }
    }

    /// Create a vision request for image analysis
    pub fn create_vision_request(
        image_content: String,
        features: Vec<VisionFeature>,
        image_context: Option<VisionImageContext>,
    ) -> VisionRequest {
        VisionRequest {
            image: VisionImage {
                content: Some(image_content),
                source: None,
            },
            features,
            image_context,
        }
    }

    /// Create a vision request for image URL analysis
    pub fn create_vision_request_from_url(
        image_url: String,
        features: Vec<VisionFeature>,
        image_context: Option<VisionImageContext>,
    ) -> VisionRequest {
        VisionRequest {
            image: VisionImage {
                content: None,
                source: Some(VisionImageSource {
                    gcs_image_uri: None,
                    image_uri: Some(image_url),
                }),
            },
            features,
            image_context,
        }
    }

    /// Create common vision features
    pub fn create_label_detection_feature(max_results: Option<u32>) -> VisionFeature {
        VisionFeature {
            feature_type: VisionFeatureType::LabelDetection,
            max_results,
            model: None,
        }
    }

    pub fn create_text_detection_feature() -> VisionFeature {
        VisionFeature {
            feature_type: VisionFeatureType::TextDetection,
            max_results: None,
            model: None,
        }
    }

    pub fn create_face_detection_feature(max_results: Option<u32>) -> VisionFeature {
        VisionFeature {
            feature_type: VisionFeatureType::FaceDetection,
            max_results,
            model: None,
        }
    }

    pub fn create_object_localization_feature(max_results: Option<u32>) -> VisionFeature {
        VisionFeature {
            feature_type: VisionFeatureType::ObjectLocalization,
            max_results,
            model: None,
        }
    }

    pub fn create_safe_search_feature() -> VisionFeature {
        VisionFeature {
            feature_type: VisionFeatureType::SafeSearchDetection,
            max_results: None,
            model: None,
        }
    }
}

#[async_trait]
impl ServiceIntegration for GoogleVisionIntegration {
    async fn make_request(&self, request: ServiceRequest, api_key: &ApiKey) -> AppResult<ServiceResponse> {
        debug!("Making Google Vision API request: {:?}", request.endpoint);

        // Decrypt the API key
        let decrypted_key = &api_key.encrypted_key; // TODO: Implement proper decryption

        // Build the request URL
        let url = format!("{}{}", self.config.base_url, request.endpoint);

        // Create HTTP client
        let client = reqwest::Client::new();

        // Make the request
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", decrypted_key))
            .header("Content-Type", "application/json")
            .json(&request.body)
            .timeout(std::time::Duration::from_millis(self.config.timeout_ms))
            .send()
            .await
            .map_err(|e| ApiError::request_failed(
                "google_vision".to_string(),
                format!("Request failed: {}", e)
            ))?;

        let status = response.status();
        let response_text = response.text().await.map_err(|e| ApiError::request_failed(
            "google_vision".to_string(),
            format!("Failed to read response: {}", e)
        ))?;

        if status.is_success() {
            info!("Google Vision API request successful");
            Ok(ServiceResponse {
                status_code: status.as_u16(),
                body: serde_json::from_str(&response_text).unwrap_or_else(|_| {
                    serde_json::json!({ "raw_response": response_text })
                }),
                headers: HashMap::new(),
                response_time_ms: 0, // TODO: Implement proper timing
            })
        } else {
            error!("Google Vision API request failed: {} - {}", status, response_text);
            Err(ApiError::request_failed(
                "google_vision".to_string(),
                format!("API request failed with status {}: {}", status, response_text)
            ).into())
        }
    }

    async fn health_check(&self, api_key: &ApiKey) -> AppResult<ServiceHealth> {
        debug!("Performing Google Vision API health check");

        // Create a simple test request
        let test_request = ServiceRequest {
            endpoint: "/images:annotate".to_string(),
            method: "POST".to_string(),
            headers: HashMap::new(),
            body: serde_json::json!({
                "requests": [{
                    "image": {
                        "content": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    },
                    "features": [{
                        "type": "LABEL_DETECTION",
                        "maxResults": 1
                    }]
                }]
            }),
            timeout_ms: Some(10000),
        };

        let start_time = std::time::Instant::now();
        let result = self.make_request(test_request, api_key).await;
        let response_time = start_time.elapsed().as_millis() as u32;

        match result {
            Ok(_) => {
                info!("Google Vision API health check passed");
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
                error!("Google Vision API health check failed: {}", e);
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
        debug!("Validating Google Vision API key");
        
        // Perform a simple health check to validate the key
        let health = self.health_check(api_key).await?;
        Ok(matches!(health.status, ServiceStatus::Healthy))
    }

    fn get_endpoints(&self) -> Vec<String> {
        self.config.endpoints.clone()
    }
}
