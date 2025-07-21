use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Computer vision processing job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProcessingJob {
    pub id: Uuid,
    pub user_id: Uuid,
    pub image_url: Option<String>,
    pub image_data: Option<String>, // Base64 encoded image
    pub processing_type: VisionProcessingType,
    pub provider: VisionProvider,
    pub status: ProcessingStatus,
    pub results: Option<VisionResults>,
    pub error_message: Option<String>,
    pub processing_time_ms: Option<u32>,
    pub cost_cents: Option<u32>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub metadata: Option<serde_json::Value>,
}

/// Types of computer vision processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionProcessingType {
    LabelDetection,
    TextDetection,
    FaceDetection,
    ObjectDetection,
    SafeSearchDetection,
    ImageProperties,
    CropHints,
    WebDetection,
    DocumentTextDetection,
    LandmarkDetection,
    LogoDetection,
    CelebrityRecognition,
    ModerationLabels,
    ImageAnalysis, // Azure comprehensive analysis
}

/// Computer vision service providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionProvider {
    GoogleVision,
    AWSRekognition,
    AzureComputerVision,
}

/// Processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

/// Computer vision results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionResults {
    pub labels: Option<Vec<VisionLabel>>,
    pub text_detections: Option<Vec<VisionTextDetection>>,
    pub faces: Option<Vec<VisionFace>>,
    pub objects: Option<Vec<VisionObject>>,
    pub safe_search: Option<VisionSafeSearch>,
    pub image_properties: Option<VisionImageProperties>,
    pub crop_hints: Option<Vec<VisionCropHint>>,
    pub web_detection: Option<VisionWebDetection>,
    pub landmarks: Option<Vec<VisionLandmark>>,
    pub logos: Option<Vec<VisionLogo>>,
    pub celebrities: Option<Vec<VisionCelebrity>>,
    pub moderation_labels: Option<Vec<VisionModerationLabel>>,
    pub raw_response: Option<serde_json::Value>,
}

/// Vision label detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLabel {
    pub name: String,
    pub confidence: f32,
    pub description: Option<String>,
    pub category: Option<String>,
    pub bounding_box: Option<VisionBoundingBox>,
}

/// Text detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionTextDetection {
    pub text: String,
    pub confidence: f32,
    pub language: Option<String>,
    pub bounding_box: Option<VisionBoundingBox>,
    pub text_type: Option<String>, // WORD, LINE, PARAGRAPH, BLOCK
}

/// Face detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionFace {
    pub bounding_box: VisionBoundingBox,
    pub confidence: f32,
    pub age_range: Option<VisionAgeRange>,
    pub gender: Option<String>,
    pub emotions: Option<Vec<VisionEmotion>>,
    pub landmarks: Option<Vec<VisionFaceLandmark>>,
    pub attributes: Option<VisionFaceAttributes>,
}

/// Object detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionObject {
    pub name: String,
    pub confidence: f32,
    pub bounding_box: VisionBoundingBox,
    pub category: Option<String>,
}

/// Safe search detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionSafeSearch {
    pub adult: VisionLikelihood,
    pub spoof: VisionLikelihood,
    pub medical: VisionLikelihood,
    pub violence: VisionLikelihood,
    pub racy: VisionLikelihood,
}

/// Image properties result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionImageProperties {
    pub dominant_colors: Vec<VisionColor>,
    pub accent_color: Option<String>,
    pub is_black_and_white: Option<bool>,
    pub brightness: Option<f32>,
    pub contrast: Option<f32>,
    pub sharpness: Option<f32>,
}

/// Crop hint result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCropHint {
    pub bounding_box: VisionBoundingBox,
    pub confidence: f32,
    pub importance_fraction: Option<f32>,
}

/// Web detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebDetection {
    pub web_entities: Vec<VisionWebEntity>,
    pub full_matching_images: Vec<VisionWebImage>,
    pub partial_matching_images: Vec<VisionWebImage>,
    pub pages_with_matching_images: Vec<VisionWebPage>,
    pub visually_similar_images: Vec<VisionWebImage>,
    pub best_guess_labels: Vec<String>,
}

/// Landmark detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLandmark {
    pub name: String,
    pub confidence: f32,
    pub bounding_box: Option<VisionBoundingBox>,
    pub location: Option<VisionLocation>,
}

/// Logo detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLogo {
    pub name: String,
    pub confidence: f32,
    pub bounding_box: VisionBoundingBox,
}

/// Celebrity recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionCelebrity {
    pub name: String,
    pub confidence: f32,
    pub bounding_box: VisionBoundingBox,
    pub urls: Option<Vec<String>>,
    pub known_gender: Option<String>,
}

/// Moderation label result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionModerationLabel {
    pub name: String,
    pub confidence: f32,
    pub parent_name: Option<String>,
}

/// Supporting types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionBoundingBox {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionAgeRange {
    pub low: u32,
    pub high: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEmotion {
    pub emotion_type: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionFaceLandmark {
    pub landmark_type: String,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionFaceAttributes {
    pub smile: Option<bool>,
    pub eyeglasses: Option<bool>,
    pub sunglasses: Option<bool>,
    pub beard: Option<bool>,
    pub mustache: Option<bool>,
    pub eyes_open: Option<bool>,
    pub mouth_open: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionLikelihood {
    Unknown,
    VeryUnlikely,
    Unlikely,
    Possible,
    Likely,
    VeryLikely,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionColor {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub hex_code: String,
    pub pixel_fraction: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebEntity {
    pub entity_id: Option<String>,
    pub description: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebImage {
    pub url: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionWebPage {
    pub url: String,
    pub title: Option<String>,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLocation {
    pub latitude: f64,
    pub longitude: f64,
}

/// Request types for computer vision processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProcessingRequest {
    pub image_url: Option<String>,
    pub image_data: Option<String>, // Base64 encoded
    pub processing_types: Vec<VisionProcessingType>,
    pub provider: Option<VisionProvider>,
    pub options: Option<VisionProcessingOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProcessingOptions {
    pub max_results: Option<u32>,
    pub min_confidence: Option<f32>,
    pub language: Option<String>,
    pub include_geo_results: Option<bool>,
    pub crop_hints_aspect_ratios: Option<Vec<f32>>,
    pub face_attributes: Option<Vec<String>>,
}

/// Response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionProcessingResponse {
    pub job_id: Uuid,
    pub status: ProcessingStatus,
    pub results: Option<VisionResults>,
    pub processing_time_ms: Option<u32>,
    pub cost_cents: Option<u32>,
    pub provider_used: VisionProvider,
    pub error_message: Option<String>,
}

impl VisionProcessingJob {
    pub fn new(
        user_id: Uuid,
        image_url: Option<String>,
        image_data: Option<String>,
        processing_type: VisionProcessingType,
        provider: VisionProvider,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id,
            image_url,
            image_data,
            processing_type,
            provider,
            status: ProcessingStatus::Pending,
            results: None,
            error_message: None,
            processing_time_ms: None,
            cost_cents: None,
            created_at: Utc::now(),
            completed_at: None,
            metadata: None,
        }
    }

    pub fn mark_completed(&mut self, results: VisionResults, processing_time_ms: u32, cost_cents: u32) {
        self.status = ProcessingStatus::Completed;
        self.results = Some(results);
        self.processing_time_ms = Some(processing_time_ms);
        self.cost_cents = Some(cost_cents);
        self.completed_at = Some(Utc::now());
    }

    pub fn mark_failed(&mut self, error_message: String) {
        self.status = ProcessingStatus::Failed;
        self.error_message = Some(error_message);
        self.completed_at = Some(Utc::now());
    }

    pub fn is_completed(&self) -> bool {
        matches!(self.status, ProcessingStatus::Completed | ProcessingStatus::Failed | ProcessingStatus::Cancelled)
    }
}

impl VisionProvider {
    pub fn display_name(&self) -> &'static str {
        match self {
            VisionProvider::GoogleVision => "Google Vision API",
            VisionProvider::AWSRekognition => "AWS Rekognition",
            VisionProvider::AzureComputerVision => "Azure Computer Vision",
        }
    }

    pub fn supports_processing_type(&self, processing_type: &VisionProcessingType) -> bool {
        match self {
            VisionProvider::GoogleVision => matches!(
                processing_type,
                VisionProcessingType::LabelDetection
                    | VisionProcessingType::TextDetection
                    | VisionProcessingType::FaceDetection
                    | VisionProcessingType::ObjectDetection
                    | VisionProcessingType::SafeSearchDetection
                    | VisionProcessingType::ImageProperties
                    | VisionProcessingType::CropHints
                    | VisionProcessingType::WebDetection
                    | VisionProcessingType::DocumentTextDetection
                    | VisionProcessingType::LandmarkDetection
                    | VisionProcessingType::LogoDetection
            ),
            VisionProvider::AWSRekognition => matches!(
                processing_type,
                VisionProcessingType::LabelDetection
                    | VisionProcessingType::TextDetection
                    | VisionProcessingType::FaceDetection
                    | VisionProcessingType::CelebrityRecognition
                    | VisionProcessingType::ModerationLabels
                    | VisionProcessingType::ObjectDetection
            ),
            VisionProvider::AzureComputerVision => matches!(
                processing_type,
                VisionProcessingType::LabelDetection
                    | VisionProcessingType::TextDetection
                    | VisionProcessingType::FaceDetection
                    | VisionProcessingType::ObjectDetection
                    | VisionProcessingType::ImageAnalysis
                    | VisionProcessingType::DocumentTextDetection
                    | VisionProcessingType::ImageProperties
            ),
        }
    }
}
