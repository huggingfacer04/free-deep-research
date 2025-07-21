use tauri::State;
use tracing::{info, error};
use uuid::Uuid;

use crate::error::AppResult;
use crate::models::computer_vision::*;
use crate::services::computer_vision::ComputerVisionService;
use crate::AppState;

/// Process an image with computer vision
#[tauri::command]
pub async fn process_image(
    state: State<'_, AppState>,
    user_id: String,
    request: VisionProcessingRequest,
) -> Result<VisionProcessingResponse, String> {
    info!("Processing image via Tauri command for user: {}", user_id);

    let user_uuid = Uuid::parse_str(&user_id)
        .map_err(|e| format!("Invalid user ID: {}", e))?;

    let computer_vision = state.computer_vision.read().await;
    
    computer_vision
        .process_image(user_uuid, request)
        .await
        .map_err(|e| {
            error!("Failed to process image: {}", e);
            format!("Failed to process image: {}", e)
        })
}

/// Get processing job status
#[tauri::command]
pub async fn get_job_status(
    state: State<'_, AppState>,
    job_id: String,
) -> Result<Option<VisionProcessingJob>, String> {
    info!("Getting job status for: {}", job_id);

    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|e| format!("Invalid job ID: {}", e))?;

    let computer_vision = state.computer_vision.read().await;
    
    computer_vision
        .get_job_status(job_uuid)
        .await
        .map_err(|e| {
            error!("Failed to get job status: {}", e);
            format!("Failed to get job status: {}", e)
        })
}

/// Get user's processing history
#[tauri::command]
pub async fn get_user_vision_history(
    state: State<'_, AppState>,
    user_id: String,
    limit: Option<u32>,
) -> Result<Vec<VisionProcessingJob>, String> {
    info!("Getting vision processing history for user: {}", user_id);

    let user_uuid = Uuid::parse_str(&user_id)
        .map_err(|e| format!("Invalid user ID: {}", e))?;

    let computer_vision = state.computer_vision.read().await;
    
    computer_vision
        .get_user_history(user_uuid, limit)
        .await
        .map_err(|e| {
            error!("Failed to get user history: {}", e);
            format!("Failed to get user history: {}", e)
        })
}

/// Get cost statistics
#[tauri::command]
pub async fn get_vision_cost_stats(
    state: State<'_, AppState>,
) -> Result<VisionCostTracker, String> {
    info!("Getting vision cost statistics");

    let computer_vision = state.computer_vision.read().await;
    
    computer_vision
        .get_cost_stats()
        .await
        .map_err(|e| {
            error!("Failed to get cost stats: {}", e);
            format!("Failed to get cost stats: {}", e)
        })
}

/// Convert image file to base64 for processing
#[tauri::command]
pub async fn convert_image_to_base64(
    file_path: String,
) -> Result<String, String> {
    info!("Converting image to base64: {}", file_path);

    tokio::task::spawn_blocking(move || {
        std::fs::read(&file_path)
            .map_err(|e| format!("Failed to read file: {}", e))
            .map(|bytes| base64::encode(&bytes))
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Validate image format and size
#[tauri::command]
pub async fn validate_image(
    file_path: String,
) -> Result<ImageValidationResult, String> {
    info!("Validating image: {}", file_path);

    tokio::task::spawn_blocking(move || {
        let metadata = std::fs::metadata(&file_path)
            .map_err(|e| format!("Failed to read file metadata: {}", e))?;

        let file_size = metadata.len();
        let max_size = 10 * 1024 * 1024; // 10MB limit

        if file_size > max_size {
            return Ok(ImageValidationResult {
                is_valid: false,
                error_message: Some(format!("File size ({} bytes) exceeds maximum allowed size ({} bytes)", file_size, max_size)),
                file_size: Some(file_size),
                image_format: None,
                dimensions: None,
            });
        }

        // Check file extension
        let path = std::path::Path::new(&file_path);
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        let supported_formats = vec!["jpg", "jpeg", "png", "gif", "bmp", "webp"];
        let is_supported_format = extension
            .as_ref()
            .map(|ext| supported_formats.contains(&ext.as_str()))
            .unwrap_or(false);

        if !is_supported_format {
            return Ok(ImageValidationResult {
                is_valid: false,
                error_message: Some(format!("Unsupported image format. Supported formats: {}", supported_formats.join(", "))),
                file_size: Some(file_size),
                image_format: extension,
                dimensions: None,
            });
        }

        // Try to get image dimensions (simplified - in production you'd use an image library)
        Ok(ImageValidationResult {
            is_valid: true,
            error_message: None,
            file_size: Some(file_size),
            image_format: extension,
            dimensions: None, // Would be populated with actual image dimensions
        })
    })
    .await
    .map_err(|e| format!("Task failed: {}", e))?
}

/// Get supported processing types for a provider
#[tauri::command]
pub async fn get_supported_processing_types(
    provider: VisionProvider,
) -> Result<Vec<VisionProcessingType>, String> {
    info!("Getting supported processing types for provider: {:?}", provider);

    let all_types = vec![
        VisionProcessingType::LabelDetection,
        VisionProcessingType::TextDetection,
        VisionProcessingType::FaceDetection,
        VisionProcessingType::ObjectDetection,
        VisionProcessingType::SafeSearchDetection,
        VisionProcessingType::ImageProperties,
        VisionProcessingType::CropHints,
        VisionProcessingType::WebDetection,
        VisionProcessingType::DocumentTextDetection,
        VisionProcessingType::LandmarkDetection,
        VisionProcessingType::LogoDetection,
        VisionProcessingType::CelebrityRecognition,
        VisionProcessingType::ModerationLabels,
        VisionProcessingType::ImageAnalysis,
    ];

    let supported_types: Vec<VisionProcessingType> = all_types
        .into_iter()
        .filter(|pt| provider.supports_processing_type(pt))
        .collect();

    Ok(supported_types)
}

/// Get available computer vision providers
#[tauri::command]
pub async fn get_available_vision_providers(
    state: State<'_, AppState>,
) -> Result<Vec<ProviderInfo>, String> {
    info!("Getting available computer vision providers");

    let api_manager = state.api_manager.read().await;
    let mut providers = Vec::new();

    // Check Google Vision
    if let Ok(Some(_)) = api_manager.select_best_key_for_service(crate::models::api_key::ServiceProvider::GoogleVision).await {
        providers.push(ProviderInfo {
            provider: VisionProvider::GoogleVision,
            display_name: "Google Vision API".to_string(),
            is_available: true,
            supported_types: get_supported_processing_types(VisionProvider::GoogleVision).await.unwrap_or_default(),
        });
    }

    // Check AWS Rekognition
    if let Ok(Some(_)) = api_manager.select_best_key_for_service(crate::models::api_key::ServiceProvider::AWSRekognition).await {
        providers.push(ProviderInfo {
            provider: VisionProvider::AWSRekognition,
            display_name: "AWS Rekognition".to_string(),
            is_available: true,
            supported_types: get_supported_processing_types(VisionProvider::AWSRekognition).await.unwrap_or_default(),
        });
    }

    // Check Azure Computer Vision
    if let Ok(Some(_)) = api_manager.select_best_key_for_service(crate::models::api_key::ServiceProvider::AzureComputerVision).await {
        providers.push(ProviderInfo {
            provider: VisionProvider::AzureComputerVision,
            display_name: "Azure Computer Vision".to_string(),
            is_available: true,
            supported_types: get_supported_processing_types(VisionProvider::AzureComputerVision).await.unwrap_or_default(),
        });
    }

    Ok(providers)
}

/// Supporting types for Tauri commands
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageValidationResult {
    pub is_valid: bool,
    pub error_message: Option<String>,
    pub file_size: Option<u64>,
    pub image_format: Option<String>,
    pub dimensions: Option<ImageDimensions>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProviderInfo {
    pub provider: VisionProvider,
    pub display_name: String,
    pub is_available: bool,
    pub supported_types: Vec<VisionProcessingType>,
}

// Re-export the VisionCostTracker for Tauri
pub use crate::services::computer_vision::VisionCostTracker;
