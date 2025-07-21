use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

/// Image classification model types and configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageClassificationModel {
    pub id: Uuid,
    pub name: String,
    pub model_type: ImageModelType,
    pub version: String,
    pub framework: MLFramework,
    pub architecture: ModelArchitecture,
    pub status: ModelStatus,
    pub accuracy_metrics: Option<AccuracyMetrics>,
    pub training_config: Option<TrainingConfiguration>,
    pub deployment_config: Option<DeploymentConfiguration>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: Uuid,
    pub metadata: serde_json::Value,
}

/// Types of image classification models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageModelType {
    /// General purpose image classification
    GeneralClassification,
    /// Document classification (research papers, reports, etc.)
    DocumentClassification,
    /// Medical image classification
    MedicalImageClassification,
    /// Scientific diagram classification
    ScientificDiagramClassification,
    /// Quality assessment model
    ImageQualityAssessment,
    /// Content moderation model
    ContentModerationClassification,
    /// Custom domain-specific model
    CustomDomainClassification { domain: String },
}

/// Machine learning frameworks supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLFramework {
    TensorFlow,
    PyTorch,
    Keras,
    ScikitLearn,
    Huggingface,
}

/// Model architectures available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Convolutional Neural Networks
    CNN {
        layers: Vec<CNNLayer>,
        input_shape: (u32, u32, u32), // (height, width, channels)
    },
    /// ResNet architectures
    ResNet {
        variant: ResNetVariant,
        pretrained: bool,
    },
    /// EfficientNet architectures
    EfficientNet {
        variant: EfficientNetVariant,
        pretrained: bool,
    },
    /// Vision Transformer
    VisionTransformer {
        patch_size: u32,
        embed_dim: u32,
        num_heads: u32,
        num_layers: u32,
    },
    /// Custom architecture
    Custom {
        architecture_json: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResNetVariant {
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EfficientNetVariant {
    B0, B1, B2, B3, B4, B5, B6, B7,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNNLayer {
    pub layer_type: CNNLayerType,
    pub filters: Option<u32>,
    pub kernel_size: Option<(u32, u32)>,
    pub stride: Option<(u32, u32)>,
    pub padding: Option<String>,
    pub activation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CNNLayerType {
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Dropout,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
}

/// Model deployment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Draft,
    Training,
    Trained,
    Validating,
    Deployed,
    Deprecated,
    Failed,
}

/// Model accuracy and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub top_5_accuracy: Option<f64>,
    pub confusion_matrix: Option<Vec<Vec<u32>>>,
    pub class_metrics: HashMap<String, ClassMetrics>,
    pub validation_loss: f64,
    pub training_loss: f64,
    pub inference_time_ms: f64,
    pub model_size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: u32,
}

/// Training configuration for image classification models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfiguration {
    pub dataset_config: DatasetConfiguration,
    pub training_params: TrainingParameters,
    pub augmentation_config: Option<DataAugmentationConfig>,
    pub validation_config: ValidationConfiguration,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub learning_rate_schedule: Option<LearningRateSchedule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfiguration {
    pub dataset_name: String,
    pub dataset_path: String,
    pub num_classes: u32,
    pub class_names: Vec<String>,
    pub train_split: f64,
    pub validation_split: f64,
    pub test_split: f64,
    pub image_size: (u32, u32),
    pub batch_size: u32,
    pub shuffle: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParameters {
    pub epochs: u32,
    pub learning_rate: f64,
    pub optimizer: OptimizerConfig,
    pub loss_function: String,
    pub metrics: Vec<String>,
    pub regularization: Option<RegularizationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    Adam,
    SGD,
    RMSprop,
    AdamW,
    Adagrad,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    pub l1_lambda: Option<f64>,
    pub l2_lambda: Option<f64>,
    pub dropout_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAugmentationConfig {
    pub rotation_range: Option<f64>,
    pub width_shift_range: Option<f64>,
    pub height_shift_range: Option<f64>,
    pub shear_range: Option<f64>,
    pub zoom_range: Option<f64>,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub brightness_range: Option<(f64, f64)>,
    pub contrast_range: Option<(f64, f64)>,
    pub saturation_range: Option<(f64, f64)>,
    pub hue_range: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfiguration {
    pub validation_frequency: u32, // Every N epochs
    pub validation_metrics: Vec<String>,
    pub save_best_model: bool,
    pub monitor_metric: String,
    pub mode: ValidationMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    Min, // For loss
    Max, // For accuracy
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub monitor: String,
    pub patience: u32,
    pub min_delta: f64,
    pub mode: ValidationMode,
    pub restore_best_weights: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateSchedule {
    pub schedule_type: LearningRateScheduleType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateScheduleType {
    StepDecay,
    ExponentialDecay,
    CosineAnnealing,
    ReduceOnPlateau,
    Constant,
}

/// Deployment configuration for TensorFlow Serving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfiguration {
    pub serving_config: ServingConfiguration,
    pub scaling_config: ScalingConfiguration,
    pub monitoring_config: MonitoringConfiguration,
    pub a_b_testing_config: Option<ABTestingConfiguration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingConfiguration {
    pub model_name: String,
    pub model_version: String,
    pub serving_platform: ServingPlatform,
    pub input_signature: InputSignature,
    pub output_signature: OutputSignature,
    pub preprocessing_config: Option<PreprocessingConfig>,
    pub postprocessing_config: Option<PostprocessingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServingPlatform {
    TensorFlowServing,
    TorchServe,
    ONNX,
    Triton,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSignature {
    pub input_name: String,
    pub input_shape: Vec<i32>,
    pub input_dtype: String,
    pub preprocessing_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSignature {
    pub output_name: String,
    pub output_shape: Vec<i32>,
    pub output_dtype: String,
    pub class_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub resize_method: String,
    pub normalization: NormalizationConfig,
    pub color_space: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationConfig {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessingConfig {
    pub apply_softmax: bool,
    pub top_k: Option<u32>,
    pub confidence_threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfiguration {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_utilization: f64,
    pub target_memory_utilization: f64,
    pub scale_up_cooldown: u32,
    pub scale_down_cooldown: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfiguration {
    pub enable_metrics: bool,
    pub enable_logging: bool,
    pub log_level: String,
    pub metrics_port: u16,
    pub health_check_path: String,
    pub prometheus_metrics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestingConfiguration {
    pub test_name: String,
    pub control_model_version: String,
    pub treatment_model_version: String,
    pub traffic_split: f64, // 0.0 to 1.0
    pub success_metrics: Vec<String>,
    pub duration_days: u32,
}

/// Image classification training job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageClassificationJob {
    pub job_id: Uuid,
    pub model_id: Uuid,
    pub user_id: Uuid,
    pub job_name: String,
    pub status: TrainingJobStatus,
    pub progress: f64,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_metrics: Option<TrainingMetrics>,
    pub best_metrics: Option<TrainingMetrics>,
    pub logs: Vec<TrainingLogEntry>,
    pub started_at: DateTime<Utc>,
    pub estimated_completion: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub resource_usage: Option<ResourceUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingJobStatus {
    Queued,
    Initializing,
    DataLoading,
    Training,
    Validating,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: u32,
    pub training_loss: f64,
    pub training_accuracy: f64,
    pub validation_loss: f64,
    pub validation_accuracy: f64,
    pub learning_rate: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub epoch: Option<u32>,
    pub metrics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub gpu_usage_percent: Option<f64>,
    pub gpu_memory_usage_mb: Option<f64>,
    pub disk_usage_mb: f64,
    pub network_io_mb: f64,
}

/// Image classification inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageClassificationRequest {
    pub request_id: Uuid,
    pub model_id: Uuid,
    pub image_data: ImageData,
    pub inference_config: InferenceConfig,
    pub user_id: Uuid,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageData {
    Base64 { data: String, format: String },
    Url { url: String },
    FilePath { path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub top_k: Option<u32>,
    pub confidence_threshold: Option<f64>,
    pub return_probabilities: bool,
    pub return_features: bool,
    pub batch_size: Option<u32>,
}

/// Image classification inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageClassificationResponse {
    pub request_id: Uuid,
    pub model_id: Uuid,
    pub predictions: Vec<ClassificationPrediction>,
    pub inference_time_ms: f64,
    pub model_version: String,
    pub features: Option<Vec<f64>>,
    pub metadata: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationPrediction {
    pub class_name: String,
    pub class_id: u32,
    pub confidence: f64,
    pub probability: f64,
}

impl ImageClassificationModel {
    pub fn new(
        name: String,
        model_type: ImageModelType,
        framework: MLFramework,
        architecture: ModelArchitecture,
        created_by: Uuid,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name,
            model_type,
            version: "1.0.0".to_string(),
            framework,
            architecture,
            status: ModelStatus::Draft,
            accuracy_metrics: None,
            training_config: None,
            deployment_config: None,
            created_at: now,
            updated_at: now,
            created_by,
            metadata: serde_json::json!({}),
        }
    }

    pub fn update_status(&mut self, status: ModelStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    pub fn update_metrics(&mut self, metrics: AccuracyMetrics) {
        self.accuracy_metrics = Some(metrics);
        self.updated_at = Utc::now();
    }

    pub fn is_ready_for_deployment(&self) -> bool {
        matches!(self.status, ModelStatus::Trained | ModelStatus::Deployed) 
            && self.accuracy_metrics.is_some()
            && self.deployment_config.is_some()
    }

    pub fn get_serving_endpoint(&self) -> Option<String> {
        if let Some(deployment_config) = &self.deployment_config {
            Some(format!(
                "/v1/models/{}:predict",
                deployment_config.serving_config.model_name
            ))
        } else {
            None
        }
    }
}

impl ImageClassificationJob {
    pub fn new(
        model_id: Uuid,
        user_id: Uuid,
        job_name: String,
        total_epochs: u32,
    ) -> Self {
        Self {
            job_id: Uuid::new_v4(),
            model_id,
            user_id,
            job_name,
            status: TrainingJobStatus::Queued,
            progress: 0.0,
            current_epoch: 0,
            total_epochs,
            current_metrics: None,
            best_metrics: None,
            logs: Vec::new(),
            started_at: Utc::now(),
            estimated_completion: None,
            completed_at: None,
            error_message: None,
            resource_usage: None,
        }
    }

    pub fn update_progress(&mut self, epoch: u32, metrics: TrainingMetrics) {
        self.current_epoch = epoch;
        self.progress = (epoch as f64 / self.total_epochs as f64) * 100.0;
        self.current_metrics = Some(metrics.clone());
        
        // Update best metrics if this is better
        if let Some(ref best) = self.best_metrics {
            if metrics.validation_accuracy > best.validation_accuracy {
                self.best_metrics = Some(metrics);
            }
        } else {
            self.best_metrics = Some(metrics);
        }
    }

    pub fn add_log(&mut self, level: LogLevel, message: String, metrics: Option<serde_json::Value>) {
        self.logs.push(TrainingLogEntry {
            timestamp: Utc::now(),
            level,
            message,
            epoch: Some(self.current_epoch),
            metrics,
        });
    }

    pub fn mark_completed(&mut self) {
        self.status = TrainingJobStatus::Completed;
        self.progress = 100.0;
        self.completed_at = Some(Utc::now());
    }

    pub fn mark_failed(&mut self, error_message: String) {
        self.status = TrainingJobStatus::Failed;
        self.error_message = Some(error_message);
        self.completed_at = Some(Utc::now());
    }
}
