use tauri::State;
use tracing::{info, error};
use uuid::Uuid;

use crate::error::AppResult;
use crate::models::image_classification::*;
use crate::services::ServiceManager;

/// Create a new image classification model
#[tauri::command]
pub async fn create_image_classification_model(
    state: State<'_, ServiceManager>,
    name: String,
    model_type: ImageModelType,
    framework: MLFramework,
    architecture: ModelArchitecture,
    created_by: String,
) -> Result<String, String> {
    info!("Creating image classification model: {}", name);

    let created_by_uuid = Uuid::parse_str(&created_by)
        .map_err(|e| format!("Invalid user ID: {}", e))?;

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .create_image_classification_model(name, model_type, framework, architecture, created_by_uuid)
        .await
        .map(|id| id.to_string())
        .map_err(|e| {
            error!("Failed to create image classification model: {}", e);
            format!("Failed to create model: {}", e)
        })
}

/// Start training an image classification model
#[tauri::command]
pub async fn start_image_classification_training(
    state: State<'_, ServiceManager>,
    model_id: String,
    training_config: TrainingConfiguration,
    user_id: String,
) -> Result<String, String> {
    info!("Starting image classification training for model: {}", model_id);

    let model_uuid = Uuid::parse_str(&model_id)
        .map_err(|e| format!("Invalid model ID: {}", e))?;
    
    let user_uuid = Uuid::parse_str(&user_id)
        .map_err(|e| format!("Invalid user ID: {}", e))?;

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .start_image_classification_training(model_uuid, training_config, user_uuid)
        .await
        .map(|id| id.to_string())
        .map_err(|e| {
            error!("Failed to start image classification training: {}", e);
            format!("Failed to start training: {}", e)
        })
}

/// Get image classification training job status
#[tauri::command]
pub async fn get_image_classification_job(
    state: State<'_, ServiceManager>,
    job_id: String,
) -> Result<Option<ImageClassificationJob>, String> {
    info!("Getting image classification job status: {}", job_id);

    let job_uuid = Uuid::parse_str(&job_id)
        .map_err(|e| format!("Invalid job ID: {}", e))?;

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .get_image_classification_job(job_uuid)
        .await
        .map_err(|e| {
            error!("Failed to get image classification job: {}", e);
            format!("Failed to get job: {}", e)
        })
}

/// Get image classification model
#[tauri::command]
pub async fn get_image_classification_model(
    state: State<'_, ServiceManager>,
    model_id: String,
) -> Result<Option<ImageClassificationModel>, String> {
    info!("Getting image classification model: {}", model_id);

    let model_uuid = Uuid::parse_str(&model_id)
        .map_err(|e| format!("Invalid model ID: {}", e))?;

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .get_image_classification_model(model_uuid)
        .await
        .map_err(|e| {
            error!("Failed to get image classification model: {}", e);
            format!("Failed to get model: {}", e)
        })
}

/// List user's image classification models
#[tauri::command]
pub async fn list_user_image_classification_models(
    state: State<'_, ServiceManager>,
    user_id: String,
) -> Result<Vec<ImageClassificationModel>, String> {
    info!("Listing image classification models for user: {}", user_id);

    let user_uuid = Uuid::parse_str(&user_id)
        .map_err(|e| format!("Invalid user ID: {}", e))?;

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .list_user_image_classification_models(user_uuid)
        .await
        .map_err(|e| {
            error!("Failed to list image classification models: {}", e);
            format!("Failed to list models: {}", e)
        })
}

/// Deploy an image classification model
#[tauri::command]
pub async fn deploy_image_classification_model(
    state: State<'_, ServiceManager>,
    model_id: String,
    deployment_config: DeploymentConfiguration,
) -> Result<String, String> {
    info!("Deploying image classification model: {}", model_id);

    let model_uuid = Uuid::parse_str(&model_id)
        .map_err(|e| format!("Invalid model ID: {}", e))?;

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .deploy_image_classification_model(model_uuid, deployment_config)
        .await
        .map_err(|e| {
            error!("Failed to deploy image classification model: {}", e);
            format!("Failed to deploy model: {}", e)
        })
}

/// Classify an image using a deployed model
#[tauri::command]
pub async fn classify_image(
    state: State<'_, ServiceManager>,
    request: ImageClassificationRequest,
) -> Result<ImageClassificationResponse, String> {
    info!("Classifying image with model: {}", request.model_id);

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .classify_image(request)
        .await
        .map_err(|e| {
            error!("Failed to classify image: {}", e);
            format!("Failed to classify image: {}", e)
        })
}

/// Get image classification training statistics
#[tauri::command]
pub async fn get_image_classification_statistics(
    state: State<'_, ServiceManager>,
) -> Result<crate::services::ml_engine::image_classification::TrainingStatistics, String> {
    info!("Getting image classification statistics");

    let ml_engine = state.ml_engine.read().await;
    
    ml_engine
        .get_image_classification_statistics()
        .await
        .map_err(|e| {
            error!("Failed to get image classification statistics: {}", e);
            format!("Failed to get statistics: {}", e)
        })
}

/// Create a default CNN architecture for image classification
#[tauri::command]
pub async fn create_default_cnn_architecture(
    input_height: u32,
    input_width: u32,
    input_channels: u32,
    num_classes: u32,
) -> Result<ModelArchitecture, String> {
    info!("Creating default CNN architecture for {}x{}x{} input with {} classes", 
          input_width, input_height, input_channels, num_classes);

    let layers = vec![
        CNNLayer {
            layer_type: CNNLayerType::Conv2D,
            filters: Some(32),
            kernel_size: Some((3, 3)),
            stride: Some((1, 1)),
            padding: Some("same".to_string()),
            activation: Some("relu".to_string()),
        },
        CNNLayer {
            layer_type: CNNLayerType::MaxPooling2D,
            filters: None,
            kernel_size: Some((2, 2)),
            stride: Some((2, 2)),
            padding: None,
            activation: None,
        },
        CNNLayer {
            layer_type: CNNLayerType::Conv2D,
            filters: Some(64),
            kernel_size: Some((3, 3)),
            stride: Some((1, 1)),
            padding: Some("same".to_string()),
            activation: Some("relu".to_string()),
        },
        CNNLayer {
            layer_type: CNNLayerType::MaxPooling2D,
            filters: None,
            kernel_size: Some((2, 2)),
            stride: Some((2, 2)),
            padding: None,
            activation: None,
        },
        CNNLayer {
            layer_type: CNNLayerType::Conv2D,
            filters: Some(128),
            kernel_size: Some((3, 3)),
            stride: Some((1, 1)),
            padding: Some("same".to_string()),
            activation: Some("relu".to_string()),
        },
        CNNLayer {
            layer_type: CNNLayerType::GlobalAveragePooling2D,
            filters: None,
            kernel_size: None,
            stride: None,
            padding: None,
            activation: None,
        },
        CNNLayer {
            layer_type: CNNLayerType::Dense,
            filters: Some(num_classes),
            kernel_size: None,
            stride: None,
            padding: None,
            activation: Some("softmax".to_string()),
        },
    ];

    Ok(ModelArchitecture::CNN {
        layers,
        input_shape: (input_height, input_width, input_channels),
    })
}

/// Create a default training configuration
#[tauri::command]
pub async fn create_default_training_config(
    dataset_name: String,
    dataset_path: String,
    num_classes: u32,
    class_names: Vec<String>,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
) -> Result<TrainingConfiguration, String> {
    info!("Creating default training configuration for dataset: {}", dataset_name);

    let dataset_config = DatasetConfiguration {
        dataset_name,
        dataset_path,
        num_classes,
        class_names,
        train_split: 0.7,
        validation_split: 0.2,
        test_split: 0.1,
        image_size: (224, 224),
        batch_size,
        shuffle: true,
    };

    let training_params = TrainingParameters {
        epochs,
        learning_rate,
        optimizer: OptimizerConfig {
            optimizer_type: OptimizerType::Adam,
            parameters: std::collections::HashMap::from([
                ("beta_1".to_string(), 0.9),
                ("beta_2".to_string(), 0.999),
                ("epsilon".to_string(), 1e-7),
            ]),
        },
        loss_function: "categorical_crossentropy".to_string(),
        metrics: vec!["accuracy".to_string(), "top_5_accuracy".to_string()],
        regularization: Some(RegularizationConfig {
            l1_lambda: None,
            l2_lambda: Some(0.001),
            dropout_rate: Some(0.5),
        }),
    };

    let augmentation_config = DataAugmentationConfig {
        rotation_range: Some(20.0),
        width_shift_range: Some(0.2),
        height_shift_range: Some(0.2),
        shear_range: Some(0.2),
        zoom_range: Some(0.2),
        horizontal_flip: true,
        vertical_flip: false,
        brightness_range: Some((0.8, 1.2)),
        contrast_range: Some((0.8, 1.2)),
        saturation_range: Some((0.8, 1.2)),
        hue_range: Some(0.1),
    };

    let validation_config = ValidationConfiguration {
        validation_frequency: 1,
        validation_metrics: vec!["accuracy".to_string(), "loss".to_string()],
        save_best_model: true,
        monitor_metric: "val_accuracy".to_string(),
        mode: ValidationMode::Max,
    };

    let early_stopping = EarlyStoppingConfig {
        monitor: "val_accuracy".to_string(),
        patience: 10,
        min_delta: 0.001,
        mode: ValidationMode::Max,
        restore_best_weights: true,
    };

    let learning_rate_schedule = LearningRateSchedule {
        schedule_type: LearningRateScheduleType::ReduceOnPlateau,
        parameters: std::collections::HashMap::from([
            ("factor".to_string(), 0.5),
            ("patience".to_string(), 5.0),
            ("min_lr".to_string(), 1e-7),
        ]),
    };

    Ok(TrainingConfiguration {
        dataset_config,
        training_params,
        augmentation_config: Some(augmentation_config),
        validation_config,
        early_stopping: Some(early_stopping),
        learning_rate_schedule: Some(learning_rate_schedule),
    })
}

/// Create a default deployment configuration
#[tauri::command]
pub async fn create_default_deployment_config(
    model_name: String,
    model_version: String,
) -> Result<DeploymentConfiguration, String> {
    info!("Creating default deployment configuration for model: {}", model_name);

    let serving_config = ServingConfiguration {
        model_name: model_name.clone(),
        model_version,
        serving_platform: ServingPlatform::TensorFlowServing,
        input_signature: InputSignature {
            input_name: "input_image".to_string(),
            input_shape: vec![-1, 224, 224, 3],
            input_dtype: "float32".to_string(),
            preprocessing_required: true,
        },
        output_signature: OutputSignature {
            output_name: "predictions".to_string(),
            output_shape: vec![-1, 1000], // Assuming 1000 classes
            output_dtype: "float32".to_string(),
            class_names: None,
        },
        preprocessing_config: Some(PreprocessingConfig {
            resize_method: "bilinear".to_string(),
            normalization: NormalizationConfig {
                mean: vec![0.485, 0.456, 0.406],
                std: vec![0.229, 0.224, 0.225],
            },
            color_space: "RGB".to_string(),
        }),
        postprocessing_config: Some(PostprocessingConfig {
            apply_softmax: true,
            top_k: Some(5),
            confidence_threshold: Some(0.1),
        }),
    };

    let scaling_config = ScalingConfiguration {
        min_replicas: 1,
        max_replicas: 10,
        target_cpu_utilization: 70.0,
        target_memory_utilization: 80.0,
        scale_up_cooldown: 60,
        scale_down_cooldown: 300,
    };

    let monitoring_config = MonitoringConfiguration {
        enable_metrics: true,
        enable_logging: true,
        log_level: "INFO".to_string(),
        metrics_port: 8080,
        health_check_path: "/health".to_string(),
        prometheus_metrics: true,
    };

    Ok(DeploymentConfiguration {
        serving_config,
        scaling_config,
        monitoring_config,
        a_b_testing_config: None,
    })
}
