#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::image_classification::*;
    use uuid::Uuid;
    use chrono::Utc;

    fn create_test_config() -> ImageClassificationConfig {
        ImageClassificationConfig {
            max_concurrent_training_jobs: 2,
            default_training_timeout_hours: 1,
            model_storage_path: "/tmp/test_models".to_string(),
            dataset_storage_path: "/tmp/test_datasets".to_string(),
            inference_cache_ttl_seconds: 300,
            gpu_memory_limit_gb: 4.0,
            enable_model_versioning: true,
            enable_a_b_testing: false,
        }
    }

    fn create_test_training_config() -> TrainingConfiguration {
        TrainingConfiguration {
            dataset_config: DatasetConfiguration {
                dataset_name: "test_dataset".to_string(),
                dataset_path: "/tmp/test_dataset".to_string(),
                num_classes: 10,
                class_names: (0..10).map(|i| format!("class_{}", i)).collect(),
                train_split: 0.7,
                validation_split: 0.2,
                test_split: 0.1,
                image_size: (224, 224),
                batch_size: 32,
                shuffle: true,
            },
            training_params: TrainingParameters {
                epochs: 5,
                learning_rate: 0.001,
                optimizer: OptimizerConfig {
                    optimizer_type: OptimizerType::Adam,
                    parameters: std::collections::HashMap::from([
                        ("beta_1".to_string(), 0.9),
                        ("beta_2".to_string(), 0.999),
                    ]),
                },
                loss_function: "categorical_crossentropy".to_string(),
                metrics: vec!["accuracy".to_string()],
                regularization: Some(RegularizationConfig {
                    l1_lambda: None,
                    l2_lambda: Some(0.001),
                    dropout_rate: Some(0.5),
                }),
            },
            augmentation_config: Some(DataAugmentationConfig {
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
            }),
            validation_config: ValidationConfiguration {
                validation_frequency: 1,
                validation_metrics: vec!["accuracy".to_string(), "loss".to_string()],
                save_best_model: true,
                monitor_metric: "val_accuracy".to_string(),
                mode: ValidationMode::Max,
            },
            early_stopping: Some(EarlyStoppingConfig {
                monitor: "val_accuracy".to_string(),
                patience: 3,
                min_delta: 0.001,
                mode: ValidationMode::Max,
                restore_best_weights: true,
            }),
            learning_rate_schedule: Some(LearningRateSchedule {
                schedule_type: LearningRateScheduleType::ReduceOnPlateau,
                parameters: std::collections::HashMap::from([
                    ("factor".to_string(), 0.5),
                    ("patience".to_string(), 2.0),
                ]),
            }),
        }
    }

    #[tokio::test]
    async fn test_service_creation() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await;
        assert!(service.is_ok());
    }

    #[tokio::test]
    async fn test_model_creation() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        let model_id = service.create_model(
            "Test Model".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::CNN {
                layers: vec![
                    CNNLayer {
                        layer_type: CNNLayerType::Conv2D,
                        filters: Some(32),
                        kernel_size: Some((3, 3)),
                        stride: Some((1, 1)),
                        padding: Some("same".to_string()),
                        activation: Some("relu".to_string()),
                    },
                ],
                input_shape: (224, 224, 3),
            },
            Uuid::new_v4(),
        ).await;

        assert!(model_id.is_ok());
        let model_id = model_id.unwrap();

        // Verify model was created
        let model = service.get_model(model_id).await.unwrap();
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.name, "Test Model");
        assert_eq!(model.status, ModelStatus::Draft);
    }

    #[tokio::test]
    async fn test_training_job_creation() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        // Create a model first
        let model_id = service.create_model(
            "Training Test Model".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::ResNet {
                variant: ResNetVariant::ResNet50,
                pretrained: true,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        // Start training
        let training_config = create_test_training_config();
        let user_id = Uuid::new_v4();
        
        let job_id = service.start_training(model_id, training_config, user_id).await;
        assert!(job_id.is_ok());
        let job_id = job_id.unwrap();

        // Verify training job was created
        let job = service.get_training_job(job_id).await.unwrap();
        assert!(job.is_some());
        let job = job.unwrap();
        assert_eq!(job.model_id, model_id);
        assert_eq!(job.user_id, user_id);
        assert_eq!(job.status, TrainingJobStatus::Queued);
    }

    #[tokio::test]
    async fn test_training_progress_update() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        // Create model and start training
        let model_id = service.create_model(
            "Progress Test Model".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::EfficientNet {
                variant: EfficientNetVariant::B0,
                pretrained: true,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        let training_config = create_test_training_config();
        let job_id = service.start_training(model_id, training_config, Uuid::new_v4()).await.unwrap();

        // Update training progress
        let metrics = TrainingMetrics {
            epoch: 1,
            training_loss: 0.5,
            training_accuracy: 0.8,
            validation_loss: 0.6,
            validation_accuracy: 0.75,
            learning_rate: 0.001,
            timestamp: Utc::now(),
        };

        let result = service.update_training_progress(job_id, 1, metrics).await;
        assert!(result.is_ok());

        // Verify progress was updated
        let job = service.get_training_job(job_id).await.unwrap().unwrap();
        assert_eq!(job.current_epoch, 1);
        assert!(job.current_metrics.is_some());
        assert_eq!(job.current_metrics.unwrap().training_accuracy, 0.8);
    }

    #[tokio::test]
    async fn test_training_completion() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        // Create model and start training
        let model_id = service.create_model(
            "Completion Test Model".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::PyTorch,
            ModelArchitecture::VisionTransformer {
                patch_size: 16,
                embed_dim: 768,
                num_heads: 12,
                num_layers: 12,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        let training_config = create_test_training_config();
        let job_id = service.start_training(model_id, training_config, Uuid::new_v4()).await.unwrap();

        // Complete training
        let final_metrics = AccuracyMetrics {
            accuracy: 0.95,
            precision: 0.94,
            recall: 0.96,
            f1_score: 0.95,
            top_5_accuracy: Some(0.99),
            confusion_matrix: None,
            class_metrics: std::collections::HashMap::new(),
            validation_loss: 0.15,
            training_loss: 0.12,
            inference_time_ms: 45.0,
            model_size_mb: 25.5,
        };

        let result = service.complete_training(job_id, final_metrics).await;
        assert!(result.is_ok());

        // Verify job completion
        let job = service.get_training_job(job_id).await.unwrap().unwrap();
        assert_eq!(job.status, TrainingJobStatus::Completed);

        // Verify model status update
        let model = service.get_model(model_id).await.unwrap().unwrap();
        assert_eq!(model.status, ModelStatus::Trained);
        assert!(model.accuracy_metrics.is_some());
        assert_eq!(model.accuracy_metrics.unwrap().accuracy, 0.95);
    }

    #[tokio::test]
    async fn test_model_deployment() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        // Create and train a model
        let model_id = service.create_model(
            "Deployment Test Model".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::ResNet {
                variant: ResNetVariant::ResNet50,
                pretrained: true,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        // Simulate trained model
        {
            let mut models = service.models.write().await;
            if let Some(model) = models.get_mut(&model_id) {
                model.update_status(ModelStatus::Trained);
                model.update_metrics(AccuracyMetrics {
                    accuracy: 0.92,
                    precision: 0.91,
                    recall: 0.93,
                    f1_score: 0.92,
                    top_5_accuracy: Some(0.98),
                    confusion_matrix: None,
                    class_metrics: std::collections::HashMap::new(),
                    validation_loss: 0.18,
                    training_loss: 0.15,
                    inference_time_ms: 50.0,
                    model_size_mb: 30.0,
                });
            }
        }

        // Deploy model
        let deployment_config = DeploymentConfiguration {
            serving_config: ServingConfiguration {
                model_name: "test_model".to_string(),
                model_version: "1.0.0".to_string(),
                serving_platform: ServingPlatform::TensorFlowServing,
                input_signature: InputSignature {
                    input_name: "input_image".to_string(),
                    input_shape: vec![-1, 224, 224, 3],
                    input_dtype: "float32".to_string(),
                    preprocessing_required: true,
                },
                output_signature: OutputSignature {
                    output_name: "predictions".to_string(),
                    output_shape: vec![-1, 10],
                    output_dtype: "float32".to_string(),
                    class_names: Some((0..10).map(|i| format!("class_{}", i)).collect()),
                },
                preprocessing_config: None,
                postprocessing_config: None,
            },
            scaling_config: ScalingConfiguration {
                min_replicas: 1,
                max_replicas: 5,
                target_cpu_utilization: 70.0,
                target_memory_utilization: 80.0,
                scale_up_cooldown: 60,
                scale_down_cooldown: 300,
            },
            monitoring_config: MonitoringConfiguration {
                enable_metrics: true,
                enable_logging: true,
                log_level: "INFO".to_string(),
                metrics_port: 8080,
                health_check_path: "/health".to_string(),
                prometheus_metrics: true,
            },
            a_b_testing_config: None,
        };

        let endpoint = service.deploy_model(model_id, deployment_config).await;
        assert!(endpoint.is_ok());

        // Verify model deployment
        let model = service.get_model(model_id).await.unwrap().unwrap();
        assert_eq!(model.status, ModelStatus::Deployed);
        assert!(model.deployment_config.is_some());
    }

    #[tokio::test]
    async fn test_image_classification_inference() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        // Create and deploy a model
        let model_id = service.create_model(
            "Inference Test Model".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::ResNet {
                variant: ResNetVariant::ResNet50,
                pretrained: true,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        // Simulate deployed model
        {
            let mut models = service.models.write().await;
            if let Some(model) = models.get_mut(&model_id) {
                model.update_status(ModelStatus::Deployed);
                model.deployment_config = Some(DeploymentConfiguration {
                    serving_config: ServingConfiguration {
                        model_name: "test_model".to_string(),
                        model_version: "1.0.0".to_string(),
                        serving_platform: ServingPlatform::TensorFlowServing,
                        input_signature: InputSignature {
                            input_name: "input_image".to_string(),
                            input_shape: vec![-1, 224, 224, 3],
                            input_dtype: "float32".to_string(),
                            preprocessing_required: true,
                        },
                        output_signature: OutputSignature {
                            output_name: "predictions".to_string(),
                            output_shape: vec![-1, 10],
                            output_dtype: "float32".to_string(),
                            class_names: Some((0..10).map(|i| format!("class_{}", i)).collect()),
                        },
                        preprocessing_config: None,
                        postprocessing_config: None,
                    },
                    scaling_config: ScalingConfiguration {
                        min_replicas: 1,
                        max_replicas: 5,
                        target_cpu_utilization: 70.0,
                        target_memory_utilization: 80.0,
                        scale_up_cooldown: 60,
                        scale_down_cooldown: 300,
                    },
                    monitoring_config: MonitoringConfiguration {
                        enable_metrics: true,
                        enable_logging: true,
                        log_level: "INFO".to_string(),
                        metrics_port: 8080,
                        health_check_path: "/health".to_string(),
                        prometheus_metrics: true,
                    },
                    a_b_testing_config: None,
                });
            }
        }

        // Perform inference
        let request = ImageClassificationRequest {
            request_id: Uuid::new_v4(),
            model_id,
            image_data: ImageData::Base64 {
                data: "fake_base64_data".to_string(),
                format: "jpeg".to_string(),
            },
            inference_config: InferenceConfig {
                top_k: Some(3),
                confidence_threshold: Some(0.1),
                return_probabilities: true,
                return_features: false,
                batch_size: Some(1),
            },
            user_id: Uuid::new_v4(),
            timestamp: Utc::now(),
        };

        let response = service.classify_image(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        assert_eq!(response.model_id, model_id);
        assert!(!response.predictions.is_empty());
        assert!(response.inference_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_concurrent_training_limit() {
        let mut config = create_test_config();
        config.max_concurrent_training_jobs = 1; // Set limit to 1
        let service = ImageClassificationService::new(config).await.unwrap();

        // Create two models
        let model_id_1 = service.create_model(
            "Model 1".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::ResNet {
                variant: ResNetVariant::ResNet50,
                pretrained: true,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        let model_id_2 = service.create_model(
            "Model 2".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::ResNet {
                variant: ResNetVariant::ResNet50,
                pretrained: true,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        let training_config = create_test_training_config();
        let user_id = Uuid::new_v4();

        // Start first training job - should succeed
        let job_1 = service.start_training(model_id_1, training_config.clone(), user_id).await;
        assert!(job_1.is_ok());

        // Start second training job - should fail due to limit
        let job_2 = service.start_training(model_id_2, training_config, user_id).await;
        assert!(job_2.is_err());
    }

    #[tokio::test]
    async fn test_training_statistics() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        // Create some models and jobs
        let model_id = service.create_model(
            "Stats Test Model".to_string(),
            ImageModelType::GeneralClassification,
            MLFramework::TensorFlow,
            ModelArchitecture::ResNet {
                variant: ResNetVariant::ResNet50,
                pretrained: true,
            },
            Uuid::new_v4(),
        ).await.unwrap();

        let training_config = create_test_training_config();
        let job_id = service.start_training(model_id, training_config, Uuid::new_v4()).await.unwrap();

        // Get statistics
        let stats = service.get_training_statistics().await.unwrap();
        assert_eq!(stats.total_models, 1);
        assert_eq!(stats.total_jobs, 1);
        assert_eq!(stats.active_jobs, 0); // Job is queued, not active
        assert_eq!(stats.completed_jobs, 0);
        assert_eq!(stats.failed_jobs, 0);
        assert_eq!(stats.deployed_models, 0);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        // Test cache cleanup
        let result = service.cleanup_cache().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_model_architecture_variants() {
        let config = create_test_config();
        let service = ImageClassificationService::new(config).await.unwrap();

        let user_id = Uuid::new_v4();

        // Test different architectures
        let architectures = vec![
            ModelArchitecture::ResNet {
                variant: ResNetVariant::ResNet18,
                pretrained: false,
            },
            ModelArchitecture::EfficientNet {
                variant: EfficientNetVariant::B0,
                pretrained: true,
            },
            ModelArchitecture::VisionTransformer {
                patch_size: 16,
                embed_dim: 768,
                num_heads: 12,
                num_layers: 12,
            },
        ];

        for (i, architecture) in architectures.into_iter().enumerate() {
            let model_id = service.create_model(
                format!("Architecture Test Model {}", i),
                ImageModelType::GeneralClassification,
                MLFramework::TensorFlow,
                architecture,
                user_id,
            ).await;
            assert!(model_id.is_ok());
        }

        // Verify all models were created
        let user_models = service.list_user_models(user_id).await.unwrap();
        assert_eq!(user_models.len(), 3);
    }
}
