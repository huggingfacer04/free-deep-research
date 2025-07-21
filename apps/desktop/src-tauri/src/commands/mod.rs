pub mod api_management;
pub mod research_workflow;
pub mod template_management;
pub mod research;
pub mod config;
pub mod monitoring;
pub mod output_processor;
pub mod analytics;
pub mod performance;
pub mod v1_1_features;
pub mod v1_2_v2_0_features;

// V3.0.0 Commands - Global Intelligence Network
pub mod federated_research;
pub mod ai_marketplace;
pub mod quantum_ready;
pub mod nlp_engine;
pub mod blockchain;
pub mod knowledge_graph;
pub mod bmad_integration;

// Phase 4 Advanced Features Commands
pub mod ml_commands;
pub mod mobile_commands;
pub mod advanced_analytics;

// Phase 5.1: Computer Vision Commands
pub mod computer_vision;
pub mod image_classification;

pub use api_management::*;
pub use research::*;
pub use config::*;
pub use monitoring::*;
pub use analytics::*;
pub use performance::*;
pub use v1_1_features::*;
pub use v1_2_v2_0_features::*;

// V3.0.0 Command exports
pub use federated_research::*;
pub use ai_marketplace::*;
pub use quantum_ready::*;
pub use nlp_engine::*;
pub use blockchain::*;
pub use knowledge_graph::*;

// Phase 4 Advanced Features exports
pub use ml_commands::*;
pub use mobile_commands::*;
pub use advanced_analytics::*;

// Phase 5.1: Computer Vision exports
pub use computer_vision::*;
pub use image_classification::*;
