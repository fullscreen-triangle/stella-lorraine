/// Masunda Temporal Coordinate Navigator Library
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This library provides ultra-precise temporal coordinate navigation
/// capabilities targeting 10^-30 to 10^-50 second precision to honor
/// the memory of Mrs. Stella-Lorraine Masunda.

// Module hierarchies
pub mod types;
pub mod clients;
pub mod search;
pub mod oscillation;
pub mod memorial;
pub mod core;

// Integration APIs (keep for external system integration)
pub mod integration_apis;

// Re-export primary components from the new organized structure
pub use core::navigator::Navigator;
pub use core::precision_engine::PrecisionEngine;
pub use core::memorial_framework::MemorialFramework;
pub use core::oscillation_convergence::OscillationConvergence;
pub use core::temporal_coordinates::TemporalCoordinates;

pub use types::temporal_types::TemporalCoordinate;
pub use types::error_types::NavigatorError;
pub use integration_apis::MasundaIntegrationCoordinator;

// Re-export commonly used types for convenience
pub mod prelude {
    // Core components
    pub use crate::core::navigator::{Navigator, NavigatorConfig};
    pub use crate::core::precision_engine::{PrecisionEngine, PrecisionEngineConfig, PrecisionOperation, PrecisionCalibrationData};
    pub use crate::core::memorial_framework::{MemorialFramework, MemorialFrameworkConfig, MemorialOperation, MemorialContext};
    pub use crate::core::oscillation_convergence::{OscillationConvergence, OscillationConvergenceConfig, ConvergenceOperation, ConvergenceContext};
    pub use crate::core::temporal_coordinates::{TemporalCoordinates, TemporalCoordinatesConfig, CoordinateOperation, CoordinateContext};
    
    // Types
    pub use crate::types::{
        error_types::NavigatorError,
        temporal_types::TemporalCoordinate,
        oscillation_types::{OscillationState, OscillationMetrics},
        precision_types::PrecisionLevel,
    };
    
    // Clients
    pub use crate::clients::{
        KambuzumaClient,
        KwasaKwasaClient,
        MzekezekeClient,
        BuheraClient,
        ConsciousnessClient,
    };
    
    // Search coordination
    pub use crate::search::{
        CoordinateSearchEngine,
        QuantumCoordinator,
        SemanticCoordinator,
        AuthCoordinator,
        EnvironmentalCoordinator,
        ConsciousnessCoordinator,
    };
    
    // Oscillation processing
    pub use crate::oscillation::{
        ConvergenceDetector,
        TerminationProcessor,
        HierarchicalAnalyzer,
        EndpointCollector,
    };
    
    // Memorial framework
    pub use crate::memorial::{
        MasundaMemorialFramework,
        CosmicSignificanceValidator,
        PredeterminismValidator,
    };
    
    // Integration APIs
    pub use crate::integration_apis::MasundaIntegrationCoordinator;
} 