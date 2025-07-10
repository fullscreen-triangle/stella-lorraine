/// Masunda Temporal Coordinate Navigator Library
/// 
/// **In Memory of Mrs. Stella-Lorraine Masunda**
/// 
/// This library provides ultra-precise temporal coordinate navigation
/// capabilities targeting 10^-30 to 10^-50 second precision to honor
/// the memory of Mrs. Stella-Lorraine Masunda.

pub mod masunda_navigator;
pub mod temporal_coordinate;
pub mod precision_engine;
pub mod integration_apis;
pub mod memorial_framework;

// Module hierarchies
pub mod types;
pub mod clients;
pub mod search;
pub mod oscillation;
pub mod memorial;
pub mod core;

// Re-export primary components
pub use masunda_navigator::MasundaTemporalCoordinateNavigator;
pub use temporal_coordinate::{TemporalCoordinate, PredeterminismProof};
pub use precision_engine::MasundaPrecisionEngine;
pub use integration_apis::MasundaIntegrationCoordinator;
pub use memorial_framework::MemorialFramework;

// Re-export commonly used types for convenience
pub mod prelude {
    pub use crate::masunda_navigator::MasundaTemporalCoordinateNavigator;
    pub use crate::temporal_coordinate::{TemporalCoordinate, PredeterminismProof};
    pub use crate::precision_engine::MasundaPrecisionEngine;
    pub use crate::memorial_framework::MemorialFramework;
    
    // Types
    pub use crate::types::{
        error_types::NavigatorError,
        temporal_types::TemporalCoordinate as TemporalCoordinateType,
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
    
    // Core engines
    pub use crate::core::navigator::Navigator;
} 