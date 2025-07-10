/// Temporal coordinate types and structures
pub mod temporal_types;

/// Oscillation analysis types and structures
pub mod oscillation_types;

/// Precision measurement types and structures
pub mod precision_types;

/// Client interface types and structures
pub mod client_types;

/// Error types and result structures
pub mod error_types;

// Re-export commonly used types
pub use temporal_types::*;
pub use oscillation_types::*;
pub use precision_types::*;
pub use client_types::*;
pub use error_types::*; 