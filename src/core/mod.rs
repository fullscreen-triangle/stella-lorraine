//! Core modules for the Masunda Temporal Coordinate Navigator
//!
//! This module contains the fundamental components of the S-constant framework
//! and temporal coordinate navigation system.

/// S-constant framework implementation
pub mod s_constant;

/// Temporal coordinate navigation
pub mod temporal_navigation;

/// Oscillation convergence analysis
pub mod oscillation_convergence;

/// Memorial validation framework
pub mod memorial_validation;

/// Tri-dimensional S alignment
pub mod tri_dimensional_alignment;

// Re-export core types for convenience
pub use s_constant::SConstantFramework;
pub use temporal_navigation::TemporalNavigator;
pub use oscillation_convergence::OscillationAnalyzer;
pub use memorial_validation::MemorialValidator;
pub use tri_dimensional_alignment::TriDimensionalAligner;
