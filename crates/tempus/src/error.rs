//! Error types for the Tempus kernel.

use thiserror::Error;
use crate::partition::PartitionLabel;

pub type Result<T> = std::result::Result<T, TempusError>;

#[derive(Debug, Error)]
pub enum TempusError {
    // --- vaHera errors ---------------------------------------------------
    #[error("hole '{name}' was not resolved before dispatch")]
    UnresolvedHole { name: String },

    #[error("operation '{op}' is not registered in the operation registry")]
    UnregisteredOperation { op: String },

    #[error("argument '{key}' missing for operation '{op}'")]
    MissingArgument { op: String, key: String },

    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    // --- Partition / selection-rule errors --------------------------------
    #[error("partition coordinate out of range: n={n}, l={l}, m={m}")]
    InvalidPartitionCoord { n: u32, l: u32, m: i32 },

    #[error(
        "forbidden transition from {from:?} to {to:?}: \
         violates rule '{rule}'"
    )]
    ForbiddenTransition {
        from: PartitionLabel,
        to: PartitionLabel,
        rule: String,
    },

    // --- Kernel errors ---------------------------------------------------
    #[error("PVE rejected fragment: {reason}")]
    ValidationRejected { reason: String },

    #[error("executor returned no value for fragment")]
    ExecutorFailed,

    // --- Timing / trigger errors -----------------------------------------
    #[error("timing residual {delta_p_ns:.3} ns exceeds cell half-width {hw_ns:.3} ns for channel {channel}")]
    CellMiss { channel: u32, delta_p_ns: f64, hw_ns: f64 },

    #[error("cell id '{0}' not found in registry")]
    UnknownCell(String),
}
