//! # Tempus — Partition-Physics Trigger Kernel
//!
//! A real-time event-selection substrate grounded in the partition algebra of
//! bounded phase space.  The implementation realises every construction proved
//! in the monograph series:
//!
//! - **vaHera AST** (four frozen variants: `Literal`, `Call`, `Compose`, `Hole`)
//! - **Partition-coordinate refinement types** (selection rules as type predicates)
//! - **Five-subsystem dispatch kernel** (PVE · CMM · PSS · DIC · TEM)
//! - **Tempus trigger programs** (timing cells, cell registry, disjunctive conditions)
//! - **LHC instantiation** (four detector-subsystem providers, parallel PSS dispatch)
//!
//! ## Architecture
//!
//! ```text
//! TempusProgram
//!   └─ compiles to ──► Fragment (vaHera AST)
//!                          │
//!                     Kernel::dispatch()
//!                          │
//!          ┌───────────────┼────────────────────┐
//!         PVE             CMM                  PSS
//!      (typecheck)    (memo cache)        (parallel scheduler)
//!          │               │                    │
//!          └───────────────┴────────────────────┘
//!                          │
//!                       Executor
//!                  (backward completion)
//!                          │
//!                         TEM
//!                   (efficiency monitor)
//! ```
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use tempus::{
//!     kernel::Kernel,
//!     trigger::{TempusProgram, TimingCell, CellRegistry},
//!     lhc::AtlasKernel,
//! };
//!
//! // Build a di-muon trigger program
//! let registry = CellRegistry::with_cells(vec![
//!     TimingCell::new("mu_mu", 5.0e-9), // ±5 ns cell
//! ]);
//! let program = TempusProgram::dimuon(registry);
//!
//! // Instantiate an ATLAS kernel and dispatch an event
//! let mut kernel = AtlasKernel::new();
//! let fragment = program.compile_event(&event);
//! let result = kernel.dispatch(fragment);
//! ```

pub mod error;
pub mod vahera;
pub mod partition;
pub mod selection;
pub mod typecheck;
pub mod kernel;
pub mod trigger;
pub mod lhc;

pub use error::{TempusError, Result};
pub use vahera::{Fragment, Value, OperationName};
pub use partition::{PartitionLabel, Spin};
pub use selection::{SelectionRule, SelectionRegistry};
pub use kernel::{Kernel, DispatchEvent, DispatchOutcome};
pub use trigger::{TempusProgram, TimingCell, CellRegistry, CellId};
