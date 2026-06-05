//! # Tempus Trigger Programs
//!
//! Implements the Tempus trigger program formalism from the runtime-operations
//! paper (§2):
//!
//! ```text
//! C_i  = { k | |ΔP(k)| < δ_i }            (timing cell)
//! P    = ∨_i ( ∧_{k ∈ K_i} C_i(k) )       (trigger condition)
//! ```
//!
//! A `TempusProgram` compiles to a vaHera `Fragment` via `compile_event`.
//! The compiled fragment is type-checked by the PVE and dispatched by the kernel.

use std::collections::{BTreeMap, HashMap};
use serde::{Deserialize, Serialize};

use crate::{
    partition::PartitionLabel,
    vahera::{Fragment, Value},
};

// ---------------------------------------------------------------------------
// Cell identity
// ---------------------------------------------------------------------------

/// Unique identifier for a timing cell.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellId(pub String);

impl CellId {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
}

impl std::fmt::Display for CellId { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) } }

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// A detector channel: identified by an integer `id` (e.g.\ ATLAS cell index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId(pub u32);

// ---------------------------------------------------------------------------
// Timing residual ΔP(k)
// ---------------------------------------------------------------------------

/// Timing residual for one detector channel:
/// `ΔP(k) = T_ref(k) − t_rec(k)` (in seconds).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimingResidual {
    pub channel: ChannelId,
    /// Value in seconds.  On-time signal → ΔP ≈ 0.
    pub delta_p_s: f64,
}

impl TimingResidual {
    pub fn new(channel: impl Into<ChannelId>, delta_p_s: f64) -> Self {
        Self { channel: channel.into(), delta_p_s }
    }
    /// Convert to nanoseconds.
    pub fn delta_p_ns(&self) -> f64 { self.delta_p_s * 1e9 }
}

impl From<u32> for ChannelId { fn from(v: u32) -> Self { Self(v) } }

// ---------------------------------------------------------------------------
// Timing cell
// ---------------------------------------------------------------------------

/// A timing cell `C_i = { k | |ΔP(k)| < δ_i }`.
///
/// The cell *fires* when all required channels have timing residuals within
/// the half-width.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingCell {
    pub id: CellId,
    /// Cell half-width in seconds (|ΔP(k)| < half_width fires the cell).
    pub half_width_s: f64,
    /// Channels that must all be within the half-width for the cell to fire.
    pub required_channels: Vec<ChannelId>,
    /// Optional minimum number of channels that must fire (default: all).
    pub min_channels: usize,
}

impl TimingCell {
    /// Create a cell requiring all channels to be within `half_width_s`.
    pub fn new(id: impl Into<String>, half_width_s: f64) -> Self {
        Self {
            id: CellId::new(id),
            half_width_s,
            required_channels: vec![],
            min_channels: 0,
        }
    }

    pub fn with_channels(mut self, channels: Vec<u32>) -> Self {
        self.required_channels = channels.into_iter().map(ChannelId).collect();
        self.min_channels = self.required_channels.len(); // all required by default
        self
    }

    pub fn with_min_channels(mut self, min: usize) -> Self {
        self.min_channels = min;
        self
    }

    /// Check if this cell fires given the provided timing residuals.
    pub fn fires(&self, residuals: &[TimingResidual]) -> bool {
        let residual_map: HashMap<u32, f64> = residuals.iter()
            .map(|r| (r.channel.0, r.delta_p_s.abs()))
            .collect();

        if self.required_channels.is_empty() {
            // No channel requirements — fires if at least one residual is in range.
            return residuals.iter().any(|r| r.delta_p_s.abs() < self.half_width_s);
        }

        let in_cell: usize = self.required_channels.iter()
            .filter(|ch| residual_map.get(&ch.0).copied().unwrap_or(f64::INFINITY) < self.half_width_s)
            .count();

        in_cell >= self.min_channels.max(1)
    }

    /// Compile this cell's firing condition as a vaHera Literal carrying a Bool.
    pub fn compile_firing(&self, residuals: &[TimingResidual]) -> Fragment {
        Fragment::literal(Value::Bool(self.fires(residuals)))
    }
}

// ---------------------------------------------------------------------------
// Cell registry
// ---------------------------------------------------------------------------

/// The cell registry `Γ = { C_i }` — the immutable partition of ΔP-space into
/// cells.  This is the PVE's frozen surface for timing-only triggers.
///
/// Once constructed, the registry cannot be mutated (Stability Contract).
#[derive(Debug, Clone, Default)]
pub struct CellRegistry {
    cells: HashMap<String, TimingCell>,
}

impl CellRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn with_cells(cells: Vec<TimingCell>) -> Self {
        let mut r = Self::new();
        for c in cells { r.insert(c); }
        r
    }

    /// Insert a cell.  Panics if a cell with the same id already exists
    /// (registry is append-only per the stability contract).
    pub fn insert(&mut self, cell: TimingCell) {
        let id = cell.id.0.clone();
        assert!(!self.cells.contains_key(&id), "duplicate cell id: {id}");
        self.cells.insert(id, cell);
    }

    pub fn get(&self, id: &str) -> Option<&TimingCell> { self.cells.get(id) }
    pub fn len(&self) -> usize { self.cells.len() }
    pub fn is_empty(&self) -> bool { self.cells.is_empty() }
    pub fn ids(&self) -> impl Iterator<Item = &str> { self.cells.keys().map(String::as_str) }
}

// ---------------------------------------------------------------------------
// Trigger path
// ---------------------------------------------------------------------------

/// One path `P_i` in the trigger condition: a conjunction of cell firings.
#[derive(Debug, Clone)]
pub struct TriggerPath {
    pub label: String,
    /// Cells that must ALL fire for this path to accept.
    pub required_cells: Vec<CellId>,
    /// Optional expected partition labels for type-checking.
    pub expected_labels: Option<Vec<PartitionLabel>>,
}

impl TriggerPath {
    pub fn new(label: impl Into<String>, cells: Vec<&str>) -> Self {
        Self {
            label: label.into(),
            required_cells: cells.into_iter().map(|s| CellId::new(s)).collect(),
            expected_labels: None,
        }
    }

    /// Check if this path fires.
    pub fn fires(&self, registry: &CellRegistry, residuals: &[TimingResidual]) -> bool {
        self.required_cells.iter().all(|cid| {
            registry.get(cid.0.as_str()).map_or(false, |cell| cell.fires(residuals))
        })
    }
}

// ---------------------------------------------------------------------------
// Tempus program
// ---------------------------------------------------------------------------

/// A Tempus trigger program: a disjunction of trigger paths.
///
/// Fires (accepts the event) if at least one path fires.
#[derive(Debug, Clone)]
pub struct TempusProgram {
    pub registry: CellRegistry,
    pub paths: Vec<TriggerPath>,
    /// Reference timing window τ_BX in seconds (computed from detector geometry).
    pub tau_bx_s: f64,
}

impl TempusProgram {
    pub fn new(registry: CellRegistry, paths: Vec<TriggerPath>, tau_bx_s: f64) -> Self {
        Self { registry, paths, tau_bx_s }
    }

    /// Standard LHC bunch-crossing period: τ_BX = Δx_ATLAS / c = 25 ns.
    pub const LHC_TAU_BX_S: f64 = 25e-9;

    /// Di-muon trigger: two opposite-sign muon channels within ±5 ns.
    pub fn dimuon(mut registry: CellRegistry) -> Self {
        registry.insert(
            TimingCell::new("cell_muon_pos", 5e-9)
                .with_channels(vec![1001])
        );
        registry.insert(
            TimingCell::new("cell_muon_neg", 5e-9)
                .with_channels(vec![1002])
        );
        let paths = vec![
            TriggerPath::new("di_muon", vec!["cell_muon_pos", "cell_muon_neg"]),
        ];
        Self::new(registry, paths, Self::LHC_TAU_BX_S)
    }

    /// Decide whether the event should be accepted.
    pub fn decide(&self, residuals: &[TimingResidual]) -> bool {
        self.paths.iter().any(|p| p.fires(&self.registry, residuals))
    }

    /// Compile the trigger decision for a given event into a vaHera `Fragment`.
    ///
    /// The compiled fragment is a `Compose` chain over per-path `Literal(Bool)` nodes
    /// followed by a logical-OR `Call` that reduces them to a single `Bool`.
    pub fn compile_event(&self, residuals: &[TimingResidual]) -> Fragment {
        if self.paths.is_empty() {
            return Fragment::literal(Value::Bool(false));
        }

        // Each path compiles to a Literal(Bool).
        let path_frags: Vec<Fragment> = self.paths.iter()
            .map(|p| Fragment::literal(Value::Bool(p.fires(&self.registry, residuals))))
            .collect();

        if path_frags.len() == 1 {
            return path_frags.into_iter().next().unwrap();
        }

        // Compose chain: carry = result of first path; each subsequent stage OR's with carry.
        let mut stages = Vec::with_capacity(path_frags.len() + 1);
        // First stage: the first path's result.
        stages.push(path_frags[0].clone());
        // Subsequent stages: OR with accumulated carry.
        for frag in &path_frags[1..] {
            let mut args = BTreeMap::new();
            args.insert("rhs".to_string(), frag.clone());
            stages.push(Fragment::Call {
                op: crate::vahera::OperationName::new("logical_or"),
                args,
            });
        }
        Fragment::compose(stages)
    }

    /// Number of trigger paths.
    pub fn num_paths(&self) -> usize { self.paths.len() }
}

// ---------------------------------------------------------------------------
// Timing window from detector geometry
//
// Theorem 3.1 (runtime-operations paper):
//   τ_BX = Δx_detector / c
// ---------------------------------------------------------------------------

/// Compute the bunch crossing period from detector geometry.
///
/// For ATLAS: `Δx = 7.5 m`, giving `τ_BX = 25 ns`.
pub fn tau_bx_from_geometry(detector_half_diameter_m: f64) -> f64 {
    const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;
    detector_half_diameter_m / SPEED_OF_LIGHT_M_S
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residuals(pairs: &[(u32, f64)]) -> Vec<TimingResidual> {
        pairs.iter().map(|(ch, dp)| TimingResidual::new(*ch, *dp)).collect()
    }

    #[test]
    fn timing_cell_fires_within_halfwidth() {
        let cell = TimingCell::new("test", 5e-9).with_channels(vec![1]);
        assert!(cell.fires(&residuals(&[(1, 3e-9)])));   // |3 ns| < 5 ns ✓
        assert!(!cell.fires(&residuals(&[(1, 6e-9)]))); // |6 ns| > 5 ns ✗
    }

    #[test]
    fn timing_cell_misses_if_channel_absent() {
        let cell = TimingCell::new("test", 5e-9).with_channels(vec![1]);
        assert!(!cell.fires(&residuals(&[(2, 1e-9)]))); // channel 1 absent ✗
    }

    #[test]
    fn dimuon_program_accepts_valid_event() {
        let prog = TempusProgram::dimuon(CellRegistry::new());
        // Both muon channels within 5 ns.
        let r = residuals(&[(1001, 1e-9), (1002, -2e-9)]);
        assert!(prog.decide(&r));
    }

    #[test]
    fn dimuon_program_rejects_late_muon() {
        let prog = TempusProgram::dimuon(CellRegistry::new());
        // Second muon is 7 ns late.
        let r = residuals(&[(1001, 1e-9), (1002, 7e-9)]);
        assert!(!prog.decide(&r));
    }

    #[test]
    fn compile_event_returns_bool_literal_for_single_path() {
        let prog = TempusProgram::dimuon(CellRegistry::new());
        // Only one path → compiles to Literal(Bool).
        let r = residuals(&[(1001, 1e-9), (1002, 2e-9)]);
        let frag = prog.compile_event(&r);
        match frag {
            Fragment::Literal(Value::Bool(true)) => {}
            other => panic!("expected Literal(Bool(true)), got {other:?}"),
        }
    }

    #[test]
    fn tau_bx_from_atlas_geometry() {
        let tau = tau_bx_from_geometry(7.5);
        // Should be ≈ 25.02 ns (7.5 m / c ≈ 25.02 ns)
        let tau_ns = tau * 1e9;
        assert!((tau_ns - 25.02).abs() < 0.1, "τ_BX ≈ 25 ns, got {tau_ns:.3} ns");
    }
}
