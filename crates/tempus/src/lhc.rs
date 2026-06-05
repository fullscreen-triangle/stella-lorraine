//! # LHC-Specific Kernel Instantiation
//!
//! Provides the ATLAS kernel and the four detector-subsystem providers
//! (tracker/n, ECAL/ℓ, HCAL/m, muon/s) that map to the four partition
//! coordinates.
//!
//! ## Physics grounding
//!
//! Each detector subsystem is registered as a vaHera operation whose output
//! carries one partition coordinate.  Because the four coordinates commute
//! (QND / Jackson independence), the PSS dispatches them in parallel.
//!
//! | Provider       | Coordinate | Physical quantity       |
//! |----------------|------------|-------------------------|
//! | `op_n`         | n          | p_T (cyclotron scale)   |
//! | `op_l`         | ℓ          | shower shape (E1 level) |
//! | `op_m`         | m          | ECAL/HCAL energy ratio  |
//! | `op_s`         | s          | track curvature sign    |
//!
//! ## Planck depth
//!
//! `n_P = 60` for LHC (τ_BX = 25 ns, d = 3, t_P ≈ 5.39 × 10⁻⁴⁴ s).

use std::collections::BTreeMap;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::{
    kernel::{
        DispatchEvent, EfficiencyMonitor, FetchAllDic, HashCmm,
        Kernel, MonitorSubsystem, ParallelPss, ValidationSubsystem,
    },
    partition::{PartitionLabel, Spin, planck_depth},
    selection::SelectionRegistry,
    trigger::{CellRegistry, TempusProgram, TimingResidual, tau_bx_from_geometry},
    typecheck::PartitionTypePve,
    vahera::{Fragment, OperationRegistry, Value},
    Result,
};

// ---------------------------------------------------------------------------
// Physical constants for LHC
// ---------------------------------------------------------------------------

pub const LHC_PLANCK_DEPTH: u32 = 60;
/// LHC bunch crossing period (s): τ_BX = 25 ns
pub const LHC_TAU_BX_S: f64 = 25e-9;
/// ATLAS interaction region half-diameter (m).
pub const ATLAS_HALF_DIAMETER_M: f64 = 7.5;
/// Reference momentum scale for principal number: p_ref = 1 GeV/c.
pub const P_REF_GEV: f64 = 1.0;
/// ATLAS solenoid field (T).
pub const ATLAS_B_FIELD_T: f64 = 2.0;

// ---------------------------------------------------------------------------
// Partition coordinate extraction from detector hits
// ---------------------------------------------------------------------------

/// Extract principal number n from transverse momentum p_T (GeV/c).
///
/// `n = floor(sqrt(p_T / p_ref)) + 1`
pub fn n_from_pt(pt_gev: f64) -> u32 {
    ((pt_gev / P_REF_GEV).sqrt().floor() as u32).saturating_add(1)
}

/// Extract angular sublevel ℓ from EM shower shape parameter.
///
/// ℓ = 0 for electromagnetic (narrow) showers; ℓ ≥ 1 for hadronic.
/// The shower shape parameter `f_em ∈ [0,1]` is the EM fraction of total energy.
pub fn l_from_em_fraction(n: u32, f_em: f64) -> u32 {
    // EM-dominant shower → ℓ=0; hadronic → ℓ=1..n-1
    if f_em >= 0.9 { 0 } else { (n - 1).min(1) }
}

/// Extract azimuthal projection m from ECAL/HCAL energy ratio.
///
/// Maps the ratio to m ∈ [−ℓ, +ℓ] by linear scaling.
pub fn m_from_energy_ratio(l: u32, ratio: f64) -> i32 {
    if l == 0 { return 0; }
    let scaled = (ratio - 0.5) * 2.0 * l as f64;
    scaled.round() as i32
}

/// Extract spin s from track curvature sign (+1 = positive charge).
pub fn s_from_charge_sign(positive: bool) -> Spin {
    if positive { Spin::Up } else { Spin::Down }
}

// ---------------------------------------------------------------------------
// Detector subsystem providers (vaHera operations)
// ---------------------------------------------------------------------------

/// Register the four LHC detector subsystem operations into a registry.
///
/// Each operation takes detector measurements as named arguments and returns
/// a `Value::Label` carrying the extracted partition coordinate embedded in a
/// `PartitionLabel` with the other coordinates at their ground values.
pub fn register_detector_ops(registry: &mut OperationRegistry) {
    // op_n: tracker → principal n
    registry.register("op_n", Box::new(|args: BTreeMap<String, Value>| {
        let pt_gev = args.get("pt_gev")
            .and_then(Value::as_num)
            .unwrap_or(1.0);
        let n = n_from_pt(pt_gev).min(LHC_PLANCK_DEPTH);
        let label = PartitionLabel::new(n.max(1), 0, 0, Spin::Up)?;
        Ok(Value::Label(label))
    }));

    // op_l: ECAL → angular ℓ
    registry.register("op_l", Box::new(|args: BTreeMap<String, Value>| {
        let n = args.get("n").and_then(Value::as_num).map(|v| v as u32).unwrap_or(1);
        let f_em = args.get("f_em").and_then(Value::as_num).unwrap_or(1.0);
        let l = l_from_em_fraction(n.max(1), f_em);
        let label = PartitionLabel::new(n.max(1), l, 0, Spin::Up)?;
        Ok(Value::Label(label))
    }));

    // op_m: HCAL → projection m
    registry.register("op_m", Box::new(|args: BTreeMap<String, Value>| {
        let n = args.get("n").and_then(Value::as_num).map(|v| v as u32).unwrap_or(1);
        let l = args.get("l").and_then(Value::as_num).map(|v| v as u32).unwrap_or(0);
        let ratio = args.get("ratio").and_then(Value::as_num).unwrap_or(0.5);
        let m = m_from_energy_ratio(l, ratio);
        let label = PartitionLabel::new(n.max(1), l, m, Spin::Up)?;
        Ok(Value::Label(label))
    }));

    // op_s: muon spectrometer → spin s (charge sign)
    registry.register("op_s", Box::new(|args: BTreeMap<String, Value>| {
        let positive = args.get("positive_charge")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let n = args.get("n").and_then(Value::as_num).map(|v| v as u32).unwrap_or(1);
        let l = args.get("l").and_then(Value::as_num).map(|v| v as u32).unwrap_or(0);
        let m = args.get("m").and_then(Value::as_num).map(|v| v as i32).unwrap_or(0);
        let spin = s_from_charge_sign(positive);
        let label = PartitionLabel::new(n.max(1), l, m, spin)?;
        Ok(Value::Label(label))
    }));

    // logical_or: reduce two Bool values under Compose carry-threading
    registry.register("logical_or", Box::new(|args: BTreeMap<String, Value>| {
        let carry = args.get("input").and_then(Value::as_bool).unwrap_or(false);
        let rhs   = args.get("rhs").and_then(Value::as_bool).unwrap_or(false);
        Ok(Value::Bool(carry || rhs))
    }));
}

// ---------------------------------------------------------------------------
// ATLAS PVE — partition refinement types + LHC selection rules
// ---------------------------------------------------------------------------

/// Build the LHC PVE: E1 + spin-conserved selection rules.
pub fn atlas_pve() -> PartitionTypePve {
    PartitionTypePve::new(SelectionRegistry::lhc())
}

// ---------------------------------------------------------------------------
// ATLAS TEM — η_C monitor with timing histogram
// ---------------------------------------------------------------------------

pub struct AtlasTem {
    efficiency: EfficiencyMonitor,
    delta_p_samples: Mutex<Vec<f64>>,
}

impl AtlasTem {
    pub fn new() -> Self {
        Self {
            efficiency: EfficiencyMonitor::new(),
            delta_p_samples: Mutex::new(Vec::new()),
        }
    }

    pub fn eta_c(&self) -> f64 { self.efficiency.eta_c() }
    pub fn dispatched(&self) -> u64 { self.efficiency.dispatched() }
    pub fn mean_latency_ns(&self) -> f64 { self.efficiency.mean_latency_ns() }
    pub fn delta_p_samples(&self) -> Vec<f64> { self.delta_p_samples.lock().clone() }
}

impl Default for AtlasTem { fn default() -> Self { Self::new() } }

impl MonitorSubsystem for AtlasTem {
    fn observe(&self, event: &DispatchEvent) {
        self.efficiency.observe(event);
        // Extract ΔP value from event metadata if present.
        // (In production, events would carry ΔP in a structured field.)
    }
}

// ---------------------------------------------------------------------------
// AtlasKernel — the fully instantiated LHC kernel
// ---------------------------------------------------------------------------

pub type AtlasKernel = Kernel<PartitionTypePve, HashCmm, ParallelPss, FetchAllDic, Arc<AtlasTem>>;

/// Build the ATLAS kernel with all five subsystems instantiated.
pub fn atlas_kernel() -> (AtlasKernel, Arc<AtlasTem>) {
    let mut registry = OperationRegistry::default();
    register_detector_ops(&mut registry);

    let pve = atlas_pve();
    let cmm = HashCmm::new();
    let pss = ParallelPss::lhc();
    let dic = FetchAllDic;
    let tem = Arc::new(AtlasTem::new());

    let kernel = Kernel::new(registry, pve, cmm, pss, dic, Arc::clone(&tem));
    (kernel, tem)
}

impl MonitorSubsystem for Arc<AtlasTem> {
    fn observe(&self, event: &DispatchEvent) {
        // Deref Arc<AtlasTem> → AtlasTem, then call the AtlasTem impl.
        self.efficiency.observe(event);
    }
}

// ---------------------------------------------------------------------------
// Di-muon complete event trace (from runtime-operations paper §6)
// ---------------------------------------------------------------------------

/// A detected LHC event (minimally described for the trigger).
#[derive(Debug, Clone)]
pub struct DetectedEvent {
    /// Timing residuals for all fired channels.
    pub residuals: Vec<TimingResidual>,
    /// Tracker: transverse momentum of each track (GeV/c).
    pub track_pt: Vec<f64>,
    /// Tracker: charge sign of each track (+1 = positive).
    pub track_charge_positive: Vec<bool>,
    /// ECAL: EM energy fraction for each track.
    pub em_fraction: Vec<f64>,
    /// HCAL: ECAL/HCAL energy ratio for each track.
    pub energy_ratio: Vec<f64>,
}

impl DetectedEvent {
    /// Compile this event into a vaHera `Fragment` for kernel dispatch.
    pub fn compile(&self, program: &TempusProgram) -> Fragment {
        program.compile_event(&self.residuals)
    }

    /// Construct the partition label of the leading track.
    pub fn leading_partition_label(&self) -> Result<PartitionLabel> {
        let pt = self.track_pt.first().copied().unwrap_or(1.0);
        let positive = self.track_charge_positive.first().copied().unwrap_or(true);
        let f_em = self.em_fraction.first().copied().unwrap_or(1.0);
        let ratio = self.energy_ratio.first().copied().unwrap_or(0.5);

        let n = n_from_pt(pt).min(LHC_PLANCK_DEPTH).max(1);
        let l = l_from_em_fraction(n, f_em);
        let m = m_from_energy_ratio(l, ratio);
        let s = s_from_charge_sign(positive);
        PartitionLabel::new(n, l, m, s)
    }
}

/// Build a typical Z→μ⁺μ⁻ test event.
pub fn dimuon_test_event() -> DetectedEvent {
    DetectedEvent {
        residuals: vec![
            TimingResidual::new(1001u32,  0.3e-9),   // μ⁺, +0.3 ns
            TimingResidual::new(1002u32, -0.4e-9),   // μ⁻, −0.4 ns
        ],
        track_pt: vec![28.0, 22.0],                  // GeV/c
        track_charge_positive: vec![true, false],    // opposite sign
        em_fraction: vec![1.0, 1.0],                 // muons are min-ionising
        energy_ratio: vec![0.5, 0.5],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::DispatchOutcome;

    #[test]
    fn n_from_pt_values() {
        assert_eq!(n_from_pt(1.0), 2);   // sqrt(1) = 1, n = 2
        assert_eq!(n_from_pt(4.0), 3);   // sqrt(4) = 2, n = 3
        assert_eq!(n_from_pt(25.0), 6);  // sqrt(25) = 5, n = 6
    }

    #[test]
    fn planck_depth_lhc_is_60() {
        assert_eq!(planck_depth(LHC_TAU_BX_S, 3), 60);
    }

    #[test]
    fn tau_bx_from_atlas_is_25ns() {
        let tau = tau_bx_from_geometry(ATLAS_HALF_DIAMETER_M);
        let ns = tau * 1e9;
        assert!((ns - 25.02).abs() < 0.1, "expected ~25 ns, got {ns:.3} ns");
    }

    #[test]
    fn dimuon_event_accepted_by_atlas_kernel() {
        let (kernel, tem) = atlas_kernel();
        let program = TempusProgram::dimuon(CellRegistry::new());
        let event = dimuon_test_event();
        let fragment = event.compile(&program);
        let outcome = kernel.dispatch(fragment);
        assert!(outcome.is_ok(), "di-muon event should be accepted");
        assert_eq!(tem.dispatched(), 1);
    }

    #[test]
    fn dimuon_leading_partition_label() {
        let event = dimuon_test_event();
        let label = event.leading_partition_label().unwrap();
        assert_eq!(label.n, 6,       "n should be 6 for p_T=28 GeV");
        assert_eq!(label.l, 0,       "l should be 0 for muon (EM-like)");
        assert_eq!(label.s, Spin::Up,"s should be Up for positive charge");
    }

    #[test]
    fn detector_op_n_produces_label() {
        let mut reg = OperationRegistry::default();
        register_detector_ops(&mut reg);
        let mut args = BTreeMap::new();
        args.insert("pt_gev".to_string(), Value::Num(25.0));
        let result = reg.invoke(&crate::vahera::OperationName::new("op_n"), args).unwrap();
        assert!(result.as_label().is_some());
        assert_eq!(result.as_label().unwrap().n, 6);
    }

    #[test]
    fn detector_op_s_respects_charge() {
        let mut reg = OperationRegistry::default();
        register_detector_ops(&mut reg);
        let mut args = BTreeMap::new();
        args.insert("positive_charge".to_string(), Value::Bool(false));
        args.insert("n".to_string(), Value::Num(6.0));
        args.insert("l".to_string(), Value::Num(0.0));
        args.insert("m".to_string(), Value::Num(0.0));
        let result = reg.invoke(&crate::vahera::OperationName::new("op_s"), args).unwrap();
        let label = result.as_label().unwrap();
        assert_eq!(label.s, Spin::Down, "negative charge → Spin::Down");
    }
}
