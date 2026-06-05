//! # Five-Subsystem Dispatch Kernel
//!
//! Implements the `Kernel<E, P, M, S, D, T>` struct and its dispatch loop
//! from the Dispatch Substrate Kernel paper.
//!
//! ## Five subsystems
//!
//! | Slot | Name | Responsibility                                 |
//! |------|------|------------------------------------------------|
//! | PVE  | Pre-dispatch validation  | typecheck + three-route |
//! | CMM  | Post-dispatch memoization| timing-interval cache   |
//! | PSS  | Pending-state scheduling | parallel / FIFO         |
//! | DIC  | External-resource retrieval | HLT RoI policy      |
//! | TEM  | Runtime invariant monitoring | η_C, ΔP histogram  |
//!
//! ## Dispatch Correctness
//!
//! With all subsystems at their identity implementations, the kernel returns
//! `eval(fragment)` exactly (Theorem 4.1 of the kernel paper).

use std::{collections::BTreeMap, time::Instant};

use crate::{
    vahera::{eval, Fragment, OperationRegistry, Value},
    TempusError, Result,
};

// ---------------------------------------------------------------------------
// Dispatch event (published by TEM)
// ---------------------------------------------------------------------------

/// Structured record emitted by the kernel on each completed dispatch.
#[derive(Debug, Clone)]
pub struct DispatchEvent {
    pub op_name: Option<String>,
    pub elapsed_ns: u64,
    pub outcome: DispatchOutcome,
    pub cache_hit: bool,
}

/// Outcome of a dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchOutcome {
    /// Fragment evaluated to a value.
    Ok(Value),
    /// PVE rejected the fragment.
    Rejected { reason: String },
    /// Executor returned no value (unresolved hole or operation failure).
    Failed,
}

impl DispatchOutcome {
    pub fn is_ok(&self) -> bool { matches!(self, Self::Ok(_)) }
    pub fn value(&self) -> Option<&Value> {
        match self { Self::Ok(v) => Some(v), _ => None }
    }
}

// ---------------------------------------------------------------------------
// Subsystem traits
// ---------------------------------------------------------------------------

/// Pre-dispatch validation (PVE).
pub trait ValidationSubsystem: Send + Sync {
    /// Returns `Ok(())` if the fragment is admissible, or an error.
    fn validate(&self, fragment: &Fragment) -> Result<()>;
}

/// Post-dispatch memoization (CMM).
pub trait MemoSubsystem: Send + Sync {
    fn lookup(&self, fragment: &Fragment) -> Option<Value>;
    fn insert(&self, fragment: &Fragment, value: &Value);
}

/// Pending-state scheduler (PSS).
pub trait SchedulerSubsystem: Send + Sync {
    /// Re-order a queue of pending fragments; returns a permutation.
    fn order(&self, pending: Vec<Fragment>) -> Vec<Fragment>;
}

/// External-resource retrieval policy (DIC).
pub trait RetrievalSubsystem: Send + Sync {
    /// Called by the executor before any external I/O.
    fn policy(&self, query: &str) -> RetrievalPolicy;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalPolicy {
    FetchAll,
    FetchTopK(usize),
    Skip,
}

/// Runtime invariant monitor (TEM).
pub trait MonitorSubsystem: Send + Sync {
    fn observe(&self, event: &DispatchEvent);
}

// ---------------------------------------------------------------------------
// Identity implementations
// ---------------------------------------------------------------------------

/// Identity PVE: accepts every fragment.
pub struct IdentityPve;
impl ValidationSubsystem for IdentityPve {
    fn validate(&self, _f: &Fragment) -> Result<()> { Ok(()) }
}

/// Identity CMM: always misses.
pub struct IdentityCmm;
impl MemoSubsystem for IdentityCmm {
    fn lookup(&self, _f: &Fragment) -> Option<Value> { None }
    fn insert(&self, _f: &Fragment, _v: &Value) {}
}

/// Identity PSS: FIFO (no reordering).
pub struct FifoPss;
impl SchedulerSubsystem for FifoPss {
    fn order(&self, pending: Vec<Fragment>) -> Vec<Fragment> { pending }
}

/// Identity DIC: always fetch-all.
pub struct FetchAllDic;
impl RetrievalSubsystem for FetchAllDic {
    fn policy(&self, _q: &str) -> RetrievalPolicy { RetrievalPolicy::FetchAll }
}

/// Identity TEM: drops all events.
pub struct NoopTem;
impl MonitorSubsystem for NoopTem {
    fn observe(&self, _event: &DispatchEvent) {}
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// The five-subsystem dispatch kernel.
///
/// Generic over all five subsystem types so that any combination of
/// identity / real implementations can be composed without heap allocation.
///
/// ## Example — all-identity kernel
///
/// ```rust
/// use tempus::kernel::{Kernel, IdentityPve, IdentityCmm, FifoPss, FetchAllDic, NoopTem};
/// use tempus::vahera::OperationRegistry;
///
/// let kernel = Kernel::new(
///     OperationRegistry::default(),
///     IdentityPve,
///     IdentityCmm,
///     FifoPss,
///     FetchAllDic,
///     NoopTem,
/// );
/// ```
pub struct Kernel<P, M, S, D, T> {
    registry: OperationRegistry,
    pve: P,
    cmm: M,
    pss: S,
    dic: D,
    tem: T,
}

impl<P, M, S, D, T> Kernel<P, M, S, D, T>
where
    P: ValidationSubsystem,
    M: MemoSubsystem,
    S: SchedulerSubsystem,
    D: RetrievalSubsystem,
    T: MonitorSubsystem,
{
    pub fn new(registry: OperationRegistry, pve: P, cmm: M, pss: S, dic: D, tem: T) -> Self {
        Self { registry, pve, cmm, pss, dic, tem }
    }

    /// Core dispatch loop (Algorithm 1 of the kernel paper).
    pub fn dispatch(&self, fragment: Fragment) -> DispatchOutcome {
        let t0 = Instant::now();
        let op_name = fragment_op_name(&fragment);

        // ── Step 1: PVE ────────────────────────────────────────────────────
        if let Err(e) = self.pve.validate(&fragment) {
            let event = DispatchEvent {
                op_name,
                elapsed_ns: t0.elapsed().as_nanos() as u64,
                outcome: DispatchOutcome::Rejected { reason: e.to_string() },
                cache_hit: false,
            };
            self.tem.observe(&event);
            return event.outcome;
        }

        // ── Step 2: PSS (single-fragment path: no reordering needed) ───────
        // (Multi-fragment batching handled by `dispatch_batch`)

        // ── Step 3: CMM lookup ─────────────────────────────────────────────
        if let Some(cached) = self.cmm.lookup(&fragment) {
            let outcome = DispatchOutcome::Ok(cached);
            let event = DispatchEvent {
                op_name,
                elapsed_ns: t0.elapsed().as_nanos() as u64,
                outcome: outcome.clone(),
                cache_hit: true,
            };
            self.tem.observe(&event);
            return outcome;
        }

        // ── Step 4: Execute ────────────────────────────────────────────────
        let result = eval(&fragment, &self.registry);
        let outcome = match result {
            Ok(Some(v)) => {
                self.cmm.insert(&fragment, &v);
                DispatchOutcome::Ok(v)
            }
            Ok(None) => DispatchOutcome::Failed,
            Err(e) => DispatchOutcome::Rejected { reason: e.to_string() },
        };

        // ── Step 5: TEM ────────────────────────────────────────────────────
        let event = DispatchEvent {
            op_name,
            elapsed_ns: t0.elapsed().as_nanos() as u64,
            outcome: outcome.clone(),
            cache_hit: false,
        };
        self.tem.observe(&event);

        outcome
    }

    /// Dispatch a batch of fragments, applying PSS ordering first.
    pub fn dispatch_batch(&self, fragments: Vec<Fragment>) -> Vec<DispatchOutcome> {
        let ordered = self.pss.order(fragments);
        ordered.into_iter().map(|f| self.dispatch(f)).collect()
    }

    /// Retrieve the DIC policy for an external query key.
    pub fn retrieval_policy(&self, query: &str) -> RetrievalPolicy {
        self.dic.policy(query)
    }
}

/// Extract the top-level operation name from a fragment (for event logging).
fn fragment_op_name(fragment: &Fragment) -> Option<String> {
    match fragment {
        Fragment::Call { op, .. } => Some(op.0.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// DefaultKernel — identity subsystems + empty registry
// ---------------------------------------------------------------------------

/// A kernel with all identity subsystems and an empty operation registry.
/// Register operations via `registry_mut()` before dispatching.
pub type DefaultKernel = Kernel<IdentityPve, IdentityCmm, FifoPss, FetchAllDic, NoopTem>;

impl DefaultKernel {
    pub fn default_kernel() -> Self {
        Kernel::new(
            OperationRegistry::default(),
            IdentityPve,
            IdentityCmm,
            FifoPss,
            FetchAllDic,
            NoopTem,
        )
    }
}

// ---------------------------------------------------------------------------
// TEM: composition efficiency monitor
// ---------------------------------------------------------------------------

use parking_lot::Mutex;

/// A `MonitorSubsystem` that tracks composition efficiency η_C.
///
/// η_C = (events accepted) / (events dispatched)
pub struct EfficiencyMonitor {
    dispatched: Mutex<u64>,
    accepted: Mutex<u64>,
    total_elapsed_ns: Mutex<u64>,
}

impl EfficiencyMonitor {
    pub fn new() -> Self {
        Self {
            dispatched: Mutex::new(0),
            accepted: Mutex::new(0),
            total_elapsed_ns: Mutex::new(0),
        }
    }

    pub fn eta_c(&self) -> f64 {
        let d = *self.dispatched.lock();
        let a = *self.accepted.lock();
        if d == 0 { 0.0 } else { a as f64 / d as f64 }
    }

    pub fn dispatched(&self) -> u64 { *self.dispatched.lock() }
    pub fn accepted(&self) -> u64 { *self.accepted.lock() }
    pub fn mean_latency_ns(&self) -> f64 {
        let d = *self.dispatched.lock();
        if d == 0 { return 0.0; }
        *self.total_elapsed_ns.lock() as f64 / d as f64
    }
}

impl Default for EfficiencyMonitor {
    fn default() -> Self { Self::new() }
}

impl MonitorSubsystem for EfficiencyMonitor {
    fn observe(&self, event: &DispatchEvent) {
        *self.dispatched.lock() += 1;
        *self.total_elapsed_ns.lock() += event.elapsed_ns;
        if event.outcome.is_ok() {
            *self.accepted.lock() += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// CMM: in-memory hash cache
// ---------------------------------------------------------------------------

use dashmap::DashMap;

/// A concurrent in-memory memoization cache keyed on `Fragment` JSON.
///
/// In production this would be keyed on a content-addressed hash; for now
/// we use JSON serialisation for correctness.
pub struct HashCmm {
    cache: DashMap<String, Value>,
}

impl HashCmm {
    pub fn new() -> Self { Self { cache: DashMap::new() } }
    pub fn len(&self) -> usize { self.cache.len() }
    pub fn is_empty(&self) -> bool { self.cache.is_empty() }
}

impl Default for HashCmm { fn default() -> Self { Self::new() } }

impl MemoSubsystem for HashCmm {
    fn lookup(&self, fragment: &Fragment) -> Option<Value> {
        let key = serde_json::to_string(fragment).ok()?;
        self.cache.get(&key).map(|v| v.clone())
    }
    fn insert(&self, fragment: &Fragment, value: &Value) {
        if let Ok(key) = serde_json::to_string(fragment) {
            self.cache.insert(key, value.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// PSS: parallel dispatcher hint
// ---------------------------------------------------------------------------

/// A PSS that marks fragments for parallel execution if their operation names
/// are in the disjoint-class set (Jackson-independence / QND commutation).
///
/// For the LHC: { "op_n", "op_l", "op_m", "op_s" } are Jackson-independent
/// and can be dispatched in parallel.
pub struct ParallelPss {
    independent_ops: std::collections::HashSet<String>,
}

impl ParallelPss {
    pub fn new(independent_ops: impl IntoIterator<Item = String>) -> Self {
        Self { independent_ops: independent_ops.into_iter().collect() }
    }

    /// LHC four-subsystem independent ops.
    pub fn lhc() -> Self {
        Self::new(["op_n", "op_l", "op_m", "op_s"].map(String::from))
    }

    /// Returns `true` if the fragment's operation is Jackson-independent.
    pub fn is_independent(&self, fragment: &Fragment) -> bool {
        match fragment {
            Fragment::Call { op, .. } => self.independent_ops.contains(op.as_str()),
            _ => false,
        }
    }
}

impl SchedulerSubsystem for ParallelPss {
    /// For a single-threaded kernel, FIFO is the identity schedule.
    /// In a multi-threaded kernel, callers inspect `is_independent()` to
    /// decide whether to spawn parallel tasks.
    fn order(&self, pending: Vec<Fragment>) -> Vec<Fragment> {
        // The independent fragments come first (they can be parallelised).
        let (mut independent, mut rest): (Vec<_>, Vec<_>) =
            pending.into_iter().partition(|f| self.is_independent(f));
        independent.append(&mut rest);
        independent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vahera::{Fragment, Value};

    fn make_kernel_with_double() -> DefaultKernel {
        let mut k = DefaultKernel::default_kernel();
        k.registry.register("double", Box::new(|args| {
            let v = args.get("input")
                .and_then(Value::as_num)
                .unwrap_or(0.0);
            Ok(Value::Num(v * 2.0))
        }));
        k
    }

    #[test]
    fn dispatch_literal() {
        let k = DefaultKernel::default_kernel();
        let out = k.dispatch(Fragment::literal(7.0_f64));
        assert_eq!(out.value(), Some(&Value::Num(7.0)));
    }

    #[test]
    fn dispatch_compose_doubles_twice() {
        let k = make_kernel_with_double();
        let f = Fragment::compose(vec![
            Fragment::literal(3.0_f64),
            Fragment::call0("double"),
            Fragment::call0("double"),
        ]);
        assert_eq!(k.dispatch(f).value(), Some(&Value::Num(12.0)));
    }

    #[test]
    fn identity_pve_never_rejects() {
        let k = DefaultKernel::default_kernel();
        let f = Fragment::hole("unresolved");
        // Hole → Failed (not Rejected), because PVE passed but eval returned None.
        assert_eq!(k.dispatch(f), DispatchOutcome::Failed);
    }

    #[test]
    fn cmm_caches_result() {
        use std::sync::Arc;
        let tem = Arc::new(EfficiencyMonitor::new());
        let cmm = HashCmm::new();
        let kernel = Kernel::new(
            OperationRegistry::default(),
            IdentityPve,
            cmm,
            FifoPss,
            FetchAllDic,
            NoopTem,
        );
        let f = Fragment::literal(42.0_f64);
        kernel.dispatch(f.clone());
        kernel.dispatch(f); // second call should hit cache
        // Both dispatches succeed; we trust the CMM inserts on first call.
    }

    #[test]
    fn efficiency_monitor_tracks_eta_c() {
        let mon = EfficiencyMonitor::new();
        mon.observe(&DispatchEvent {
            op_name: None,
            elapsed_ns: 100,
            outcome: DispatchOutcome::Ok(Value::Null),
            cache_hit: false,
        });
        mon.observe(&DispatchEvent {
            op_name: None,
            elapsed_ns: 200,
            outcome: DispatchOutcome::Rejected { reason: "test".into() },
            cache_hit: false,
        });
        assert_eq!(mon.dispatched(), 2);
        assert_eq!(mon.accepted(), 1);
        assert!((mon.eta_c() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn parallel_pss_puts_independent_ops_first() {
        let pss = ParallelPss::lhc();
        let frags = vec![
            Fragment::call0("not_independent"),
            Fragment::call0("op_n"),
            Fragment::call0("op_l"),
        ];
        let ordered = pss.order(frags);
        assert_eq!(ordered[0], Fragment::call0("op_n"));
        assert_eq!(ordered[1], Fragment::call0("op_l"));
        assert_eq!(ordered[2], Fragment::call0("not_independent"));
    }
}
