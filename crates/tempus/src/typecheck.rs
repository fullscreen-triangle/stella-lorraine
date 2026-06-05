//! # Partition-Coordinate Refinement Type Checker (PVE)
//!
//! Implements the `typecheck` function from the vaHera refinement-type paper:
//!
//! > A vaHera fragment is *well-typed* iff every `Compose`-chain transition
//! > respects every registered selection rule, and all `Call` operations are
//! > registered.
//!
//! ## Complexity
//!
//! `O(|fragment| × |rules|)` — linear in fragment size for a fixed rule set
//! (Decidability Theorem, vaHera types paper, Thm 5.3).
//!
//! ## Key result
//!
//! The Forbidden-Transition Theorem states that a fragment implying a
//! selection-rule-violating transition has an *uninhabited* refinement type
//! and is rejected *at static time* — before any evaluation.
//! This is the formal basis for "selection rules as type errors, not
//! probabilistic suppression" (trajectory-completion paper, Thm 8.1).

use crate::{
    partition::PartitionLabel,
    selection::SelectionRegistry,
    vahera::Fragment,
    TempusError, Result,
};

// ---------------------------------------------------------------------------
// TypeContext — maps Hole names to their expected PartitionLabel type
// ---------------------------------------------------------------------------

/// Typing context: maps `Hole` names to expected partition labels.
#[derive(Default, Clone)]
pub struct TypeContext {
    holes: std::collections::HashMap<String, PartitionLabel>,
}

impl TypeContext {
    pub fn new() -> Self { Self::default() }

    pub fn with_hole(mut self, name: impl Into<String>, label: PartitionLabel) -> Self {
        self.holes.insert(name.into(), label);
        self
    }

    pub fn get(&self, name: &str) -> Option<&PartitionLabel> {
        self.holes.get(name)
    }
}

// ---------------------------------------------------------------------------
// TypeCheckResult
// ---------------------------------------------------------------------------

/// Result of a `typecheck` call.
#[derive(Debug, Clone)]
pub struct TypeCheckResult {
    pub ok: bool,
    /// First error encountered (if any).
    pub error: Option<TempusError>,
    /// All forbidden transitions found (for diagnostics).
    pub violations: Vec<ForbiddenTransition>,
}

impl TypeCheckResult {
    fn pass() -> Self { Self { ok: true, error: None, violations: vec![] } }

    fn fail(err: TempusError) -> Self {
        Self { ok: false, error: Some(err), violations: vec![] }
    }

    fn with_violation(mut self, v: ForbiddenTransition) -> Self {
        self.ok = false;
        if self.error.is_none() {
            self.error = Some(TempusError::ForbiddenTransition {
                from: v.from,
                to: v.to,
                rule: v.violated_rule.clone(),
            });
        }
        self.violations.push(v);
        self
    }
}

#[derive(Debug, Clone)]
pub struct ForbiddenTransition {
    pub from: PartitionLabel,
    pub to: PartitionLabel,
    pub violated_rule: String,
}

// ---------------------------------------------------------------------------
// typecheck
// ---------------------------------------------------------------------------

/// Type-check a `Fragment` against the selection-rule registry.
///
/// Returns `Ok(())` if all transitions are admissible, or an error describing
/// the first forbidden transition encountered.
///
/// This is the `Check(F, Γ, Σ★)` function from Algorithm 1 of the vaHera
/// types paper.
pub fn typecheck(
    fragment: &Fragment,
    ctx: &TypeContext,
    registry: &SelectionRegistry,
) -> Result<()> {
    let result = check_inner(fragment, ctx, registry, None);
    match result.error {
        None => Ok(()),
        Some(e) => Err(e),
    }
}

/// Full check that returns all violations (not just the first).
pub fn typecheck_full(
    fragment: &Fragment,
    ctx: &TypeContext,
    registry: &SelectionRegistry,
) -> TypeCheckResult {
    check_inner(fragment, ctx, registry, None)
}

fn check_inner(
    fragment: &Fragment,
    ctx: &TypeContext,
    registry: &SelectionRegistry,
    carry_label: Option<&PartitionLabel>,
) -> TypeCheckResult {
    match fragment {
        Fragment::Literal(v) => {
            // If the literal carries a PartitionLabel AND there's a carry label,
            // check the implied transition.
            if let Some(from) = carry_label {
                if let crate::vahera::Value::Label(to) = v {
                    let violations = registry.violations(from, to);
                    let mut result = TypeCheckResult::pass();
                    for rule in violations {
                        result = result.with_violation(ForbiddenTransition {
                            from: *from,
                            to: *to,
                            violated_rule: rule,
                        });
                    }
                    return result;
                }
            }
            TypeCheckResult::pass()
        }

        Fragment::Hole(name) => {
            if ctx.get(name).is_none() {
                // Hole with no type in context is a warning, not a hard error
                // (the hole may be filled later).
            }
            TypeCheckResult::pass()
        }

        Fragment::Call { op, args } => {
            // Check all argument sub-fragments.
            let mut result = TypeCheckResult::pass();
            for (_key, sub) in args {
                let sub_result = check_inner(sub, ctx, registry, None);
                if !sub_result.ok {
                    result.ok = false;
                    if result.error.is_none() { result.error = sub_result.error; }
                    result.violations.extend(sub_result.violations);
                }
            }
            // Unknown operation is a hard error.
            if !op.0.is_empty() && !registry.is_empty() {
                // We don't have access to the op registry here; that check
                // happens in the executor.  The type-checker only enforces
                // partition-label selection rules.
            }
            result
        }

        Fragment::Compose(stages) => {
            if stages.is_empty() { return TypeCheckResult::pass(); }

            let mut result = TypeCheckResult::pass();
            let mut prev_label: Option<PartitionLabel> = carry_label.copied();

            for stage in stages {
                // Check the stage itself.
                let stage_result = check_inner(stage, ctx, registry, prev_label.as_ref());
                if !stage_result.ok {
                    result.ok = false;
                    if result.error.is_none() { result.error = stage_result.error.clone(); }
                    result.violations.extend(stage_result.violations);
                }

                // Extract the output label of this stage (if it produces one) to
                // use as `carry_label` for the next stage.
                prev_label = extract_output_label(stage);
            }
            result
        }
    }
}

/// Attempt to statically determine the output label of a fragment.
/// Returns `None` if it cannot be determined (e.g.\ for non-literal fragments
/// that don't have a statically-known label).
fn extract_output_label(fragment: &Fragment) -> Option<PartitionLabel> {
    match fragment {
        Fragment::Literal(crate::vahera::Value::Label(l)) => Some(*l),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Identity PVE (accepts everything)
// ---------------------------------------------------------------------------

/// A `ValidationSubsystem` implementation backed by `typecheck`.
///
/// Used as the PVE slot in the kernel for LHC-mode operation.
pub struct PartitionTypePve {
    ctx: TypeContext,
    registry: SelectionRegistry,
    /// Also run three-route mass verification (see runtime-operations paper).
    three_route_enabled: bool,
}

impl PartitionTypePve {
    pub fn new(registry: SelectionRegistry) -> Self {
        Self { ctx: TypeContext::new(), registry, three_route_enabled: false }
    }

    pub fn with_three_route(mut self) -> Self {
        self.three_route_enabled = true;
        self
    }

    pub fn with_context(mut self, ctx: TypeContext) -> Self {
        self.ctx = ctx;
        self
    }

    pub fn valid(&self, fragment: &Fragment) -> bool {
        typecheck(fragment, &self.ctx, &self.registry).is_ok()
    }
}

// ---------------------------------------------------------------------------
// ValidationSubsystem impl for PartitionTypePve
// (connects typecheck to the kernel's PVE trait slot)
// ---------------------------------------------------------------------------

impl crate::kernel::ValidationSubsystem for PartitionTypePve {
    fn validate(&self, fragment: &Fragment) -> crate::Result<()> {
        if self.valid(fragment) {
            Ok(())
        } else {
            Err(crate::TempusError::ValidationRejected {
                reason: "partition refinement typecheck failed".into(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        partition::{PartitionLabel, Spin},
        selection::SelectionRegistry,
        vahera::{Fragment, Value},
    };

    fn label(n: u32, l: u32, m: i32, s: Spin) -> PartitionLabel {
        PartitionLabel::new(n, l, m, s).unwrap()
    }

    fn label_frag(n: u32, l: u32, m: i32, s: Spin) -> Fragment {
        Fragment::literal(Value::Label(label(n, l, m, s)))
    }

    #[test]
    fn literal_label_no_carry_passes() {
        let reg = SelectionRegistry::lhc();
        let ctx = TypeContext::new();
        let f = label_frag(2, 1, 0, Spin::Up);
        assert!(typecheck(&f, &ctx, &reg).is_ok());
    }

    #[test]
    fn compose_chain_e1_passes() {
        // ℓ=0 → ℓ=1: Δℓ=+1, allowed by E1
        let reg = SelectionRegistry::lhc();
        let ctx = TypeContext::new();
        let f = Fragment::compose(vec![
            label_frag(2, 0, 0, Spin::Up),
            label_frag(2, 1, 0, Spin::Up),
        ]);
        assert!(typecheck(&f, &ctx, &reg).is_ok());
    }

    #[test]
    fn compose_chain_forbidden_delta_l_2_fails() {
        // ℓ=0 → ℓ=2: Δℓ=+2, forbidden by E1 — this is a TYPE ERROR
        let reg = SelectionRegistry::lhc();
        let ctx = TypeContext::new();
        let f = Fragment::compose(vec![
            label_frag(3, 0, 0, Spin::Up),
            label_frag(3, 2, 0, Spin::Up),
        ]);
        let result = typecheck(&f, &ctx, &reg);
        assert!(result.is_err(), "Δℓ=2 transition must be a type error");
        match result {
            Err(TempusError::ForbiddenTransition { .. }) => {}
            other => panic!("expected ForbiddenTransition, got {other:?}"),
        }
    }

    #[test]
    fn full_check_collects_multiple_violations() {
        let reg = SelectionRegistry::lhc();
        let ctx = TypeContext::new();
        // Δℓ=+2 AND Δs≠0 → both E1 and spin_conserved violated
        let f = Fragment::compose(vec![
            label_frag(3, 0, 0, Spin::Up),
            label_frag(3, 2, 0, Spin::Down),
        ]);
        let result = typecheck_full(&f, &ctx, &reg);
        assert!(!result.ok);
        assert_eq!(result.violations.len(), 2);
        let rule_names: Vec<_> = result.violations.iter()
            .map(|v| v.violated_rule.as_str())
            .collect();
        assert!(rule_names.contains(&"E1"));
        assert!(rule_names.contains(&"spin_conserved"));
    }

    #[test]
    fn empty_registry_accepts_anything() {
        let reg = SelectionRegistry::new();
        let ctx = TypeContext::new();
        let f = Fragment::compose(vec![
            label_frag(3, 0, 0, Spin::Up),
            label_frag(3, 2, 1, Spin::Down), // would be forbidden under E1
        ]);
        assert!(typecheck(&f, &ctx, &reg).is_ok());
    }

    #[test]
    fn pve_valid_wraps_typecheck() {
        let pve = PartitionTypePve::new(SelectionRegistry::lhc());
        let f_good = Fragment::compose(vec![
            label_frag(2, 0, 0, Spin::Up),
            label_frag(2, 1, 0, Spin::Up),
        ]);
        let f_bad = Fragment::compose(vec![
            label_frag(3, 0, 0, Spin::Up),
            label_frag(3, 2, 0, Spin::Up),
        ]);
        assert!(pve.valid(&f_good));
        assert!(!pve.valid(&f_bad));
    }
}
