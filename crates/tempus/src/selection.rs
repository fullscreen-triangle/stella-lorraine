//! # Selection Rules as Refinement Predicates
//!
//! Every registered selection rule is a binary predicate on partition labels.
//! The `SelectionRegistry` holds a finite set of rules.  A transition
//! `(from, to)` is *admissible* iff every registered rule permits it.
//!
//! ## Physical rules
//!
//! | Name              | Condition                         |
//! |-------------------|-----------------------------------|
//! | `E1`              | Δℓ ∈ {−1, +1}, Δm ∈ {−1, 0, +1} |
//! | `M1`              | Δℓ = 0,        Δm ∈ {−1, 0, +1} |
//! | `spin_conserved`  | Δs = 0                            |
//! | `n_monotone`      | n₂ ≥ n₁  (principal non-decreasing)|
//!
//! All rules are computationally independent (O(1) per evaluation) as
//! required by the Decidability Theorem (vaHera types paper, Thm 5.3).

use crate::partition::{LabelDelta, PartitionLabel};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A computable binary predicate on pairs of partition labels.
///
/// The trait is object-safe so rules can be stored as `Arc<dyn SelectionRule>`.
pub trait SelectionRule: Send + Sync {
    /// Returns `true` iff the transition `from → to` is admissible.
    fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool;

    /// Human-readable rule name (used in error messages and diagnostics).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Built-in physical rules
// ---------------------------------------------------------------------------

/// Electric-dipole (E1) selection rule: Δℓ = ±1, Δm ∈ {−1, 0, +1}.
pub struct E1Rule;
impl SelectionRule for E1Rule {
    fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool {
        let d = from.delta(*to);
        (d.dl == 1 || d.dl == -1) && (-1..=1).contains(&d.dm)
    }
    fn name(&self) -> &str { "E1" }
}

/// Magnetic-dipole (M1) selection rule: Δℓ = 0, Δm ∈ {−1, 0, +1}.
pub struct M1Rule;
impl SelectionRule for M1Rule {
    fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool {
        let d = from.delta(*to);
        d.dl == 0 && (-1..=1).contains(&d.dm)
    }
    fn name(&self) -> &str { "M1" }
}

/// Spin-conservation rule: Δs = 0.
pub struct SpinConservedRule;
impl SelectionRule for SpinConservedRule {
    fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool {
        from.delta(*to).ds == 0
    }
    fn name(&self) -> &str { "spin_conserved" }
}

/// Principal-level monotonicity: n₂ ≥ n₁.
pub struct PrincipalMonotoneRule;
impl SelectionRule for PrincipalMonotoneRule {
    fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool {
        from.delta(*to).dn >= 0
    }
    fn name(&self) -> &str { "n_monotone" }
}

/// Closure rule: Δℓ ∈ {−1, 0, +1}, Δm ∈ {−1, 0, +1}  (combines E1+M1).
pub struct DipoleRule;
impl SelectionRule for DipoleRule {
    fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool {
        let d = from.delta(*to);
        (-1..=1).contains(&d.dl) && (-1..=1).contains(&d.dm)
    }
    fn name(&self) -> &str { "dipole" }
}

/// Custom rule backed by a closure.
pub struct ClosureRule {
    name: String,
    f: Arc<dyn Fn(LabelDelta) -> bool + Send + Sync>,
}

impl ClosureRule {
    pub fn new(
        name: impl Into<String>,
        f: impl Fn(LabelDelta) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self { name: name.into(), f: Arc::new(f) }
    }
}

impl SelectionRule for ClosureRule {
    fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool {
        (self.f)(from.delta(*to))
    }
    fn name(&self) -> &str { &self.name }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// A finite set of selection rules.
///
/// A transition `(from, to)` is *admissible under the registry* iff every
/// rule in the registry permits it.  The empty registry admits all transitions.
#[derive(Default)]
pub struct SelectionRegistry {
    rules: Vec<Arc<dyn SelectionRule>>,
}

impl SelectionRegistry {
    pub fn new() -> Self { Self::default() }

    /// Standard LHC registry: E1 + spin-conserved.
    pub fn lhc() -> Self {
        let mut r = Self::new();
        r.add(Arc::new(E1Rule));
        r.add(Arc::new(SpinConservedRule));
        r
    }

    /// Full dipole registry: dipole (E1+M1) + spin-conserved.
    pub fn dipole() -> Self {
        let mut r = Self::new();
        r.add(Arc::new(DipoleRule));
        r.add(Arc::new(SpinConservedRule));
        r
    }

    /// Add a rule to the registry.
    pub fn add(&mut self, rule: Arc<dyn SelectionRule>) {
        self.rules.push(rule);
    }

    /// Number of registered rules.
    pub fn len(&self) -> usize { self.rules.len() }
    pub fn is_empty(&self) -> bool { self.rules.is_empty() }

    /// Check if `from → to` is admissible.
    ///
    /// Returns `Ok(())` if all rules pass, or `Err(name)` with the name of
    /// the first violated rule.
    pub fn check(
        &self,
        from: &PartitionLabel,
        to: &PartitionLabel,
    ) -> Result<(), String> {
        for rule in &self.rules {
            if !rule.allows(from, to) {
                return Err(rule.name().to_string());
            }
        }
        Ok(())
    }

    /// Returns `true` iff the transition is admissible under every rule.
    pub fn allows(&self, from: &PartitionLabel, to: &PartitionLabel) -> bool {
        self.check(from, to).is_ok()
    }

    /// Collect all violated rule names for a given transition.
    pub fn violations(&self, from: &PartitionLabel, to: &PartitionLabel) -> Vec<String> {
        self.rules.iter()
            .filter(|r| !r.allows(from, to))
            .map(|r| r.name().to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::{PartitionLabel, Spin};

    fn label(n: u32, l: u32, m: i32, s: Spin) -> PartitionLabel {
        PartitionLabel::new(n, l, m, s).unwrap()
    }

    #[test]
    fn e1_rule_allows_delta_l_1() {
        let from = label(2, 0, 0, Spin::Up);
        let to   = label(2, 1, 0, Spin::Up);
        assert!(E1Rule.allows(&from, &to));   // Δℓ = +1 ✓
    }

    #[test]
    fn e1_rule_forbids_delta_l_2() {
        let from = label(3, 0, 0, Spin::Up);
        let to   = label(3, 2, 0, Spin::Up);
        assert!(!E1Rule.allows(&from, &to));  // Δℓ = +2 ✗
    }

    #[test]
    fn spin_conserved_forbids_flip() {
        let from = label(2, 1, 0, Spin::Up);
        let to   = label(2, 0, 0, Spin::Down);
        assert!(!SpinConservedRule.allows(&from, &to)); // Δs ≠ 0 ✗
    }

    #[test]
    fn registry_check_collects_violations() {
        let reg = SelectionRegistry::lhc();
        let from = label(3, 0, 0, Spin::Up);
        let to   = label(3, 2, 0, Spin::Down); // Δℓ=2 (E1 fails), Δs≠0 (spin fails)
        let violations = reg.violations(&from, &to);
        assert!(violations.contains(&"E1".to_string()));
        assert!(violations.contains(&"spin_conserved".to_string()));
    }

    #[test]
    fn empty_registry_admits_everything() {
        let reg = SelectionRegistry::new();
        let from = label(3, 0, 0, Spin::Up);
        let to   = label(3, 2, 1, Spin::Down);
        assert!(reg.allows(&from, &to));
    }
}
