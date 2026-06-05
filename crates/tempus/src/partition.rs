//! # Partition Coordinates
//!
//! The four-tuple `(n, ℓ, m, s)` labelling every state of a bounded
//! three-dimensional oscillatory system.  Constraints:
//!
//!   n ≥ 1,   0 ≤ ℓ ≤ n−1,   −ℓ ≤ m ≤ +ℓ,   s ∈ {−½, +½}
//!
//! Shell capacity: `C(n) = 2n²`.
//! Bijection `Φ: ℤ⁺ → 𝒫` (Partition Bijection Theorem, Introduction paper).
//!
//! This module re-exports the validated `PartitionCoord` from `catcount` and
//! defines the lighter `PartitionLabel` used internally by the vaHera type
//! system (avoids a hard dependency on catcount's error type in hot paths).

use serde::{Deserialize, Serialize};

// Re-export catcount's validated type for code that needs full construction.
pub use catcount::partition::PartitionCoord;

// ---------------------------------------------------------------------------
// Spin — typed ±½
// ---------------------------------------------------------------------------

/// Spin projection.  `Up` = +½, `Down` = −½.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Spin {
    Up,
    Down,
}

impl Spin {
    pub fn value(self) -> f64 {
        match self { Spin::Up => 0.5, Spin::Down => -0.5 }
    }

    pub fn sign(self) -> i8 {
        match self { Spin::Up => 1, Spin::Down => -1 }
    }
}

impl TryFrom<i8> for Spin {
    type Error = crate::TempusError;
    fn try_from(v: i8) -> crate::Result<Self> {
        match v {
            1  => Ok(Spin::Up),
            -1 => Ok(Spin::Down),
            _  => Err(crate::TempusError::InvalidPartitionCoord { n: 0, l: 0, m: v as i32 }),
        }
    }
}

// ---------------------------------------------------------------------------
// PartitionLabel — lightweight label for type-checking
// ---------------------------------------------------------------------------

/// A validated partition label `(n, ℓ, m, s)`.
///
/// Construction is only possible via [`PartitionLabel::new`], which enforces
/// the boundary conditions.  Use [`PartitionLabel::new_unchecked`] only when
/// the constraints are known to hold (e.g.\ inside the type-checker itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionLabel {
    /// Principal level n ≥ 1.
    pub n: u32,
    /// Angular sublevel 0 ≤ ℓ ≤ n−1.
    pub l: u32,
    /// Azimuthal projection −ℓ ≤ m ≤ +ℓ.
    pub m: i32,
    /// Chirality / spin.
    pub s: Spin,
}

impl PartitionLabel {
    /// Construct with full validation.
    pub fn new(n: u32, l: u32, m: i32, s: Spin) -> crate::Result<Self> {
        if n == 0 {
            return Err(crate::TempusError::InvalidPartitionCoord { n, l, m });
        }
        if l >= n {
            return Err(crate::TempusError::InvalidPartitionCoord { n, l, m });
        }
        if m.unsigned_abs() > l {
            return Err(crate::TempusError::InvalidPartitionCoord { n, l, m });
        }
        Ok(Self { n, l, m, s })
    }

    /// Construct without validation (caller guarantees constraints).
    #[inline]
    pub const fn new_unchecked(n: u32, l: u32, m: i32, s: Spin) -> Self {
        Self { n, l, m, s }
    }

    /// Ground state (n=1, ℓ=0, m=0, s=Up).
    pub const fn ground() -> Self {
        Self::new_unchecked(1, 0, 0, Spin::Up)
    }

    /// Shell capacity at this principal level: C(n) = 2n².
    pub fn capacity(self) -> u64 { 2 * (self.n as u64).pow(2) }

    /// Transition delta to another label.
    pub fn delta(self, other: Self) -> LabelDelta {
        LabelDelta {
            dn: other.n as i32 - self.n as i32,
            dl: other.l as i32 - self.l as i32,
            dm: other.m - self.m,
            ds: other.s.sign() - self.s.sign(),
        }
    }
}

impl std::fmt::Display for PartitionLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(n={}, ℓ={}, m={}, s={})",
               self.n, self.l, self.m,
               if self.s == Spin::Up { "+½" } else { "−½" })
    }
}

/// Componentwise difference between two partition labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelDelta {
    pub dn: i32,
    pub dl: i32,
    pub dm: i32,
    pub ds: i8,
}

// ---------------------------------------------------------------------------
// Bijection Φ: ℤ⁺ → 𝒫
// ---------------------------------------------------------------------------

/// Enumerate the `index`-th partition label (1-indexed, `index ≥ 1`).
///
/// Implements the bijection `Φ` proved in the Introduction paper.
/// The enumeration order is: level by level (increasing n), then by (ℓ, m, s)
/// in lexicographic order within each level.
pub fn phi(index: u64) -> Option<PartitionLabel> {
    if index == 0 { return None; }
    let mut remaining = index;
    let mut n = 1u32;
    loop {
        let cap = 2 * (n as u64).pow(2); // C(n) = 2n²
        if remaining <= cap { break; }
        remaining -= cap;
        n += 1;
    }
    // remaining ∈ [1, C(n)] — enumerate (ℓ, m, s) lexicographically
    for l in 0..n {
        let deg = 2 * l + 1; // 2ℓ+1 m-values
        for s in [Spin::Up, Spin::Down] {
            for m_offset in 0..=(2 * l as i32) {
                let m = m_offset - l as i32; // m ∈ [−ℓ, +ℓ]
                remaining -= 1;
                if remaining == 0 {
                    return Some(PartitionLabel::new_unchecked(n, l, m, s));
                }
                // suppress unused warning for `deg`
                let _ = deg;
            }
        }
    }
    None // unreachable for valid n
}

/// Inverse bijection: `PartitionLabel` → `ℤ⁺` index.
pub fn phi_inv(label: PartitionLabel) -> u64 {
    // Sum capacities of all complete levels below n.
    let mut idx: u64 = (0..label.n)
        .map(|k| 2 * (k as u64).pow(2))
        .sum();
    // Position within level n.
    for l in 0..label.n {
        for s in [Spin::Up, Spin::Down] {
            for m_offset in 0..=(2 * l as i32) {
                let m = m_offset - l as i32;
                idx += 1;
                if l == label.l && m == label.m && s == label.s {
                    return idx;
                }
            }
        }
    }
    unreachable!("label must be reachable by the enumeration");
}

// ---------------------------------------------------------------------------
// Composition-inflation T(n, d) = d(d+1)^{n-1}
// ---------------------------------------------------------------------------

/// Number of distinct partition states accessible at depth `n` in a
/// `d`-branching hierarchy.  `T(n, d) = d·(d+1)^{n−1}`.
pub fn composition_inflation(n: u32, d: u32) -> u64 {
    (d as u64) * (d as u64 + 1).pow(n.saturating_sub(1))
}

/// Planck depth for a given timing window and dimension.
///
/// `n_P = 1 + ⌈log_{d+1}(τ / (d · t_P))⌉`
///
/// For the LHC: `τ = 25 ns`, `d = 3`, `t_P = 5.391 × 10⁻⁴⁴ s` → `n_P = 60`.
pub fn planck_depth(tau_s: f64, d: u32) -> u32 {
    const PLANCK_TIME_S: f64 = 5.391e-44;
    let base = (d + 1) as f64;
    let arg = tau_s / (d as f64 * PLANCK_TIME_S);
    1 + arg.log(base).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ground_state_is_valid() {
        let g = PartitionLabel::ground();
        assert_eq!(g.n, 1);
        assert_eq!(g.l, 0);
        assert_eq!(g.m, 0);
        assert_eq!(g.s, Spin::Up);
        assert_eq!(g.capacity(), 2);
    }

    #[test]
    fn label_construction_validates() {
        assert!(PartitionLabel::new(0, 0, 0, Spin::Up).is_err()); // n=0
        assert!(PartitionLabel::new(2, 2, 0, Spin::Up).is_err()); // l >= n
        assert!(PartitionLabel::new(2, 1, 2, Spin::Up).is_err()); // |m| > l
        assert!(PartitionLabel::new(2, 1, 1, Spin::Up).is_ok());
    }

    #[test]
    fn phi_round_trips() {
        for idx in 1u64..=50 {
            let label = phi(idx).expect("phi(idx) should be Some for idx >= 1");
            assert_eq!(phi_inv(label), idx, "phi_inv(phi({idx})) != {idx}");
        }
    }

    #[test]
    fn composition_inflation_lhc() {
        // T(60, 3) should be enormous
        let t = composition_inflation(1, 3);
        assert_eq!(t, 3); // T(1,3) = 3
        let t2 = composition_inflation(2, 3);
        assert_eq!(t2, 12); // T(2,3) = 3*4 = 12
    }

    #[test]
    fn planck_depth_lhc() {
        let n_p = planck_depth(25e-9, 3);
        assert_eq!(n_p, 60, "LHC Planck depth should be 60, got {n_p}");
    }

    #[test]
    fn planck_depth_caesium() {
        let tau_cs = 1.0 / 9.192631770e9; // one Cs hyperfine period
        let n_p = planck_depth(tau_cs, 3);
        assert_eq!(n_p, 56, "Caesium Planck depth should be 56, got {n_p}");
    }
}
