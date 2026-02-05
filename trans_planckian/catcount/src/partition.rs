//! Partition Coordinate Algebra
//!
//! The partition coordinate system (n, l, m, s) provides the discrete
//! algebraic structure for categorical state space, analogous to
//! atomic quantum numbers.

use crate::error::{CatCountError, Result};
use serde::{Deserialize, Serialize};

/// Partition coordinates (n, l, m, s)
///
/// These coordinates are analogous to hydrogen atom quantum numbers:
/// - n: Principal partition number (n ≥ 1)
/// - l: Angular partition number (0 ≤ l ≤ n-1)
/// - m: Magnetic partition number (-l ≤ m ≤ l)
/// - s: Spin partition number (s = ±1/2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionCoord {
    /// Principal partition number (n ≥ 1)
    pub n: u32,
    /// Angular partition number (0 ≤ l ≤ n-1)
    pub l: u32,
    /// Magnetic partition number (-l ≤ m ≤ l)
    pub m: i32,
    /// Spin partition number (+1 for up, -1 for down)
    pub s: i8,
}

impl PartitionCoord {
    /// Create new partition coordinates with validation
    pub fn new(n: u32, l: u32, m: i32, s: i8) -> Result<Self> {
        // Validate n
        if n == 0 {
            return Err(CatCountError::invalid_partition("n", n as i64, "n ≥ 1"));
        }

        // Validate l
        if l >= n {
            return Err(CatCountError::invalid_partition(
                "l", l as i64, format!("0 ≤ l ≤ n-1 = {}", n - 1)
            ));
        }

        // Validate m
        if m.abs() > l as i32 {
            return Err(CatCountError::invalid_partition(
                "m", m as i64, format!("-l ≤ m ≤ l, l = {}", l)
            ));
        }

        // Validate s (spin)
        if s != 1 && s != -1 {
            return Err(CatCountError::invalid_partition(
                "s", s as i64, "s = +1 (up) or s = -1 (down)"
            ));
        }

        Ok(Self { n, l, m, s })
    }

    /// Create partition coordinates without validation (use with caution)
    #[inline]
    pub fn new_unchecked(n: u32, l: u32, m: i32, s: i8) -> Self {
        Self { n, l, m, s }
    }

    /// Ground state partition (n=1, l=0, m=0, s=+1)
    pub fn ground_state() -> Self {
        Self { n: 1, l: 0, m: 0, s: 1 }
    }

    /// Check if this is the ground state
    pub fn is_ground_state(&self) -> bool {
        self.n == 1 && self.l == 0 && self.m == 0
    }

    /// Get the spin as a fraction (±1/2)
    pub fn spin_fraction(&self) -> f64 {
        self.s as f64 / 2.0
    }

    /// Calculate degeneracy g_n = 2n² for this shell
    pub fn shell_degeneracy(&self) -> u64 {
        2 * (self.n as u64).pow(2)
    }

    /// Calculate cumulative state count up to and including this shell
    /// G_N = Σ_{n=1}^{N} 2n² = N(N+1)(2N+1)/3
    pub fn cumulative_state_count(&self) -> u64 {
        let n = self.n as u64;
        n * (n + 1) * (2 * n + 1) / 3
    }

    /// Convert to a unique state index
    ///
    /// The index is computed by summing:
    /// 1. All states in shells 1 to n-1
    /// 2. All states in shell n with l' < l
    /// 3. All states with m' < m in current subshell
    /// 4. Spin offset
    pub fn to_index(&self) -> u64 {
        let mut index: u64 = 0;

        // States in shells 1 to n-1
        if self.n > 1 {
            let nm1 = (self.n - 1) as u64;
            index += nm1 * (nm1 + 1) * (2 * nm1 + 1) / 3;
        }

        // States in subshells l' < l within shell n
        for l_prime in 0..self.l {
            index += 2 * (2 * l_prime as u64 + 1); // 2(2l'+1) states per subshell
        }

        // States with m' < m in current subshell
        index += (self.m + self.l as i32) as u64 * 2;

        // Spin offset: s=+1 -> 0, s=-1 -> 1
        if self.s == -1 {
            index += 1;
        }

        index
    }

    /// Create from state index (inverse of to_index)
    pub fn from_index(mut index: u64) -> Result<Self> {
        // Find shell n
        let mut n: u32 = 1;
        loop {
            let shell_states = 2 * (n as u64).pow(2);
            let cumulative = n as u64 * (n as u64 + 1) * (2 * n as u64 + 1) / 3;
            if index < cumulative {
                break;
            }
            n += 1;
            if n > 1000 {
                return Err(CatCountError::Overflow(format!("Index {} too large", index)));
            }
        }

        // Subtract states from previous shells
        if n > 1 {
            let nm1 = (n - 1) as u64;
            index -= nm1 * (nm1 + 1) * (2 * nm1 + 1) / 3;
        }

        // Find subshell l
        let mut l: u32 = 0;
        while l < n {
            let subshell_states = 2 * (2 * l as u64 + 1);
            if index < subshell_states {
                break;
            }
            index -= subshell_states;
            l += 1;
        }

        // Find m and s from remaining index
        let m = (index / 2) as i32 - l as i32;
        let s = if index % 2 == 0 { 1 } else { -1 };

        Self::new(n, l, m, s)
    }

    /// Check if a transition to another state is allowed by selection rules
    ///
    /// Selection rules:
    /// - Δl = 0, ±1 (with Δl = 0 forbidden when Δm = 0)
    /// - Δm = 0, ±1
    /// - Δs = 0
    pub fn is_transition_allowed(&self, other: &Self) -> bool {
        let delta_l = (other.l as i32 - self.l as i32).abs();
        let delta_m = (other.m - self.m).abs();
        let delta_s = other.s - self.s;

        // Spin must be conserved
        if delta_s != 0 {
            return false;
        }

        // Check Δl
        if delta_l > 1 {
            return false;
        }

        // Check Δm
        if delta_m > 1 {
            return false;
        }

        // Δl = 0 is forbidden when Δm = 0
        if delta_l == 0 && delta_m == 0 {
            return false;
        }

        true
    }

    /// Get all states in the same shell
    pub fn shell_states(&self) -> Vec<Self> {
        let mut states = Vec::with_capacity(self.shell_degeneracy() as usize);
        for l in 0..self.n {
            for m in -(l as i32)..=(l as i32) {
                for s in [-1, 1] {
                    states.push(Self::new_unchecked(self.n, l, m, s));
                }
            }
        }
        states
    }
}

impl std::fmt::Display for PartitionCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let spin = if self.s == 1 { "↑" } else { "↓" };
        write!(f, "|n={}, l={}, m={}, s={}>", self.n, self.l, self.m, spin)
    }
}

/// Represents a complete partition state with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionState {
    /// Partition coordinates
    pub coord: PartitionCoord,
    /// Energy of this state (if applicable)
    pub energy: Option<f64>,
    /// Occupation probability
    pub occupation: f64,
}

impl PartitionState {
    /// Create a new partition state
    pub fn new(coord: PartitionCoord) -> Self {
        Self {
            coord,
            energy: None,
            occupation: 0.0,
        }
    }

    /// Create with energy
    pub fn with_energy(coord: PartitionCoord, energy: f64) -> Self {
        Self {
            coord,
            energy: Some(energy),
            occupation: 0.0,
        }
    }
}

/// Calculate total degeneracy for principal quantum number n
/// g_n = 2n²
#[inline]
pub fn shell_degeneracy(n: u32) -> u64 {
    2 * (n as u64).pow(2)
}

/// Calculate cumulative state count up to and including shell N
/// G_N = N(N+1)(2N+1)/3
#[inline]
pub fn cumulative_states(n: u32) -> u64 {
    let n = n as u64;
    n * (n + 1) * (2 * n + 1) / 3
}

/// Iterator over all partition states up to a maximum principal number
pub struct PartitionIterator {
    n_max: u32,
    current_n: u32,
    current_l: u32,
    current_m: i32,
    current_s: i8,
}

impl PartitionIterator {
    /// Create a new iterator up to n_max
    pub fn new(n_max: u32) -> Self {
        Self {
            n_max,
            current_n: 1,
            current_l: 0,
            current_m: 0,
            current_s: 1,
        }
    }
}

impl Iterator for PartitionIterator {
    type Item = PartitionCoord;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_n > self.n_max {
            return None;
        }

        let result = PartitionCoord::new_unchecked(
            self.current_n,
            self.current_l,
            self.current_m,
            self.current_s,
        );

        // Advance to next state
        if self.current_s == 1 {
            self.current_s = -1;
        } else {
            self.current_s = 1;
            if self.current_m < self.current_l as i32 {
                self.current_m += 1;
            } else {
                self.current_m = -(self.current_l as i32) - 1;
                if self.current_l + 1 < self.current_n {
                    self.current_l += 1;
                    self.current_m = -(self.current_l as i32);
                } else {
                    self.current_n += 1;
                    self.current_l = 0;
                    self.current_m = 0;
                }
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_partition() {
        let p = PartitionCoord::new(3, 2, -1, 1).unwrap();
        assert_eq!(p.n, 3);
        assert_eq!(p.l, 2);
        assert_eq!(p.m, -1);
        assert_eq!(p.s, 1);
    }

    #[test]
    fn test_invalid_n() {
        assert!(PartitionCoord::new(0, 0, 0, 1).is_err());
    }

    #[test]
    fn test_invalid_l() {
        assert!(PartitionCoord::new(2, 2, 0, 1).is_err()); // l must be < n
    }

    #[test]
    fn test_invalid_m() {
        assert!(PartitionCoord::new(2, 1, 2, 1).is_err()); // |m| must be <= l
    }

    #[test]
    fn test_invalid_s() {
        assert!(PartitionCoord::new(1, 0, 0, 0).is_err());
        assert!(PartitionCoord::new(1, 0, 0, 2).is_err());
    }

    #[test]
    fn test_shell_degeneracy() {
        assert_eq!(shell_degeneracy(1), 2);   // 2 * 1² = 2
        assert_eq!(shell_degeneracy(2), 8);   // 2 * 2² = 8
        assert_eq!(shell_degeneracy(3), 18);  // 2 * 3² = 18
        assert_eq!(shell_degeneracy(4), 32);  // 2 * 4² = 32
    }

    #[test]
    fn test_cumulative_states() {
        assert_eq!(cumulative_states(1), 2);   // 2
        assert_eq!(cumulative_states(2), 10);  // 2 + 8 = 10
        assert_eq!(cumulative_states(3), 28);  // 2 + 8 + 18 = 28
    }

    #[test]
    fn test_index_roundtrip() {
        for n in 1..=4 {
            for l in 0..n {
                for m in -(l as i32)..=(l as i32) {
                    for s in [-1, 1] {
                        let p = PartitionCoord::new_unchecked(n, l, m, s);
                        let idx = p.to_index();
                        let p2 = PartitionCoord::from_index(idx).unwrap();
                        assert_eq!(p, p2, "Roundtrip failed for {:?}", p);
                    }
                }
            }
        }
    }

    #[test]
    fn test_transition_rules() {
        let s1 = PartitionCoord::new(2, 1, 0, 1).unwrap();
        let s2 = PartitionCoord::new(2, 0, 0, 1).unwrap();

        // Δl = 1, Δm = 0, Δs = 0 - should be allowed
        assert!(s1.is_transition_allowed(&s2));

        // Same state - not allowed (Δl = 0, Δm = 0)
        assert!(!s1.is_transition_allowed(&s1));
    }

    #[test]
    fn test_iterator() {
        let iter = PartitionIterator::new(2);
        let states: Vec<_> = iter.collect();
        assert_eq!(states.len(), 10); // 2 + 8 = 10 states
    }
}
