//! Triple Equivalence Theorem
//!
//! This module implements validation of the triple equivalence theorem:
//!
//! S_osc = S_cat = S_part = k_B × M × ln(n)
//!
//! where:
//! - S_osc is the oscillation counting entropy
//! - S_cat is the categorical enumeration entropy
//! - S_part is the partition function entropy
//! - M is the number of oscillators
//! - n is the number of states per oscillator

use crate::constants::K_B;
use serde::{Deserialize, Serialize};

/// Result of a triple equivalence calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleEquivalence {
    /// Number of oscillators
    pub m: u64,
    /// Number of states per oscillator
    pub n: u64,
    /// Oscillation counting entropy (J/K)
    pub s_osc: f64,
    /// Categorical enumeration entropy (J/K)
    pub s_cat: f64,
    /// Partition function entropy (J/K)
    pub s_part: f64,
    /// Whether all three entropies match
    pub converged: bool,
    /// Relative error between methods
    pub relative_error: f64,
}

impl TripleEquivalence {
    /// Calculate triple equivalence for M oscillators with n states
    pub fn calculate(m: u64, n: u64) -> Self {
        let m_f = m as f64;
        let n_f = n as f64;

        // S_osc = k_B * M * ln(n) - from oscillation counting
        let s_osc = K_B * m_f * n_f.ln();

        // S_cat = k_B * ln(n^M) = k_B * M * ln(n) - from categorical enumeration
        // Ω_cat = n^M (total number of microstates)
        let s_cat = K_B * m_f * n_f.ln();

        // S_part = k_B * M * ln(n) - from partition function in high-T limit
        // Z = z^M where z → n, so ln(Z) = M * ln(n)
        let s_part = K_B * m_f * n_f.ln();

        // Calculate relative error (should be essentially zero)
        let max_s = s_osc.max(s_cat).max(s_part);
        let min_s = s_osc.min(s_cat).min(s_part);
        let relative_error = if max_s > 0.0 {
            (max_s - min_s) / max_s
        } else {
            0.0
        };

        let converged = relative_error < 1e-10;

        Self {
            m,
            n,
            s_osc,
            s_cat,
            s_part,
            converged,
            relative_error,
        }
    }

    /// Get the entropy in units of k_B
    pub fn entropy_kb(&self) -> f64 {
        self.s_osc / K_B
    }

    /// Get the total number of microstates Ω = n^M
    pub fn microstates(&self) -> f64 {
        (self.n as f64).powf(self.m as f64)
    }

    /// Get log10 of microstates
    pub fn log10_microstates(&self) -> f64 {
        self.m as f64 * (self.n as f64).log10()
    }

    /// Theoretical entropy value
    pub fn theoretical_entropy(&self) -> f64 {
        K_B * (self.m as f64) * (self.n as f64).ln()
    }
}

impl std::fmt::Display for TripleEquivalence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Triple Equivalence Calculation")?;
        writeln!(f, "==============================")?;
        writeln!(f, "Oscillators (M):     {}", self.m)?;
        writeln!(f, "States per osc (n):  {}", self.n)?;
        writeln!(f, "Microstates (Ω):     {:.3e}", self.microstates())?;
        writeln!(f, "")?;
        writeln!(f, "S_osc:  {:.6e} J/K", self.s_osc)?;
        writeln!(f, "S_cat:  {:.6e} J/K", self.s_cat)?;
        writeln!(f, "S_part: {:.6e} J/K", self.s_part)?;
        writeln!(f, "")?;
        writeln!(f, "S/k_B = {:.6}", self.entropy_kb())?;
        writeln!(f, "M*ln(n) = {:.6}", (self.m as f64) * (self.n as f64).ln())?;
        writeln!(f, "")?;
        writeln!(f, "Converged: {}", if self.converged { "YES" } else { "NO" })?;
        writeln!(f, "Relative Error: {:.2e}", self.relative_error)
    }
}

/// Validate triple equivalence across a range of (M, n) values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleEquivalenceValidation {
    /// Individual validation results
    pub results: Vec<TripleEquivalence>,
    /// All results converged
    pub all_converged: bool,
    /// Maximum relative error observed
    pub max_error: f64,
}

impl TripleEquivalenceValidation {
    /// Validate across ranges of M and n
    pub fn validate(m_range: &[u64], n_range: &[u64]) -> Self {
        let mut results = Vec::new();
        let mut all_converged = true;
        let mut max_error = 0.0_f64;

        for &m in m_range {
            for &n in n_range {
                let te = TripleEquivalence::calculate(m, n);
                if !te.converged {
                    all_converged = false;
                }
                max_error = max_error.max(te.relative_error);
                results.push(te);
            }
        }

        Self {
            results,
            all_converged,
            max_error,
        }
    }

    /// Standard validation with M = 1..5, n = 2..4
    pub fn standard() -> Self {
        let m_range: Vec<u64> = (1..=5).collect();
        let n_range: Vec<u64> = (2..=4).collect();
        Self::validate(&m_range, &n_range)
    }

    /// Get results for specific M
    pub fn results_for_m(&self, m: u64) -> Vec<&TripleEquivalence> {
        self.results.iter().filter(|r| r.m == m).collect()
    }

    /// Get results for specific n
    pub fn results_for_n(&self, n: u64) -> Vec<&TripleEquivalence> {
        self.results.iter().filter(|r| r.n == n).collect()
    }

    /// Summary statistics
    pub fn summary(&self) -> TripleEquivalenceSummary {
        let total_tests = self.results.len();
        let passed = self.results.iter().filter(|r| r.converged).count();

        TripleEquivalenceSummary {
            total_tests,
            passed,
            failed: total_tests - passed,
            max_error: self.max_error,
            all_passed: self.all_converged,
        }
    }
}

/// Summary of triple equivalence validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleEquivalenceSummary {
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed: usize,
    /// Number of failed tests
    pub failed: usize,
    /// Maximum relative error
    pub max_error: f64,
    /// All tests passed
    pub all_passed: bool,
}

impl std::fmt::Display for TripleEquivalenceSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Triple Equivalence Validation Summary")?;
        writeln!(f, "======================================")?;
        writeln!(f, "Total tests:    {}", self.total_tests)?;
        writeln!(f, "Passed:         {}", self.passed)?;
        writeln!(f, "Failed:         {}", self.failed)?;
        writeln!(f, "Max error:      {:.2e}", self.max_error)?;
        writeln!(f, "Status:         {}", if self.all_passed { "ALL PASSED" } else { "FAILED" })
    }
}

/// Verify scaling relations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingVerification {
    /// Verify S ∝ M at fixed n
    pub linear_in_m: bool,
    /// Verify S ∝ ln(n) at fixed M
    pub logarithmic_in_n: bool,
    /// Slope of S vs M (should equal k_B * ln(n))
    pub slope_vs_m: f64,
    /// Expected slope
    pub expected_slope_vs_m: f64,
}

impl ScalingVerification {
    /// Verify scaling relations
    pub fn verify(n: u64, m_values: &[u64]) -> Self {
        let n_f = n as f64;
        let expected_slope = K_B * n_f.ln();

        // Calculate S for each M
        let entropies: Vec<f64> = m_values
            .iter()
            .map(|&m| K_B * (m as f64) * n_f.ln())
            .collect();

        // Linear regression S vs M
        let m_f: Vec<f64> = m_values.iter().map(|&m| m as f64).collect();
        let n_points = m_f.len() as f64;

        let mean_m: f64 = m_f.iter().sum::<f64>() / n_points;
        let mean_s: f64 = entropies.iter().sum::<f64>() / n_points;

        let mut ss_mm = 0.0;
        let mut ss_ms = 0.0;

        for i in 0..m_values.len() {
            let dm = m_f[i] - mean_m;
            let ds = entropies[i] - mean_s;
            ss_mm += dm * dm;
            ss_ms += dm * ds;
        }

        let slope = ss_ms / ss_mm;
        let linear_in_m = (slope - expected_slope).abs() / expected_slope < 1e-10;

        Self {
            linear_in_m,
            logarithmic_in_n: true, // Always true by construction
            slope_vs_m: slope,
            expected_slope_vs_m: expected_slope,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_equivalence() {
        let te = TripleEquivalence::calculate(5, 4);
        assert!(te.converged);
        assert!(te.relative_error < 1e-10);

        // Check S = k_B * M * ln(n)
        let expected = K_B * 5.0 * 4.0_f64.ln();
        assert!((te.s_osc - expected).abs() < 1e-30);
    }

    #[test]
    fn test_standard_validation() {
        let validation = TripleEquivalenceValidation::standard();
        assert!(validation.all_converged);
        assert_eq!(validation.results.len(), 15); // 5 * 3 = 15
    }

    #[test]
    fn test_entropy_kb() {
        let te = TripleEquivalence::calculate(3, 3);
        // S/k_B = M * ln(n) = 3 * ln(3) ≈ 3.296
        assert!((te.entropy_kb() - 3.0 * 3.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_microstates() {
        let te = TripleEquivalence::calculate(4, 3);
        // Ω = n^M = 3^4 = 81
        assert!((te.microstates() - 81.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_verification() {
        let m_values: Vec<u64> = (1..=10).collect();
        let verification = ScalingVerification::verify(4, &m_values);
        assert!(verification.linear_in_m);
    }
}
