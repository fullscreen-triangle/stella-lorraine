//! S-Entropy Coordinate System
//!
//! The S-entropy coordinate system (S_k, S_t, S_e) provides the natural
//! geometry for categorical state space. These coordinates are defined
//! in terms of entropy-like quantities.

use crate::constants::K_B;
use crate::error::{CatCountError, Result};
use serde::{Deserialize, Serialize};

/// Reference values for S-entropy coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SEntropyReferences {
    /// Reference phase deviation (radians)
    pub phi_0: f64,
    /// Reference period (seconds)
    pub tau_0: f64,
    /// Reference energy (joules)
    pub e_0: f64,
}

impl Default for SEntropyReferences {
    fn default() -> Self {
        Self {
            phi_0: 1e-10,  // Small reference phase
            tau_0: 1e-15,  // Femtosecond reference period
            e_0: K_B * 300.0, // Thermal energy at room temperature
        }
    }
}

/// S-Entropy coordinates in categorical state space
///
/// The S-entropy coordinate system is defined by:
/// - S_k = k_B * ln((|δφ| + φ₀) / φ₀)  - phase deviation entropy
/// - S_t = k_B * ln(τ / τ₀)            - period entropy
/// - S_e = k_B * ln((E + E₀) / E₀)     - energy entropy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SEntropyCoord {
    /// S_k: Phase deviation entropy (J/K)
    pub s_k: f64,
    /// S_t: Period entropy (J/K)
    pub s_t: f64,
    /// S_e: Energy entropy (J/K)
    pub s_e: f64,
}

impl SEntropyCoord {
    /// Create new S-entropy coordinates with validation
    pub fn new(s_k: f64, s_t: f64, s_e: f64) -> Result<Self> {
        if s_k < 0.0 {
            return Err(CatCountError::invalid_s_entropy("S_k", s_k));
        }
        if s_t < 0.0 {
            return Err(CatCountError::invalid_s_entropy("S_t", s_t));
        }
        if s_e < 0.0 {
            return Err(CatCountError::invalid_s_entropy("S_e", s_e));
        }
        Ok(Self { s_k, s_t, s_e })
    }

    /// Create S-entropy coordinates without validation (unsafe but fast)
    #[inline]
    pub fn new_unchecked(s_k: f64, s_t: f64, s_e: f64) -> Self {
        Self { s_k, s_t, s_e }
    }

    /// Create from physical quantities
    ///
    /// # Arguments
    /// * `delta_phi` - Phase deviation from reference oscillator (radians)
    /// * `tau` - Categorical period (seconds)
    /// * `energy` - State energy (joules)
    /// * `refs` - Reference values
    pub fn from_physical(
        delta_phi: f64,
        tau: f64,
        energy: f64,
        refs: &SEntropyReferences,
    ) -> Result<Self> {
        if tau <= 0.0 {
            return Err(CatCountError::InvalidEnhancementParam(
                format!("Period must be positive: {}", tau)
            ));
        }

        let s_k = K_B * ((delta_phi.abs() + refs.phi_0) / refs.phi_0).ln();
        let s_t = K_B * (tau / refs.tau_0).ln();
        let s_e = K_B * ((energy + refs.e_0) / refs.e_0).ln();

        Self::new(s_k, s_t, s_e)
    }

    /// Create from physical quantities with default references
    pub fn from_physical_default(delta_phi: f64, tau: f64, energy: f64) -> Result<Self> {
        Self::from_physical(delta_phi, tau, energy, &SEntropyReferences::default())
    }

    /// Origin of S-entropy space (all coordinates zero)
    pub fn origin() -> Self {
        Self { s_k: 0.0, s_t: 0.0, s_e: 0.0 }
    }

    /// Calculate the Euclidean distance from origin (magnitude)
    pub fn magnitude(&self) -> f64 {
        (self.s_k.powi(2) + self.s_t.powi(2) + self.s_e.powi(2)).sqrt()
    }

    /// Calculate the Euclidean distance between two S-entropy coordinates
    pub fn distance(&self, other: &Self) -> f64 {
        let dk = self.s_k - other.s_k;
        let dt = self.s_t - other.s_t;
        let de = self.s_e - other.s_e;
        (dk.powi(2) + dt.powi(2) + de.powi(2)).sqrt()
    }

    /// Calculate the total entropy (sum of components)
    pub fn total_entropy(&self) -> f64 {
        self.s_k + self.s_t + self.s_e
    }

    /// Calculate the entropy in units of k_B
    pub fn total_entropy_kb(&self) -> f64 {
        self.total_entropy() / K_B
    }

    /// Interpolate between two S-entropy coordinates
    ///
    /// # Arguments
    /// * `other` - Target coordinates
    /// * `t` - Interpolation parameter [0, 1]
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            s_k: self.s_k + t * (other.s_k - self.s_k),
            s_t: self.s_t + t * (other.s_t - self.s_t),
            s_e: self.s_e + t * (other.s_e - self.s_e),
        }
    }

    /// Generate a geodesic path (straight line) between two coordinates
    ///
    /// # Arguments
    /// * `end` - End point
    /// * `n_points` - Number of points along the path
    pub fn geodesic(&self, end: &Self, n_points: usize) -> Vec<Self> {
        (0..n_points)
            .map(|i| {
                let t = i as f64 / (n_points - 1).max(1) as f64;
                self.interpolate(end, t)
            })
            .collect()
    }

    /// Convert to array representation
    pub fn to_array(&self) -> [f64; 3] {
        [self.s_k, self.s_t, self.s_e]
    }

    /// Create from array representation
    pub fn from_array(arr: [f64; 3]) -> Result<Self> {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl std::ops::Add for SEntropyCoord {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            s_k: self.s_k + other.s_k,
            s_t: self.s_t + other.s_t,
            s_e: self.s_e + other.s_e,
        }
    }
}

impl std::ops::Sub for SEntropyCoord {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            s_k: self.s_k - other.s_k,
            s_t: self.s_t - other.s_t,
            s_e: self.s_e - other.s_e,
        }
    }
}

impl std::ops::Mul<f64> for SEntropyCoord {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            s_k: self.s_k * scalar,
            s_t: self.s_t * scalar,
            s_e: self.s_e * scalar,
        }
    }
}

impl std::fmt::Display for SEntropyCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "S-coord(S_k={:.3e}, S_t={:.3e}, S_e={:.3e}) J/K",
            self.s_k, self.s_t, self.s_e
        )
    }
}

/// Metric tensor for S-entropy space (Euclidean)
#[derive(Debug, Clone, Copy)]
pub struct SEntropyMetric;

impl SEntropyMetric {
    /// Calculate the line element ds² = dS_k² + dS_t² + dS_e²
    pub fn line_element(ds: &SEntropyCoord) -> f64 {
        ds.s_k.powi(2) + ds.s_t.powi(2) + ds.s_e.powi(2)
    }

    /// Calculate geodesic distance (same as Euclidean distance)
    pub fn geodesic_distance(a: &SEntropyCoord, b: &SEntropyCoord) -> f64 {
        a.distance(b)
    }

    /// Check if the metric is flat (always true for Euclidean)
    pub fn is_flat() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_entropy_creation() {
        let coord = SEntropyCoord::new(1e-23, 2e-23, 3e-23).unwrap();
        assert_eq!(coord.s_k, 1e-23);
        assert_eq!(coord.s_t, 2e-23);
        assert_eq!(coord.s_e, 3e-23);
    }

    #[test]
    fn test_s_entropy_negative_rejection() {
        assert!(SEntropyCoord::new(-1e-23, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_from_physical() {
        let refs = SEntropyReferences::default();
        let coord = SEntropyCoord::from_physical(1e-8, 1e-12, 1e-20, &refs).unwrap();
        assert!(coord.s_k > 0.0);
        assert!(coord.s_t > 0.0);
        assert!(coord.s_e >= 0.0);
    }

    #[test]
    fn test_distance() {
        let a = SEntropyCoord::origin();
        let b = SEntropyCoord::new_unchecked(3e-23, 4e-23, 0.0);
        let dist = a.distance(&b);
        assert!((dist - 5e-23).abs() < 1e-30);
    }

    #[test]
    fn test_interpolation() {
        let a = SEntropyCoord::origin();
        let b = SEntropyCoord::new_unchecked(1e-23, 2e-23, 3e-23);
        let mid = a.interpolate(&b, 0.5);
        assert!((mid.s_k - 0.5e-23).abs() < 1e-30);
        assert!((mid.s_t - 1.0e-23).abs() < 1e-30);
        assert!((mid.s_e - 1.5e-23).abs() < 1e-30);
    }

    #[test]
    fn test_geodesic() {
        let a = SEntropyCoord::origin();
        let b = SEntropyCoord::new_unchecked(1e-23, 0.0, 0.0);
        let path = a.geodesic(&b, 11);
        assert_eq!(path.len(), 11);
        assert_eq!(path[0].s_k, 0.0);
        assert!((path[10].s_k - 1e-23).abs() < 1e-30);
    }
}
