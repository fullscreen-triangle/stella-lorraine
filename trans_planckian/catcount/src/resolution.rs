//! Temporal Resolution Calculations
//!
//! This module calculates categorical temporal resolution using the formula:
//!
//! δt = t_P / (E × (ν/ν_P))
//!
//! where:
//! - t_P is the Planck time
//! - E is the total enhancement factor
//! - ν is the process frequency
//! - ν_P is the Planck frequency

use crate::constants::*;
use crate::enhancement::EnhancementChain;
use serde::{Deserialize, Serialize};

/// Result of a temporal resolution calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalResolution {
    /// Process frequency (Hz)
    pub frequency_hz: f64,
    /// Categorical temporal resolution (seconds)
    pub delta_t: f64,
    /// Log10 of the resolution
    pub log10_delta_t: f64,
    /// Log10 of the enhancement factor used
    pub log10_enhancement: f64,
    /// Whether this resolution is below Planck time
    pub is_trans_planckian: bool,
}

impl TemporalResolution {
    /// Calculate temporal resolution at given frequency with enhancement chain
    ///
    /// Formula: δt = t_P / (E × (ν/ν_P))
    ///
    /// In log form: log10(δt) = log10(t_P) - log10(E) - log10(ν) + log10(ν_P)
    ///
    /// Since ν_P = 1/t_P, we have log10(ν_P) = -log10(t_P), so:
    /// log10(δt) = -log10(E) - log10(ν) + 2*log10(t_P)
    ///
    /// But more simply: log10(δt) = log10(t_P) - log10(E) - log10(ν/ν_P)
    pub fn calculate(frequency_hz: f64, chain: &EnhancementChain) -> Self {
        let log10_enhancement = chain.total_log10();

        // Calculate log10(δt) = log10(t_P) - log10(E) - log10(ν) + log10(ν_P)
        // = log10(t_P) - log10(E) - log10(ν/ν_P)
        let log10_freq_ratio = (frequency_hz / PLANCK_FREQUENCY).log10();
        let log10_delta_t = PLANCK_TIME.log10() - log10_enhancement - log10_freq_ratio;

        let delta_t = 10.0_f64.powf(log10_delta_t);
        let is_trans_planckian = delta_t < PLANCK_TIME;

        Self {
            frequency_hz,
            delta_t,
            log10_delta_t,
            log10_enhancement,
            is_trans_planckian,
        }
    }

    /// Calculate resolution with full enhancement chain
    pub fn with_full_enhancement(frequency_hz: f64) -> Self {
        Self::calculate(frequency_hz, &EnhancementChain::full())
    }

    /// Calculate resolution with no enhancement (base Planck resolution)
    pub fn without_enhancement(frequency_hz: f64) -> Self {
        Self::calculate(frequency_hz, &EnhancementChain::none())
    }

    /// Calculate orders of magnitude below Planck time
    pub fn orders_below_planck(&self) -> f64 {
        PLANCK_TIME.log10() - self.log10_delta_t
    }

    /// Calculate the frequency ratio ν/ν_P
    pub fn frequency_ratio(&self) -> f64 {
        self.frequency_hz / PLANCK_FREQUENCY
    }

    /// Calculate log10 of frequency ratio
    pub fn log10_frequency_ratio(&self) -> f64 {
        self.frequency_ratio().log10()
    }

    /// Check if the resolution exceeds a target number of orders below Planck
    pub fn exceeds_target(&self, target_orders: f64) -> bool {
        self.orders_below_planck() >= target_orders
    }

    /// Get resolution relative to Planck time
    pub fn planck_ratio(&self) -> f64 {
        self.delta_t / PLANCK_TIME
    }
}

impl std::fmt::Display for TemporalResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Temporal Resolution Calculation")?;
        writeln!(f, "================================")?;
        writeln!(f, "Process frequency:      {:.3e} Hz", self.frequency_hz)?;
        writeln!(f, "Enhancement applied:    10^{:.2}", self.log10_enhancement)?;
        writeln!(f, "Categorical resolution: {:.3e} s", self.delta_t)?;
        writeln!(f, "Orders below Planck:    {:.2}", self.orders_below_planck())?;
        writeln!(f, "Trans-Planckian:        {}", if self.is_trans_planckian { "YES" } else { "NO" })
    }
}

/// Multi-scale resolution calculation across frequency range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleResolution {
    /// Individual resolution calculations
    pub resolutions: Vec<TemporalResolution>,
    /// Scaling law slope (should be -1)
    pub scaling_slope: f64,
    /// Scaling law intercept
    pub scaling_intercept: f64,
    /// R² of the scaling fit
    pub r_squared: f64,
}

impl MultiScaleResolution {
    /// Calculate resolutions across multiple frequencies
    pub fn calculate(frequencies: &[f64], chain: &EnhancementChain) -> Self {
        let resolutions: Vec<_> = frequencies
            .iter()
            .map(|&f| TemporalResolution::calculate(f, chain))
            .collect();

        // Linear regression: log10(δt) vs log10(ν)
        let (slope, intercept, r_squared) = Self::linear_regression(&resolutions);

        Self {
            resolutions,
            scaling_slope: slope,
            scaling_intercept: intercept,
            r_squared,
        }
    }

    /// Calculate at standard reference frequencies
    pub fn standard() -> Self {
        let frequencies = vec![
            FREQ_CO_VIBRATION,
            FREQ_LYMAN_ALPHA,
            FREQ_ELECTRON_COMPTON,
            PLANCK_FREQUENCY,
            FREQ_SCHWARZSCHILD_ELECTRON,
        ];
        Self::calculate(&frequencies, &EnhancementChain::full())
    }

    /// Perform linear regression on log-log data
    fn linear_regression(resolutions: &[TemporalResolution]) -> (f64, f64, f64) {
        let n = resolutions.len() as f64;

        let log_f: Vec<f64> = resolutions.iter().map(|r| r.frequency_hz.log10()).collect();
        let log_t: Vec<f64> = resolutions.iter().map(|r| r.log10_delta_t).collect();

        let mean_f: f64 = log_f.iter().sum::<f64>() / n;
        let mean_t: f64 = log_t.iter().sum::<f64>() / n;

        let mut ss_ff = 0.0;
        let mut ss_ft = 0.0;
        let mut ss_tt = 0.0;

        for i in 0..resolutions.len() {
            let df = log_f[i] - mean_f;
            let dt = log_t[i] - mean_t;
            ss_ff += df * df;
            ss_ft += df * dt;
            ss_tt += dt * dt;
        }

        let slope = ss_ft / ss_ff;
        let intercept = mean_t - slope * mean_f;
        let r_squared = (ss_ft * ss_ft) / (ss_ff * ss_tt);

        (slope, intercept, r_squared)
    }

    /// Check if scaling law is validated (slope ≈ -1)
    pub fn is_scaling_validated(&self, tolerance: f64) -> bool {
        (self.scaling_slope + 1.0).abs() < tolerance
    }

    /// Get minimum resolution achieved
    pub fn min_resolution(&self) -> Option<&TemporalResolution> {
        self.resolutions.iter().min_by(|a, b| {
            a.delta_t.partial_cmp(&b.delta_t).unwrap()
        })
    }

    /// Get maximum orders below Planck
    pub fn max_orders_below_planck(&self) -> f64 {
        self.resolutions
            .iter()
            .map(|r| r.orders_below_planck())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Calculate resolution at a single frequency with full enhancement
pub fn resolve(frequency_hz: f64) -> TemporalResolution {
    TemporalResolution::with_full_enhancement(frequency_hz)
}

/// Calculate resolution from wavenumber (cm⁻¹)
pub fn resolve_wavenumber(wavenumber_cm: f64) -> TemporalResolution {
    let frequency = wavenumber_to_frequency(wavenumber_cm);
    resolve(frequency)
}

/// Calculate resolution from energy (eV)
pub fn resolve_energy(energy_ev: f64) -> TemporalResolution {
    let frequency = ev_to_hz(energy_ev);
    resolve(frequency)
}

/// Calculate resolution from temperature (K)
pub fn resolve_temperature(temperature_k: f64) -> TemporalResolution {
    let frequency = temperature_to_frequency(temperature_k);
    resolve(frequency)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_calculation() {
        let res = resolve(5.13e13);
        assert!(res.is_trans_planckian);
        assert!(res.orders_below_planck() > 90.0);
    }

    #[test]
    fn test_scaling_law() {
        let multi = MultiScaleResolution::standard();

        // Slope should be very close to -1
        assert!((multi.scaling_slope + 1.0).abs() < 0.001);

        // R² should be essentially 1
        assert!(multi.r_squared > 0.9999);
    }

    #[test]
    fn test_no_enhancement() {
        let res = TemporalResolution::without_enhancement(5.13e13);

        // Without enhancement, resolution should be worse than Planck time
        // at molecular frequencies
        assert!(res.delta_t > PLANCK_TIME);
    }

    #[test]
    fn test_planck_frequency_resolution() {
        let res = resolve(PLANCK_FREQUENCY);

        // At Planck frequency with full enhancement, should get maximum resolution
        assert!(res.orders_below_planck() > 120.0);
    }

    #[test]
    fn test_wavenumber_conversion() {
        // 1715 cm⁻¹ (C=O stretch of vanillin)
        let res = resolve_wavenumber(1715.0);
        assert!(res.is_trans_planckian);
    }

    #[test]
    fn test_display() {
        let res = resolve(5.13e13);
        let display = format!("{}", res);
        assert!(display.contains("Trans-Planckian"));
        assert!(display.contains("YES"));
    }
}
