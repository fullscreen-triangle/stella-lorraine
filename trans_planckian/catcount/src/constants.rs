//! Physical constants for trans-Planckian calculations
//!
//! This module provides fundamental physical constants used throughout
//! the categorical state counting framework.

use std::f64::consts::{E, PI};

// =============================================================================
// FUNDAMENTAL CONSTANTS (CODATA 2018)
// =============================================================================

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380649e-23;

/// Reduced Planck constant (J·s)
pub const H_BAR: f64 = 1.054571817e-34;

/// Planck constant (J·s)
pub const H: f64 = 6.62607015e-34;

/// Speed of light in vacuum (m/s)
pub const C: f64 = 299792458.0;

/// Gravitational constant (m³/(kg·s²))
pub const G: f64 = 6.67430e-11;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602176634e-19;

/// Electron mass (kg)
pub const M_E: f64 = 9.1093837015e-31;

/// Proton mass (kg)
pub const M_P: f64 = 1.67262192369e-27;

/// Avogadro constant (1/mol)
pub const N_A: f64 = 6.02214076e23;

// =============================================================================
// PLANCK UNITS
// =============================================================================

/// Planck time (s)
/// t_P = √(ℏG/c⁵) ≈ 5.391 × 10⁻⁴⁴ s
pub const PLANCK_TIME: f64 = 5.391247e-44;

/// Planck length (m)
/// l_P = √(ℏG/c³) ≈ 1.616 × 10⁻³⁵ m
pub const PLANCK_LENGTH: f64 = 1.616255e-35;

/// Planck mass (kg)
/// m_P = √(ℏc/G) ≈ 2.176 × 10⁻⁸ kg
pub const PLANCK_MASS: f64 = 2.176434e-8;

/// Planck energy (J)
/// E_P = √(ℏc⁵/G) ≈ 1.956 × 10⁹ J
pub const PLANCK_ENERGY: f64 = 1.9561e9;

/// Planck temperature (K)
/// T_P = √(ℏc⁵/(Gk_B²)) ≈ 1.417 × 10³² K
pub const PLANCK_TEMPERATURE: f64 = 1.416785e32;

/// Planck frequency (Hz)
/// ν_P = 1/t_P ≈ 1.855 × 10⁴³ Hz
pub const PLANCK_FREQUENCY: f64 = 1.854859e43;

// =============================================================================
// REFERENCE FREQUENCIES
// =============================================================================

/// CO molecular vibration frequency (Hz)
pub const FREQ_CO_VIBRATION: f64 = 5.13e13;

/// Lyman-alpha electronic frequency (Hz)
pub const FREQ_LYMAN_ALPHA: f64 = 2.47e15;

/// Electron Compton frequency (Hz)
pub const FREQ_ELECTRON_COMPTON: f64 = 1.24e20;

/// Schwarzschild frequency for electron (Hz)
pub const FREQ_SCHWARZSCHILD_ELECTRON: f64 = 1.35e53;

// =============================================================================
// ENHANCEMENT PARAMETERS
// =============================================================================

/// Number of ternary encoding levels
pub const N_TERNARY_LEVELS: u32 = 20;

/// Number of measurement modalities
pub const N_MODALITIES: u32 = 5;

/// Measurements per modality
pub const N_MEASUREMENTS_PER_MODALITY: u32 = 100;

/// Number of harmonic coincidences
pub const N_HARMONIC_COINCIDENCES: u32 = 12;

/// Number of categorical states for Poincaré computing
pub const N_POINCARE_STATES: u64 = 1_000_000;

/// Observation time for Poincaré computing (s)
pub const T_POINCARE_OBSERVATION: f64 = 100.0;

/// Characteristic recurrence time (s)
pub const TAU_RECURRENCE: f64 = 1e-4;

/// Continuous refinement integration time (s)
pub const T_REFINEMENT: f64 = 100.0;

/// Refinement recurrence time (s)
pub const TAU_REFINEMENT: f64 = 1.0;

// =============================================================================
// SPECTROSCOPY CONSTANTS
// =============================================================================

/// Wavenumber to frequency conversion factor (Hz/cm⁻¹)
/// ν = c̃ × c where c̃ is wavenumber in cm⁻¹
pub const WAVENUMBER_TO_HZ: f64 = C * 100.0; // cm⁻¹ → m⁻¹ → Hz

/// eV to Hz conversion factor
pub const EV_TO_HZ: f64 = E_CHARGE / H;

// =============================================================================
// COMPUTED CONSTANTS
// =============================================================================

/// Log₁₀ of Planck time
pub fn log10_planck_time() -> f64 {
    PLANCK_TIME.log10()
}

/// Log₁₀ of Planck frequency
pub fn log10_planck_frequency() -> f64 {
    PLANCK_FREQUENCY.log10()
}

/// Calculate Planck time from fundamentals
pub fn compute_planck_time() -> f64 {
    (H_BAR * G / C.powi(5)).sqrt()
}

/// Calculate Planck length from fundamentals
pub fn compute_planck_length() -> f64 {
    (H_BAR * G / C.powi(3)).sqrt()
}

/// Calculate Planck mass from fundamentals
pub fn compute_planck_mass() -> f64 {
    (H_BAR * C / G).sqrt()
}

// =============================================================================
// UNIT CONVERSIONS
// =============================================================================

/// Convert wavenumber (cm⁻¹) to frequency (Hz)
#[inline]
pub fn wavenumber_to_frequency(wavenumber_cm: f64) -> f64 {
    wavenumber_cm * WAVENUMBER_TO_HZ
}

/// Convert frequency (Hz) to wavenumber (cm⁻¹)
#[inline]
pub fn frequency_to_wavenumber(frequency_hz: f64) -> f64 {
    frequency_hz / WAVENUMBER_TO_HZ
}

/// Convert eV to Hz
#[inline]
pub fn ev_to_hz(ev: f64) -> f64 {
    ev * EV_TO_HZ
}

/// Convert Hz to eV
#[inline]
pub fn hz_to_ev(hz: f64) -> f64 {
    hz / EV_TO_HZ
}

/// Convert temperature (K) to thermal frequency (Hz)
/// ν_thermal = k_B T / h
#[inline]
pub fn temperature_to_frequency(temperature_k: f64) -> f64 {
    K_B * temperature_k / H
}

/// Convert frequency (Hz) to temperature (K)
#[inline]
pub fn frequency_to_temperature(frequency_hz: f64) -> f64 {
    frequency_hz * H / K_B
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planck_time_computation() {
        let computed = compute_planck_time();
        let relative_error = (computed - PLANCK_TIME).abs() / PLANCK_TIME;
        assert!(relative_error < 1e-4, "Planck time computation error: {}", relative_error);
    }

    #[test]
    fn test_wavenumber_conversion() {
        // 1000 cm⁻¹ should be about 3e13 Hz
        let freq = wavenumber_to_frequency(1000.0);
        assert!((freq - 2.998e13).abs() / 2.998e13 < 0.01);

        // Round-trip conversion
        let wavenumber = frequency_to_wavenumber(freq);
        assert!((wavenumber - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_planck_frequency() {
        let computed = 1.0 / PLANCK_TIME;
        let relative_error = (computed - PLANCK_FREQUENCY).abs() / PLANCK_FREQUENCY;
        assert!(relative_error < 1e-4);
    }
}
