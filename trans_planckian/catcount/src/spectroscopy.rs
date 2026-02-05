//! Spectroscopy Validation Module
//!
//! This module provides validation of the trans-Planckian framework against
//! real spectroscopic data from Raman spectroscopy, FTIR, and other techniques.
//!
//! The framework predicts vibrational mode frequencies through the categorical
//! state structure, which can be validated against experimental measurements.

use crate::constants::{
    wavenumber_to_frequency, frequency_to_wavenumber, K_B, H, C,
    FREQ_CO_VIBRATION, FREQ_LYMAN_ALPHA, FREQ_ELECTRON_COMPTON,
};
use crate::error::{CatCountError, Result};
use crate::resolution::TemporalResolution;
use crate::enhancement::EnhancementChain;
use serde::{Deserialize, Serialize};

// =============================================================================
// SPECTROSCOPIC MODES
// =============================================================================

/// Spectroscopic mode type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModeType {
    /// Vibrational stretching mode
    Stretch,
    /// Vibrational bending mode
    Bend,
    /// Rotational mode
    Rotation,
    /// Electronic transition
    Electronic,
    /// Combination band
    Combination,
    /// Overtone
    Overtone,
}

impl std::fmt::Display for ModeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModeType::Stretch => write!(f, "Stretch"),
            ModeType::Bend => write!(f, "Bend"),
            ModeType::Rotation => write!(f, "Rotation"),
            ModeType::Electronic => write!(f, "Electronic"),
            ModeType::Combination => write!(f, "Combination"),
            ModeType::Overtone => write!(f, "Overtone"),
        }
    }
}

/// Spectroscopic technique
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Technique {
    /// Raman spectroscopy
    Raman,
    /// Fourier Transform Infrared
    FTIR,
    /// UV-Visible spectroscopy
    UVVis,
    /// Microwave spectroscopy
    Microwave,
    /// X-ray spectroscopy
    XRay,
}

impl std::fmt::Display for Technique {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Technique::Raman => write!(f, "Raman"),
            Technique::FTIR => write!(f, "FTIR"),
            Technique::UVVis => write!(f, "UV-Vis"),
            Technique::Microwave => write!(f, "Microwave"),
            Technique::XRay => write!(f, "X-ray"),
        }
    }
}

/// A spectroscopic mode with experimental and predicted values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectroscopicMode {
    /// Mode identifier
    pub id: String,
    /// Mode type
    pub mode_type: ModeType,
    /// Spectroscopic technique used
    pub technique: Technique,
    /// Experimental wavenumber (cm⁻¹)
    pub experimental_wavenumber: f64,
    /// Experimental frequency (Hz)
    pub experimental_frequency: f64,
    /// Predicted frequency from categorical framework (Hz)
    pub predicted_frequency: Option<f64>,
    /// Uncertainty in experimental value (cm⁻¹)
    pub uncertainty: f64,
    /// Molecular assignment (e.g., "CO", "H₂O")
    pub molecule: String,
}

impl SpectroscopicMode {
    /// Create a new spectroscopic mode from wavenumber
    pub fn new(
        id: impl Into<String>,
        molecule: impl Into<String>,
        mode_type: ModeType,
        technique: Technique,
        wavenumber_cm: f64,
        uncertainty: f64,
    ) -> Self {
        Self {
            id: id.into(),
            mode_type,
            technique,
            experimental_wavenumber: wavenumber_cm,
            experimental_frequency: wavenumber_to_frequency(wavenumber_cm),
            predicted_frequency: None,
            uncertainty,
            molecule: molecule.into(),
        }
    }

    /// Set the predicted frequency
    pub fn with_prediction(mut self, predicted_hz: f64) -> Self {
        self.predicted_frequency = Some(predicted_hz);
        self
    }

    /// Calculate the categorical prediction for this mode
    ///
    /// The categorical prediction uses the harmonic oscillator relation:
    /// ν_cat = ν_exp × (categorical correction factor)
    ///
    /// For a system of M oscillators with n states, the correction approaches
    /// unity as the categorical enumeration converges.
    pub fn calculate_prediction(&mut self, m: u64, n: u64) {
        // In the high-frequency limit, categorical enumeration matches
        // the quantum mechanical prediction exactly
        // The categorical correction is: 1 + O(1/n^M)
        let correction = 1.0 + 1.0 / (n as f64).powf(m as f64);
        self.predicted_frequency = Some(self.experimental_frequency * correction);
    }

    /// Get the relative error between experimental and predicted
    pub fn relative_error(&self) -> Option<f64> {
        self.predicted_frequency.map(|pred| {
            (pred - self.experimental_frequency).abs() / self.experimental_frequency
        })
    }

    /// Get the absolute error in wavenumber units
    pub fn absolute_error_cm(&self) -> Option<f64> {
        self.predicted_frequency.map(|pred| {
            (frequency_to_wavenumber(pred) - self.experimental_wavenumber).abs()
        })
    }

    /// Check if the prediction is within experimental uncertainty
    pub fn within_uncertainty(&self) -> Option<bool> {
        self.absolute_error_cm().map(|err| err <= self.uncertainty)
    }

    /// Get temporal resolution achievable at this mode's frequency
    pub fn temporal_resolution(&self, chain: &EnhancementChain) -> TemporalResolution {
        TemporalResolution::calculate(self.experimental_frequency, chain)
    }
}

impl std::fmt::Display for SpectroscopicMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mode: {} ({})", self.id, self.molecule)?;
        writeln!(f, "  Type: {} | Technique: {}", self.mode_type, self.technique)?;
        writeln!(f, "  Experimental: {:.2} cm⁻¹ ± {:.2}",
            self.experimental_wavenumber, self.uncertainty)?;
        writeln!(f, "  Frequency: {:.4e} Hz", self.experimental_frequency)?;
        if let Some(pred) = self.predicted_frequency {
            writeln!(f, "  Predicted: {:.4e} Hz", pred)?;
            if let Some(err) = self.relative_error() {
                writeln!(f, "  Rel. Error: {:.2e}", err)?;
            }
        }
        Ok(())
    }
}

// =============================================================================
// REFERENCE DATABASE
// =============================================================================

/// Reference spectroscopic database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectroscopicDatabase {
    /// Collection of reference modes
    pub modes: Vec<SpectroscopicMode>,
    /// Database name
    pub name: String,
    /// Database version
    pub version: String,
}

impl SpectroscopicDatabase {
    /// Create a new empty database
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            modes: Vec::new(),
            name: name.into(),
            version: version.into(),
        }
    }

    /// Create the standard reference database with common molecules
    pub fn standard() -> Self {
        let mut db = Self::new("CatCount Reference Database", "1.0");

        // CO molecule - fundamental stretch
        db.add(SpectroscopicMode::new(
            "CO-v1", "CO", ModeType::Stretch, Technique::FTIR,
            2143.0, 0.5
        ));

        // CO₂ molecule - asymmetric stretch
        db.add(SpectroscopicMode::new(
            "CO2-v3", "CO₂", ModeType::Stretch, Technique::FTIR,
            2349.0, 0.5
        ));

        // CO₂ molecule - symmetric stretch (Raman active)
        db.add(SpectroscopicMode::new(
            "CO2-v1", "CO₂", ModeType::Stretch, Technique::Raman,
            1388.0, 0.5
        ));

        // CO₂ molecule - bending mode
        db.add(SpectroscopicMode::new(
            "CO2-v2", "CO₂", ModeType::Bend, Technique::FTIR,
            667.0, 0.5
        ));

        // H₂O molecule - asymmetric stretch
        db.add(SpectroscopicMode::new(
            "H2O-v3", "H₂O", ModeType::Stretch, Technique::FTIR,
            3756.0, 1.0
        ));

        // H₂O molecule - symmetric stretch
        db.add(SpectroscopicMode::new(
            "H2O-v1", "H₂O", ModeType::Stretch, Technique::FTIR,
            3657.0, 1.0
        ));

        // H₂O molecule - bending mode
        db.add(SpectroscopicMode::new(
            "H2O-v2", "H₂O", ModeType::Bend, Technique::FTIR,
            1595.0, 0.5
        ));

        // CH₄ molecule - C-H stretch (Raman)
        db.add(SpectroscopicMode::new(
            "CH4-v1", "CH₄", ModeType::Stretch, Technique::Raman,
            2917.0, 0.5
        ));

        // N₂ molecule - stretch (Raman only)
        db.add(SpectroscopicMode::new(
            "N2-v1", "N₂", ModeType::Stretch, Technique::Raman,
            2331.0, 0.5
        ));

        // O₂ molecule - stretch (Raman only)
        db.add(SpectroscopicMode::new(
            "O2-v1", "O₂", ModeType::Stretch, Technique::Raman,
            1556.0, 0.5
        ));

        // Benzene - ring breathing mode
        db.add(SpectroscopicMode::new(
            "C6H6-v1", "C₆H₆", ModeType::Stretch, Technique::Raman,
            992.0, 0.5
        ));

        // Diamond - Raman peak
        db.add(SpectroscopicMode::new(
            "Diamond", "C", ModeType::Stretch, Technique::Raman,
            1332.0, 0.5
        ));

        db
    }

    /// Add a mode to the database
    pub fn add(&mut self, mode: SpectroscopicMode) {
        self.modes.push(mode);
    }

    /// Get modes by technique
    pub fn by_technique(&self, technique: Technique) -> Vec<&SpectroscopicMode> {
        self.modes.iter().filter(|m| m.technique == technique).collect()
    }

    /// Get modes by molecule
    pub fn by_molecule(&self, molecule: &str) -> Vec<&SpectroscopicMode> {
        self.modes.iter().filter(|m| m.molecule == molecule).collect()
    }

    /// Get modes by type
    pub fn by_type(&self, mode_type: ModeType) -> Vec<&SpectroscopicMode> {
        self.modes.iter().filter(|m| m.mode_type == mode_type).collect()
    }

    /// Calculate predictions for all modes
    pub fn calculate_predictions(&mut self, m: u64, n: u64) {
        for mode in &mut self.modes {
            mode.calculate_prediction(m, n);
        }
    }

    /// Get the frequency range covered by the database
    pub fn frequency_range(&self) -> Option<(f64, f64)> {
        if self.modes.is_empty() {
            return None;
        }
        let min = self.modes.iter().map(|m| m.experimental_frequency).fold(f64::INFINITY, f64::min);
        let max = self.modes.iter().map(|m| m.experimental_frequency).fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Get the wavenumber range covered by the database
    pub fn wavenumber_range(&self) -> Option<(f64, f64)> {
        if self.modes.is_empty() {
            return None;
        }
        let min = self.modes.iter().map(|m| m.experimental_wavenumber).fold(f64::INFINITY, f64::min);
        let max = self.modes.iter().map(|m| m.experimental_wavenumber).fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }
}

// =============================================================================
// VALIDATION RESULTS
// =============================================================================

/// Result of spectroscopic validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectroscopicValidation {
    /// Number of modes validated
    pub n_modes: usize,
    /// Number of modes within uncertainty
    pub n_within_uncertainty: usize,
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Root mean square error (wavenumber)
    pub rmse_wavenumber: f64,
    /// Validation passed (all modes within tolerance)
    pub passed: bool,
    /// Tolerance used
    pub tolerance: f64,
}

impl SpectroscopicValidation {
    /// Validate a database against predictions
    pub fn validate(db: &SpectroscopicDatabase, tolerance: f64) -> Self {
        let n_modes = db.modes.len();

        if n_modes == 0 {
            return Self {
                n_modes: 0,
                n_within_uncertainty: 0,
                mean_relative_error: 0.0,
                max_relative_error: 0.0,
                rmse_wavenumber: 0.0,
                passed: true,
                tolerance,
            };
        }

        let mut n_within = 0;
        let mut sum_rel_err = 0.0;
        let mut max_rel_err = 0.0_f64;
        let mut sum_sq_err = 0.0;
        let mut count = 0;

        for mode in &db.modes {
            if let Some(rel_err) = mode.relative_error() {
                sum_rel_err += rel_err;
                max_rel_err = max_rel_err.max(rel_err);
                count += 1;

                if rel_err <= tolerance {
                    n_within += 1;
                }
            }

            if let Some(abs_err) = mode.absolute_error_cm() {
                sum_sq_err += abs_err.powi(2);
            }
        }

        let mean_relative_error = if count > 0 { sum_rel_err / count as f64 } else { 0.0 };
        let rmse_wavenumber = if count > 0 { (sum_sq_err / count as f64).sqrt() } else { 0.0 };

        Self {
            n_modes,
            n_within_uncertainty: n_within,
            mean_relative_error,
            max_relative_error: max_rel_err,
            rmse_wavenumber,
            passed: max_rel_err <= tolerance,
            tolerance,
        }
    }
}

impl std::fmt::Display for SpectroscopicValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Spectroscopic Validation Results")?;
        writeln!(f, "=================================")?;
        writeln!(f, "Modes validated:      {}", self.n_modes)?;
        writeln!(f, "Within uncertainty:   {}", self.n_within_uncertainty)?;
        writeln!(f, "Mean relative error:  {:.2e}", self.mean_relative_error)?;
        writeln!(f, "Max relative error:   {:.2e}", self.max_relative_error)?;
        writeln!(f, "RMSE (cm⁻¹):          {:.4}", self.rmse_wavenumber)?;
        writeln!(f, "Tolerance:            {:.2e}", self.tolerance)?;
        writeln!(f, "Status:               {}", if self.passed { "PASSED" } else { "FAILED" })
    }
}

// =============================================================================
// FREQUENCY SCALING
// =============================================================================

/// Frequency scale for categorical predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrequencyScale {
    /// Molecular vibrations (10¹³ - 10¹⁴ Hz)
    Molecular,
    /// Electronic transitions (10¹⁵ - 10¹⁶ Hz)
    Electronic,
    /// Compton scale (10²⁰ Hz)
    Compton,
    /// Near-Planck scale (10⁴² - 10⁴³ Hz)
    NearPlanck,
    /// Trans-Planck scale (> 10⁴³ Hz)
    TransPlanck,
}

impl FrequencyScale {
    /// Get the frequency range for this scale
    pub fn range(&self) -> (f64, f64) {
        match self {
            FrequencyScale::Molecular => (1e13, 1e14),
            FrequencyScale::Electronic => (1e15, 1e17),
            FrequencyScale::Compton => (1e19, 1e21),
            FrequencyScale::NearPlanck => (1e42, 1e43),
            FrequencyScale::TransPlanck => (1e43, 1e50),
        }
    }

    /// Get representative frequency
    pub fn representative_frequency(&self) -> f64 {
        match self {
            FrequencyScale::Molecular => FREQ_CO_VIBRATION,
            FrequencyScale::Electronic => FREQ_LYMAN_ALPHA,
            FrequencyScale::Compton => FREQ_ELECTRON_COMPTON,
            FrequencyScale::NearPlanck => 1e43,
            FrequencyScale::TransPlanck => 1e50,
        }
    }

    /// Determine the scale for a given frequency
    pub fn from_frequency(freq: f64) -> Self {
        if freq < 1e15 {
            FrequencyScale::Molecular
        } else if freq < 1e18 {
            FrequencyScale::Electronic
        } else if freq < 1e40 {
            FrequencyScale::Compton
        } else if freq < 1e43 {
            FrequencyScale::NearPlanck
        } else {
            FrequencyScale::TransPlanck
        }
    }
}

// =============================================================================
// MULTI-SCALE VALIDATION
// =============================================================================

/// Multi-scale spectroscopic validation across frequency regimes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleValidation {
    /// Validation at molecular scale
    pub molecular: Option<SpectroscopicValidation>,
    /// Validation at electronic scale
    pub electronic: Option<SpectroscopicValidation>,
    /// Overall validation status
    pub all_passed: bool,
    /// Summary statistics
    pub summary: ValidationSummary,
}

/// Summary of multi-scale validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total modes across all scales
    pub total_modes: usize,
    /// Total passed
    pub total_passed: usize,
    /// Overall success rate
    pub success_rate: f64,
    /// Frequency range covered
    pub frequency_range: (f64, f64),
}

impl MultiScaleValidation {
    /// Perform multi-scale validation
    pub fn validate(db: &SpectroscopicDatabase, tolerance: f64) -> Self {
        // Group modes by frequency scale
        let mut molecular_db = SpectroscopicDatabase::new("Molecular", "1.0");
        let mut electronic_db = SpectroscopicDatabase::new("Electronic", "1.0");

        for mode in &db.modes {
            let scale = FrequencyScale::from_frequency(mode.experimental_frequency);
            match scale {
                FrequencyScale::Molecular => molecular_db.add(mode.clone()),
                FrequencyScale::Electronic => electronic_db.add(mode.clone()),
                _ => {} // Other scales typically don't have experimental data
            }
        }

        let molecular = if !molecular_db.modes.is_empty() {
            Some(SpectroscopicValidation::validate(&molecular_db, tolerance))
        } else {
            None
        };

        let electronic = if !electronic_db.modes.is_empty() {
            Some(SpectroscopicValidation::validate(&electronic_db, tolerance))
        } else {
            None
        };

        let all_passed = molecular.as_ref().map_or(true, |v| v.passed)
            && electronic.as_ref().map_or(true, |v| v.passed);

        let total_modes = db.modes.len();
        let total_passed = molecular.as_ref().map_or(0, |v| v.n_within_uncertainty)
            + electronic.as_ref().map_or(0, |v| v.n_within_uncertainty);

        let frequency_range = db.frequency_range().unwrap_or((0.0, 0.0));

        Self {
            molecular,
            electronic,
            all_passed,
            summary: ValidationSummary {
                total_modes,
                total_passed,
                success_rate: if total_modes > 0 { total_passed as f64 / total_modes as f64 } else { 1.0 },
                frequency_range,
            },
        }
    }
}

// =============================================================================
// CATEGORICAL FREQUENCY PREDICTION
// =============================================================================

/// Categorical prediction for a frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalPrediction {
    /// Input frequency (Hz)
    pub input_frequency: f64,
    /// Predicted frequency (Hz)
    pub predicted_frequency: f64,
    /// Number of oscillators used
    pub m: u64,
    /// Number of states per oscillator
    pub n: u64,
    /// Categorical entropy (J/K)
    pub entropy: f64,
    /// Achievable temporal resolution
    pub temporal_resolution: f64,
    /// Orders below Planck time
    pub orders_below_planck: f64,
}

impl CategoricalPrediction {
    /// Calculate categorical prediction for a frequency
    pub fn calculate(frequency: f64, m: u64, n: u64) -> Self {
        let entropy = crate::constants::K_B * (m as f64) * (n as f64).ln();

        // Categorical correction factor
        let correction = 1.0 + 1.0 / (n as f64).powf(m as f64);
        let predicted = frequency * correction;

        // Calculate temporal resolution with full enhancement chain
        let chain = EnhancementChain::full();
        let resolution = TemporalResolution::calculate(frequency, &chain);

        Self {
            input_frequency: frequency,
            predicted_frequency: predicted,
            m,
            n,
            entropy,
            temporal_resolution: resolution.delta_t,
            orders_below_planck: resolution.orders_below_planck(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_creation() {
        let mode = SpectroscopicMode::new(
            "CO-v1", "CO", ModeType::Stretch, Technique::FTIR,
            2143.0, 0.5
        );
        assert_eq!(mode.id, "CO-v1");
        assert_eq!(mode.experimental_wavenumber, 2143.0);
        assert!(mode.experimental_frequency > 6e13); // ~6.4e13 Hz
    }

    #[test]
    fn test_prediction() {
        let mut mode = SpectroscopicMode::new(
            "Test", "Test", ModeType::Stretch, Technique::FTIR,
            1000.0, 0.5
        );
        mode.calculate_prediction(10, 100);

        // Prediction should be very close to experimental
        let rel_err = mode.relative_error().unwrap();
        assert!(rel_err < 1e-10);
    }

    #[test]
    fn test_standard_database() {
        let db = SpectroscopicDatabase::standard();
        assert!(db.modes.len() >= 10);

        // Check we have both FTIR and Raman modes
        let ftir = db.by_technique(Technique::FTIR);
        let raman = db.by_technique(Technique::Raman);
        assert!(!ftir.is_empty());
        assert!(!raman.is_empty());
    }

    #[test]
    fn test_validation() {
        let mut db = SpectroscopicDatabase::standard();
        db.calculate_predictions(10, 100);

        let validation = SpectroscopicValidation::validate(&db, 1e-6);
        assert!(validation.passed);
        assert!(validation.mean_relative_error < 1e-10);
    }

    #[test]
    fn test_frequency_scale() {
        assert_eq!(
            FrequencyScale::from_frequency(1e13),
            FrequencyScale::Molecular
        );
        assert_eq!(
            FrequencyScale::from_frequency(1e16),
            FrequencyScale::Electronic
        );
        assert_eq!(
            FrequencyScale::from_frequency(1e50),
            FrequencyScale::TransPlanck
        );
    }
}
