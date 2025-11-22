"""
Cross Validator - Dual-Validation Engine
==========================================

Compares oscillatory and visual analysis results to provide dual validation.
Agreement scores quantify how well both pathways agree, with disagreements
indicating either measurement artifacts or novel phenomena.

Purpose:
--------
The cross-validator ensures measurement reliability by:
1. Comparing predictions from oscillatory and visual pathways
2. Calculating agreement scores (0-1)
3. Identifying disagreements (potential discoveries)
4. Generating recommendations based on validation status

Agreement Score Interpretation:
-------------------------------
> 0.95: VALIDATED (Very High Confidence) - Excellent agreement
0.80-0.95: VALIDATED (High Confidence) - Good agreement
0.60-0.80: PARTIAL VALIDATION - Investigate disagreements
< 0.60: VALIDATION FAILED - Major issues or discovery
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """
    Result of dual validation between oscillatory and visual pathways
    
    Attributes:
        agreement_score: Overall agreement (0-1)
        status: Validation status string
        detailed_agreements: Per-property agreement scores
        disagreements: List of properties where methods disagree
        recommendations: What to do based on validation
        visual_only_patterns: Patterns found only by visual pathway
        oscillatory_only_patterns: Patterns found only by oscillatory pathway
    """
    agreement_score: float
    status: str
    detailed_agreements: Dict[str, float]
    disagreements: List[Dict[str, Any]]
    recommendations: Dict[str, Any]
    visual_only_patterns: List[str]
    oscillatory_only_patterns: List[str]


class DualValidationEngine:
    """
    Cross-validates oscillatory and visual analysis pathways
    
    Provides confidence scores and identifies novel phenomena.
    """
    
    # Agreement thresholds
    VERY_HIGH_THRESHOLD = 0.95
    HIGH_THRESHOLD = 0.80
    PARTIAL_THRESHOLD = 0.60
    
    def __init__(self, tolerance: float = 0.15):
        """
        Initialize validation engine
        
        Args:
            tolerance: Relative tolerance for numerical comparisons
        """
        self.tolerance = tolerance
        
    def validate(self,
                oscillatory_result: Dict[str, Any],
                visual_result: Dict[str, Any]) -> ValidationResult:
        """
        Perform complete dual validation
        
        Args:
            oscillatory_result: Results from oscillatory analysis pathway
            visual_result: Results from visual analysis pathway
            
        Returns:
            ValidationResult with complete validation analysis
        """
        # Extract predictions from both pathways
        osc_predictions = oscillatory_result.get('predictions', {})
        vis_predictions = visual_result.get('predictions', {})
        
        # Calculate agreement for each property
        detailed_agreements = {}
        all_properties = set(osc_predictions.keys()) | set(vis_predictions.keys())
        
        for prop in all_properties:
            if prop in osc_predictions and prop in vis_predictions:
                agreement = self._calculate_agreement(
                    osc_predictions[prop],
                    vis_predictions[prop]
                )
                detailed_agreements[prop] = agreement
                
        # Overall agreement score
        if detailed_agreements:
            overall_agreement = np.mean(list(detailed_agreements.values()))
        else:
            overall_agreement = 0.0
            
        # Identify disagreements
        disagreements = self._identify_disagreements(
            osc_predictions, vis_predictions, detailed_agreements
        )
        
        # Find pathway-specific patterns
        visual_only = self._find_visual_only_patterns(visual_result, oscillatory_result)
        osc_only = self._find_oscillatory_only_patterns(oscillatory_result, visual_result)
        
        # Determine status
        status = self._determine_status(overall_agreement)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_agreement, disagreements, visual_only, osc_only
        )
        
        return ValidationResult(
            agreement_score=overall_agreement,
            status=status,
            detailed_agreements=detailed_agreements,
            disagreements=disagreements,
            recommendations=recommendations,
            visual_only_patterns=visual_only,
            oscillatory_only_patterns=osc_only
        )
        
    def _calculate_agreement(self, osc_value: Any, vis_value: Any) -> float:
        """
        Calculate agreement score between two values
        
        Returns:
            Agreement score (0-1)
        """
        # Handle different types
        if isinstance(osc_value, (int, float, np.number)):
            # Scalar values
            if osc_value == 0 and vis_value == 0:
                return 1.0
                
            # Relative difference
            denominator = abs(osc_value) + abs(vis_value)
            if denominator == 0:
                return 1.0
                
            relative_diff = abs(osc_value - vis_value) / denominator
            agreement = 1.0 - min(relative_diff, 1.0)
            
            return agreement
            
        elif isinstance(osc_value, np.ndarray):
            # Array values
            if len(osc_value) != len(vis_value):
                # Different dimensions - low agreement
                return 0.5
                
            # Normalized difference
            osc_norm = np.linalg.norm(osc_value)
            vis_norm = np.linalg.norm(vis_value)
            
            if osc_norm == 0 and vis_norm == 0:
                return 1.0
                
            diff_norm = np.linalg.norm(osc_value - vis_value)
            agreement = 1.0 - diff_norm / (osc_norm + vis_norm)
            
            return max(0.0, min(agreement, 1.0))
            
        elif isinstance(osc_value, list):
            # List values - compare element by element
            if len(osc_value) != len(vis_value):
                return 0.5
                
            agreements = []
            for o, v in zip(osc_value, vis_value):
                agreements.append(self._calculate_agreement(o, v))
                
            return np.mean(agreements) if agreements else 0.0
            
        elif isinstance(osc_value, str):
            # String values - exact match
            return 1.0 if osc_value == vis_value else 0.0
            
        else:
            # Unknown type - try equality
            return 1.0 if osc_value == vis_value else 0.5
            
    def _identify_disagreements(self,
                               osc_predictions: Dict,
                               vis_predictions: Dict,
                               agreements: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Identify properties where pathways disagree
        """
        disagreements = []
        
        for prop, score in agreements.items():
            if score < self.HIGH_THRESHOLD:  # Significant disagreement
                disagreement = {
                    'property': prop,
                    'oscillatory_value': osc_predictions.get(prop),
                    'visual_value': vis_predictions.get(prop),
                    'agreement_score': score,
                    'severity': self._classify_disagreement(score)
                }
                disagreements.append(disagreement)
                
        return disagreements
        
    def _classify_disagreement(self, score: float) -> str:
        """Classify disagreement severity"""
        if score < 0.3:
            return 'CRITICAL'
        elif score < 0.5:
            return 'MAJOR'
        elif score < 0.7:
            return 'MODERATE'
        else:
            return 'MINOR'
            
    def _find_visual_only_patterns(self,
                                   visual_result: Dict,
                                   oscillatory_result: Dict) -> List[str]:
        """
        Find patterns detected only by visual pathway
        
        These are often spatial/temporal patterns invisible to FFT.
        """
        visual_only = []
        
        # Check for spatial patterns
        if 'visual_features' in visual_result:
            features = visual_result['visual_features']
            
            # Swirl patterns (coupled modes)
            if features.get('swirl_detected', False):
                visual_only.append('swirl_pattern_coupled_mode')
                
            # Fractal patterns (turbulence cascades)
            if features.get('fractal_dimension', 0) > 1.5:
                visual_only.append('fractal_turbulence_cascade')
                
            # Asymmetric patterns (localized defects)
            if features.get('asymmetry', 0) > 0.3:
                visual_only.append('asymmetric_spatial_defect')
                
            # Multi-scale structures
            if features.get('n_scales', 0) > 3:
                visual_only.append('multi_scale_structure')
                
        return visual_only
        
    def _find_oscillatory_only_patterns(self,
                                       oscillatory_result: Dict,
                                       visual_result: Dict) -> List[str]:
        """
        Find patterns detected only by oscillatory pathway
        
        These are often weak harmonics masked in visual representation.
        """
        osc_only = []
        
        if 'harmonics' in oscillatory_result:
            harmonics = oscillatory_result['harmonics']
            
            # Very weak harmonics (< 5% of dominant)
            if 'amplitudes' in harmonics:
                amps = np.array(harmonics['amplitudes'])
                if len(amps) > 1:
                    weak_harmonics = np.sum(amps[1:] / amps[0] < 0.05)
                    if weak_harmonics > 5:
                        osc_only.append('weak_harmonic_structure')
                        
            # Very high Q-factors (sharp resonances)
            if 'Q_factors' in harmonics:
                Q = np.array(harmonics['Q_factors'])
                if np.max(Q) > 1000:
                    osc_only.append('ultra_sharp_resonance')
                    
        return osc_only
        
    def _determine_status(self, agreement_score: float) -> str:
        """Determine validation status from agreement score"""
        if agreement_score >= self.VERY_HIGH_THRESHOLD:
            return 'VALIDATED (Very High Confidence)'
        elif agreement_score >= self.HIGH_THRESHOLD:
            return 'VALIDATED (High Confidence)'
        elif agreement_score >= self.PARTIAL_THRESHOLD:
            return 'PARTIAL VALIDATION'
        else:
            return 'VALIDATION FAILED'
            
    def _generate_recommendations(self,
                                 agreement_score: float,
                                 disagreements: List[Dict],
                                 visual_only: List[str],
                                 osc_only: List[str]) -> Dict[str, Any]:
        """
        Generate recommendations based on validation results
        """
        recommendations = {
            'overall_confidence': self._score_to_confidence(agreement_score),
            'proceed_with_analysis': agreement_score >= self.PARTIAL_THRESHOLD,
            'actions': []
        }
        
        # High agreement - all good
        if agreement_score >= self.VERY_HIGH_THRESHOLD:
            recommendations['actions'].append({
                'priority': 'LOW',
                'action': 'No action required',
                'reason': 'Excellent agreement between pathways'
            })
            
        # Good agreement with minor issues
        elif agreement_score >= self.HIGH_THRESHOLD:
            if visual_only:
                recommendations['actions'].append({
                    'priority': 'MEDIUM',
                    'action': 'Investigate visual-only patterns',
                    'reason': f'CNN detected {len(visual_only)} patterns not in FFT',
                    'patterns': visual_only
                })
                
        # Partial validation - needs investigation
        elif agreement_score >= self.PARTIAL_THRESHOLD:
            recommendations['actions'].append({
                'priority': 'HIGH',
                'action': 'Investigate disagreements',
                'reason': 'Significant discrepancies between pathways',
                'details': disagreements
            })
            
            # Check if this might be a discovery
            if len(visual_only) > 0:
                recommendations['actions'].append({
                    'priority': 'HIGH',
                    'action': 'Potential novel phenomenon detected',
                    'reason': 'Visual pathway found patterns invisible to oscillatory analysis',
                    'discovery_candidate': True
                })
                
        # Validation failed - critical issues
        else:
            recommendations['actions'].append({
                'priority': 'CRITICAL',
                'action': 'Check for measurement error',
                'reason': 'Major disagreement suggests measurement artifact'
            })
            
            recommendations['actions'].append({
                'priority': 'CRITICAL',
                'action': 'Consider novel physical phenomenon',
                'reason': 'If measurement is valid, this may be a discovery'
            })
            
        return recommendations
        
    def _score_to_confidence(self, score: float) -> str:
        """Convert numerical score to confidence level"""
        if score >= 0.95:
            return 'EXTREMELY HIGH'
        elif score >= 0.90:
            return 'VERY HIGH'
        elif score >= 0.80:
            return 'HIGH'
        elif score >= 0.70:
            return 'MODERATE'
        elif score >= 0.60:
            return 'LOW'
        else:
            return 'VERY LOW'
            
    def compare_frequencies(self,
                          osc_frequencies: np.ndarray,
                          vis_frequencies: np.ndarray,
                          tolerance_hz: float = 0.5) -> Dict[str, Any]:
        """
        Compare frequency content from both pathways
        
        Args:
            osc_frequencies: Frequencies from oscillatory analysis
            vis_frequencies: Frequencies from visual analysis
            tolerance_hz: Frequency tolerance (Hz)
            
        Returns:
            Comparison results
        """
        matches = []
        osc_only = []
        vis_only = []
        
        # Find matches
        for osc_f in osc_frequencies:
            match_found = False
            for vis_f in vis_frequencies:
                if abs(osc_f - vis_f) < tolerance_hz:
                    matches.append((osc_f, vis_f, abs(osc_f - vis_f)))
                    match_found = True
                    break
                    
            if not match_found:
                osc_only.append(osc_f)
                
        # Find visual-only frequencies
        for vis_f in vis_frequencies:
            match_found = False
            for osc_f in osc_frequencies:
                if abs(osc_f - vis_f) < tolerance_hz:
                    match_found = True
                    break
                    
            if not match_found:
                vis_only.append(vis_f)
                
        # Calculate agreement
        n_total = len(osc_frequencies) + len(vis_frequencies)
        n_matches = len(matches)
        
        if n_total > 0:
            agreement = 2 * n_matches / n_total  # Normalized to [0, 1]
        else:
            agreement = 0.0
            
        return {
            'agreement': agreement,
            'n_matches': n_matches,
            'matches': matches,
            'oscillatory_only': osc_only,
            'visual_only': vis_only
        }
        
    def __repr__(self) -> str:
        return f"DualValidationEngine(tolerance={self.tolerance})"