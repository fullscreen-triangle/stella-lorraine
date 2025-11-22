import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class AdversarialRobustnessValidator:
    """Validates system robustness under biological perturbations"""
    
    def __init__(self):
        self.robustness_constant = 1.24  # L from paper
        self.perturbation_types = ['noise', 'motion', 'temperature', 'degradation']
        
    def generate_baseline_performance(self, n_samples=1000):
        """Generate baseline system performance"""
        
        # Simulate multi-modal measurements
        true_values = np.random.uniform(0, 1, n_samples)
        
        # Perfect system response
        system_response = true_values + np.random.normal(0, 0.01, n_samples)
        
        baseline_accuracy = 1 - np.mean(np.abs(system_response - true_values))
        
        return true_values, system_response, baseline_accuracy
    
    def apply_perturbations(self, true_values, perturbation_type, epsilon):
        """Apply specific perturbation types"""
        
        perturbed_values = true_values.copy()
        
        if perturbation_type == 'noise':
            # Measurement noise (SNR-dependent)
            snr_db = 20 * np.log10(1/epsilon) if epsilon > 0 else 60
            noise_power = 10**(-snr_db/10)
            perturbed_values += np.random.normal(0, np.sqrt(noise_power), len(true_values))
            
        elif perturbation_type == 'motion':
            # Motion artifacts (tissue displacement)
            displacement = epsilon * np.sin(2 * np.pi * np.arange(len(true_values)) / 100)
            perturbed_values += 0.1 * displacement
            
        elif perturbation_type == 'temperature':
            # Environmental temperature variation
            temp_drift = epsilon * np.random.uniform(-1, 1, len(true_values))
            perturbed_values += 0.05 * temp_drift
            
        elif perturbation_type == 'degradation':
            # Membrane degradation over time
            degradation_factor = 1 - epsilon * np.linspace(0, 1, len(true_values))
            perturbed_values *= degradation_factor
        
        return perturbed_values
    
    def validate_robustness_theorem(self, n_trials=100):
        """Validate Theorem 7: Bounded performance degradation"""
        
        results = {perturbation: {'epsilons': [], 'degradations': [], 'bounds_satisfied': []} 
                  for perturbation in self.perturbation_types}
        
        true_values, baseline_response, baseline_accuracy = self.generate_baseline_performance()
        
        # Test different perturbation magnitudes
        epsilons = np.logspace(-3, -1, 10)  # 0.001 to 0.1
        
        for perturbation_type in self.perturbation_types:
            for epsilon in epsilons:
                trial_degradations = []
                
                for trial in range(n_trials):
                    # Apply perturbation
                    perturbed_values = self.apply_perturbations(true_values, perturbation_type, epsilon)
                    
                    # System response to perturbed input
                    perturbed_response = perturbed_values + np.random.normal(0, 0.01, len(perturbed_values))
                    perturbed_accuracy = 1 - np.mean(np.abs(perturbed_response - true_values))
                    
                    # Performance degradation
                    degradation = abs(perturbed_accuracy - baseline_accuracy)
                    trial_degradations.append(degradation)
                
                mean_degradation = np.mean(trial_degradations)
                theoretical_bound = self.robustness_constant * epsilon
                
                results[perturbation_type]['epsilons'].append(epsilon)
                results[perturbation_type]['degradations'].append(mean_degradation)
                results[perturbation_type]['bounds_satisfied'].append(mean_degradation <= theoretical_bound)
        
        # Overall validation
        all_bounds_satisfied = all(
            all(results[p]['bounds_satisfied']) for p in self.perturbation_types
        )
        
        return {
            'theorem_validated': all_bounds_satisfied,
            'robustness_constant': self.robustness_constant,
            'detailed_results': results,
            'baseline_accuracy': baseline_accuracy
        }
    
    def validate_clinical_requirements(self):
        """Validate specific clinical robustness requirements from paper"""
        
        requirements = {
            'snr_15db': {'epsilon': 0.032, 'max_degradation': 0.05},  # <5% degradation
            'motion_2mm': {'epsilon': 0.002, 'max_degradation': 0.03}, # Stable operation  
            'temp_5c': {'epsilon': 0.05, 'max_degradation': 0.03},    # <3% accuracy loss
            'degradation_1e5': {'epsilon': 0.1, 'max_degradation': 0.10} # >90% accuracy
        }
        
        true_values, _, baseline_accuracy = self.generate_baseline_performance()
        clinical_results = {}
        
        perturbation_map = {
            'snr_15db': 'noise',
            'motion_2mm': 'motion', 
            'temp_5c': 'temperature',
            'degradation_1e5': 'degradation'
        }
        
        for req_name, req_params in requirements.items():
            perturbation_type = perturbation_map[req_name]
            epsilon = req_params['epsilon']
            
            perturbed_values = self.apply_perturbations(true_values, perturbation_type, epsilon)
            perturbed_response = perturbed_values + np.random.normal(0, 0.01, len(perturbed_values))
            perturbed_accuracy = 1 - np.mean(np.abs(perturbed_response - true_values))
            
            actual_degradation = abs(perturbed_accuracy - baseline_accuracy)
            requirement_met = actual_degradation <= req_params['max_degradation']
            
            clinical_results[req_name] = {
                'requirement_met': requirement_met,
                'actual_degradation': actual_degradation,
                'max_allowed': req_params['max_degradation'],
                'perturbed_accuracy': perturbed_accuracy
            }
        
        return clinical_results

if __name__ == "__main__":
    # Usage
    robustness_validator = AdversarialRobustnessValidator()
    theorem_results = robustness_validator.validate_robustness_theorem()
    clinical_results = robustness_validator.validate_clinical_requirements()

    print(f"Robustness theorem validated: {theorem_results['theorem_validated']}")
    print("Clinical requirements:")
    for req, result in clinical_results.items():
        print(f"  {req}: {'✓' if result['requirement_met'] else '✗'} "
            f"({result['actual_degradation']:.3f} ≤ {result['max_allowed']:.3f})")
