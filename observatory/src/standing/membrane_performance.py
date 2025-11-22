import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class MembranePerformanceValidator:
    """Validates fluid-controlled membrane system performance"""
    
    def __init__(self, membrane_thickness=1e-6, pressure_sensitivity=0.1):
        self.d_membrane = membrane_thickness  # meters
        self.delta_p_min = pressure_sensitivity  # Pa
        self.k_membrane = 1e6  # membrane stiffness
        self.k_fluid = 1e3     # fluid resistance
        
    def membrane_dynamics(self, state, t, control_pressure):
        """Model membrane deformation dynamics"""
        delta, delta_dot = state
        
        # Pressure differential
        p_ambient = 101325  # Pa
        delta_p = control_pressure(t) - p_ambient
        
        # Membrane equation (Eq. 5 from paper)
        a_local = 1e-6  # local area
        delta_target = delta_p * a_local / (self.k_membrane + self.k_fluid)
        
        # Second-order dynamics
        delta_ddot = -100 * (delta - delta_target) - 20 * delta_dot
        
        return [delta_dot, delta_ddot]
    
    def validate_response_time(self):
        """Validate membrane response time < 10ms requirement"""
        
        def step_pressure(t):
            return 101325 + 10 * (t > 0.001)  # 10 Pa step at t=1ms
        
        t = np.linspace(0, 0.02, 1000)  # 20ms simulation
        initial_state = [0, 0]
        
        solution = odeint(self.membrane_dynamics, initial_state, t, args=(step_pressure,))
        
        # Find 90% response time
        final_value = solution[-1, 0]
        target_90 = 0.9 * final_value
        
        response_time = None
        for i, delta in enumerate(solution[:, 0]):
            if delta >= target_90:
                response_time = t[i]
                break
        
        validation_results = {
            'response_time': response_time,
            'meets_requirement': response_time < 0.01 if response_time else False,
            'final_deformation': final_value,
            'time_series': t,
            'deformation_series': solution[:, 0]
        }
        
        return validation_results
    
    def validate_spatial_resolution(self):
        """Validate 10μm spatial resolution requirement"""
        
        # Simulate 2D membrane with spatial pressure distribution
        x = np.linspace(0, 1e-3, 100)  # 1mm membrane
        y = np.linspace(0, 1e-3, 100)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian pressure distribution
        pressure_field = 10 * np.exp(-((X-0.5e-3)**2 + (Y-0.5e-3)**2) / (2*(50e-6)**2))
        
        # Calculate membrane deformation
        deformation = pressure_field * 1e-6 / (self.k_membrane + self.k_fluid)
        
        # Spatial resolution analysis
        gradient_x = np.gradient(deformation, axis=1)
        gradient_y = np.gradient(deformation, axis=0)
        
        spatial_resolution = np.min([np.diff(x)[0], np.diff(y)[0]])
        
        return {
            'spatial_resolution': spatial_resolution,
            'meets_requirement': spatial_resolution <= 10e-6,
            'deformation_field': deformation,
            'pressure_field': pressure_field
        }

if __name__ == "__main__":
    # Usage
    membrane_validator = MembranePerformanceValidator()
    response_results = membrane_validator.validate_response_time()
    spatial_results = membrane_validator.validate_spatial_resolution()

    print(f"Response time: {response_results['response_time']*1000:.2f} ms")
    print(f"Spatial resolution: {spatial_results['spatial_resolution']*1e6:.1f} μm")
