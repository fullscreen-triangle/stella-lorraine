"""
Observatory - Advanced Precision Measurement & Categorical Alignment Framework

This module ensures that all functions used in experiments are imported and instantiated
with correct inputs and outputs. It provides the main interface for accessing all
observatory components and running complete experiments.
"""

# Core framework version
__version__ = "1.0.0"

# Import all major components for easy access

# Signal Processing Components
try:
    from .signal import (
        create_mimo_signal_amplification_system,
        create_precise_clock_api_system,
        create_satellite_temporal_gps_system,
        create_signal_fusion_system,
        create_signal_latency_analyzer,
        create_temporal_information_system,
        # Enums and classes
        FrequencyBand,
        ClockType,
        VirtualInfrastructureLayer,
        FusionAlgorithm,
        LatencyComponent,
        TemporalPrecision,
        # Main classes
        MIMOSignalAmplificationEngine,
        PreciseClockAPIManager,
        SatelliteTemporalGPSEngine,
        SignalFusionEngine,
        SignalLatencyAnalyzer,
        TemporalDatabaseEngine
    )
    SIGNAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Signal processing components not available: {e}")
    SIGNAL_AVAILABLE = False

# Precision Validation Components
try:
    from .precision import (
        # Main validation classes
        StrategicDisagreementValidator,
        PrecisionStatistics,
        ValidationFramework,
        # Enums and types
        ValidationMethod,
        MeasurementSystem,
        DisagreementType,
        StatisticalTest,
        HypothesisType,
        # Factory functions
        create_strategic_disagreement_validator,
        create_precision_statistics_suite,
        create_validation_framework
    )
    PRECISION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Precision validation components not available: {e}")
    PRECISION_AVAILABLE = False

# S-Entropy Oscillatory Framework
try:
    from .oscillatory import (
        # Core S-Entropy classes
        SemanticDistanceAmplifier,
        NavigationAlgorithm,
        HierarchicalOscillatorySystem,
        TranscendentObserver,
        SemanticDistanceCalculator,
        PrecisionAmplifier,
        # Compression and sequencing
        CompressionResistanceCalculator,
        AmbiguousInformationSegment,
        MetaInformationExtractor,
        SequentialEncoder,
        PositionalContextEncoder,
        DirectionalTransformer,
        # Factory functions
        create_s_entropy_system,
        create_hierarchical_navigation,
        create_semantic_amplifier
    )
    OSCILLATORY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Oscillatory framework components not available: {e}")
    OSCILLATORY_AVAILABLE = False

# Recursive Enhancement Components
try:
    from .recursion import (
        # Dual function and processing
        DualFunctionMolecule,
        MolecularDualFunctionNetwork,
        RecursivePrecisionEnhancer,
        AtmosphericMolecularNetwork,
        QuantumTimeVirtualProcessor,
        VirtualProcessorAccelerationSystem,
        # Enums and types
        MoleculeType,
        PrecisionMetric,
        AtmosphericLayer,
        ProcessorType,
        # Factory functions
        create_dual_function_network,
        create_precision_enhancer,
        create_atmospheric_network,
        create_virtual_processor_system
    )
    RECURSION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Recursive enhancement components not available: {e}")
    RECURSION_AVAILABLE = False

# Wave Simulation Framework
try:
    from .simulation import (
        # Core simulation components
        InfiniteComplexityWave,
        Observer,
        WavePropagationOrchestrator,
        StrategicDisagreementValidator as SimulationValidator,
        TranscendentObserver as SimulationTranscendent,
        # Enums and types
        WaveComplexity,
        ObserverType,
        InteractionMode,
        ObserverLimitation,
        ObservationStrategy,
        TranscendentDecision,
        # Factory functions
        create_infinite_complexity_wave,
        create_observer_network,
        create_wave_propagation_orchestrator,
        create_transcendent_observer,
        create_basic_observer
    )
    SIMULATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Simulation framework components not available: {e}")
    SIMULATION_AVAILABLE = False

# Experiment Management
try:
    from .experiment_config import ExperimentConfig, ExperimentParameters, create_default_config
    from .experiment import ExperimentOrchestrator, BayesianOptimizer, create_experiment_orchestrator
    EXPERIMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Experiment management components not available: {e}")
    EXPERIMENT_AVAILABLE = False

# Main framework functions
def get_framework_status():
    """Get status of all framework components"""
    status = {
        'framework_version': __version__,
        'signal_processing': SIGNAL_AVAILABLE,
        'precision_validation': PRECISION_AVAILABLE,
        'oscillatory_framework': OSCILLATORY_AVAILABLE,
        'recursive_enhancement': RECURSION_AVAILABLE,
        'wave_simulation': SIMULATION_AVAILABLE,
        'experiment_management': EXPERIMENT_AVAILABLE
    }

    available_components = sum(status.values()) - 1  # Exclude version
    total_components = 6

    status['framework_completeness'] = f"{available_components}/{total_components} components available"
    status['ready_for_experiments'] = available_components >= 4  # Minimum for basic experiments

    return status

def create_complete_observatory_system():
    """Create a complete observatory system with all available components"""

    system_components = {}

    # Signal processing system
    if SIGNAL_AVAILABLE:
        system_components['signal_processing'] = {
            'mimo_system': create_mimo_signal_amplification_system(),
            'clock_system': create_precise_clock_api_system(),
            'gps_system': create_satellite_temporal_gps_system(),
            'fusion_system': create_signal_fusion_system(),
            'latency_analyzer': create_signal_latency_analyzer(),
            'temporal_db': create_temporal_information_system()
        }

    # Precision validation system
    if PRECISION_AVAILABLE:
        system_components['precision_validation'] = {
            'strategic_validator': create_strategic_disagreement_validator(),
            'statistics_suite': create_precision_statistics_suite(),
            'validation_framework': create_validation_framework()
        }

    # S-Entropy oscillatory system
    if OSCILLATORY_AVAILABLE:
        system_components['s_entropy_framework'] = {
            's_entropy_system': create_s_entropy_system(),
            'hierarchical_navigation': create_hierarchical_navigation(),
            'semantic_amplifier': create_semantic_amplifier()
        }

    # Recursive enhancement system
    if RECURSION_AVAILABLE:
        system_components['recursive_enhancement'] = {
            'dual_function_network': create_dual_function_network(),
            'precision_enhancer': create_precision_enhancer(),
            'atmospheric_network': create_atmospheric_network(),
            'virtual_processor_system': create_virtual_processor_system()
        }

    # Wave simulation system
    if SIMULATION_AVAILABLE:
        system_components['wave_simulation'] = {
            'reality_wave': create_infinite_complexity_wave(),
            'observer_network': create_observer_network([
                {'observer_id': 'quantum_1', 'observer_type': ObserverType.QUANTUM_OBSERVER},
                {'observer_id': 'precision_2', 'observer_type': ObserverType.RESONANT_OBSERVER},
                {'observer_id': 'adaptive_3', 'observer_type': ObserverType.ADAPTIVE_OBSERVER}
            ]),
            'propagation_orchestrator': create_wave_propagation_orchestrator(),
            'transcendent_observer': create_transcendent_observer()
        }

    # Experiment management system
    if EXPERIMENT_AVAILABLE:
        system_components['experiment_management'] = {
            'config': create_default_config(),
            'orchestrator': create_experiment_orchestrator()
        }

    return system_components

def run_comprehensive_demo():
    """Run the comprehensive wave simulation demo"""

    # Check if demo is available
    try:
        # Import the demo from the parent directory
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))

        from comprehensive_wave_simulation_demo import ComprehensiveWaveSimulationDemo

        demo = ComprehensiveWaveSimulationDemo()
        results = demo.run_complete_demonstration()
        return results

    except ImportError as e:
        print(f"Demo not available: {e}")
        print("Please run: python comprehensive_wave_simulation_demo.py")
        return None

def validate_installation():
    """Validate that the installation is working correctly"""

    print("🔍 Observatory Framework Installation Validation")
    print("="*50)

    status = get_framework_status()

    print(f"Framework Version: {status['framework_version']}")
    print(f"Completeness: {status['framework_completeness']}")
    print()

    # Test each component
    components = [
        ('Signal Processing', status['signal_processing']),
        ('Precision Validation', status['precision_validation']),
        ('Oscillatory Framework', status['oscillatory_framework']),
        ('Recursive Enhancement', status['recursive_enhancement']),
        ('Wave Simulation', status['wave_simulation']),
        ('Experiment Management', status['experiment_management'])
    ]

    for component_name, available in components:
        status_icon = "✅" if available else "❌"
        print(f"{status_icon} {component_name}: {'Available' if available else 'Not Available'}")

    print()

    if status['ready_for_experiments']:
        print("🎉 Observatory is ready for experiments!")
        print("Run: observatory.run_comprehensive_demo()")
    else:
        print("⚠️  Some components are missing. Please check installation.")
        print("Run: pip install -r requirements.txt")

    return status

# Framework metadata
__all__ = [
    # Core functions
    'get_framework_status',
    'create_complete_observatory_system',
    'run_comprehensive_demo',
    'validate_installation',

    # Signal processing (if available)
    *([] if not SIGNAL_AVAILABLE else [
        'create_mimo_signal_amplification_system',
        'create_precise_clock_api_system',
        'create_satellite_temporal_gps_system',
        'create_signal_fusion_system',
        'create_signal_latency_analyzer',
        'create_temporal_information_system',
        'MIMOSignalAmplificationEngine',
        'PreciseClockAPIManager',
        'SatelliteTemporalGPSEngine',
        'SignalFusionEngine',
        'SignalLatencyAnalyzer',
        'TemporalDatabaseEngine'
    ]),

    # Precision validation (if available)
    *([] if not PRECISION_AVAILABLE else [
        'StrategicDisagreementValidator',
        'PrecisionStatistics',
        'ValidationFramework',
        'create_strategic_disagreement_validator',
        'create_precision_statistics_suite',
        'create_validation_framework'
    ]),

    # Oscillatory framework (if available)
    *([] if not OSCILLATORY_AVAILABLE else [
        'SemanticDistanceAmplifier',
        'NavigationAlgorithm',
        'HierarchicalOscillatorySystem',
        'SemanticDistanceCalculator',
        'PrecisionAmplifier',
        'create_s_entropy_system',
        'create_hierarchical_navigation',
        'create_semantic_amplifier'
    ]),

    # Recursive enhancement (if available)
    *([] if not RECURSION_AVAILABLE else [
        'DualFunctionMolecule',
        'RecursivePrecisionEnhancer',
        'AtmosphericMolecularNetwork',
        'QuantumTimeVirtualProcessor',
        'create_dual_function_network',
        'create_precision_enhancer',
        'create_atmospheric_network',
        'create_virtual_processor_system'
    ]),

    # Wave simulation (if available)
    *([] if not SIMULATION_AVAILABLE else [
        'InfiniteComplexityWave',
        'Observer',
        'WavePropagationOrchestrator',
        'create_infinite_complexity_wave',
        'create_observer_network',
        'create_wave_propagation_orchestrator',
        'create_transcendent_observer'
    ]),

    # Experiment management (if available)
    *([] if not EXPERIMENT_AVAILABLE else [
        'ExperimentConfig',
        'ExperimentOrchestrator',
        'create_default_config',
        'create_experiment_orchestrator'
    ])
]

# Print status on import (optional)
if __name__ == "__main__":
    validate_installation()
