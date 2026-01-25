"""
Master Runner for Gas Law Validation Instruments
=================================================
Runs all virtual instruments and generates comprehensive validation report.

This script executes the complete instrument suite:
1. Triple Equivalence Validator (TEV)
2. Categorical Temperature Spectrometer (CTS)
3. Categorical Pressure Gauge (CPG)
4. Maxwell-Boltzmann Reconstructor (MBCR)
5. Van der Waals Corrector (VWCC)
6. Quantum Statistics Classifier (QSCC)
7. Categorical Heat Capacity Analyzer (CHCA)
8. Ideal Gas Law Triangulator (IGLT)
9. S-Entropy Coordinate Extractor (SECE)
"""

import os
import sys
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_instrument(instrument_class, name, systems, output_dir):
    """Run a single instrument for multiple systems."""
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    
    results = {}
    for system in systems:
        try:
            print(f"  Running {name} for {system}...")
            
            if name == "QSCC":
                # QSCC uses particle_type instead of system_name
                inst = instrument_class(system)
            else:
                inst = instrument_class(system)
            
            # Run validation
            if hasattr(inst, 'full_validation'):
                inst.full_validation()
            elif hasattr(inst, 'validate_equivalence'):
                inst.validate_equivalence()
            
            # Save outputs
            fig_path = os.path.join(figures_dir, f"panel_{name.lower()}_{system}.png")
            data_path = os.path.join(data_dir, f"{name.lower()}_{system}.json")
            
            inst.create_panel_chart(fig_path)
            inst.save_data(data_path)
            
            results[system] = {"status": "SUCCESS", "figure": fig_path, "data": data_path}
            plt.close('all')
            
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            results[system] = {"status": "ERROR", "error": str(e)}
    
    return results


def generate_summary_report(all_results, output_dir):
    """Generate comprehensive validation summary."""
    report = {
        "title": "Gas Law Validation Instrument Suite - Summary Report",
        "timestamp": datetime.now().isoformat(),
        "instruments": {},
        "statistics": {
            "total_instruments": 0,
            "total_systems": 0,
            "successful": 0,
            "failed": 0,
        }
    }
    
    for instrument, systems in all_results.items():
        report["instruments"][instrument] = systems
        report["statistics"]["total_instruments"] += 1
        
        for system, result in systems.items():
            report["statistics"]["total_systems"] += 1
            if result["status"] == "SUCCESS":
                report["statistics"]["successful"] += 1
            else:
                report["statistics"]["failed"] += 1
    
    # Save report
    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary figure
    create_summary_figure(all_results, output_dir)
    
    return report


def create_summary_figure(all_results, output_dir):
    """Create a summary visualization of all instruments."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Gas Law Validation Instrument Suite - Summary', 
                fontsize=18, fontweight='bold', y=0.98)
    
    instruments = list(all_results.keys())
    
    for idx, (ax, instrument) in enumerate(zip(axes.flatten(), instruments)):
        systems = all_results[instrument]
        
        # Simple status visualization
        system_names = list(systems.keys())
        statuses = [1 if s["status"] == "SUCCESS" else 0 for s in systems.values()]
        colors = ['green' if s == 1 else 'red' for s in statuses]
        
        ax.bar(system_names, statuses, color=colors)
        ax.set_ylim([0, 1.2])
        ax.set_title(instrument, fontsize=12, fontweight='bold')
        ax.set_ylabel('Status (1=Pass)')
        
        # Add status text
        for i, (sys, status) in enumerate(zip(system_names, statuses)):
            text = 'PASS' if status == 1 else 'FAIL'
            ax.text(i, status + 0.05, text, ha='center', fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(instruments), 9):
        axes.flatten()[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    summary_path = os.path.join(output_dir, "figures", "summary_all_instruments.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary figure to {summary_path}")


def main():
    """Run all validation instruments."""
    print("=" * 70)
    print("GAS LAW VALIDATION INSTRUMENT SUITE")
    print("Running all virtual instruments for comprehensive validation")
    print("=" * 70)
    
    # Setup output directories
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    all_results = {}
    
    # Import and run each instrument
    try:
        from triple_equivalence_validator import TripleEquivalenceValidator
        print("\n[1/9] Triple Equivalence Validator (TEV)")
        all_results["TEV"] = run_instrument(
            TripleEquivalenceValidator, "TEV", ["N2", "CO2", "He"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run TEV: {e}")
        all_results["TEV"] = {"error": str(e)}
    
    try:
        from categorical_temperature_spectrometer import CategoricalTemperatureSpectrometer
        print("\n[2/9] Categorical Temperature Spectrometer (CTS)")
        all_results["CTS"] = run_instrument(
            CategoricalTemperatureSpectrometer, "CTS", ["N2", "H2", "He"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run CTS: {e}")
        all_results["CTS"] = {"error": str(e)}
    
    try:
        from categorical_pressure_gauge import CategoricalPressureGauge
        print("\n[3/9] Categorical Pressure Gauge (CPG)")
        all_results["CPG"] = run_instrument(
            CategoricalPressureGauge, "CPG", ["N2", "He", "CO2"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run CPG: {e}")
        all_results["CPG"] = {"error": str(e)}
    
    try:
        from maxwell_boltzmann_reconstructor import MaxwellBoltzmannReconstructor
        print("\n[4/9] Maxwell-Boltzmann Reconstructor (MBCR)")
        all_results["MBCR"] = run_instrument(
            MaxwellBoltzmannReconstructor, "MBCR", ["N2", "H2", "Xe"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run MBCR: {e}")
        all_results["MBCR"] = {"error": str(e)}
    
    try:
        from van_der_waals_corrector import VanDerWaalsCorrector
        print("\n[5/9] Van der Waals Corrector (VWCC)")
        all_results["VWCC"] = run_instrument(
            VanDerWaalsCorrector, "VWCC", ["N2", "CO2", "Ar"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run VWCC: {e}")
        all_results["VWCC"] = {"error": str(e)}
    
    try:
        from quantum_statistics_classifier import QuantumStatisticsClassifier
        print("\n[6/9] Quantum Statistics Classifier (QSCC)")
        all_results["QSCC"] = run_instrument(
            QuantumStatisticsClassifier, "QSCC", ["boson", "fermion"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run QSCC: {e}")
        all_results["QSCC"] = {"error": str(e)}
    
    try:
        from categorical_heat_capacity_analyzer import CategoricalHeatCapacityAnalyzer
        print("\n[7/9] Categorical Heat Capacity Analyzer (CHCA)")
        all_results["CHCA"] = run_instrument(
            CategoricalHeatCapacityAnalyzer, "CHCA", ["Ar", "N2", "CO2"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run CHCA: {e}")
        all_results["CHCA"] = {"error": str(e)}
    
    try:
        from ideal_gas_law_triangulator import IdealGasLawTriangulator
        print("\n[8/9] Ideal Gas Law Triangulator (IGLT)")
        all_results["IGLT"] = run_instrument(
            IdealGasLawTriangulator, "IGLT", ["N2", "He", "CO2"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run IGLT: {e}")
        all_results["IGLT"] = {"error": str(e)}
    
    try:
        from s_entropy_coordinate_extractor import SEntropyCoordinateExtractor
        print("\n[9/9] S-Entropy Coordinate Extractor (SECE)")
        all_results["SECE"] = run_instrument(
            SEntropyCoordinateExtractor, "SECE", ["N2", "He", "CO2"], output_dir
        )
    except Exception as e:
        print(f"  Failed to run SECE: {e}")
        all_results["SECE"] = {"error": str(e)}
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)
    
    report = generate_summary_report(all_results, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal Instruments: {report['statistics']['total_instruments']}")
    print(f"Total System Tests: {report['statistics']['total_systems']}")
    print(f"Successful: {report['statistics']['successful']}")
    print(f"Failed: {report['statistics']['failed']}")
    print(f"\nReport saved to: {os.path.join(output_dir, 'validation_report.json')}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Data saved to: {data_dir}")
    
    return report


if __name__ == "__main__":
    main()

