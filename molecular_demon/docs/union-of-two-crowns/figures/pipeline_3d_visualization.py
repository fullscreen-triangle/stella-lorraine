"""
3D Pipeline Visualization
==========================

Creates publication-quality panel charts for 3D object pipeline.
Integrates with existing virtual MS visualization framework.

Author: Kundai Farai Sachikonye
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    from .pipeline_3d_transformation import Object3DState, Pipeline3DTransformation
except ImportError:
    from pipeline_3d_transformation import Object3DState, Pipeline3DTransformation

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

logger = logging.getLogger(__name__)


class Pipeline3DVisualizer:
    """
    Visualizes 3D object pipeline with panel charts.
    
    Creates:
    1. 3D object grid (2x3 panel)
    2. Property evolution charts
    3. Physics validation charts
    4. Conservation validation
    """
    
    def __init__(self, objects_3d: Dict[str, Object3DState], experiment_name: str):
        """
        Initialize visualizer.
        
        Args:
            objects_3d: Dictionary of 3D objects by stage
            experiment_name: Name of experiment
        """
        self.objects_3d = objects_3d
        self.experiment_name = experiment_name
        
        # Stage order
        self.stage_order = [
            'solution',
            'chromatography',
            'ionization',
            'ms1',
            'ms2',
            'droplet'
        ]
    
    def create_all_visualizations(self, output_dir: Path) -> Dict[str, Path]:
        """
        Create all visualization panels.
        
        Args:
            output_dir: Directory to save figures
            
        Returns:
            Dictionary mapping figure name to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating visualizations for {self.experiment_name}")
        
        figures = {}
        
        # 1. 3D object grid (main figure)
        figures['grid'] = self.create_3d_object_grid(output_dir)
        
        # 2. Property evolution
        figures['properties'] = self.create_property_evolution(output_dir)
        
        # 3. Physics validation
        figures['physics'] = self.create_physics_validation(output_dir)
        
        # 4. S-entropy trajectory
        figures['sentropy'] = self.create_sentropy_trajectory(output_dir)
        
        logger.info(f"Created {len(figures)} visualization panels")
        
        return figures
    
    def create_3d_object_grid(self, output_dir: Path) -> Path:
        """
        Create 2x3 grid of 3D object projections.
        
        Main visualization showing transformation through pipeline.
        """
        logger.info("  Creating 3D object grid...")
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'3D Object Pipeline Transformation - {self.experiment_name}',
                     fontsize=16, fontweight='bold', y=0.98)
        
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, stage in enumerate(self.stage_order):
            if stage not in self.objects_3d:
                continue
            
            obj = self.objects_3d[stage]
            
            row = idx // 3
            col = idx % 3
            
            ax = fig.add_subplot(gs[row, col], projection='3d')
            
            self._plot_3d_object(ax, obj, stage)
        
        output_file = output_dir / f'{self.experiment_name}_grid.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved: {output_file.name}")
        return output_file
    
    def _plot_3d_object(self, ax: Axes3D, obj: Object3DState, stage: str) -> None:
        """Plot a single 3D object."""
        # Get object properties
        S_k, S_t, S_e = obj.S_k, obj.S_t, obj.S_e
        a, b, c = obj.dimensions
        color = obj.color
        
        # Generate surface based on shape
        if obj.shape == 'sphere':
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x = S_k + a * np.outer(np.cos(u), np.sin(v))
            y = S_t + b * np.outer(np.sin(u), np.sin(v))
            z = S_e + c * np.outer(np.ones(np.size(u)), np.cos(v))
            
        elif obj.shape == 'ellipsoid':
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x = S_k + a * np.outer(np.cos(u), np.sin(v))
            y = S_t + b * np.outer(np.sin(u), np.sin(v))
            z = S_e + c * np.outer(np.ones(np.size(u)), np.cos(v))
            
        elif obj.shape == 'fragmenting_sphere':
            # Sphere with fractures
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            r = a * (1 + 0.1 * np.sin(5 * np.outer(u, np.ones(len(v)))))
            x = S_k + r * np.outer(np.cos(u), np.sin(v))
            y = S_t + r * np.outer(np.sin(u), np.sin(v))
            z = S_e + c * np.outer(np.ones(np.size(u)), np.cos(v))
            
        elif obj.shape == 'sphere_array':
            # Multiple small spheres
            n_spheres = min(50, obj.molecule_count // 20)
            np.random.seed(42)
            
            for i in range(n_spheres):
                offset_x = (np.random.rand() - 0.5) * a
                offset_y = (np.random.rand() - 0.5) * b
                offset_z = (np.random.rand() - 0.5) * c
                
                r = 0.02
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 8)
                x = S_k + offset_x + r * np.outer(np.cos(u), np.sin(v))
                y = S_t + offset_y + r * np.outer(np.sin(u), np.sin(v))
                z = S_e + offset_z + r * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x, y, z, color=color, alpha=0.6, edgecolor='none')
            
            # Set limits and return early
            ax.set_xlim([S_k - a/2, S_k + a/2])
            ax.set_ylim([S_t - b/2, S_t + b/2])
            ax.set_zlim([S_e - c/2, S_e + c/2])
            ax.set_xlabel('S_k', fontweight='bold')
            ax.set_ylabel('S_t', fontweight='bold')
            ax.set_zlabel('S_e', fontweight='bold')
            ax.set_title(f'{stage.upper()}\n{obj.shape}\nN={obj.molecule_count}',
                        fontsize=11, fontweight='bold')
            return
            
        elif obj.shape == 'cascade':
            # Cascading spheres
            n_levels = 3
            for level in range(n_levels):
                scale = 1.0 - level * 0.2
                offset_z = level * 0.15
                
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 15)
                x = S_k + a * scale * np.outer(np.cos(u), np.sin(v))
                y = S_t + b * scale * np.outer(np.sin(u), np.sin(v))
                z = S_e + offset_z + c * scale * np.outer(np.ones(np.size(u)), np.cos(v))
                
                alpha = 0.8 - level * 0.2
                ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')
            
            ax.set_xlim([S_k - a, S_k + a])
            ax.set_ylim([S_t - b, S_t + b])
            ax.set_zlim([S_e - c/2, S_e + c + 0.5])
            ax.set_xlabel('S_k', fontweight='bold')
            ax.set_ylabel('S_t', fontweight='bold')
            ax.set_zlabel('S_e', fontweight='bold')
            ax.set_title(f'{stage.upper()}\n{obj.shape}\nN={obj.molecule_count}',
                        fontsize=11, fontweight='bold')
            return
            
        elif obj.shape == 'wave_pattern':
            # Sphere with wave pattern
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 30)
            r = a * (1 + 0.05 * np.sin(8 * np.outer(u, np.ones(len(v)))) * 
                     np.sin(8 * np.outer(np.ones(len(u)), v)))
            x = S_k + r * np.outer(np.cos(u), np.sin(v))
            y = S_t + r * np.outer(np.sin(u), np.sin(v))
            z = S_e + c * np.outer(np.ones(np.size(u)), np.cos(v))
            
        else:
            # Default sphere
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x = S_k + a * np.outer(np.cos(u), np.sin(v))
            y = S_t + b * np.outer(np.sin(u), np.sin(v))
            z = S_e + c * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot surface
        ax.plot_surface(x, y, z, color=color, alpha=0.7, edgecolor='none')
        
        # Set labels and title
        ax.set_xlabel('S_k', fontweight='bold')
        ax.set_ylabel('S_t', fontweight='bold')
        ax.set_zlabel('S_e', fontweight='bold')
        
        # Title with metadata
        title = f'{stage.upper()}\n{obj.shape}\nN={obj.molecule_count}'
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
    
    def create_property_evolution(self, output_dir: Path) -> Path:
        """
        Create property evolution charts.
        
        Shows how thermodynamic properties change through pipeline.
        """
        logger.info("  Creating property evolution charts...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Thermodynamic Property Evolution - {self.experiment_name}',
                     fontsize=16, fontweight='bold')
        
        # Extract properties
        stages = []
        temperatures = []
        pressures = []
        entropies = []
        volumes = []
        molecule_counts = []
        
        for stage in self.stage_order:
            if stage in self.objects_3d:
                obj = self.objects_3d[stage]
                stages.append(stage)
                temperatures.append(obj.temperature)
                pressures.append(obj.pressure)
                entropies.append(obj.entropy)
                volumes.append(obj.volume)
                molecule_counts.append(obj.molecule_count)
        
        x_pos = np.arange(len(stages))
        
        # Temperature
        ax = axes[0, 0]
        ax.plot(x_pos, temperatures, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_ylabel('Temperature (S-variance)', fontweight='bold')
        ax.set_title('Categorical Temperature', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Pressure
        ax = axes[0, 1]
        ax.plot(x_pos, pressures, 'o-', linewidth=2, markersize=8, color='blue')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_ylabel('Pressure (sampling rate)', fontweight='bold')
        ax.set_title('Categorical Pressure', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Entropy
        ax = axes[0, 2]
        ax.plot(x_pos, entropies, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_ylabel('Entropy (S-spread)', fontweight='bold')
        ax.set_title('Categorical Entropy', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Volume
        ax = axes[1, 0]
        ax.plot(x_pos, volumes, 'o-', linewidth=2, markersize=8, color='purple')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_ylabel('Volume (S-space)', fontweight='bold')
        ax.set_title('S-Space Volume', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Molecule count
        ax = axes[1, 1]
        ax.plot(x_pos, molecule_counts, 'o-', linewidth=2, markersize=8, color='orange')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_ylabel('Molecule Count', fontweight='bold')
        ax.set_title('Molecular Population', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # Combined view
        ax = axes[1, 2]
        ax.plot(x_pos, np.array(temperatures) / max(temperatures), 'o-', label='T', linewidth=2)
        ax.plot(x_pos, np.array(pressures) / max(pressures), 's-', label='P', linewidth=2)
        ax.plot(x_pos, np.array(entropies) / max(entropies), '^-', label='S', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value', fontweight='bold')
        ax.set_title('Normalized Properties', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f'{self.experiment_name}_properties.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved: {output_file.name}")
        return output_file
    
    def create_physics_validation(self, output_dir: Path) -> Path:
        """
        Create physics validation charts.
        
        Shows dimensionless numbers for droplet validation.
        """
        logger.info("  Creating physics validation charts...")
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Physics Validation - Dimensionless Numbers - {self.experiment_name}',
                     fontsize=16, fontweight='bold')
        
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get droplet object
        if 'droplet' not in self.objects_3d:
            logger.warning("No droplet object found for physics validation")
            plt.close()
            return None
        
        droplet = self.objects_3d['droplet']
        
        # Weber number
        ax = fig.add_subplot(gs[0, 0])
        We = droplet.weber_number or 0
        We_valid = 0.1 < We < 1000
        color = 'green' if We_valid else 'red'
        ax.bar(['Weber'], [We], color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(0.1, color='orange', linestyle='--', label='Min valid')
        ax.axhline(1000, color='orange', linestyle='--', label='Max valid')
        ax.set_ylabel('We = ρv²L/σ', fontweight='bold')
        ax.set_title(f'Weber Number\n{"VALID" if We_valid else "INVALID"}', fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Reynolds number
        ax = fig.add_subplot(gs[0, 1])
        Re = droplet.reynolds_number or 0
        Re_valid = 10 < Re < 10000
        color = 'green' if Re_valid else 'red'
        ax.bar(['Reynolds'], [Re], color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(10, color='orange', linestyle='--', label='Min valid')
        ax.axhline(10000, color='orange', linestyle='--', label='Max valid')
        ax.set_ylabel('Re = ρvL/μ', fontweight='bold')
        ax.set_title(f'Reynolds Number\n{"VALID" if Re_valid else "INVALID"}', fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Ohnesorge number
        ax = fig.add_subplot(gs[0, 2])
        Oh = droplet.ohnesorge_number or 0
        Oh_valid = 0.001 < Oh < 1.0
        color = 'green' if Oh_valid else 'red'
        ax.bar(['Ohnesorge'], [Oh], color=color, alpha=0.7, edgecolor='black', linewidth=2)
        ax.axhline(0.001, color='orange', linestyle='--', label='Min valid')
        ax.axhline(1.0, color='orange', linestyle='--', label='Max valid')
        ax.set_ylabel('Oh = √We / Re', fontweight='bold')
        ax.set_title(f'Ohnesorge Number\n{"VALID" if Oh_valid else "INVALID"}', fontweight='bold')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Physical properties
        ax = fig.add_subplot(gs[1, 0])
        props = ['Radius (μm)', 'Velocity (m/s)', 'Surface Tension (N/m)']
        values = [
            (droplet.radius or 0) * 1e6,
            droplet.velocity or 0,
            droplet.surface_tension or 0
        ]
        ax.bar(props, values, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Physical Droplet Properties', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3e}', ha='center', va='bottom', fontweight='bold')
        
        # Validation summary
        ax = fig.add_subplot(gs[1, 1:])
        ax.axis('off')
        
        summary_text = f"""
PHYSICS VALIDATION SUMMARY

Weber Number (We):     {We:.4e}  {"✓ VALID" if We_valid else "✗ INVALID"}
  Range: 0.1 < We < 1000
  Meaning: Inertial forces vs surface tension

Reynolds Number (Re):  {Re:.4e}  {"✓ VALID" if Re_valid else "✗ INVALID"}
  Range: 10 < Re < 10000
  Meaning: Inertial forces vs viscous forces

Ohnesorge Number (Oh): {Oh:.4e}  {"✓ VALID" if Oh_valid else "✗ INVALID"}
  Range: 0.001 < Oh < 1.0
  Meaning: √We / Re (viscosity vs inertia & surface tension)

OVERALL: {"✓ PHYSICALLY REALIZABLE" if (We_valid and Re_valid and Oh_valid) else "✗ NEEDS ADJUSTMENT"}

Droplet Properties:
  Radius:          {(droplet.radius or 0)*1e6:.2f} μm
  Velocity:        {droplet.velocity or 0:.2f} m/s
  Surface Tension: {droplet.surface_tension or 0:.4f} N/m
  Temperature:     {droplet.temperature:.4e} (S-variance)
"""
        
        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        output_file = output_dir / f'{self.experiment_name}_physics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved: {output_file.name}")
        return output_file
    
    def create_sentropy_trajectory(self, output_dir: Path) -> Path:
        """
        Create S-entropy trajectory through pipeline.
        
        Shows path through (S_k, S_t, S_e) space.
        """
        logger.info("  Creating S-entropy trajectory...")
        
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f'S-Entropy Trajectory - {self.experiment_name}',
                     fontsize=16, fontweight='bold')
        
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        
        # Extract S-coordinates
        stages = []
        S_k_vals = []
        S_t_vals = []
        S_e_vals = []
        
        for stage in self.stage_order:
            if stage in self.objects_3d:
                obj = self.objects_3d[stage]
                stages.append(stage)
                S_k_vals.append(obj.S_k)
                S_t_vals.append(obj.S_t)
                S_e_vals.append(obj.S_e)
        
        # 3D trajectory
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        
        # Plot trajectory
        ax.plot(S_k_vals, S_t_vals, S_e_vals, 'o-', linewidth=3, markersize=10,
               color='purple', label='Pipeline trajectory')
        
        # Mark start and end
        ax.scatter([S_k_vals[0]], [S_t_vals[0]], [S_e_vals[0]], 
                  s=200, c='green', marker='o', edgecolor='black', linewidth=2,
                  label='Start (solution)')
        ax.scatter([S_k_vals[-1]], [S_t_vals[-1]], [S_e_vals[-1]], 
                  s=200, c='red', marker='s', edgecolor='black', linewidth=2,
                  label='End (droplet)')
        
        # Label stages
        for i, stage in enumerate(stages):
            ax.text(S_k_vals[i], S_t_vals[i], S_e_vals[i], f'  {stage}',
                   fontsize=9, fontweight='bold')
        
        ax.set_xlabel('S_k (Knowledge)', fontweight='bold')
        ax.set_ylabel('S_t (Time)', fontweight='bold')
        ax.set_zlabel('S_e (Entropy)', fontweight='bold')
        ax.set_title('3D Trajectory in S-Entropy Space', fontweight='bold')
        ax.legend()
        ax.view_init(elev=20, azim=45)
        
        # 2D projections
        ax = fig.add_subplot(gs[0, 1])
        
        x_pos = np.arange(len(stages))
        
        ax.plot(x_pos, S_k_vals, 'o-', linewidth=2, markersize=8, label='S_k', color='blue')
        ax.plot(x_pos, S_t_vals, 's-', linewidth=2, markersize=8, label='S_t', color='green')
        ax.plot(x_pos, S_e_vals, '^-', linewidth=2, markersize=8, label='S_e', color='red')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=45, ha='right')
        ax.set_ylabel('S-Coordinate Value', fontweight='bold')
        ax.set_title('S-Coordinate Evolution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / f'{self.experiment_name}_sentropy.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"    Saved: {output_file.name}")
        return output_file


def visualize_experiment(experiment_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Generate all visualizations for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        output_dir: Optional output directory (defaults to experiment_dir/visualizations)
        
    Returns:
        Dictionary mapping figure name to file path
    """
    # Generate 3D objects
    transformer = Pipeline3DTransformation(experiment_dir)
    objects = transformer.generate_all_objects()
    
    # Create visualizations
    if output_dir is None:
        output_dir = experiment_dir / 'visualizations'
    
    visualizer = Pipeline3DVisualizer(objects, transformer.experiment_name)
    figures = visualizer.create_all_visualizations(output_dir)
    
    return figures


if __name__ == "__main__":
    # Test with one experiment
    logging.basicConfig(level=logging.INFO)
    
    experiment_dir = Path("results/ucdavis_fast_analysis/A_M3_negPFP_03")
    
    figures = visualize_experiment(experiment_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATIONS CREATED")
    print("="*70)
    print(f"\nExperiment: {experiment_dir.name}")
    print(f"\nGenerated {len(figures)} figures:")
    for name, path in figures.items():
        print(f"  {name:15s} -> {path.name}")

