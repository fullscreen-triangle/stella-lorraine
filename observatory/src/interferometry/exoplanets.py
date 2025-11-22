# figures/publication_exoplanet_results.py

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import Circle, Wedge
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.constants as const
from scipy.ndimage import gaussian_filter
import seaborn as sns

# Publication settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.5
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'brown': '#949494',
    'yellow': '#ECE133'
}


def generate_exoplanet_results_figure():
    """
    Figure 6: Exoplanet Imaging Results (4 panels)

    Based on validated results from angular_resolution_validation.py
    """

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    panel_labels = ['A', 'B', 'C', 'D']

    # ========== PANEL A: Resolution Elements vs Distance ==========
    ax_a = fig.add_subplot(gs[0, 0])

    # Distance range
    distances_pc = np.logspace(0, 3, 100)  # 1 to 1000 pc

    # Angular resolution
    baseline = 1e7  # 10^4 km
    wavelength = 500e-9
    theta_resolution = wavelength / baseline  # rad

    # Planet types (FIXED: Use R_E instead of R⊕)
    R_earth = 6.371e6  # m
    planets = {
        'Earth (1 R_E)': 1.0 * R_earth,
        'Super-Earth (2 R_E)': 2.0 * R_earth,
        'Neptune (4 R_E)': 4.0 * R_earth,
        'Jupiter (11 R_E)': 11.2 * R_earth
    }

    colors_planets = [COLORS['blue'], COLORS['green'],
                     COLORS['orange'], COLORS['red']]

    for (name, radius), color in zip(planets.items(), colors_planets):
        # Angular size vs distance
        angular_size = radius / (distances_pc * 3.086e16)  # rad

        # Resolution elements
        resolution_elements = angular_size / theta_resolution

        ax_a.loglog(distances_pc, resolution_elements,
                   color=color, linewidth=2.5, label=name)

    # Mark detectability threshold
    ax_a.axhline(1, color='black', linestyle='--', linewidth=2,
                label='Detection limit (1 pixel)')
    ax_a.axhline(10, color='gray', linestyle=':', linewidth=2,
                label='Imaging threshold (10 pixels)')

    # Mark validated scenarios
    scenarios = {
        'Earth @ 10 pc': (10, 412.9),
        'Earth @ 100 pc': (100, 41.3),
        'Jupiter @ 10 pc': (10, 4624.4)
    }

    for name, (dist, res_elem) in scenarios.items():
        ax_a.scatter(dist, res_elem, s=200, marker='*',
                    edgecolors='black', linewidth=2, zorder=5)
        ax_a.annotate(name, (dist, res_elem),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax_a.set_xlabel('Distance (pc)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Resolution elements (pixels)', fontsize=11, fontweight='bold')
    ax_a.set_title('Imaging Capability vs Distance', fontsize=12, fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=8)
    ax_a.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax_a.set_xlim(1, 1000)
    ax_a.set_ylim(0.1, 1e5)
    ax_a.text(-0.15, 1.05, panel_labels[0], transform=ax_a.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Shade imageable region
    ax_a.fill_between(distances_pc, 10, 1e5, alpha=0.1,
                     color=COLORS['green'], label='Imageable')

    # ========== PANEL B: Spatial Resolution Map ==========
    ax_b = fig.add_subplot(gs[0, 1])

    # Create feature detectability map
    distances = np.logspace(0, 3, 50)

    # Spatial resolution in km
    spatial_res = theta_resolution * (distances * 3.086e16) / 1e3

    # Feature scales
    features = {
        'Cities': 10,
        'Mountain ranges': 100,
        'Continents': 1000,
        'Hemispheres': 6371
    }

    # Plot spatial resolution
    ax_b.loglog(distances, spatial_res, color=COLORS['blue'],
               linewidth=3, label='Spatial resolution')

    # Mark feature scales
    colors_features = [COLORS['red'], COLORS['orange'],
                      COLORS['green'], COLORS['purple']]

    for (feature, scale), color in zip(features.items(), colors_features):
        ax_b.axhline(scale, color=color, linestyle='--',
                    linewidth=2, alpha=0.7, label=feature)

    ax_b.set_xlabel('Distance (pc)', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Spatial resolution (km)', fontsize=11, fontweight='bold')
    ax_b.set_title('Detectable Feature Scales', fontsize=12, fontweight='bold')
    ax_b.legend(loc='upper left', fontsize=8)
    ax_b.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax_b.text(-0.15, 1.05, panel_labels[1], transform=ax_b.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add annotations
    ax_b.annotate('', xy=(10, 15.4), xytext=(10, 1),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax_b.text(12, 5, 'Earth @ 10 pc\n15.4 km resolution',
             fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # ========== PANEL C: Simulated Earth Image ==========
    ax_c = fig.add_subplot(gs[1, 0])

    # Generate realistic Earth-like planet (413 pixels diameter)
    image_size = 512
    planet_diameter_pix = 413

    # Create image
    planet_image = np.zeros((image_size, image_size, 3))  # RGB

    # Planet disk
    center = image_size // 2
    Y, X = np.ogrid[:image_size, :image_size]
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    planet_mask = dist_from_center <= planet_diameter_pix / 2

    # Ocean (blue)
    planet_image[planet_mask, 2] = 0.2  # Blue channel

    # Continents (green/brown)
    np.random.seed(42)
    for _ in range(8):
        # Random continent
        cx = center + np.random.randint(-150, 150)
        cy = center + np.random.randint(-150, 150)
        size = np.random.randint(40, 100)

        continent_mask = np.sqrt((X - cx)**2 + (Y - cy)**2) <= size
        combined_mask = continent_mask & planet_mask

        # Land color (green + brown)
        planet_image[combined_mask, 1] = 0.6  # Green
        planet_image[combined_mask, 0] = 0.3  # Red (for brown)

    # Ice caps (white)
    ice_north = (Y < center - planet_diameter_pix / 3) & planet_mask
    ice_south = (Y > center + planet_diameter_pix / 3) & planet_mask
    planet_image[ice_north | ice_south] = [0.9, 0.9, 0.95]

    # Clouds (white, semi-transparent)
    for _ in range(15):
        cx = center + np.random.randint(-180, 180)
        cy = center + np.random.randint(-180, 180)
        size = np.random.randint(20, 60)

        cloud_mask = np.sqrt((X - cx)**2 + (Y - cy)**2) <= size
        combined_mask = cloud_mask & planet_mask

        # Add clouds with transparency
        planet_image[combined_mask] = planet_image[combined_mask] * 0.5 + \
                                      np.array([0.95, 0.95, 0.95]) * 0.5

    # Apply Gaussian blur (telescope PSF)
    for i in range(3):
        planet_image[:, :, i] = gaussian_filter(planet_image[:, :, i], sigma=1.5)

    # Add noise (photon shot noise)
    noise = 0.02 * np.random.randn(image_size, image_size, 3)
    planet_image = np.clip(planet_image + noise, 0, 1)

    # Plot
    ax_c.imshow(planet_image, origin='lower', interpolation='bilinear')
    ax_c.set_xlabel('RA offset (pixels)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Dec offset (pixels)', fontsize=11, fontweight='bold')
    ax_c.set_title('Simulated: Earth-like Planet at 10 pc\n(413 resolution elements)',
                  fontsize=12, fontweight='bold')
    ax_c.text(-0.15, 1.05, panel_labels[2], transform=ax_c.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add scale bar
    scale_km = 1000  # 1000 km
    scale_pix = scale_km / 15.4  # 15.4 km/pixel

    ax_c.plot([50, 50 + scale_pix], [50, 50], 'w-', linewidth=4)
    ax_c.text(50 + scale_pix/2, 65, f'{scale_km} km',
             color='white', fontsize=10, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    # Add compass
    arrow_length = 30
    ax_c.arrow(450, 50, 0, arrow_length, head_width=10, head_length=8,
              fc='white', ec='white', linewidth=2)
    ax_c.text(450, 50 + arrow_length + 15, 'N', color='white',
             fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    # ========== PANEL D: Comparison Gallery ==========
    ax_d = fig.add_subplot(gs[1, 1])

    # Create comparison: JWST vs This Work
    comparison_size = 256

    # JWST simulation (unresolved)
    jwst_image = np.zeros((comparison_size, comparison_size))
    jwst_resolution_pix = 2  # ~2 pixels for Earth at 10 pc

    center_jwst = comparison_size // 2
    Y_jwst, X_jwst = np.ogrid[:comparison_size, :comparison_size]
    dist_jwst = np.sqrt((X_jwst - center_jwst)**2 + (Y_jwst - center_jwst)**2)

    jwst_image[dist_jwst <= jwst_resolution_pix] = 1.0
    jwst_image = gaussian_filter(jwst_image, sigma=2)

    # This work simulation (resolved)
    thiswork_image = np.zeros((comparison_size, comparison_size))
    thiswork_resolution_pix = 100  # 413 pixels scaled to 100

    dist_thiswork = np.sqrt((X_jwst - center_jwst)**2 + (Y_jwst - center_jwst)**2)
    thiswork_mask = dist_thiswork <= thiswork_resolution_pix / 2

    # Add continents
    thiswork_image[thiswork_mask] = 0.3
    for _ in range(5):
        cx = center_jwst + np.random.randint(-40, 40)
        cy = center_jwst + np.random.randint(-40, 40)
        size = np.random.randint(10, 25)

        continent = np.sqrt((X_jwst - cx)**2 + (Y_jwst - cy)**2) <= size
        thiswork_image[continent & thiswork_mask] = 0.8

    thiswork_image = gaussian_filter(thiswork_image, sigma=1)

    # Create side-by-side comparison
    comparison = np.hstack([jwst_image, thiswork_image])

    im = ax_d.imshow(comparison, cmap='terrain', origin='lower', vmin=0, vmax=1)

    # Add dividing line
    ax_d.axvline(comparison_size, color='white', linewidth=3, linestyle='--')

    # Labels
    ax_d.text(comparison_size / 2, 10, 'JWST\n(unresolved)',
             color='white', fontsize=11, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax_d.text(comparison_size * 1.5, 10, 'This Work\n(resolved)',
             color='white', fontsize=11, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    ax_d.set_xlabel('Angular position', fontsize=11, fontweight='bold')
    ax_d.set_ylabel('Angular position', fontsize=11, fontweight='bold')
    ax_d.set_title('Comparison: JWST vs Categorical Interferometry',
                  fontsize=12, fontweight='bold')
    ax_d.text(-0.15, 1.05, panel_labels[3], transform=ax_d.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Colorbar
    divider = make_axes_locatable(ax_d)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Brightness', fontsize=9, fontweight='bold')

    # Save
    plt.savefig('Figure6_Exoplanet_Results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Figure6_Exoplanet_Results.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 6 saved: Exoplanet imaging results")

    plt.close(fig)  # Free memory
    return fig


def generate_error_budget_figure():
    """
    Figure 7: Error Budget Analysis (4 panels)
    """

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    panel_labels = ['A', 'B', 'C', 'D']

    # ========== PANEL A: Angular Resolution Error Waterfall ==========
    ax_a = fig.add_subplot(gs[0, 0])

    # Error components (from corrected analysis)
    components = {
        'Measurement': 0.0103,
        'Wavelength\ncalibration': 1.03e-8,
        'Baseline GPS': 1.03e-11,
        'Baseline\norientation': 1.03e-9,
        'Clock drift': 2.06e-20,
        'Atmospheric\njitter': 1.03e-10,
        'Photon shot\nnoise': 1.03e-5,
        'Detector\nthermal': 1.03e-6,
        'Total\nuncertainty': 1.03e-5
    }

    names = list(components.keys())
    values = list(components.values())

    # Waterfall chart
    x_pos = np.arange(len(names))
    colors_bars = ['blue'] + ['red']*7 + ['green']

    bars = ax_a.bar(x_pos, values, color=colors_bars,
                   edgecolor='black', linewidth=1.5, alpha=0.7)

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax_a.set_ylabel('Uncertainty (μas)', fontsize=11, fontweight='bold')
    ax_a.set_yscale('log')
    ax_a.set_title('Angular Resolution Error Budget', fontsize=12, fontweight='bold')
    ax_a.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax_a.text(-0.15, 1.05, panel_labels[0], transform=ax_a.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add connecting lines
    for i in range(1, len(values)-1):
        ax_a.plot([i-0.5, i+0.5], [values[i], values[i]],
                 'k--', linewidth=1, alpha=0.3)

    # ========== PANEL B: Temperature Error Waterfall ==========
    ax_b = fig.add_subplot(gs[0, 1])

    # Temperature error components
    temp_components = {
        'Measurement': 100000,  # 100 nK = 100,000 pK
        'Timing\nprecision': 17,
        'Statistical\nsampling': 316,
        'State\nreconstruction': 1000,
        'Measurement\nheating': 0.001,
        'Magnetic field\nnoise': 1,
        'Total\nuncertainty': 1049
    }

    temp_names = list(temp_components.keys())
    temp_values = list(temp_components.values())

    colors_temp = ['blue'] + ['red']*5 + ['green']

    bars_temp = ax_b.bar(range(len(temp_names)), temp_values,
                        color=colors_temp, edgecolor='black',
                        linewidth=1.5, alpha=0.7)

    ax_b.set_xticks(range(len(temp_names)))
    ax_b.set_xticklabels(temp_names, rotation=45, ha='right', fontsize=9)
    ax_b.set_ylabel('Uncertainty (pK)', fontsize=11, fontweight='bold')
    ax_b.set_yscale('log')
    ax_b.set_title('Temperature Error Budget', fontsize=12, fontweight='bold')
    ax_b.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax_b.text(-0.15, 1.05, panel_labels[1], transform=ax_b.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # ========== PANEL C: Sensitivity Analysis ==========
    ax_c = fig.add_subplot(gs[1, 0])

    # How error scales with parameters
    baselines = np.logspace(3, 8, 50)  # 1 km to 100,000 km

    # Fixed errors
    wavelength_error = 1e-12  # m
    pointing_error = 1e-6  # rad

    # Scaling
    theta_resolution = 500e-9 / baselines

    # Error contributions
    error_wavelength = theta_resolution * (wavelength_error / 500e-9)
    error_pointing = 0.7 * theta_resolution * pointing_error
    error_total = np.sqrt(error_wavelength**2 + error_pointing**2)

    # Convert to μas
    error_wavelength_uas = error_wavelength * (180 * 3600 / np.pi) * 1e6
    error_pointing_uas = error_pointing * (180 * 3600 / np.pi) * 1e6
    error_total_uas = error_total * (180 * 3600 / np.pi) * 1e6

    ax_c.loglog(baselines / 1e3, error_wavelength_uas,
               color=COLORS['blue'], linewidth=2.5,
               label='Wavelength calibration')
    ax_c.loglog(baselines / 1e3, error_pointing_uas,
               color=COLORS['red'], linewidth=2.5,
               label='Pointing error')
    ax_c.loglog(baselines / 1e3, error_total_uas,
               color=COLORS['green'], linewidth=3,
               linestyle='--', label='Total')

    # Mark operational baseline
    ax_c.axvline(1e4, color='black', linestyle=':',
                linewidth=2, alpha=0.5, label='Operational (10^4 km)')

    ax_c.set_xlabel('Baseline length (km)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Uncertainty (μas)', fontsize=11, fontweight='bold')
    ax_c.set_title('Error Scaling with Baseline', fontsize=12, fontweight='bold')
    ax_c.legend(loc='upper right', fontsize=9)
    ax_c.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax_c.text(-0.15, 1.05, panel_labels[2], transform=ax_c.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # ========== PANEL D: Systematic vs Statistical (FIXED) ==========
    ax_d = fig.add_subplot(gs[1, 1])

    # Angular resolution
    angular_systematic = 1.03e-8 + 1.03e-11 + 1.03e-9 + 2.06e-20
    angular_statistical = np.sqrt((1.03e-10)**2 + (1.03e-5)**2 + (1.03e-6)**2)

    angular_total = np.sqrt(angular_systematic**2 + angular_statistical**2)
    angular_sys_frac = (angular_systematic / angular_total) * 100
    angular_stat_frac = (angular_statistical / angular_total) * 100

    # Temperature
    temp_systematic = 17 + 1000 + 0.001
    temp_statistical = np.sqrt(316**2 + 1**2)

    temp_total = np.sqrt(temp_systematic**2 + temp_statistical**2)
    temp_sys_frac = (temp_systematic / temp_total) * 100
    temp_stat_frac = (temp_statistical / temp_total) * 100

    # Create grouped bar chart (FIXED: no nested subplots)
    categories = ['Angular\nResolution', 'Temperature']
    systematic_vals = [angular_sys_frac, temp_sys_frac]
    statistical_vals = [angular_stat_frac, temp_stat_frac]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax_d.bar(x - width/2, systematic_vals, width,
                    label='Systematic', color=COLORS['red'],
                    edgecolor='black', linewidth=1.5, alpha=0.7)
    bars2 = ax_d.bar(x + width/2, statistical_vals, width,
                    label='Statistical', color=COLORS['blue'],
                    edgecolor='black', linewidth=1.5, alpha=0.7)

    ax_d.set_ylabel('Error contribution (%)', fontsize=11, fontweight='bold')
    ax_d.set_title('Systematic vs Statistical Error Composition',
                  fontsize=12, fontweight='bold')
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(categories, fontsize=10)
    ax_d.legend(loc='upper right', fontsize=10)
    ax_d.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax_d.set_ylim(0, 105)

    # Add percentage labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_d.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_d.text(-0.15, 1.05, panel_labels[3], transform=ax_d.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Save
    plt.savefig('Figure7_Error_Budgets.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Figure7_Error_Budgets.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 7 saved: Error budget analysis")

    plt.close(fig)  # Free memory
    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING ADDITIONAL PUBLICATION FIGURES")
    print("=" * 70)

    sns.set_style("whitegrid")

    # Generate figures
    fig6 = generate_exoplanet_results_figure()
    fig7 = generate_error_budget_figure()

    print("\n" + "=" * 70)
    print("FIGURES GENERATED")
    print("=" * 70)
    print("\nFigure 6: Exoplanet imaging results (4 panels)")
    print("Figure 7: Error budget analysis (4 panels)")
    print("\n✓ All figures saved successfully")
    print("=" * 70)
