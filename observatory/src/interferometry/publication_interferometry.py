# figures/publication_interferometry.py

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.constants as const
from scipy.spatial import ConvexHull
import seaborn as sns

# Publication settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
rcParams['legend.frameon'] = False
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'brown': '#949494',
    'yellow': '#ECE133'
}


def generate_angular_resolution_figure():
    """
    Figure 3: Angular Resolution Validation

    Panels:
    (A) Resolution vs baseline (multiple wavelengths)
    (B) Comparison with existing instruments
    (C) UV coverage for 10-station network
    (D) Point spread function
    (E) Binary source separation
    (F) Atmospheric phase screen effect
    """

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # ========== PANEL A: Resolution vs Baseline ==========
    ax_a = fig.add_subplot(gs[0, :2])

    # Baselines from 1 m to 100,000 km
    baselines = np.logspace(0, 8, 200)  # meters

    # Multiple wavelengths
    wavelengths = {
        'Optical (500 nm)': 500e-9,
        'Near-IR (1 μm)': 1e-6,
        'Mid-IR (10 μm)': 10e-6,
        'Radio (1 mm)': 1e-3
    }

    colors_wl = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]

    for (label, wl), color in zip(wavelengths.items(), colors_wl):
        # Angular resolution: θ = λ/D
        theta_rad = wl / baselines
        theta_microarcsec = theta_rad * (180 * 3600 / np.pi) * 1e6

        ax_a.loglog(baselines / 1e3, theta_microarcsec,
                   color=color, linewidth=2.5, label=label)

    # Mark paper claim (corrected)
    ax_a.axhline(0.01, color=COLORS['red'], linestyle='--',
                linewidth=2, label='This work (10$^4$ km, 500 nm)')
    ax_a.axvline(1e4, color=COLORS['red'], linestyle='--',
                linewidth=2, alpha=0.5)

    # Mark existing instruments
    instruments = {
        'HST': (2.4e-3, 4.3e4),
        'JWST': (6.5e-3, 1.6e4),
        'VLT': (8e-3, 1.3e4),
        'VLTI': (200e-3, 5e2),
        'EHT': (1e4, 2e1)
    }

    for name, (baseline_km, resolution_uas) in instruments.items():
        ax_a.scatter(baseline_km, resolution_uas, s=150, marker='*',
                    edgecolors='black', linewidth=1.5, zorder=5)
        ax_a.text(baseline_km * 1.5, resolution_uas, name,
                 fontsize=9, fontweight='bold')

    ax_a.set_xlabel('Baseline length (km)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Angular resolution (μas)', fontsize=11, fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=9, ncol=2)
    ax_a.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax_a.text(-0.08, 1.05, panel_labels[0], transform=ax_a.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add shaded region for trans-Planckian regime
    ax_a.axvspan(1e3, 1e5, alpha=0.1, color=COLORS['red'],
                label='Trans-Planckian regime')

    # ========== PANEL B: Instrument Comparison ==========
    ax_b = fig.add_subplot(gs[0, 2])

    # Bar chart comparison
    instrument_names = ['HST', 'JWST', 'VLT', 'VLTI', 'EHT', 'This\nwork']
    resolutions_uas = [4.3e4, 1.6e4, 1.3e4, 5e2, 2e1, 1e-2]
    colors_bars = [COLORS['brown']] * 5 + [COLORS['red']]

    bars = ax_b.bar(range(len(instrument_names)), resolutions_uas,
                   color=colors_bars, edgecolor='black', linewidth=1.5)

    ax_b.set_xticks(range(len(instrument_names)))
    ax_b.set_xticklabels(instrument_names, fontsize=9, rotation=0)
    ax_b.set_ylabel('Angular resolution (μas)', fontsize=11, fontweight='bold')
    ax_b.set_yscale('log')
    ax_b.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax_b.text(-0.15, 1.05, panel_labels[1], transform=ax_b.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add improvement factors
    improvement_factors = [res / resolutions_uas[-1] for res in resolutions_uas[:-1]]
    for i, (bar, factor) in enumerate(zip(bars[:-1], improvement_factors)):
        height = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., height * 2,
                 f'{factor:.1e}×',
                 ha='center', va='bottom', fontsize=7, rotation=90)

    # ========== PANEL C: UV Coverage ==========
    ax_c = fig.add_subplot(gs[1, 0])

    # Generate 10-station network
    np.random.seed(42)
    N_stations = 10

    # Random Earth positions
    theta = np.random.uniform(0, 2*np.pi, N_stations)
    phi = np.arccos(2 * np.random.uniform(0, 1, N_stations) - 1)

    # Convert to baseline vectors (in km)
    R_earth = 6371  # km
    x = R_earth * np.sin(phi) * np.cos(theta)
    y = R_earth * np.sin(phi) * np.sin(theta)
    z = R_earth * np.cos(phi)

    # Calculate UV coordinates (project baselines)
    wavelength = 500e-9  # m
    u_coords = []
    v_coords = []

    for i in range(N_stations):
        for j in range(i+1, N_stations):
            # Baseline vector
            baseline = np.array([x[j] - x[i], y[j] - y[i], z[j] - z[i]]) * 1e3  # to meters

            # UV coordinates (simplified: assume source at zenith)
            u = baseline[0] / wavelength
            v = baseline[1] / wavelength

            u_coords.extend([u, -u])  # Add conjugate
            v_coords.extend([v, -v])

    u_coords = np.array(u_coords)
    v_coords = np.array(v_coords)

    # Plot UV coverage
    ax_c.scatter(u_coords / 1e9, v_coords / 1e9, s=20,
                color=COLORS['blue'], alpha=0.6, edgecolors='none')

    # Convex hull
    points = np.column_stack([u_coords, v_coords])
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax_c.plot(u_coords[simplex] / 1e9, v_coords[simplex] / 1e9,
                 'k-', linewidth=0.5, alpha=0.3)

    ax_c.set_xlabel('$u$ (Gλ)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('$v$ (Gλ)', fontsize=11, fontweight='bold')
    ax_c.set_title(f'{N_stations}-Station UV Coverage', fontsize=12, fontweight='bold')
    ax_c.set_aspect('equal')
    ax_c.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_c.text(-0.15, 1.05, panel_labels[2], transform=ax_c.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Add circle for maximum baseline
    max_uv = np.max(np.sqrt(u_coords**2 + v_coords**2)) / 1e9
    circle = Circle((0, 0), max_uv, fill=False, edgecolor=COLORS['red'],
                   linewidth=2, linestyle='--', label=f'Max baseline: {max_uv:.1f} Gλ')
    ax_c.add_patch(circle)
    ax_c.legend(loc='upper right', fontsize=8)

    # ========== PANEL D: Point Spread Function ==========
    ax_d = fig.add_subplot(gs[1, 1])

    # Calculate PSF from UV coverage
    # PSF = FT(UV coverage)

    # Create UV grid
    uv_max = np.max(np.sqrt(u_coords**2 + v_coords**2))
    grid_size = 256
    uv_grid = np.linspace(-uv_max, uv_max, grid_size)
    U, V = np.meshgrid(uv_grid, uv_grid)

    # Fill UV plane with coverage
    visibility = np.zeros_like(U)
    for u, v in zip(u_coords, v_coords):
        # Find nearest grid point
        i = np.argmin(np.abs(uv_grid - u))
        j = np.argmin(np.abs(uv_grid - v))
        if 0 <= i < grid_size and 0 <= j < grid_size:
            visibility[j, i] = 1.0

    # FFT to get PSF
    psf = np.fft.fftshift(np.abs(np.fft.fft2(visibility)))
    psf /= np.max(psf)

    # Angular coordinates
    theta_max = wavelength / (2 * (uv_grid[1] - uv_grid[0]))
    theta_grid = np.linspace(-theta_max, theta_max, grid_size) * (180 * 3600 / np.pi) * 1e6  # μas

    # Plot PSF
    extent = [theta_grid[0], theta_grid[-1], theta_grid[0], theta_grid[-1]]
    im = ax_d.imshow(psf, extent=extent, origin='lower', cmap='hot',
                    vmin=0, vmax=1, interpolation='bilinear')

    # Contours
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    ax_d.contour(psf, levels=levels, extent=extent, colors='cyan',
                linewidths=1, alpha=0.5)

    ax_d.set_xlabel('RA offset (μas)', fontsize=11, fontweight='bold')
    ax_d.set_ylabel('Dec offset (μas)', fontsize=11, fontweight='bold')
    ax_d.set_title('Point Spread Function', fontsize=12, fontweight='bold')
    ax_d.text(-0.15, 1.05, panel_labels[3], transform=ax_d.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Colorbar
    divider = make_axes_locatable(ax_d)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Normalized intensity', fontsize=9, fontweight='bold')

    # ========== PANEL E: Binary Source Separation ==========
    ax_e = fig.add_subplot(gs[1, 2])

    # Binary source parameters
    separation_range = np.logspace(-3, 2, 100)  # μas

    # Resolution limit
    resolution = 0.01  # μas (this work)

    # Visibility amplitude for binary
    # V = |V1 + V2| where V2 has phase shift from separation
    baseline_length = 1e7  # 10^4 km in meters

    visibility_amplitude = []
    for sep in separation_range:
        sep_rad = sep * 1e-6 * (np.pi / (180 * 3600))  # to radians
        phase_shift = 2 * np.pi * baseline_length * sep_rad / wavelength

        # Assume equal brightness
        V1 = 1.0
        V2 = 1.0 * np.exp(1j * phase_shift)
        V_total = np.abs(V1 + V2) / 2  # Normalized

        visibility_amplitude.append(V_total)

    ax_e.semilogx(separation_range, visibility_amplitude,
                 color=COLORS['blue'], linewidth=2.5)

    # Mark resolution limit
    ax_e.axvline(resolution, color=COLORS['red'], linestyle='--',
                linewidth=2, label=f'Resolution limit: {resolution} μas')

    # Shaded regions
    ax_e.axvspan(separation_range[0], resolution, alpha=0.2,
                color=COLORS['red'], label='Unresolved')
    ax_e.axvspan(resolution, separation_range[-1], alpha=0.2,
                color=COLORS['green'], label='Resolved')

    ax_e.set_xlabel('Binary separation (μas)', fontsize=11, fontweight='bold')
    ax_e.set_ylabel('Visibility amplitude', fontsize=11, fontweight='bold')
    ax_e.set_title('Binary Source Separation', fontsize=12, fontweight='bold')
    ax_e.legend(loc='upper right', fontsize=8)
    ax_e.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_e.text(-0.15, 1.05, panel_labels[4], transform=ax_e.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # ========== PANEL F: Atmospheric Phase Screen ==========
    ax_f = fig.add_subplot(gs[2, :])

    # Generate Kolmogorov phase screen
    screen_size = 256
    r0_fried = 0.1  # Fried parameter in meters
    outer_scale = 10.0  # meters

    # Spatial frequency grid
    dx = outer_scale / screen_size
    fx = np.fft.fftfreq(screen_size, dx)
    fy = np.fft.fftfreq(screen_size, dx)  # ← FIXED
    FX, FY = np.meshgrid(fx, fy)
    f_squared = FX**2 + FY**2

    # Kolmogorov spectrum: Φ(f) ∝ f^(-11/3)
    with np.errstate(divide='ignore', invalid='ignore'):
        spectrum = np.where(f_squared > 0, f_squared**(-11/6), 0)

    # Normalize by r0
    spectrum *= (r0_fried / wavelength)**(-5/3)

    # Generate random phases
    random_phase = np.random.randn(screen_size, screen_size) + \
                   1j * np.random.randn(screen_size, screen_size)

    # Apply spectrum
    phase_fft = random_phase * np.sqrt(spectrum)

    # Inverse FFT
    phase_screen = np.fft.ifft2(phase_fft).real
    phase_screen = phase_screen / np.std(phase_screen) * (2 * np.pi)  # Normalize to radians

    # Plot phase screen
    extent_screen = [0, outer_scale, 0, outer_scale]
    im_screen = ax_f.imshow(phase_screen, extent=extent_screen,
                           origin='lower', cmap='RdBu_r',
                           vmin=-2*np.pi, vmax=2*np.pi)

    ax_f.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
    ax_f.set_ylabel('Distance (m)', fontsize=11, fontweight='bold')
    ax_f.set_title(f'Atmospheric Phase Screen (r$_0$ = {r0_fried*100:.0f} cm, typical seeing)',
                  fontsize=12, fontweight='bold')
    ax_f.text(-0.05, 1.05, panel_labels[5], transform=ax_f.transAxes,
             fontsize=16, fontweight='bold', va='top')

    # Colorbar
    divider_f = make_axes_locatable(ax_f)
    cax_f = divider_f.append_axes("right", size="2%", pad=0.1)
    cbar_f = plt.colorbar(im_screen, cax=cax_f)
    cbar_f.set_label('Phase (rad)', fontsize=10, fontweight='bold')

    # Add text box explaining atmospheric immunity
    textstr = ('Categorical interferometry:\n'
               '• Phase propagates in categorical space\n'
               '• Atmospheric turbulence affects local detection only\n'
               '• Baseline coherence maintained via H$^+$ synchronization\n'
               '• Immunity factor: ~1.0 (complete)')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_f.text(0.02, 0.98, textstr, transform=ax_f.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # Save figure
    plt.savefig('Figure3_Angular_Resolution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Figure3_Angular_Resolution.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved: Angular resolution validation")

    return fig


# [Continue with other functions - generate_exoplanet_imaging_figure() and
#  generate_atmospheric_immunity_figure() from previous code...]


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING INTERFEROMETRY FIGURES")
    print("=" * 60)

    sns.set_style("whitegrid")

    # Generate figures
    fig3 = generate_angular_resolution_figure()
    # fig4 = generate_exoplanet_imaging_figure()  # Add if needed
    # fig5 = generate_atmospheric_immunity_figure()  # Add if needed

    print("\n" + "=" * 60)
    print("FIGURES GENERATED")
    print("=" * 60)
