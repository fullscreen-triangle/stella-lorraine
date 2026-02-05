//! CatCount CLI - Trans-Planckian Temporal Resolution Calculator
//!
//! Command-line interface for the categorical state counting framework.

use clap::{Parser, Subcommand, Args};
use catcount::{
    EnhancementChain, TemporalResolution,
    TripleEquivalence,
    constants::*,
    resolution::{resolve, MultiScaleResolution},
    validation::{ValidationSuite, QuickValidation},
    spectroscopy::{SpectroscopicDatabase, SpectroscopicValidation},
    triple_equivalence::TripleEquivalenceValidation,
    memory::{CategoricalMemory, CategoricalAddress, MemoryTier, PrecisionByDifference},
    demon::{MaxwellDemon, TripleEquivalenceController},
    s_entropy::SEntropyCoord,
};

/// CatCount: Trans-Planckian Temporal Resolution via Categorical State Counting
#[derive(Parser)]
#[command(name = "catcount")]
#[command(author = "Stella Lorraine Framework")]
#[command(version)]
#[command(about = "Calculate categorical temporal resolution beyond the Planck scale")]
#[command(long_about = "
CatCount implements the trans-Planckian temporal resolution framework through
categorical state counting in bounded phase space. The framework achieves
temporal resolution of δt = 6.03×10⁻¹⁶⁵ s, approximately 121 orders of
magnitude below the Planck time.

The core formula is: δt = t_P / (E × (ν/ν_P))

where:
  t_P = Planck time (5.391×10⁻⁴⁴ s)
  E   = Total enhancement factor (10^120.95)
  ν   = Process frequency
  ν_P = Planck frequency (1.855×10⁴³ Hz)
")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate temporal resolution at a given frequency
    Resolve(ResolveArgs),

    /// Show enhancement chain breakdown
    Enhancement(EnhancementArgs),

    /// Calculate triple equivalence S_osc = S_cat = S_part
    Equivalence(EquivalenceArgs),

    /// Run validation suite
    Validate(ValidateArgs),

    /// Show physical constants
    Constants,

    /// Quick demonstration of framework capabilities
    Demo,

    /// Categorical memory system demonstration
    Memory(MemoryArgs),

    /// Maxwell demon controller demonstration
    Demon(DemonArgs),
}

#[derive(Args)]
struct ResolveArgs {
    /// Frequency in Hz (or use --wavenumber, --ev, --temperature)
    #[arg(short, long)]
    frequency: Option<f64>,

    /// Wavenumber in cm⁻¹
    #[arg(short, long)]
    wavenumber: Option<f64>,

    /// Energy in eV
    #[arg(short, long)]
    ev: Option<f64>,

    /// Temperature in K
    #[arg(short, long)]
    temperature: Option<f64>,

    /// Use standard reference frequencies
    #[arg(long)]
    standard: bool,

    /// Disable enhancement chain (raw Planck-limited resolution)
    #[arg(long)]
    no_enhancement: bool,
}

#[derive(Args)]
struct EnhancementArgs {
    /// Show detailed breakdown
    #[arg(short, long)]
    detailed: bool,

    /// Show only active mechanisms
    #[arg(long)]
    active_only: bool,
}

#[derive(Args)]
struct EquivalenceArgs {
    /// Number of oscillators (M)
    #[arg(short = 'M', long, default_value = "5")]
    oscillators: u64,

    /// Number of states per oscillator (n)
    #[arg(short = 'n', long, default_value = "4")]
    states: u64,

    /// Run validation across range
    #[arg(long)]
    validate: bool,
}

#[derive(Args)]
struct ValidateArgs {
    /// Run quick validation only
    #[arg(short, long)]
    quick: bool,

    /// Include spectroscopic validation
    #[arg(long)]
    spectroscopy: bool,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct MemoryArgs {
    /// Number of memory operations to simulate
    #[arg(short, long, default_value = "100")]
    operations: u32,

    /// Show detailed tier breakdown
    #[arg(short, long)]
    detailed: bool,
}

#[derive(Args)]
struct DemonArgs {
    /// Number of demon decisions to simulate
    #[arg(short, long, default_value = "50")]
    steps: u32,

    /// Oscillation frequency (Hz)
    #[arg(short, long, default_value = "1000000")]
    frequency: f64,

    /// Show trajectory details
    #[arg(short, long)]
    trajectory: bool,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Resolve(args) => cmd_resolve(args),
        Commands::Enhancement(args) => cmd_enhancement(args),
        Commands::Equivalence(args) => cmd_equivalence(args),
        Commands::Validate(args) => cmd_validate(args),
        Commands::Constants => cmd_constants(),
        Commands::Demo => cmd_demo(),
        Commands::Memory(args) => cmd_memory(args),
        Commands::Demon(args) => cmd_demon(args),
    }
}

fn cmd_resolve(args: ResolveArgs) {
    if args.standard {
        println!("Multi-Scale Resolution Analysis");
        println!("================================\n");

        let multi = MultiScaleResolution::standard();

        println!("{:<25} {:>15} {:>15} {:>12}",
            "Frequency", "Resolution (s)", "log10(δt)", "Orders < t_P");
        println!("{}", "-".repeat(70));

        for res in &multi.resolutions {
            println!("{:<25.3e} {:>15.3e} {:>15.2} {:>12.2}",
                res.frequency_hz, res.delta_t, res.log10_delta_t, res.orders_below_planck());
        }

        println!("\nScaling Law Validation:");
        println!("  Slope:     {:.6} (expected: -1.000)", multi.scaling_slope);
        println!("  R²:        {:.10}", multi.r_squared);
        println!("  Validated: {}", if multi.is_scaling_validated(1e-6) { "YES" } else { "NO" });

        return;
    }

    let frequency = if let Some(f) = args.frequency {
        f
    } else if let Some(w) = args.wavenumber {
        wavenumber_to_frequency(w)
    } else if let Some(e) = args.ev {
        ev_to_hz(e)
    } else if let Some(t) = args.temperature {
        temperature_to_frequency(t)
    } else {
        eprintln!("Error: Please specify a frequency (--frequency, --wavenumber, --ev, or --temperature)");
        eprintln!("       Or use --standard for reference frequencies");
        std::process::exit(1);
    };

    let resolution = if args.no_enhancement {
        TemporalResolution::without_enhancement(frequency)
    } else {
        resolve(frequency)
    };

    println!("{}", resolution);

    if resolution.is_trans_planckian {
        println!("\nThis resolution is {:.2} orders of magnitude below the Planck time!",
            resolution.orders_below_planck());
    }
}

fn cmd_enhancement(args: EnhancementArgs) {
    let chain = EnhancementChain::full();

    println!("Enhancement Chain Analysis");
    println!("==========================\n");

    println!("Five Enhancement Mechanisms:");
    println!();

    let breakdown = chain.breakdown();

    if args.detailed {
        println!("{:<25} {:>12} {:>15} {:>12}",
            "Mechanism", "log10(E)", "Cumulative", "Factor");
        println!("{}", "-".repeat(66));

        for b in &breakdown {
            println!("{:<25} {:>12.4} {:>15.4} {:>12.2e}",
                b.mechanism.name(),
                b.log10_enhancement,
                b.cumulative_log10,
                10.0_f64.powf(b.log10_enhancement));
        }
    } else {
        for b in &breakdown {
            println!("  {:2}. {:<22} : 10^{:.2}",
                match b.mechanism {
                    catcount::EnhancementMechanism::Ternary => 1,
                    catcount::EnhancementMechanism::MultiModal => 2,
                    catcount::EnhancementMechanism::Harmonic => 3,
                    catcount::EnhancementMechanism::Poincare => 4,
                    catcount::EnhancementMechanism::Refinement => 5,
                },
                b.mechanism.name(),
                b.log10_enhancement);
        }
    }

    println!();
    println!("Total Enhancement: 10^{:.2}", chain.total_log10());
    println!("                 = {:.2e}", chain.total_enhancement());
    println!();
    println!("Theoretical value: 10^{:.2}", catcount::enhancement::THEORETICAL_TOTAL_LOG10);
}

fn cmd_equivalence(args: EquivalenceArgs) {
    if args.validate {
        println!("Triple Equivalence Validation");
        println!("=============================\n");

        let m_range: Vec<u64> = (1..=10).collect();
        let n_range: Vec<u64> = (2..=10).collect();
        let validation = TripleEquivalenceValidation::validate(&m_range, &n_range);
        let summary = validation.summary();

        println!("{}", summary);

        return;
    }

    let te = TripleEquivalence::calculate(args.oscillators, args.states);
    println!("{}", te);
}

fn cmd_validate(args: ValidateArgs) {
    if args.quick {
        let result = QuickValidation::run();

        if args.json {
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        } else {
            println!("{}", result);
        }

        if !result.all_passed {
            std::process::exit(1);
        }

        return;
    }

    println!("Running Full Validation Suite...\n");

    let suite = ValidationSuite::run();

    if args.json {
        println!("{}", serde_json::to_string_pretty(&suite).unwrap());
    } else {
        println!("{}", suite);
    }

    if args.spectroscopy {
        println!("\nSpectroscopic Validation");
        println!("========================\n");

        let mut db = SpectroscopicDatabase::standard();
        db.calculate_predictions(10, 100);

        let validation = SpectroscopicValidation::validate(&db, 1e-6);
        println!("{}", validation);
    }

    if !suite.all_passed {
        std::process::exit(1);
    }
}

fn cmd_constants() {
    println!("Physical Constants (CODATA 2018)");
    println!("=================================\n");

    println!("Fundamental Constants:");
    println!("  k_B  = {:.6e} J/K      (Boltzmann constant)", K_B);
    println!("  ℏ    = {:.6e} J·s     (Reduced Planck constant)", H_BAR);
    println!("  h    = {:.6e} J·s     (Planck constant)", H);
    println!("  c    = {:.6e} m/s      (Speed of light)", C);
    println!("  G    = {:.6e} m³/(kg·s²) (Gravitational constant)", G);
    println!();

    println!("Planck Units:");
    println!("  t_P  = {:.6e} s       (Planck time)", PLANCK_TIME);
    println!("  l_P  = {:.6e} m       (Planck length)", PLANCK_LENGTH);
    println!("  m_P  = {:.6e} kg      (Planck mass)", PLANCK_MASS);
    println!("  ν_P  = {:.6e} Hz      (Planck frequency)", PLANCK_FREQUENCY);
    println!("  T_P  = {:.6e} K       (Planck temperature)", PLANCK_TEMPERATURE);
    println!();

    println!("Reference Frequencies:");
    println!("  CO vibration:     {:.2e} Hz", FREQ_CO_VIBRATION);
    println!("  Lyman-α:          {:.2e} Hz", FREQ_LYMAN_ALPHA);
    println!("  Electron Compton: {:.2e} Hz", FREQ_ELECTRON_COMPTON);
    println!();

    println!("Enhancement Parameters:");
    println!("  Ternary levels:   {}", N_TERNARY_LEVELS);
    println!("  Modalities:       {}", N_MODALITIES);
    println!("  Measurements:     {}", N_MEASUREMENTS_PER_MODALITY);
    println!("  Harmonic coinc.:  {}", N_HARMONIC_COINCIDENCES);
    println!("  Poincaré states:  {}", N_POINCARE_STATES);
}

fn cmd_demo() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        TRANS-PLANCKIAN TEMPORAL RESOLUTION FRAMEWORK             ║");
    println!("║                    Demonstration Output                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Show the key result
    let chain = EnhancementChain::full();
    let best = resolve(PLANCK_FREQUENCY);

    println!("KEY RESULT:");
    println!("===========");
    println!("  Categorical temporal resolution: δt = {:.2e} s", best.delta_t);
    println!("  Orders below Planck time:        {:.2}", best.orders_below_planck());
    println!("  Enhancement factor:              10^{:.2}", chain.total_log10());
    println!();

    // Show the formula
    println!("CORE FORMULA:");
    println!("=============");
    println!("  δt = t_P / (E × (ν/ν_P))");
    println!();
    println!("  where:");
    println!("    t_P = {:.3e} s (Planck time)", PLANCK_TIME);
    println!("    E   = 10^{:.2} (enhancement)", chain.total_log10());
    println!("    ν_P = {:.3e} Hz (Planck frequency)", PLANCK_FREQUENCY);
    println!();

    // Show enhancement breakdown
    println!("ENHANCEMENT CHAIN:");
    println!("==================");
    for b in chain.breakdown() {
        println!("  {:<22}: 10^{:.2}", b.mechanism.name(), b.log10_enhancement);
    }
    println!("  ─────────────────────────────");
    println!("  {:<22}: 10^{:.2}", "TOTAL", chain.total_log10());
    println!();

    // Show triple equivalence
    println!("TRIPLE EQUIVALENCE:");
    println!("===================");
    println!("  S_osc = S_cat = S_part = k_B × M × ln(n)");
    println!();
    let te = TripleEquivalence::calculate(5, 4);
    println!("  Example: M=5 oscillators, n=4 states");
    println!("    S_osc  = {:.6e} J/K", te.s_osc);
    println!("    S_cat  = {:.6e} J/K", te.s_cat);
    println!("    S_part = {:.6e} J/K", te.s_part);
    println!("    Converged: {}", if te.converged { "YES" } else { "NO" });
    println!();

    // Show validation summary
    println!("VALIDATION STATUS:");
    println!("==================");
    let quick = QuickValidation::run();
    println!("  Triple equivalence: {}", if quick.triple_equivalence { "PASSED" } else { "FAILED" });
    println!("  Enhancement chain:  {}", if quick.enhancement_chain { "PASSED" } else { "FAILED" });
    println!("  Resolution:         {}", if quick.resolution { "PASSED" } else { "FAILED" });
    println!();
    println!("  Overall: {}", if quick.all_passed { "ALL TESTS PASSED" } else { "SOME TESTS FAILED" });
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("Use 'catcount --help' for more commands and options.");
}

fn cmd_memory(args: MemoryArgs) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           CATEGORICAL MEMORY SYSTEM DEMONSTRATION                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("The categorical memory system addresses data by trajectories through");
    println!("S-entropy space. Memory locations are not physical positions but");
    println!("paths through the 3^k hierarchical structure.\n");

    // Create memory system
    let mut memory = CategoricalMemory::new();

    // Create a precision-by-difference calculator
    let mut pbd = PrecisionByDifference::new(1.0);

    println!("Simulating {} memory operations...\n", args.operations);

    // Simulate memory operations
    for i in 0..args.operations {
        // Simulate timing measurement
        let t_local = 1.0 + (i as f64 * 0.001).sin() * 0.01;
        let delta_p = pbd.measure(t_local);

        // Create categorical address from trajectory
        let coord = SEntropyCoord::new_unchecked(
            (i as f64) * 1e-25,
            (i as f64 * 0.5).sin() * 1e-24,
            (i as f64 * 0.3).cos() * 1e-24,
        );

        let addr = CategoricalAddress::new(
            vec![coord],
            vec![delta_p],
        );

        // Write to memory
        let _ = memory.write(addr.clone(), 64 + (i as u64 % 256));

        // Occasionally read back
        if i % 5 == 0 {
            let _ = memory.read(&addr);
        }
    }

    println!("Memory Statistics");
    println!("=================");
    println!("{}", memory.summary());
    println!();

    if args.detailed {
        println!("Tier Breakdown");
        println!("==============");
        for tier in [MemoryTier::L1, MemoryTier::L2, MemoryTier::L3,
                     MemoryTier::RAM, MemoryTier::Storage] {
            let pressure = memory.categorical_pressure(tier);
            println!("  {}: pressure = {:.2e} (T = {} K)",
                tier, pressure, tier.temperature());
        }
        println!();
    }

    println!("Thermodynamic Properties:");
    println!("  Total entropy:      {:.6e} J/K", memory.total_entropy());
    println!("  L1 pressure:        {:.6e}", memory.categorical_pressure(MemoryTier::L1));
    println!("  RAM pressure:       {:.6e}", memory.categorical_pressure(MemoryTier::RAM));
    println!();

    println!("From ideal gas law: P = k_B × T × (M/V)");
    println!("The categorical pressure measures how 'full' each tier is.");
}

fn cmd_demon(args: DemonArgs) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              MAXWELL DEMON CONTROLLER DEMONSTRATION               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("The Maxwell demon operates in categorical space, not physical space.");
    println!("Key insight: [Ô_cat, Ô_phys] = 0 (they commute!)");
    println!("Therefore: Zero thermodynamic cost for categorical sorting.\n");

    // Create the triple equivalence controller
    let mut controller = TripleEquivalenceController::new(args.frequency);

    println!("Triple Equivalence Controller");
    println!("=============================");
    println!("  Frequency: {:.2e} Hz", args.frequency);
    println!("  Running {} steps...\n", args.steps);

    // Run the controller
    for _ in 0..args.steps {
        controller.tick(1e-9);

        // Also move the demon through S-entropy space
        let coord = SEntropyCoord::new_unchecked(
            controller.phase.cos() * 1e-23,
            controller.phase.sin() * 1e-23,
            0.0,
        );
        controller.demon.move_to(coord, controller.phase * 1e-6);
    }

    // Verify triple equivalence
    let verified = controller.verify_triple_equivalence();

    println!("Triple Equivalence Identity:");
    println!("  dM/dt = ω/(2π/M) = 1/⟨τ_p⟩");
    println!();
    println!("  Partitions:    {}", controller.partitions);
    println!("  Category rate: {:.2e} (dM/dt)", controller.category_rate());
    println!("  Avg τ_p:       {:.2e} s", controller.avg_partition_duration());
    println!("  Verified:      {}\n", if verified { "YES" } else { "NO" });

    println!("Maxwell Demon Statistics");
    println!("========================");
    println!("  Trajectory length:      {}", controller.demon.trajectory.len());
    println!("  Predictions made:       {}", controller.demon.stats.predictions_made);
    println!("  Sort operations:        {}", controller.demon.stats.sort_operations);
    println!("  Categorical work:       {:.6e} J", controller.demon.stats.categorical_work);
    println!("  Physical work:          {:.6e} J", controller.demon.stats.physical_work);
    println!("  Zero-cost demon:        {}\n", if controller.demon.stats.is_zero_cost() { "YES" } else { "NO" });

    println!("Thermodynamic Properties");
    println!("========================");
    println!("  Demon temperature:      {:.2e} K", controller.demon.categorical_temperature());
    println!("  Trajectory entropy:     {:.6e} J/K", controller.demon.trajectory_entropy());

    if args.trajectory {
        println!("\nTrajectory (first 10 points):");
        for (i, coord) in controller.demon.trajectory.iter().take(10).enumerate() {
            println!("  {}: S=({:.2e}, {:.2e}, {:.2e})",
                i, coord.s_k, coord.s_t, coord.s_e);
        }
    }

    println!();
    println!("The demon-aperture distinction:");
    println!("  Maxwell's demon: sorts by energy → requires erasure → k_B T ln 2 cost");
    println!("  Categorical aperture: sorts by partition → no erasure → ZERO cost");
}
