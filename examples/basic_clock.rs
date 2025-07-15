/// Basic Clock Usage Example
///
/// **In Memory of Mrs. Stella-Lorraine Masunda**
///
/// This example demonstrates basic clock functionality of the Masunda
/// Temporal Coordinate Navigator, achieving ultra-precise timekeeping.
use std::error::Error;
use std::time::Duration;
use tokio;

use masunda_temporal_coordinate_navigator::config::system_config::SystemConfig;
use masunda_temporal_coordinate_navigator::output::clock_interface::ClockInterface;
use masunda_temporal_coordinate_navigator::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🕐 MASUNDA TEMPORAL COORDINATE NAVIGATOR");
    println!("   =====================================");
    println!("   Basic Clock Usage Example");
    println!("   In memory of Mrs. Stella-Lorraine Masunda");
    println!();

    // Initialize the system
    let config = SystemConfig::default();
    let navigator = MasundaNavigator::new(config).await?;

    // Create clock interface
    let clock = ClockInterface::default(Arc::new(navigator)).await?;

    println!("✅ Masunda Navigator initialized successfully");
    println!();

    // Get current time with ultra-precision
    println!("🕐 Getting current time...");
    let current_time = clock.now().await?;
    println!("   Current Time: {:?}", current_time);

    // Get time as nanoseconds
    let nanos = clock.now_nanos().await?;
    println!("   Nanoseconds since epoch: {}", nanos);

    // Get time with specific precision
    println!();
    println!("🎯 Testing different precision levels...");

    let precision_levels = vec![
        PrecisionLevel::High,
        PrecisionLevel::Ultra,
        PrecisionLevel::Target,
    ];

    for precision in precision_levels {
        let precise_time = clock.now_with_precision(precision.clone()).await?;
        println!("   {:?} precision: {:?}", precision, precise_time);
    }

    // Get current temporal coordinate
    println!();
    println!("🧭 Getting temporal coordinate...");
    let coordinate = clock.get_current_coordinate().await?;
    println!(
        "   Spatial: ({:.6e}, {:.6e}, {:.6e})",
        coordinate.spatial.x, coordinate.spatial.y, coordinate.spatial.z
    );
    println!(
        "   Temporal: {:.6e} ± {:.6e} seconds",
        coordinate.temporal.seconds, coordinate.temporal.uncertainty
    );
    println!(
        "   Precision Level: {:?}",
        coordinate.temporal.precision_level
    );

    // Check memorial significance
    if coordinate.memorial_significance.is_valid {
        println!(
            "   🌟 Memorial Significance: {:.2}% - Mrs. Masunda's memory honored",
            coordinate.memorial_significance.cosmic_significance * 100.0
        );
    }

    // Monitor clock stability
    println!();
    println!("⏰ Monitoring clock stability...");

    for i in 0..5 {
        let time = clock.now().await?;
        let coordinate = clock.get_current_coordinate().await?;

        println!(
            "   Measurement {}: {:.2e} second precision",
            i + 1,
            coordinate.temporal.uncertainty
        );

        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Get clock metrics
    println!();
    println!("📊 Clock Performance Metrics:");
    let metrics = clock.get_metrics().await;
    println!("   Time Requests: {}", metrics.time_requests);
    println!(
        "   Precision Measurements: {}",
        metrics.precision_measurements
    );
    println!(
        "   Average Response Time: {:?}",
        metrics.average_response_time
    );
    println!("   Memorial Validations: {}", metrics.memorial_validations);

    println!();
    println!("✅ Basic clock example completed successfully");
    println!("   Mrs. Stella-Lorraine Masunda's memory preserved through");
    println!("   mathematical precision in temporal coordinate navigation");

    Ok(())
}
