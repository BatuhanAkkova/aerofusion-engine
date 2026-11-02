use aerofusion_core::{EkfConfig, EkfEngine, ImuMeasurement, MagMeasurement, Vec3};
use std::time::Instant;

fn main() {
    println!("===============================================================");
    println!("   Aerofusion Core Microsecond Filter Latency Benchmark        ");
    println!("===============================================================");

    let mut filter = EkfEngine::new(EkfConfig::default());
    let dt = 0.0025; // 400 Hz
    let imu = ImuMeasurement {
        accel: Vec3::new(0.1, -0.05, 9.80665),
        gyro: Vec3::new(0.02, -0.01, 0.05),
        dt,
    };
    let mag = MagMeasurement {
        mag: Vec3::new(0.8, -0.1, 0.5),
    };

    // Warmup
    for _ in 0..5000 {
        filter.step(imu, Some(mag));
    }

    const ITERATIONS: usize = 100_000;
    let mut latencies_ns = Vec::with_capacity(ITERATIONS);

    // Benchmark 1: predict() alone
    let start_predict = Instant::now();
    for _ in 0..ITERATIONS {
        filter.predict(imu);
    }
    let dur_predict = start_predict.elapsed();
    let avg_predict_ns = dur_predict.as_nanos() as f64 / ITERATIONS as f64;

    // Benchmark 2: Full 9-DoF step (predict + accel update + mag update)
    let start_step = Instant::now();
    for _ in 0..ITERATIONS {
        let t0 = Instant::now();
        filter.step(imu, Some(mag));
        latencies_ns.push(t0.elapsed().as_nanos() as f64);
    }
    let dur_step = start_step.elapsed();
    let avg_step_ns = dur_step.as_nanos() as f64 / ITERATIONS as f64;

    latencies_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_ns[ITERATIONS / 2];
    let p95 = latencies_ns[(ITERATIONS as f64 * 0.95) as usize];
    let p99 = latencies_ns[(ITERATIONS as f64 * 0.99) as usize];
    let max = latencies_ns[ITERATIONS - 1];
    let min = latencies_ns[0];

    println!("Iterations Sampled   : {}", ITERATIONS);
    println!("predict() Avg Latency: {:8.2} ns ({:.3} µs)", avg_predict_ns, avg_predict_ns / 1000.0);
    println!("9-DoF Full Step Avg  : {:8.2} ns ({:.3} µs)", avg_step_ns, avg_step_ns / 1000.0);
    println!("---------------- Latency Distribution (9-DoF) ----------------");
    println!("Min Latency          : {:8.2} ns ({:.3} µs)", min, min / 1000.0);
    println!("Median (p50)         : {:8.2} ns ({:.3} µs)", p50, p50 / 1000.0);
    println!("95th Percentile (p95): {:8.2} ns ({:.3} µs)", p95, p95 / 1000.0);
    println!("99th Percentile (p99): {:8.2} ns ({:.3} µs)", p99, p99 / 1000.0);
    println!("Max Latency          : {:8.2} ns ({:.3} µs)", max, max / 1000.0);
    println!("Max Throughput Rate  : {:.0} Hz ({:.2} kHz)", 1e9 / avg_step_ns, (1e9 / avg_step_ns) / 1000.0);
    println!("===============================================================");
}
