use clap::Parser;
use smarts_evolution::FileLogConfig;

use npc_smarts::{ExperimentConfig, run_experiment};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExperimentConfig::parse();
    std::fs::create_dir_all(&config.output_dir)?;
    let slow_smarts_log_path = config.output_dir.join("slow-smarts.log");
    FileLogConfig::new(slow_smarts_log_path.clone())
        .append(false)
        .init()?;
    let summary = run_experiment(&config).await?;
    println!(
        "completed {} tasks, skipped {} tasks | results={} | summary={} | slow_smarts={}",
        summary.completed_tasks,
        summary.skipped_tasks,
        summary.results_path.display(),
        summary.output_dir.join("summary.json").display(),
        slow_smarts_log_path.display(),
    );
    Ok(())
}
