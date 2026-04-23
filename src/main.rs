use clap::Parser;

use npc_smarts::{ExperimentConfig, run_experiment};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExperimentConfig::parse();
    let summary = run_experiment(&config).await?;
    println!(
        "completed {} tasks, skipped {} tasks | results={} | summary={}",
        summary.completed_tasks,
        summary.skipped_tasks,
        summary.results_path.display(),
        summary.output_dir.join("summary.json").display(),
    );
    Ok(())
}
