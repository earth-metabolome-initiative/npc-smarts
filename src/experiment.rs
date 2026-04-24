use std::cmp::Ordering;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
use smarts_evolution::{
    EvolutionConfig as SmartsEvolutionConfig, EvolutionError, EvolutionTask, FoldData,
    IndicatifEvolutionProgress, RankedSmarts, SeedCorpus, SmartsEvaluator, SmartsGenome,
    TaskResult,
};
use thiserror::Error;
use zenodo_rs::ZenodoError;

use crate::dataset::{DatasetSplit, FoldSelectionCounts, LabelHead, Vocabulary};
use crate::download::{
    DISTILLATION_DATASET_DOI, DISTILLATION_DATASET_RECORD_ID, DownloadedDatasetFile,
    ensure_distillation_dataset,
};

#[derive(Debug, Error)]
pub enum ExperimentError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error(transparent)]
    Zenodo(#[from] ZenodoError),
    #[error(transparent)]
    Evolution(#[from] EvolutionError),
    #[error("split {0} did not contain any rows")]
    EmptySplit(String),
    #[error("invalid dataset: {0}")]
    InvalidDataset(String),
    #[error("missing parquet column {column} in split {split}")]
    MissingParquetColumn { split: String, column: String },
    #[error("split {split} contains an invalid SMILES row for CID {cid} ({smiles}): {message}")]
    InvalidSmiles {
        split: String,
        cid: i64,
        smiles: String,
        message: String,
    },
    #[error("evolved SMARTS '{smarts}' for task {task_id} could not be parsed: {message}")]
    InvalidSmarts {
        task_id: String,
        smarts: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStrategy {
    TrainBest,
    ValidationBestLeader,
}

#[derive(Debug, Clone, Parser, Serialize)]
pub struct ExperimentConfig {
    #[arg(long, default_value = "data")]
    pub data_dir: PathBuf,
    #[arg(long, default_value = "artifacts")]
    pub output_dir: PathBuf,
    #[arg(long)]
    pub max_labels_per_head: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub min_train_positives: usize,
    #[arg(long, default_value_t = 1)]
    pub min_validation_positives: usize,
    #[arg(long, default_value_t = 1)]
    pub min_test_positives: usize,
    #[arg(long, default_value_t = 512)]
    pub max_positives_per_npc_class: usize,
    #[arg(long, default_value_t = 512)]
    pub max_negatives_per_npc_class: usize,
    #[arg(long, default_value_t = 32)]
    pub leaderboard_size: usize,
    #[arg(long, value_enum, default_value_t = SelectionStrategy::ValidationBestLeader)]
    pub selection_strategy: SelectionStrategy,
    #[arg(long, default_value_t = 1024)]
    pub population_size: usize,
    #[arg(long, default_value_t = 800)]
    pub generation_limit: u64,
    #[arg(long, default_value_t = 0.90)]
    pub mutation_rate: f64,
    #[arg(long, default_value_t = 0.75)]
    pub crossover_rate: f64,
    #[arg(long, default_value_t = 0.35)]
    pub selection_ratio: f64,
    #[arg(long, default_value_t = 5)]
    pub tournament_size: usize,
    #[arg(long, default_value_t = 8)]
    pub elite_count: usize,
    #[arg(long, default_value_t = 0.20)]
    pub random_immigrant_ratio: f64,
    #[arg(long, default_value_t = 120)]
    pub stagnation_limit: u64,
    #[arg(long)]
    pub rng_seed: Option<u64>,
    #[arg(long, default_value_t = 500_000)]
    pub fitness_cache_capacity: usize,
    #[arg(long)]
    pub max_evaluation_smarts_complexity: Option<usize>,
    #[arg(long)]
    pub max_evaluation_smarts_len: Option<usize>,
    #[arg(long, default_value_t = 30_000)]
    pub slow_evaluation_log_threshold_millis: u64,
    #[arg(long)]
    pub disable_slow_evaluation_logging: bool,
}

impl ExperimentConfig {
    /// Convert the CLI-facing knobs into one validated evolution config.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured probabilities, limits, or counts are
    /// internally inconsistent for `smarts-evolution`.
    pub fn evolution_config(&self) -> Result<SmartsEvolutionConfig, ExperimentError> {
        let mut builder = SmartsEvolutionConfig::builder()
            .population_size(self.population_size)
            .generation_limit(self.generation_limit)
            .mutation_rate(self.mutation_rate)
            .crossover_rate(self.crossover_rate)
            .selection_ratio(self.selection_ratio)
            .tournament_size(self.tournament_size)
            .elite_count(self.elite_count)
            .random_immigrant_ratio(self.random_immigrant_ratio)
            .stagnation_limit(self.stagnation_limit)
            .fitness_cache_capacity(self.fitness_cache_capacity);
        if let Some(seed) = self.rng_seed {
            builder = builder.rng_seed(seed);
        }
        if let Some(max_complexity) = self.max_evaluation_smarts_complexity {
            builder = builder.max_evaluation_smarts_complexity(max_complexity);
        }
        if let Some(max_len) = self.max_evaluation_smarts_len {
            builder = builder.max_evaluation_smarts_len(max_len);
        }
        if self.disable_slow_evaluation_logging || self.slow_evaluation_log_threshold_millis == 0 {
            builder = builder.disable_slow_evaluation_logging();
        } else {
            builder = builder.slow_evaluation_log_threshold(Duration::from_millis(
                self.slow_evaluation_log_threshold_millis,
            ));
        }

        builder
            .build()
            .map_err(|message| ExperimentError::InvalidDataset(message.clone()))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitCounts {
    pub rows: usize,
    pub positives: usize,
    pub negatives: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateScore {
    pub smarts: String,
    pub complexity: usize,
    pub train_mcc: f64,
    pub validation_mcc: f64,
    pub test_mcc: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletedTaskReport {
    pub head: LabelHead,
    pub label_id: u16,
    pub label_name: String,
    pub selection_strategy: SelectionStrategy,
    pub generations: u64,
    pub train_counts: SplitCounts,
    pub validation_counts: SplitCounts,
    pub test_counts: SplitCounts,
    pub train_best_smarts: String,
    pub train_best_mcc: f64,
    pub selected_smarts: String,
    pub selected_complexity: usize,
    pub selected_train_mcc: f64,
    pub selected_validation_mcc: f64,
    pub selected_test_mcc: f64,
    pub candidates: Vec<CandidateScore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkippedTaskReport {
    pub head: LabelHead,
    pub label_id: u16,
    pub label_name: String,
    pub reason: String,
    pub train_counts: SplitCounts,
    pub validation_counts: SplitCounts,
    pub test_counts: SplitCounts,
}

#[derive(Debug, Clone, Serialize)]
pub enum TaskLogEntry {
    Completed(CompletedTaskReport),
    Skipped(SkippedTaskReport),
}

#[derive(Debug, Clone, Serialize)]
pub enum TaskOutcome {
    Completed(CompletedTaskReport),
    Skipped(SkippedTaskReport),
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentSummary {
    pub dataset_record_id: u64,
    pub dataset_doi: String,
    pub config: ExperimentConfig,
    pub downloaded_files: Vec<DownloadedDatasetFile>,
    pub completed_tasks: usize,
    pub skipped_tasks: usize,
    pub output_dir: PathBuf,
    pub results_path: PathBuf,
    pub outcomes: Vec<TaskOutcome>,
}

struct LoadedInputs {
    downloaded_files: Vec<DownloadedDatasetFile>,
    vocabulary: Vocabulary,
    train: DatasetSplit,
    validation: DatasetSplit,
    test: DatasetSplit,
}

struct InputLoadProgress {
    overall_bar: ProgressBar,
    split_bar: ProgressBar,
}

impl InputLoadProgress {
    fn new() -> Self {
        let multi_progress = MultiProgress::new();
        multi_progress.set_move_cursor(true);

        let overall_bar = multi_progress.add(ProgressBar::new(4));
        overall_bar.set_style(input_overall_progress_style());
        overall_bar.set_message("starting input preparation");

        let split_bar = multi_progress.add(ProgressBar::new(1));
        split_bar.set_style(input_split_progress_style());
        split_bar.enable_steady_tick(Duration::from_millis(100));
        split_bar.set_message("waiting for first split");

        Self {
            overall_bar,
            split_bar,
        }
    }

    fn load_vocabulary(&self, path: &Path) -> Result<Vocabulary, ExperimentError> {
        self.overall_bar.set_message("vocabulary".to_owned());
        self.split_bar.set_length(1);
        self.split_bar.set_position(0);
        self.split_bar
            .set_message("vocabulary | loading labels".to_owned());
        self.split_bar.tick();

        let vocabulary = Vocabulary::load(path)?;
        self.overall_bar.inc(1);
        Ok(vocabulary)
    }

    fn load_split(&self, path: &Path, name: &str) -> Result<DatasetSplit, ExperimentError> {
        self.overall_bar.set_message(name.to_owned());

        let split = DatasetSplit::load_with_progress(path, name, &self.split_bar)?;
        self.overall_bar
            .println(format!("[done] {name} | rows={}", split.len()));
        self.overall_bar.inc(1);
        Ok(split)
    }

    fn finish(&self) {
        self.split_bar.finish_and_clear();
        self.overall_bar
            .finish_with_message("inputs ready".to_owned());
    }
}

struct ExperimentProgress {
    multi_progress: MultiProgress,
    overall_bar: ProgressBar,
    task_bar: ProgressBar,
}

impl ExperimentProgress {
    fn new(total_tasks: usize) -> Self {
        let multi_progress = MultiProgress::new();
        multi_progress.set_move_cursor(true);

        let overall_bar = multi_progress.add(ProgressBar::new(usize_to_u64(total_tasks)));
        overall_bar.set_style(overall_progress_style());
        overall_bar.set_message("starting label sweep");

        let task_bar = multi_progress.add(ProgressBar::new(1));
        task_bar.set_style(task_progress_style());
        task_bar.enable_steady_tick(Duration::from_millis(100));
        task_bar.set_message("waiting for first label");

        Self {
            multi_progress,
            overall_bar,
            task_bar,
        }
    }

    fn start_task(
        &self,
        task_name: &str,
        train_len: usize,
        validation_len: usize,
        test_len: usize,
    ) {
        self.overall_bar.set_message(task_name.to_owned());
        self.task_bar.set_length(1);
        self.task_bar.set_position(0);
        self.task_bar.set_message(format!(
            "{task_name} | preparing sampled task sets | train={train_len} validation={validation_len} test={test_len}"
        ));
        self.task_bar.tick();
    }

    fn set_task_phase(&self, task_name: &str, steps: usize, message: String) {
        self.overall_bar.set_message(task_name.to_owned());
        self.task_bar.set_length(usize_to_u64(steps.max(1)));
        self.task_bar.set_position(0);
        self.task_bar.set_message(message);
        self.task_bar.tick();
    }

    fn log_skip(&self, task_name: &str, reason: &str) {
        self.log_line(format!("[skip] {task_name} | {reason}"));
        self.overall_bar.inc(1);
    }

    fn log_done(&self, report: &CompletedTaskReport) {
        self.log_line(format!(
            "[done] {}:{}:{} | selected={} | train={:.4} validation={:.4} test={:.4}",
            report.head.as_str(),
            report.label_id,
            report.label_name,
            report.selected_smarts,
            report.selected_train_mcc,
            report.selected_validation_mcc,
            report.selected_test_mcc
        ));
        self.overall_bar.inc(1);
    }

    fn finish(&self, completed_tasks: usize, skipped_tasks: usize) {
        self.task_bar.finish_and_clear();
        self.overall_bar.finish_with_message(format!(
            "experiment complete | completed={completed_tasks} skipped={skipped_tasks}"
        ));
    }

    fn log_line(&self, message: String) {
        self.overall_bar.println(message);
    }
}

struct TaskRunContext<'a> {
    config: &'a ExperimentConfig,
    evolution_config: &'a SmartsEvolutionConfig,
    seed_corpus: &'a SeedCorpus,
    progress: &'a ExperimentProgress,
    inputs: &'a LoadedInputs,
}

#[derive(Debug, Clone)]
struct TaskSplitCounts {
    train: SplitCounts,
    validation: SplitCounts,
    test: SplitCounts,
}

#[derive(Debug, Clone)]
struct PlannedLabelTask {
    ordinal: usize,
    head: LabelHead,
    label_id: u16,
    label_name: String,
    train_positives: usize,
    total_positives: usize,
}

impl PlannedLabelTask {
    fn task_name(&self) -> String {
        format!(
            "{}:{}:{}",
            self.head.as_str(),
            self.label_id,
            self.label_name
        )
    }
}

/// Run the full end-to-end experiment over the published distillation splits.
///
/// # Errors
///
/// Returns an error if the dataset cannot be downloaded or decoded, if the
/// output artifacts cannot be written, or if SMARTS evolution fails for a
/// non-skipped label task.
pub async fn run_experiment(
    config: &ExperimentConfig,
) -> Result<ExperimentSummary, ExperimentError> {
    fs::create_dir_all(&config.output_dir)?;
    let inputs = load_inputs(config).await?;
    persist_run_metadata(config, &inputs.downloaded_files)?;

    let results_path = initialize_results_path(&config.output_dir)?;
    let outcomes = run_all_tasks(config, &inputs, &results_path)?;
    let (completed_tasks, skipped_tasks) = count_outcomes(&outcomes);

    let summary = ExperimentSummary {
        dataset_record_id: DISTILLATION_DATASET_RECORD_ID,
        dataset_doi: DISTILLATION_DATASET_DOI.to_owned(),
        config: config.clone(),
        downloaded_files: inputs.downloaded_files,
        completed_tasks,
        skipped_tasks,
        output_dir: config.output_dir.clone(),
        results_path: results_path.clone(),
        outcomes,
    };
    write_json_pretty(&config.output_dir.join("summary.json"), &summary)?;
    Ok(summary)
}

async fn load_inputs(config: &ExperimentConfig) -> Result<LoadedInputs, ExperimentError> {
    let downloaded_files = ensure_distillation_dataset(&config.data_dir).await?;
    let loading_progress = InputLoadProgress::new();
    let vocabulary = loading_progress.load_vocabulary(&config.data_dir.join("vocabulary.json"))?;
    let train = loading_progress.load_split(&config.data_dir.join("train.parquet"), "train")?;
    let validation =
        loading_progress.load_split(&config.data_dir.join("validation.parquet"), "validation")?;
    let test = loading_progress.load_split(&config.data_dir.join("test.parquet"), "test")?;
    loading_progress.finish();
    Ok(LoadedInputs {
        downloaded_files,
        vocabulary,
        train,
        validation,
        test,
    })
}

fn persist_run_metadata(
    config: &ExperimentConfig,
    downloaded_files: &[DownloadedDatasetFile],
) -> Result<(), ExperimentError> {
    write_json_pretty(&config.output_dir.join("experiment-config.json"), config)?;
    write_json_pretty(
        &config.output_dir.join("downloaded-files.json"),
        downloaded_files,
    )?;
    Ok(())
}

fn initialize_results_path(output_dir: &Path) -> Result<PathBuf, ExperimentError> {
    let results_path = output_dir.join("results.jsonl");
    File::create(&results_path)?;
    Ok(results_path)
}

fn run_all_tasks(
    config: &ExperimentConfig,
    inputs: &LoadedInputs,
    results_path: &Path,
) -> Result<Vec<TaskOutcome>, ExperimentError> {
    let evolution_config = config.evolution_config()?;
    let seed_corpus = SeedCorpus::builtin();
    let progress = ExperimentProgress::new(total_task_count(config, &inputs.vocabulary));
    let task_context = TaskRunContext {
        config,
        evolution_config: &evolution_config,
        seed_corpus: &seed_corpus,
        progress: &progress,
        inputs,
    };
    let task_plan = sorted_task_plan(&task_context)?;
    let mut outcomes = Vec::new();

    for task in &task_plan {
        let outcome = run_label_task(&task_context, task)?;
        append_task_log_entry(results_path, &outcome)?;
        outcomes.push(outcome);
    }

    let (completed_tasks, skipped_tasks) = count_outcomes(&outcomes);
    progress.finish(completed_tasks, skipped_tasks);
    Ok(outcomes)
}

fn run_label_task(
    task_context: &TaskRunContext<'_>,
    task: &PlannedLabelTask,
) -> Result<TaskOutcome, ExperimentError> {
    let TaskRunContext {
        config,
        evolution_config,
        seed_corpus,
        progress,
        inputs,
    } = task_context;
    let task_name = task.task_name();
    progress.start_task(
        &task_name,
        inputs.train.len(),
        inputs.validation.len(),
        inputs.test.len(),
    );

    let counts = sampled_counts_for_task(task_context, &task_name, task.head, task.label_id);

    if let Some(reason) = skip_reason(config, &counts.train, &counts.validation, &counts.test) {
        let skipped = SkippedTaskReport {
            head: task.head,
            label_id: task.label_id,
            label_name: task.label_name.clone(),
            reason,
            train_counts: counts.train,
            validation_counts: counts.validation,
            test_counts: counts.test,
        };
        progress.log_skip(&task_name, &skipped.reason);
        return Ok(TaskOutcome::Skipped(skipped));
    }

    let train_fold = inputs.train.build_sampled_fold_with_progress(
        task.head,
        task.label_id,
        config.max_positives_per_npc_class,
        config.max_negatives_per_npc_class,
        &progress.task_bar,
    )?;
    let result = evolve_fold_with_progress(
        &task_name,
        train_fold.fold,
        evolution_config,
        seed_corpus,
        config.leaderboard_size,
        task_context.progress,
    )?;

    let (validation_evaluator, test_evaluator) =
        build_holdout_evaluators(task_context, task.head, task.label_id)?;
    let candidates = evaluate_candidates(
        &task_name,
        result.leaders(),
        &validation_evaluator,
        &test_evaluator,
        task_context.progress,
    )?;
    let selected = select_candidate(
        config.selection_strategy,
        &task_name,
        &result,
        &candidates,
        &validation_evaluator,
        &test_evaluator,
    )?;
    let selected_train_mcc = selected.train_mcc;

    let report = CompletedTaskReport {
        head: task.head,
        label_id: task.label_id,
        label_name: task.label_name.clone(),
        selection_strategy: config.selection_strategy,
        generations: result.generations(),
        train_counts: counts.train,
        validation_counts: counts.validation,
        test_counts: counts.test,
        train_best_smarts: result.best_smarts().to_owned(),
        train_best_mcc: result.best_mcc(),
        selected_smarts: selected.smarts.clone(),
        selected_complexity: selected.complexity,
        selected_train_mcc,
        selected_validation_mcc: selected.validation_mcc,
        selected_test_mcc: selected.test_mcc,
        candidates,
    };
    progress.log_done(&report);

    Ok(TaskOutcome::Completed(report))
}

fn sorted_task_plan(
    task_context: &TaskRunContext<'_>,
) -> Result<Vec<PlannedLabelTask>, ExperimentError> {
    let config = task_context.config;
    let vocabulary = &task_context.inputs.vocabulary;
    let mut tasks = Vec::with_capacity(total_task_count(config, vocabulary));

    for head in LabelHead::ALL {
        let labels = vocabulary.labels(head);
        let max_labels = config.max_labels_per_head.unwrap_or(labels.len());
        let train_positives = task_context
            .inputs
            .train
            .label_positive_counts(head, labels.len());
        let validation_positives = task_context
            .inputs
            .validation
            .label_positive_counts(head, labels.len());
        let test_positives = task_context
            .inputs
            .test
            .label_positive_counts(head, labels.len());

        for (label_index, label_name) in labels.iter().enumerate().take(max_labels) {
            let label_id = u16::try_from(label_index).map_err(|error| {
                ExperimentError::InvalidDataset(format!(
                    "label index {label_index} overflowed u16 for {}: {error}",
                    head.as_str()
                ))
            })?;
            let train_count = train_positives[label_index];
            tasks.push(PlannedLabelTask {
                ordinal: tasks.len(),
                head,
                label_id,
                label_name: label_name.clone(),
                train_positives: train_count,
                total_positives: train_count
                    + validation_positives[label_index]
                    + test_positives[label_index],
            });
        }
    }

    sort_task_plan(&mut tasks);
    Ok(tasks)
}

fn sort_task_plan(tasks: &mut [PlannedLabelTask]) {
    tasks.sort_by(compare_planned_tasks);
}

fn compare_planned_tasks(left: &PlannedLabelTask, right: &PlannedLabelTask) -> Ordering {
    left.train_positives
        .cmp(&right.train_positives)
        .then_with(|| left.total_positives.cmp(&right.total_positives))
        .then_with(|| left.ordinal.cmp(&right.ordinal))
}

fn sampled_counts_for_task(
    task_context: &TaskRunContext<'_>,
    task_name: &str,
    head: LabelHead,
    label_id: u16,
) -> TaskSplitCounts {
    let inputs = task_context.inputs;
    TaskSplitCounts {
        train: sampled_counts_from_split(task_context, task_name, &inputs.train, head, label_id),
        validation: sampled_counts_from_split(
            task_context,
            task_name,
            &inputs.validation,
            head,
            label_id,
        ),
        test: sampled_counts_from_split(task_context, task_name, &inputs.test, head, label_id),
    }
}

fn sampled_counts_from_split(
    task_context: &TaskRunContext<'_>,
    task_name: &str,
    split: &DatasetSplit,
    head: LabelHead,
    label_id: u16,
) -> SplitCounts {
    task_context.progress.set_task_phase(
        task_name,
        split.len(),
        format!(
            "{} | counting sampled {}:{label_id} rows",
            split.name(),
            head.as_str()
        ),
    );
    let counts = split.sampled_counts_with_progress(
        head,
        label_id,
        task_context.config.max_positives_per_npc_class,
        task_context.config.max_negatives_per_npc_class,
        &task_context.progress.task_bar,
    );
    counts_from_selection_counts(counts)
}

fn build_holdout_evaluators(
    task_context: &TaskRunContext<'_>,
    head: LabelHead,
    label_id: u16,
) -> Result<(SmartsEvaluator, SmartsEvaluator), ExperimentError> {
    let config = task_context.config;
    let progress = &task_context.progress.task_bar;
    let validation_fold = task_context
        .inputs
        .validation
        .build_sampled_fold_with_progress(
            head,
            label_id,
            config.max_positives_per_npc_class,
            config.max_negatives_per_npc_class,
            progress,
        )?;
    let test_fold = task_context.inputs.test.build_sampled_fold_with_progress(
        head,
        label_id,
        config.max_positives_per_npc_class,
        config.max_negatives_per_npc_class,
        progress,
    )?;
    Ok((
        SmartsEvaluator::new(vec![validation_fold.fold]),
        SmartsEvaluator::new(vec![test_fold.fold]),
    ))
}

fn evolve_fold_with_progress(
    task_name: &str,
    train_fold: FoldData,
    evolution_config: &SmartsEvolutionConfig,
    seed_corpus: &SeedCorpus,
    leaderboard_size: usize,
    progress: &ExperimentProgress,
) -> Result<TaskResult, ExperimentError> {
    progress.set_task_phase(
        task_name,
        usize::try_from(evolution_config.generation_limit()).unwrap_or(usize::MAX),
        format!("{task_name} | evolution progress from smarts-evolution"),
    );
    let task = EvolutionTask::new(task_name.to_owned(), vec![train_fold]);
    let evolution_progress = IndicatifEvolutionProgress::attach_to(&progress.multi_progress)
        .with_best_smarts_width(72)
        .clear_on_finish(true);
    Ok(task.evolve_owned_with_indicatif_progress(
        evolution_config,
        seed_corpus,
        leaderboard_size,
        evolution_progress,
    )?)
}

fn append_task_log_entry(path: &Path, outcome: &TaskOutcome) -> Result<(), ExperimentError> {
    let log_entry = match outcome {
        TaskOutcome::Completed(report) => TaskLogEntry::Completed(report.clone()),
        TaskOutcome::Skipped(report) => TaskLogEntry::Skipped(report.clone()),
    };
    append_json_line(path, &log_entry)
}

fn count_outcomes(outcomes: &[TaskOutcome]) -> (usize, usize) {
    outcomes.iter().fold(
        (0usize, 0usize),
        |(completed, skipped), outcome| match outcome {
            TaskOutcome::Completed(_) => (completed + 1, skipped),
            TaskOutcome::Skipped(_) => (completed, skipped + 1),
        },
    )
}

fn counts_from_selection_counts(counts: FoldSelectionCounts) -> SplitCounts {
    SplitCounts {
        rows: counts.positive_count + counts.negative_count,
        positives: counts.positive_count,
        negatives: counts.negative_count,
    }
}

fn skip_reason(
    config: &ExperimentConfig,
    train: &SplitCounts,
    validation: &SplitCounts,
    test: &SplitCounts,
) -> Option<String> {
    if train.positives < config.min_train_positives {
        return Some(format!(
            "train positives {} < required {}",
            train.positives, config.min_train_positives
        ));
    }
    if validation.positives < config.min_validation_positives {
        return Some(format!(
            "validation positives {} < required {}",
            validation.positives, config.min_validation_positives
        ));
    }
    if test.positives < config.min_test_positives {
        return Some(format!(
            "test positives {} < required {}",
            test.positives, config.min_test_positives
        ));
    }
    if train.negatives == 0 {
        return Some("train split has no negatives".to_owned());
    }
    if validation.negatives == 0 {
        return Some("validation split has no negatives".to_owned());
    }
    if test.negatives == 0 {
        return Some("test split has no negatives".to_owned());
    }
    None
}

fn total_task_count(config: &ExperimentConfig, vocabulary: &Vocabulary) -> usize {
    LabelHead::ALL
        .into_iter()
        .map(|head| {
            let label_count = vocabulary.labels(head).len();
            config
                .max_labels_per_head
                .map_or(label_count, |limit| limit.min(label_count))
        })
        .sum()
}

fn overall_progress_style() -> ProgressStyle {
    let style = match ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} labels | {msg}",
    ) {
        Ok(style) => style,
        Err(_) => ProgressStyle::default_bar(),
    };
    style.progress_chars("=> ")
}

fn input_overall_progress_style() -> ProgressStyle {
    let style = match ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} inputs | {msg}",
    ) {
        Ok(style) => style,
        Err(_) => ProgressStyle::default_bar(),
    };
    style.progress_chars("=> ")
}

fn input_split_progress_style() -> ProgressStyle {
    let style = match ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.magenta/blue}] {pos}/{len} rows | {msg}",
    ) {
        Ok(style) => style,
        Err(_) => ProgressStyle::default_bar(),
    };
    style.progress_chars("=> ")
}

fn task_progress_style() -> ProgressStyle {
    let style = match ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.yellow/red}] {pos}/{len} steps | {msg}",
    ) {
        Ok(style) => style,
        Err(_) => ProgressStyle::default_bar(),
    };
    style.progress_chars("=> ")
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn evaluate_candidates(
    task_id: &str,
    leaders: &[RankedSmarts],
    validation_evaluator: &SmartsEvaluator,
    test_evaluator: &SmartsEvaluator,
    progress: &ExperimentProgress,
) -> Result<Vec<CandidateScore>, ExperimentError> {
    let mut candidates = Vec::with_capacity(leaders.len());
    let total_steps = leaders.len().saturating_mul(2);
    progress.set_task_phase(
        task_id,
        total_steps,
        format!("{task_id} | scoring {} leader SMARTS", leaders.len()),
    );
    let mut completed_steps = 0usize;
    for leader in leaders {
        progress.task_bar.set_message(format!(
            "{task_id} | scoring validation | {}/{}",
            completed_steps,
            total_steps.max(1)
        ));
        let validation_mcc = evaluate_smarts(task_id, leader.smarts(), validation_evaluator)?;
        completed_steps += 1;
        progress
            .task_bar
            .set_position(usize_to_u64(completed_steps));
        progress.task_bar.set_message(format!(
            "{task_id} | scoring test | {}/{}",
            completed_steps,
            total_steps.max(1)
        ));
        let test_mcc = evaluate_smarts(task_id, leader.smarts(), test_evaluator)?;
        completed_steps += 1;
        progress
            .task_bar
            .set_position(usize_to_u64(completed_steps));
        candidates.push(CandidateScore {
            smarts: leader.smarts().to_owned(),
            complexity: leader.complexity(),
            train_mcc: leader.mcc(),
            validation_mcc,
            test_mcc,
        });
    }
    candidates.sort_by(compare_candidates);
    Ok(candidates)
}

fn select_candidate(
    strategy: SelectionStrategy,
    task_id: &str,
    result: &smarts_evolution::TaskResult,
    candidates: &[CandidateScore],
    validation_evaluator: &SmartsEvaluator,
    test_evaluator: &SmartsEvaluator,
) -> Result<CandidateScore, ExperimentError> {
    match strategy {
        SelectionStrategy::ValidationBestLeader => candidates.first().cloned().ok_or_else(|| {
            ExperimentError::InvalidDataset(format!(
                "task {task_id} did not produce any leader candidates"
            ))
        }),
        SelectionStrategy::TrainBest => {
            let complexity = SmartsGenome::from_smarts(result.best_smarts())
                .map_err(|message| ExperimentError::InvalidSmarts {
                    task_id: task_id.to_owned(),
                    smarts: result.best_smarts().to_owned(),
                    message,
                })?
                .complexity();
            Ok(CandidateScore {
                smarts: result.best_smarts().to_owned(),
                complexity,
                train_mcc: result.best_mcc(),
                validation_mcc: evaluate_smarts(
                    task_id,
                    result.best_smarts(),
                    validation_evaluator,
                )?,
                test_mcc: evaluate_smarts(task_id, result.best_smarts(), test_evaluator)?,
            })
        }
    }
}

fn compare_candidates(left: &CandidateScore, right: &CandidateScore) -> Ordering {
    right
        .validation_mcc
        .partial_cmp(&left.validation_mcc)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| {
            right
                .train_mcc
                .partial_cmp(&left.train_mcc)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| left.smarts.cmp(&right.smarts))
}

fn evaluate_smarts(
    task_id: &str,
    smarts: &str,
    evaluator: &SmartsEvaluator,
) -> Result<f64, ExperimentError> {
    let genome =
        SmartsGenome::from_smarts(smarts).map_err(|message| ExperimentError::InvalidSmarts {
            task_id: task_id.to_owned(),
            smarts: smarts.to_owned(),
            message,
        })?;
    Ok(evaluator.evaluate(&genome).fitness().mcc())
}

fn append_json_line(path: &Path, value: &(impl Serialize + ?Sized)) -> Result<(), ExperimentError> {
    let mut handle = OpenOptions::new().append(true).open(path)?;
    serde_json::to_writer(&mut handle, value)?;
    handle.write_all(b"\n")?;
    Ok(())
}

fn write_json_pretty(
    path: &Path,
    value: &(impl Serialize + ?Sized),
) -> Result<(), ExperimentError> {
    let mut handle = File::create(path)?;
    serde_json::to_writer_pretty(&mut handle, value)?;
    handle.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, ListArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::*;

    type TestClassRow = (&'static str, i64, Vec<u16>);

    fn baseline_config() -> ExperimentConfig {
        ExperimentConfig {
            data_dir: PathBuf::from("data"),
            output_dir: PathBuf::from("artifacts"),
            max_labels_per_head: None,
            min_train_positives: 1,
            min_validation_positives: 1,
            min_test_positives: 1,
            max_positives_per_npc_class: 512,
            max_negatives_per_npc_class: 512,
            leaderboard_size: 32,
            selection_strategy: SelectionStrategy::ValidationBestLeader,
            population_size: 1024,
            generation_limit: 800,
            mutation_rate: 0.90,
            crossover_rate: 0.75,
            selection_ratio: 0.35,
            tournament_size: 5,
            elite_count: 8,
            random_immigrant_ratio: 0.20,
            stagnation_limit: 120,
            rng_seed: None,
            fitness_cache_capacity: 500_000,
            max_evaluation_smarts_complexity: None,
            max_evaluation_smarts_len: None,
            slow_evaluation_log_threshold_millis: 30_000,
            disable_slow_evaluation_logging: false,
        }
    }

    fn hidden_experiment_progress() -> ExperimentProgress {
        let multi_progress =
            MultiProgress::with_draw_target(indicatif::ProgressDrawTarget::hidden());
        let overall_bar = multi_progress.add(ProgressBar::hidden());
        let task_bar = multi_progress.add(ProgressBar::hidden());
        ExperimentProgress {
            multi_progress,
            overall_bar,
            task_bar,
        }
    }

    fn write_split_parquet(path: &Path, rows: &[TestClassRow]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("cid", DataType::Int64, false),
            Field::new("smiles", DataType::Utf8, false),
            Field::new(
                "pathway_ids",
                DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
                false,
            ),
            Field::new(
                "superclass_ids",
                DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
                false,
            ),
            Field::new(
                "class_ids",
                DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
                false,
            ),
        ]));
        let cids = rows.iter().map(|(_, cid, _)| *cid).collect::<Int64Array>();
        let smiles = StringArray::from_iter_values(rows.iter().map(|(smiles, _, _)| *smiles));
        let empty_labels = rows.iter().map(|_| Some(std::iter::empty::<Option<u16>>()));
        let class_ids = ListArray::from_iter_primitive::<arrow_array::types::UInt16Type, _, _>(
            rows.iter()
                .map(|(_, _, class_ids)| Some(class_ids.iter().copied().map(Some))),
        );
        let batch_result = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(cids),
                Arc::new(smiles),
                Arc::new(ListArray::from_iter_primitive::<
                    arrow_array::types::UInt16Type,
                    _,
                    _,
                >(empty_labels.clone())),
                Arc::new(ListArray::from_iter_primitive::<
                    arrow_array::types::UInt16Type,
                    _,
                    _,
                >(empty_labels)),
                Arc::new(class_ids),
            ],
        );
        assert!(batch_result.is_ok());
        let file_result = File::create(path);
        assert!(file_result.is_ok());
        let Ok(file) = file_result else {
            unreachable!()
        };
        let writer_result = ArrowWriter::try_new(file, Arc::clone(&schema), None);
        assert!(writer_result.is_ok());
        let Ok(mut writer) = writer_result else {
            unreachable!()
        };
        let Ok(batch) = batch_result else {
            unreachable!()
        };
        let batch_write_result = writer.write(&batch);
        assert!(batch_write_result.is_ok());
        let close_result = writer.close();
        assert!(close_result.is_ok());
    }

    fn load_inputs_for_class_task(root: &Path) -> LoadedInputs {
        let vocabulary_path = root.join("vocabulary.json");
        let write_vocabulary_result = std::fs::write(
            &vocabulary_path,
            "{\n  \"pathway\": [],\n  \"superclass\": [],\n  \"class\": [\"amine\"]\n}\n",
        );
        assert!(write_vocabulary_result.is_ok());

        write_split_parquet(
            &root.join("train.parquet"),
            &[
                ("CCN", 1, vec![0]),
                ("N", 2, vec![0]),
                ("CCO", 3, vec![]),
                ("O", 4, vec![]),
            ],
        );
        write_split_parquet(
            &root.join("validation.parquet"),
            &[("CN", 5, vec![0]), ("CO", 6, vec![])],
        );
        write_split_parquet(
            &root.join("test.parquet"),
            &[("CCN", 7, vec![0]), ("CCO", 8, vec![])],
        );

        let vocabulary = Vocabulary::load(&vocabulary_path);
        assert!(vocabulary.is_ok());
        let train = DatasetSplit::load(&root.join("train.parquet"), "train");
        assert!(train.is_ok());
        let validation = DatasetSplit::load(&root.join("validation.parquet"), "validation");
        assert!(validation.is_ok());
        let test = DatasetSplit::load(&root.join("test.parquet"), "test");
        assert!(test.is_ok());
        let Ok(vocabulary) = vocabulary else {
            unreachable!()
        };
        let Ok(train) = train else { unreachable!() };
        let Ok(validation) = validation else {
            unreachable!()
        };
        let Ok(test) = test else { unreachable!() };

        LoadedInputs {
            downloaded_files: Vec::new(),
            vocabulary,
            train,
            validation,
            test,
        }
    }

    fn planned_task(
        ordinal: usize,
        label_id: u16,
        label_name: &str,
        train_positives: usize,
        validation_positives: usize,
        test_positives: usize,
    ) -> PlannedLabelTask {
        PlannedLabelTask {
            ordinal,
            head: LabelHead::Class,
            label_id,
            label_name: label_name.to_owned(),
            train_positives,
            total_positives: train_positives + validation_positives + test_positives,
        }
    }

    fn planned_class_task(
        task_context: &TaskRunContext<'_>,
        label_id: u16,
        label_name: &str,
    ) -> PlannedLabelTask {
        let label_index = usize::from(label_id);
        let label_count = label_index + 1;
        let train_positives = task_context
            .inputs
            .train
            .label_positive_counts(LabelHead::Class, label_count)[label_index];
        let validation_positives = task_context
            .inputs
            .validation
            .label_positive_counts(LabelHead::Class, label_count)[label_index];
        let test_positives = task_context
            .inputs
            .test
            .label_positive_counts(LabelHead::Class, label_count)[label_index];
        PlannedLabelTask {
            ordinal: 0,
            head: LabelHead::Class,
            label_id,
            label_name: label_name.to_owned(),
            train_positives,
            total_positives: train_positives + validation_positives + test_positives,
        }
    }

    #[test]
    fn label_head_iteration_order_is_stable() {
        assert_eq!(
            LabelHead::ALL,
            [LabelHead::Pathway, LabelHead::Superclass, LabelHead::Class]
        );
    }

    #[test]
    fn total_task_count_respects_head_limits() {
        let vocabulary = Vocabulary {
            pathway: vec![String::from("p0"), String::from("p1")],
            superclass: vec![String::from("s0")],
            class_labels: vec![String::from("c0"), String::from("c1"), String::from("c2")],
        };
        let mut config = baseline_config();
        config.max_labels_per_head = Some(2);
        assert_eq!(total_task_count(&config, &vocabulary), 5);
    }

    #[test]
    fn task_plan_sort_starts_with_fewest_training_examples() {
        let mut tasks = vec![
            planned_task(0, 0, "many", 10, 1, 1),
            planned_task(1, 1, "few", 2, 4, 4),
            planned_task(2, 2, "same_train_fewer_total", 2, 1, 1),
        ];

        sort_task_plan(&mut tasks);

        assert_eq!(tasks[0].label_name, "same_train_fewer_total");
        assert_eq!(tasks[1].label_name, "few");
        assert_eq!(tasks[2].label_name, "many");
    }

    #[test]
    fn evolution_config_uses_aggressive_defaults() {
        let config = baseline_config();
        let built = config.evolution_config();
        assert!(built.is_ok());
        let Ok(built) = built else { unreachable!() };
        assert_eq!(built.population_size(), 1024);
        assert_eq!(built.generation_limit(), 800);
        assert_eq!(built.stagnation_limit(), 120);
        assert_eq!(built.tournament_size(), 5);
        assert_eq!(built.elite_count(), 8);
        assert_eq!(built.fitness_cache_capacity(), 500_000);
        assert_eq!(built.max_evaluation_smarts_complexity(), 1536);
        assert_eq!(
            built.slow_evaluation_log_threshold(),
            Some(Duration::from_secs(30))
        );
        assert_eq!(built.max_evaluation_smarts_len(), None);
        assert!((built.mutation_rate() - 0.90).abs() < f64::EPSILON);
        assert!((built.crossover_rate() - 0.75).abs() < f64::EPSILON);
        assert!((built.selection_ratio() - 0.35).abs() < f64::EPSILON);
        assert!((built.random_immigrant_ratio() - 0.20).abs() < f64::EPSILON);
    }

    #[test]
    fn evolution_config_forwards_slow_smarts_controls() {
        let mut config = baseline_config();
        config.max_evaluation_smarts_complexity = Some(8);
        config.max_evaluation_smarts_len = Some(96);
        config.slow_evaluation_log_threshold_millis = 250;

        let built = config.evolution_config();
        assert!(built.is_ok());
        let Ok(built) = built else { unreachable!() };
        assert_eq!(built.max_evaluation_smarts_complexity(), 8);
        assert_eq!(built.max_evaluation_smarts_len(), Some(96));
        assert_eq!(
            built.slow_evaluation_log_threshold(),
            Some(Duration::from_millis(250))
        );
    }

    #[test]
    fn evolution_config_can_disable_slow_smarts_logging() {
        let mut config = baseline_config();
        config.disable_slow_evaluation_logging = true;

        let built = config.evolution_config();
        assert!(built.is_ok());
        let Ok(built) = built else { unreachable!() };
        assert_eq!(built.slow_evaluation_log_threshold(), None);
    }

    #[test]
    fn count_outcomes_splits_completed_and_skipped() {
        let completed = CompletedTaskReport {
            head: LabelHead::Class,
            label_id: 0,
            label_name: String::from("test"),
            selection_strategy: SelectionStrategy::ValidationBestLeader,
            generations: 1,
            train_counts: SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
            validation_counts: SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
            test_counts: SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
            train_best_smarts: String::from("[#6]"),
            train_best_mcc: 1.0,
            selected_smarts: String::from("[#6]"),
            selected_complexity: 1,
            selected_train_mcc: 1.0,
            selected_validation_mcc: 1.0,
            selected_test_mcc: 1.0,
            candidates: Vec::new(),
        };
        let skipped = SkippedTaskReport {
            head: LabelHead::Pathway,
            label_id: 1,
            label_name: String::from("skip"),
            reason: String::from("not enough positives"),
            train_counts: SplitCounts {
                rows: 1,
                positives: 0,
                negatives: 1,
            },
            validation_counts: SplitCounts {
                rows: 1,
                positives: 0,
                negatives: 1,
            },
            test_counts: SplitCounts {
                rows: 1,
                positives: 0,
                negatives: 1,
            },
        };

        let outcomes = vec![
            TaskOutcome::Completed(completed),
            TaskOutcome::Skipped(skipped),
        ];
        assert_eq!(count_outcomes(&outcomes), (1, 1));
    }

    #[test]
    fn skip_reason_reports_negative_only_train_fold() {
        let mut config = baseline_config();
        config.min_train_positives = 2;
        let reason = skip_reason(
            &config,
            &SplitCounts {
                rows: 1,
                positives: 1,
                negatives: 0,
            },
            &SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
            &SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
        );
        assert_eq!(reason.as_deref(), Some("train positives 1 < required 2"));
    }

    #[test]
    fn compare_candidates_prefers_validation_then_simplicity() {
        let mut candidates = [
            CandidateScore {
                smarts: String::from("[#6]~[#7]"),
                complexity: 2,
                train_mcc: 0.9,
                validation_mcc: 0.8,
                test_mcc: 0.7,
            },
            CandidateScore {
                smarts: String::from("[#7]"),
                complexity: 1,
                train_mcc: 0.7,
                validation_mcc: 0.8,
                test_mcc: 0.7,
            },
            CandidateScore {
                smarts: String::from("[#8]"),
                complexity: 1,
                train_mcc: 1.0,
                validation_mcc: 0.6,
                test_mcc: 0.6,
            },
        ];
        candidates.sort_by(compare_candidates);
        assert_eq!(candidates[0].smarts, "[#7]");
        assert_eq!(candidates[1].smarts, "[#6]~[#7]");
    }

    #[test]
    fn skip_reason_reports_missing_negatives() {
        let config = baseline_config();
        let reason = skip_reason(
            &config,
            &SplitCounts {
                rows: 2,
                positives: 2,
                negatives: 0,
            },
            &SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
            &SplitCounts {
                rows: 2,
                positives: 1,
                negatives: 1,
            },
        );
        assert_eq!(reason.as_deref(), Some("train split has no negatives"));
    }

    #[test]
    fn json_helpers_write_expected_payloads() {
        let temp_dir =
            std::env::temp_dir().join(format!("npc-smarts-json-helpers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let create_dir_result = std::fs::create_dir_all(&temp_dir);
        assert!(create_dir_result.is_ok());

        let jsonl_path = temp_dir.join("results.jsonl");
        let init_result = File::create(&jsonl_path);
        assert!(init_result.is_ok());
        let json_path = temp_dir.join("summary.json");

        let append_result = append_json_line(
            &jsonl_path,
            &SplitCounts {
                rows: 3,
                positives: 1,
                negatives: 2,
            },
        );
        assert!(append_result.is_ok());
        let pretty_write_result = write_json_pretty(
            &json_path,
            &SplitCounts {
                rows: 4,
                positives: 2,
                negatives: 2,
            },
        );
        assert!(pretty_write_result.is_ok());

        let jsonl_read_result = std::fs::read_to_string(&jsonl_path);
        assert!(jsonl_read_result.is_ok());
        let Ok(jsonl_payload) = jsonl_read_result else {
            unreachable!()
        };
        assert_eq!(
            jsonl_payload,
            "{\"rows\":3,\"positives\":1,\"negatives\":2}\n"
        );

        let summary_read_result = std::fs::read_to_string(&json_path);
        assert!(summary_read_result.is_ok());
        let Ok(summary_payload) = summary_read_result else {
            unreachable!()
        };
        assert!(summary_payload.contains("\"rows\": 4"));
        assert!(summary_payload.ends_with('\n'));

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn run_label_task_completes_on_small_class_problem() {
        let temp_dir =
            std::env::temp_dir().join(format!("npc-smarts-run-label-task-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let create_dir_result = std::fs::create_dir_all(&temp_dir);
        assert!(create_dir_result.is_ok());

        let inputs = load_inputs_for_class_task(&temp_dir);
        let mut config = baseline_config();
        config.population_size = 16;
        config.generation_limit = 6;
        config.stagnation_limit = 3;
        config.leaderboard_size = 4;
        config.rng_seed = Some(7);
        let evolution_config = config.evolution_config();
        assert!(evolution_config.is_ok());
        let Ok(evolution_config) = evolution_config else {
            unreachable!()
        };
        let progress = hidden_experiment_progress();
        let seed_corpus = SeedCorpus::builtin();
        let task_context = TaskRunContext {
            config: &config,
            evolution_config: &evolution_config,
            seed_corpus: &seed_corpus,
            progress: &progress,
            inputs: &inputs,
        };
        let task = planned_class_task(&task_context, 0, "amine");
        let outcome = run_label_task(&task_context, &task);
        assert!(outcome.is_ok());
        let Ok(TaskOutcome::Completed(report)) = outcome else {
            unreachable!();
        };
        assert_eq!(report.head, LabelHead::Class);
        assert_eq!(report.label_id, 0);
        assert_eq!(report.train_counts.positives, 2);
        assert_eq!(report.validation_counts.positives, 1);
        assert_eq!(report.test_counts.positives, 1);
        assert!(!report.selected_smarts.is_empty());
        assert!(!report.candidates.is_empty());

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn run_label_task_returns_skipped_when_thresholds_are_not_met() {
        let temp_dir =
            std::env::temp_dir().join(format!("npc-smarts-run-label-skip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let create_dir_result = std::fs::create_dir_all(&temp_dir);
        assert!(create_dir_result.is_ok());

        let inputs = load_inputs_for_class_task(&temp_dir);
        let mut config = baseline_config();
        config.min_train_positives = 10;
        let evolution_config = config.evolution_config();
        assert!(evolution_config.is_ok());
        let Ok(evolution_config) = evolution_config else {
            unreachable!()
        };
        let progress = hidden_experiment_progress();
        let seed_corpus = SeedCorpus::builtin();
        let task_context = TaskRunContext {
            config: &config,
            evolution_config: &evolution_config,
            seed_corpus: &seed_corpus,
            progress: &progress,
            inputs: &inputs,
        };
        let task = planned_class_task(&task_context, 0, "amine");
        let outcome = run_label_task(&task_context, &task);
        assert!(matches!(outcome, Ok(TaskOutcome::Skipped(_))));

        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
