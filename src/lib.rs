#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod dataset;
pub mod download;
pub mod experiment;

pub use dataset::{DatasetSplit, LabelHead, Vocabulary};
pub use download::{
    DISTILLATION_DATASET_DOI, DISTILLATION_DATASET_FILES, DISTILLATION_DATASET_RECORD_ID,
    DownloadedDatasetFile, ensure_distillation_dataset, missing_distillation_dataset_files,
};
pub use experiment::{
    ExperimentConfig, ExperimentError, ExperimentSummary, TaskLogEntry, TaskOutcome, run_experiment,
};
