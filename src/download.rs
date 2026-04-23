use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use zenodo_rs::{Auth, RecordId, ZenodoClient, ZenodoError};

pub const DISTILLATION_DATASET_RECORD_ID: u64 = 19_701_295;
pub const DISTILLATION_DATASET_DOI: &str = "10.5281/zenodo.19701295";
pub const DISTILLATION_DATASET_FILES: &[&str] = &[
    "README.md",
    "LICENSE",
    "manifest.json",
    "SHA256SUMS.txt",
    "vocabulary.json",
    "train.parquet",
    "train.pathway-vectors.f16.zst",
    "train.superclass-vectors.f16.zst",
    "train.class-vectors.f16.zst",
    "validation.parquet",
    "validation.pathway-vectors.f16.zst",
    "validation.superclass-vectors.f16.zst",
    "validation.class-vectors.f16.zst",
    "test.parquet",
    "test.pathway-vectors.f16.zst",
    "test.superclass-vectors.f16.zst",
    "test.class-vectors.f16.zst",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DownloadedDatasetFile {
    pub key: String,
    pub path: PathBuf,
    pub bytes_written: u64,
    pub skipped: bool,
}

#[must_use]
pub fn missing_distillation_dataset_files(data_dir: &Path) -> Vec<&'static str> {
    DISTILLATION_DATASET_FILES
        .iter()
        .copied()
        .filter(|key| !data_dir.join(key).exists())
        .collect()
}

/// Download the published distillation split files into `data_dir`.
///
/// # Errors
///
/// Returns an error if the dataset directory cannot be created, the Zenodo
/// client cannot be initialized, or any required file cannot be downloaded and
/// atomically moved into place.
pub async fn ensure_distillation_dataset(
    data_dir: &Path,
) -> Result<Vec<DownloadedDatasetFile>, ZenodoError> {
    fs::create_dir_all(data_dir)?;
    let client = ZenodoClient::builder(Auth::new(
        std::env::var(Auth::TOKEN_ENV_VAR).unwrap_or_default(),
    ))
    .user_agent("npc-smarts/distillation-dataset")
    .build()?;

    let progress_bar = download_progress_bar(DISTILLATION_DATASET_FILES.len());
    let mut downloaded = Vec::with_capacity(DISTILLATION_DATASET_FILES.len());
    let mut downloaded_count = 0usize;
    let mut skipped_count = 0usize;
    for key in DISTILLATION_DATASET_FILES {
        let final_path = data_dir.join(key);
        if final_path.exists() {
            skipped_count += 1;
            downloaded.push(DownloadedDatasetFile {
                key: (*key).to_owned(),
                path: final_path,
                bytes_written: 0,
                skipped: true,
            });
            progress_bar.set_message(format!("dataset | skipping {key}"));
            progress_bar.inc(1);
            continue;
        }

        progress_bar.set_message(format!("dataset | downloading {key}"));
        let part_path = temporary_download_path(&final_path);
        if part_path.exists() {
            fs::remove_file(&part_path)?;
        }
        let resolved = client
            .download_record_file_by_key_to_path(
                RecordId(DISTILLATION_DATASET_RECORD_ID),
                key,
                &part_path,
            )
            .await?;
        fs::rename(&part_path, &final_path)?;
        downloaded_count += 1;
        downloaded.push(DownloadedDatasetFile {
            key: (*key).to_owned(),
            path: final_path,
            bytes_written: resolved.bytes_written,
            skipped: false,
        });
        progress_bar.inc(1);
        progress_bar.println(format!(
            "[download] {key} | {} bytes",
            resolved.bytes_written
        ));
    }

    progress_bar.finish_with_message(format!(
        "dataset ready | downloaded={downloaded_count} skipped={skipped_count}"
    ));
    Ok(downloaded)
}

fn temporary_download_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("download");
    path.with_file_name(format!("{file_name}.part"))
}

fn download_progress_bar(total_files: usize) -> ProgressBar {
    let progress_bar = if std::io::stderr().is_terminal() {
        ProgressBar::new(usize_to_u64(total_files))
    } else {
        ProgressBar::hidden()
    };
    progress_bar.set_style(download_progress_style());
    progress_bar.enable_steady_tick(Duration::from_millis(100));
    progress_bar.set_message("dataset | checking required files");
    progress_bar
}

fn download_progress_style() -> ProgressStyle {
    let style = match ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {msg}",
    ) {
        Ok(style) => style,
        Err(_) => ProgressStyle::default_bar(),
    };
    style.progress_chars("=> ")
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_files_only_reports_absent_dataset_entries() {
        let temp_dir =
            std::env::temp_dir().join(format!("npc-smarts-download-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        let create_dir_result = fs::create_dir_all(&temp_dir);
        assert!(create_dir_result.is_ok());

        let write_result = fs::write(temp_dir.join("manifest.json"), "{}\n");
        assert!(write_result.is_ok());

        let missing = missing_distillation_dataset_files(&temp_dir);
        assert!(!missing.contains(&"manifest.json"));
        assert!(missing.contains(&"train.parquet"));
        assert!(missing.contains(&"test.class-vectors.f16.zst"));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn temporary_download_path_uses_part_suffix() {
        let path = Path::new("data/train.parquet");
        assert_eq!(
            temporary_download_path(path),
            PathBuf::from("data/train.parquet.part")
        );
    }
}
