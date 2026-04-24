use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use arrow_array::cast::{as_list_array, as_primitive_array, as_string_array};
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch, UInt16Array};
use indicatif::ProgressBar;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smarts_evolution::{FoldData, FoldSample};
use smarts_rs::PreparedTarget;
use smiles_parser::Smiles;

use crate::experiment::ExperimentError;

const FOLD_PROGRESS_GRANULARITY: usize = 8_192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelHead {
    Pathway,
    Superclass,
    Class,
}

impl LabelHead {
    pub const ALL: [Self; 3] = [Self::Pathway, Self::Superclass, Self::Class];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pathway => "pathway",
            Self::Superclass => "superclass",
            Self::Class => "class",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocabulary {
    pub pathway: Vec<String>,
    pub superclass: Vec<String>,
    #[serde(rename = "class")]
    pub class_labels: Vec<String>,
}

impl Vocabulary {
    /// Load the label vocabulary JSON emitted with the published dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the JSON payload does
    /// not match the expected schema.
    pub fn load(path: &Path) -> Result<Self, ExperimentError> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    #[must_use]
    pub fn labels(&self, head: LabelHead) -> &[String] {
        match head {
            LabelHead::Pathway => &self.pathway,
            LabelHead::Superclass => &self.superclass,
            LabelHead::Class => &self.class_labels,
        }
    }
}

#[derive(Clone)]
pub struct SplitRow {
    pub cid: i64,
    pub smiles: String,
    pub pathway_ids: Vec<u16>,
    pub superclass_ids: Vec<u16>,
    pub class_ids: Vec<u16>,
}

impl SplitRow {
    #[must_use]
    pub fn labels(&self, head: LabelHead) -> &[u16] {
        match head {
            LabelHead::Pathway => &self.pathway_ids,
            LabelHead::Superclass => &self.superclass_ids,
            LabelHead::Class => &self.class_ids,
        }
    }
}

#[derive(Clone)]
pub struct DatasetSplit {
    name: String,
    rows: Vec<SplitRow>,
}

impl DatasetSplit {
    /// Load one published parquet split without preparing molecule match data.
    ///
    /// # Errors
    ///
    /// Returns an error if the parquet file cannot be read, if required columns
    /// are missing, if labels are malformed, or if any SMILES row cannot be
    /// parsed into the in-memory row representation.
    pub fn load(path: &Path, name: impl Into<String>) -> Result<Self, ExperimentError> {
        let progress_bar = ProgressBar::hidden();
        Self::load_with_progress(path, name, &progress_bar)
    }

    /// Load one published parquet split while updating a progress bar by row.
    ///
    /// # Errors
    ///
    /// Returns an error if the parquet file cannot be read, if required columns
    /// are missing, if labels are malformed, or if any SMILES row cannot be
    /// parsed into the in-memory row representation.
    pub fn load_with_progress(
        path: &Path,
        name: impl Into<String>,
        progress_bar: &ProgressBar,
    ) -> Result<Self, ExperimentError> {
        let name = name.into();
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let total_rows =
            usize::try_from(builder.metadata().file_metadata().num_rows()).map_err(|error| {
                ExperimentError::InvalidDataset(format!(
                    "split {name} reported an invalid row count: {error}"
                ))
            })?;
        progress_bar.set_length(usize_to_u64(total_rows));
        progress_bar.set_position(0);
        progress_bar.set_message(format!("{name} | loading parquet batches"));
        let reader = builder.build()?;

        let mut rows = Vec::with_capacity(total_rows);
        for batch in reader {
            let batch =
                batch.map_err(|error| ExperimentError::InvalidDataset(error.to_string()))?;
            progress_bar.set_message(format!("{name} | loading raw rows"));
            let prepared_rows = prepare_batch_rows(&batch, &name)?;
            progress_bar.inc(usize_to_u64(prepared_rows.len()));
            rows.extend(prepared_rows);
        }

        if rows.is_empty() {
            return Err(ExperimentError::EmptySplit(name));
        }

        Ok(Self { name, rows })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn rows(&self) -> &[SplitRow] {
        &self.rows
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Build an unsampled labeled evaluation set.
    ///
    /// # Errors
    ///
    /// Returns an error if any selected SMILES cannot be parsed.
    pub fn build_fold(
        &self,
        head: LabelHead,
        label_id: u16,
    ) -> Result<LabeledFold, ExperimentError> {
        let progress_bar = ProgressBar::hidden();
        self.build_fold_with_progress(head, label_id, &progress_bar)
    }

    /// Build an unsampled labeled evaluation set with progress.
    ///
    /// # Errors
    ///
    /// Returns an error if any selected SMILES cannot be parsed.
    pub fn build_fold_with_progress(
        &self,
        head: LabelHead,
        label_id: u16,
        progress_bar: &ProgressBar,
    ) -> Result<LabeledFold, ExperimentError> {
        self.build_sampled_fold_with_progress(head, label_id, usize::MAX, usize::MAX, progress_bar)
    }

    #[must_use]
    pub fn sampled_counts_with_progress(
        &self,
        head: LabelHead,
        label_id: u16,
        max_positives_per_class: usize,
        max_negatives_per_class: usize,
        progress_bar: &ProgressBar,
    ) -> FoldSelectionCounts {
        let selection = self.select_sample_indices(
            head,
            label_id,
            max_positives_per_class,
            max_negatives_per_class,
            progress_bar,
        );
        FoldSelectionCounts {
            positive_count: selection.positive_count,
            negative_count: selection.negative_count,
        }
    }

    /// Build a sampled labeled evaluation set with progress.
    ///
    /// # Errors
    ///
    /// Returns an error if any selected SMILES cannot be parsed.
    pub fn build_sampled_fold_with_progress(
        &self,
        head: LabelHead,
        label_id: u16,
        max_positives_per_class: usize,
        max_negatives_per_class: usize,
        progress_bar: &ProgressBar,
    ) -> Result<LabeledFold, ExperimentError> {
        let SampleSelection {
            indices,
            positive_count,
            negative_count,
        } = self.select_sample_indices(
            head,
            label_id,
            max_positives_per_class,
            max_negatives_per_class,
            progress_bar,
        );
        progress_bar.set_length(usize_to_u64(indices.len()));
        progress_bar.set_position(0);
        progress_bar.set_message(format!(
            "{} | preparing {} selected {}:{label_id} targets",
            self.name,
            indices.len(),
            head.as_str()
        ));

        let completed_count = AtomicUsize::new(0);
        let selected_count = indices.len();
        let samples = indices
            .par_iter()
            .map(|&row_index| {
                let row = &self.rows[row_index];
                let sample = prepare_fold_sample(row, &self.name, head, label_id);
                let completed = completed_count.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                if completed.is_multiple_of(FOLD_PROGRESS_GRANULARITY)
                    || completed == selected_count
                {
                    progress_bar.set_position(usize_to_u64(completed));
                }
                sample
            })
            .collect::<Result<Vec<_>, _>>()?;
        drop(indices);

        progress_bar.set_message(format!(
            "{} | indexing {} selected {}:{label_id} targets",
            self.name,
            samples.len(),
            head.as_str()
        ));
        progress_bar.tick();
        Ok(LabeledFold {
            fold: FoldData::new(samples),
            positive_count,
            negative_count,
        })
    }

    fn select_sample_indices(
        &self,
        head: LabelHead,
        label_id: u16,
        max_positives_per_class: usize,
        max_negatives_per_class: usize,
        progress_bar: &ProgressBar,
    ) -> SampleSelection {
        let row_count = self.rows.len();
        progress_bar.set_length(usize_to_u64(row_count));
        progress_bar.set_position(0);
        progress_bar.set_message(format!(
            "{} | selecting {}:{label_id} evaluation rows | max_positives_per_class={max_positives_per_class} max_negatives_per_class={max_negatives_per_class}",
            self.name,
            head.as_str()
        ));

        let mut all_positive_indices = Vec::new();
        let mut all_negative_indices = Vec::new();
        let mut positive_buckets: HashMap<Option<u16>, BinaryHeap<RankedIndex>> = HashMap::new();
        let mut negative_buckets: HashMap<Option<u16>, BinaryHeap<RankedIndex>> = HashMap::new();

        for (row_index, row) in self.rows.iter().enumerate() {
            if row.labels(head).contains(&label_id) {
                if max_positives_per_class == usize::MAX {
                    all_positive_indices.push(row_index);
                } else if max_positives_per_class > 0 {
                    push_sample_candidates(
                        &mut positive_buckets,
                        row,
                        row_index,
                        max_positives_per_class,
                    );
                }
            } else if max_negatives_per_class == usize::MAX {
                all_negative_indices.push(row_index);
            } else if max_negatives_per_class > 0 {
                push_sample_candidates(
                    &mut negative_buckets,
                    row,
                    row_index,
                    max_negatives_per_class,
                );
            }

            let completed = row_index + 1;
            if completed.is_multiple_of(FOLD_PROGRESS_GRANULARITY) || completed == row_count {
                progress_bar.set_position(usize_to_u64(completed));
            }
        }

        let mut positive_indices = if max_positives_per_class == usize::MAX {
            all_positive_indices
        } else {
            sampled_indices(positive_buckets)
        };
        positive_indices.sort_unstable();
        positive_indices.dedup();

        let mut negative_indices = if max_negatives_per_class == usize::MAX {
            all_negative_indices
        } else {
            sampled_indices(negative_buckets)
        };
        negative_indices.sort_unstable();
        negative_indices.dedup();

        let positive_count = positive_indices.len();
        let negative_count = negative_indices.len();
        positive_indices.extend(negative_indices);
        positive_indices.sort_unstable();

        SampleSelection {
            indices: positive_indices,
            positive_count,
            negative_count,
        }
    }
}

#[derive(Clone)]
pub struct LabeledFold {
    pub fold: FoldData,
    pub positive_count: usize,
    pub negative_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldSelectionCounts {
    pub positive_count: usize,
    pub negative_count: usize,
}

struct SampleSelection {
    indices: Vec<usize>,
    positive_count: usize,
    negative_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RankedIndex {
    score: u64,
    index: usize,
}

impl Ord for RankedIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for RankedIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn push_sample_candidates(
    buckets: &mut HashMap<Option<u16>, BinaryHeap<RankedIndex>>,
    row: &SplitRow,
    row_index: usize,
    max_per_class: usize,
) {
    if row.class_ids.is_empty() {
        push_sample_candidate(
            buckets,
            None,
            RankedIndex {
                score: sample_score(row, row_index, None),
                index: row_index,
            },
            max_per_class,
        );
        return;
    }

    for &class_id in &row.class_ids {
        push_sample_candidate(
            buckets,
            Some(class_id),
            RankedIndex {
                score: sample_score(row, row_index, Some(class_id)),
                index: row_index,
            },
            max_per_class,
        );
    }
}

fn push_sample_candidate(
    buckets: &mut HashMap<Option<u16>, BinaryHeap<RankedIndex>>,
    class_id: Option<u16>,
    candidate: RankedIndex,
    max_per_class: usize,
) {
    let bucket = buckets.entry(class_id).or_default();
    if bucket.len() < max_per_class {
        bucket.push(candidate);
    } else if bucket.peek().is_some_and(|worst| candidate < *worst) {
        bucket.pop();
        bucket.push(candidate);
    }
}

fn sampled_indices(buckets: HashMap<Option<u16>, BinaryHeap<RankedIndex>>) -> Vec<usize> {
    buckets
        .into_values()
        .flat_map(BinaryHeap::into_iter)
        .map(|candidate| candidate.index)
        .collect()
}

fn sample_score(row: &SplitRow, row_index: usize, class_id: Option<u16>) -> u64 {
    let cid_bits = u64::from_ne_bytes(row.cid.to_ne_bytes());
    let row_bits = u64::try_from(row_index).unwrap_or(u64::MAX);
    let class_bits = class_id.map_or(u64::MAX, u64::from);
    mix64(cid_bits ^ row_bits.rotate_left(17) ^ class_bits.rotate_left(41))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn column<'a>(
    batch: &'a RecordBatch,
    split: &str,
    name: &str,
) -> Result<&'a dyn Array, ExperimentError> {
    batch
        .column_by_name(name)
        .map(std::convert::AsRef::as_ref)
        .ok_or_else(|| ExperimentError::MissingParquetColumn {
            split: split.to_owned(),
            column: name.to_owned(),
        })
}

fn label_column<'a>(
    batch: &'a RecordBatch,
    split: &str,
    name: &str,
) -> Result<LabelColumnView<'a>, ExperimentError> {
    let list = as_list_array(column(batch, split, name)?);
    let values = list
        .values()
        .as_any()
        .downcast_ref::<UInt16Array>()
        .ok_or_else(|| {
            ExperimentError::InvalidDataset(format!(
                "list column {name} did not contain uint16 values"
            ))
        })?;
    Ok(LabelColumnView { list, values })
}

struct LabelColumnView<'a> {
    list: &'a arrow_array::ListArray,
    values: &'a UInt16Array,
}

impl LabelColumnView<'_> {
    fn values(&self, row_index: usize) -> Result<Vec<u16>, ExperimentError> {
        let offsets = self.list.value_offsets();
        let start = usize::try_from(offsets[row_index]).map_err(|error| {
            ExperimentError::InvalidDataset(format!(
                "row offset for index {row_index} overflowed usize: {error}"
            ))
        })?;
        let end = usize::try_from(offsets[row_index + 1]).map_err(|error| {
            ExperimentError::InvalidDataset(format!(
                "row offset for index {} overflowed usize: {error}",
                row_index + 1
            ))
        })?;
        Ok(self.values.values()[start..end].to_vec())
    }
}

struct RawSplitRow {
    cid: i64,
    smiles: String,
    pathway_ids: Vec<u16>,
    superclass_ids: Vec<u16>,
    class_ids: Vec<u16>,
}

fn prepare_batch_rows(batch: &RecordBatch, split: &str) -> Result<Vec<SplitRow>, ExperimentError> {
    let smiles_array = as_string_array(column(batch, split, "smiles")?);
    let cid_array = as_primitive_array::<Int64Type>(column(batch, split, "cid")?);
    let pathway_view = label_column(batch, split, "pathway_ids")?;
    let superclass_view = label_column(batch, split, "superclass_ids")?;
    let class_view = label_column(batch, split, "class_ids")?;

    let raw_rows = (0..batch.num_rows())
        .map(|row_index| {
            Ok(RawSplitRow {
                cid: cid_array.value(row_index),
                smiles: smiles_array.value(row_index).to_owned(),
                pathway_ids: pathway_view.values(row_index)?,
                superclass_ids: superclass_view.values(row_index)?,
                class_ids: class_view.values(row_index)?,
            })
        })
        .collect::<Result<Vec<_>, ExperimentError>>()?;

    Ok(raw_rows.into_iter().map(prepare_raw_row).collect())
}

fn prepare_raw_row(row: RawSplitRow) -> SplitRow {
    let RawSplitRow {
        cid,
        smiles,
        pathway_ids,
        superclass_ids,
        class_ids,
    } = row;
    SplitRow {
        cid,
        smiles,
        pathway_ids,
        superclass_ids,
        class_ids,
    }
}

fn prepare_fold_sample(
    row: &SplitRow,
    split: &str,
    head: LabelHead,
    label_id: u16,
) -> Result<FoldSample, ExperimentError> {
    let parsed = row
        .smiles
        .parse::<Smiles>()
        .map_err(|error| ExperimentError::InvalidSmiles {
            split: split.to_owned(),
            cid: row.cid,
            smiles: row.smiles.clone(),
            message: error.to_string(),
        })?;
    let target = PreparedTarget::new(parsed);
    if row.labels(head).contains(&label_id) {
        Ok(FoldSample::positive(target))
    } else {
        Ok(FoldSample::negative(target))
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, ListArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::*;

    type TestSplitRow = (&'static str, i64, Vec<u16>, Vec<u16>, Vec<u16>);

    fn write_split_parquet(path: &Path, rows: &[TestSplitRow]) {
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
        let cids = rows
            .iter()
            .map(|(_, cid, _, _, _)| *cid)
            .collect::<Int64Array>();
        let smiles = StringArray::from_iter_values(rows.iter().map(|(smiles, _, _, _, _)| *smiles));
        let pathway_ids = ListArray::from_iter_primitive::<arrow_array::types::UInt16Type, _, _>(
            rows.iter()
                .map(|(_, _, pathway, _, _)| Some(pathway.iter().copied().map(Some))),
        );
        let superclass_ids = ListArray::from_iter_primitive::<arrow_array::types::UInt16Type, _, _>(
            rows.iter()
                .map(|(_, _, _, superclass, _)| Some(superclass.iter().copied().map(Some))),
        );
        let class_ids = ListArray::from_iter_primitive::<arrow_array::types::UInt16Type, _, _>(
            rows.iter()
                .map(|(_, _, _, _, class_ids)| Some(class_ids.iter().copied().map(Some))),
        );
        let batch_result = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(cids),
                Arc::new(smiles),
                Arc::new(pathway_ids),
                Arc::new(superclass_ids),
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

    #[test]
    fn dataset_split_loads_rows_and_builds_binary_fold() {
        let temp_dir = std::env::temp_dir().join(format!(
            "npc-smarts-dataset-load-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let create_dir_result = std::fs::create_dir_all(&temp_dir);
        assert!(create_dir_result.is_ok());

        let split_path = temp_dir.join("train.parquet");
        write_split_parquet(
            &split_path,
            &[
                ("CCN", 1, vec![0], vec![1], vec![2]),
                ("CCO", 2, vec![], vec![], vec![]),
                ("NC", 3, vec![0], vec![1], vec![2]),
            ],
        );

        let loaded = DatasetSplit::load(&split_path, "train");
        assert!(loaded.is_ok());
        let Ok(loaded) = loaded else { unreachable!() };
        assert_eq!(loaded.name(), "train");
        assert_eq!(loaded.len(), 3);
        assert!(!loaded.is_empty());
        assert_eq!(loaded.rows()[0].cid, 1);
        assert_eq!(loaded.rows()[0].smiles, "CCN");
        assert_eq!(loaded.rows()[0].labels(LabelHead::Class), &[2]);
        assert!(loaded.rows()[1].labels(LabelHead::Pathway).is_empty());

        let fold = loaded.build_fold(LabelHead::Class, 2);
        assert!(fold.is_ok());
        let Ok(fold) = fold else { unreachable!() };
        assert_eq!(fold.positive_count, 2);
        assert_eq!(fold.negative_count, 1);
        assert_eq!(fold.fold.len(), 3);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn sampled_fold_caps_positives_and_negatives_by_class() {
        let temp_dir = std::env::temp_dir().join(format!(
            "npc-smarts-dataset-sampling-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let create_dir_result = std::fs::create_dir_all(&temp_dir);
        assert!(create_dir_result.is_ok());

        let split_path = temp_dir.join("train.parquet");
        write_split_parquet(
            &split_path,
            &[
                ("CCN", 1, vec![], vec![], vec![0]),
                ("CN", 2, vec![], vec![], vec![0]),
                ("CCO", 3, vec![], vec![], vec![1]),
                ("CO", 4, vec![], vec![], vec![1]),
                ("CCC", 5, vec![], vec![], vec![2]),
                ("CC", 6, vec![], vec![], vec![2]),
                ("O", 7, vec![], vec![], vec![]),
                ("N", 8, vec![], vec![], vec![]),
            ],
        );

        let loaded = DatasetSplit::load(&split_path, "train");
        assert!(loaded.is_ok());
        let Ok(loaded) = loaded else { unreachable!() };
        let progress_bar = ProgressBar::hidden();
        let counts = loaded.sampled_counts_with_progress(LabelHead::Class, 0, 1, 1, &progress_bar);
        assert_eq!(
            counts,
            FoldSelectionCounts {
                positive_count: 1,
                negative_count: 3,
            }
        );

        let fold =
            loaded.build_sampled_fold_with_progress(LabelHead::Class, 0, 1, 1, &progress_bar);
        assert!(fold.is_ok());
        let Ok(fold) = fold else { unreachable!() };
        assert_eq!(fold.positive_count, 1);
        assert_eq!(fold.negative_count, 3);
        assert_eq!(fold.fold.len(), 4);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn vocabulary_load_and_label_lookup_follow_head_selection() {
        let temp_dir =
            std::env::temp_dir().join(format!("npc-smarts-vocabulary-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        let create_dir_result = std::fs::create_dir_all(&temp_dir);
        assert!(create_dir_result.is_ok());

        let vocabulary_path = temp_dir.join("vocabulary.json");
        let write_result = std::fs::write(
            &vocabulary_path,
            "{\n  \"pathway\": [\"p0\"],\n  \"superclass\": [\"s0\", \"s1\"],\n  \"class\": [\"c0\", \"c1\", \"c2\"]\n}\n",
        );
        assert!(write_result.is_ok());

        let loaded = Vocabulary::load(&vocabulary_path);
        assert!(loaded.is_ok());
        let Ok(loaded) = loaded else { unreachable!() };
        assert_eq!(loaded.labels(LabelHead::Pathway), ["p0"]);
        assert_eq!(loaded.labels(LabelHead::Superclass), ["s0", "s1"]);
        assert_eq!(loaded.labels(LabelHead::Class), ["c0", "c1", "c2"]);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn label_heads_render_expected_names() {
        assert_eq!(LabelHead::Pathway.as_str(), "pathway");
        assert_eq!(LabelHead::Superclass.as_str(), "superclass");
        assert_eq!(LabelHead::Class.as_str(), "class");
    }
}
