use std::fs::File;
use std::path::Path;

use arrow_array::cast::{as_list_array, as_primitive_array, as_string_array};
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch, UInt16Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use smarts_evolution::{FoldData, FoldSample};
use smarts_rs::PreparedTarget;
use smiles_parser::Smiles;

use crate::experiment::ExperimentError;

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
    pub target: PreparedTarget,
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
    /// Load one published parquet split and prepare every SMILES for matching.
    ///
    /// # Errors
    ///
    /// Returns an error if the parquet file cannot be read, if required columns
    /// are missing, if labels are malformed, or if any SMILES row cannot be
    /// parsed into a prepared target.
    pub fn load(path: &Path, name: impl Into<String>) -> Result<Self, ExperimentError> {
        let name = name.into();
        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut rows = Vec::new();
        for batch in reader {
            let batch =
                batch.map_err(|error| ExperimentError::InvalidDataset(error.to_string()))?;
            let smiles_array = as_string_array(column(&batch, &name, "smiles")?);
            let cid_array = as_primitive_array::<Int64Type>(column(&batch, &name, "cid")?);
            let pathway_view = label_column(&batch, &name, "pathway_ids")?;
            let superclass_view = label_column(&batch, &name, "superclass_ids")?;
            let class_view = label_column(&batch, &name, "class_ids")?;

            for row_index in 0..batch.num_rows() {
                let cid = cid_array.value(row_index);
                let smiles = smiles_array.value(row_index).to_owned();
                let parsed =
                    smiles
                        .parse::<Smiles>()
                        .map_err(|error| ExperimentError::InvalidSmiles {
                            split: name.clone(),
                            cid,
                            smiles: smiles.clone(),
                            message: error.to_string(),
                        })?;
                rows.push(SplitRow {
                    cid,
                    smiles,
                    target: PreparedTarget::new(parsed),
                    pathway_ids: pathway_view.values(row_index)?,
                    superclass_ids: superclass_view.values(row_index)?,
                    class_ids: class_view.values(row_index)?,
                });
            }
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

    #[must_use]
    pub fn build_fold(&self, head: LabelHead, label_id: u16) -> LabeledFold {
        let mut samples = Vec::with_capacity(self.rows.len());
        let mut positive_count = 0usize;
        for row in &self.rows {
            let is_positive = row.labels(head).contains(&label_id);
            positive_count += usize::from(is_positive);
            let sample = if is_positive {
                FoldSample::positive(row.target.clone())
            } else {
                FoldSample::negative(row.target.clone())
            };
            samples.push(sample);
        }

        let negative_count = samples.len().saturating_sub(positive_count);
        LabeledFold {
            fold: FoldData::new(samples),
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
        assert_eq!(fold.positive_count, 2);
        assert_eq!(fold.negative_count, 1);
        assert_eq!(fold.fold.len(), 3);

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
