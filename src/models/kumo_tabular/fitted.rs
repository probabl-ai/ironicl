//! A fitted Kumo-Tabular classifier or regressor: its preprocessing, each
//! member's context, and what turns the members' outputs into probabilities
//! or the regressor's outputs.
//!
//! It saves as its training data in `IRPREP` (architecture 2); loading fits
//! again, which gives the same model, since fitting is deterministic.

use std::sync::Arc;

use super::cpu::{Context, Model, Targets};
use super::preprocess::{MAX_CLASSES, Preprocessor};
use crate::preprocess::labels::Labels;
use crate::progress::{self, Stage};
use crate::{
    DecoderOutputs, Error, RegressionOutput, Result, TaskKind, format::preprocessor,
    models::tabicl_v2::quantiles::summarize,
};

/// Query rows preprocessed at once.
const PREDICT_CHUNK_ROWS: usize = 4096;

/// What a fit starts from, and all that saving keeps: the encoded
/// `[rows, features]` training matrix, each feature's categories (`None` for
/// a numeric one), the labels (a regressor's targets as floats), the members,
/// and the seed.
#[derive(Clone, Debug, PartialEq)]
pub struct Training {
    pub x: Vec<f64>,
    pub categories: Vec<Option<Labels>>,
    pub labels: Labels,
    pub members: usize,
    pub seed: i128,
    pub feature_names: Option<Vec<String>>,
}

/// A fitted model: per member, one context, or for a classifier of more than
/// 10 classes one per codebook task.
pub struct Fitted {
    model: Arc<Model>,
    pool: rayon::ThreadPool,
    training: Training,
    preprocessor: Preprocessor,
    contexts: Vec<Vec<Context>>,
}

impl Fitted {
    /// Fit `model` on `training` with `threads` worker threads, `None` for
    /// one per core, reporting members built.
    ///
    /// # Errors
    ///
    /// When the training data is invalid, or as [`Model::fit`].
    pub fn fit(model: Arc<Model>, training: Training, threads: Option<usize>) -> Result<Self> {
        let categorical: Vec<bool> = training.categories.iter().map(Option::is_some).collect();
        let (members, seed) = (training.members, training.seed);
        let task = model.release.task;
        let preprocessor =
            Preprocessor::fit(&training.x, &categorical, &training.labels, task, members, seed)?;
        Self::with_preprocessor(model, training, preprocessor, threads)
    }

    /// [`Self::fit`] with the given draws.
    #[cfg(test)]
    pub(crate) fn fit_with(
        model: Arc<Model>,
        training: Training,
        draws: super::preprocess::Draws,
    ) -> Result<Self> {
        let categorical: Vec<bool> = training.categories.iter().map(Option::is_some).collect();
        let preprocessor = Preprocessor::fit_with(
            &training.x,
            &categorical,
            &training.labels,
            model.release.task,
            training.members,
            |_, _, _| draws,
        )?;
        Self::with_preprocessor(model, training, preprocessor, None)
    }

    fn with_preprocessor(
        model: Arc<Model>,
        training: Training,
        preprocessor: Preprocessor,
        threads: Option<usize>,
    ) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.unwrap_or(0))
            .build()
            .map_err(|error| Error::InvalidArgument(format!("cannot start {threads:?} threads: {error}")))?;
        let members = preprocessor.members();
        let mut contexts = Vec::with_capacity(members);
        for member in 0..members {
            let rows = preprocessor.transform(&training.x, member)?;
            if model.release.task == TaskKind::Regressor {
                let values = preprocessor.target_values(&training.labels, member)?;
                let context =
                    pool.install(|| model.fit(&rows.x, &rows.categorical, Targets::Values(&values)))?;
                contexts.push(vec![context]);
                progress::report(Stage::Build, member + 1, members)?;
                continue;
            }
            let codes = preprocessor.class_codes(&training.labels, member)?;
            let codebook = &preprocessor.draws.codebooks[member];
            let tasks: Vec<Vec<usize>> = if codebook.is_empty() {
                vec![codes]
            } else {
                codebook
                    .iter()
                    .map(|task| codes.iter().map(|&code| task[code]).collect())
                    .collect()
            };
            contexts.push(
                tasks
                    .iter()
                    .map(|codes| {
                        pool.install(|| model.fit(&rows.x, &rows.categorical, Targets::Classes(codes)))
                    })
                    .collect::<Result<_>>()?,
            );
            progress::report(Stage::Build, member + 1, members)?;
        }
        Ok(Self {
            model,
            pool,
            training,
            preprocessor,
            contexts,
        })
    }

    #[must_use]
    pub fn task(&self) -> TaskKind {
        self.model.release.task
    }

    /// Where the transformer blocks run: `"metal"` or `"cpu"`.
    #[must_use]
    pub fn device(&self) -> &'static str {
        self.model.device()
    }

    /// A classifier's classes, sorted.
    #[must_use]
    pub fn classes(&self) -> Option<&Labels> {
        self.preprocessor.classes.as_ref()
    }

    /// The SHA-256 of the checkpoint this model was fitted with.
    #[must_use]
    pub fn checkpoint_sha256(&self) -> [u8; 32] {
        self.model.sha256()
    }

    #[must_use]
    pub fn training(&self) -> &Training {
        &self.training
    }

    /// The probability of each class for each of the encoded `[rows, features]`
    /// rows `x`, `[rows, classes]`, reporting rows predicted.
    ///
    /// # Errors
    ///
    /// When `x` does not have the training features.
    pub fn predict_proba(&self, x: &[f64]) -> Result<Vec<f32>> {
        if self.classes().is_none() {
            return Err(Error::InvalidArgument(
                "a regressor has no class probabilities".into(),
            ));
        }
        Ok(self.chunks(x, |x| self.probabilities(x))?.concat())
    }

    /// A regressor's `outputs` for each of the encoded `[rows, features]` rows
    /// `x`, each row-major `[rows, columns]` with its columns, in the targets'
    /// units, reporting rows predicted. The members' quantiles are combined
    /// first, as NVIDIA's code does, then summarized as `TabICLRegressor`'s
    /// are.
    ///
    /// # Errors
    ///
    /// For a classifier, or when `x` does not have the training features.
    pub fn predict_regression(
        &self,
        x: &[f64],
        outputs: &[RegressionOutput],
    ) -> Result<Vec<(Vec<f32>, usize)>> {
        if self.preprocessor.target_scale.is_none() {
            return Err(Error::InvalidArgument(
                "a classifier has no regression outputs".into(),
            ));
        }
        let chunks = self.chunks(x, |x| {
            let quantiles = self.quantiles(x)?;
            let rows = quantiles.len() / 999;
            summarize(&DecoderOutputs::new([1, rows, 999], quantiles)?, outputs)
        })?;
        Ok((0..outputs.len())
            .map(|index| {
                let values = chunks
                    .iter()
                    .flat_map(|chunk| chunk[index].0.iter().copied())
                    .collect();
                (values, chunks[0][index].1)
            })
            .collect())
    }

    /// `predict` on each chunk of the `[rows, features]` rows `x`, reporting rows done.
    fn chunks<T>(&self, x: &[f64], predict: impl Fn(&[f64]) -> Result<T>) -> Result<Vec<T>> {
        let features = self.training.categories.len();
        if x.is_empty() || !x.len().is_multiple_of(features) {
            return Err(Error::InvalidArgument("X must have the training features".into()));
        }
        let rows = x.len() / features;
        let mut out = Vec::new();
        for start in (0..rows).step_by(PREDICT_CHUNK_ROWS) {
            let end = (start + PREDICT_CHUNK_ROWS).min(rows);
            out.push(progress::with_offset(start, rows, || {
                predict(&x[start * features..end * features])
            })?);
            progress::report(Stage::Predict, end, rows)?;
        }
        Ok(out)
    }

    /// Each member's outputs of the rows `x`, per context.
    fn outputs(&self, x: &[f64], member: usize) -> Result<Vec<Vec<f32>>> {
        let query = self.preprocessor.transform(x, member)?;
        self.contexts[member]
            .iter()
            .map(|context| self.pool.install(|| self.model.predict(context, &query.x)))
            .collect()
    }

    /// The members' 999 quantiles, in the targets' units and sorted, with the
    /// lowest and highest fifth of the members dropped per quantile, then
    /// averaged.
    ///
    /// Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/processing/output/reduce.py#L38-L66>.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "Member counts are small; the output is FP32"
    )]
    fn quantiles(&self, x: &[f64]) -> Result<Vec<f32>> {
        let (mean, scale) = self.preprocessor.target_scale.expect("a regressor");
        let members = self.contexts.len();
        let mut all = Vec::with_capacity(members);
        for member in 0..members {
            let sign = self.preprocessor.draws.target_signs[member];
            let mut values: Vec<f64> = self
                .outputs(x, member)?
                .remove(0)
                .into_iter()
                .map(|q| f64::from(q) * sign * scale + mean)
                .collect();
            for row in values.chunks_mut(999) {
                row.sort_by(f64::total_cmp);
            }
            all.push(values);
        }
        let cut = (0.2 * members as f64) as usize;
        let mut column = vec![0.0; members];
        Ok((0..all[0].len())
            .map(|index| {
                for (value, member) in column.iter_mut().zip(&all) {
                    *value = member[index];
                }
                column.sort_by(f64::total_cmp);
                let kept = &column[cut..members - cut];
                (kept.iter().sum::<f64>() / kept.len() as f64) as f32
            })
            .collect())
    }

    /// The mean of the members' class scores, softmaxed.
    #[expect(clippy::cast_precision_loss, reason = "Member counts are small")]
    fn probabilities(&self, x: &[f64]) -> Result<Vec<f32>> {
        let classes = self.classes().map_or(0, Labels::len);
        let rows = x.len() / self.training.categories.len();
        let mut sums = vec![0.0_f32; rows * classes];
        for member in 0..self.contexts.len() {
            let logits = self.outputs(x, member)?;
            let scores = if logits.len() == 1 {
                logits[0]
                    .chunks(MAX_CLASSES)
                    .flat_map(|row| &row[..classes])
                    .copied()
                    .collect()
            } else {
                ecoc_scores(&logits, &self.preprocessor.draws.codebooks[member], classes)
            };
            let order = &self.preprocessor.draws.class_orders[member];
            for (sum, row) in sums.chunks_mut(classes).zip(scores.chunks(classes)) {
                for (class, value) in sum.iter_mut().enumerate() {
                    *value += row[order[class]];
                }
            }
        }
        let members = self.contexts.len() as f32;
        for row in sums.chunks_mut(classes) {
            for value in row.iter_mut() {
                *value /= members;
            }
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0;
            for value in row.iter_mut() {
                *value = (*value - maximum).exp();
                total += *value;
            }
            for value in row.iter_mut() {
                *value /= total;
            }
        }
        Ok(sums)
    }

    /// The `IRPREP` bytes this model loads from.
    ///
    /// # Errors
    ///
    /// When the training data does not fit the format.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        preprocessor::encode_kumo_tabular(&self.training, &self.model.sha256())
    }

    /// Load the model that [`Self::to_bytes`] saved, fitting it again.
    ///
    /// # Errors
    ///
    /// When the bytes are invalid or for another checkpoint, or as [`Self::fit`].
    pub fn from_bytes(model: Arc<Model>, bytes: &[u8], threads: Option<usize>) -> Result<Self> {
        let (checkpoint, training) = preprocessor::decode_kumo_tabular(bytes)?;
        if checkpoint != model.sha256() {
            return Err(Error::Artifact(
                "the saved model was fitted with another checkpoint".into(),
            ));
        }
        Self::fit(model, training, threads)
    }
}

/// Each class's mean log probability over the codebook tasks that separate
/// it, `[rows, classes]`, from each task's `[rows, 10]` logits.
///
/// Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/base/ecoc.py#L125-L139>.
#[expect(clippy::cast_precision_loss, reason = "Task counts are small")]
fn ecoc_scores(logits: &[Vec<f32>], codebook: &[Vec<usize>], classes: usize) -> Vec<f32> {
    let rows = logits[0].len() / MAX_CLASSES;
    let mut scores = vec![0.0; rows * classes];
    for row in 0..rows {
        for (task, symbols) in logits.iter().zip(codebook) {
            let values = &task[row * MAX_CLASSES..][..MAX_CLASSES];
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_total = values.iter().map(|v| (v - maximum).exp()).sum::<f32>().ln();
            for (class, &symbol) in symbols.iter().enumerate() {
                if symbol != MAX_CLASSES - 1 {
                    scores[row * classes + class] += values[symbol] - maximum - log_total;
                }
            }
        }
        for class in 0..classes {
            let active = codebook
                .iter()
                .filter(|symbols| symbols[class] != MAX_CLASSES - 1)
                .count();
            scores[row * classes + class] /= active as f32;
        }
    }
    scores
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{Fitted, Training};
    use crate::models::kumo_tabular::cpu::tests::model;
    use crate::models::kumo_tabular::preprocess::fixtures::{case, rows};

    fn training(name: &str) -> (crate::models::kumo_tabular::preprocess::fixtures::Case, Training) {
        let case = case(name);
        let categories = case
            .categorical
            .iter()
            .map(|&categorical| categorical.then(|| crate::Labels::Int(Vec::new())))
            .collect();
        let training = Training {
            x: case.x[..case.context * case.features].to_vec(),
            categories,
            labels: case.labels.clone(),
            members: case.json["members"].as_array().unwrap().len(),
            seed: 0,
            feature_names: None,
        };
        (case, training)
    }

    #[test]
    #[ignore = "needs IRONICL_CHECKPOINT_FIXTURES with the Kumo-Tabular checkpoints"]
    fn probabilities_match_nvidias() {
        for (name, size) in [
            ("numeric", "small"),
            ("mixed", "small"),
            ("many_classes", "small"),
        ] {
            let (case, training) = training(name);
            let fitted =
                Fitted::fit_with(Arc::new(model(size, "classifier")), training, case.draws.clone()).unwrap();
            let probabilities = fitted
                .predict_proba(&case.x[case.context * case.features..])
                .unwrap();
            // NVIDIA's columns are the first member's class order.
            let columns: Vec<String> = case.json["prediction"]["columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().into())
                .collect();
            let classes = match fitted.classes().unwrap() {
                crate::Labels::Text(names) => names.clone(),
                crate::Labels::Int(values) => values.iter().map(ToString::to_string).collect(),
                _ => unreachable!(),
            };
            let expected = rows(&case.json["prediction"]["values"]);
            let mut worst = 0.0_f64;
            for (row, values) in expected.iter().enumerate() {
                for (column, value) in columns.iter().zip(values) {
                    let class = classes.iter().position(|c| c == column).unwrap();
                    worst = worst.max((f64::from(probabilities[row * classes.len() + class]) - value).abs());
                }
            }
            println!("{name}: max |probability error| {worst:.2e}");
            assert!(worst < 1e-5, "{name}: {worst}");
        }
    }

    #[test]
    #[ignore = "needs IRONICL_CHECKPOINT_FIXTURES with the Kumo-Tabular checkpoints"]
    fn quantiles_match_nvidias() {
        for (name, size) in [
            ("numeric_regression", "small"),
            ("mixed_regression_large", "large"),
        ] {
            let (case, training) = training(name);
            let fitted =
                Fitted::fit_with(Arc::new(model(size, "regressor")), training, case.draws.clone()).unwrap();
            let x = &case.x[case.context * case.features..];
            let outputs = [
                crate::RegressionOutput::RawQuantiles,
                crate::RegressionOutput::Mean,
            ];
            let results = fitted.predict_regression(x, &outputs).unwrap();
            let expected = rows(&case.json["prediction"]["values"]);
            let mut worst = 0.0_f64;
            for (row, values) in expected.iter().enumerate() {
                for (index, value) in values.iter().enumerate() {
                    let error =
                        (f64::from(results[0].0[row * 999 + index]) - value).abs() / (1.0 + value.abs());
                    worst = worst.max(error);
                }
                let mean = values.iter().sum::<f64>() / 999.0;
                assert!(
                    (f64::from(results[1].0[row]) - mean).abs() < 1e-4 * (1.0 + mean.abs()),
                    "{name} mean"
                );
            }
            println!("{name}: max relative quantile error {worst:.2e}");
            assert!(worst < 1e-5, "{name}: {worst}");
        }
    }
}
