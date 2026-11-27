//! A fitted `TabICLv2` model: preprocessing, the training context, and what
//! turns predictions into the caller's units (class indices, or the
//! regressor's target scale). Python's estimators run on it.
//!
//! Fitting is [`Preprocessed::fit`], whose shape plans the memory, then
//! [`Executor::build_fitted`]. A fitted model saves its preprocessing
//! ([`FittedModel::preprocessor_bytes`]), with or without its preprocessed
//! training rows, and optionally its context;
//! [`Executor::load_fitted`] loads them, or rebuilds the context from the
//! training rows.

use std::{path::Path, sync::Arc};

use super::preprocess::{
    Features, PreprocessOptions, Preprocessor, Targets, flatten_orders, invalid, matrix_rows,
};
use crate::preprocess::labels::Labels;
use crate::{
    ClassifierAggregation, ClassifierProbabilities, Error, Executor, Optimize, QueryRequest,
    RegressionOutput, Result, StaticExecutor, TaskKind,
    format::{
        encode_sha256, preprocessor,
        static_context::{StaticContextArtifact, StaticContextMetadata},
    },
    memory::MemoryLimits,
    progress::{self, Stage},
};

/// Query rows preprocessed at once, which bounds the `[B, rows, H]` copies a
/// prediction makes. Rust splits each chunk further to fit its own buffers.
const PREDICT_CHUNK_ROWS: usize = 8192;

/// The preprocessed training rows: each group's `[rows, H]` matrix and each
/// row's target, a class index or a standardized value.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Training {
    pub(crate) group_rows: Vec<Vec<f32>>,
    pub(crate) targets: Vec<f32>,
}

/// Fitted preprocessing, with the training rows a context is built from.
#[derive(Clone, Debug, PartialEq)]
pub struct Preprocessed {
    pub(crate) preprocessor: Preprocessor,
    /// Absent when loaded with a context and without the training rows.
    pub(crate) training: Option<Training>,
}

/// What [`Executor::build_context`] takes: `[B, N, H]` rows, `[B, N]` targets,
/// and each member's class and feature order.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingInputs {
    pub rows: Vec<f32>,
    pub shape: [usize; 3],
    pub targets: Vec<f32>,
    pub class_permutations: Vec<u64>,
    pub feature_permutations: Vec<u64>,
}

impl Preprocessed {
    /// Fit preprocessing on `x`, `[rows, columns]`, whose features have
    /// `categories`, and preprocess it.
    ///
    /// # Errors
    ///
    /// As [`Preprocessor::fit`].
    pub fn fit(
        options: &PreprocessOptions,
        x: Features<'_>,
        columns: usize,
        categories: Vec<Option<Labels>>,
        targets: &Targets,
    ) -> Result<Self> {
        let preprocessor = Preprocessor::fit(options, x, columns, categories, targets)?;
        Self::with_training(preprocessor, x, columns, targets)
    }

    /// `preprocessor` with the training rows `x`, `[rows, columns]`, and their
    /// targets, preprocessed as fitting preprocesses them.
    fn with_training(
        preprocessor: Preprocessor,
        x: Features<'_>,
        columns: usize,
        targets: &Targets,
    ) -> Result<Self> {
        let group_rows = preprocessor.transform(x, columns)?;
        let rows = x.len() / columns;
        let targets = match targets {
            Targets::Labels(labels) if labels.len() == rows => {
                codes_as_targets(&preprocessor.encode_labels(labels)?)
            }
            Targets::Values(values) if values.len() == rows => {
                let (mean, scale) = preprocessor
                    .target_scale
                    .ok_or_else(|| invalid("a classifier's targets are class labels"))?;
                #[expect(clippy::cast_possible_truncation, reason = "The model reads FP32")]
                let scaled = values
                    .iter()
                    .map(|&value| ((f64::from(value) - mean) / scale) as f32)
                    .collect();
                scaled
            }
            _ => return Err(invalid("X and y must have the same length")),
        };
        Ok(Self {
            preprocessor,
            training: Some(Training { group_rows, targets }),
        })
    }

    #[must_use]
    pub fn preprocessor(&self) -> &Preprocessor {
        &self.preprocessor
    }

    #[must_use]
    pub fn task(&self) -> TaskKind {
        self.preprocessor.task()
    }

    /// Whether the training rows are kept, so a context can be built.
    #[must_use]
    pub fn has_training(&self) -> bool {
        self.training.is_some()
    }

    /// The `[B, N, H]` shape of the context these training rows build.
    #[must_use]
    pub fn shape(&self) -> Option<[usize; 3]> {
        let features = self.preprocessor.n_features_out();
        self.training
            .as_ref()
            .map(|training| [self.preprocessor.member_count(), training.targets.len(), features])
    }

    /// Each member's training rows and targets, for [`Executor::build_context`].
    ///
    /// # Errors
    ///
    /// When the training rows were not kept.
    pub fn training_inputs(&self) -> Result<TrainingInputs> {
        let training = self
            .training
            .as_ref()
            .ok_or_else(|| invalid("the training rows were not kept, so no context can be built"))?;
        let (rows, shape) = self.preprocessor.ensemble_rows(&training.group_rows)?;
        let targets = match self.preprocessor.classes() {
            None => training.targets.repeat(shape[0]),
            Some(_) => self
                .preprocessor
                .class_permutations()
                .iter()
                .flat_map(|order| {
                    let order = codes_as_targets(order);
                    training.targets.iter().map(move |&code| order[class_index(code)])
                })
                .collect(),
        };
        Ok(TrainingInputs {
            rows,
            shape,
            targets,
            class_permutations: flatten_orders(self.preprocessor.class_permutations()),
            feature_permutations: flatten_orders(self.preprocessor.feature_permutations()),
        })
    }

    /// Each member's preprocessed rows of `x`, `[rows, columns]`, as FP32 `[B, rows, H]`.
    ///
    /// # Errors
    ///
    /// As [`Preprocessor::transform`].
    pub fn query_rows(&self, x: Features<'_>, columns: usize) -> Result<(Vec<f32>, [usize; 3])> {
        let group_rows = self.preprocessor.transform(x, columns)?;
        self.preprocessor.ensemble_rows(&group_rows)
    }

    /// Run `predict` on the preprocessed rows of `x`, `[rows, columns]`, `step`
    /// rows at a time, reporting rows done of all rows.
    fn chunks<T>(
        &self,
        x: Features<'_>,
        columns: usize,
        step: usize,
        mut predict: impl FnMut(QueryRequest<'_>) -> Result<T>,
    ) -> Result<Vec<T>> {
        let rows = matrix_rows(x, columns)?;
        if rows == 0 {
            return Err(invalid("X must be a nonempty matrix of features"));
        }
        let mut results = Vec::new();
        let mut start = 0;
        while start < rows {
            let end = (start + step).min(rows);
            let (stacked, [members, chunk, features]) =
                self.query_rows(x.rows(start, end, columns), columns)?;
            let request = QueryRequest::new(members, chunk, features, &stacked)?;
            results.push(progress::with_offset(start, rows, || predict(request))?);
            progress::report(Stage::Predict, end, rows)?;
            start = end;
        }
        Ok(results)
    }
}

/// A class index stored as an exact small `f32` target.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Validated class indices are exact small integers"
)]
fn class_index(target: f32) -> usize {
    target as usize
}

#[expect(clippy::cast_precision_loss, reason = "Class indices are below ten")]
fn codes_as_targets(codes: &[usize]) -> Vec<f32> {
    codes.iter().map(|&code| code as f32).collect()
}

/// `values` of `output` from standardized units to the target's, in `f64` and
/// then FP32.
///
/// Equivalent of <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/regressor.py#L755-L770>.
fn unscale(values: &mut [f32], output: &RegressionOutput, (mean, scale): (f64, f64)) {
    // The variance is a squared spread, so it scales by σ² and the mean drops
    // out. Every other output is a location in target units, σ·v + μ.
    let variance = matches!(output, RegressionOutput::Variance);
    for value in values {
        let wide = f64::from(*value);
        #[expect(clippy::cast_possible_truncation, reason = "Outputs are FP32")]
        let unscaled = (if variance {
            wide * (scale * scale)
        } else {
            wide * scale + mean
        }) as f32;
        *value = unscaled;
    }
}

/// A fitted model: its preprocessing and its training context.
#[derive(Debug, Clone)]
pub struct FittedModel {
    pub(crate) preprocessed: Arc<Preprocessed>,
    /// Shared, not copied: an owned context holds its tensors itself.
    pub(crate) context: Arc<StaticExecutor>,
}

impl FittedModel {
    #[must_use]
    pub fn preprocessed(&self) -> &Arc<Preprocessed> {
        &self.preprocessed
    }

    #[must_use]
    pub fn task(&self) -> TaskKind {
        self.preprocessed.task()
    }

    /// Where predictions run; see [`StaticExecutor::device`].
    #[must_use]
    pub fn device(&self) -> &'static str {
        self.context.device()
    }

    /// See [`StaticExecutor::batch_rows`].
    #[must_use]
    pub fn batch_rows(&self) -> Option<usize> {
        self.context.batch_rows()
    }

    /// Class probabilities `[rows, classes]` for `x`, `[rows, columns]`.
    ///
    /// # Errors
    ///
    /// For a regressor, invalid `x`, or a failed prediction.
    pub fn predict_probabilities(
        &self,
        x: Features<'_>,
        columns: usize,
        aggregation: ClassifierAggregation,
    ) -> Result<ClassifierProbabilities> {
        let classes = self
            .preprocessed
            .preprocessor
            .classes()
            .map(Labels::len)
            .ok_or_else(|| invalid("a regressor has no class probabilities"))?;
        let chunks = self.chunks(x, columns, |request| {
            self.context.predict_probabilities(request, aggregation)
        })?;
        let values = chunks
            .iter()
            .flat_map(|chunk| chunk.values().iter().copied())
            .collect();
        Ok(ClassifierProbabilities::from_rows(values, classes))
    }

    /// Each requested regression output for `x`, `[rows, columns]`, in the
    /// target's units, with its column count.
    ///
    /// # Errors
    ///
    /// For a classifier, invalid `x`, or a failed prediction.
    pub fn predict_regression(
        &self,
        x: Features<'_>,
        columns: usize,
        outputs: &[RegressionOutput],
    ) -> Result<Vec<(Vec<f32>, usize)>> {
        let rows = matrix_rows(x, columns)?;
        Ok(self
            .predict_regression_batch(x, columns, &[(rows, outputs)])?
            .pop()
            .unwrap_or_default())
    }

    /// Like [`Self::predict_regression`], for a batch of queries whose row
    /// counts sum to `x`'s rows: the neural pass runs once on the whole batch
    /// and each query summarizes only its own rows with its own outputs.
    pub(crate) fn predict_regression_batch(
        &self,
        x: Features<'_>,
        columns: usize,
        queries: &[(usize, &[RegressionOutput])],
    ) -> Result<Vec<crate::models::quantiles::RegressionOutputs>> {
        let target_scale = self
            .preprocessed
            .preprocessor
            .target_scale
            .ok_or_else(|| invalid("a classifier has no regression outputs"))?;
        let mut results: Vec<crate::models::quantiles::RegressionOutputs> = queries
            .iter()
            .map(|(_, outputs)| vec![(Vec::new(), 0); outputs.len()])
            .collect();
        let rows = matrix_rows(x, columns)?;
        if rows == 0 {
            return Err(invalid("X must be a nonempty matrix of features"));
        }
        let step = self
            .context
            .batch_rows()
            .unwrap_or(PREDICT_CHUNK_ROWS)
            .clamp(1, super::cpu::QUERY_CHUNK_ROWS);
        let mut start = 0;
        while start < rows {
            let end = (start + step).min(rows);
            let (stacked, [members, chunk, features]) = self
                .preprocessed
                .query_rows(x.rows(start, end, columns), columns)?;
            let request = QueryRequest::new(members, chunk, features, &stacked)?;
            let decoded = progress::with_offset(start, rows, || self.context.predict_logits(request))?;
            crate::models::quantiles::extend_query_summaries(&decoded, start, queries, &mut results)?;
            progress::report(Stage::Predict, end, rows)?;
            start = end;
        }
        for (query_results, &(_, outputs)) in results.iter_mut().zip(queries) {
            for ((values, _), output) in query_results.iter_mut().zip(outputs) {
                unscale(values, output, target_scale);
            }
        }
        Ok(results)
    }

    /// Run `predict` over `x` a chunk at a time: one pass's rows when the
    /// context streams its cache, else [`PREDICT_CHUNK_ROWS`].
    fn chunks<T>(
        &self,
        x: Features<'_>,
        columns: usize,
        predict: impl FnMut(QueryRequest<'_>) -> Result<T>,
    ) -> Result<Vec<T>> {
        let step = self.context.batch_rows().unwrap_or(PREDICT_CHUNK_ROWS).max(1);
        self.preprocessed.chunks(x, columns, step, predict)
    }

    /// The preprocessing as container bytes, for [`Executor::load_fitted`], with
    /// the training rows when `training` is set and they were kept.
    ///
    /// # Errors
    ///
    /// When a text label is longer than 4 GiB.
    pub fn preprocessor_bytes(&self, training: bool) -> Result<Vec<u8>> {
        let metadata = self.context.context_metadata();
        preprocessor::encode(
            &self.preprocessed.preprocessor,
            self.preprocessed.training.as_ref().filter(|_| training),
            &metadata.model_sha256,
            &metadata.context_sha256,
        )
    }
}

/// Without training rows we cannot recompute the identity. Check the saved binding
/// and the parts of the serving contract the preprocessor still knows.
fn matches_without_rows(
    preprocessor: &Preprocessor,
    identity: Option<&str>,
    m: &StaticContextMetadata,
) -> bool {
    identity == Some(encode_sha256(&m.context_sha256).as_str())
        && m.ensemble_count == preprocessor.member_count()
        && m.feature_count == preprocessor.n_features_out()
        && m.class_count == preprocessor.classes().map_or(0, Labels::len)
        && m.class_permutations == flatten_orders(preprocessor.class_permutations())
        && m.feature_permutations == flatten_orders(preprocessor.feature_permutations())
}

impl Executor {
    /// Build the context of `preprocessed`, as [`Self::build_context`] does.
    ///
    /// # Errors
    ///
    /// When the preprocessing is for another task or has no training rows, or
    /// as [`Self::build_context`].
    pub fn build_fitted(
        &self,
        preprocessed: Arc<Preprocessed>,
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
        cache_dir: Option<&Path>,
    ) -> Result<FittedModel> {
        if preprocessed.task() != self.task() {
            return Err(invalid(format!(
                "the preprocessing is for a {}, but the model is a {}",
                preprocessed.task(),
                self.task()
            )));
        }
        let inputs = preprocessed.training_inputs()?;
        let [members, rows, features] = inputs.shape;
        let context = self.build_context(
            members,
            rows,
            features,
            &inputs.rows,
            &inputs.targets,
            inputs.class_permutations,
            inputs.feature_permutations,
            optimize,
            threads,
            limits,
            cache_dir,
        )?;
        Ok(FittedModel {
            preprocessed,
            context: Arc::new(context),
        })
    }

    /// Load a fitted model from its preprocessing bytes and optional `context`
    /// bytes. Without a context, one is built from the training rows in the
    /// preprocessing bytes (through `cache_dir`) as fitting builds it.
    ///
    /// # Errors
    ///
    /// When the bytes are invalid or were fitted with another checkpoint, the
    /// context does not match the preprocessing, or as [`Self::build_fitted`].
    pub fn load_fitted(
        &self,
        preprocessor: &[u8],
        context: Option<&[u8]>,
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
        cache_dir: Option<&Path>,
    ) -> Result<FittedModel> {
        let (preprocessed, saved_identity) = self.decode_preprocessed(preprocessor)?;
        let preprocessed = Arc::new(preprocessed);
        let Some(context) = context else {
            return self.build_fitted(preprocessed, optimize, threads, limits, cache_dir);
        };
        self.restore_fitted(
            preprocessed,
            saved_identity.as_deref(),
            StaticContextArtifact::from_bytes(context)?,
            optimize,
            threads,
            limits,
        )
    }

    /// Validate the fit binding before placing the context on a device. Both
    /// pickles and model directories use this path.
    pub(crate) fn restore_fitted(
        &self,
        preprocessed: Arc<Preprocessed>,
        saved_identity: Option<&str>,
        artifact: StaticContextArtifact,
        optimize: Optimize,
        threads: Option<usize>,
        limits: MemoryLimits,
    ) -> Result<FittedModel> {
        let metadata = artifact.metadata();
        let matches = if preprocessed.has_training() {
            *metadata == self.fitted_context_metadata(&preprocessed, optimize)?
        } else {
            metadata.optimize == optimize
                && matches_without_rows(&preprocessed.preprocessor, saved_identity, metadata)
        };
        if !matches {
            return Err(Error::Artifact(
                "the context does not match its preprocessing".into(),
            ));
        }
        let context = self.load_context_artifact(artifact, optimize, threads, limits)?;
        Ok(FittedModel {
            preprocessed,
            context: Arc::new(context),
        })
    }

    /// Container bytes as preprocessing fitted with this model's checkpoint,
    /// and the saved context identity when training rows were omitted.
    ///
    /// # Errors
    ///
    /// When the bytes are invalid or were fitted with another checkpoint.
    pub(crate) fn decode_preprocessed(&self, bytes: &[u8]) -> Result<(Preprocessed, Option<String>)> {
        let (checkpoint_sha256, preprocessed, identity) = preprocessor::decode(bytes)?;
        let checkpoint = encode_sha256(&checkpoint_sha256);
        if checkpoint != self.source_sha256() {
            return Err(Error::Artifact(format!(
                "this {} was fitted with checkpoint {checkpoint}, not {}",
                preprocessed.task(),
                self.source_sha256()
            )));
        }
        Ok((preprocessed, identity))
    }

    /// The metadata of the context that [`Self::build_fitted`] builds for
    /// `preprocessed` at `optimize`, to tell whether a saved one is it.
    ///
    /// # Errors
    ///
    /// When the training rows were not kept, or as [`Self::build_context`].
    fn fitted_context_metadata(
        &self,
        preprocessed: &Preprocessed,
        optimize: Optimize,
    ) -> Result<StaticContextMetadata> {
        let inputs = preprocessed.training_inputs()?;
        let [members, rows, features] = inputs.shape;
        self.context_metadata(
            (members, rows, features),
            &inputs.rows,
            &inputs.targets,
            inputs.class_permutations,
            inputs.feature_permutations,
            optimize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Preprocessed, matches_without_rows, unscale};
    use crate::{
        Features, Labels, NormMethod, Optimize, PreprocessOptions, RegressionOutput, StaticContextMetadata,
        Targets,
        format::preprocessor::{decode, encode},
    };

    const SHA: [u8; 32] = [5; 32];

    fn x() -> Vec<f64> {
        (0..60).map(|i| f64::from((i * 37 % 23) - 11) / 3.0).collect()
    }

    fn fit(targets: &Targets) -> Preprocessed {
        let options = PreprocessOptions {
            n_estimators: 4,
            random_state: 3,
            norm_methods: vec![NormMethod::None, NormMethod::Power],
            outlier_threshold: 4.0,
        };
        Preprocessed::fit(&options, Features::F64(&x()), 3, vec![None; 3], targets).unwrap()
    }

    fn values(a: f32, b: f32) -> Targets {
        Targets::Values((0..20_u8).map(|i| a * (f32::from(i) * 0.5 - 3.0) + b).collect())
    }

    #[test]
    fn saved_preprocessing_keeps_the_training_rows() {
        for targets in [
            Targets::Labels(Labels::Int((0..20).map(|i| i % 3).collect())),
            values(1.0, 0.0),
        ] {
            let fitted = fit(&targets);
            let bytes = encode(&fitted.preprocessor, fitted.training.as_ref(), &SHA, &[1; 32]).unwrap();
            let (sha, saved, identity) = decode(&bytes).unwrap();
            assert_eq!((sha, identity), (SHA, None));
            assert_eq!(saved, fitted);
            // The saved rows are the ones the raw rows preprocess to.
            let rebuilt =
                Preprocessed::with_training(saved.preprocessor.clone(), Features::F64(&x()), 3, &targets)
                    .unwrap();
            assert_eq!(
                rebuilt.training_inputs().unwrap(),
                saved.training_inputs().unwrap()
            );
        }
    }

    #[test]
    fn a_regressor_predicts_in_the_targets_units() {
        // For targets a·y + b, the standardized targets are y's, so the same
        // standardized prediction v means a·p + b, with a²-times the variance.
        let (a, b) = (3.0, -40.0);
        let (base, shifted) = (fit(&values(1.0, 0.0)), fit(&values(a, b)));
        let predicted = [-1.5_f32, 0.0, 0.25, 2.0];
        let unscaled = |preprocessed: &Preprocessed, output| {
            let mut values = predicted.to_vec();
            unscale(
                &mut values,
                &output,
                preprocessed.preprocessor.target_scale.unwrap(),
            );
            values
        };
        for output in [RegressionOutput::Mean, RegressionOutput::Variance] {
            let (p, q) = (
                unscaled(&base, output.clone()),
                unscaled(&shifted, output.clone()),
            );
            for (p, q) in p.iter().zip(&q) {
                let expected = if output == RegressionOutput::Variance {
                    a * a * p
                } else {
                    a * p + b
                };
                assert!(
                    (q - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                    "{output:?}: {q} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn chunked_prediction_sees_every_row_once() {
        let fitted = fit(&values(1.0, 0.0));
        let (whole, [members, rows, features]) = fitted.query_rows(Features::F64(&x()), 3).unwrap();
        let chunks = fitted
            .chunks(Features::F64(&x()), 3, 6, |request| {
                Ok((request.query_rows(), request.x().to_vec()))
            })
            .unwrap();
        assert_eq!(
            chunks.iter().map(|(rows, _)| *rows).collect::<Vec<_>>(),
            [6, 6, 6, 2]
        );
        for member in 0..members {
            let joined: Vec<f32> = chunks
                .iter()
                .flat_map(|(rows, values)| &values[member * rows * features..][..rows * features])
                .copied()
                .collect();
            assert_eq!(joined, whole[member * rows * features..][..rows * features]);
        }
        assert!(fitted.chunks(Features::F64(&[]), 3, 6, |_| Ok(())).is_err());
    }

    #[test]
    fn a_context_for_other_preprocessing_is_rejected() {
        let fitted = fit(&Targets::Labels(Labels::Int((0..20).map(|i| i % 3).collect())));
        let inputs = fitted.training_inputs().unwrap();
        let [members, rows, features] = inputs.shape;
        let metadata = StaticContextMetadata {
            architecture: crate::format::static_context::ContextArchitecture::TabIclV2,
            model_sha256: SHA,
            context_sha256: [1; 32],
            ensemble_count: members,
            training_rows: rows,
            feature_count: features,
            class_count: 3,
            class_permutations: inputs.class_permutations,
            feature_permutations: inputs.feature_permutations,
            optimize: Optimize::Safe,
        };
        let identity = crate::format::encode_sha256(&metadata.context_sha256);
        let matches = |m| matches_without_rows(&fitted.preprocessor, Some(&identity), m);
        assert!(matches(&metadata));
        assert!(!matches_without_rows(&fitted.preprocessor, None, &metadata));
        let mut changes: Vec<fn(&mut StaticContextMetadata)> = vec![
            |context| context.ensemble_count -= 1,
            |context| context.context_sha256 = [3; 32],
            |context| context.feature_count += 1,
            |context| context.class_count = 2,
            |context| context.class_permutations.swap(0, 1),
            |context| context.feature_permutations.swap(0, 1),
        ];
        for (index, change) in changes.iter_mut().enumerate() {
            let mut other = metadata.clone();
            change(&mut other);
            assert!(
                !matches_without_rows(&fitted.preprocessor, Some(&identity), &other),
                "change {index}"
            );
        }
    }
}
