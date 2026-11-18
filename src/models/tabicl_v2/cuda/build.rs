//! A context's build on the device: the training rows' pass through the
//! column, row, and ICL stages, which keeps each block's keys and values.
//!
//! Estimators build in groups that fit the device. In a group, the column
//! stage runs in chunks of columns: in each of the three blocks, the inducing
//! points attend to the column's training rows, then the rows attend back to
//! them through the cached keys and values. The row stage and the ICL blocks
//! run in runs of rows; each ICL block stores every training row's keys and
//! values, in FP32 for `safe` and FP16 otherwise, before any row attends to
//! them. The caches go to the host
//! as each block finishes, and when the whole context builds at once and
//! they fit, they also stay on the device, encoded as predictions read them,
//! for the context's predictor.

use cuda_core::DeviceBuffer;
use rayon::prelude::*;

use super::{
    COLUMN_BLOCKS, COLUMN_CACHE, COLUMN_DIM, HEAD_DIM, HEADS, ICL_BLOCKS, ICL_DIM, INDUCING, Precision,
    RESERVED,
    cache::{BlockCache, BuiltCache, KeyValues, Scales, units},
    device::{Device, gpu_error},
    launch::{CacheOutput, SmallAttention, padded_rows},
    plan::{MIN_RUN as PREDICT_MIN_RUN, row_bytes},
    predictor::{RowTokens, Stages, upload_all},
    weights::ModelWeights,
    workspace::{Run, Shape, Workspace},
};
use crate::{
    error::{Error, Result},
    progress::Stage,
};

/// Most rows per run of the row and ICL stages.
const MAX_RUN: usize = 1024;
/// Fewest rows per run.
const MIN_RUN: usize = 64;
/// Fewest training rows the device builds for; below, the CPU is as fast.
const MIN_ROWS: usize = 64;

/// What a build reads.
pub struct BuildInputs<'a> {
    /// Estimators, features, and training rows; the outputs are unused.
    pub shape: Shape,
    /// Preprocessed training rows `[B, N, H]`; no grouped triple is the
    /// all-missing marker.
    pub x: &'a [f32],
    /// Label embeddings for the column stage, `[B, N, 128]`.
    pub column_labels: &'a [f32],
    /// Label embeddings for the ICL stage, `[B, N, 512]`.
    pub icl_labels: &'a [f32],
    /// Each column block's inducing points as its first attention's queries,
    /// after `norm1`, the query projection, and `QASSMax`: `[128, 128]`.
    pub inducing_queries: &'a [Vec<f32>],
    /// Each ICL block's `QASSMax` base scale, `[8, 64]`.
    pub base_scales: &'a [Vec<f32>],
    /// The row transformer's `RoPE` frequencies.
    pub rope_frequencies: &'a [f32],
    /// The row transformer's CLS tokens, at least `[4, 128]`.
    pub cls_tokens: &'a [f32],
    /// The device memory the build may use, in bytes.
    pub budget: usize,
}

/// Where a build writes the context.
pub struct BuildOutputs<'a> {
    /// Each column block's keys and values, `[B, H + 4, 8, 128, 16]`; the
    /// build leaves the reserved streams as they are.
    pub column: Vec<(&'a mut [f32], &'a mut [f32])>,
    /// Each ICL block's keys and values, `[B, 8, N, 64]`, as bytes of FP32
    /// values at [`Precision::Fp32`] and of FP16 ones otherwise (see
    /// [`context_bytes`]).
    pub icl: Vec<(&'a mut [u8], &'a mut [u8])>,
}

/// How a build uses device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plan {
    /// Estimators per group.
    group: usize,
    /// Columns per chunk of the column stage.
    columns: usize,
    /// Rows per run of the row and ICL stages.
    run: usize,
    /// Whether the ICL keys and values stay on the device.
    keep: bool,
}

/// Bytes per ICL key or value in a context built at `precision`: FP32 at
/// [`Precision::Fp32`], FP16 otherwise.
#[must_use]
pub const fn context_bytes(precision: Precision) -> usize {
    match precision {
        Precision::Fp32 => 4,
        Precision::Split | Precision::Half => 2,
    }
}

/// Device bytes one estimator of a group keeps through its build with `H`
/// features and `N` rows at `precision`: its column embeddings, row states,
/// label embeddings, and training rows, and an ICL block's keys and values.
const fn estimator_bytes(precision: Precision, h: usize, n: usize) -> usize {
    4 * n * (h * COLUMN_DIM + 2 * ICL_DIM + COLUMN_DIM + h)
        + 2 * context_bytes(precision) * HEADS * padded_rows(n) * HEAD_DIM
}

/// Device bytes one column of a chunk of the column stage takes: its state
/// and in-projection over `N` rows, and its inducing points' state,
/// attention, in-projection, and keys and values.
const fn column_bytes(n: usize) -> usize {
    4 * (n * 4 * COLUMN_DIM + INDUCING * (2 * COLUMN_DIM + 3 * COLUMN_DIM + 2 * COLUMN_DIM))
}

/// The plan for a build of `shape` within `budget` bytes, a tenth of which
/// stays free for the driver and fragmentation, or `None` when one estimator
/// does not fit. The kernels index their buffers with 32-bit sizes, which
/// caps groups and chunks too. The ICL keys and values stay on the device,
/// stored at `precision`, when every estimator builds at once and they fit
/// with the build and with room to predict.
fn plan(budget: usize, precision: Precision, shape: Shape) -> Option<Plan> {
    let Shape {
        ensembles: b,
        features: h,
        rows: n,
        ..
    } = shape;
    let budget = budget / 10 * 9;
    let limit = u32::MAX as usize;
    let build = |group: usize| {
        group * estimator_bytes(precision, h, n) + column_bytes(n).max(MIN_RUN * row_bytes((group, h)))
    };
    let group = (1..=b)
        .rev()
        .find(|&group| build(group) <= budget && group * h * n * COLUMN_DIM <= limit)?;
    let free = budget - group * estimator_bytes(precision, h, n);
    let columns = (free / column_bytes(n))
        .min(limit / (n * 3 * COLUMN_DIM))
        .clamp(1, group * h);
    let run = (free / row_bytes((group, h)))
        .min(MAX_RUN)
        .min(n.next_multiple_of(MIN_RUN))
        .max(MIN_RUN)
        / MIN_RUN
        * MIN_RUN;
    let kept = ICL_BLOCKS * 4 * units(precision, b * HEADS, n);
    let keep = group == b && build(b) + kept + 2 * PREDICT_MIN_RUN * row_bytes((b, h)) <= budget;
    Some(Plan {
        group,
        columns,
        run,
        keep,
    })
}

/// Whether a context of `shape` builds on the device at `precision` within
/// `budget` bytes.
#[must_use]
pub fn fits(budget: usize, precision: Precision, shape: Shape) -> bool {
    shape.rows >= MIN_ROWS && plan(budget, precision, shape).is_some()
}

/// Build a context's caches into `outputs`, with products at `precision`.
/// Returns the ICL keys and values when they stay on the device, encoded for
/// predictions at `precision`.
///
/// # Errors
///
/// Returns an error for inconsistent inputs, a context that does not
/// [`fits`], a failed launch or copy, a cached value FP16 cannot hold, or the
/// progress hook's.
pub fn build(
    weights: &ModelWeights,
    precision: Precision,
    inputs: &BuildInputs<'_>,
    outputs: &mut BuildOutputs<'_>,
) -> Result<Option<BuiltCache>> {
    check(precision, inputs, outputs)?;
    let Some(plan) = plan(inputs.budget, precision, inputs.shape) else {
        return Err(Error::InvalidArgument(
            "the context does not fit the CUDA device".into(),
        ));
    };
    let device = &weights.device;
    let Shape {
        ensembles,
        features,
        rows,
        ..
    } = inputs.shape;
    let tokens = RowTokens::upload(device, features, inputs.rope_frequencies, inputs.cls_tokens)?;
    let base_scales = upload_all(device, inputs.base_scales)?;
    let builder = Builder {
        stages: Stages {
            weights,
            precision,
            tokens: &tokens,
            base_scales: &base_scales,
        },
        plan,
        features,
        rows,
        queries: upload_all(device, inputs.inducing_queries)?,
    };
    crate::progress::report(Stage::Build, 0, ensembles)?;
    let mut built = None;
    for first in (0..ensembles).step_by(plan.group) {
        let group = plan.group.min(ensembles - first);
        built = builder.group(inputs, outputs, (first, group))?;
        crate::progress::report(Stage::Build, first + group, ensembles)?;
    }
    Ok(built)
}

/// Check that the inputs and outputs hold the shapes they claim.
fn check(precision: Precision, inputs: &BuildInputs<'_>, outputs: &BuildOutputs<'_>) -> Result<()> {
    let Shape {
        ensembles: b,
        features: h,
        rows: n,
        ..
    } = inputs.shape;
    let consistent = inputs.x.len() == b * n * h
        && inputs.column_labels.len() == b * n * COLUMN_DIM
        && inputs.icl_labels.len() == b * n * ICL_DIM
        && inputs.inducing_queries.len() == COLUMN_BLOCKS
        && inputs
            .inducing_queries
            .iter()
            .all(|queries| queries.len() == INDUCING * COLUMN_DIM)
        && inputs.base_scales.len() == ICL_BLOCKS
        && inputs.base_scales.iter().all(|scale| scale.len() == ICL_DIM)
        && outputs.column.len() == COLUMN_BLOCKS
        && outputs.column.iter().all(|(keys, values)| {
            keys.len() == b * (h + RESERVED) * COLUMN_CACHE && values.len() == keys.len()
        })
        && outputs.icl.len() == ICL_BLOCKS
        && outputs.icl.iter().all(|(keys, values)| {
            keys.len() == b * n * ICL_DIM * context_bytes(precision) && values.len() == keys.len()
        });
    if consistent {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "CUDA build inputs and outputs do not match their shape".into(),
        ))
    }
}

/// What every group of a build shares.
struct Builder<'a> {
    stages: Stages<'a>,
    plan: Plan,
    features: usize,
    rows: usize,
    /// Each column block's inducing points as queries, `[128, 128]`.
    queries: Vec<DeviceBuffer<f32>>,
}

impl Builder<'_> {
    fn device(&self) -> &Device {
        &self.stages.weights.device
    }

    /// Estimators `first..first + group`: the column stage, the row stage,
    /// and the ICL blocks, whose keys and values it returns when they stay on
    /// the device.
    fn group(
        &self,
        inputs: &BuildInputs<'_>,
        outputs: &mut BuildOutputs<'_>,
        (first, group): (usize, usize),
    ) -> Result<Option<BuiltCache>> {
        let shape = Shape {
            ensembles: group,
            features: self.features,
            rows: self.rows,
            outputs: 1,
        };
        // The column stage's buffers are gone before the workspace takes their
        // place, and the column embeddings once the row stage has read them.
        let mut work = Workspace::default();
        let mut runs = {
            let embeddings = self.columns(inputs, outputs, (first, group))?;
            work.reserve(self.device(), self.stages.precision, shape, self.plan.run)
                .map_err(gpu_error)?;
            self.row_runs(&mut work, inputs, &embeddings, (first, group))?
        };
        self.icl(&mut work, &mut runs, outputs, (first, group))
    }

    /// The column stage of estimators `first..first + group`, chunk by chunk
    /// of columns, into the column caches; returns the column embeddings
    /// `[group * H, N, 128]`.
    fn columns(
        &self,
        inputs: &BuildInputs<'_>,
        outputs: &mut BuildOutputs<'_>,
        (first, group): (usize, usize),
    ) -> Result<DeviceBuffer<f32>> {
        let device = self.device();
        let (features, rows) = (self.features, self.rows);
        let training = device.upload(&inputs.x[first * rows * features..][..group * rows * features])?;
        let labels =
            device.upload(&inputs.column_labels[first * rows * COLUMN_DIM..][..group * rows * COLUMN_DIM])?;
        let mut embeddings = device
            .zeroed::<f32>(group * features * rows * COLUMN_DIM)
            .map_err(gpu_error)?;
        let mut chunk = ColumnChunk::new(device, self.plan.columns.min(group * features), rows)?;
        for start in (0..group * features).step_by(chunk.columns) {
            let count = chunk.columns.min(group * features - start);
            device.column_train(
                (&training, &labels),
                &self.stages.weights.column_input,
                &mut chunk.state,
                (features, rows),
                (start, count),
            )?;
            for block in 0..COLUMN_BLOCKS {
                self.column_block(&mut chunk, block, count)?;
                let (keys, values) = (device.download(&chunk.keys)?, device.download(&chunk.values)?);
                let (keys_out, values_out) = &mut outputs.column[block];
                for column in 0..count {
                    let (estimator, feature) = ((start + column) / features, (start + column) % features);
                    let to =
                        ((first + estimator) * (features + RESERVED) + feature + RESERVED) * COLUMN_CACHE;
                    let from = column * COLUMN_CACHE;
                    for (out, host) in [(&mut **keys_out, &keys), (&mut **values_out, &values)] {
                        out[to..][..COLUMN_CACHE].copy_from_slice(&host[from..][..COLUMN_CACHE]);
                    }
                }
            }
            device.copy_into(
                &chunk.state,
                &mut embeddings,
                (start * rows * COLUMN_DIM, count * rows * COLUMN_DIM),
            )?;
        }
        let streams = features + RESERVED;
        let built = (first * streams * COLUMN_CACHE)..((first + group) * streams * COLUMN_CACHE);
        let finite = outputs.column.iter().all(|(keys, values)| {
            keys[built.clone()]
                .iter()
                .chain(&values[built.clone()])
                .all(|value| value.is_finite())
        });
        if !finite {
            return Err(Error::InvalidArgument(
                "the built context contains a non-finite value".into(),
            ));
        }
        Ok(embeddings)
    }

    /// Column block `block` on the chunk's first `count` columns: the
    /// inducing points attend to each column's training rows, then the rows
    /// attend back through the inducing points' keys and values, which stay
    /// in the chunk's cache.
    fn column_block(&self, chunk: &mut ColumnChunk, block: usize, count: usize) -> Result<()> {
        let Stages {
            weights,
            precision,
            tokens,
            ..
        } = self.stages;
        let device = &weights.device;
        let rows = self.rows;
        let column = &weights.columns[block];
        device.token_head(
            precision,
            (&chunk.state, &mut chunk.projected),
            &column.inducing_attention,
            tokens.rope(),
            (count * rows, 3 * COLUMN_DIM, 0),
        )?;
        // The same 128 queries for every column.
        let layout = SmallAttention {
            seqs: count,
            lq: INDUCING,
            lk: rows,
            inner: 1,
            q_seq: 0,
            q_tok: COLUMN_DIM,
            k_base: COLUMN_DIM,
            v_base: 2 * COLUMN_DIM,
            kv_outer: rows * 3 * COLUMN_DIM,
            kv_inner: 0,
            kv_tok: 3 * COLUMN_DIM,
            kv_head: 16,
            o_seq: INDUCING * COLUMN_DIM,
            o_tok: COLUMN_DIM,
        };
        device.attention16(
            precision,
            (&self.queries[block], &chunk.projected, &chunk.projected),
            &mut chunk.attended,
            &layout,
        )?;
        device.repeat(
            &column.inducing_points,
            &mut chunk.inducing,
            (INDUCING * COLUMN_DIM, count * INDUCING * COLUMN_DIM),
        )?;
        device.token_tail(
            precision,
            (&chunk.attended, &mut chunk.inducing),
            &column.inducing_attention,
            count * INDUCING,
        )?;
        device.token_head(
            precision,
            (&chunk.inducing, &mut chunk.inducing_projected),
            &column.attention,
            tokens.rope(),
            (count * INDUCING, 3 * COLUMN_DIM, 0),
        )?;
        device.column_kv(
            &chunk.inducing_projected,
            (&mut chunk.keys, &mut chunk.values),
            count,
        )?;
        device.column_block(
            precision,
            &mut chunk.state,
            &column.attention,
            (&chunk.keys, &chunk.values),
            (count, 1, rows, 0),
        )
    }

    /// The row stage of estimators `first..first + group` from their column
    /// embeddings, plus their ICL label embeddings, run by run of rows.
    fn row_runs(
        &self,
        work: &mut Workspace,
        inputs: &BuildInputs<'_>,
        embeddings: &DeviceBuffer<f32>,
        (first, group): (usize, usize),
    ) -> Result<Vec<Run<DeviceBuffer<f32>>>> {
        let device = self.device();
        let rows = self.rows;
        let labels = device.upload(&inputs.icl_labels[first * rows * ICL_DIM..][..group * rows * ICL_DIM])?;
        (0..rows)
            .step_by(self.plan.run)
            .map(|start| {
                let len = self.plan.run.min(rows - start);
                let mut hidden = device.zeroed::<f32>(group * len * ICL_DIM).map_err(gpu_error)?;
                self.stages.row_stage(
                    work,
                    Some(embeddings),
                    &mut hidden,
                    (group, self.features, len),
                    (rows, start),
                )?;
                device.add_labels(&mut hidden, &labels, (group, len, ICL_DIM), (rows, start))?;
                Ok(Run { start, len, hidden })
            })
            .collect()
    }

    /// The twelve ICL blocks on the runs' row states `[group, len, 512]`:
    /// each block's keys and values for every training row, FP32 at
    /// [`Precision::Fp32`] and FP16 otherwise, then each run's attention to
    /// them; the keys and values then go to the host, and stay on the device
    /// when the plan keeps them.
    fn icl(
        &self,
        work: &mut Workspace,
        runs: &mut [Run<DeviceBuffer<f32>>],
        outputs: &mut BuildOutputs<'_>,
        (first, group): (usize, usize),
    ) -> Result<Option<BuiltCache>> {
        let device = self.device();
        let precision = self.stages.precision;
        let (rows, padded, pairs) = (self.rows, padded_rows(self.rows), group * HEADS);
        let len = pairs * padded * HEAD_DIM;
        let ones = device.upload(&vec![1.0f32; pairs * HEAD_DIM])?;
        let mut built = self.plan.keep.then(|| BuiltCache::new(precision));
        for block in 0..ICL_BLOCKS {
            // A kept `fast` cache is FP8, encoded from the FP16 one.
            let mut fp8 = None;
            let (stored, read_at) = if precision == Precision::Fp32 {
                let zeroed = || device.zeroed::<f32>(len).map_err(gpu_error);
                let (mut keys, mut values) = (zeroed()?, zeroed()?);
                self.keys_values(work, runs, block, group, CacheOutput::F32(&mut keys, &mut values))?;
                (KeyValues::from_typed(keys, values)?, Precision::Fp32)
            } else {
                let zeroed = || device.zeroed::<u16>(len).map_err(gpu_error);
                let (mut keys, mut values) = (zeroed()?, zeroed()?);
                self.keys_values(work, runs, block, group, CacheOutput::F16(&mut keys, &mut values))?;
                if built.is_some() && precision == Precision::Half {
                    let encode = |values: &DeviceBuffer<u16>| {
                        device.encode_fp8(values, (pairs, rows), (padded, padded))
                    };
                    fp8 = Some((encode(&keys)?, encode(&values)?));
                }
                (KeyValues::from_typed(keys, values)?, Precision::Split)
            };
            let cache = BlockCache {
                precision: read_at,
                keys: &stored.keys,
                values: &stored.values,
                key_scales: &ones,
                value_scales: &ones,
            };
            for run in runs.iter_mut() {
                self.stages
                    .icl_block(work, &mut run.hidden, block, cache, (group, run.len, rows))?;
            }
            let (keys_out, values_out) = &mut outputs.icl[block];
            let size = context_bytes(precision);
            for (out, cache) in [
                (&mut **keys_out, &stored.keys),
                (&mut **values_out, &stored.values),
            ] {
                let out = &mut out[first * rows * ICL_DIM * size..][..group * rows * ICL_DIM * size];
                unpad(&device.download(cache)?, out, rows, size)?;
            }
            if let Some(built) = &mut built {
                match fp8 {
                    Some(((keys, key_scales), (values, value_scales))) => built.push(
                        KeyValues { keys, values },
                        Some(Scales {
                            keys: key_scales,
                            values: value_scales,
                        }),
                    ),
                    None => built.push(stored, None),
                }
            }
        }
        Ok(built)
    }

    /// ICL block `block`'s keys and values for every training row of the
    /// runs, into `cache`.
    fn keys_values(
        &self,
        work: &mut Workspace,
        runs: &[Run<DeviceBuffer<f32>>],
        block: usize,
        group: usize,
        mut cache: CacheOutput<'_>,
    ) -> Result<()> {
        let padded = padded_rows(self.rows);
        for run in runs {
            self.stages
                .icl_keys_values(work, &run.hidden, block, group * run.len)?;
            self.device().icl_kv(
                work.wide.get(),
                cache.reborrow(),
                (group, run.len, run.start, padded),
            )?;
        }
        Ok(())
    }
}

/// A chunk of columns' buffers in the column stage.
struct ColumnChunk {
    /// Columns per chunk.
    columns: usize,
    /// The columns' training rows, `[columns, N, 128]`.
    state: DeviceBuffer<f32>,
    /// Their in-projections, `[columns, N, 384]`.
    projected: DeviceBuffer<f32>,
    /// The inducing points per column, `[columns, 128, 128]`.
    inducing: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    /// The inducing points' in-projections, `[columns, 128, 384]`.
    inducing_projected: DeviceBuffer<f32>,
    /// The inducing points' keys and values, `[columns, 8, 128, 16]`.
    keys: DeviceBuffer<f32>,
    values: DeviceBuffer<f32>,
}

impl ColumnChunk {
    fn new(device: &Device, columns: usize, rows: usize) -> Result<Self> {
        let zeroed = |len: usize| device.zeroed::<f32>(len).map_err(gpu_error);
        Ok(Self {
            columns,
            state: zeroed(columns * rows * COLUMN_DIM)?,
            projected: zeroed(columns * rows * 3 * COLUMN_DIM)?,
            inducing: zeroed(columns * INDUCING * COLUMN_DIM)?,
            attended: zeroed(columns * INDUCING * COLUMN_DIM)?,
            inducing_projected: zeroed(columns * INDUCING * 3 * COLUMN_DIM)?,
            keys: zeroed(columns * COLUMN_CACHE)?,
            values: zeroed(columns * COLUMN_CACHE)?,
        })
    }
}

/// Keys or values `[pairs, padded_rows(rows), 64]` of `size` bytes each (FP32
/// or FP16), as bytes, into `[pairs, rows, 64]`, checking that every value is
/// finite.
fn unpad(padded: &[u8], out: &mut [u8], rows: usize, size: usize) -> Result<()> {
    let finite = out
        .par_chunks_mut(rows * HEAD_DIM * size)
        .zip(padded.par_chunks(size * padded_rows(rows) * HEAD_DIM))
        .map(|(to, from)| {
            to.copy_from_slice(&from[..to.len()]);
            // Infinities and NaNs have every exponent bit set.
            if size == 4 {
                to.as_chunks::<4>()
                    .0
                    .iter()
                    .all(|&bytes| f32::from_le_bytes(bytes).is_finite())
            } else {
                to.as_chunks::<2>()
                    .0
                    .iter()
                    .all(|&bytes| u16::from_le_bytes(bytes) & 0x7c00 != 0x7c00)
            }
        })
        .reduce(|| true, |a, b| a && b);
    if finite {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "static context tensor has a value its storage cannot hold".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn shape(ensembles: usize, features: usize, rows: usize) -> Shape {
        Shape {
            ensembles,
            features,
            rows,
            outputs: 1,
        }
    }

    #[test]
    fn a_large_device_builds_every_estimator_at_once() {
        let plan = plan(14 << 30, Precision::Split, shape(8, 20, 10_000)).expect("fits");
        assert!(plan.keep);
        assert_eq!(plan.group, 8);
        assert_eq!(plan.columns, 8 * 20);
        assert_eq!(plan.run, MAX_RUN);
    }

    #[test]
    fn a_large_context_builds_in_groups_and_chunks() {
        let plan = plan(14 << 30, Precision::Split, shape(8, 100, 50_000)).expect("fits");
        assert!(!plan.keep);
        assert!(plan.group < 8 && u32::try_from(plan.group * 100 * 50_000 * COLUMN_DIM).is_ok());
        assert!(plan.columns < plan.group * 100);
        assert!(plan.run.is_multiple_of(MIN_RUN));
    }

    #[test]
    fn without_room_for_one_estimator_the_cpu_builds() {
        assert!(!fits(1 << 30, Precision::Split, shape(8, 100, 50_000)));
        assert!(!fits(14 << 30, Precision::Split, shape(8, 20, MIN_ROWS - 1)));
    }

    #[test]
    fn an_fp32_cache_takes_twice_the_room() {
        let (fp16, fp32) = (
            plan(14 << 30, Precision::Split, shape(8, 20, 30_000)).expect("fits"),
            plan(14 << 30, Precision::Fp32, shape(8, 20, 30_000)).expect("fits"),
        );
        assert!(fp16.keep && !fp32.keep);
    }
}
