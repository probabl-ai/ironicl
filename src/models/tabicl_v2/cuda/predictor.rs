//! A context's prediction on the device: the column stage against the cached
//! inducing points, the row transformer, the ICL blocks against the cached
//! keys and values, and the decoder, in passes over the queries.

use std::sync::{Arc, Mutex, PoisonError};

use cuda_core::DeviceBuffer;

use super::{
    COLUMN_BLOCKS, COLUMN_CACHE, COLUMN_DIM, HEAD_DIM, HEADS, ICL_BLOCKS, ICL_DIM, INDUCING, Precision,
    RESERVED, ROW_BLOCKS,
    cache::{BlockCache, BuiltCache, IclCache, IclSource, Slot, units},
    device::{Device, gpu_error, out_of_memory},
    launch::{Activation, Output, SmallAttention, padded_query_rows},
    plan::{MIN_RUN, plan},
    weights::ModelWeights,
    workspace::{Buffer, Buffers, Halves, Run, Shape, Workspace},
};
use crate::{
    error::{Error, Result},
    progress::Stage,
};

/// Whether `context` matches its shape and the model, since the kernels index it unchecked: each
/// column block's keys and values `[B, H + 4, 8, 128, 16]`, a base scale per ICL block, the CLS
/// tokens, and no more outputs than the decoder has rows.
fn check(weights: &ModelWeights, context: &ContextTensors<'_>) -> Result<()> {
    let Shape {
        ensembles: b,
        features: h,
        outputs,
        ..
    } = context.shape;
    let column = b
        .checked_mul(h + RESERVED)
        .and_then(|len| len.checked_mul(COLUMN_CACHE));
    let consistent = column.is_some_and(|len| {
        context.column.len() == COLUMN_BLOCKS
            && context
                .column
                .iter()
                .all(|(keys, values)| keys.len() == len && values.len() == len)
    }) && context.base_scales.len() == ICL_BLOCKS
        && context.base_scales.iter().all(|scale| scale.len() == ICL_DIM)
        && context.cls_tokens.len() >= 4 * COLUMN_DIM
        && (1..=weights.decoder.out.b.len()).contains(&outputs);
    if consistent {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "CUDA context tensors do not match their shape or the model".into(),
        ))
    }
}

/// A context's tensors the predictor takes.
pub struct ContextTensors<'a> {
    /// Estimators, features, training rows, and decoder outputs per row.
    pub shape: Shape,
    /// Each column block's keys and values, `[B, H + 4, 8, 128, 16]`.
    pub column: Vec<(&'a [f32], &'a [f32])>,
    /// The ICL blocks' keys and values.
    pub icl: Arc<dyn IclSource>,
    /// Each ICL block's `QASSMax` base scale, `[8, 64]`.
    pub base_scales: &'a [Vec<f32>],
    /// The row transformer's `RoPE` frequencies.
    pub rope_frequencies: &'a [f32],
    /// The row transformer's CLS tokens, at least `[4, 128]`.
    pub cls_tokens: &'a [f32],
    /// The device memory this context may use, in bytes.
    pub budget: usize,
}

/// Decoder outputs for a prediction's queries, and the queries only the CPU
/// path can predict.
pub struct Prediction {
    /// `[B, Q, outputs]`; rows in `cpu_rows` hold no result.
    pub logits: Vec<f32>,
    /// Query rows with a grouped feature triple equal to the all-missing
    /// marker, ascending.
    pub cpu_rows: Vec<usize>,
}

/// What a prediction reuses and adjusts.
struct State {
    buffers: Buffers,
    slots: Vec<Slot>,
    /// Query rows per run; halved when the device runs out of memory.
    run_rows: usize,
}

/// A context's caches on the device and the buffers its predictions reuse.
pub struct Predictor {
    weights: Arc<ModelWeights>,
    precision: Precision,
    shape: Shape,
    /// Each column block's cached keys and values.
    columns: Vec<(DeviceBuffer<f32>, DeviceBuffer<f32>)>,
    icl: IclCache,
    /// Each ICL block's `QASSMax` base scale.
    base_scales: Vec<DeviceBuffer<f32>>,
    tokens: RowTokens,
    pass_rows: usize,
    state: Mutex<State>,
}

impl std::fmt::Debug for Predictor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Predictor")
            .field("shape", &self.shape)
            .field("precision", &self.precision)
            .field("streamed", &self.streamed())
            .field("pass_rows", &self.pass_rows)
            .finish_non_exhaustive()
    }
}

/// The row stage's CLS tokens and its `RoPE` cos and sin tables for every
/// position of a `[4 + H]`-token sequence.
pub(super) struct RowTokens {
    cls: DeviceBuffer<f32>,
    cos: DeviceBuffer<f32>,
    sin: DeviceBuffer<f32>,
}

impl RowTokens {
    pub(super) fn upload(device: &Device, features: usize, frequencies: &[f32], cls: &[f32]) -> Result<Self> {
        let (cos, sin) = crate::models::tabicl_v2::cpu::rope_tables(features + RESERVED, frequencies);
        Ok(Self {
            cls: device.upload(&cls[..RESERVED * COLUMN_DIM])?,
            cos: device.upload(&cos)?,
            sin: device.upload(&sin)?,
        })
    }

    pub(super) const fn rope(&self) -> (&DeviceBuffer<f32>, &DeviceBuffer<f32>) {
        (&self.cos, &self.sin)
    }
}

/// Each ICL block's `QASSMax` base scale on the device.
pub(super) fn upload_all(device: &Device, values: &[Vec<f32>]) -> Result<Vec<DeviceBuffer<f32>>> {
    values.iter().map(|values| device.upload(values)).collect()
}

impl Predictor {
    /// Upload a context's caches, its ICL keys and values too when they fit
    /// the device with room to predict; otherwise they stream from the host on
    /// every pass. `built`, the ICL keys and values a build left on the
    /// device, replaces their upload when they can stay there. `None` when not
    /// even the streamed cache fits.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent tensors or a failed upload.
    pub fn new(
        weights: Arc<ModelWeights>,
        precision: Precision,
        context: &ContextTensors<'_>,
        built: Option<BuiltCache>,
    ) -> Result<Option<Self>> {
        check(&weights, context)?;
        let device = &weights.device;
        let shape = context.shape;
        let (b, h, n) = (shape.ensembles, shape.features, shape.rows);
        // Built keys and values already take their share of the budget.
        // Encoding FP8 on the device holds one FP16 tensor of a block there,
        // as many bytes as the block's FP8 keys and values.
        let block_bytes = 4 * units(precision, b * HEADS, n);
        let budget = match &built {
            Some(built) => context.budget + built.bytes(),
            None if precision == Precision::Half => context.budget.saturating_sub(block_bytes),
            None => context.budget,
        };
        let Some(plan) = plan(budget, (b, h), block_bytes) else {
            return Ok(None);
        };
        let icl = match built.filter(|built| !plan.streamed && built.precision() == precision) {
            Some(built) => IclCache::built(device, built, (b * HEADS, n))?,
            None => IclCache::new(device, precision, &context.icl, (b * HEADS, n), plan.streamed)?,
        };
        let columns = context
            .column
            .iter()
            .map(|(keys, values)| Ok((device.upload(keys)?, device.upload(values)?)))
            .collect::<Result<_>>()?;
        let slots = icl.slots(device)?;
        Ok(Some(Self {
            precision,
            shape,
            columns,
            icl,
            base_scales: upload_all(device, context.base_scales)?,
            tokens: RowTokens::upload(device, h, context.rope_frequencies, context.cls_tokens)?,
            pass_rows: plan.pass_rows,
            state: Mutex::new(State {
                buffers: Buffers::default(),
                slots,
                run_rows: plan.run_rows,
            }),
            weights,
        }))
    }

    /// Whether the ICL cache streams from the host on every pass.
    #[must_use]
    pub const fn streamed(&self) -> bool {
        self.icl.streamed()
    }

    /// Query rows a prediction should bring at once to use whole passes.
    #[must_use]
    pub const fn pass_rows(&self) -> usize {
        self.pass_rows
    }

    /// Decoder outputs `[B, Q, outputs]` for queries `x[B, Q, H]`. Reports
    /// progress after each pass, which can cancel the prediction.
    ///
    /// # Errors
    ///
    /// Returns an error if a launch or copy fails, or the progress hook's.
    pub fn predict(&self, x: &[f32], q: usize) -> Result<Prediction> {
        let (b, h) = (self.shape.ensembles, self.shape.features);
        if x.len() != b * q * h {
            return Err(Error::InvalidArgument(format!(
                "CUDA queries hold {} values, not [B, Q, H] = [{b}, {q}, {h}]",
                x.len()
            )));
        }
        let device = &self.weights.device;
        let mut logits = vec![0.0; b * q * self.shape.outputs];
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let queries = device.upload(x)?;
        let mut missing = device.zeroed::<f32>(q.max(1)).map_err(gpu_error)?;
        let mut start = 0;
        while start < q {
            let run = state.run_rows;
            let pass = if self.streamed() {
                (self.pass_rows / run).max(1) * run
            } else {
                run
            };
            let rows = pass.min(q - start);
            // Out of device memory before the pass launches anything: free
            // the buffers and retry with runs half as long.
            match state
                .buffers
                .reserve(device, self.precision, self.shape, (run, rows))
            {
                Err(error) if out_of_memory(error) && run > MIN_RUN => {
                    state.buffers = Buffers::default();
                    state.run_rows = run / 2;
                    continue;
                }
                reserved => reserved.map_err(gpu_error)?,
            }
            self.run_pass(
                &mut state,
                (&queries, &mut missing, q),
                (start, rows),
                &mut logits,
            )?;
            start += rows;
            crate::progress::report(Stage::Predict, start, q)?;
        }
        let flags = device.download(&missing)?;
        let cpu_rows = (0..q).filter(|&row| flags[row] != 0.0).collect();
        Ok(Prediction { logits, cpu_rows })
    }

    /// One pass over query rows `[start, start + rows)` of the device queries
    /// `[B, total, H]`: the column and row stages run by run into each run's
    /// ICL state, then each ICL block runs over every run while the next
    /// streamed block copies, then the decoder, into `logits` `[B, total,
    /// outputs]`.
    fn run_pass(
        &self,
        state: &mut State,
        (queries, missing, total): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>, usize),
        (start, rows): (usize, usize),
        logits: &mut [f32],
    ) -> Result<()> {
        let device = &self.weights.device;
        let Shape {
            ensembles: b,
            features: h,
            rows: n,
            outputs,
        } = self.shape;
        let stages = self.stages();
        let State {
            buffers: Buffers { work, hidden },
            slots,
            run_rows,
        } = state;
        let mut runs: Vec<_> = (start..start + rows)
            .step_by(*run_rows)
            .zip(hidden.iter_mut())
            .map(|(row, hidden)| Run {
                start: row,
                len: (*run_rows).min(start + rows - row),
                hidden: hidden.get_mut(),
            })
            .collect();
        for run in &mut runs {
            self.column_stage(work, (queries, &mut *missing, total, run.start), run.len)?;
            stages.row_stage(work, None, run.hidden, (b, h, run.len), (run.len, 0))?;
        }
        self.icl.begin(device, slots)?;
        for block in 0..ICL_BLOCKS {
            let cache = self.icl.acquire(device, slots, block)?;
            for run in &mut runs {
                stages.icl_block(work, run.hidden, block, cache, (b, run.len, n))?;
            }
            self.icl.release(device, slots, block)?;
        }
        for run in &runs {
            stages.decoder(work, run.hidden, (b * run.len, outputs))?;
            let host = device.download(work.logits.get())?;
            let width = run.len * outputs;
            for (ensemble, part) in host[..b * width].chunks_exact(width).enumerate() {
                logits[(ensemble * total + run.start) * outputs..][..width].copy_from_slice(part);
            }
        }
        Ok(())
    }

    /// Query rows `[start, start + q)` of the queries `[B, total, H]` to
    /// `[B * H, q, 128]` column embeddings in `x0`.
    fn column_stage(
        &self,
        work: &mut Workspace,
        (queries, missing, total, start): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>, usize, usize),
        q: usize,
    ) -> Result<()> {
        let weights = &*self.weights;
        let device = &weights.device;
        let (b, h) = (self.shape.ensembles, self.shape.features);
        let columns = b * h * q;
        let (x0, x1, attended) = (work.x0.get_mut(), work.x1.get_mut(), work.attended.get_mut());
        device.column_input(
            queries,
            &weights.column_input,
            (x0, missing),
            (b, h, total, start, q),
        )?;
        for (block, (keys, values)) in weights.columns.iter().zip(&self.columns) {
            let attention = &block.attention;
            // From one 32-row tile per column on, a whole block is one kernel.
            if q >= 32 {
                device.column_block(self.precision, x0, attention, (keys, values), (b, h, q, RESERVED))?;
                continue;
            }
            device.token_head(
                self.precision,
                (x0, x1),
                attention,
                self.tokens.rope(),
                (columns, COLUMN_DIM, 0),
            )?;
            // Feature h of ensemble b attends to cache stream h + 4.
            let layout = SmallAttention {
                seqs: b * h,
                lq: q,
                lk: INDUCING,
                inner: h,
                q_seq: q * COLUMN_DIM,
                q_tok: COLUMN_DIM,
                k_base: RESERVED * COLUMN_CACHE,
                v_base: RESERVED * COLUMN_CACHE,
                kv_outer: (h + RESERVED) * COLUMN_CACHE,
                kv_inner: COLUMN_CACHE,
                kv_tok: 16,
                kv_head: INDUCING * 16,
                o_seq: q * COLUMN_DIM,
                o_tok: COLUMN_DIM,
            };
            device.attention16(self.precision, (x1, keys, values), attended, &layout)?;
            device.token_tail(self.precision, (attended, x0), attention, columns)?;
        }
        debug_assert_eq!(weights.columns.len(), COLUMN_BLOCKS);
        Ok(())
    }

    /// The stages this predictor shares with a context's build.
    fn stages(&self) -> Stages<'_> {
        Stages {
            weights: &self.weights,
            precision: self.precision,
            tokens: &self.tokens,
            base_scales: &self.base_scales,
        }
    }
}

/// What a prediction's and a build's stages share: the weights, the
/// precision of the products, the row stage's tokens, and each ICL block's
/// `QASSMax` base scale.
pub(super) struct Stages<'a> {
    pub(super) weights: &'a ModelWeights,
    pub(super) precision: Precision,
    pub(super) tokens: &'a RowTokens,
    pub(super) base_scales: &'a [DeviceBuffer<f32>],
}

impl Stages<'_> {
    /// Rows `start..start + q` of the column embeddings `[B * H, total, 128]`
    /// in `columns`, or else in the workspace's `x0`, to row representations
    /// `[B * q, 512]` in `out`.
    pub(super) fn row_stage(
        &self,
        work: &mut Workspace,
        columns: Option<&DeviceBuffer<f32>>,
        out: &mut DeviceBuffer<f32>,
        (b, h, q): (usize, usize, usize),
        (total, start): (usize, usize),
    ) -> Result<()> {
        let weights = self.weights;
        let device = &weights.device;
        let len = h + RESERVED;
        let seqs = b * q;
        let tokens = seqs * len;
        let qkv = 3 * COLUMN_DIM;
        let (x0, x1, wide, attended, cls) = (
            columns.unwrap_or_else(|| work.x0.get()),
            work.x1.get_mut(),
            work.wide.get_mut(),
            work.attended.get_mut(),
            work.cls.get_mut(),
        );
        device.build_rows(x0, &self.tokens.cls, x1, (b, h, q), (total, start))?;
        for (index, block) in weights.rows.iter().enumerate() {
            let last = index + 1 == ROW_BLOCKS;
            device.token_head(
                self.precision,
                (x1, wide),
                block,
                self.tokens.rope(),
                (tokens, qkv, len),
            )?;
            // The last block updates only the CLS tokens.
            let query_rows = if last { RESERVED } else { len };
            let layout = SmallAttention {
                seqs,
                lq: query_rows,
                lk: len,
                inner: 1,
                q_seq: len * qkv,
                q_tok: qkv,
                k_base: COLUMN_DIM,
                v_base: 2 * COLUMN_DIM,
                kv_outer: len * qkv,
                kv_inner: 0,
                kv_tok: qkv,
                kv_head: 16,
                o_seq: query_rows * COLUMN_DIM,
                o_tok: COLUMN_DIM,
            };
            device.attention16(self.precision, (wide, wide, wide), attended, &layout)?;
            if last {
                device.gather_cls(x1, cls, seqs, len)?;
                device.token_tail(self.precision, (attended, cls), block, seqs * RESERVED)?;
            } else {
                device.token_tail(self.precision, (attended, x1), block, tokens)?;
            }
        }
        // [B * q * 4, 128] normalized and read as [B * q, 512].
        device.layer_norm(
            cls,
            &weights.row_norm,
            Output::F32(out),
            seqs * RESERVED,
            COLUMN_DIM,
        )
    }

    /// ICL block `block` on row representations `hidden` `[B * q, 512]`, in
    /// place, attending to `rows` training rows' keys and values `cache`.
    pub(super) fn icl_block(
        &self,
        work: &mut Workspace,
        hidden: &mut DeviceBuffer<f32>,
        block: usize,
        cache: BlockCache<'_>,
        (b, q, rows): (usize, usize, usize),
    ) -> Result<()> {
        let weights = self.weights;
        let device = &weights.device;
        let layer = &weights.icl[block];
        let seqs = b * q;
        let pairs = b * HEADS;
        let padded = padded_query_rows(q);
        let tensor = self.tensor(seqs);
        let normed = self.output(tensor, &mut work.a, &mut work.normed);
        device.layer_norm(hidden, &layer.norm1, normed, seqs, ICL_DIM)?;
        // The queries in FP32 for QASSMax in x1, and the gate's operand.
        device.product(
            self.input(tensor, &work.a, &work.normed),
            &layer.query,
            None,
            (Some(work.x1.get_mut()), tensor.then(|| self.halves(&mut work.w))),
            work.partial.get_mut(),
            (seqs, ICL_DIM, ICL_DIM),
            false,
        )?;
        // QASSMax: an MLP over each 64-wide head of the query gates it.
        let gate_shape = (seqs * HEADS, HEAD_DIM, HEAD_DIM);
        device.product(
            self.input(tensor, &work.w, &work.x1),
            &layer.gate[0],
            None,
            outputs(self.output(tensor, &mut work.a, &mut work.gate_hidden)),
            work.partial.get_mut(),
            gate_shape,
            true,
        )?;
        device.product(
            self.input(tensor, &work.a, &work.gate_hidden),
            &layer.gate[1],
            None,
            (Some(work.gate.get_mut()), None),
            work.partial.get_mut(),
            gate_shape,
            false,
        )?;
        let fp32 = cache.precision == Precision::Fp32;
        device.qass_pack(
            (work.x1.get(), work.gate.get()),
            (&self.base_scales[block], cache.key_scales),
            if fp32 {
                Output::F32(work.packed32.get_mut())
            } else {
                Output::Half(work.packed.get_mut())
            },
            (b, q, padded),
        )?;
        device.attend_icl(
            cache.precision,
            if fp32 {
                Activation::F32(work.packed32.get())
            } else {
                Activation::Half(work.packed.get())
            },
            (cache.keys, cache.values),
            (work.part.get_mut(), work.lse.get_mut(), work.attention.get_mut()),
            (pairs, padded, rows),
        )?;
        let attended = self.output(tensor, &mut work.a, &mut work.attended);
        device.heads_to_tokens(
            (work.attention.get(), cache.value_scales),
            attended,
            (b, q, padded),
        )?;
        self.icl_feed_forward(work, hidden, block, seqs)
    }

    /// The rest of ICL block `block` once the workspace holds its attention
    /// output: `mid = hidden + out_proj(attended)`, then
    /// `hidden = mid + linear2(gelu(linear1(norm2(mid))))`. The queries in x1
    /// are spent, so x1 holds `mid`.
    fn icl_feed_forward(
        &self,
        work: &mut Workspace,
        hidden: &mut DeviceBuffer<f32>,
        block: usize,
        seqs: usize,
    ) -> Result<()> {
        let device = &self.weights.device;
        let layer = &self.weights.icl[block];
        let tensor = self.tensor(seqs);
        device.product(
            self.input(tensor, &work.a, &work.attended),
            &layer.out_proj,
            Some(hidden),
            (Some(work.x1.get_mut()), None),
            work.partial.get_mut(),
            (seqs, ICL_DIM, ICL_DIM),
            false,
        )?;
        let normed = self.output(tensor, &mut work.a, &mut work.normed);
        device.layer_norm(work.x1.get(), &layer.norm2, normed, seqs, ICL_DIM)?;
        device.product(
            self.input(tensor, &work.a, &work.normed),
            &layer.linear1,
            None,
            outputs(self.output(tensor, &mut work.w, &mut work.wide)),
            work.partial.get_mut(),
            (seqs, 2 * ICL_DIM, ICL_DIM),
            true,
        )?;
        device.product(
            self.input(tensor, &work.w, &work.wide),
            &layer.linear2,
            Some(work.x1.get()),
            (Some(hidden), None),
            work.partial.get_mut(),
            (seqs, ICL_DIM, 2 * ICL_DIM),
            false,
        )
    }

    /// ICL block `block`'s keys and values `[seqs, 1024]` in FP32 into the
    /// workspace's `wide`, from row representations `hidden` `[seqs, 512]`.
    pub(super) fn icl_keys_values(
        &self,
        work: &mut Workspace,
        hidden: &DeviceBuffer<f32>,
        block: usize,
        seqs: usize,
    ) -> Result<()> {
        let device = &self.weights.device;
        let layer = &self.weights.icl[block];
        let tensor = self.tensor(seqs);
        let normed = self.output(tensor, &mut work.a, &mut work.normed);
        device.layer_norm(hidden, &layer.norm1, normed, seqs, ICL_DIM)?;
        device.product(
            self.input(tensor, &work.a, &work.normed),
            &layer.keys_values,
            None,
            (Some(work.wide.get_mut()), None),
            work.partial.get_mut(),
            (seqs, 2 * ICL_DIM, ICL_DIM),
            false,
        )
    }

    /// The decoder on row representations `hidden` `[seqs, 512]`, into the
    /// workspace's logits `[seqs, width]`.
    pub(super) fn decoder(
        &self,
        work: &mut Workspace,
        hidden: &DeviceBuffer<f32>,
        (seqs, width): (usize, usize),
    ) -> Result<()> {
        let device = &self.weights.device;
        let tensor = self.tensor(seqs);
        let decoder = &self.weights.decoder;
        let normed = self.output(tensor, &mut work.a, &mut work.normed);
        device.layer_norm(hidden, &decoder.norm, normed, seqs, ICL_DIM)?;
        device.product(
            self.input(tensor, &work.a, &work.normed),
            &decoder.hidden,
            None,
            outputs(self.output(tensor, &mut work.w, &mut work.wide)),
            work.partial.get_mut(),
            (seqs, 2 * ICL_DIM, ICL_DIM),
            true,
        )?;
        device.product(
            self.input(tensor, &work.w, &work.wide),
            &decoder.out,
            None,
            (Some(work.logits.get_mut()), None),
            work.partial.get_mut(),
            (seqs, width, 2 * ICL_DIM),
            false,
        )
    }

    /// Whether the ICL and decoder products over `rows` token rows run on
    /// tensor cores, above [`Precision::tensor_rows`], rather than in FP32.
    const fn tensor(&self, rows: usize) -> bool {
        rows > self.precision.tensor_rows()
    }

    /// A product's operand as it reads it: on tensor cores, the FP16 values
    /// in `halves`, with their low parts at split precision; else FP32 in `f32`.
    fn input<'a>(&self, tensor: bool, halves: &'a Halves, f32: &'a Buffer<f32>) -> Activation<'a> {
        match self.precision {
            _ if !tensor => Activation::F32(f32.get()),
            Precision::Fp32 => unreachable!("FP32 runs no tensor-core products"),
            Precision::Half => Activation::Half(halves.hi.get()),
            Precision::Split => Activation::Split {
                hi: halves.hi.get(),
                lo: halves.lo.get(),
            },
        }
    }

    /// Where a kernel writes the operand [`Self::input`] reads.
    fn output<'a>(&self, tensor: bool, halves: &'a mut Halves, f32: &'a mut Buffer<f32>) -> Output<'a> {
        if tensor {
            self.halves(halves)
        } else {
            Output::F32(f32.get_mut())
        }
    }

    /// `halves` as a tensor-core operand at this precision.
    fn halves<'a>(&self, halves: &'a mut Halves) -> Output<'a> {
        match self.precision {
            Precision::Fp32 => unreachable!("FP32 runs no tensor-core products"),
            Precision::Half => Output::Half(halves.hi.get_mut()),
            Precision::Split => Output::Split {
                hi: halves.hi.get_mut(),
                lo: halves.lo.get_mut(),
            },
        }
    }
}

/// Where a product writes `out`: FP32 into its buffer, or as the next
/// product's operand.
fn outputs(out: Output<'_>) -> (Option<&mut DeviceBuffer<f32>>, Option<Output<'_>>) {
    match out {
        Output::F32(y) => (Some(y), None),
        half => (None, Some(half)),
    }
}
