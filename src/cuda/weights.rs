//! The model's weights on the device, resolved once into typed blocks, so a
//! prediction looks nothing up by name.

use cuda_core::{DeviceBuffer, DeviceCopy};

use super::{COLUMN_BLOCKS, COLUMN_DIM, HEAD_DIM, ICL_BLOCKS, ICL_DIM, LO_SCALE, ROW_BLOCKS, device::Device};
use crate::error::Result;

/// A dense layer `[n, k]`: FP32 for the CUDA cores, FP16 high and scaled low
/// parts for the tensor cores, and its bias.
pub(super) struct Dense {
    pub(super) w: DeviceBuffer<f32>,
    pub(super) hi: DeviceBuffer<u16>,
    pub(super) lo: DeviceBuffer<u16>,
    pub(super) b: DeviceBuffer<f32>,
}

impl Dense {
    /// A dense layer from its FP32 weight `[n, k]` and bias.
    pub(super) fn upload(device: &Device, w: &[f32], b: &[f32]) -> Result<Self> {
        let (hi, lo): (Vec<u16>, Vec<u16>) = w
            .iter()
            .map(|&v| {
                (
                    half::f16::from_f32(v).to_bits(),
                    half::f16::from_f32(low_part(v)).to_bits(),
                )
            })
            .unzip();
        Ok(Self {
            w: device.upload(w)?,
            hi: device.upload(&hi)?,
            lo: device.upload(&lo)?,
            b: device.upload(b)?,
        })
    }

    /// Device memory the layer takes.
    fn bytes(&self) -> usize {
        4 * (self.w.len() + self.b.len()) + 2 * (self.hi.len() + self.lo.len())
    }
}

/// A layer norm's scale and shift.
pub(super) struct Norm {
    pub(super) gamma: DeviceBuffer<f32>,
    pub(super) beta: DeviceBuffer<f32>,
}

/// A 128-wide block's dense layer in the B-fragment order of
/// `mma.m16n8k16` (see [`fragment_order`]), high and low parts, and its bias.
pub(super) struct Fragments {
    pub(super) hi: DeviceBuffer<u32>,
    pub(super) lo: DeviceBuffer<u32>,
    pub(super) b: DeviceBuffer<f32>,
}

/// A 128-wide transformer block of the column or row stage, run by the fused
/// block kernels.
pub(super) struct TokenBlock {
    pub(super) norm1: Norm,
    /// Queries, keys, and values, `[384, 128]`.
    pub(super) in_proj: Fragments,
    pub(super) out_proj: Fragments,
    pub(super) norm2: Norm,
    pub(super) linear1: Fragments,
    pub(super) linear2: Fragments,
}

/// An ICL transformer block: the fused in-projection's query rows, which
/// predictions use, and its key and value rows, which builds use.
pub(super) struct IclBlock {
    pub(super) norm1: Norm,
    pub(super) query: Dense,
    /// Keys, then values: `[1024, 512]`.
    pub(super) keys_values: Dense,
    /// `QASSMax`'s per-head gating MLP, `64 -> 64 -> 64`.
    pub(super) gate: [Dense; 2],
    pub(super) out_proj: Dense,
    pub(super) norm2: Norm,
    pub(super) linear1: Dense,
    pub(super) linear2: Dense,
}

/// The column stage's input layer, `[128, 3]`, and its bias.
pub(super) struct ColumnInput {
    pub(super) weight: DeviceBuffer<f32>,
    pub(super) bias: DeviceBuffer<f32>,
}

/// A column block: an induced set attention block.
pub(super) struct ColumnBlock {
    /// The inducing points attending to the training rows: builds only.
    pub(super) inducing_attention: TokenBlock,
    /// The inducing points, `[128, 128]`.
    pub(super) inducing_points: DeviceBuffer<f32>,
    /// The training and query rows attending to the inducing points.
    pub(super) attention: TokenBlock,
}

/// The decoder after the ICL blocks.
pub(super) struct Decoder {
    pub(super) norm: Norm,
    pub(super) hidden: Dense,
    pub(super) out: Dense,
}

/// A model's weights on its device, shared by every context of the model.
pub struct ModelWeights {
    pub(super) device: Device,
    pub(super) column_input: ColumnInput,
    pub(super) columns: Vec<ColumnBlock>,
    pub(super) rows: Vec<TokenBlock>,
    /// The row stage's output norm.
    pub(super) row_norm: Norm,
    pub(super) icl: Vec<IclBlock>,
    pub(super) decoder: Decoder,
    /// Device memory the weights take.
    pub(super) bytes: usize,
}

impl std::fmt::Debug for ModelWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelWeights")
            .field("device", &self.device)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

/// A value's low part, `(v - f16(v)) * LO_SCALE`.
pub(super) fn low_part(value: f32) -> f32 {
    (value - half::f16::from_f32(value).to_f32()) * LO_SCALE
}

/// Two values as packed FP16, the first in the low half.
fn pack_f16(a: f32, b: f32) -> u32 {
    u32::from(half::f16::from_f32(a).to_bits()) | (u32::from(half::f16::from_f32(b).to_bits()) << 16)
}

/// A weight `[n, k]` in the B-fragment order of `mma.m16n8k16`:
/// `[n / 8, k / 16, 32 lanes, 2]` packed FP16 pairs, high and low parts.
fn fragment_order(values: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<u32>) {
    let mut high = Vec::with_capacity(n * k / 2);
    let mut low = Vec::with_capacity(n * k / 2);
    for tile in 0..n / 8 {
        for step in 0..k / 16 {
            for lane in 0..32 {
                for half in 0..2 {
                    let at = (tile * 8 + lane / 4) * k + step * 16 + (lane % 4) * 2 + half * 8;
                    let (a, b) = (values[at], values[at + 1]);
                    high.push(pack_f16(a, b));
                    low.push(pack_f16(low_part(a), low_part(b)));
                }
            }
        }
    }
    (high, low)
}

/// Uploads weights by name from the checkpoint and adds up their size.
struct Loader<'d, F> {
    device: &'d Device,
    tensor: F,
    bytes: usize,
}

impl<'a, F: Fn(&str) -> Result<&'a [f32]>> Loader<'_, F> {
    fn upload<T: DeviceCopy>(&mut self, values: &[T]) -> Result<DeviceBuffer<T>> {
        self.bytes += std::mem::size_of_val(values);
        self.device.upload(values)
    }

    fn tensor(&mut self, name: &str) -> Result<DeviceBuffer<f32>> {
        let values = (self.tensor)(name)?;
        self.upload(values)
    }

    fn norm(&mut self, prefix: &str) -> Result<Norm> {
        Ok(Norm {
            gamma: self.tensor(&format!("{prefix}.weight"))?,
            beta: self.tensor(&format!("{prefix}.bias"))?,
        })
    }

    /// Output rows `rows` of a dense layer `[_, k]` and its bias.
    fn dense_rows(
        &mut self,
        weight: &str,
        bias: &str,
        rows: std::ops::Range<usize>,
        k: usize,
    ) -> Result<Dense> {
        let dense = Dense::upload(
            self.device,
            &(self.tensor)(weight)?[rows.start * k..rows.end * k],
            &(self.tensor)(bias)?[rows],
        )?;
        self.bytes += dense.bytes();
        Ok(dense)
    }

    /// A whole dense layer `[_, k]` and its bias.
    fn dense(&mut self, prefix: &str, k: usize) -> Result<Dense> {
        let weight = format!("{prefix}.weight");
        let rows = (self.tensor)(&weight)?.len() / k;
        self.dense_rows(&weight, &format!("{prefix}.bias"), 0..rows, k)
    }

    fn fragments(&mut self, weight: &str, bias: &str, n: usize, k: usize) -> Result<Fragments> {
        let (hi, lo) = fragment_order((self.tensor)(weight)?, n, k);
        Ok(Fragments {
            hi: self.upload(&hi)?,
            lo: self.upload(&lo)?,
            b: self.tensor(bias)?,
        })
    }

    fn token_block(&mut self, prefix: &str) -> Result<TokenBlock> {
        let d = COLUMN_DIM;
        Ok(TokenBlock {
            norm1: self.norm(&format!("{prefix}.norm1"))?,
            in_proj: self.fragments(
                &format!("{prefix}.attn.in_proj_weight"),
                &format!("{prefix}.attn.in_proj_bias"),
                3 * d,
                d,
            )?,
            out_proj: self.fragments(
                &format!("{prefix}.attn.out_proj.weight"),
                &format!("{prefix}.attn.out_proj.bias"),
                d,
                d,
            )?,
            norm2: self.norm(&format!("{prefix}.norm2"))?,
            linear1: self.fragments(
                &format!("{prefix}.linear1.weight"),
                &format!("{prefix}.linear1.bias"),
                2 * d,
                d,
            )?,
            linear2: self.fragments(
                &format!("{prefix}.linear2.weight"),
                &format!("{prefix}.linear2.bias"),
                d,
                2 * d,
            )?,
        })
    }

    fn icl_block(&mut self, prefix: &str) -> Result<IclBlock> {
        let (d, h) = (ICL_DIM, HEAD_DIM);
        let gate = format!("{prefix}.attn.ssmax_layer.query_mlp");
        let (weight, bias) = (
            format!("{prefix}.attn.in_proj_weight"),
            format!("{prefix}.attn.in_proj_bias"),
        );
        Ok(IclBlock {
            norm1: self.norm(&format!("{prefix}.norm1"))?,
            query: self.dense_rows(&weight, &bias, 0..d, d)?,
            keys_values: self.dense_rows(&weight, &bias, d..3 * d, d)?,
            gate: [
                self.dense(&format!("{gate}.0"), h)?,
                self.dense(&format!("{gate}.2"), h)?,
            ],
            out_proj: self.dense(&format!("{prefix}.attn.out_proj"), d)?,
            norm2: self.norm(&format!("{prefix}.norm2"))?,
            linear1: self.dense(&format!("{prefix}.linear1"), d)?,
            linear2: self.dense(&format!("{prefix}.linear2"), 2 * d)?,
        })
    }
}

impl ModelWeights {
    /// The device memory a context may use, in bytes: what is free now,
    /// within `device_limit`, the user's cap on Ironicl's device memory, less
    /// these weights.
    ///
    /// # Errors
    ///
    /// Returns an error if the driver cannot report the free memory.
    pub fn budget(&self, device_limit: Option<u64>) -> Result<usize> {
        let free = self.device.free_memory()?;
        Ok(device_limit.map_or(free, |limit| {
            let limit = usize::try_from(limit).unwrap_or(usize::MAX);
            free.min(limit.saturating_sub(self.bytes))
        }))
    }

    /// Upload every weight predictions and builds read, looked up by `tensor`.
    ///
    /// # Errors
    ///
    /// Returns an error if a weight is missing or an upload fails.
    pub fn new<'a>(device: Device, tensor: impl Fn(&str) -> Result<&'a [f32]>) -> Result<Self> {
        let mut load = Loader {
            device: &device,
            tensor,
            bytes: 0,
        };
        let column_input = ColumnInput {
            weight: load.tensor("col_embedder.in_linear.weight")?,
            bias: load.tensor("col_embedder.in_linear.bias")?,
        };
        let columns = (0..COLUMN_BLOCKS)
            .map(|block| {
                let prefix = format!("col_embedder.tf_col.blocks.{block}");
                Ok(ColumnBlock {
                    inducing_attention: load.token_block(&format!("{prefix}.multihead_attn1"))?,
                    inducing_points: load.tensor(&format!("{prefix}.ind_vectors"))?,
                    attention: load.token_block(&format!("{prefix}.multihead_attn2"))?,
                })
            })
            .collect::<Result<_>>()?;
        let rows = (0..ROW_BLOCKS)
            .map(|block| load.token_block(&format!("row_interactor.tf_row.blocks.{block}")))
            .collect::<Result<_>>()?;
        let row_norm = load.norm("row_interactor.out_ln")?;
        let icl = (0..ICL_BLOCKS)
            .map(|block| load.icl_block(&format!("icl_predictor.tf_icl.blocks.{block}")))
            .collect::<Result<_>>()?;
        let decoder = Decoder {
            norm: load.norm("icl_predictor.ln")?,
            hidden: load.dense("icl_predictor.decoder.0", ICL_DIM)?,
            out: load.dense("icl_predictor.decoder.2", 2 * ICL_DIM)?,
        };
        let bytes = load.bytes;
        Ok(Self {
            device,
            column_input,
            columns,
            rows,
            row_norm,
            icl,
            decoder,
            bytes,
        })
    }
}
