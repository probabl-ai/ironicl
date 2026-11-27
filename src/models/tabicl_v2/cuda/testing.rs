//! Entry points for the kernel tests: one kernel family run on host data the
//! way a predictor runs it, compared against references in `tests/`.

use super::{
    HEAD_DIM, Precision,
    cache::UploadedBlock,
    device::{Device, gpu_error},
    launch::{Activation, Output, padded_query_rows},
    weights::Dense,
};
use crate::error::{Error, Result};

impl Device {
    /// Softmax attention of `queries` against `keys` and `values`, all FP16
    /// bits, for `pairs` (estimator, head) pairs, with the cache stored and the
    /// products run as a predictor does at `precision`.
    ///
    /// - `queries`: `[pairs, query_rows, 64]`.
    /// - `keys`, `values`: `[pairs, rows, 64]`.
    ///
    /// Returns `softmax(q kᵀ / 8) v` as FP32, `[pairs, query_rows, 64]`.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent shapes or a failed launch.
    #[doc(hidden)]
    pub fn icl_attention(
        &self,
        precision: Precision,
        queries: &[u16],
        (keys, values): (&[u16], &[u16]),
        (pairs, query_rows, rows): (usize, usize, usize),
    ) -> Result<Vec<f32>> {
        if rows == 0
            || query_rows == 0
            || queries.len() != pairs * query_rows * HEAD_DIM
            || keys.len() != pairs * rows * HEAD_DIM
            || values.len() != keys.len()
        {
            return Err(Error::InvalidArgument(format!(
                "ICL attention shapes do not match: {pairs} pairs, {query_rows} queries, {rows} rows"
            )));
        }
        let block = UploadedBlock::new(self, precision, (keys, values), (pairs, rows))?;
        let (key_scales, value_scales) = (&block.key_scales, &block.value_scales);
        let padded = padded_query_rows(query_rows);
        let mut packed = vec![0u16; pairs * padded * HEAD_DIM];
        for pair in 0..pairs {
            let rows = &queries[pair * query_rows * HEAD_DIM..][..query_rows * HEAD_DIM];
            for (at, &query) in rows.iter().enumerate() {
                // The keys' scale folds into the queries, as QASSMax packs them.
                let scale = key_scales[pair * HEAD_DIM + at % HEAD_DIM];
                packed[pair * padded * HEAD_DIM + at] =
                    half::f16::from_f32(half::f16::from_bits(query).to_f32() * scale).to_bits();
            }
        }
        // FP32 queries hold the FP16 ones exactly; the scales are ones.
        let queries16 = self.upload(&packed)?;
        let widened: Vec<f32> = packed
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        let queries32 = self.upload(&widened)?;
        let queries = if precision == Precision::Fp32 {
            Activation::F32(&queries32)
        } else {
            Activation::Half(&queries16)
        };
        let (_, splits) = self.icl_split(precision, pairs, padded, rows);
        let zeroed = |len: usize| self.zeroed::<f32>(len).map_err(gpu_error);
        let (mut part, mut lse, mut out) = (
            zeroed(pairs * splits * padded * HEAD_DIM)?,
            zeroed(pairs * splits * padded)?,
            zeroed(pairs * padded * HEAD_DIM)?,
        );
        self.attend_icl(
            precision,
            queries,
            None,
            (&block.cache.keys, &block.cache.values),
            (&mut part, &mut lse, &mut out),
            (
                pairs,
                padded,
                rows,
                crate::kernels::cuda::launch::padded_rows(rows),
            ),
            1,
        )?;
        let padded_out = self.download(&out)?;
        let mut result = Vec::with_capacity(pairs * query_rows * HEAD_DIM);
        for pair in 0..pairs {
            let rows = &padded_out[pair * padded * HEAD_DIM..][..query_rows * HEAD_DIM];
            result.extend(
                rows.iter()
                    .enumerate()
                    .map(|(at, value)| value * value_scales[pair * HEAD_DIM + at % HEAD_DIM]),
            );
        }
        Ok(result)
    }

    /// Two products chained at `precision`, as a predictor runs them: `first
    /// = gelu(input w1ᵀ + b1)`, written in FP32 and, on tensor cores, as the
    /// next product's operand, then `second = first w2ᵀ + b2`, plus `first`
    /// when `w2` is square. `input` is `[rows, depth]`, encoded on the host as a
    /// producer would, `w1` `[width, depth]`, `w2` `[width2, width]`. Returns
    /// `(first, second)`.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent shapes or a failed launch.
    #[doc(hidden)]
    pub fn tensor_products(
        &self,
        precision: Precision,
        input: &[f32],
        (w1, b1): (&[f32], &[f32]),
        (w2, b2): (&[f32], &[f32]),
        (rows, width, depth, width2): (usize, usize, usize, usize),
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if input.len() != rows * depth
            || w1.len() != width * depth
            || b1.len() != width
            || w2.len() != width2 * width
            || b2.len() != width2
            || !depth.is_multiple_of(32)
            || !width.is_multiple_of(32)
        {
            return Err(Error::InvalidArgument(
                "tensor product shapes do not match".into(),
            ));
        }
        let (layer1, layer2) = (Dense::upload(self, w1, b1)?, Dense::upload(self, w2, b2)?);
        if precision == Precision::Fp32 {
            let input = self.upload(input)?;
            let zeroed = |len: usize| self.zeroed::<f32>(len).map_err(gpu_error);
            let (mut first, mut second) = (zeroed(rows * width)?, zeroed(rows * width2)?);
            // Room for the split-K partial sums, as a workspace reserves it.
            let mut partial = zeroed(4 * self.multiprocessors() * 64 * 64)?;
            self.product(
                Activation::F32(&input),
                &layer1,
                None,
                (Some(&mut first), None),
                &mut partial,
                (rows, width, depth),
                true,
            )?;
            let residual = (width2 == width).then_some(&first);
            self.product(
                Activation::F32(&first),
                &layer2,
                residual,
                (Some(&mut second), None),
                &mut partial,
                (rows, width2, width),
                false,
            )?;
            return Ok((self.download(&first)?, self.download(&second)?));
        }
        let high: Vec<u16> = input.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let low: Vec<u16> = input
            .iter()
            .map(|&v| half::f16::from_f32(super::weights::low_part(v)).to_bits())
            .collect();
        let (input_hi, input_lo) = (self.upload(&high)?, self.upload(&low)?);
        let zeroed16 = |len: usize| self.zeroed::<u16>(len).map_err(gpu_error);
        let zeroed32 = |len: usize| self.zeroed::<f32>(len).map_err(gpu_error);
        let (mut next_hi, mut next_lo) = (zeroed16(rows * width)?, zeroed16(rows * width)?);
        let (mut first, mut second) = (zeroed32(rows * width)?, zeroed32(rows * width2)?);
        // Tensor-core products write no split-K partial sums.
        let mut partial = zeroed32(0)?;
        let (input, next) = match precision {
            Precision::Half => (Activation::Half(&input_hi), Output::Half(&mut next_hi)),
            Precision::Split | Precision::Fp32 => (
                Activation::Split {
                    hi: &input_hi,
                    lo: &input_lo,
                },
                Output::Split {
                    hi: &mut next_hi,
                    lo: &mut next_lo,
                },
            ),
        };
        self.product(
            input,
            &layer1,
            None,
            (Some(&mut first), Some(next)),
            &mut partial,
            (rows, width, depth),
            true,
        )?;
        let next = match precision {
            Precision::Half => Activation::Half(&next_hi),
            Precision::Split | Precision::Fp32 => Activation::Split {
                hi: &next_hi,
                lo: &next_lo,
            },
        };
        let residual = (width2 == width).then_some(&first);
        self.product(
            next,
            &layer2,
            residual,
            (Some(&mut second), None),
            &mut partial,
            (rows, width2, width),
            false,
        )?;
        Ok((self.download(&first)?, self.download(&second)?))
    }
}
