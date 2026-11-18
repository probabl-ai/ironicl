//! `MPSGraph` construction: placeholders bound to shared weight buffers, and
//! the operations models build their graphs from.

#![allow(unsafe_code)]
// Objective-C calls; each `unsafe` block states its invariant.
// Tensor dimensions are model constants or validated sizes far below `isize::MAX`.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use objc2::rc::Retained;
use objc2_foundation::NSData;
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::MPSGraph;

use super::{Binding, Buffer, FLOAT, ModelWeights, Tensor, Weights, shape};
use crate::Result;

/// `MPSGraph` construction helpers. Weights become placeholders bound to the
/// model's shared buffers; the bindings are kept for every run.
pub(crate) struct Builder<'w, 'a> {
    pub(crate) graph: Retained<MPSGraph>,
    pub(crate) weights: &'w Weights<'w, 'a>,
    pub(crate) model: &'w ModelWeights,
    pub(crate) bindings: std::cell::RefCell<Vec<Binding>>,
    /// The element type attention runs in when its keys and values are FP32.
    pub(crate) attention: std::cell::Cell<MPSDataType>,
}

// SAFETY (all methods): plain `MPSGraph` construction. Shapes come from the
// model's fixed dimensions and weights checked by its artifact.
impl<'w, 'a> Builder<'w, 'a> {
    pub(crate) fn new(model: &'w ModelWeights, weights: &'w Weights<'w, 'a>) -> Self {
        // SAFETY: see the impl comment.
        Self {
            graph: unsafe { MPSGraph::new() },
            weights,
            model,
            bindings: std::cell::RefCell::default(),
            attention: std::cell::Cell::new(FLOAT),
        }
    }

    pub(crate) fn set_attention(&self, kind: MPSDataType) {
        self.attention.set(kind);
    }

    pub(crate) fn weight(&self, name: &str) -> Result<&'a [f32]> {
        (self.weights)(name)
    }

    pub(crate) fn placeholder(&self, dimensions: &[isize]) -> Tensor {
        unsafe {
            self.graph
                .placeholderWithShape_dataType_name(Some(&shape(dimensions)), FLOAT, None)
        }
    }

    /// A placeholder bound to `buffer` for every run.
    pub(crate) fn bound(&self, buffer: Buffer, dimensions: &[isize]) -> Tensor {
        self.bound_typed(buffer, dimensions, FLOAT)
    }

    /// A placeholder of element type `kind` bound to `buffer`, in that type.
    pub(crate) fn bound_typed(&self, buffer: Buffer, dimensions: &[isize], kind: MPSDataType) -> Tensor {
        let tensor = unsafe {
            self.graph
                .placeholderWithShape_dataType_name(Some(&shape(dimensions)), kind, None)
        };
        self.bindings
            .borrow_mut()
            .push((tensor.clone(), buffer, dimensions.to_vec(), kind));
        tensor
    }

    /// `input` converted to element type `kind`; a no-op for the same type.
    pub(crate) fn cast(&self, input: &Tensor, kind: MPSDataType) -> Tensor {
        if unsafe { input.dataType() } == kind {
            return input.clone();
        }
        unsafe { self.graph.castTensor_toType_name(input, kind, None) }
    }

    /// A model weight under `key`, converted by `values` on first use.
    pub(crate) fn model_weight(
        &self,
        key: &str,
        values: impl FnOnce() -> Vec<f32>,
        dimensions: &[isize],
    ) -> Result<Tensor> {
        Ok(self.bound(self.model.get(key, values)?, dimensions))
    }

    pub(crate) fn named_weight(&self, name: &str, dimensions: &[isize]) -> Result<Tensor> {
        let values = self.weight(name)?;
        self.model_weight(name, || values.to_vec(), dimensions)
    }

    pub(crate) fn constant(&self, values: &[f32], dimensions: &[isize]) -> Tensor {
        let bytes = NSData::with_bytes(bytemuck::cast_slice(values));
        unsafe {
            self.graph
                .constantWithData_shape_dataType(&bytes, &shape(dimensions), FLOAT)
        }
    }

    pub(crate) fn scalar(&self, value: f64) -> Tensor {
        unsafe { self.graph.constantWithScalar_dataType(value, FLOAT) }
    }

    pub(crate) fn zeros(&self, dimensions: &[isize]) -> Tensor {
        unsafe {
            self.graph
                .constantWithScalar_shape_dataType(0.0, &shape(dimensions), FLOAT)
        }
    }

    pub(crate) fn concat(&self, left: &Tensor, right: &Tensor, dimension: isize) -> Tensor {
        unsafe {
            self.graph
                .concatTensor_withTensor_dimension_name(left, right, dimension, None)
        }
    }

    pub(crate) fn add(&self, left: &Tensor, right: &Tensor) -> Tensor {
        unsafe {
            self.graph
                .additionWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    pub(crate) fn multiply(&self, left: &Tensor, right: &Tensor) -> Tensor {
        unsafe {
            self.graph
                .multiplicationWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    pub(crate) fn matmul(&self, left: &Tensor, right: &Tensor) -> Tensor {
        unsafe {
            self.graph
                .matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(left, right, None)
        }
    }

    pub(crate) fn reshape(&self, input: &Tensor, dimensions: &[isize]) -> Tensor {
        unsafe {
            self.graph
                .reshapeTensor_withShape_name(input, &shape(dimensions), None)
        }
    }

    pub(crate) fn transpose(&self, input: &Tensor, first: usize, second: usize) -> Tensor {
        unsafe {
            self.graph
                .transposeTensor_dimension_withDimension_name(input, first, second, None)
        }
    }

    pub(crate) fn slice(&self, input: &Tensor, dimension: usize, start: isize, length: isize) -> Tensor {
        unsafe {
            self.graph
                .sliceTensor_dimension_start_length_name(input, dimension, start, length, None)
        }
    }

    /// `input @ weight^T + bias` for a `[outputs, inputs]` weight and its bias,
    /// or for block `part` of a weight and bias that stack several of them.
    pub(crate) fn affine(
        &self,
        input: &Tensor,
        (weight_name, bias_name): (&str, &str),
        part: usize,
        (inputs, outputs): (usize, usize),
    ) -> Result<Tensor> {
        let weight = &self.weight(weight_name)?[part * inputs * outputs..][..inputs * outputs];
        let transposed = self.model_weight(
            &format!("{weight_name}[{part}]^T"),
            || {
                let mut transposed = vec![0.0; inputs * outputs];
                for out in 0..outputs {
                    for index in 0..inputs {
                        transposed[index * outputs + out] = weight[out * inputs + index];
                    }
                }
                transposed
            },
            &[inputs as isize, outputs as isize],
        )?;
        let bias = &self.weight(bias_name)?[part * outputs..][..outputs];
        let bias = self.model_weight(
            &format!("{bias_name}[{part}]"),
            || bias.to_vec(),
            &[outputs as isize],
        )?;
        Ok(self.add(&self.matmul(input, &transposed), &bias))
    }

    pub(crate) fn linear(
        &self,
        input: &Tensor,
        prefix: &str,
        inputs: usize,
        outputs: usize,
    ) -> Result<Tensor> {
        self.affine(
            input,
            (&format!("{prefix}.weight"), &format!("{prefix}.bias")),
            0,
            (inputs, outputs),
        )
    }

    /// Exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    pub(crate) fn gelu(&self, input: &Tensor) -> Tensor {
        let scaled = self.multiply(input, &self.scalar(std::f64::consts::FRAC_1_SQRT_2));
        let one_plus = self.add(
            &unsafe { self.graph.erfWithTensor_name(&scaled, None) },
            &self.scalar(1.0),
        );
        self.multiply(&self.multiply(input, &self.scalar(0.5)), &one_plus)
    }

    pub(crate) fn layer_norm(&self, input: &Tensor, prefix: &str, width: usize) -> Result<Tensor> {
        let axes = shape(&[-1]);
        let gamma = self.named_weight(&format!("{prefix}.weight"), &[width as isize])?;
        let beta = self.named_weight(&format!("{prefix}.bias"), &[width as isize])?;
        unsafe {
            let mean = self.graph.meanOfTensor_axes_name(input, &axes, None);
            let variance = self
                .graph
                .varianceOfTensor_meanTensor_axes_name(input, &mean, &axes, None);
            Ok(self
                .graph
                .normalizationWithTensor_meanTensor_varianceTensor_gammaTensor_betaTensor_epsilon_name(
                    input,
                    &mean,
                    &variance,
                    Some(&gamma),
                    Some(&beta),
                    1e-5,
                    None,
                ))
        }
    }

    /// Non-interleaved `RoPE`: `x * cos + rotate_half(x) * sin`, with
    /// `rotate_half([left, right]) = [-right, left]`.
    pub(crate) fn rope(&self, input: &Tensor, (cos, sin): (&Tensor, &Tensor), head_dim: usize) -> Tensor {
        // `sin` carries the sign: [-sin, sin], so rotate_half is a half swap.
        let half = (head_dim / 2) as isize;
        let swapped = self.concat(
            &self.slice(input, 3, half, half),
            &self.slice(input, 3, 0, half),
            3,
        );
        self.add(&self.multiply(input, cos), &self.multiply(&swapped, sin))
    }
}
