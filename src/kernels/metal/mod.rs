//! Metal on Apple Silicon: the device, shared buffers, and the `MPSGraph`
//! plumbing models build their graphs with.

#![allow(unsafe_code)]
// Objective-C calls; each `unsafe` block states its invariant.
// Tensor dimensions are model constants or validated sizes far below `isize::MAX`.
#![allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]

use objc2::{AllocAnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphCompilationDescriptor, MPSGraphExecutableExecutionDescriptor,
    MPSGraphExecutionDescriptor, MPSGraphOptimization, MPSGraphTensor, MPSGraphTensorData,
};

use crate::{Error, Result, format::static_context::TensorStorage};

/// The `MPSGraph` element type for a stored tensor type.
pub(crate) fn kind(storage: TensorStorage) -> MPSDataType {
    match storage {
        TensorStorage::Fp32 => FLOAT,
        TensorStorage::F16 => MPSDataType::Float16,
        TensorStorage::Bf16 => MPSDataType::BFloat16,
    }
}

pub(crate) const FLOAT: MPSDataType = MPSDataType::Float32;

pub(crate) type Tensor = Retained<MPSGraphTensor>;
pub(crate) type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub(crate) type Weights<'w, 'a> = dyn Fn(&str) -> Result<&'a [f32]> + 'w;
/// A graph input or output: its tensor, buffer, shape, and data type.
pub(crate) type Binding = (Tensor, Buffer, Vec<isize>, MPSDataType);

pub(crate) fn shape(dimensions: &[isize]) -> Retained<MPSShape> {
    let numbers = dimensions
        .iter()
        .map(|&value| NSNumber::new_isize(value))
        .collect::<Vec<_>>();
    NSArray::from_retained_slice(&numbers)
}

pub(crate) fn metal_error(message: &str) -> Error {
    Error::InvalidArgument(format!("Metal backend: {message}"))
}

/// The memory Metal recommends the system GPU use without paging, or `None`
/// without a GPU. Creates no command queue.
pub(crate) fn system_working_set() -> Option<u64> {
    MTLCreateSystemDefaultDevice().map(|device| device.recommendedMaxWorkingSetSize())
}

/// The system GPU's name and the memory Metal recommends it use, or `None`
/// without a GPU.
pub(crate) fn system_gpu() -> Option<(String, u64)> {
    MTLCreateSystemDefaultDevice()
        .map(|device| (device.name().to_string(), device.recommendedMaxWorkingSetSize()))
}

/// The shared device and command queue, and how graphs run on them.
#[derive(Clone)]
pub(crate) struct Gpu {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) compilation: Retained<MPSGraphCompilationDescriptor>,
    pub(crate) execution: Retained<MPSGraphExecutionDescriptor>,
    pub(crate) executable_execution: Retained<MPSGraphExecutableExecutionDescriptor>,
}

impl Gpu {
    /// The system GPU, or `None` without one.
    pub(crate) fn new() -> Result<Option<Self>> {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            return Ok(None);
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| metal_error("cannot create a command queue"))?;
        // Optimization level 0 measured as fast as the default level 1 on an
        // M1 Max. Level 1's pass that places operations on the Neural Engine
        // or CPU printed "Failed to import MPS module" on every compile.
        // SAFETY: plain descriptor construction.
        let (compilation, execution, executable_execution) = unsafe {
            let compilation = MPSGraphCompilationDescriptor::new();
            compilation.setOptimizationLevel(MPSGraphOptimization::Level0);
            let execution = MPSGraphExecutionDescriptor::new();
            execution.setWaitUntilCompleted(true);
            execution.setCompilationDescriptor(Some(&compilation));
            let executable_execution = MPSGraphExecutableExecutionDescriptor::new();
            executable_execution.setWaitUntilCompleted(true);
            (compilation, execution, executable_execution)
        };
        Ok(Some(Self {
            device,
            queue,
            compilation,
            execution,
            executable_execution,
        }))
    }

    pub(crate) fn buffer(&self, values: &[f32]) -> Result<Buffer> {
        self.buffer_bytes(bytemuck::cast_slice(values))
    }

    pub(crate) fn buffer_bytes(&self, bytes: &[u8]) -> Result<Buffer> {
        let pointer = std::ptr::NonNull::new(bytes.as_ptr().cast_mut().cast())
            .ok_or_else(|| metal_error("empty buffer"))?;
        // SAFETY: Metal copies the bytes of a live slice into a new shared buffer.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Wrap `bytes` of existing page-aligned memory without copying it. The
    /// memory must outlive the buffer.
    pub(crate) fn wrap(&self, pointer: std::ptr::NonNull<u8>, bytes: usize) -> Result<Buffer> {
        // SAFETY: callers pass a page-aligned, page-padded region that outlives
        // the buffer; no deallocator, because the region's owner frees it.
        unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                pointer.cast(),
                bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or_else(|| metal_error("cannot wrap page-aligned memory"))
    }

    /// A buffer over a tensor's `bytes`: wrapped in place when `page_aligned`
    /// (the bytes start on a page inside page-padded memory), otherwise copied.
    pub(crate) fn tensor_buffer(&self, bytes: &[u8], page_aligned: bool) -> Result<Buffer> {
        if !page_aligned {
            return self.buffer_bytes(bytes);
        }
        let pointer = std::ptr::NonNull::from(bytes).cast::<u8>();
        self.wrap(
            pointer,
            bytes
                .len()
                .next_multiple_of(crate::format::static_context::TENSOR_PAGE_BYTES),
        )
    }

    /// The memory Metal recommends the device use without paging.
    pub(crate) fn working_set_bytes(&self) -> usize {
        usize::try_from(self.device.recommendedMaxWorkingSetSize()).unwrap_or(usize::MAX)
    }

    pub(crate) fn empty(&self, elements: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(elements * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Run `graph` synchronously, feeding `inputs` and writing `outputs`.
    pub(crate) fn run(&self, graph: &MPSGraph, inputs: &[&Binding], outputs: &[&Binding]) {
        // SAFETY: the descriptor waits until the run completes, and the
        // dictionaries keep their buffers alive.
        unsafe {
            graph.runAsyncWithMTLCommandQueue_feeds_targetOperations_resultsDictionary_executionDescriptor(
                &self.queue,
                &dictionary(inputs),
                None,
                &dictionary(outputs),
                Some(&self.execution),
            );
        }
    }
}

pub(crate) type Dictionary = Retained<NSDictionary<MPSGraphTensor, MPSGraphTensorData>>;

/// Map each binding's tensor to its buffer viewed with the binding's shape.
pub(crate) fn dictionary(items: &[&Binding]) -> Dictionary {
    let tensors: Vec<&MPSGraphTensor> = items.iter().map(|item| &*item.0).collect();
    let data: Vec<_> = items
        .iter()
        .map(|(_, buffer, dimensions, kind)| {
            // SAFETY: each buffer is live, shared, and at least as long as its shape.
            unsafe {
                MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                    MPSGraphTensorData::alloc(),
                    buffer,
                    &shape(dimensions),
                    *kind,
                )
            }
        })
        .collect();
    NSDictionary::from_retained_objects(&tensors, &data)
}
