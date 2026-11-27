//! Metal on Apple Silicon: the device, shared buffers, and the `MPSGraph`
//! plumbing models build their graphs with.

#![expect(
    unsafe_code,
    reason = "Objective-C calls; each unsafe block states its invariant"
)]
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "tensor dimensions are model constants or validated sizes far below isize::MAX"
)]

use std::{collections::HashMap, sync::Mutex};

use objc2::{AllocAnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal::{MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::{MPSDataType, MPSShape};
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphCompilationDescriptor, MPSGraphExecutableExecutionDescriptor,
    MPSGraphExecutionDescriptor, MPSGraphOptimization, MPSGraphTensor, MPSGraphTensorData,
};

use crate::{Error, Result, format::container::DType};

/// The `MPSGraph` element type for a stored tensor type.
pub(crate) fn kind(storage: DType) -> MPSDataType {
    match storage {
        DType::F32 => FLOAT,
        DType::F16 => MPSDataType::Float16,
        DType::Bf16 => MPSDataType::BFloat16,
        other => unreachable!("Metal admission excludes {}", other.as_str()),
    }
}

pub(crate) const FLOAT: MPSDataType = MPSDataType::Float32;

pub(crate) mod block;
pub(crate) mod builder;

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
#[cfg(feature = "python")]
pub(crate) fn system_gpu() -> Option<(String, u64)> {
    MTLCreateSystemDefaultDevice()
        .map(|device| (device.name().to_string(), device.recommendedMaxWorkingSetSize()))
}

/// The shared device and command queue, and how graphs run on them.
#[derive(Clone)]
pub(crate) struct Gpu {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
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
        Ok(Some(Self { device, queue }))
    }

    pub(crate) fn buffer(&self, values: &[f32]) -> Result<Buffer> {
        self.buffer_bytes(bytemuck::cast_slice(values))
    }

    pub(crate) fn buffer_bytes(&self, bytes: &[u8]) -> Result<Buffer> {
        let pointer = std::ptr::NonNull::new(bytes.as_ptr().cast_mut().cast())
            .ok_or_else(|| metal_error("empty buffer"))?;
        // SAFETY: `pointer` and `bytes.len()` describe the live `bytes` slice, which
        // Metal copies into the new buffer before returning; it keeps no pointer.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                pointer,
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Wrap `bytes` of existing page-aligned memory without copying it.
    ///
    /// # Safety
    ///
    /// `pointer` starts a page and `bytes` is a page multiple inside one live allocation; the
    /// memory outlives every GPU use of the buffer; and no Rust reference reads it while the GPU
    /// writes it, or writes it while the GPU reads it.
    pub(crate) unsafe fn wrap(&self, pointer: std::ptr::NonNull<u8>, bytes: usize) -> Result<Buffer> {
        // SAFETY: the caller upholds the contract above (page-aligned, page-multiple, live for
        // the buffer's GPU use); no deallocator, because the region's owner frees it.
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
    ///
    /// # Safety
    ///
    /// With `page_aligned`, [`Self::wrap`]'s contract holds for `bytes` rounded up to
    /// [`TENSOR_PAGE_BYTES`](crate::format::static_context::TENSOR_PAGE_BYTES), and nothing
    /// writes `bytes` while the buffer lives: the returned buffer does not borrow them.
    pub(crate) unsafe fn tensor_buffer(&self, bytes: &[u8], page_aligned: bool) -> Result<Buffer> {
        if !page_aligned {
            return self.buffer_bytes(bytes);
        }
        let pointer = std::ptr::NonNull::from(bytes).cast::<u8>();
        // SAFETY: the caller upholds `wrap`'s contract for the rounded length (see above).
        unsafe {
            self.wrap(
                pointer,
                bytes
                    .len()
                    .next_multiple_of(crate::format::static_context::TENSOR_PAGE_BYTES),
            )
        }
    }

    /// The memory Metal recommends the device use without paging.
    pub(crate) fn working_set_bytes(&self) -> usize {
        usize::try_from(self.device.recommendedMaxWorkingSetSize()).unwrap_or(usize::MAX)
    }

    pub(crate) fn empty(&self, elements: usize) -> Result<Buffer> {
        self.empty_bytes(elements * size_of::<f32>())
    }

    pub(crate) fn empty_bytes(&self, bytes: usize) -> Result<Buffer> {
        self.device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| metal_error("cannot allocate a buffer"))
    }

    /// Run `graph` synchronously, feeding `inputs` and writing `outputs`.
    /// A fresh compilation descriptor. Apple does not document descriptors as safe to share
    /// across threads, so each compile and run makes its own; they are small.
    ///
    /// Optimization level 0 measured as fast as the default level 1 on an M1 Max. Level 1's
    /// pass that places operations on the Neural Engine or CPU printed "Failed to import MPS
    /// module" on every compile.
    pub(crate) fn compilation() -> Retained<MPSGraphCompilationDescriptor> {
        // SAFETY: a fresh descriptor; the setter takes a valid enum value.
        unsafe {
            let compilation = MPSGraphCompilationDescriptor::new();
            compilation.setOptimizationLevel(MPSGraphOptimization::Level0);
            compilation
        }
    }

    /// A fresh descriptor for a graph run that waits until the GPU finishes.
    fn execution() -> Retained<MPSGraphExecutionDescriptor> {
        // SAFETY: a fresh descriptor; the setters take a bool and an owned descriptor.
        unsafe {
            let execution = MPSGraphExecutionDescriptor::new();
            execution.setWaitUntilCompleted(true);
            execution.setCompilationDescriptor(Some(&Self::compilation()));
            execution
        }
    }

    /// A fresh descriptor for a compiled-executable run that waits until the GPU finishes.
    pub(crate) fn executable_execution() -> Retained<MPSGraphExecutableExecutionDescriptor> {
        // SAFETY: a fresh descriptor; the setter takes a bool.
        unsafe {
            let execution = MPSGraphExecutableExecutionDescriptor::new();
            execution.setWaitUntilCompleted(true);
            execution
        }
    }

    pub(crate) fn run(&self, graph: &MPSGraph, inputs: &[&Binding], outputs: &[&Binding]) {
        // SAFETY: the execution descriptor sets `waitUntilCompleted`, so the call returns
        // after the GPU finishes; the caller's bindings retain every buffer the
        // tensor data (which may not retain them) refer to for that whole time.
        unsafe {
            graph.runAsyncWithMTLCommandQueue_feeds_targetOperations_resultsDictionary_executionDescriptor(
                &self.queue,
                &dictionary(inputs),
                None,
                &dictionary(outputs),
                Some(&Self::execution()),
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
            // SAFETY: `buffer` is retained by the binding; every binding is built with a
            // buffer of at least `dimensions` elements of `kind` (see the callers).
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

/// A model's weights in shared GPU buffers, converted once and reused by every
/// context's graphs.
pub(crate) struct ModelWeights {
    pub(crate) gpu: Gpu,
    pub(crate) buffers: Mutex<HashMap<String, Buffer>>,
}

impl std::fmt::Debug for ModelWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ModelWeights")
    }
}

// SAFETY: the device and queue are thread-safe (`Send + Sync` in objc2-metal),
// and descriptors are made per compile and run, never shared; each buffer is filled when
// created and only read after; the map is behind a mutex. Retain and release
// are atomic.
unsafe impl Send for ModelWeights {}
// SAFETY: as above.
unsafe impl Sync for ModelWeights {}

impl ModelWeights {
    pub(crate) fn new(gpu: Gpu) -> Self {
        Self {
            gpu,
            buffers: Mutex::default(),
        }
    }

    pub(crate) fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub(crate) fn get(&self, key: &str, values: impl FnOnce() -> Vec<f32>) -> Result<Buffer> {
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| metal_error("weight cache lock poisoned"))?;
        if let Some(buffer) = buffers.get(key) {
            return Ok(buffer.clone());
        }
        let buffer = self.gpu.buffer(&values())?;
        buffers.insert(key.to_owned(), buffer.clone());
        Ok(buffer)
    }
}
