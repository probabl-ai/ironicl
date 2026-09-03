//! C ABI for resident cached-query benchmarks.
//!
//! This module is intentionally narrow. It exists so the Python benchmark can
//! time its common raw-query preprocessing and call a resident Ironicl executor
//! without process launch, artifact loading, or file I/O in the timed region.

#![allow(unsafe_code)] // C ABI pointer validation and copies are confined to this module.

use std::{
    ffi::{CStr, c_char, c_int},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
};

use crate::{ClassifierQueryRequest, StaticClassifierExecutor, StaticContextArtifact, load_model_artifact};

const SUCCESS: c_int = 0;
const INVALID_ARGUMENT: c_int = 1;
const EXECUTION_ERROR: c_int = 2;
const PANIC: c_int = 3;

pub struct CachedExecutor {
    executor: StaticClassifierExecutor,
}

fn write_error(buffer: *mut c_char, length: usize, message: &str) {
    if buffer.is_null() || length == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let count = bytes.len().min(length - 1);
    // The caller provides a writable C buffer with `length` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), count);
        *buffer.add(count) = 0;
    }
}

fn c_path(path: *const c_char, name: &str) -> Result<PathBuf, String> {
    if path.is_null() {
        return Err(format!("{name} path is null"));
    }
    // The caller must pass one NUL-terminated path for the duration of this call.
    let path = unsafe { CStr::from_ptr(path) };
    let path = path
        .to_str()
        .map_err(|_| format!("{name} path is not valid UTF-8"))?;
    Ok(PathBuf::from(path))
}

/// Load a model and static context once and return a resident opaque handle.
///
/// Returns null on error. If provided, `error_buffer` receives a NUL-terminated
/// diagnostic of at most `error_buffer_length - 1` bytes.
///
/// # Safety
///
/// The path pointers must be valid NUL-terminated UTF-8 strings. When provided,
/// `error_buffer` must be writable for `error_buffer_length` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironicl_static_executor_create(
    model_path: *const c_char,
    context_path: *const c_char,
    error_buffer: *mut c_char,
    error_buffer_length: usize,
) -> *mut CachedExecutor {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let model_path = c_path(model_path, "model")?;
        let context_path = c_path(context_path, "context")?;
        let model = load_model_artifact(&model_path).map_err(|error| error.to_string())?;
        let context = StaticContextArtifact::read(&context_path).map_err(|error| error.to_string())?;
        let executor = StaticClassifierExecutor::new(model, context).map_err(|error| error.to_string())?;
        Ok::<_, String>(Box::into_raw(Box::new(CachedExecutor { executor })))
    }));
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            write_error(error_buffer, error_buffer_length, &error);
            std::ptr::null_mut()
        }
        Err(_) => {
            write_error(
                error_buffer,
                error_buffer_length,
                "Ironicl executor creation panicked",
            );
            std::ptr::null_mut()
        }
    }
}

/// Run one resident cached prediction.
///
/// `query` has `ensemble_count * query_rows * feature_count` finite FP32 values
/// in `[B,Q,H]` row-major order. `output` has `query_rows * class_count` FP32
/// slots. Return zero on success and a nonzero error code otherwise.
///
/// # Safety
///
/// `handle` must be live and returned by `ironicl_static_executor_create`.
/// `query`, `output`, and an optional `error_buffer` must be valid for the
/// lengths specified by their arguments and must not alias mutable Rust state.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironicl_static_executor_predict_probabilities(
    handle: *const CachedExecutor,
    query: *const f32,
    ensemble_count: usize,
    query_rows: usize,
    feature_count: usize,
    output: *mut f32,
    output_length: usize,
    error_buffer: *mut c_char,
    error_buffer_length: usize,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || query.is_null() || output.is_null() {
            return Err((
                INVALID_ARGUMENT,
                "Ironicl handle, query, and output must be non-null".to_owned(),
            ));
        }
        let query_length = ensemble_count
            .checked_mul(query_rows)
            .and_then(|length| length.checked_mul(feature_count))
            .ok_or_else(|| (INVALID_ARGUMENT, "query dimensions overflow usize".to_owned()))?;
        // The caller promises the query and output ranges are valid for this call.
        let query = unsafe { std::slice::from_raw_parts(query, query_length) };
        let request = ClassifierQueryRequest::new(ensemble_count, query_rows, feature_count, query)
            .map_err(|error| (INVALID_ARGUMENT, error.to_string()))?;
        let executor = unsafe { &(*handle).executor };
        let probabilities = executor
            .predict_probabilities(request)
            .map_err(|error| (EXECUTION_ERROR, error.to_string()))?;
        if output_length != probabilities.values().len() {
            return Err((
                INVALID_ARGUMENT,
                format!(
                    "output length {output_length} does not match expected {}",
                    probabilities.values().len()
                ),
            ));
        }
        unsafe { std::ptr::copy_nonoverlapping(probabilities.values().as_ptr(), output, output_length) };
        Ok(())
    }));
    match result {
        Ok(Ok(())) => SUCCESS,
        Ok(Err((code, error))) => {
            write_error(error_buffer, error_buffer_length, &error);
            code
        }
        Err(_) => {
            write_error(
                error_buffer,
                error_buffer_length,
                "Ironicl cached prediction panicked",
            );
            PANIC
        }
    }
}

/// Run equal-sized prepared requests as one resident cache-sharing microbatch.
///
/// `query` stores `[request_count, B, Q, H]` row-major FP32 values. `output`
/// receives `[request_count, Q, C]` FP32 probabilities in request order.
///
/// # Safety
///
/// `handle`, `query`, `output`, and an optional `error_buffer` follow the same
/// validity requirements as [`ironicl_static_executor_predict_probabilities`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironicl_static_executor_predict_probabilities_batch(
    handle: *const CachedExecutor,
    query: *const f32,
    request_count: usize,
    ensemble_count: usize,
    query_rows: usize,
    feature_count: usize,
    output: *mut f32,
    output_length: usize,
    error_buffer: *mut c_char,
    error_buffer_length: usize,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || query.is_null() || output.is_null() || request_count == 0 {
            return Err((
                INVALID_ARGUMENT,
                "Ironicl handle, query, output, and request count must be valid".to_owned(),
            ));
        }
        let request_length = ensemble_count
            .checked_mul(query_rows)
            .and_then(|length| length.checked_mul(feature_count))
            .ok_or_else(|| (INVALID_ARGUMENT, "query dimensions overflow usize".to_owned()))?;
        let query_length = request_count.checked_mul(request_length).ok_or_else(|| {
            (
                INVALID_ARGUMENT,
                "query microbatch dimensions overflow usize".to_owned(),
            )
        })?;
        // The caller promises the query and output ranges are valid for this call.
        let query = unsafe { std::slice::from_raw_parts(query, query_length) };
        let requests = query
            .chunks_exact(request_length)
            .map(|values| ClassifierQueryRequest::new(ensemble_count, query_rows, feature_count, values))
            .collect::<crate::Result<Vec<_>>>()
            .map_err(|error| (INVALID_ARGUMENT, error.to_string()))?;
        let executor = unsafe { &(*handle).executor };
        let probabilities = executor
            .predict_probabilities_batch(&requests)
            .map_err(|error| (EXECUTION_ERROR, error.to_string()))?;
        let expected_length = request_count
            .checked_mul(query_rows)
            .and_then(|length| length.checked_mul(probabilities[0].shape()[1]))
            .ok_or_else(|| (INVALID_ARGUMENT, "output dimensions overflow usize".to_owned()))?;
        if output_length != expected_length {
            return Err((
                INVALID_ARGUMENT,
                format!("output length {output_length} does not match expected {expected_length}"),
            ));
        }
        let mut offset = 0_usize;
        for probabilities in probabilities {
            let values = probabilities.values();
            unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), output.add(offset), values.len()) };
            offset += values.len();
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => SUCCESS,
        Ok(Err((code, error))) => {
            write_error(error_buffer, error_buffer_length, &error);
            code
        }
        Err(_) => {
            write_error(
                error_buffer,
                error_buffer_length,
                "Ironicl cached microbatch prediction panicked",
            );
            PANIC
        }
    }
}

/// Reset sparse-attention decisions collected by a resident executor.
///
/// # Safety
///
/// `handle` must be live and returned by `ironicl_static_executor_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironicl_static_executor_reset_sparse_attention_statistics(
    handle: *const CachedExecutor,
) -> c_int {
    if handle.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { &(*handle).executor }.reset_sparse_attention_statistics();
    SUCCESS
}

/// Read sparse-attention decisions collected by a resident executor.
///
/// # Safety
///
/// `handle` and every output pointer must be valid for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironicl_static_executor_sparse_attention_statistics(
    handle: *const CachedExecutor,
    attempted: *mut u64,
    accepted: *mut u64,
    fallback: *mut u64,
) -> c_int {
    if handle.is_null() || attempted.is_null() || accepted.is_null() || fallback.is_null() {
        return INVALID_ARGUMENT;
    }
    let statistics = unsafe { &(*handle).executor }.sparse_attention_statistics();
    unsafe {
        *attempted = statistics.attempted;
        *accepted = statistics.accepted;
        *fallback = statistics.fallback;
    }
    SUCCESS
}

/// Destroy a handle returned by [`ironicl_static_executor_create`].
///
/// # Safety
///
/// `handle` must be null or a live handle returned by this ABI. It must be
/// destroyed exactly once and not be used after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironicl_static_executor_destroy(handle: *mut CachedExecutor) {
    if !handle.is_null() {
        // The caller transfers ownership of a handle created by this ABI.
        unsafe { drop(Box::from_raw(handle)) };
    }
}
