#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#![allow(unsafe_code)] // One libproc call to read the process's peak footprint.

use std::path::Path;

use ironicl::{
    Executor, Optimize, QueryRequest, TaskKind, load_model_artifact, memory::MemoryLimits, plan_context,
};

/// `ri_phys_footprint` and `ri_lifetime_max_phys_footprint` in `rusage_info_v4`:
/// the process's current and peak physical footprint, which include GPU
/// buffers on unified memory.
const CURRENT: usize = 7;
const PEAK: usize = 28;

fn footprint(field: usize) -> u64 {
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    }
    let mut info = [0_u64; 64];
    // SAFETY: the buffer is larger than rusage_info_v4.
    let status = unsafe { proc_pid_rusage(std::process::id().cast_signed(), 4, info.as_mut_ptr()) };
    assert_eq!(status, 0, "proc_pid_rusage failed");
    // Skip the 16-byte UUID.
    info[2 + field]
}

/// The tests measure the whole process, so they run one at a time. The build
/// test's peak covers whatever ran before it, which stays below its plan.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn executor() -> Executor {
    let directory = std::env::var_os("IRONICL_CHECKPOINT_FIXTURES")
        .expect("set IRONICL_CHECKPOINT_FIXTURES to run this ignored test; see CONTRIBUTING.md");
    let model = load_model_artifact(&Path::new(&directory).join("tabicl-classifier-v2.irmodel"))
        .expect("load classifier artifact");
    Executor::new(model)
}

#[allow(clippy::cast_precision_loss)]
fn value(index: usize) -> f32 {
    ((index * 7919) % 1000) as f32 / 250.0 - 2.0
}

/// A wide context once built 4 estimators per Metal graph, sized to Metal's
/// whole working set: 21.4 GiB peak on an M1 Max for a 2 GiB context. The
/// memory plan sizes the groups from the memory available, and the build must
/// stay near the plan's peak, whatever this machine has free.
#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
#[allow(clippy::cast_precision_loss)]
fn wide_gpu_build_stays_within_the_memory_plan() {
    let _serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let executor = executor();
    let (b, n, h) = (8, 10_000, 100);
    let plan = plan_context(
        TaskKind::Classifier,
        (b, n, h),
        Optimize::Auto,
        MemoryLimits::default(),
    )
    .expect("memory plan");
    let train: Vec<f32> = (0..b * n * h).map(value).collect();
    let y: Vec<f32> = (0..b * n).map(|row| (row % 3) as f32).collect();
    let context = executor
        .build_context(
            b,
            n,
            h,
            &train,
            &y,
            (0..b).flat_map(|_| 0..3).collect(),
            (0..b).flat_map(|_| 0..h as u64).collect(),
            Optimize::Auto,
            None,
            ironicl::memory::MemoryLimits::default(),
        )
        .expect("build context");
    drop(context);
    let gib = |bytes: u64| bytes as f64 / f64::from(1 << 30);
    let (peak, planned) = (gib(footprint(PEAK)), gib(plan.peak_bytes));
    assert!(
        peak <= planned * 1.15,
        "peak footprint {peak:.1} GiB exceeds the plan's {planned:.1} GiB by more than 15%:\n{plan}"
    );
}

/// `MPSGraph` keeps every query shape's buffers until its graph is freed, about
/// 2.3 MB per row at 8 estimators x 100 features, so predicts of many sizes
/// kept more memory with each. Queries now split into a set of shapes sized to
/// fit the device's share, here the memory limit.
#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
#[allow(clippy::cast_precision_loss)]
fn gpu_predict_keeps_buckets_within_the_memory_limit() {
    let _serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let (b, n, h) = (8, 2_000, 100);
    let train: Vec<f32> = (0..b * n * h).map(value).collect();
    let y: Vec<f32> = (0..b * n).map(|row| (row % 3) as f32).collect();
    let context = executor()
        .build_context(
            b,
            n,
            h,
            &train,
            &y,
            (0..b).flat_map(|_| 0..3).collect(),
            (0..b).flat_map(|_| 0..h as u64).collect(),
            Optimize::Auto,
            None,
            MemoryLimits::resolve(Some("4G"), None).expect("limits"),
        )
        .expect("build context");
    // Metal frees the build's buffers asynchronously.
    let settled = || {
        std::thread::sleep(std::time::Duration::from_secs(5));
        footprint(CURRENT)
    };
    let before = settled();
    let query: Vec<f32> = (0..b * 1_000 * h).map(|index| value(index + 1)).collect();
    // Sizes that each end in a different bucket.
    for rows in (1..=24).map(|step| step * 41) {
        let request = QueryRequest::new(b, rows, h, &query[..b * rows * h]).expect("query");
        let logits = context.predict_logits(request).expect("predict");
        assert!(logits.values().iter().all(|value| value.is_finite()));
    }
    let kept = settled().saturating_sub(before) as f64 / f64::from(1 << 30);
    assert!(kept <= 4.0, "predicts kept {kept:.1} GiB, past the 4 GiB limit");
}
