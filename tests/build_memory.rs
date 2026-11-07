#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#![allow(unsafe_code)] // One libproc call to read the process's peak footprint.

use std::path::Path;

use ironicl::{Executor, Optimize, TaskKind, load_model_artifact, memory::MemoryLimits, plan_context};

/// The process's lifetime peak physical footprint, which includes GPU buffers on
/// unified memory (`ri_lifetime_max_phys_footprint`, `rusage_info_v4`).
fn peak_footprint() -> u64 {
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    }
    let mut info = [0_u64; 64];
    // SAFETY: the buffer is larger than rusage_info_v4.
    let status = unsafe { proc_pid_rusage(std::process::id().cast_signed(), 4, info.as_mut_ptr()) };
    assert_eq!(status, 0, "proc_pid_rusage failed");
    // Skip the 16-byte UUID; field 28 is the lifetime peak footprint.
    info[2 + 28]
}

/// A wide context once built 4 estimators per Metal graph, sized to Metal's
/// whole working set: 21.4 GiB peak on an M1 Max for a 2 GiB context. The
/// memory plan sizes the groups from the memory available, and the build must
/// stay near the plan's peak, whatever this machine has free.
#[test]
#[ignore = "needs official checkpoint fixtures; run with --ignored once IRONICL_CHECKPOINT_FIXTURES is set. See CONTRIBUTING.md"]
#[allow(clippy::cast_precision_loss)]
fn wide_gpu_build_stays_within_the_memory_plan() {
    let directory = std::env::var_os("IRONICL_CHECKPOINT_FIXTURES")
        .expect("set IRONICL_CHECKPOINT_FIXTURES to run this ignored test; see CONTRIBUTING.md");
    let model = load_model_artifact(&Path::new(&directory).join("tabicl-classifier-v2.irmodel"))
        .expect("load classifier artifact");
    let executor = Executor::new(model);
    let (b, n, h) = (8, 10_000, 100);
    let plan = plan_context(
        TaskKind::Classifier,
        (b, n, h),
        Optimize::Auto,
        MemoryLimits::default(),
    )
    .expect("memory plan");
    let value = |index: usize| ((index * 7919) % 1000) as f32 / 250.0 - 2.0;
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
    let (peak, planned) = (gib(peak_footprint()), gib(plan.peak_bytes));
    assert!(
        peak <= planned * 1.15,
        "peak footprint {peak:.1} GiB exceeds the plan's {planned:.1} GiB by more than 15%:\n{plan}"
    );
}
