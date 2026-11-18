//! The CUDA architectures the kernels build for: `IRONICL_CUDA_ARCH`, a
//! comma-separated list of targets from [`ARCHES`], `sm_80,sm_89` when unset:
//! Ampere and later, and Ada and later with FP8 conversions in hardware.
//!
//! cuda-oxide's backend builds one target per crate build. The first target is
//! this build's, set through `CUDA_OXIDE_TARGET`; each further target is a
//! nested build of this crate with that one target, in its own directory under
//! `OUT_DIR`, whose PTX this build embeds. Every build sets the `ironicl_cap_*`
//! cfgs of its target's capabilities. `OUT_DIR/cuda_modules.rs` lists the
//! modules for the device to choose from at run time (see `src/cuda/device.rs`).

use std::{env, fmt::Write as _, fs, path::PathBuf, process::Command};

/// Every capability the kernels test for: `fp8_cvt`, FP8 conversions in hardware.
const CAPABILITIES: &[&str] = &["fp8_cvt"];

/// The targets the kernels build for, and the capabilities they use on each.
/// A target is `sm_<major><minor>`: plain, for its compute capability and later
/// ones; `f`, for later ones of its major version; or `a`, for exactly its own.
const ARCHES: &[(&str, &[&str])] = &[
    ("sm_80", &[]),
    ("sm_86", &[]),
    ("sm_87", &[]),
    ("sm_89", &["fp8_cvt"]),
    ("sm_90", &["fp8_cvt"]),
    ("sm_90a", &["fp8_cvt"]),
    ("sm_100", &["fp8_cvt"]),
    ("sm_100f", &["fp8_cvt"]),
    ("sm_100a", &["fp8_cvt"]),
    ("sm_103", &["fp8_cvt"]),
    ("sm_103f", &["fp8_cvt"]),
    ("sm_103a", &["fp8_cvt"]),
    ("sm_110", &["fp8_cvt"]),
    ("sm_110f", &["fp8_cvt"]),
    ("sm_110a", &["fp8_cvt"]),
    ("sm_120", &["fp8_cvt"]),
    ("sm_120f", &["fp8_cvt"]),
    ("sm_120a", &["fp8_cvt"]),
    ("sm_121", &["fp8_cvt"]),
    ("sm_121f", &["fp8_cvt"]),
    ("sm_121a", &["fp8_cvt"]),
];

fn main() {
    for capability in CAPABILITIES {
        println!("cargo::rustc-check-cfg=cfg(ironicl_cap_{capability})");
    }
    println!("cargo::rerun-if-env-changed=IRONICL_CUDA_ARCH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let list = env::var("IRONICL_CUDA_ARCH").unwrap_or_default();
    let targets: Vec<&str> = list
        .split(',')
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .collect();
    let targets = if targets.is_empty() {
        vec!["sm_80", "sm_89"]
    } else {
        targets
    };
    for target in &targets {
        if !ARCHES.iter().any(|(name, _)| name == target) {
            let names: Vec<&str> = ARCHES.iter().map(|(name, _)| *name).collect();
            panic!(
                "IRONICL_CUDA_ARCH: unknown target {target}; choose from {}",
                names.join(", ")
            );
        }
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));

    let (first, rest) = targets.split_first().expect("at least one target");
    println!("cargo::rustc-env=CUDA_OXIDE_TARGET={first}");
    let (_, capabilities) = ARCHES
        .iter()
        .find(|(name, _)| name == first)
        .expect("checked above");
    for capability in *capabilities {
        println!("cargo::rustc-cfg=ironicl_cap_{capability}");
    }

    let mut modules = format!("&[\n    {},\n", module(first, "None"));
    if !rest.is_empty() {
        // A nested build rebuilds when the crate's sources change.
        for path in ["src", "build.rs", "Cargo.toml", "Cargo.lock"] {
            println!("cargo::rerun-if-changed={path}");
        }
    }
    for target in rest {
        let ptx = nested_build(&out.join("cuda").join(target), target);
        writeln!(
            modules,
            "    {},",
            module(target, &format!("Some(include_str!({:?}))", ptx.display()))
        )
        .expect("writing to a String");
    }
    modules.push(']');
    fs::write(out.join("cuda_modules.rs"), modules).expect("OUT_DIR is writable");
}

/// One entry of `cuda_modules.rs`: `Module { name, ptx }`.
fn module(target: &str, ptx: &str) -> String {
    format!("Module {{ name: {target:?}, ptx: {ptx} }}")
}

/// Builds this crate's library for `target` alone in `dir`, and returns its PTX.
fn nested_build(dir: &std::path::Path, target: &str) -> PathBuf {
    let ptx_dir = dir.join("ptx");
    fs::create_dir_all(&ptx_dir).expect("OUT_DIR is writable");
    let mut command = Command::new(env::var_os("CARGO").expect("Cargo sets CARGO"));
    command
        .args([
            "build",
            "--locked",
            "--lib",
            "--features",
            "cuda",
            "--manifest-path",
        ])
        .arg(PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets it")).join("Cargo.toml"))
        .arg("--target-dir")
        .arg(dir.join("target"))
        .env("IRONICL_CUDA_ARCH", target)
        .env("CUDA_OXIDE_PTX_DIR", &ptx_dir)
        // Under `cargo clippy`, compile the nested build without linting it.
        .env_remove("RUSTC_WORKSPACE_WRAPPER");
    if env::var("PROFILE").as_deref() == Ok("release") {
        command.arg("--release");
    }
    let output = command.output().expect("running Cargo");
    assert!(
        output.status.success(),
        "the nested build for {target} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ptx = ptx_dir.join("ironicl.ptx");
    assert!(
        ptx.is_file(),
        "the nested build for {target} wrote no PTX at {}",
        ptx.display()
    );
    ptx
}
