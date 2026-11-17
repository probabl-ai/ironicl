//! The CUDA kernels for Ada and later GPUs: a build at `sm_89` (`make
//! cuda-sm89`) writes their PTX, which `IRONICL_CUDA_SM89_PTX` names and this
//! build embeds. Without it, every GPU runs the `sm_80` kernels the build
//! itself embeds.

use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo::rerun-if-env-changed=IRONICL_CUDA_SM89_PTX");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR")).join("kernels_sm89.ptx");
    let ptx = match env::var_os("IRONICL_CUDA_SM89_PTX").map(PathBuf::from) {
        Some(path) => {
            println!("cargo::rerun-if-changed={}", path.display());
            fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("IRONICL_CUDA_SM89_PTX {}: {error}", path.display()))
        }
        None => String::new(),
    };
    fs::write(&out, ptx).expect("OUT_DIR is writable");
}
