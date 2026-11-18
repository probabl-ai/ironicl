# Source this file from the repository root before building Ironicl's `cuda`
# feature with a tool that runs Cargo itself, such as maturin, Clippy, or
# `cargo test`. It sets what `cargo oxide build` sets: cuda-oxide's codegen
# backend, which compiles the kernels in src/cuda/kernels.rs to PTX and embeds
# it, and the profile settings the backend requires. The backend also writes
# the PTX to target/cuda-oxide/.
#
#   . tools/cuda-env.sh && cargo build --features cuda
#
# build.rs picks the target architectures from IRONICL_CUDA_ARCH, `sm_80,sm_89`
# by default; `IRONICL_CUDA_ARCH=sm_120` builds for one GPU.
#
# The backend comes from `cargo oxide setup`, built from the cuda-rust
# revision in Cargo.toml; set CUDA_OXIDE_BACKEND if it is not in the default
# place.

backend=${CUDA_OXIDE_BACKEND:-$HOME/.cargo/cuda-oxide/librustc_codegen_cuda.so}
if [ ! -f "$backend" ]; then
    echo "cuda-env.sh: no cuda-oxide backend at $backend" >&2
    return 1 2>/dev/null || exit 1
fi
digest=$(sha256sum "$backend" | cut -d' ' -f1)
separator=$(printf '\037')
flags="--cfg${separator}cuda_oxide_internal_backend_identity=\"$digest\""
for flag in "-Zcodegen-backend=$backend" -Zmir-enable-passes=-JumpThreading -Zalways-encode-mir \
    -Csymbol-mangling-version=v0; do
    flags="$flags$separator$flag"
done
export CARGO_ENCODED_RUSTFLAGS="$flags" RUSTC_BOOTSTRAP=1 CUDA_OXIDE_PTX_DIR="$PWD/target/cuda-oxide"
mkdir -p "$CUDA_OXIDE_PTX_DIR"
unset RUSTFLAGS
for profile in DEV RELEASE; do
    export "CARGO_PROFILE_${profile}_OPT_LEVEL=3" "CARGO_PROFILE_${profile}_DEBUG_ASSERTIONS=false" \
        "CARGO_PROFILE_${profile}_OVERFLOW_CHECKS=false"
done
unset backend digest separator flags flag profile
