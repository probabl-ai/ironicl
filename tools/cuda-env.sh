# Source this file from the repository root before building Ironicl's `cuda`
# feature with a tool that runs Cargo itself, such as maturin, Clippy, or
# `cargo test`. It sets what `cargo oxide build` sets: cuda-oxide's codegen
# backend, which compiles the kernels in src/cuda/kernels.rs to PTX and embeds
# it, the target architecture, and the profile settings the backend requires.
# The backend also writes the PTX to target/cuda-oxide/<architecture>/.
#
#   . tools/cuda-env.sh                          # sm_80: Ampere and later
#   IRONICL_CUDA_ARCH=sm_89 . tools/cuda-env.sh  # Ada and later, with `ironicl_sm89`
#
# The sm_89 build only makes the PTX the main build embeds (`make cuda-sm89`);
# give it its own target directory, as its flags differ.
#
# The backend comes from `cargo oxide setup`, built from the cuda-rust
# revision in Cargo.toml; set CUDA_OXIDE_BACKEND if it is not in the default
# place.

arch=${IRONICL_CUDA_ARCH:-sm_80}
case $arch in
sm_80) cfgs= ;;
sm_89) cfgs=ironicl_sm89 ;;
*)
    echo "cuda-env.sh: unknown architecture $arch, not sm_80 or sm_89" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
backend=${CUDA_OXIDE_BACKEND:-$HOME/.cargo/cuda-oxide/librustc_codegen_cuda.so}
if [ ! -f "$backend" ]; then
    echo "cuda-env.sh: no cuda-oxide backend at $backend" >&2
    return 1 2>/dev/null || exit 1
fi
digest=$(sha256sum "$backend" | cut -d' ' -f1)
separator=$(printf '\037')
flags="--cfg${separator}cuda_oxide_internal_backend_identity=\"$digest\""
for cfg in $cfgs; do
    flags="$flags$separator--cfg$separator$cfg"
done
for flag in "-Zcodegen-backend=$backend" -Zmir-enable-passes=-JumpThreading -Zalways-encode-mir \
    -Csymbol-mangling-version=v0; do
    flags="$flags$separator$flag"
done
export CARGO_ENCODED_RUSTFLAGS="$flags" RUSTC_BOOTSTRAP=1 CUDA_OXIDE_TARGET=$arch \
    CUDA_OXIDE_PTX_DIR="$PWD/target/cuda-oxide/$arch"
mkdir -p "$CUDA_OXIDE_PTX_DIR"
unset RUSTFLAGS
# Cargo.toml's release LTO is untested with the cuda-oxide backend: keep the
# defaults until it is.
export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
for profile in DEV RELEASE; do
    export "CARGO_PROFILE_${profile}_OPT_LEVEL=3" "CARGO_PROFILE_${profile}_DEBUG_ASSERTIONS=false" \
        "CARGO_PROFILE_${profile}_OVERFLOW_CHECKS=false"
done
unset arch cfgs cfg backend digest separator flags flag profile
