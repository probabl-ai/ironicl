//! Ironicl's CUDA kernels, compiled to PTX by cuda-oxide's codegen backend
//! and embedded in the library, once per target in `IRONICL_CUDA_ARCH` (see
//! `build.rs`). A target's capabilities set `ironicl_cap_*` cfgs:
//! `ironicl_cap_fp8_cvt` (Ada and later) converts FP8 in hardware.
//!
//! ICL attention: every (estimator, head) pair attends with Q query rows to N
//! cached training rows, in Ironicl's `[estimators x heads, N, 64]` cache
//! layout. The attention kernels write a normalized partial output and its
//! log-sum-exp per chunk of N; `icl_attention_merge` combines the chunks.
//!
//! Each tensor-core family (`_half`, `_split`) has an `_f32` counterpart on
//! the CUDA cores for `optimize="safe"`, which never touches FP16.

use cuda_host::cuda_module;

#[cuda_module]
#[allow(
    unsafe_code,
    reason = "shared memory, tensor-core intrinsics, and unchecked indexing; each unsafe block states its invariant"
)]
#[allow(
    clippy::no_mangle_with_rust_abi,
    reason = "cuda-oxide exports kernels and device functions with #[no_mangle] and the Rust ABI"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's parameters are its launch ABI, which device bodies take on"
)]
#[allow(clippy::too_many_lines, reason = "a kernel body is one loop nest, kept whole")]
#[allow(clippy::many_single_char_names, reason = "the math's names: q, k, v, m, n, k")]
#[allow(
    clippy::similar_names,
    reason = "paired high and low parts such as w_hi and w_lo"
)]
#[allow(
    clippy::needless_range_loop,
    reason = "#[unroll] loops index register arrays in step"
)]
#[allow(
    clippy::cast_ptr_alignment,
    reason = "vector loads at offsets aligned by construction"
)]
#[allow(
    clippy::items_after_statements,
    reason = "#[launch_bounds] expands ahead of a kernel's shared statics"
)]
pub(crate) mod device {
    use cuda_device::vector::F32x4;
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread, warp};

    use crate::kernels::cuda::ops::{
        D, attend16_f32, attention16_tc_body, block_head_body, block_tail_body, block_tail_core_f32,
        column_block_body, e4m3, fast_body, linear16_body, load_normed_f32, max16, product4x4,
        store_activation, sum16,
    };

    /// Tensor-core partial attention over FP16 keys and values, 64 queries per
    /// block (wide batches).
    #[kernel]
    pub fn icl_attention_wide(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<64, false, false, false>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// Tensor-core partial attention over FP16 keys and values, 16 queries per
    /// block, keys split over warps.
    #[kernel]
    pub fn icl_attention_decode(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<16, true, false, false>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// [`icl_attention_wide`] over FP8 keys and values, accumulating in FP16, for
    /// `optimize="fast"`.
    #[kernel]
    pub fn icl_attention_wide_fp8(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<64, false, true, true>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// [`icl_attention_decode`] over FP8 keys and values, for `optimize="fast"`.
    #[kernel]
    pub fn icl_attention_decode_fp8(
        q: &[u16],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        fast_body::<16, true, false, true>(
            q,
            k,
            v,
            &mut part,
            &mut lse,
            qp,
            np,
            n_valid,
            chunk_tiles,
            splits,
            scale_log2,
        );
    }

    /// Combines the chunks' partial outputs by their log2-domain log-sum-exp:
    /// one thread per output element of `[BH, Qp, 64]`.
    #[kernel]
    pub fn icl_attention_merge(
        part: &[f32],
        lse: &[f32],
        mut out: DisjointSlice<f32>,
        bh_count: u32,
        qp: u32,
        splits: u32,
    ) {
        let gid = thread::index_1d();
        let i = gid.get();
        let (qp, splits) = (qp as usize, splits as usize);
        if i >= bh_count as usize * qp * D {
            return;
        }
        let (bh, rest) = (i / (qp * D), i % (qp * D));
        let (q, d) = (rest / D, rest % D);
        let mut max = f32::NEG_INFINITY;
        for s in 0..splits {
            max = max.max(lse[(bh * splits + s) * qp + q]);
        }
        let (mut total, mut acc) = (0.0f32, 0.0f32);
        for s in 0..splits {
            let row = (bh * splits + s) * qp + q;
            let w = (lse[row] - max).exp2();
            total += w;
            acc += w * part[row * D + d];
        }
        if let Some(slot) = out.get_mut(gid) {
            *slot = acc / total;
        }
    }

    // ---------------------------------------------------------------------------
    // Dense stages in FP32, for products too small or narrow for the tensor-core
    // kernels.
    // ---------------------------------------------------------------------------

    /// `LayerNorm` over `width` values (at most 512) of each of `rows` rows (row
    /// strides `ldx`, `ldy`), epsilon 1e-5, population variance, written as
    /// [`store_activation`] does with `format`. One warp per row; each lane loads
    /// its values once, all loads in flight together, and keeps them in
    /// registers. 256 threads per block.
    #[kernel]
    pub fn layer_norm(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut y: DisjointSlice<f32>,
        mut y_hi: DisjointSlice<u16>,
        mut y_lo: DisjointSlice<u16>,
        rows: u32,
        width: u32,
        ldx: u32,
        ldy: u32,
        format: u32,
    ) {
        let global =
            thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize;
        let (row, lane) = (global / 32, global % 32);
        let (width, ldx, ldy) = (width as usize, ldx as usize, ldy as usize);
        if row >= rows as usize {
            return;
        }
        let mut values = [0.0f32; 16];
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0..16 {
            let i = lane + 32 * j;
            if i < width {
                // SAFETY: `row < rows` (returned above, warp-uniformly) and `i < width = ldx`; the
                // launcher passes `x` as `[rows, width]`.
                values[j] = unsafe { *x.get_unchecked(row * ldx + i) };
                sum += values[j];
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "At most 512")]
        let count = width as f32;
        let mean = warp::reduce_sum_f32(sum) / count;
        let mut squares = 0.0f32;
        #[unroll]
        for j in 0..16 {
            if lane + 32 * j < width {
                let d = values[j] - mean;
                squares += d * d;
            }
        }
        let inverse = 1.0 / libm::sqrtf(warp::reduce_sum_f32(squares) / count + 1e-5);
        #[unroll]
        for j in 0..16 {
            let i = lane + 32 * j;
            if i < width {
                let value = (values[j] - mean) * inverse * gamma[i] + beta[i];
                // SAFETY: `row < rows` and `i < width = ldy`; the launcher sizes the buffers
                // `format` names to `[rows, width]`, and each (row, i) is one lane's.
                unsafe { store_activation((&mut y, &mut y_hi, &mut y_lo), format, row * ldy + i, value) };
            }
        }
    }

    /// `x wᵀ` for 64 x 64 output tiles over one split of k, double-buffered in
    /// 16-wide K steps with 16-byte loads; 256 threads with 4 x 4 outputs each.
    /// With one split it applies the epilogue like [`linear`] and writes `y`;
    /// otherwise split `s` (grid z) covers k in `[s * k_chunk, (s + 1) * k_chunk)`
    /// and writes its raw sums to `partial[s, m, n]` for [`linear_reduce`].
    /// Needs `k`, `k_chunk`, and `ldx` to be multiples of 4 (`k_chunk` of 16) and
    /// 16-byte aligned `x` and `w`. Grid: (ceil(n / 64), ceil(m / 64), splits).
    #[kernel]
    pub fn linear64(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        mut partial: DisjointSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
        ldx: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
        k_chunk: u32,
    ) {
        static mut AS: SharedArray<f32, { 2 * 16 * 64 }, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, { 2 * 16 * 64 }, 16> = SharedArray::UNINIT;
        // SAFETY: `AS` and `BS` are this kernel's own shared statics, reached only through these
        // raw pointers; every access below stays in their `2 * 16 * 64` floats and is ordered by
        // `sync_threads`.
        let a_tile = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        // SAFETY: as above, for `BS`.
        let b_tile = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BS)) };
        let tid = thread::threadIdx_x() as usize;
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let (ldx, ldy, ldr) = (ldx as usize, ldy as usize, ldr as usize);
        let row0 = thread::blockIdx_y() as usize * 64;
        let col0 = thread::blockIdx_x() as usize * 64;
        let split = thread::blockIdx_z() as usize;
        let splits = thread::gridDim_z() as usize;
        let k_begin = split * k_chunk as usize;
        let k_end = if k_begin + (k_chunk as usize) < k {
            k_begin + k_chunk as usize
        } else {
            k
        };
        let (ty, tx) = (tid / 16, tid % 16);
        let (load_row, load_k) = (tid / 4, (tid % 4) * 4);
        let (xp, wp) = (x.as_ptr(), w.as_ptr());
        let fetch = |k0: usize| -> (F32x4, F32x4) {
            let gk = k0 + load_k;
            let (gr, gc) = (row0 + load_row, col0 + load_row);
            let a = if gr < m && gk < k_end {
                // SAFETY: `gr < m` and `gk < k_end <= k <= ldx`, with `gk`, `k_end`, and `k`
                // multiples of 4, so all 16 bytes lie in `x` `[m, ldx]`; 16-byte aligned since
                // `ldx` and `gk` are multiples of 4 and `x` is aligned.
                unsafe { *xp.add(gr * ldx + gk).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            let bv = if gc < n && gk < k_end {
                // SAFETY: `gc < n` and, as for `x`, `gk + 4 <= k`, so the 16 aligned bytes lie in
                // `w` `[n, k]`.
                unsafe { *wp.add(gc * k + gk).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            (a, bv)
        };
        let store = |buf: usize, (a, bv): (F32x4, F32x4)| {
            #[unroll]
            for i in 0..4 {
                // SAFETY: `buf < 2`, `load_k + i < 16`, and `load_row = tid / 4 < 64` (256
                // threads), so the index is below the tiles' `2 * 1024` floats; each thread writes
                // only its own slots, into the buffer no thread reads before the next barrier.
                unsafe {
                    *a_tile.add(buf * 1024 + (load_k + i) * 64 + load_row) = a.0[i];
                    *b_tile.add(buf * 1024 + (load_k + i) * 64 + load_row) = bv.0[i];
                }
            }
        };
        let mut acc = [[0.0f32; 4]; 4];
        store(0, fetch(k_begin));
        thread::sync_threads();
        let steps = (k_end - k_begin).div_ceil(16);
        let mut step = 0;
        while step < steps {
            let buf = step % 2;
            let next = if step + 1 < steps {
                fetch(k_begin + (step + 1) * 16)
            } else {
                (F32x4::splat(0.0), F32x4::splat(0.0))
            };
            #[unroll]
            for kk in 0..16 {
                let base = buf * 1024 + kk * 64;
                // SAFETY: `buf < 2`, `kk < 16`, `ty < 16`: `base + ty * 4 + 4 <= 2 * 1024`, a
                // 16-byte aligned offset of the 16-byte aligned tile, stored before the last
                // barrier.
                let a = unsafe { *(a_tile.add(base + ty * 4) as *const F32x4) };
                // SAFETY: as above, with `tx < 16`.
                let bv = unsafe { *(b_tile.add(base + tx * 4) as *const F32x4) };
                #[unroll]
                for i in 0..4 {
                    #[unroll]
                    for j in 0..4 {
                        acc[i][j] += a.0[i] * bv.0[j];
                    }
                }
            }
            if step + 1 < steps {
                store(1 - buf, next);
            }
            thread::sync_threads();
            step += 1;
        }
        #[unroll]
        for i in 0..4 {
            let row = row0 + ty * 4 + i;
            if row < m {
                #[unroll]
                for j in 0..4 {
                    let col = col0 + tx * 4 + j;
                    if col < n {
                        if splits > 1 {
                            // SAFETY: `split < splits`, `row < m`, `col < n`; with more than one
                            // split the launcher asserts `splits * m * n <= partial.len()`, and
                            // each (split, row, col) belongs to this thread alone.
                            unsafe { *partial.get_unchecked_mut((split * m + row) * n + col) = acc[i][j] };
                        } else {
                            let mut value = acc[i][j] + b[col];
                            if mode & 1 != 0 {
                                value = 0.5
                                    * value
                                    * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2));
                            }
                            if mode & 2 != 0 {
                                value += r[row * ldr + col];
                            }
                            // SAFETY: `row < m` and `col < n = ldy` (the launcher), inside its `[m,
                            // n]` output; each (row, col) belongs to this thread alone.
                            unsafe { *y.get_unchecked_mut(row * ldy + col) = value };
                        }
                    }
                }
            }
        }
    }

    /// Sum [`linear64`]'s split partials `[splits, m, n]` and apply the epilogue:
    /// bias, then GELU if `mode` bit 1, then the residual `r` if bit 2.
    #[kernel]
    pub fn linear_reduce(
        partial: &[f32],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        m: u32,
        n: u32,
        splits: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
    ) {
        let i = thread::index_1d().get();
        let (m, n) = (m as usize, n as usize);
        if i >= m * n {
            return;
        }
        let (row, col) = (i / n, i % n);
        let mut value = b[col];
        let mut s = 0;
        while s < splits as usize {
            value += partial[s * m * n + i];
            s += 1;
        }
        if mode & 1 != 0 {
            value = 0.5 * value * (1.0 + libm::erff(value * core::f32::consts::FRAC_1_SQRT_2));
        }
        if mode & 2 != 0 {
            value += r[row * ldr as usize + col];
        }
        // SAFETY: `i < m * n` and `ldy = n` (the launcher), so the index is `i`: this thread's own,
        // inside `y` `[m, n]`.
        unsafe { *y.get_unchecked_mut(row * ldy as usize + col) = value };
    }

    /// The column stage's input layer from raw query rows: for ensemble `e`,
    /// feature `f`, and query row `i` of rows `x[e, start + i, :]` (`[B, total, H]`),
    /// groups the features `f + 1`, `f + 2`, and `f + 4` (mod H) and writes
    /// `w · group + b` as `out[(e * H + f) * rows + i, 0..128]`. Sets
    /// `missing[start + i]` to 1 when a group of the row is the all-missing marker
    /// (-100), which only the CPU column stage handles. One thread per output value.
    #[kernel]
    pub fn column_input(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        mut out: DisjointSlice<f32>,
        mut missing: DisjointSlice<f32>,
        ensembles: u32,
        features: u32,
        total: u32,
        start: u32,
        rows: u32,
    ) {
        let i = thread::index_1d().get();
        let (h, rows) = (features as usize, rows as usize);
        if i >= ensembles as usize * h * rows * 128 {
            return;
        }
        let (c, rest) = (i % 128, i / 128);
        let (row, column) = (rest % rows, rest / rows);
        let (e, f) = (column / h, column % h);
        let base = (e * total as usize + start as usize + row) * h;
        let group = [
            x[base + (f + 1) % h],
            x[base + (f + 2) % h],
            x[base + (f + 4) % h],
        ];
        #[allow(clippy::float_cmp, reason = "The all-missing marker is exactly -100")]
        let marked = group[0] == -100.0 && group[1] == -100.0 && group[2] == -100.0;
        if c == 0 && marked {
            // The `c == 0` thread of every (e, f) may flag the same row, so the store is atomic.
            // SAFETY: `row < rows`, and the launcher's `missing` holds `start + rows` rows, so the
            // pointer is in bounds and aligned for `f32`; this kernel accesses `missing` only
            // through this atomic store.
            let flag = unsafe {
                cuda_device::atomic::DeviceAtomicF32::from_ptr(missing.as_mut_ptr().add(start as usize + row))
            };
            flag.store(1.0, cuda_device::atomic::AtomicOrdering::Relaxed);
        }
        let value = group[0] * w[c * 3] + group[1] * w[c * 3 + 1] + group[2] * w[c * 3 + 2] + b[c];
        // SAFETY: `i < ensembles * H * rows * 128`, the size of the launcher's `out` `[B * H, rows,
        // 128]`; `i` is this thread's own.
        unsafe { *out.get_unchecked_mut(i) = value };
    }

    /// Row-stage sequences `[B * Q, 4 + H, 128]`: the four CLS tokens, then each
    /// feature's column embedding of rows `start..start + Q` of `[B * H, total, 128]`.
    #[kernel]
    pub fn build_rows(
        columns: &[f32],
        cls: &[f32],
        mut rows: DisjointSlice<f32>,
        ensembles: u32,
        features: u32,
        queries: u32,
        total: u32,
        start: u32,
    ) {
        let i = thread::index_1d().get();
        let (h, q) = (features as usize, queries as usize);
        let len = h + 4;
        if i >= ensembles as usize * q * len * 128 {
            return;
        }
        let (c, token, seq) = (i % 128, (i / 128) % len, i / (128 * len));
        let value = if token < 4 {
            cls[token * 128 + c]
        } else {
            let (b, query) = (seq / q, seq % q);
            columns[((b * h + token - 4) * total as usize + start as usize + query) * 128 + c]
        };
        // SAFETY: `i < ensembles * q * (H + 4) * 128`, the size of the launcher's `rows`; `i` is
        // this thread's own.
        unsafe { *rows.get_unchecked_mut(i) = value };
    }

    /// The CLS tokens of each `[len, 128]` sequence, as `[seqs * 4, 128]`.
    #[kernel]
    pub fn gather_cls(rows: &[f32], mut cls: DisjointSlice<f32>, seqs: u32, len: u32) {
        let i = thread::index_1d().get();
        if i >= seqs as usize * 4 * 128 {
            return;
        }
        let (c, token, seq) = (i % 128, (i / 128) % 4, i / 512);
        // SAFETY: `i < seqs * 4 * 128`, the size of the launcher's `cls`; `i` is this thread's own.
        unsafe { *cls.get_unchecked_mut(i) = rows[(seq * len as usize + token) * 128 + c] };
    }

    /// `QASSMax` and packing for ICL attention: for token `t = b * Q + q` and head
    /// `j`, `q * base[j] * (1 + tanh(modulation)) * key_scale[b * 8 + j]` at
    /// `[(b * 8 + j) * qp + q, 64]`, in FP32 into `packed` (`format` 0) or in
    /// FP16 into `packed_hi` (1); `key_scale` is the FP8 keys' scale per dim, or
    /// ones.
    #[kernel]
    pub fn qass_pack(
        query: &[f32],
        modulation: &[f32],
        base: &[f32],
        key_scale: &[f32],
        mut packed: DisjointSlice<f32>,
        mut packed_hi: DisjointSlice<u16>,
        ensembles: u32,
        queries: u32,
        qp: u32,
        format: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let gate = 1.0 + modulation[(token * 8 + head) * 64 + d].tanh();
        let value = query[i] * base[head * 64 + d] * gate * key_scale[(b * 8 + head) * 64 + d];
        let at = ((b * 8 + head) * qp as usize + row) * 64 + d;
        // SAFETY: `b < ensembles`, `head < 8`, `row < q <= qp`, `d < 64`: `at` lies in `[B * 8, qp,
        // 64]` and is this thread's alone. `format` selects the buffer the launcher sized
        // (`Unused::outputs`); the other may be empty and is not touched.
        unsafe {
            if format == 0 {
                *packed.get_unchecked_mut(at) = value;
            } else {
                *packed_hi.get_unchecked_mut(at) = (value as f16).to_bits();
            }
        }
    }

    /// Attention outputs `[(b * 8 + j) * qp + q, 64]` back to tokens `[B * Q, 512]`,
    /// times `value_scale[b * 8 + j]`, the FP8 values' scale per dim, or ones;
    /// written as [`store_activation`] does with `format`.
    #[kernel]
    pub fn heads_to_tokens(
        attended: &[f32],
        value_scale: &[f32],
        mut tokens: DisjointSlice<f32>,
        mut tokens_hi: DisjointSlice<u16>,
        mut tokens_lo: DisjointSlice<u16>,
        ensembles: u32,
        queries: u32,
        qp: u32,
        format: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let pair = b * 8 + head;
        let value = attended[(pair * qp as usize + row) * 64 + d] * value_scale[pair * 64 + d];
        // SAFETY: `i < B * q * 512` (returned above), within the buffers `format` names, which the
        // launcher sizes to at least `[B q, 512]`; one thread per `i`.
        unsafe { store_activation((&mut tokens, &mut tokens_hi, &mut tokens_lo), format, i, value) };
    }

    // ---------------------------------------------------------------------------
    // Building a context: the training rows' pass, which keeps keys and values.
    // ---------------------------------------------------------------------------

    /// The column stage's input layer for training columns `first..first +
    /// columns` of `[B * H]`, over all `rows` training rows `x[B, rows, H]`,
    /// plus each estimator's label embeddings `labels[B, rows, 128]`, into
    /// `out[columns, rows, 128]`. The caller has checked for the all-missing
    /// marker. One thread per output value.
    #[kernel]
    pub fn column_train(
        x: &[f32],
        w: &[f32],
        b: &[f32],
        labels: &[f32],
        mut out: DisjointSlice<f32>,
        features: u32,
        rows: u32,
        first: u32,
        columns: u32,
    ) {
        let i = thread::index_1d().get();
        let (h, rows) = (features as usize, rows as usize);
        if i >= columns as usize * rows * 128 {
            return;
        }
        let (c, rest) = (i % 128, i / 128);
        let (row, column) = (rest % rows, first as usize + rest / rows);
        let (e, f) = (column / h, column % h);
        let base = (e * rows + row) * h;
        let group = [
            x[base + (f + 1) % h],
            x[base + (f + 2) % h],
            x[base + (f + 4) % h],
        ];
        let value = group[0] * w[c * 3] + group[1] * w[c * 3 + 1] + group[2] * w[c * 3 + 2] + b[c];
        // SAFETY: `i < columns * rows * 128`, the size of the launcher's `out`; `i` is this
        // thread's own.
        unsafe { *out.get_unchecked_mut(i) = value + labels[(e * rows + row) * 128 + c] };
    }

    /// `out[i] = source[i % period]` for `len` values.
    #[kernel]
    pub fn repeat(source: &[f32], mut out: DisjointSlice<f32>, period: u32, len: u32) {
        let i = thread::index_1d().get();
        if i < len as usize {
            // SAFETY: `i < len`, and the launcher passes `len <= out.len()` (checked by
            // `debug_assert!` only); `i` is this thread's own.
            unsafe { *out.get_unchecked_mut(i) = source[i % period as usize] };
        }
    }

    /// `out[offset + i] = source[i]` for `len` values.
    #[kernel]
    pub fn copy_into(source: &[f32], mut out: DisjointSlice<f32>, offset: u32, len: u32) {
        let i = thread::index_1d().get();
        if i < len as usize {
            // SAFETY: `offset + i < offset + len <= out.len()`, asserted by the launcher; each
            // thread writes its own index.
            unsafe { *out.get_unchecked_mut(offset as usize + i) = source[i] };
        }
    }

    /// A column block's inducing-point keys and values from the in-projection
    /// `kv[columns * 128, 384]` (queries, keys, values) into `[columns, 8, 128,
    /// 16]`, which [`column_block_body`] reads with one feature and no reserved
    /// streams. One thread per key value.
    #[kernel]
    pub fn column_kv(kv: &[f32], mut keys: DisjointSlice<f32>, mut values: DisjointSlice<f32>, columns: u32) {
        let i = thread::index_1d().get();
        if i >= columns as usize * 128 * 128 {
            return;
        }
        let (d, head, point, column) = (i % 16, (i / 16) % 8, (i / 128) % 128, i / (128 * 128));
        let from = (column * 128 + point) * 384 + head * 16 + d;
        let to = (column * 8 + head) * 128 * 16 + point * 16 + d;
        // SAFETY: `to < columns * 128 * 128 <= keys.len() == values.len()`, asserted by the
        // launcher; `i` maps to `to` one to one, so each thread writes its own slots.
        unsafe {
            *keys.get_unchecked_mut(to) = kv[from + 128];
            *values.get_unchecked_mut(to) = kv[from + 256];
        }
    }

    /// Rows `[B, q, width]` of `x` plus rows `start..start + q` of
    /// `labels[B, total, width]`, in place.
    #[kernel]
    pub fn add_labels(
        mut x: DisjointSlice<f32>,
        labels: &[f32],
        queries: u32,
        total: u32,
        start: u32,
        width: u32,
        len: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let (q, width) = (queries as usize, width as usize);
        let (c, token) = (i % width, i / width);
        let (e, row) = (token / q, token % q);
        let label = labels[(e * total as usize + start as usize + row) * width + c];
        // SAFETY: `i < len <= x.len()`, asserted by the launcher; the thread reads and writes only
        // `x[i]`.
        unsafe { *x.get_unchecked_mut(i) += label };
    }

    /// An ICL block's keys and values for training rows `start..start + q`,
    /// from the in-projection `kv[B * q, 1024]` (keys, then values), into the
    /// FP16 caches `[B * 8, np, 64]`. One thread per key value.
    #[kernel]
    pub fn icl_kv(
        kv: &[f32],
        mut keys: DisjointSlice<u16>,
        mut values: DisjointSlice<u16>,
        ensembles: u32,
        queries: u32,
        start: u32,
        np: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let to = ((b * 8 + head) * np as usize + start as usize + row) * 64 + d;
        // SAFETY: `b < ensembles`, `head < 8`, `start + row < start + q <= np`, `d < 64`, and the
        // launcher asserts both caches hold `[B * 8, np, 64]`; distinct threads write distinct
        // `to`.
        unsafe {
            *keys.get_unchecked_mut(to) = (kv[token * 1024 + head * 64 + d] as f16).to_bits();
            *values.get_unchecked_mut(to) = (kv[token * 1024 + 512 + head * 64 + d] as f16).to_bits();
        }
    }

    /// [`icl_kv`] into FP32 caches, for `optimize="safe"`.
    #[kernel]
    pub fn icl_kv_f32(
        kv: &[f32],
        mut keys: DisjointSlice<f32>,
        mut values: DisjointSlice<f32>,
        ensembles: u32,
        queries: u32,
        start: u32,
        np: u32,
    ) {
        let i = thread::index_1d().get();
        let q = queries as usize;
        if i >= ensembles as usize * q * 512 {
            return;
        }
        let (d, head, token) = (i % 64, (i / 64) % 8, i / 512);
        let (b, row) = (token / q, token % q);
        let to = ((b * 8 + head) * np as usize + start as usize + row) * 64 + d;
        // SAFETY: as in [`icl_kv`]: `to` lies in the launcher-checked `[B * 8, np, 64]` caches and
        // is this thread's own.
        unsafe {
            *keys.get_unchecked_mut(to) = kv[token * 1024 + head * 64 + d];
            *values.get_unchecked_mut(to) = kv[token * 1024 + 512 + head * 64 + d];
        }
    }

    /// The largest magnitude of each (pair, dim) of FP16 `values` `[pairs,
    /// stride, 64]` over a chunk of 256 of their first `n` rows, into
    /// `peaks[(pair * chunks + chunk) * 64 + dim]`. One thread per peak.
    #[kernel]
    pub fn fp8_peaks(
        values: &[u16],
        mut peaks: DisjointSlice<f32>,
        stride: u32,
        n: u32,
        chunks: u32,
        len: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let (dim, rest) = (i % 64, i / 64);
        let (chunk, pair) = (rest % chunks as usize, rest / chunks as usize);
        let start = chunk * 256;
        let end = if start + 256 < n as usize {
            start + 256
        } else {
            n as usize
        };
        let mut peak = 0.0f32;
        for row in start..end {
            let value = f16::from_bits(values[(pair * stride as usize + row) * 64 + dim]) as f32;
            peak = peak.max(value.abs());
        }
        // SAFETY: `i < len = peaks.len()` (the launcher); `i` is this thread's own.
        unsafe { *peaks.get_unchecked_mut(i) = peak };
    }

    /// Each (pair, dim)'s FP8 scale from its chunks' peaks: the scale that
    /// maps its largest magnitude to E4M3's largest, 448, or 1 for zeros.
    /// One thread per scale.
    #[kernel]
    pub fn fp8_scales(peaks: &[f32], mut scales: DisjointSlice<f32>, chunks: u32, len: u32) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let (dim, pair, chunks) = (i % 64, i / 64, chunks as usize);
        let mut peak = 0.0f32;
        for chunk in 0..chunks {
            peak = peak.max(peaks[(pair * chunks + chunk) * 64 + dim]);
        }
        // SAFETY: `i < len = scales.len()` (the launcher); `i` is this thread's own.
        unsafe { *scales.get_unchecked_mut(i) = if peak > 0.0 { peak / 448.0 } else { 1.0 } };
    }

    /// The first `n` rows of FP16 `values` `[pairs, stride, 64]` divided by
    /// their scales `[pairs, 64]` as FP8 E4M3 bytes `[pairs, np, 64]`, two to
    /// a `u16`; rows from `n` on are zero. One thread per pair of bytes.
    #[kernel]
    pub fn fp8_encode(
        values: &[u16],
        scales: &[f32],
        mut packed: DisjointSlice<u16>,
        stride: u32,
        np: u32,
        n: u32,
        len: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= len as usize {
            return;
        }
        let np = np as usize;
        let (dim, row, pair) = ((i % 32) * 2, (i / 32) % np, i / (32 * np));
        let mut bytes = 0u32;
        if row < n as usize {
            let at = (pair * stride as usize + row) * 64 + dim;
            let low = f16::from_bits(values[at]) as f32 / scales[pair * 64 + dim];
            let high = f16::from_bits(values[at + 1]) as f32 / scales[pair * 64 + dim + 1];
            bytes = e4m3(low) | (e4m3(high) << 8);
        }
        #[allow(clippy::cast_possible_truncation, reason = "Two bytes")]
        let bytes = bytes as u16;
        // SAFETY: `i < len = packed.len()` (the launcher); `i` is this thread's own.
        unsafe { *packed.get_unchecked_mut(i) = bytes };
    }

    // ---------------------------------------------------------------------------
    // Dense layers on tensor cores.
    // ---------------------------------------------------------------------------

    // ---------------------------------------------------------------------------
    // Fused 128-wide transformer block tail.
    // ---------------------------------------------------------------------------

    /// [`block_tail_body`] with one FP16 product per layer.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_tail_half(
        attended: &[f32],
        mut state: DisjointSlice<f32>,
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        m: u32,
    ) {
        block_tail_body::<false>(
            attended, &mut state, wo_hi, wo_lo, bo, gamma, beta, w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, m,
        );
    }

    /// [`block_tail_body`] with split, near-FP32 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_tail_split(
        attended: &[f32],
        mut state: DisjointSlice<f32>,
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        m: u32,
    ) {
        block_tail_body::<true>(
            attended, &mut state, wo_hi, wo_lo, bo, gamma, beta, w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, m,
        );
    }

    /// [`block_head_body`] with one FP16 product.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_head_half(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w_hi: &[u32],
        w_lo: &[u32],
        bias: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut out: DisjointSlice<f32>,
        m: u32,
        n: u32,
        len: u32,
    ) {
        block_head_body::<false>(x, gamma, beta, w_hi, w_lo, bias, cos, sin, &mut out, m, n, len);
    }

    /// [`block_head_body`] with split, near-FP32 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_head_split(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w_hi: &[u32],
        w_lo: &[u32],
        bias: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut out: DisjointSlice<f32>,
        m: u32,
        n: u32,
        len: u32,
    ) {
        block_head_body::<true>(x, gamma, beta, w_hi, w_lo, bias, cos, sin, &mut out, m, n, len);
    }

    /// [`attention16_tc_body`] with FP16 products.
    #[kernel]
    pub fn attention16_half(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
        lq: u32,
        lk: u32,
        inner: u32,
        q_seq: u32,
        q_tok: u32,
        k_base: u32,
        v_base: u32,
        kv_outer: u32,
        kv_inner: u32,
        kv_tok: u32,
        kv_head: u32,
        o_seq: u32,
        o_tok: u32,
        scale: f32,
    ) {
        attention16_tc_body::<false>(
            q, k, v, &mut out, lq, lk, inner, q_seq, q_tok, k_base, v_base, kv_outer, kv_inner, kv_tok,
            kv_head, o_seq, o_tok, scale,
        );
    }

    /// [`attention16_tc_body`] with split, near-FP32 products.
    #[kernel]
    pub fn attention16_split(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
        lq: u32,
        lk: u32,
        inner: u32,
        q_seq: u32,
        q_tok: u32,
        k_base: u32,
        v_base: u32,
        kv_outer: u32,
        kv_inner: u32,
        kv_tok: u32,
        kv_head: u32,
        o_seq: u32,
        o_tok: u32,
        scale: f32,
    ) {
        attention16_tc_body::<true>(
            q, k, v, &mut out, lq, lk, inner, q_seq, q_tok, k_base, v_base, kv_outer, kv_inner, kv_tok,
            kv_head, o_seq, o_tok, scale,
        );
    }

    /// [`column_block_body`] with FP16 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn column_block_half(
        mut state: DisjointSlice<f32>,
        gamma1: &[f32],
        beta1: &[f32],
        wq_hi: &[u32],
        wq_lo: &[u32],
        bq: &[f32],
        keys: &[f32],
        values: &[f32],
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma2: &[f32],
        beta2: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        features: u32,
        queries: u32,
        reserved: u32,
    ) {
        column_block_body::<false>(
            &mut state, gamma1, beta1, wq_hi, wq_lo, bq, keys, values, wo_hi, wo_lo, bo, gamma2, beta2,
            w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, features, queries, reserved,
        );
    }

    /// [`column_block_body`] with split, near-FP32 products.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn column_block_split(
        mut state: DisjointSlice<f32>,
        gamma1: &[f32],
        beta1: &[f32],
        wq_hi: &[u32],
        wq_lo: &[u32],
        bq: &[f32],
        keys: &[f32],
        values: &[f32],
        wo_hi: &[u32],
        wo_lo: &[u32],
        bo: &[f32],
        gamma2: &[f32],
        beta2: &[f32],
        w1_hi: &[u32],
        w1_lo: &[u32],
        b1: &[f32],
        w2_hi: &[u32],
        w2_lo: &[u32],
        b2: &[f32],
        features: u32,
        queries: u32,
        reserved: u32,
    ) {
        column_block_body::<true>(
            &mut state, gamma1, beta1, wq_hi, wq_lo, bq, keys, values, wo_hi, wo_lo, bo, gamma2, beta2,
            w1_hi, w1_lo, b1, w2_hi, w2_lo, b2, features, queries, reserved,
        );
    }

    /// [`linear16_body`] with FP16 operands, for `optimize="fast"`.
    #[kernel]
    pub fn linear16(
        x: &[u16],
        x_lo: &[u16],
        w: &[u16],
        w_lo: &[u16],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        mut y_hi: DisjointSlice<u16>,
        mut y_lo: DisjointSlice<u16>,
        m: u32,
        n: u32,
        k: u32,
        ldx: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
    ) {
        static mut AS: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        // SAFETY: `AS` and `BS` are this kernel's own shared statics, reached only through these
        // raw pointers, which [`linear16_body`] keeps in bounds and orders by its barriers.
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        // SAFETY: as above, for `BS`.
        let bt = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BS)) };
        // SAFETY: `AS`, `BS` are distinct `3 * 64 * 16` u32 tiles, 128-byte aligned (the low-part
        // slots alias them, untouched without `SPLIT`). The launcher's grid, block, and buffers
        // match the contract (`ldx = k`, `ldy = ldr = n`, `k` a multiple of 32).
        unsafe {
            linear16_body::<false>(
                (a, a, bt, bt),
                (x, x_lo, w, w_lo),
                (b, r),
                (&mut y, &mut y_hi, &mut y_lo),
                (m, n, k),
                (ldx, ldy, ldr, mode),
            );
        }
    }

    /// [`linear16_body`] with split, near-FP32 operands, for `optimize="auto"`.
    #[kernel]
    pub fn linear16_split(
        x: &[u16],
        x_lo: &[u16],
        w: &[u16],
        w_lo: &[u16],
        b: &[f32],
        r: &[f32],
        mut y: DisjointSlice<f32>,
        mut y_hi: DisjointSlice<u16>,
        mut y_lo: DisjointSlice<u16>,
        m: u32,
        n: u32,
        k: u32,
        ldx: u32,
        ldy: u32,
        ldr: u32,
        mode: u32,
    ) {
        static mut AH: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut AL: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut BH: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        static mut BL: SharedArray<u32, { 3 * 64 * 16 }, 128> = SharedArray::UNINIT;
        // SAFETY: this kernel's own four shared statics, reached only through these raw pointers,
        // which [`linear16_body`] keeps in bounds and orders by its barriers.
        let tiles = unsafe {
            (
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AH)),
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AL)),
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BH)),
                SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(BL)),
            )
        };
        // SAFETY: four distinct `3 * 64 * 16` u32 tiles, 128-byte aligned. The launcher's grid,
        // block, and buffers match the contract (`ldx = k`, `ldy = ldr = n`, `k` a multiple of 32,
        // split `next` only with a split input).
        unsafe {
            linear16_body::<true>(
                tiles,
                (x, x_lo, w, w_lo),
                (b, r),
                (&mut y, &mut y_hi, &mut y_lo),
                (m, n, k),
                (ldx, ldy, ldr, mode),
            );
        }
    }

    // ---------------------------------------------------------------------------
    // FP32 on the CUDA cores, for `optimize="safe"`: the counterparts of the
    // tensor-core block, attention, and ICL kernels, every product, sum, and
    // softmax in FP32. Weights are FP32 `[n, k]` in row order.
    // ---------------------------------------------------------------------------

    /// [`block_head_body`] in FP32: `out = norm1(x) Wᵀ + b` for `n` (128 or
    /// 384) outputs, then `RoPE` on the first 256 by each row's position
    /// `row % len` when `len` is not zero. Thread `(warp, lane)` computes rows
    /// `4 warp..` by columns `4 lane..` of each 128-column pass; a head's dims
    /// `d` and `d + 8` sit two lanes apart.
    /// Grid: ceil(m / 32); block: 256 threads.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_head_f32(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w: &[f32],
        bias: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut out: DisjointSlice<f32>,
        m: u32,
        n: u32,
        len: u32,
    ) {
        static mut AS: SharedArray<f32, { 32 * 128 }, 16> = SharedArray::UNINIT;
        // SAFETY: `AS` is this kernel's own shared static, reached only through this pointer:
        // `load_normed_f32` writes its rows below 32, and after the barrier `product4x4` only reads
        // them.
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        let tid = thread::threadIdx_x() as usize;
        let (r0, lane) = ((tid / 32) * 4, tid % 32);
        let (m, n, len) = (m as usize, n as usize, len as usize);
        let row0 = thread::blockIdx_x() as usize * 32;
        // SAFETY: the launcher runs 256 threads and passes `x` `[m, 128]` (16-byte aligned rows);
        // `AS` is `[32, 128]`, 16-byte aligned, and the barrier below ends the stores.
        unsafe { load_normed_f32(x.as_ptr(), gamma, beta, a, row0, m) };
        thread::sync_threads();
        let mut pass = 0;
        while pass < n / 128 {
            let c0 = pass * 128 + lane * 4;
            let mut acc = [[0.0f32; 4]; 4];
            // SAFETY: `AS` `[32, 128]`, published by the barrier above and not written after, at
            // rows `r0 + 4 <= 32`; `w` the in-projection `[n, 128]` at rows `c0 + 4 <= n`.
            unsafe { product4x4((a.cast_const(), 128, r0), (w.as_ptr(), 128, c0), 128, &mut acc) };
            #[unroll]
            for i in 0..4 {
                #[unroll]
                for j in 0..4 {
                    acc[i][j] += bias[c0 + j];
                }
            }
            if len != 0 && pass < 2 {
                // With the tables holding [-sin, sin], both halves of a head
                // take `x cos + partner sin`.
                #[unroll]
                for i in 0..4 {
                    let t = (row0 + r0 + i) % len;
                    #[unroll]
                    for j in 0..4 {
                        let partner = warp::shuffle_xor_f32(acc[i][j], 2);
                        let d = (c0 + j) % 16;
                        acc[i][j] = acc[i][j] * cos[t * 16 + d] + partner * sin[t * 16 + d];
                    }
                }
            }
            #[unroll]
            for i in 0..4 {
                let row = row0 + r0 + i;
                if row < m {
                    #[unroll]
                    for j in 0..4 {
                        // SAFETY: `row < m` and `c0 + j < (pass + 1) * 128 <= n`: inside `out` `[m,
                        // n]`; each (row, column) belongs to one thread of one block.
                        unsafe { *out.get_unchecked_mut(row * n + c0 + j) = acc[i][j] };
                    }
                }
            }
            pass += 1;
        }
    }

    /// [`block_tail_body`] in FP32 (see [`block_tail_core_f32`]).
    /// Grid: ceil(m / 32); block: 256 threads.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn block_tail_f32(
        attended: &[f32],
        mut state: DisjointSlice<f32>,
        wo: &[f32],
        bo: &[f32],
        gamma: &[f32],
        beta: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        m: u32,
    ) {
        static mut AS: SharedArray<f32, { 32 * 128 }, 16> = SharedArray::UNINIT;
        static mut HS: SharedArray<f32, { 32 * 256 }, 16> = SharedArray::UNINIT;
        // SAFETY: `AS` and `HS` are this kernel's own shared statics, reached only through these
        // pointers, here and in [`block_tail_core_f32`], in bounds and between barriers.
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        // SAFETY: as above, for `HS`.
        let h = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HS)) };
        let tid = thread::threadIdx_x() as usize;
        let m = m as usize;
        let row0 = thread::blockIdx_x() as usize * 32;
        let ap = attended.as_ptr();
        #[unroll]
        for i in 0..4 {
            let index = tid + i * 256;
            let (row, col) = (index / 32, (index % 32) * 4);
            let v = if row0 + row < m {
                // SAFETY: `row0 + row < m` and `col + 4 <= 128`: 16 aligned bytes inside `attended`
                // `[m, 128]`.
                unsafe { *ap.add((row0 + row) * 128 + col).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            // SAFETY: `row < 32` (`index < 1024`) and `col + 4 <= 128`: inside `AS` `[32, 128]`,
            // 16-byte aligned; each thread writes its own four values, read after the barrier.
            unsafe { *a.add(row * 128 + col).cast::<F32x4>() = v };
        }
        thread::sync_threads();
        // SAFETY: all 256 threads of block `row0 / 32` call it; `AS` holds the tile, published by
        // the barrier above, and `HS` is untouched. The launcher passes `state` `[m, 128]` and the
        // 128- and 256-wide FP32 layers.
        unsafe {
            block_tail_core_f32(
                a,
                h,
                &mut state,
                (row0, m),
                (wo, bo),
                (gamma, beta),
                (w1, b1),
                (w2, b2),
            );
        }
    }

    /// [`column_block_body`] in FP32: `norm1`, the query projection (the first
    /// 128 rows of the in-projection), attention of each 16-wide head to the
    /// column's 128 cached inducing points, then [`block_tail_core_f32`], in
    /// place on `state` `[B * H, q, 128]`. Thread `t` attends for row `t / 8`
    /// and head `t % 8`.
    /// Grid: (ceil(q / 32), B * H); block: 256 threads.
    #[kernel]
    #[launch_bounds(256, 2)]
    pub fn column_block_f32(
        mut state: DisjointSlice<f32>,
        gamma1: &[f32],
        beta1: &[f32],
        wq: &[f32],
        bq: &[f32],
        keys: &[f32],
        values: &[f32],
        wo: &[f32],
        bo: &[f32],
        gamma2: &[f32],
        beta2: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        features: u32,
        queries: u32,
        reserved: u32,
    ) {
        static mut AS: SharedArray<f32, { 32 * 128 }, 16> = SharedArray::UNINIT;
        static mut HS: SharedArray<f32, { 32 * 256 }, 16> = SharedArray::UNINIT;
        // SAFETY: `AS` and `HS` are this kernel's own shared statics, reached only through these
        // pointers, here and in [`block_tail_core_f32`], in bounds and between barriers.
        let a = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(AS)) };
        // SAFETY: as above, for `HS`.
        let h = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(HS)) };
        let tid = thread::threadIdx_x() as usize;
        let (r0, c0) = ((tid / 32) * 4, (tid % 32) * 4);
        let (features, q) = (features as usize, queries as usize);
        let column = thread::blockIdx_y() as usize;
        let tile = thread::blockIdx_x() as usize * 32;
        let row0 = column * q + tile;
        let m = row0 + if q - tile < 32 { q - tile } else { 32 };

        // norm1 into A, then Q = A Wqᵀ + bq into the first 128 columns of H.
        // SAFETY: 256 threads call it; rows `[row0, m)` with `m <= (column + 1) * q` lie in the
        // launcher's `state` `[B * H * q, 128]`, which this block writes only later, after
        // barriers. `AS` is `[32, 128]`, 16-byte aligned, and the barrier below ends the stores.
        unsafe { load_normed_f32(state.as_mut_ptr(), gamma1, beta1, a, row0, m) };
        thread::sync_threads();
        let mut acc = [[0.0f32; 4]; 4];
        // SAFETY: `AS` `[32, 128]`, published by the barrier above, at rows `r0 + 4 <= 32`; `wq`
        // the in-projection (128 or more rows of 128) at rows `c0 + 4 <= 128`.
        unsafe { product4x4((a.cast_const(), 128, r0), (wq.as_ptr(), 128, c0), 128, &mut acc) };
        #[unroll]
        for i in 0..4 {
            let y = [
                acc[i][0] + bq[c0],
                acc[i][1] + bq[c0 + 1],
                acc[i][2] + bq[c0 + 2],
                acc[i][3] + bq[c0 + 3],
            ];
            // SAFETY: `r0 + i < 32` and `c0 + 4 <= 128`: inside `HS` `[32, 256]`, 16-byte aligned;
            // each thread writes its own values, read only after the barrier.
            unsafe { *h.add((r0 + i) * 256 + c0).cast::<F32x4>() = F32x4(y) };
        }
        thread::sync_threads();

        // Attention to the column's cache stream `f + reserved` of estimator `e`.
        let (row, head) = (tid / 8, tid % 8);
        let (e, f) = (column / features, column % features);
        let kv0 =
            (e * (features + reserved as usize) + f + reserved as usize) * (8 * 128 * 16) + head * 128 * 16;
        let mut query = [0.0f32; 16];
        #[unroll]
        for c in 0..4 {
            // SAFETY: `row < 32` and `head * 16 + c * 4 + 4 <= 128`: inside `HS`, 16-byte aligned,
            // written before the last barrier.
            let v = unsafe { *h.add(row * 256 + head * 16 + c * 4).cast::<F32x4>() }.0;
            #[unroll]
            for t in 0..4 {
                query[c * 4 + t] = v[t];
            }
        }
        let mut out = [0.0f32; 16];
        // SAFETY: `kv0`, a multiple of 2048, starts head `head` of stream `e * (H + reserved) + f +
        // reserved` of `B * (H + reserved)`; its 128 keys of stride 16 lie in the caches, which
        // the launcher asserts both hold `B * (H + reserved) * 16384`.
        unsafe {
            attend16_f32(
                &query,
                (
                    keys.as_ptr().wrapping_add(kv0),
                    values.as_ptr().wrapping_add(kv0),
                    16,
                ),
                (0, 128),
                &mut out,
            );
        }
        // The attention output replaces the normalized rows in A.
        #[unroll]
        for c in 0..4 {
            let y = [out[c * 4], out[c * 4 + 1], out[c * 4 + 2], out[c * 4 + 3]];
            // SAFETY: as for the read above, inside `AS` `[32, 128]`; `product4x4` finished reading
            // A before the last barrier, and each (row, head) slot is one thread's.
            unsafe { *a.add(row * 128 + head * 16 + c * 4).cast::<F32x4>() = F32x4(y) };
        }
        thread::sync_threads();
        // SAFETY: all 256 threads call it with `AS` published by the barrier above and `HS` last
        // read before it. Block (tile, column) alone owns rows `[row0, m)` of `state` `[B * H * q,
        // 128]`; the launcher passes the 128- and 256-wide FP32 layers.
        unsafe {
            block_tail_core_f32(
                a,
                h,
                &mut state,
                (row0, m),
                (wo, bo),
                (gamma2, beta2),
                (w1, b1),
                (w2, b2),
            );
        }
    }

    /// [`attention16_tc_body`] in FP32, with its layout: one thread per query
    /// of a (sequence, head), each query against every key in turn.
    /// Grid: (ceil(lq / 64), 8, sequences); block: 64 threads.
    #[kernel]
    pub fn attention16_f32(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
        lq: u32,
        lk: u32,
        inner: u32,
        q_seq: u32,
        q_tok: u32,
        k_base: u32,
        v_base: u32,
        kv_outer: u32,
        kv_inner: u32,
        kv_tok: u32,
        kv_head: u32,
        o_seq: u32,
        o_tok: u32,
    ) {
        let query_row = thread::blockIdx_x() as usize * 64 + thread::threadIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let seq = thread::blockIdx_z() as usize;
        let inner = inner as usize;
        if query_row >= lq as usize {
            return;
        }
        let kv0 =
            (seq / inner) * kv_outer as usize + (seq % inner) * kv_inner as usize + head * kv_head as usize;
        let mut query = [0.0f32; 16];
        let at = seq * q_seq as usize + query_row * q_tok as usize + head * 16;
        #[unroll]
        for d in 0..16 {
            query[d] = q[at + d];
        }
        let (kp, vp) = (
            k.as_ptr().wrapping_add(k_base as usize + kv0),
            v.as_ptr().wrapping_add(v_base as usize + kv0),
        );
        let mut result = [0.0f32; 16];
        // SAFETY: the launcher's layouts keep `k_base + kv0 + key * kv_tok + 16 <= k.len()` for
        // `key < lk` (likewise `v`), with offsets and strides multiples of 4 for 16-byte alignment.
        unsafe { attend16_f32(&query, (kp, vp, kv_tok as usize), (0, lk as usize), &mut result) };
        let at = seq * o_seq as usize + query_row * o_tok as usize + head * 16;
        #[unroll]
        for d in 0..16 {
            // SAFETY: `at + d` lies in `out` `[seqs, lq, 128]`: the launchers' layouts have `o_tok
            // = 128` and `o_seq = lq * 128`, so each (sequence, query, head, dim) has its own slot.
            unsafe { *out.get_unchecked_mut(at + d) = result[d] };
        }
    }

    /// Partial ICL attention in FP32 on the CUDA cores over FP32 keys and
    /// values (bytes of `[pairs, np, 64]`), with [`fast_body`]'s split of the
    /// keys into chunks and its outputs: 32 queries per block, keys in 32-key
    /// steps through shared memory. Thread `(y, x)` = `(t / 16, t % 16)`
    /// scores rows `y + 8i` against keys `x + 16j` and accumulates dims
    /// `4x..4x + 4` of the same rows.
    /// Grid: (ceil(qp / 32), pairs * splits); block: 128 threads.
    #[kernel]
    #[launch_bounds(128)]
    pub fn icl_attention_f32(
        q: &[f32],
        k: &[u8],
        v: &[u8],
        mut part: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        qp: u32,
        np: u32,
        n_valid: u32,
        chunk_tiles: u32,
        splits: u32,
        scale_log2: f32,
    ) {
        // Q and K rows padded to 68 values, so that rows four apart and keys
        // one apart start in different banks.
        static mut QS: SharedArray<f32, { 32 * 68 }, 16> = SharedArray::UNINIT;
        static mut KS: SharedArray<f32, { 32 * 68 }, 16> = SharedArray::UNINIT;
        static mut VS: SharedArray<f32, { 32 * 64 }, 16> = SharedArray::UNINIT;
        static mut PS: SharedArray<f32, { 32 * 33 }, 16> = SharedArray::UNINIT;
        // SAFETY: `QS`, `KS`, `VS`, and `PS` are this kernel's own shared statics, reached only
        // through these pointers; indices stay in their sizes, and the key loop's barriers order
        // the accesses.
        let qs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(QS)) };
        // SAFETY: as above.
        let ks = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(KS)) };
        // SAFETY: as above.
        let vs = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(VS)) };
        // SAFETY: as above.
        let ps = unsafe { SharedArray::as_raw_mut_ptr(core::ptr::addr_of_mut!(PS)) };
        let tid = thread::threadIdx_x() as usize;
        let (ty, tx) = (tid / 16, tid % 16);
        let block = thread::blockIdx_y() as usize;
        let (bh, chunk) = (block / splits as usize, block % splits as usize);
        let q0 = thread::blockIdx_x() as usize * 32;
        let (qp, np) = (qp as usize, np as usize);
        let first_tile = chunk * chunk_tiles as usize;
        let (kp, vp) = (k.as_ptr().cast::<f32>(), v.as_ptr().cast::<f32>());

        // The block's 32 query rows; rows from qp on are zeros.
        #[unroll]
        for i in 0..4 {
            let index = tid + i * 128;
            let (row, c) = (index / 16, (index % 16) * 4);
            let value = if q0 + row < qp {
                // SAFETY: `q0 + row < qp`, `c + 4 <= 64`, and `bh < pairs` by the grid: 16 aligned
                // bytes inside `q` `[pairs, qp, 64]`.
                unsafe { *q.as_ptr().add((bh * qp + q0 + row) * 64 + c).cast::<F32x4>() }
            } else {
                F32x4::splat(0.0)
            };
            // SAFETY: `row < 32` (`index < 512`) and `c + 4 <= 64`, so `row * 68 + c + 4 <= 32 *
            // 68`; 16-byte aligned since a row is 272 bytes. Read only after the key loop's first
            // barrier.
            unsafe { *qs.add(row * 68 + c).cast::<F32x4>() = value };
        }

        let mut acc = [[0.0f32; 4]; 4];
        let mut max = [f32::NEG_INFINITY; 4];
        let mut total = [0.0f32; 4];
        // The cache holds np / 64 tiles, so the last chunk can be shorter.
        let tiles = (chunk_tiles as usize).min(np / 64 - first_tile);
        for step in 0..2 * tiles {
            let key0 = first_tile * 64 + step * 32;
            // The previous step is done reading K, V, and P.
            thread::sync_threads();
            #[unroll]
            for i in 0..4 {
                let index = tid + i * 128;
                let (key, c) = (index / 16, (index % 16) * 4);
                let at = (bh * np + key0 + key) * 64 + c;
                // SAFETY: `first_tile < np / 64` (the launcher's chunks), so `key0 + key <
                // (first_tile + tiles) * 64 <= np` and, with `bh < pairs`, 16 aligned bytes lie in
                // the FP32 `[pairs, np, 64]` caches; the shared indices stay in `KS` and `VS`,
                // whose readers the barrier above retired.
                unsafe {
                    *ks.add(key * 68 + c).cast::<F32x4>() = *kp.add(at).cast::<F32x4>();
                    *vs.add(key * 64 + c).cast::<F32x4>() = *vp.add(at).cast::<F32x4>();
                }
            }
            thread::sync_threads();

            // Scores, masked past N, in the exp2 domain.
            let mut s = [[0.0f32; 2]; 4];
            let mut c = 0;
            while c < 64 {
                let mut qv = [[0.0f32; 4]; 4];
                #[unroll]
                for i in 0..4 {
                    // SAFETY: `ty + 8 * i < 32` and `c + 4 <= 64`: inside `QS`, 16-byte aligned,
                    // written before a barrier.
                    qv[i] = unsafe { *qs.add((ty + 8 * i) * 68 + c).cast::<F32x4>() }.0;
                }
                #[unroll]
                for j in 0..2 {
                    // SAFETY: `tx + 16 * j < 32` and `c + 4 <= 64`: inside `KS`, 16-byte aligned,
                    // written before the barrier above.
                    let kv = unsafe { *ks.add((tx + 16 * j) * 68 + c).cast::<F32x4>() }.0;
                    #[unroll]
                    for i in 0..4 {
                        #[unroll]
                        for e in 0..4 {
                            s[i][j] += qv[i][e] * kv[e];
                        }
                    }
                }
                c += 4;
            }
            #[unroll]
            for i in 0..4 {
                #[unroll]
                for j in 0..2 {
                    let valid = key0 + tx + 16 * j < n_valid as usize;
                    s[i][j] = if valid {
                        s[i][j] * scale_log2
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                // Online softmax over the 16 lanes sharing the row.
                let new = max[i].max(max16(s[i][0].max(s[i][1])));
                // A row with no valid key yet keeps zero weight instead of NaN.
                let base = if new == f32::NEG_INFINITY { 0.0 } else { new };
                let alpha = (max[i] - base).exp2();
                let (p0, p1) = ((s[i][0] - base).exp2(), (s[i][1] - base).exp2());
                total[i] = total[i] * alpha + sum16(p0 + p1);
                max[i] = new;
                #[unroll]
                for e in 0..4 {
                    acc[i][e] *= alpha;
                }
                // SAFETY: row `ty + 8 * i < 32` and column `tx + 16 < 32`: inside `PS` `[32, 33]`;
                // each slot is this thread's, and the previous step's readers passed the loop's
                // first barrier.
                unsafe {
                    *ps.add((ty + 8 * i) * 33 + tx) = p0;
                    *ps.add((ty + 8 * i) * 33 + tx + 16) = p1;
                }
            }
            thread::sync_threads();

            // O += P V.
            for key in 0..32 {
                // SAFETY: `key < 32` and `tx * 4 + 4 <= 64`: inside `VS`, 16-byte aligned, written
                // before a barrier.
                let value = unsafe { *vs.add(key * 64 + tx * 4).cast::<F32x4>() }.0;
                #[unroll]
                for i in 0..4 {
                    // SAFETY: `ty + 8 * i < 32` and `key < 32`: inside `PS`, written before the
                    // barrier above.
                    let p = unsafe { *ps.add((ty + 8 * i) * 33 + key) };
                    #[unroll]
                    for e in 0..4 {
                        acc[i][e] += p * value[e];
                    }
                }
            }
        }

        let out_row = (bh * splits as usize + chunk) * qp + q0;
        #[unroll]
        for i in 0..4 {
            let row = ty + 8 * i;
            if q0 + row < qp {
                #[unroll]
                for e in 0..4 {
                    // SAFETY: `q0 + row < qp`, so `out_row + row` is a row of `part` `[pairs *
                    // splits, qp, 64]` (the workspace reserves chunks times padded rows); each
                    // (row, dim) belongs to this thread.
                    unsafe {
                        *part.get_unchecked_mut((out_row + row) * 64 + tx * 4 + e) = acc[i][e] / total[i];
                    }
                }
                if tx == 0 {
                    // SAFETY: as above, inside `lse` `[pairs * splits, qp]`; only the `tx == 0`
                    // thread writes the row.
                    unsafe { *lse.get_unchecked_mut(out_row + row) = max[i] + total[i].log2() };
                }
            }
        }
    }
}
