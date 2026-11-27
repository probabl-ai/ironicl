#![cfg(feature = "cuda")]

use ironicl::{
    CheckpointKind, Executor, Optimize, QueryRequest,
    format::{container::DType, static_context::ContextHeader},
    load_checkpoint_model,
    memory::MemoryLimits,
};
use std::path::PathBuf;

#[test]
#[ignore = "needs official TabICLv2 checkpoints and a CUDA GPU"]
#[expect(clippy::cast_precision_loss, reason = "small deterministic test indices")]
fn tabicl_native_cache_survives_reload() {
    let fixtures = PathBuf::from(std::env::var_os("IRONICL_CHECKPOINT_FIXTURES").unwrap());
    let (b, n, h, q) = (2, 65, 4, 3);
    let train: Vec<_> = (0..b * n * h).map(|i| (i % 53) as f32 / 27.0 - 1.0).collect();
    let labels: Vec<_> = (0..b * n).map(|i| (i % 3) as f32).collect();
    let mut query: Vec<_> = (0..b * q * h).map(|i| (i % 19) as f32 / 11.0).collect();
    for e in 0..b {
        query[(e * q + 1) * h..][..h].fill(-100.0);
    }
    let request = || QueryRequest::new(b, q, h, &query).unwrap();
    for (task, classes) in [("classifier", 3), ("regressor", 0)] {
        let checkpoint = fixtures.join(format!("tabicl-{task}-v2-20260212.ckpt"));
        let model = || Executor::new(load_checkpoint_model(&checkpoint, CheckpointKind::TabIclV2).unwrap());
        for (optimize, dtype, width) in [
            (Optimize::Safe, DType::F32, 4),
            (Optimize::Auto, DType::F16, 2),
            (Optimize::Fast, DType::F8E4M3, 1),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(ironicl::models::directory::CONTEXT);
            let expected = {
                let fitted = model()
                    .build_context(
                        b,
                        n,
                        h,
                        &train,
                        &labels,
                        (0..b).flat_map(|_| 0..classes).collect(),
                        (0..b).flat_map(|_| 0..h as u64).collect(),
                        optimize,
                        Some(2),
                        MemoryLimits::default(),
                        None,
                    )
                    .unwrap();
                assert_eq!(fitted.device(), "cuda");
                let expected = fitted.predict_logits(request()).unwrap();
                fitted.save_context(&path).unwrap();
                let mut bytes = Vec::new();
                fitted.encode_context(&mut bytes).unwrap();
                assert_eq!(bytes.len(), fitted.context_encoded_len().unwrap());
                expected
            };
            let header = ContextHeader::open(&path).unwrap();
            let tensor = header.tensors.iter().find(|t| t.name == "icl.11.value").unwrap();
            assert_eq!(tensor.dtype, dtype);
            assert_eq!(tensor.bytes, b * 8 * n * 64 * width);
            assert_eq!(
                header
                    .tensors
                    .iter()
                    .filter(|t| t.name.ends_with("_scale"))
                    .count(),
                if optimize == Optimize::Fast { 24 } else { 0 }
            );
            let fresh = model();
            let restored = fresh
                .load_context(&path, optimize, Some(2), MemoryLimits::default())
                .unwrap();
            assert_eq!(restored.device(), "cuda");
            assert_eq!(
                restored.predict_logits(request()).unwrap().values(),
                expected.values()
            );
            if optimize == Optimize::Fast {
                assert!(
                    fresh
                        .load_context(&path, Optimize::Safe, Some(2), MemoryLimits::default())
                        .is_err()
                );
            }
        }
    }
}
