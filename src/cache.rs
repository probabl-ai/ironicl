use std::fmt;

use crate::{
    config::{CacheMode, ExecutionConfig},
    error::{Error, Result},
};

/// Byte breakdown for the fixed training-context cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheEstimate {
    pub column_bytes: u64,
    pub context_bytes: u64,
}

impl CacheEstimate {
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.column_bytes + self.context_bytes
    }
}

/// Persistent memory required for weights and one static context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    pub weight_bytes: u64,
    pub cache: CacheEstimate,
}

impl MemoryPlan {
    #[must_use]
    pub const fn persistent_bytes(self) -> u64 {
        self.weight_bytes + self.cache.total_bytes()
    }

    /// Ideal lower bound for one full cache scan. This excludes all compute.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cache_scan_seconds(self, bandwidth_gbps: f64) -> Option<f64> {
        if bandwidth_gbps.is_finite() && bandwidth_gbps > 0.0 {
            Some(self.cache.total_bytes() as f64 / (bandwidth_gbps * 1_000_000_000.0))
        } else {
            None
        }
    }
}

/// Estimate payload memory. Quantization scales, alignment, and scratch space
/// are intentionally excluded and must be added by the artifact planner.
///
/// # Errors
///
/// Returns an error when the configuration is invalid or a size calculation
/// overflows `u64`.
pub fn estimate_memory(config: ExecutionConfig) -> Result<MemoryPlan> {
    config.validate()?;

    let model = config.model;
    let workload = config.workload;
    let feature_streams = checked_add(workload.features, model.row_cls_tokens, "feature streams")?;

    // Each column block caches K and V over all inducing points for every
    // feature stream and ensemble member.
    let column_elements = checked_product(
        &[
            2,
            workload.estimators,
            model.column_blocks,
            feature_streams,
            model.column_inducing_points,
            model.embed_dim,
        ],
        "column cache",
    )?;

    let context_elements = match config.cache_mode {
        CacheMode::Kv => checked_product(
            &[
                2,
                workload.estimators,
                model.icl_blocks,
                workload.training_rows,
                model.icl_dim(),
            ],
            "ICL KV cache",
        )?,
        CacheMode::Representation => checked_product(
            &[workload.estimators, workload.training_rows, model.icl_dim()],
            "row representation cache",
        )?,
    };

    let kv_bytes = config.kv_dtype.bytes_per_element();
    let column_bytes = checked_product(&[column_elements, kv_bytes], "column cache bytes")?;
    let context_bytes = checked_product(&[context_elements, kv_bytes], "context cache bytes")?;
    let weight_bytes = checked_product(
        &[model.parameter_count(), config.weight_dtype.bytes_per_element()],
        "weight bytes",
    )?;

    Ok(MemoryPlan {
        weight_bytes,
        cache: CacheEstimate {
            column_bytes,
            context_bytes,
        },
    })
}

fn checked_add(left: u64, right: u64, name: &'static str) -> Result<u64> {
    left.checked_add(right).ok_or(Error::ArithmeticOverflow(name))
}

fn checked_product(values: &[u64], name: &'static str) -> Result<u64> {
    values.iter().try_fold(1_u64, |product, value| {
        product.checked_mul(*value).ok_or(Error::ArithmeticOverflow(name))
    })
}

/// Human-readable binary byte count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSize(pub u64);

impl fmt::Display for ByteSize {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const KIB: f64 = 1024.0;
        const MIB: f64 = KIB * 1024.0;
        const GIB: f64 = MIB * 1024.0;
        let bytes = self.0 as f64;
        if bytes >= GIB {
            write!(f, "{:.2} GiB", bytes / GIB)
        } else if bytes >= MIB {
            write!(f, "{:.2} MiB", bytes / MIB)
        } else if bytes >= KIB {
            write!(f, "{:.2} KiB", bytes / KIB)
        } else {
            write!(f, "{} B", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{StorageDType, Workload};

    #[test]
    fn full_bf16_cache_matches_reference_formula() {
        let config = ExecutionConfig {
            workload: Workload {
                training_rows: 50_000,
                features: 100,
                estimators: 8,
                query_microbatch: 1,
            },
            weight_dtype: StorageDType::Bf16,
            kv_dtype: StorageDType::Bf16,
            ..ExecutionConfig::default()
        };
        let plan = estimate_memory(config).expect("valid plan");

        assert_eq!(plan.weight_bytes, 55_104_516);
        assert_eq!(plan.cache.column_bytes, 163_577_856);
        assert_eq!(plan.cache.context_bytes, 9_830_400_000);
        assert_eq!(plan.cache.total_bytes(), 9_993_977_856);
    }

    #[test]
    fn representation_cache_is_twenty_four_times_smaller_per_context_row() {
        let kv = ExecutionConfig {
            workload: Workload {
                training_rows: 1,
                features: 1,
                estimators: 1,
                query_microbatch: 1,
            },
            cache_mode: CacheMode::Kv,
            ..ExecutionConfig::default()
        };
        let repr = ExecutionConfig {
            cache_mode: CacheMode::Representation,
            ..kv
        };
        let kv_plan = estimate_memory(kv).expect("valid KV plan");
        let repr_plan = estimate_memory(repr).expect("valid representation plan");

        assert_eq!(kv_plan.cache.context_bytes, repr_plan.cache.context_bytes * 24);
    }
}
