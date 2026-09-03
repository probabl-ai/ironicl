//! Summary statistics for resident-model inference benchmarks.

use std::time::Duration;

/// Sorted latency statistics in milliseconds.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkStats {
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
}

impl BenchmarkStats {
    /// Calculate latency statistics from one or more inference durations.
    ///
    /// Percentiles use nearest-rank: rank `ceil(percentile * N)`, with ranks
    /// starting at one.
    ///
    /// # Panics
    ///
    /// Panics when `durations` is empty.
    #[must_use]
    pub fn from_durations(durations: &[Duration]) -> Self {
        assert!(!durations.is_empty(), "benchmark requires at least one duration");
        let mut milliseconds = durations
            .iter()
            .map(|duration| duration.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        milliseconds.sort_by(f64::total_cmp);
        let count = milliseconds.len();
        Self {
            min_ms: milliseconds[0],
            p50_ms: nearest_rank(&milliseconds, 50),
            p95_ms: nearest_rank(&milliseconds, 95),
            p99_ms: nearest_rank(&milliseconds, 99),
            max_ms: milliseconds[count - 1],
            mean_ms: milliseconds.iter().sum::<f64>()
                / f64::from(u32::try_from(count).expect("benchmark count fits u32")),
        }
    }
}

fn nearest_rank(sorted: &[f64], percentile: usize) -> f64 {
    let rank = (sorted.len() * percentile).div_ceil(100);
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_nearest_rank_statistics() {
        let durations = [1, 2, 3, 4, 5].map(Duration::from_millis);
        assert_eq!(
            BenchmarkStats::from_durations(&durations),
            BenchmarkStats {
                min_ms: 1.0,
                p50_ms: 3.0,
                p95_ms: 5.0,
                p99_ms: 5.0,
                max_ms: 5.0,
                mean_ms: 3.0,
            }
        );
    }
}
