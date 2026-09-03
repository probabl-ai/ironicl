//! Immutable worker partitioning for static-context execution.

/// Worker ownership for one static-context prediction.
///
/// The plan is compiled before request execution. It intentionally contains no
/// mutable scheduler state so future throughput schedulers can reuse the same
/// admission and partitioning decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExecutionPlan {
    pub(crate) ensemble_workers: usize,
    pub(crate) inner_workers: usize,
    pub(crate) flat_pool_ensembles: bool,
}

impl ExecutionPlan {
    /// Compile the automatic partitioning policy without environment overrides.
    #[must_use]
    pub(crate) fn automatic(ensembles: usize, total_threads: usize, force_serial: bool) -> Self {
        let total_threads = total_threads.max(1);
        if !force_serial && total_threads >= 16 && (4..=total_threads).contains(&ensembles) {
            let inner_workers = total_threads / ensembles.min(total_threads / 4);
            return Self {
                ensemble_workers: ensembles,
                inner_workers,
                flat_pool_ensembles: true,
            };
        }
        if !force_serial && total_threads > 1 && ensembles >= total_threads {
            return Self {
                ensemble_workers: total_threads,
                inner_workers: 1,
                flat_pool_ensembles: false,
            };
        }
        Self {
            ensemble_workers: 1,
            inner_workers: total_threads,
            flat_pool_ensembles: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExecutionPlan;

    #[test]
    fn automatic_plan_parallelizes_full_estimator_waves() {
        assert_eq!(
            ExecutionPlan::automatic(8, 8, false),
            ExecutionPlan {
                ensemble_workers: 8,
                inner_workers: 1,
                flat_pool_ensembles: false,
            }
        );
        assert_eq!(
            ExecutionPlan::automatic(32, 8, false),
            ExecutionPlan {
                ensemble_workers: 8,
                inner_workers: 1,
                flat_pool_ensembles: false,
            }
        );
        for plan in [
            ExecutionPlan::automatic(1, 8, false),
            ExecutionPlan::automatic(8, 8, true),
        ] {
            assert_eq!(
                plan,
                ExecutionPlan {
                    ensemble_workers: 1,
                    inner_workers: 8,
                    flat_pool_ensembles: false,
                }
            );
        }
        assert_eq!(
            ExecutionPlan::automatic(8, 16, false),
            ExecutionPlan {
                ensemble_workers: 8,
                inner_workers: 4,
                flat_pool_ensembles: true,
            }
        );
        assert_eq!(
            ExecutionPlan::automatic(32, 16, false),
            ExecutionPlan {
                ensemble_workers: 16,
                inner_workers: 1,
                flat_pool_ensembles: false,
            }
        );
    }
}
