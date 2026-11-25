//! The worker: one thread that owns the model and predicts the queued queries
//! together, in batches of up to `--max-batch-rows` rows, and the queue that
//! bounds how long a query waits for its batch.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use tokio::sync::oneshot;

use super::{
    error::ApiError,
    request::Rows,
    response::{classifier_response, regressor_response},
};
use crate::{Labels, RegressionOutput, models::directory::Fitted};

/// A checked request: its rows and the outputs it asks for.
pub(super) struct Query {
    pub rows: Rows,
    pub outputs: Outputs,
}

pub(super) enum Outputs {
    Classifier {
        labels: bool,
        probabilities: bool,
    },
    /// Each output once, and the quantile levels as the request gave them.
    Regressor {
        outputs: Vec<RegressionOutput>,
        alphas: Vec<f32>,
    },
}

/// A query waiting for its batch, and where its answer goes.
pub(super) struct Job {
    pub query: Query,
    pub enqueued_at: Instant,
    pub reply: oneshot::Sender<Reply>,
}

/// A response body, its rows, and how long its request waited and its batch ran.
pub(super) struct Answer {
    pub body: Vec<u8>,
    pub rows: usize,
    pub queue: Duration,
    pub predict: Duration,
}

pub(super) type Reply = Result<Answer, ApiError>;

/// The rows waiting for a batch, and what one costs to predict.
pub(super) struct Queue {
    rows: AtomicUsize,
    /// Nanoseconds a row takes in a full batch, as the worker measures it.
    row_cost: AtomicU64,
    max_wait: Duration,
}

impl Queue {
    pub(super) fn new(max_wait: Duration) -> Self {
        Self {
            rows: AtomicUsize::new(0),
            row_cost: AtomicU64::new(0),
            max_wait,
        }
    }

    /// Whether `rows` more can wait: the queued rows and theirs would take at
    /// most `max_wait` to predict. An empty queue takes any query, so one
    /// larger than the queue still runs.
    fn fits(&self, queued: usize, rows: usize) -> bool {
        let wait = (queued + rows) as u128 * u128::from(self.row_cost.load(Ordering::Relaxed));
        queued == 0 || wait <= self.max_wait.as_nanos()
    }

    pub(super) fn has_room(&self) -> bool {
        self.fits(self.rows.load(Ordering::SeqCst), 0)
    }

    /// Count `rows` as waiting if they fit.
    pub(super) fn admit(&self, rows: usize) -> bool {
        self.rows
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |queued| {
                self.fits(queued, rows).then_some(queued + rows)
            })
            .is_ok()
    }

    /// Fold a batch of `rows` that took `predict` into the row cost.
    pub(super) fn measure(&self, rows: usize, predict: Duration) {
        if rows == 0 {
            return;
        }
        // Floor at 1 ns/row so a sub-nanosecond/row batch or coarse clock
        // never collapses row_cost to 0 and disables admission control.
        let cost = u64::try_from((predict.as_nanos() / rows as u128).max(1)).unwrap_or(u64::MAX);
        let old = self.row_cost.load(Ordering::Relaxed);
        let new = if old == 0 {
            cost
        } else {
            u64::try_from((3 * u128::from(old) + u128::from(cost)) / 4).unwrap_or(u64::MAX)
        };
        self.row_cost.store(new, Ordering::Relaxed);
    }
}

/// Owns the model and predicts each batch of queued queries at once.
pub(super) struct Worker {
    pub model: Fitted,
    pub classes: Option<Labels>,
    pub max_batch_rows: usize,
    pub queue: Arc<Queue>,
}

impl Worker {
    pub(super) fn run(self, jobs: &mpsc::Receiver<Job>) {
        let mut next = None;
        while let Some(first) = next.take().or_else(|| jobs.recv().ok()) {
            let (batch, rest) = next_batch(
                first,
                |job: &Job| job.query.rows.count,
                || jobs.try_recv().ok(),
                self.max_batch_rows,
            );
            next = rest;
            let total_rows: usize = batch.iter().map(|job| job.query.rows.count).sum();
            // Clients that hung up get no answer, so release and skip their rows now.
            let batch: Vec<Job> = batch.into_iter().filter(|job| !job.reply.is_closed()).collect();
            let active_rows: usize = batch.iter().map(|job| job.query.rows.count).sum();
            let dropped_rows = total_rows - active_rows;
            if dropped_rows > 0 {
                self.queue.rows.fetch_sub(dropped_rows, Ordering::SeqCst);
            }
            if !batch.is_empty() {
                self.answer(batch, active_rows);
            }
        }
    }

    fn answer(&self, batch: Vec<Job>, active_rows: usize) {
        let start = Instant::now();
        let queries: Vec<&Query> = batch.iter().map(|job| &job.query).collect();
        let bodies = self.predict(&queries);
        let predict = start.elapsed();
        // Release active rows after prediction finishes and before sending replies,
        // so in-flight work counts toward queue wait without blocking sequential retries.
        self.queue.rows.fetch_sub(active_rows, Ordering::SeqCst);
        // Smaller batches overstate the row cost.
        if 2 * active_rows >= self.max_batch_rows {
            self.queue.measure(active_rows, predict);
        }
        match bodies {
            Ok(bodies) => {
                for (job, body) in batch.into_iter().zip(bodies) {
                    let answer = Answer {
                        body,
                        rows: job.query.rows.count,
                        queue: start.saturating_duration_since(job.enqueued_at),
                        predict,
                    };
                    let _ = job.reply.send(Ok(answer));
                }
            }
            Err(error) => {
                for job in batch {
                    let _ = job.reply.send(Err(error.clone()));
                }
            }
        }
    }

    /// One response body per query, from one prediction over all their rows.
    pub(super) fn predict(&self, queries: &[&Query]) -> Result<Vec<Vec<u8>>, ApiError> {
        let values: Vec<f64> = queries
            .iter()
            .flat_map(|query| query.rows.values.iter().copied())
            .collect();
        if let Some(classes) = &self.classes {
            let probabilities = self.model.predict_probabilities(&values)?;
            let mut rows = probabilities.chunks(classes.len());
            return queries
                .iter()
                .map(|query| {
                    let Outputs::Classifier {
                        labels,
                        probabilities,
                    } = query.outputs
                    else {
                        unreachable!("a classifier's server makes classifier queries")
                    };
                    let rows = rows.by_ref().take(query.rows.count).collect();
                    classifier_response(labels, probabilities, classes, rows)
                })
                .collect();
        }
        let requested: Vec<(&[RegressionOutput], &[f32])> = queries
            .iter()
            .map(|query| match &query.outputs {
                Outputs::Regressor { outputs, alphas } => (&outputs[..], &alphas[..]),
                Outputs::Classifier { .. } => unreachable!("a regressor's server makes regressor queries"),
            })
            .collect();
        // Run the neural pass once for the whole batch and summarize each query only
        // over its own rows with its own alphas, preventing cross-query amplification.
        let batch_queries: Vec<(usize, &[RegressionOutput])> = queries
            .iter()
            .zip(&requested)
            .map(|(query, &(outputs, _))| (query.rows.count, outputs))
            .collect();
        let results = self.model.predict_regression_batch(&values, &batch_queries)?;
        requested
            .iter()
            .zip(&results)
            .map(|(&(outputs, alphas), query_results)| {
                let parts = query_results
                    .iter()
                    .map(|(values, width)| (values.as_slice(), *width))
                    .collect();
                regressor_response(outputs, alphas, parts)
            })
            .collect()
    }
}

/// The oldest job and the jobs queued after it that fit in `max_batch_rows`
/// with it, and the first that did not fit, which leads the next batch. It
/// never waits for more jobs.
fn next_batch<T>(
    first: T,
    rows: impl Fn(&T) -> usize,
    mut queued: impl FnMut() -> Option<T>,
    max_batch_rows: usize,
) -> (Vec<T>, Option<T>) {
    let mut total = rows(&first);
    let mut batch = vec![first];
    while let Some(job) = queued() {
        total += rows(&job);
        if total > max_batch_rows {
            return (batch, Some(job));
        }
        batch.push(job);
    }
    (batch, None)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{Queue, next_batch};
    use crate::models::quantiles::extend_query_summaries;
    use crate::{DecoderOutputs, RegressionOutput};

    #[test]
    fn a_batch_takes_the_queued_jobs_that_fit() {
        // Jobs are row counts. The first job always runs, even alone over the limit.
        let batch = |first, queued: &[usize], max| {
            let mut queued = queued.iter().copied();
            next_batch(first, |&rows| rows, || queued.next(), max)
        };
        assert_eq!(batch(1, &[], 64), (vec![1], None));
        assert_eq!(batch(1, &[2, 3], 64), (vec![1, 2, 3], None));
        assert_eq!(batch(30, &[30, 30, 1], 64), (vec![30, 30], Some(30)));
        assert_eq!(batch(100, &[1], 64), (vec![100], Some(1)));
        assert_eq!(batch(1, &[1], 1), (vec![1], Some(1)));
    }

    #[test]
    fn queue_measure_floors_at_one_nanosecond_per_row() {
        let queue = Queue::new(Duration::from_nanos(10));
        queue.measure(100, Duration::ZERO);
        assert!(queue.admit(5));
        assert!(!queue.admit(6));
    }

    #[test]
    fn batch_regression_summarizes_each_query_only_over_its_own_rows() {
        let k = 10_usize;
        let row = |offset: f32| -> Vec<f32> { (1..=10_u8).map(|i| offset + f32::from(i)).collect() };
        let decoded = DecoderOutputs::new([1, 3, k], [row(0.0), row(100.0), row(200.0)].concat()).unwrap();
        let q0 = [RegressionOutput::Mean, RegressionOutput::Quantiles(vec![0.5])];
        let q1 = [RegressionOutput::Quantiles(vec![0.1, 0.9])];
        let queries: Vec<(usize, &[RegressionOutput])> = vec![(1, &q0), (2, &q1)];
        let mut results = vec![vec![(Vec::new(), 0); q0.len()], vec![(Vec::new(), 0); q1.len()]];
        extend_query_summaries(&decoded, 0, &queries, &mut results).unwrap();
        assert_eq!(results[0][0].0.len(), 1);
        assert_eq!(results[0][1].0.len(), 1);
        assert_eq!(results[1][0].0.len(), 4);
        assert_eq!(results[1][0].1, 2);
        assert!((results[0][0].0[0] - 5.5).abs() < 1e-5);
    }
}
