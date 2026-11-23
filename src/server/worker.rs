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
use crate::{ClassifierAggregation, Features, FittedModel, Labels, RegressionOutput};

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

/// A response body and how long its request waited and its batch ran.
pub(super) struct Answer {
    pub body: Vec<u8>,
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
        let cost = u64::try_from(predict.as_nanos() / rows as u128).unwrap_or(u64::MAX);
        let old = self.row_cost.load(Ordering::Relaxed);
        let new = if old == 0 { cost } else { (3 * old + cost) / 4 };
        self.row_cost.store(new, Ordering::Relaxed);
    }
}

/// Owns the model and predicts each batch of queued queries at once.
pub(super) struct Worker {
    pub model: FittedModel,
    pub features: usize,
    pub classes: Option<Labels>,
    pub aggregation: ClassifierAggregation,
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
            let rows = batch.iter().map(|job| job.query.rows.count).sum();
            self.queue.rows.fetch_sub(rows, Ordering::SeqCst);
            // Clients that hung up get no answer, so skip their rows.
            let batch: Vec<Job> = batch.into_iter().filter(|job| !job.reply.is_closed()).collect();
            if !batch.is_empty() {
                self.answer(batch);
            }
        }
    }

    fn answer(&self, batch: Vec<Job>) {
        let start = Instant::now();
        let queries: Vec<&Query> = batch.iter().map(|job| &job.query).collect();
        let bodies = self.predict(&queries);
        let predict = start.elapsed();
        let rows: usize = queries.iter().map(|query| query.rows.count).sum();
        // Smaller batches overstate the row cost.
        if 2 * rows >= self.max_batch_rows {
            self.queue.measure(rows, predict);
        }
        match bodies {
            Ok(bodies) => {
                for (job, body) in batch.into_iter().zip(bodies) {
                    let queue = start - job.enqueued_at;
                    let _ = job.reply.send(Ok(Answer { body, queue, predict }));
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
        let x = Features::F64(&values);
        if let Some(classes) = &self.classes {
            let probabilities = self
                .model
                .predict_probabilities(x, self.features, self.aggregation)?;
            let mut rows = probabilities.values().chunks(classes.len());
            return Ok(queries
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
                .collect());
        }
        let requested: Vec<(&[RegressionOutput], &[f32])> = queries
            .iter()
            .map(|query| match &query.outputs {
                Outputs::Regressor { outputs, alphas } => (&outputs[..], &alphas[..]),
                Outputs::Classifier { .. } => unreachable!("a regressor's server makes regressor queries"),
            })
            .collect();
        let (merged, picks) = merge_outputs(requested.iter().map(|(outputs, _)| *outputs));
        let results = self.model.predict_regression(x, self.features, &merged)?;
        let mut start = 0;
        Ok(queries
            .iter()
            .zip(requested)
            .zip(picks)
            .map(|((query, (outputs, alphas)), picks)| {
                let count = query.rows.count;
                let parts = picks
                    .iter()
                    .map(|&pick| {
                        let (values, width) = &results[pick];
                        (&values[start * width..(start + count) * width], *width)
                    })
                    .collect();
                start += count;
                regressor_response(outputs, alphas, parts)
            })
            .collect())
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

/// Every output that some query asks for, each once, and for each query where
/// its outputs are in that list.
fn merge_outputs<'a>(
    queries: impl Iterator<Item = &'a [RegressionOutput]>,
) -> (Vec<RegressionOutput>, Vec<Vec<usize>>) {
    let mut merged: Vec<RegressionOutput> = Vec::new();
    let mut picks = Vec::new();
    for outputs in queries {
        let mut pick = Vec::with_capacity(outputs.len());
        for output in outputs {
            let index = merged.iter().position(|seen| seen == output).unwrap_or_else(|| {
                merged.push(output.clone());
                merged.len() - 1
            });
            pick.push(index);
        }
        picks.push(pick);
    }
    (merged, picks)
}

#[cfg(test)]
mod tests {
    use super::{merge_outputs, next_batch};
    use crate::RegressionOutput;

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
    fn a_batch_computes_each_requested_output_once() {
        use RegressionOutput::{Mean, Median, Quantiles};
        let queries = [
            vec![Mean],
            vec![Quantiles(vec![0.5]), Mean],
            vec![Median, Quantiles(vec![0.5])],
            vec![Quantiles(vec![0.1])],
        ];
        let (merged, picks) = merge_outputs(queries.iter().map(Vec::as_slice));
        assert_eq!(merged, [Mean, Quantiles(vec![0.5]), Median, Quantiles(vec![0.1])]);
        assert_eq!(picks, [vec![0], vec![1, 0], vec![2, 1], vec![3]]);
    }
}
