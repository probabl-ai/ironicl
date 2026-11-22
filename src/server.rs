//! `ironicl serve`: an HTTP server for one exported model.
//!
//! `POST /v1/predict` takes `{"rows": [[...], ...]}` and options named after
//! `TabICL`'s predict arguments, and answers with one key per output.
//! `GET /v1/health` answers once the model is loaded. One worker thread owns the
//! model and predicts the queued requests together, in batches of up to
//! `--max-batch-rows` rows. A request that would wait longer than
//! `--max-queue` is refused. See `docs/server.md`.

use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderName, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{DeserializeSeed, Error as _, SeqAccess, Visitor},
};
use tokio::sync::oneshot;

use crate::{
    ClassifierAggregation, Error, Features, FittedModel, Labels, RegressionOutput, Result, ServeArgs,
    TaskKind, memory::MemoryLimits, models::tabicl_v2::saved,
};

/// Default `--max-batch-rows` on a GPU and on the CPU.
const GPU_BATCH_ROWS: usize = 1024;
const CPU_BATCH_ROWS: usize = 32;

/// `TabICLRegressor.predict`'s default quantile levels.
const DEFAULT_ALPHAS: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
/// The most quantile levels a request may ask for: as many as the model's raw
/// quantiles. Each level costs a value per row of the whole batch.
const MAX_ALPHAS: usize = 999;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassifierRequest {
    rows: Rows,
    output_type: Option<Outputs<ClassifierOutput>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegressorRequest {
    rows: Rows,
    output_type: Option<Outputs<RegressorOutput>>,
    alphas: Option<Alphas>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ClassifierOutput {
    Labels,
    Probabilities,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegressorOutput {
    Mean,
    Median,
    Variance,
    Quantiles,
    RawQuantiles,
}

/// At least one row, all of the same length, flattened in row order.
struct Rows {
    values: Vec<f64>,
    count: usize,
    features: usize,
}

/// Rows parse straight into one flat buffer, without an allocation per row,
/// and a ragged row fails as soon as it ends.
impl<'de> Deserialize<'de> for Rows {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_seq(RowsVisitor)
    }
}

struct RowsVisitor;

impl<'de> Visitor<'de> for RowsVisitor {
    type Value = Rows;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an array of rows of numbers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut rows: A) -> std::result::Result<Rows, A::Error> {
        let mut values = Vec::new();
        let (mut count, mut features) = (0, None);
        while let Some(length) = rows.next_element_seed(Row(&mut values))? {
            let features = *features.get_or_insert(length);
            if length != features {
                return Err(A::Error::custom(format!(
                    "row {count} has {length} values; row 0 has {features}"
                )));
            }
            count += 1;
        }
        let features = features.ok_or_else(|| A::Error::custom("rows is empty"))?;
        Ok(Rows {
            values,
            count,
            features,
        })
    }
}

/// Appends one row's values and returns how many there were.
struct Row<'a>(&'a mut Vec<f64>);

impl<'de> DeserializeSeed<'de> for Row<'_> {
    type Value = usize;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> std::result::Result<usize, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for Row<'_> {
    type Value = usize;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a row of numbers")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut row: A) -> std::result::Result<usize, A::Error> {
        let start = self.0.len();
        while let Some(value) = row.next_element()? {
            self.0.push(value);
        }
        Ok(self.0.len() - start)
    }
}

/// Requested outputs: at least one, none twice.
#[derive(Deserialize)]
#[serde(try_from = "Vec<T>", bound(deserialize = "T: Deserialize<'de> + PartialEq"))]
struct Outputs<T>(Vec<T>);

impl<T: PartialEq> TryFrom<Vec<T>> for Outputs<T> {
    type Error = &'static str;

    fn try_from(outputs: Vec<T>) -> std::result::Result<Self, &'static str> {
        if outputs.is_empty() {
            return Err("output_type is empty");
        }
        if (1..outputs.len()).any(|i| outputs[..i].contains(&outputs[i])) {
            return Err("output_type names an output twice");
        }
        Ok(Self(outputs))
    }
}

/// Quantile levels: at most [`MAX_ALPHAS`], each strictly between 0 and 1.
#[derive(Deserialize)]
#[serde(try_from = "Vec<f64>")]
struct Alphas(Vec<f64>);

impl TryFrom<Vec<f64>> for Alphas {
    type Error = String;

    fn try_from(alphas: Vec<f64>) -> std::result::Result<Self, String> {
        if alphas.is_empty() {
            return Err("alphas is empty".into());
        }
        if alphas.len() > MAX_ALPHAS {
            return Err(format!(
                "alphas has {} levels; the limit is {MAX_ALPHAS}",
                alphas.len()
            ));
        }
        if let Some(alpha) = alphas.iter().find(|&&alpha| !(alpha > 0.0 && alpha < 1.0)) {
            return Err(format!("alphas must be between 0 and 1, got {alpha}"));
        }
        Ok(Self(alphas))
    }
}

#[derive(Serialize)]
struct ClassifierResponse<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<Labels>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    probabilities: Option<ClassProbabilities<'a>>,
}

#[derive(Serialize)]
struct ClassProbabilities<'a> {
    /// The columns of `probabilities`.
    classes: &'a Labels,
    probabilities: Vec<&'a [f32]>,
}

#[derive(Serialize, Default)]
struct RegressorResponse<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    mean: Option<&'a [f32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    median: Option<&'a [f32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    variance: Option<&'a [f32]>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    quantiles: Option<Quantiles<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_quantiles: Option<Vec<&'a [f32]>>,
}

#[derive(Serialize)]
struct Quantiles<'a> {
    /// The columns of `quantiles`.
    alphas: &'a [f64],
    quantiles: Vec<&'a [f32]>,
}

/// An error response: `{"error": {"code": ..., "message": ...}}`.
#[derive(Clone)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn invalid(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, message)
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
    }

    fn worker_stopped() -> Self {
        Self::internal("the prediction worker stopped")
    }
}

impl From<Error> for ApiError {
    fn from(error: Error) -> Self {
        match error {
            Error::InvalidArgument(message) => Self::invalid("invalid_request", message),
            error => Self::internal(error.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({"error": {"code": self.code, "message": self.message}});
        (self.status, axum::Json(body)).into_response()
    }
}

/// What the handlers check requests against and hand them to the worker by.
struct Server {
    task: TaskKind,
    features: usize,
    max_request_rows: usize,
    max_queue: Duration,
    /// Rows waiting for a batch.
    queued: Arc<AtomicUsize>,
    /// Nanoseconds a row takes in a full batch, as the worker measures it.
    row_cost: Arc<AtomicU64>,
    jobs: mpsc::Sender<Job>,
}

enum Request {
    Classifier(ClassifierRequest),
    Regressor(RegressorRequest),
}

impl Request {
    fn rows(&self) -> &Rows {
        match self {
            Self::Classifier(request) => &request.rows,
            Self::Regressor(request) => &request.rows,
        }
    }

    /// The request, which a classifier's server parsed as a classifier's.
    fn classifier(&self) -> &ClassifierRequest {
        let Self::Classifier(request) = self else {
            unreachable!("a classifier's server parses classifier requests")
        };
        request
    }

    /// The request, which a regressor's server parsed as a regressor's.
    fn regressor(&self) -> &RegressorRequest {
        let Self::Regressor(request) = self else {
            unreachable!("a regressor's server parses regressor requests")
        };
        request
    }
}

/// A checked request waiting for its batch, and where its answer goes.
struct Job {
    request: Request,
    enqueued_at: Instant,
    reply: oneshot::Sender<Reply>,
}

/// A response body and how long its request waited and its batch ran.
struct Answer {
    body: Vec<u8>,
    queue: Duration,
    predict: Duration,
}

type Reply = std::result::Result<Answer, ApiError>;

impl Server {
    fn parse(&self, body: &[u8]) -> std::result::Result<Request, ApiError> {
        let parse_error = |error: serde_json::Error| ApiError::invalid("invalid_request", error.to_string());
        let request = match self.task {
            TaskKind::Classifier => Request::Classifier(serde_json::from_slice(body).map_err(parse_error)?),
            TaskKind::Regressor => Request::Regressor(serde_json::from_slice(body).map_err(parse_error)?),
        };
        let rows = request.rows();
        if rows.features != self.features {
            return Err(ApiError::invalid(
                "wrong_feature_count",
                format!(
                    "rows have {} values; the model expects {}",
                    rows.features, self.features
                ),
            ));
        }
        if rows.count > self.max_request_rows {
            return Err(ApiError::invalid(
                "too_many_rows",
                format!("{} rows; the limit is {}", rows.count, self.max_request_rows),
            ));
        }
        Ok(request)
    }

    /// Whether `rows` more can wait: the queued rows and theirs would take at
    /// most `max_queue` to predict. An empty queue takes any request, so one
    /// larger than the queue still runs.
    fn fits(&self, queued: usize, rows: usize) -> bool {
        let wait = (queued + rows) as u128 * u128::from(self.row_cost.load(Ordering::Relaxed));
        queued == 0 || wait <= self.max_queue.as_nanos()
    }

    /// Count `rows` as waiting if they fit.
    fn admit(&self, rows: usize) -> bool {
        self.queued
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |queued| {
                self.fits(queued, rows).then_some(queued + rows)
            })
            .is_ok()
    }
}

/// The oldest job and the jobs queued after it that fit in `max_batch_rows` with
/// it, and the first that did not fit, which leads the next batch. It never
/// waits for more jobs.
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

/// Owns the model and predicts each batch of queued requests at once.
struct Worker {
    model: FittedModel,
    features: usize,
    classes: Option<Labels>,
    aggregation: ClassifierAggregation,
    max_batch_rows: usize,
    queued: Arc<AtomicUsize>,
    row_cost: Arc<AtomicU64>,
    jobs: mpsc::Receiver<Job>,
}

impl Worker {
    fn run(self) {
        let mut next = None;
        while let Some(first) = next.take().or_else(|| self.jobs.recv().ok()) {
            let (batch, rest) = next_batch(
                first,
                |job: &Job| job.request.rows().count,
                || self.jobs.try_recv().ok(),
                self.max_batch_rows,
            );
            next = rest;
            let rows = batch.iter().map(|job| job.request.rows().count).sum();
            self.queued.fetch_sub(rows, Ordering::SeqCst);
            // Clients that hung up get no answer, so skip their rows.
            let batch: Vec<Job> = batch.into_iter().filter(|job| !job.reply.is_closed()).collect();
            if !batch.is_empty() {
                self.answer(batch);
            }
        }
    }

    fn answer(&self, batch: Vec<Job>) {
        let start = Instant::now();
        let bodies = self.predict(&batch);
        let predict = start.elapsed();
        let rows: usize = batch.iter().map(|job| job.request.rows().count).sum();
        if 2 * rows >= self.max_batch_rows {
            self.measure(rows, predict);
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

    /// Fold a batch of at least half `max_batch_rows` into the row cost, which
    /// smaller batches overstate.
    fn measure(&self, rows: usize, predict: Duration) {
        let cost = u64::try_from(predict.as_nanos() / rows as u128).unwrap_or(u64::MAX);
        let old = self.row_cost.load(Ordering::Relaxed);
        let new = if old == 0 { cost } else { (3 * old + cost) / 4 };
        self.row_cost.store(new, Ordering::Relaxed);
    }

    /// One response body per job, from one prediction over all their rows.
    fn predict(&self, batch: &[Job]) -> std::result::Result<Vec<Vec<u8>>, ApiError> {
        let values: Vec<f64> = batch
            .iter()
            .flat_map(|job| job.request.rows().values.iter().copied())
            .collect();
        let x = Features::F64(&values);
        if let Some(classes) = &self.classes {
            let probabilities = self
                .model
                .predict_probabilities(x, self.features, self.aggregation)?;
            let mut rows = probabilities.values().chunks(classes.len());
            return Ok(batch
                .iter()
                .map(|job| {
                    let request = job.request.classifier();
                    let rows = rows.by_ref().take(request.rows.count).collect();
                    classifier_response(request, classes, rows)
                })
                .collect());
        }
        let requests: Vec<&RegressorRequest> = batch.iter().map(|job| job.request.regressor()).collect();
        let (outputs, picks) = merge_outputs(requests.iter().map(|request| regression_outputs(request)));
        let results = self.model.predict_regression(x, self.features, &outputs)?;
        let mut start = 0;
        Ok(requests
            .iter()
            .zip(picks)
            .map(|(request, picks)| {
                let count = request.rows.count;
                let parts = picks
                    .iter()
                    .map(|&pick| {
                        let (values, width) = &results[pick];
                        (&values[start * width..(start + count) * width], *width)
                    })
                    .collect();
                start += count;
                regressor_response(request, parts)
            })
            .collect())
    }
}

/// Every output that some request asks for, each once, and for each request
/// where its outputs are in that list.
fn merge_outputs(
    requests: impl Iterator<Item = Vec<RegressionOutput>>,
) -> (Vec<RegressionOutput>, Vec<Vec<usize>>) {
    let mut merged: Vec<RegressionOutput> = Vec::new();
    let mut picks = Vec::new();
    for outputs in requests {
        let mut pick = Vec::with_capacity(outputs.len());
        for output in outputs {
            let index = merged.iter().position(|seen| *seen == output).unwrap_or_else(|| {
                merged.push(output);
                merged.len() - 1
            });
            pick.push(index);
        }
        picks.push(pick);
    }
    (merged, picks)
}

fn classifier_outputs(request: &ClassifierRequest) -> &[ClassifierOutput] {
    request.output_type.as_ref().map_or(
        &[ClassifierOutput::Labels, ClassifierOutput::Probabilities][..],
        |outputs| &outputs.0,
    )
}

fn classifier_response(request: &ClassifierRequest, classes: &Labels, rows: Vec<&[f32]>) -> Vec<u8> {
    let outputs = classifier_outputs(request);
    let labels = outputs.contains(&ClassifierOutput::Labels).then(|| {
        // The first most probable class, as NumPy's argmax picks it.
        let best: Vec<usize> = rows
            .iter()
            .map(|row| (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }))
            .collect();
        classes.select(&best)
    });
    let probabilities = outputs
        .contains(&ClassifierOutput::Probabilities)
        .then_some(ClassProbabilities {
            classes,
            probabilities: rows,
        });
    serde_json::to_vec(&ClassifierResponse {
        labels,
        probabilities,
    })
    .expect("the response serializes")
}

fn regressor_outputs(request: &RegressorRequest) -> &[RegressorOutput] {
    request
        .output_type
        .as_ref()
        .map_or(&[RegressorOutput::Mean][..], |outputs| &outputs.0)
}

fn alphas(request: &RegressorRequest) -> &[f64] {
    request
        .alphas
        .as_ref()
        .map_or(&DEFAULT_ALPHAS[..], |alphas| &alphas.0)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "The model computes quantiles in FP32"
)]
fn regression_outputs(request: &RegressorRequest) -> Vec<RegressionOutput> {
    regressor_outputs(request)
        .iter()
        .map(|output| match output {
            RegressorOutput::Mean => RegressionOutput::Mean,
            RegressorOutput::Median => RegressionOutput::Median,
            RegressorOutput::Variance => RegressionOutput::Variance,
            RegressorOutput::Quantiles => {
                RegressionOutput::Quantiles(alphas(request).iter().map(|&alpha| alpha as f32).collect())
            }
            RegressorOutput::RawQuantiles => RegressionOutput::RawQuantiles,
        })
        .collect()
}

/// `parts`: each requested output's rows of this request, with their width.
fn regressor_response(request: &RegressorRequest, parts: Vec<(&[f32], usize)>) -> Vec<u8> {
    let mut response = RegressorResponse::default();
    for (output, (values, width)) in regressor_outputs(request).iter().zip(parts) {
        let matrix = || values.chunks(width.max(1)).collect();
        match output {
            RegressorOutput::Mean => response.mean = Some(values),
            RegressorOutput::Median => response.median = Some(values),
            RegressorOutput::Variance => response.variance = Some(values),
            RegressorOutput::Quantiles => {
                response.quantiles = Some(Quantiles {
                    alphas: alphas(request),
                    quantiles: matrix(),
                });
            }
            RegressorOutput::RawQuantiles => response.raw_quantiles = Some(matrix()),
        }
    }
    serde_json::to_vec(&response).expect("the response serializes")
}

async fn predict(
    State(server): State<Arc<Server>>,
    body: std::result::Result<Bytes, BytesRejection>,
) -> Response {
    let body = match body {
        Ok(body) => body,
        Err(rejection) if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            return ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "body_too_large",
                rejection.body_text(),
            )
            .into_response();
        }
        Err(rejection) => return ApiError::invalid("invalid_request", rejection.body_text()).into_response(),
    };
    let overloaded = || {
        let error = ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded",
            "the queue is full; retry later",
        );
        ([(header::RETRY_AFTER, "1")], error).into_response()
    };
    // A full queue refuses the request before the work of parsing it.
    if !server.fits(server.queued.load(Ordering::SeqCst), 0) {
        return overloaded();
    }
    let request = match server.parse(&body) {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    if !server.admit(request.rows().count) {
        return overloaded();
    }
    let (reply, answer) = oneshot::channel();
    let job = Job {
        request,
        enqueued_at: Instant::now(),
        reply,
    };
    if server.jobs.send(job).is_err() {
        return ApiError::worker_stopped().into_response();
    }
    match answer.await {
        Ok(Ok(Answer { body, queue, predict })) => {
            let timing = format!(
                "queue;dur={:.3}, predict;dur={:.3}",
                queue.as_secs_f64() * 1e3,
                predict.as_secs_f64() * 1e3
            );
            (
                [
                    (header::CONTENT_TYPE, "application/json".to_owned()),
                    (HeaderName::from_static("server-timing"), timing),
                ],
                body,
            )
                .into_response()
        }
        Ok(Err(error)) => error.into_response(),
        Err(_) => ApiError::worker_stopped().into_response(),
    }
}

async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({"status": "ok"}))
}

/// Load the exported model, predict once to warm it up, and serve it until
/// Ctrl-C.
///
/// # Errors
///
/// When the model does not load, as [`saved::load`], or the address cannot
/// be bound.
pub fn serve(args: &ServeArgs) -> Result<()> {
    let model = saved::load(&args.dir, None, MemoryLimits::resolve(None, None)?)?;
    let preprocessor = model.preprocessed().preprocessor();
    let (task, features) = (model.task(), preprocessor.n_features_in());
    let device = model.device();
    let max_batch_rows = args.max_batch_rows.unwrap_or(if device == "cpu" {
        CPU_BATCH_ROWS
    } else {
        GPU_BATCH_ROWS
    });
    let (jobs, receiver) = mpsc::channel();
    let queued = Arc::new(AtomicUsize::new(0));
    let row_cost = Arc::new(AtomicU64::new(0));
    let worker = Worker {
        classes: preprocessor.classes().cloned(),
        aggregation: preprocessor.aggregation.unwrap_or_default(),
        model,
        features,
        max_batch_rows,
        queued: Arc::clone(&queued),
        row_cost: Arc::clone(&row_cost),
        jobs: receiver,
    };
    let server = Server {
        task,
        features,
        max_request_rows: args.max_request_rows,
        max_queue: args.max_queue,
        queued,
        row_cost,
        jobs,
    };
    // Predict a full batch twice: once to warm up, once to measure the row cost.
    let rows = Rows {
        values: vec![0.0; max_batch_rows * features],
        count: max_batch_rows,
        features,
    };
    let request = match task {
        TaskKind::Classifier => Request::Classifier(ClassifierRequest {
            rows,
            output_type: None,
        }),
        TaskKind::Regressor => Request::Regressor(RegressorRequest {
            rows,
            output_type: None,
            alphas: None,
        }),
    };
    let warm_up = [Job {
        request,
        enqueued_at: Instant::now(),
        reply: oneshot::channel().0,
    }];
    for measure in [false, true] {
        let start = Instant::now();
        worker
            .predict(&warm_up)
            .map_err(|error| Error::InvalidConfig(format!("warm-up prediction failed: {}", error.message)))?;
        if measure {
            worker.measure(max_batch_rows, start.elapsed());
        }
    }
    // A worker that panicked can never answer, so end the process rather than
    // keep answering /v1/health; a supervisor restarts it.
    std::thread::spawn(move || {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.run())).is_err() {
            eprintln!("the prediction worker panicked; exiting");
            std::process::exit(1);
        }
    });
    let app = Router::new()
        .route("/v1/predict", post(predict))
        .route("/v1/health", get(health))
        .layer(DefaultBodyLimit::max(args.max_body))
        .with_state(Arc::new(server));
    let failed = |error: std::io::Error| Error::InvalidConfig(format!("server: {error}"));
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(failed)?
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(args.bind).await.map_err(failed)?;
            println!(
                "Serving {} on http://{} ({device}; batches of up to {max_batch_rows} rows, \
                 waits of up to {} ms)",
                args.dir.display(),
                args.bind,
                args.max_queue.as_millis()
            );
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
                .map_err(failed)
        })
}

#[cfg(test)]
mod tests {
    use super::{
        ClassifierRequest, MAX_ALPHAS, RegressionOutput, RegressorRequest, merge_outputs, next_batch,
    };

    fn error<T: for<'de> serde::Deserialize<'de>>(json: &str) -> String {
        serde_json::from_str::<T>(json)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn requests_are_checked_as_they_parse() {
        let request: RegressorRequest = serde_json::from_str(
            r#"{"rows": [[1, 2], [3, 4]], "output_type": ["quantiles"], "alphas": [0.5]}"#,
        )
        .unwrap();
        assert_eq!(
            (request.rows.values, request.rows.count, request.rows.features),
            (vec![1.0, 2.0, 3.0, 4.0], 2, 2)
        );
        for (json, message) in [
            (r#"{"rows": []}"#, "rows is empty"),
            (r#"{"rows": [[1, 2], [3]]}"#, "row 1 has 1 values; row 0 has 2"),
            (r#"{"rows": [[1, null]]}"#, "invalid type: null"),
            (
                r#"{"rows": [[1]], "output_type": ["mean"]}"#,
                "unknown variant `mean`",
            ),
            (
                r#"{"rows": [[1]], "output_type": ["labels", "labels"]}"#,
                "names an output twice",
            ),
            (r#"{"rows": [[1]], "alphas": [0.5]}"#, "unknown field `alphas`"),
        ] {
            assert!(
                error::<ClassifierRequest>(json).contains(message),
                "{json}: {}",
                error::<ClassifierRequest>(json)
            );
        }
        assert!(
            error::<RegressorRequest>(r#"{"rows": [[1]], "alphas": [0.5, 1]}"#)
                .contains("between 0 and 1, got 1")
        );
        let alphas = vec!["0.5"; MAX_ALPHAS + 1].join(", ");
        assert!(
            error::<RegressorRequest>(&format!(r#"{{"rows": [[1]], "alphas": [{alphas}]}}"#))
                .contains("the limit is 999")
        );
        for (json, message) in [
            (r#"{"rows": [1]}"#, "expected a row of numbers"),
            (r#"{"rows": {}}"#, "expected an array of rows"),
            (r#"{"rows": [[1], [2, 3], "never read"]}"#, "row 1 has 2 values"),
        ] {
            assert!(
                error::<ClassifierRequest>(json).contains(message),
                "{json}: {}",
                error::<ClassifierRequest>(json)
            );
        }
    }

    #[test]
    fn a_batch_computes_each_requested_output_once() {
        use RegressionOutput::{Mean, Median, Quantiles};
        let (merged, picks) = merge_outputs(
            [
                vec![Mean],
                vec![Quantiles(vec![0.5]), Mean],
                vec![Median, Quantiles(vec![0.5])],
                vec![Quantiles(vec![0.1])],
            ]
            .into_iter(),
        );
        assert_eq!(merged, [Mean, Quantiles(vec![0.5]), Median, Quantiles(vec![0.1])]);
        assert_eq!(picks, [vec![0], vec![1, 0], vec![2, 1], vec![3]]);
    }

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
}
