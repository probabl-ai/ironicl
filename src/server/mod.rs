//! `ironicl serve`: an HTTP server for one model directory.
//!
//! `POST /v1/predict` takes `{"rows": [[...], ...]}` and options named after
//! `TabICL`'s predict arguments, and answers with one key per output.
//! `GET /v1/health` answers once the model is loaded. [`api`] checks each
//! request and queues it; the [`worker`] thread owns the model and predicts the
//! queued requests together, in batches of up to `--max-batch-rows` rows. A
//! request that would wait longer than `--max-queue` is refused. See
//! `docs/server.md`.

mod api;
mod worker;

use std::{
    sync::{Arc, mpsc},
    time::Instant,
};

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use tokio::sync::oneshot;

use self::{
    api::{ApiError, Limits},
    worker::{Answer, Job, Query, Queue, Worker},
};
use crate::{Error, Result, ServeArgs, memory::MemoryLimits, models::tabicl_v2::directory};

/// Default `--max-batch-rows` on a GPU and on the CPU.
const GPU_BATCH_ROWS: usize = 1024;
const CPU_BATCH_ROWS: usize = 32;

/// What the handlers share: the limits requests are checked against, the
/// queue, and the way to the worker.
struct Server {
    limits: Limits,
    queue: Arc<Queue>,
    jobs: mpsc::Sender<Job>,
}

impl Server {
    /// Queue `query` for the next batch it fits in, and wait for its answer.
    async fn submit(&self, query: Query) -> std::result::Result<Answer, ApiError> {
        if !self.queue.admit(query.rows.count) {
            return Err(ApiError::overloaded());
        }
        let (reply, answer) = oneshot::channel();
        let job = Job {
            query,
            enqueued_at: Instant::now(),
            reply,
        };
        self.jobs.send(job).map_err(|_| ApiError::worker_stopped())?;
        answer.await.map_err(|_| ApiError::worker_stopped())?
    }
}

/// Prepare the model directory as `ironicl prepare` does, predict a full batch
/// to warm it up, and serve it until Ctrl-C.
///
/// # Errors
///
/// As [`directory::prepare`], when the warm-up fails, or when the address
/// cannot be bound.
pub fn serve(args: &ServeArgs) -> Result<()> {
    let weights = args
        .weights
        .clone()
        .map_or_else(directory::default_weights_dir, Ok)?;
    let start = Instant::now();
    let limits = MemoryLimits::resolve(None, None)?;
    let prepared = directory::prepare(&args.dir, &weights, args.optimize, None, limits)?;
    let cache = if prepared.built {
        format!("KV cache built in {:.1} s", start.elapsed().as_secs_f64())
    } else {
        "KV cache loaded".to_owned()
    };
    let model = prepared.model;
    let preprocessor = model.preprocessed().preprocessor();
    let (task, features) = (model.task(), preprocessor.n_features_in());
    let (classes, aggregation, feature_names) = (
        preprocessor.classes().cloned(),
        preprocessor.aggregation.unwrap_or_default(),
        preprocessor.feature_names.clone(),
    );
    let device = model.device();
    let max_batch_rows = args.max_batch_rows.unwrap_or(if device == "cpu" {
        CPU_BATCH_ROWS
    } else {
        GPU_BATCH_ROWS
    });
    let queue = Arc::new(Queue::new(args.max_queue));
    let worker = Worker {
        model,
        features,
        classes,
        aggregation,
        max_batch_rows,
        queue: Arc::clone(&queue),
    };
    let limits = Limits::new(task, features, feature_names, args.max_request_rows);
    // Predict a full batch twice: once to warm up, once to measure the row cost.
    let failed =
        |error: ApiError| Error::InvalidConfig(format!("warm-up prediction failed: {}", error.message()));
    let warm_up = api::zeros(task, features, max_batch_rows).map_err(failed)?;
    for measure in [false, true] {
        let start = Instant::now();
        worker.predict(&[&warm_up]).map_err(failed)?;
        if measure {
            queue.measure(max_batch_rows, start.elapsed());
        }
    }
    let (jobs, receiver) = mpsc::channel();
    // A worker that panicked can never answer, so end the process rather than
    // keep answering /v1/health; a supervisor restarts it.
    std::thread::spawn(move || {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.run(&receiver))).is_err() {
            eprintln!("the prediction worker panicked; exiting");
            std::process::exit(1);
        }
    });
    let server = Server { limits, queue, jobs };
    let app = Router::new()
        .route("/v1/predict", post(api::predict))
        .route("/v1/health", get(api::health))
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
                "Serving {} ({task}, {features} features) on http://{} ({device}, {}; {cache}; \
                 batches of up to {max_batch_rows} rows, waits of up to {} ms)",
                args.dir.display(),
                args.bind,
                prepared.optimize,
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
