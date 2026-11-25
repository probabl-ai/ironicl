//! `ironicl serve`: an HTTP server for one model directory.
//!
//! `POST /v1/predict` takes a table as pandas'
//! `to_json(orient="split", index=False)` writes it, `{"columns": [...],
//! "data": [[...], ...]}`, and options named after `TabICL`'s predict
//! arguments, and answers with one key per output.
//! `GET /v1/health` answers once the model is loaded. [`api`] checks each
//! request and queues it; the [`worker`] thread owns the model and predicts the
//! queued requests together, in batches of up to `--max-batch-rows` rows. A
//! request that would wait longer than `--max-queue` is refused. See
//! `docs/server.md`.

mod api;
mod error;
mod request;
mod response;
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
use tokio::sync::{Semaphore, oneshot};

use self::{
    error::ApiError,
    request::Schema,
    worker::{Answer, Job, Query, Queue, Worker},
};
use crate::{Error, Result, ServeArgs, memory::MemoryLimits, models::directory};

/// Default `--max-batch-rows` on a GPU and on the CPU.
const GPU_BATCH_ROWS: usize = 1024;
const CPU_BATCH_ROWS: usize = 32;

/// What the handlers share: the schema requests are checked against, the
/// queue, a cap on concurrent JSON parsing, and the way to the worker.
struct Server {
    schema: Schema,
    queue: Arc<Queue>,
    parse_slots: Arc<Semaphore>,
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
        self.jobs
            .send(job)
            .map_err(|_| ApiError::internal("the prediction worker stopped"))?;
        answer
            .await
            .map_err(|_| ApiError::internal("the prediction worker stopped"))?
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
    init_logging()?;
    let weights = args
        .weights
        .clone()
        .map_or_else(directory::default_weights_dir, Ok)?;
    let start = Instant::now();
    let limits = MemoryLimits::resolve(None, None)?;
    let prepared = directory::load(&args.dir, &weights, args.optimize, None, limits)?;
    if let Some(warning) = &prepared.warning {
        tracing::warn!("{warning}");
    }
    if let Err(error) = prepared.save_cache_if_built(&args.dir) {
        tracing::warn!("could not save the KV cache; serving from memory: {error}");
    }
    let cache = if prepared.built {
        format!("KV cache built in {:.1} s", start.elapsed().as_secs_f64())
    } else {
        "KV cache loaded".to_owned()
    };
    let model = prepared.model;
    let (task, features) = (model.task(), model.categories().len());
    let (categories, classes, feature_names) = (
        model.categories().to_vec(),
        model.classes().cloned(),
        model.feature_names().map(<[String]>::to_vec),
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
        classes,
        max_batch_rows,
        queue: Arc::clone(&queue),
    };
    let schema = Schema::new(task, categories.clone(), feature_names, args.max_request_rows);
    // Predict a full batch twice: once to warm up, once to measure the row cost.
    let failed =
        |error: ApiError| Error::InvalidConfig(format!("warm-up prediction failed: {}", error.message()));
    let warm_up = request::missing_rows(task, categories, max_batch_rows).map_err(failed)?;
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
            tracing::error!("the prediction worker panicked; exiting");
            std::process::exit(1);
        }
    });
    let parse_concurrency = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .clamp(2, 16);
    let server = Server {
        schema,
        queue,
        parse_slots: Arc::new(Semaphore::new(parse_concurrency)),
        jobs,
    };
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
            tracing::info!(
                "serving {} ({task}, {features} features) on http://{} ({device}, {}; {cache}; \
                 batches of up to {max_batch_rows} rows, waits of up to {} ms)",
                args.dir.display(),
                args.bind,
                prepared.optimize,
                args.max_queue.as_millis()
            );
            if !args.bind.ip().is_loopback() {
                tracing::warn!(
                    "serving on {}; a real deployment needs a reverse proxy in front (docs/server.md)",
                    args.bind
                );
            }
            axum::serve(listener, app)
                .with_graceful_shutdown(stop_signal())
                .await
                .map_err(failed)
        })
}

/// Ctrl-C, or SIGTERM on Unix, as service managers stop a process.
async fn stop_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = signal(SignalKind::terminate()).expect("a SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutting down");
}

/// Log to stderr at the level `RUST_LOG` sets, `info` when it is unset.
fn init_logging() -> Result<()> {
    use tracing_subscriber::EnvFilter;
    let filter = match std::env::var_os("RUST_LOG") {
        None => EnvFilter::new("info"),
        Some(_) => EnvFilter::try_from_default_env()
            .map_err(|error| Error::InvalidConfig(format!("RUST_LOG: {error}")))?,
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
    Ok(())
}
