//! Ironicl reference types, model artifacts, and scalar `TabICLv2` execution.
// The CUDA kernels convert to FP16; CUDA builds run with `RUSTC_BOOTSTRAP`.
#![cfg_attr(feature = "cuda", feature(f16))]

pub mod artifact;
pub mod backend;
pub mod benchmark;
pub mod cli;
pub mod config;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod error;
pub mod input_bundle;
pub mod kernels;
pub mod memory;
pub mod model_artifact;
pub mod models;
pub mod platform;
pub mod profile;
pub mod progress;
#[cfg(feature = "python")]
mod python;
pub mod query_bundle;
pub mod static_context;

pub use backend::{BackendKind, BackendPreference, resolve_backend};
pub use benchmark::BenchmarkStats;
pub use cli::{
    BenchmarkArgs, CheckpointCacheOptions, Command, HelpTopic, ImportArgs, InspectArgs, ModelSource,
    PlanArgs, RunArgs, parse_cli,
};
pub use config::{CacheMode, ExecutionConfig, StorageDType, TabIclV2Model, TaskKind, Workload};
pub use error::{Error, Result};
pub use input_bundle::{
    CLASSIFIER_INPUT_BUNDLE_KIND, CLASSIFIER_INPUT_BUNDLE_MAGIC, CLASSIFIER_INPUT_BUNDLE_VERSION,
    InputBundle, QueryRequest, Request,
};
pub use memory::{CacheEstimate, MemoryPlan, estimate_memory};
pub use model_artifact::{
    MODEL_ARTIFACT_MAGIC, MODEL_ARTIFACT_VERSION, ModelArtifact, ModelArtifactSummary, ModelConfig,
    ModelProfile, ModelTensorF32, inspect_model_artifact, load_model_artifact,
};
pub use models::tabicl_v2::checkpoint::{
    CheckpointProfile, CheckpointSummary, import_checkpoint, inspect_checkpoint,
};
pub use models::tabicl_v2::checkpoint_cache::{PreparedCheckpoint, prepare_checkpoint};
pub use models::tabicl_v2::cpu::{
    ClassifierAggregation, ClassifierProbabilities, DecoderOutputs, Executor, ForwardTrace, Optimize,
    SparseAttentionStatistics, StaticExecutor, TraceStage, plan_context,
};
pub use models::tabicl_v2::quantiles::RegressionOutput;
pub use platform::CpuCapabilities;
pub use query_bundle::{STATIC_QUERY_BUNDLE_MAGIC, STATIC_QUERY_BUNDLE_VERSION, StaticQueryBundle};
pub use static_context::{
    STATIC_CONTEXT_MAGIC, STATIC_CONTEXT_VERSION, StaticContextArtifact, StaticContextMetadata,
    StaticContextTensor, StaticContextTensorData, StaticContextTensorRef,
};
