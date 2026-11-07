//! Ironicl reference types, model artifacts, and scalar `TabICLv2` execution.

pub mod artifact;
pub mod backend;
pub mod benchmark;
pub mod cache;
pub mod checkpoint;
pub mod checkpoint_cache;
pub mod cli;
pub mod config;
pub mod error;
mod execution_plan;
pub mod executor;
pub mod input_bundle;
pub mod kernels;
pub mod memory;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal;
pub mod model_artifact;
pub mod platform;
pub mod profile;
#[cfg(feature = "python")]
mod python;
pub mod quantiles;
pub mod query_bundle;
pub mod static_context;

pub use backend::{BackendKind, BackendPreference, resolve_backend};
pub use benchmark::BenchmarkStats;
pub use cache::{CacheEstimate, MemoryPlan, estimate_memory};
pub use checkpoint::{CheckpointProfile, CheckpointSummary, import_checkpoint, inspect_checkpoint};
pub use checkpoint_cache::{PreparedCheckpoint, prepare_checkpoint};
pub use cli::{
    BenchmarkArgs, CheckpointCacheOptions, Command, HelpTopic, ImportArgs, InspectArgs, ModelSource,
    PlanArgs, RunArgs, parse_cli,
};
pub use config::{CacheMode, ExecutionConfig, StorageDType, TabIclV2Model, TaskKind, Workload};
pub use error::{Error, Result};
pub use executor::{
    ClassifierAggregation, ClassifierProbabilities, DecoderOutputs, Executor, ForwardTrace, Optimize,
    SparseAttentionStatistics, StaticExecutor, TraceStage, plan_context,
};
pub use input_bundle::{
    CLASSIFIER_INPUT_BUNDLE_KIND, CLASSIFIER_INPUT_BUNDLE_MAGIC, CLASSIFIER_INPUT_BUNDLE_VERSION,
    InputBundle, QueryRequest, Request,
};
pub use model_artifact::{
    MODEL_ARTIFACT_MAGIC, MODEL_ARTIFACT_VERSION, ModelArtifact, ModelArtifactSummary, ModelConfig,
    ModelProfile, ModelTensorF32, inspect_model_artifact, load_model_artifact,
};
pub use platform::CpuCapabilities;
pub use quantiles::RegressionOutput;
pub use query_bundle::{STATIC_QUERY_BUNDLE_MAGIC, STATIC_QUERY_BUNDLE_VERSION, StaticQueryBundle};
pub use static_context::{
    STATIC_CONTEXT_MAGIC, STATIC_CONTEXT_VERSION, StaticContextArtifact, StaticContextMetadata,
    StaticContextTensor, StaticContextTensorData, StaticContextTensorRef,
};
