//! `IronICL`: cache-aware CPU inference for `TabICLv2` from imported checkpoints and
//! precomputed static contexts.

pub mod backend;
pub mod benchmark;
pub mod cache;
#[cfg(feature = "checkpoint")]
pub mod checkpoint;
pub mod checkpoint_cache;
pub mod cli;
pub mod config;
pub mod context_builder;
pub mod error;
mod execution_plan;
pub mod executor;
pub mod input_bundle;
pub mod kernels;
pub mod model_artifact;
pub mod platform;
pub mod profile;
#[cfg(feature = "python")]
mod python;
pub mod query_bundle;
pub mod static_context;

pub use backend::{BackendKind, BackendPreference, resolve_backend};
pub use benchmark::BenchmarkStats;
pub use cache::{CacheEstimate, MemoryPlan, estimate_memory};
#[cfg(feature = "checkpoint")]
pub use checkpoint::{CheckpointProfile, CheckpointSummary, import_checkpoint, inspect_checkpoint};
pub use checkpoint_cache::{PreparedCheckpoint, prepare_checkpoint};
pub use cli::{
    BenchmarkArgs, BuildContextArgs, CheckpointCacheOptions, Command, HelpTopic, ImportArgs, InspectArgs,
    ModelSource, PlanArgs, RunArgs, parse_cli,
};
pub use config::{CacheMode, ExecutionConfig, StorageDType, TabIclV2Model, TaskKind, Workload};
pub use context_builder::{
    StaticContextOptions, StaticContextRequest, StaticContextSummary, build_static_context,
    build_static_context_file,
};
pub use error::{Error, Result};
pub use executor::{
    ClassifierExecutor, ClassifierLogits, ClassifierLogitsTrace, ClassifierProbabilities,
    ClassifierTraceStage, SparseAttentionStatistics, StaticClassifierExecutor,
};
pub use input_bundle::{
    CLASSIFIER_INPUT_BUNDLE_KIND, CLASSIFIER_INPUT_BUNDLE_MAGIC, CLASSIFIER_INPUT_BUNDLE_VERSION,
    ClassifierInputBundle, ClassifierQueryRequest, ClassifierRequest,
};
pub use model_artifact::{
    MODEL_ARTIFACT_MAGIC, MODEL_ARTIFACT_VERSION, ModelArtifact, ModelArtifactSummary, ModelConfig,
    ModelProfile, ModelTensorF32, decode_sha256, encode_sha256, inspect_model_artifact, load_model_artifact,
};
pub use platform::CpuCapabilities;
pub use query_bundle::{STATIC_QUERY_BUNDLE_MAGIC, STATIC_QUERY_BUNDLE_VERSION, StaticQueryBundle};
pub use static_context::{
    KvDtype, STATIC_CONTEXT_MAGIC, STATIC_CONTEXT_VERSION, StaticContextArtifact, StaticContextFileSummary,
    StaticContextLayout, StaticContextMetadata, StaticContextTensor, StaticContextTensorData,
    StaticContextTensorRef, StaticContextWriter,
};
