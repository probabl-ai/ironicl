//! Ironicl reference types, model artifacts, and scalar `TabICLv2` execution.
// The CUDA kernels convert to FP16; CUDA builds run with `RUSTC_BOOTSTRAP`.
#![cfg_attr(feature = "cuda", feature(f16))]

pub mod cli;
pub mod config;
pub mod error;
pub mod format;
pub mod kernels;
pub mod memory;
pub mod models;
pub mod preprocess;
pub mod profile;
pub mod progress;
#[cfg(feature = "python")]
mod python;

pub use cli::{Command, HelpTopic, ImportArgs, InspectArgs, ModelSource, PlanArgs, parse_cli};
pub use config::{CacheMode, ExecutionConfig, StorageDType, TabIclV2Model, TaskKind, Workload};
pub use error::{Error, Result};
pub use format::model::{
    MODEL_ARTIFACT_MAGIC, MODEL_ARTIFACT_VERSION, ModelArtifact, ModelArtifactSummary, ModelConfig,
    ModelProfile, ModelTensorF32, inspect_model_artifact, load_model_artifact,
};
pub use format::preprocessor::{PREPROCESSOR_MAGIC, PREPROCESSOR_VERSION};
pub use format::static_context::{
    STATIC_CONTEXT_MAGIC, STATIC_CONTEXT_VERSION, StaticContextArtifact, StaticContextMetadata,
    StaticContextTensor, StaticContextTensorData, StaticContextTensorRef,
};
pub use kernels::cpu::{CpuCapabilities, Isa, IsaPreference};
pub use memory::{CacheEstimate, MemoryPlan, estimate_memory};
pub use models::tabicl_v2::checkpoint::{
    CheckpointProfile, CheckpointSummary, import_checkpoint, inspect_checkpoint,
};
pub use models::tabicl_v2::checkpoint_cache::{
    CheckpointCacheOptions, PreparedCheckpoint, load_checkpoint_model, prepare_checkpoint,
};
pub use models::tabicl_v2::cpu::{
    ClassifierAggregation, ClassifierProbabilities, DecoderOutputs, Executor, Optimize, StaticExecutor,
    plan_context,
};
pub use models::tabicl_v2::fitted::{FittedModel, Preprocessed, TrainingInputs};
pub use models::tabicl_v2::preprocess::{Features, NormMethod, PreprocessOptions, Preprocessor, Targets};
pub use models::tabicl_v2::quantiles::RegressionOutput;
pub use models::tabicl_v2::request::{QueryRequest, Request};
pub use preprocess::labels::Labels;
