use std::{fmt, str::FromStr};

use crate::{
    backend::BackendPreference,
    error::{Error, Result},
};

/// The released `TabICLv2` task variants supported by the design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Classifier,
    Regressor,
}

impl TaskKind {
    #[must_use]
    pub const fn parameter_count(self) -> u64 {
        match self {
            Self::Classifier => 27_552_258,
            Self::Regressor => 28_544_991,
        }
    }
}

impl fmt::Display for TaskKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Classifier => f.write_str("classifier"),
            Self::Regressor => f.write_str("regressor"),
        }
    }
}

impl FromStr for TaskKind {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "classifier" | "clf" => Ok(Self::Classifier),
            "regressor" | "reg" => Ok(Self::Regressor),
            _ => Err(Error::InvalidArgument(format!(
                "unknown task '{value}'; expected classifier or regressor"
            ))),
        }
    }
}

/// Storage formats. Arithmetic policy is separate and remains FP32 by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDType {
    Fp32,
    Bf16,
    Int8,
}

impl StorageDType {
    #[must_use]
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::Fp32 => 4,
            Self::Bf16 => 2,
            Self::Int8 => 1,
        }
    }
}

impl fmt::Display for StorageDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fp32 => f.write_str("fp32"),
            Self::Bf16 => f.write_str("bf16"),
            Self::Int8 => f.write_str("int8"),
        }
    }
}

impl FromStr for StorageDType {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "fp32" | "f32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            "int8" | "i8" => Ok(Self::Int8),
            _ => Err(Error::InvalidArgument(format!(
                "unknown storage dtype '{value}'; expected fp32, bf16, or int8"
            ))),
        }
    }
}

/// Training-context cache strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Cache K/V for every ICL layer. This is large but minimizes repeated work.
    Kv,
    /// Cache row representations and rerun the ICL stack for each request.
    Representation,
}

impl fmt::Display for CacheMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kv => f.write_str("kv"),
            Self::Representation => f.write_str("repr"),
        }
    }
}

impl FromStr for CacheMode {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "kv" | "full-kv" => Ok(Self::Kv),
            "repr" | "representation" => Ok(Self::Representation),
            _ => Err(Error::InvalidArgument(format!(
                "unknown cache mode '{value}'; expected kv or repr"
            ))),
        }
    }
}

/// Released `TabICLv2` architecture constants used by cache planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabIclV2Model {
    pub task: TaskKind,
    pub embed_dim: u64,
    pub column_blocks: u64,
    pub column_inducing_points: u64,
    pub row_cls_tokens: u64,
    pub icl_blocks: u64,
}

impl TabIclV2Model {
    #[must_use]
    pub const fn classifier() -> Self {
        Self::new(TaskKind::Classifier)
    }

    #[must_use]
    pub const fn regressor() -> Self {
        Self::new(TaskKind::Regressor)
    }

    #[must_use]
    pub const fn new(task: TaskKind) -> Self {
        Self {
            task,
            embed_dim: 128,
            column_blocks: 3,
            column_inducing_points: 128,
            row_cls_tokens: 4,
            icl_blocks: 12,
        }
    }

    #[must_use]
    pub const fn icl_dim(self) -> u64 {
        self.embed_dim * self.row_cls_tokens
    }

    #[must_use]
    pub const fn parameter_count(self) -> u64 {
        self.task.parameter_count()
    }
}

/// Data dimensions and request batching used to size a static context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workload {
    pub training_rows: u64,
    pub features: u64,
    pub estimators: u64,
    pub query_microbatch: u32,
}

impl Default for Workload {
    fn default() -> Self {
        Self {
            training_rows: 50_000,
            features: 100,
            estimators: 8,
            query_microbatch: 1,
        }
    }
}

impl Workload {
    /// Validate non-zero workload dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] when any dimension is zero.
    pub fn validate(self) -> Result<()> {
        if self.training_rows == 0 {
            return Err(Error::InvalidConfig(
                "training_rows must be greater than zero".into(),
            ));
        }
        if self.features == 0 {
            return Err(Error::InvalidConfig("features must be greater than zero".into()));
        }
        if self.estimators == 0 {
            return Err(Error::InvalidConfig(
                "estimators must be greater than zero".into(),
            ));
        }
        if self.query_microbatch == 0 {
            return Err(Error::InvalidConfig(
                "query_microbatch must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Top-level execution policy used to create an engine plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionConfig {
    pub model: TabIclV2Model,
    pub workload: Workload,
    pub cache_mode: CacheMode,
    pub weight_dtype: StorageDType,
    pub kv_dtype: StorageDType,
    pub backend: BackendPreference,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            model: TabIclV2Model::classifier(),
            workload: Workload::default(),
            cache_mode: CacheMode::Kv,
            weight_dtype: StorageDType::Bf16,
            kv_dtype: StorageDType::Bf16,
            backend: BackendPreference::Auto,
        }
    }
}

impl ExecutionConfig {
    /// Validate the workload and model-specific cache assumptions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] when the workload or model contract is
    /// not supported.
    pub fn validate(self) -> Result<()> {
        self.workload.validate()?;

        if self.model.task == TaskKind::Classifier && self.model.row_cls_tokens != 4 {
            return Err(Error::InvalidConfig(
                "the released classifier cache contract requires four row CLS tokens".into(),
            ));
        }
        Ok(())
    }
}
