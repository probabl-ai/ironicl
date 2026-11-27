use std::{fmt, str::FromStr};

use crate::error::{Error, Result};

/// The released `TabICLv2` task variants supported by the design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
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
///
/// This is the storage-dtype vocabulary of the on-disk artifact tensor
/// directories. Model *import* dtypes are a separate concern
/// (`format::model::ModelTensorDType`), since a checkpoint weight may be
/// `i64`/`f64`/`u8` that a cache never stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageDType {
    Fp32,
    Bf16,
    F16,
}

impl StorageDType {
    #[must_use]
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::Fp32 => 4,
            Self::Bf16 | Self::F16 => 2,
        }
    }
}

impl fmt::Display for StorageDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fp32 => f.write_str("fp32"),
            Self::Bf16 => f.write_str("bf16"),
            Self::F16 => f.write_str("f16"),
        }
    }
}
