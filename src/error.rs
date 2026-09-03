use std::fmt;

/// Errors produced while validating or planning an Ironicl execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidConfig(String),
    InvalidArgument(String),
    Artifact(String),
    Cache(String),
    Checkpoint(String),
    InputBundle(String),
    NotImplemented(String),
    UnsupportedBackend(String),
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::Artifact(message) => write!(f, "model artifact error: {message}"),
            Self::Cache(message) => write!(f, "model cache error: {message}"),
            Self::Checkpoint(message) => write!(f, "checkpoint error: {message}"),
            Self::InputBundle(message) => write!(f, "input bundle error: {message}"),
            Self::NotImplemented(message) => write!(f, "not implemented: {message}"),
            Self::UnsupportedBackend(message) => write!(f, "unsupported backend: {message}"),
            Self::ArithmeticOverflow(name) => write!(f, "arithmetic overflow while calculating {name}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
