//! The `ironicl` command line, parsed with clap. Each command's and option's
//! help is its doc comment.
#![expect(
    clippy::doc_markdown,
    reason = "the doc comments are the terminal help, where backticks show"
)]

use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::{Args, Parser, Subcommand, ValueEnum, builder::RangedU64ValueParser};

use crate::{CheckpointKind, Optimize};

/// Ironicl: TabICLv2 and Kumo-Tabular inference without Python.
#[derive(Debug, Parser)]
#[command(name = "ironicl", version, arg_required_else_help = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, PartialEq, Subcommand)]
pub enum Command {
    /// Show the CPU features the kernels detect.
    Platform,
    Inspect(InspectArgs),
    Import(ImportArgs),
    Prepare(PrepareArgs),
    Serve(ServeArgs),
}

/// The model a checkpoint is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Model {
    /// TabICLv2
    Tabicl,
    /// Kumo-Tabular, any size and task
    KumoTabular,
}

impl Model {
    /// A Kumo-Tabular checkpoint's size and task are those of its tensors.
    #[must_use]
    pub fn checkpoint_kind(self) -> CheckpointKind {
        match self {
            Self::Tabicl => CheckpointKind::TabIclV2,
            Self::KumoTabular => CheckpointKind::KumoTabular(None),
        }
    }
}

/// Inspect a model directory, an Ironicl artifact (.ir), or a PyTorch
/// checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct InspectArgs {
    /// A model directory, an Ironicl artifact, or a checkpoint.
    pub path: PathBuf,
    /// The model the checkpoint is [default: found from its contents]
    #[arg(long)]
    pub model: Option<Model>,
    /// The folder a model directory's weights are looked for in [default: the
    /// shared one]
    #[arg(long, value_name = "DIR")]
    pub weights: Option<PathBuf>,
}

/// Convert a PyTorch checkpoint to Ironicl weights.
///
/// By default they go into the shared weights folder,
/// ~/.cache/ironicl/models/v1 (under XDG_CACHE_HOME when set), where prepare
/// and serve look for them.
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct ImportArgs {
    /// The checkpoint file.
    pub checkpoint: PathBuf,
    /// The model the checkpoint is.
    #[arg(long, value_enum, default_value_t = Model::Tabicl)]
    pub model: Model,
    /// Write the weights to FILE instead.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

/// Build a model directory's KV cache.
///
/// It builds the cache on this machine's device and writes it to
/// DIR/context.ircntxt, unless the saved cache is already the one this model
/// builds at the level. A saved cache that cannot be read is rebuilt, with a
/// warning.
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct PrepareArgs {
    /// The model directory.
    pub dir: PathBuf,
    /// The folder of weights [default: the shared one]
    #[arg(long, value_name = "DIR")]
    pub weights: Option<PathBuf>,
    /// safe, auto, or fast [default: the saved cache's level, else the exported fit's]
    #[arg(long, value_name = "LEVEL", value_parser = level)]
    pub optimize: Option<Optimize>,
}

/// Serve a model directory over HTTP.
///
/// It serves POST /v1/predict and GET /v1/health, after preparing the KV cache
/// as ironicl prepare does. Concurrent requests are predicted together, up to
/// --max-batch-rows rows at once; a request that would wait longer than
/// --max-queue is refused with 503.
#[derive(Debug, Clone, PartialEq, Eq, Args)]
pub struct ServeArgs {
    /// The model directory.
    pub dir: PathBuf,
    /// The folder of weights [default: the shared one]
    #[arg(long, value_name = "DIR")]
    pub weights: Option<PathBuf>,
    /// safe, auto, or fast [default: the saved cache's level, else the exported fit's]
    #[arg(long, value_name = "LEVEL", value_parser = level)]
    pub optimize: Option<Optimize>,
    /// The address to listen on.
    #[arg(long, value_name = "ADDR", default_value = "127.0.0.1:8080")]
    pub bind: SocketAddr,
    /// The most rows a request may have.
    #[arg(long, value_name = "N", default_value = "10000", value_parser = positive())]
    pub max_request_rows: usize,
    /// The largest request body, such as 512K or 16M.
    #[arg(long, value_name = "SIZE", default_value = "16M", value_parser = bytes)]
    pub max_body: usize,
    /// The most rows predicted together [default: 1024 on a GPU, 32 on the CPU]
    #[arg(long, value_name = "N", value_parser = positive())]
    pub max_batch_rows: Option<usize>,
    /// How long, in milliseconds, a request may wait for its batch.
    #[arg(long, value_name = "MS", default_value = "1000", value_parser = milliseconds)]
    pub max_queue: Duration,
}

fn level(value: &str) -> Result<Optimize, String> {
    value
        .parse()
        .map_err(|_| "expected safe, auto, or fast".to_owned())
}

fn positive() -> RangedU64ValueParser<usize> {
    RangedU64ValueParser::new().range(1..)
}

fn bytes(value: &str) -> Result<usize, String> {
    let bytes = crate::memory::parse_bytes(value).map_err(|error| error.to_string())?;
    usize::try_from(bytes).map_err(|_| "too large".to_owned())
}

fn milliseconds(value: &str) -> Result<Duration, String> {
    match value.parse() {
        Ok(0) | Err(_) => Err("expected a positive number of milliseconds".to_owned()),
        Ok(milliseconds) => Ok(Duration::from_millis(milliseconds)),
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    fn parse(args: &[&str]) -> Result<Command, clap::Error> {
        Cli::try_parse_from([&["ironicl"], args].concat()).map(|cli| cli.command)
    }

    #[test]
    fn the_command_line_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_inspect_and_import() {
        assert_eq!(
            parse(&["inspect", "my-model", "--weights", "w"]).unwrap(),
            Command::Inspect(InspectArgs {
                path: PathBuf::from("my-model"),
                model: None,
                weights: Some(PathBuf::from("w")),
            })
        );
        assert_eq!(
            parse(&["inspect", "large-classifier.pt", "--model", "kumo-tabular"]).unwrap(),
            Command::Inspect(InspectArgs {
                path: PathBuf::from("large-classifier.pt"),
                model: Some(Model::KumoTabular),
                weights: None,
            })
        );
        assert_eq!(
            parse(&["import", "model.ckpt", "--model", "tabicl", "-o", "model.ir"]).unwrap(),
            Command::Import(ImportArgs {
                checkpoint: PathBuf::from("model.ckpt"),
                model: Model::Tabicl,
                output: Some(PathBuf::from("model.ir")),
            })
        );
        assert_eq!(
            parse(&["import", "model.ckpt"]).unwrap(),
            parse(&["import", "model.ckpt", "--model", "tabicl"]).unwrap()
        );
        for bad in [
            &["inspect"][..],
            &["inspect", "--checkpoint", "model.ckpt"],
            &["import", "model.ckpt", "--model", "tabicl", "-o"],
            &["import", "model.ckpt", "--model", "tabicl", "-o", "a", "-o", "b"],
            &["import", "model.ckpt", "--model", "tabicl", "extra"],
            &["import", "model.pt", "--model", "kumo-tabular/large-classifier"],
            &["import", "model.pt", "--kumo-tabular", "large-classifier"],
            &["plan"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn parses_prepare() {
        assert_eq!(
            parse(&["prepare", "my-model", "--optimize", "auto", "--weights", "w"]).unwrap(),
            Command::Prepare(PrepareArgs {
                dir: PathBuf::from("my-model"),
                weights: Some(PathBuf::from("w")),
                optimize: Some(Optimize::Auto),
            })
        );
        assert!(parse(&["prepare", "--optimize", "auto"]).is_err());
        assert!(parse(&["prepare", "my-model", "--optimize", "max"]).is_err());
    }

    #[test]
    fn parses_serve() {
        let serve = |extra: &[&str]| parse(&[&["serve", "my-model"], extra].concat());
        assert_eq!(
            serve(&[]).unwrap(),
            Command::Serve(ServeArgs {
                dir: PathBuf::from("my-model"),
                weights: None,
                optimize: None,
                bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
                max_request_rows: 10_000,
                max_body: 16 << 20,
                max_batch_rows: None,
                max_queue: Duration::from_secs(1),
            })
        );
        assert_eq!(
            serve(&[
                "--weights",
                "w",
                "--optimize",
                "fast",
                "--bind",
                "0.0.0.0:9000",
                "--max-request-rows",
                "5",
                "--max-body",
                "1M",
                "--max-batch-rows",
                "64",
                "--max-queue",
                "10",
            ])
            .unwrap(),
            Command::Serve(ServeArgs {
                dir: PathBuf::from("my-model"),
                weights: Some(PathBuf::from("w")),
                optimize: Some(Optimize::Fast),
                bind: SocketAddr::from(([0, 0, 0, 0], 9000)),
                max_request_rows: 5,
                max_body: 1 << 20,
                max_batch_rows: Some(64),
                max_queue: Duration::from_millis(10),
            })
        );
        for bad in [
            &["--max-request-rows", "0"][..],
            &["--max-batch-rows", "0"],
            &["--max-queue", "0"],
            &["--max-body", "lots"],
            &["--bind", "nowhere"],
            &["--max-request-rows", "1", "--max-request-rows", "2"],
            &["--max-rows", "5"],
        ] {
            assert!(serve(bad).is_err(), "{bad:?}");
        }
        assert!(parse(&["serve", "--bind", "0.0.0.0:9000"]).is_err());
    }
}
