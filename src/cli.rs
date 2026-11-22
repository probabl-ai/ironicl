use std::{net::SocketAddr, path::PathBuf, time::Duration};

use crate::{
    Optimize,
    error::{Error, Result},
};

/// A parsed Ironicl command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Help(Option<HelpTopic>),
    Platform,
    Inspect(InspectArgs),
    Import(ImportArgs),
    Prepare(PrepareArgs),
    Serve(ServeArgs),
}

/// Command-specific help pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTopic {
    Platform,
    Inspect,
    Import,
    Prepare,
    Serve,
}

/// Arguments for inspecting a model directory, an Ironicl model, or a
/// checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectArgs {
    pub path: PathBuf,
    /// The weights folder a model directory's weights are looked for in.
    pub weights: Option<PathBuf>,
}

/// Arguments for converting a `PyTorch` checkpoint into Ironicl weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportArgs {
    pub checkpoint: PathBuf,
    /// Where to write the weights; by default, into the shared weights folder.
    pub output: Option<PathBuf>,
}

/// Arguments for building a model directory's KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareArgs {
    pub dir: PathBuf,
    /// The weights folder; by default, the shared one.
    pub weights: Option<PathBuf>,
    /// The cache's level; by default, the saved cache's, else `safe`.
    pub optimize: Option<Optimize>,
}

/// Arguments for serving a model directory over HTTP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeArgs {
    pub dir: PathBuf,
    pub weights: Option<PathBuf>,
    pub optimize: Option<Optimize>,
    pub bind: SocketAddr,
    pub max_request_rows: usize,
    pub max_body: usize,
    /// Rows predicted together at most; by default, by device.
    pub max_batch_rows: Option<usize>,
    /// How long a request may wait for its batch before it is refused.
    pub max_queue: Duration,
}

/// Parse command-line arguments excluding the executable name.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for unknown commands or options,
/// duplicate options, missing or invalid values, and a missing path.
pub fn parse_cli(args: &[String]) -> Result<Command> {
    let Some((command, rest)) = args.split_first() else {
        return Ok(Command::Help(None));
    };

    match command.as_str() {
        "help" | "--help" | "-h" => parse_help(rest),
        "platform" => parse_no_args(rest, HelpTopic::Platform, Command::Platform),
        "inspect" => parse_inspect(rest),
        "import" => parse_import(rest),
        "prepare" => parse_prepare(rest),
        "serve" => parse_serve(rest),
        _ => Err(Error::InvalidArgument(format!("unknown command '{command}'"))),
    }
}

fn parse_help(args: &[String]) -> Result<Command> {
    match args {
        [] => Ok(Command::Help(None)),
        [topic] => Ok(Command::Help(Some(parse_help_topic(topic)?))),
        _ => Err(Error::InvalidArgument(
            "help accepts at most one command name".into(),
        )),
    }
}

fn parse_help_topic(value: &str) -> Result<HelpTopic> {
    match value {
        "platform" => Ok(HelpTopic::Platform),
        "inspect" => Ok(HelpTopic::Inspect),
        "import" => Ok(HelpTopic::Import),
        "prepare" => Ok(HelpTopic::Prepare),
        "serve" => Ok(HelpTopic::Serve),
        _ => Err(Error::InvalidArgument(format!("unknown help topic '{value}'"))),
    }
}

fn parse_no_args(args: &[String], topic: HelpTopic, command: Command) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(topic)));
    }
    if let Some(argument) = args.first() {
        return Err(Error::InvalidArgument(format!(
            "command does not accept argument '{argument}'"
        )));
    }
    Ok(command)
}

fn parse_inspect(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Inspect)));
    }
    let (path, rest) = path(args, "inspect", "a path")?;
    let mut weights = None;
    for (flag, value) in options(rest)? {
        match flag {
            "--weights" => set_once(&mut weights, PathBuf::from(value), flag)?,
            _ => return Err(unknown(flag, "inspect")),
        }
    }
    Ok(Command::Inspect(InspectArgs { path, weights }))
}

fn parse_import(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Import)));
    }
    let (checkpoint, rest) = path(args, "import", "a checkpoint")?;
    let mut output = None;
    for (flag, value) in options(rest)? {
        match flag {
            "-o" | "--output" => set_once(&mut output, PathBuf::from(value), flag)?,
            _ => return Err(unknown(flag, "import")),
        }
    }
    Ok(Command::Import(ImportArgs { checkpoint, output }))
}

fn parse_prepare(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Prepare)));
    }
    let (dir, rest) = path(args, "prepare", "a model directory")?;
    let (mut weights, mut optimize) = (None, None);
    for (flag, value) in options(rest)? {
        match flag {
            "--weights" => set_once(&mut weights, PathBuf::from(value), flag)?,
            "--optimize" => set_once(&mut optimize, value.parse()?, flag)?,
            _ => return Err(unknown(flag, "prepare")),
        }
    }
    Ok(Command::Prepare(PrepareArgs {
        dir,
        weights,
        optimize,
    }))
}

fn parse_serve(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Serve)));
    }
    let (dir, rest) = path(args, "serve", "a model directory")?;
    let (mut weights, mut optimize, mut bind) = (None, None, None);
    let (mut max_request_rows, mut max_body, mut max_batch_rows, mut max_queue) = (None, None, None, None);
    for (flag, value) in options(rest)? {
        let invalid = || Error::InvalidArgument(format!("invalid value '{value}' for {flag}"));
        match flag {
            "--weights" => set_once(&mut weights, PathBuf::from(value), flag)?,
            "--optimize" => set_once(&mut optimize, value.parse()?, flag)?,
            "--bind" => set_once(&mut bind, value.parse().map_err(|_| invalid())?, flag)?,
            "--max-request-rows" => {
                set_once(&mut max_request_rows, positive(value).ok_or_else(invalid)?, flag)?;
            }
            "--max-body" => set_once(&mut max_body, crate::memory::parse_bytes(value)?, flag)?,
            "--max-batch-rows" => set_once(&mut max_batch_rows, positive(value).ok_or_else(invalid)?, flag)?,
            "--max-queue" => {
                let milliseconds = positive(value).ok_or_else(invalid)?;
                set_once(&mut max_queue, Duration::from_millis(milliseconds as u64), flag)?;
            }
            _ => return Err(unknown(flag, "serve")),
        }
    }
    Ok(Command::Serve(ServeArgs {
        dir,
        weights,
        optimize,
        bind: bind.unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 8080))),
        max_request_rows: max_request_rows.unwrap_or(10_000),
        max_body: usize::try_from(max_body.unwrap_or(16 << 20))
            .map_err(|_| Error::InvalidArgument("--max-body is too large".into()))?,
        max_batch_rows,
        max_queue: max_queue.unwrap_or(Duration::from_secs(1)),
    }))
}

fn positive(value: &str) -> Option<usize> {
    value.parse().ok().filter(|&value| value > 0)
}

/// The path that leads `args`, and the options after it.
fn path<'a>(args: &'a [String], command: &str, what: &str) -> Result<(PathBuf, &'a [String])> {
    match args.split_first() {
        Some((path, rest)) if !path.starts_with('-') => Ok((PathBuf::from(path), rest)),
        _ => Err(Error::InvalidArgument(format!("{command} requires {what}"))),
    }
}

/// `args` as `(flag, value)` pairs.
fn options(args: &[String]) -> Result<Vec<(&str, &str)>> {
    args.chunks(2)
        .map(|pair| match pair {
            [flag, value] if flag.starts_with('-') => Ok((flag.as_str(), value.as_str())),
            [flag] if flag.starts_with('-') => {
                Err(Error::InvalidArgument(format!("missing value for {flag}")))
            }
            [argument, ..] => Err(Error::InvalidArgument(format!(
                "unexpected positional argument '{argument}'"
            ))),
            [] => unreachable!("chunks are never empty"),
        })
        .collect()
}

fn unknown(flag: &str, command: &str) -> Error {
    Error::InvalidArgument(format!("unknown option '{flag}' for {command}"))
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(Error::InvalidArgument(format!("duplicate option '{flag}'")));
    }
    Ok(())
}

fn is_help(args: &[String]) -> bool {
    matches!(args, [value] if value == "--help" || value == "-h")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_inspect_and_import() {
        assert_eq!(
            parse_cli(&args(&["inspect", "my-model", "--weights", "w"])),
            Ok(Command::Inspect(InspectArgs {
                path: PathBuf::from("my-model"),
                weights: Some(PathBuf::from("w")),
            }))
        );
        assert_eq!(
            parse_cli(&args(&["import", "model.ckpt"])),
            Ok(Command::Import(ImportArgs {
                checkpoint: PathBuf::from("model.ckpt"),
                output: None,
            }))
        );
        assert_eq!(
            parse_cli(&args(&["import", "model.ckpt", "-o", "model.irmodel"])),
            Ok(Command::Import(ImportArgs {
                checkpoint: PathBuf::from("model.ckpt"),
                output: Some(PathBuf::from("model.irmodel")),
            }))
        );
        for bad in [
            &["inspect"][..],
            &["inspect", "--checkpoint", "model.ckpt"],
            &["import", "model.ckpt", "-o"],
            &["import", "model.ckpt", "-o", "a", "-o", "b"],
            &["import", "model.ckpt", "extra"],
        ] {
            assert!(parse_cli(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn parses_prepare() {
        assert_eq!(
            parse_cli(&args(&["prepare", "my-model"])),
            Ok(Command::Prepare(PrepareArgs {
                dir: PathBuf::from("my-model"),
                weights: None,
                optimize: None,
            }))
        );
        assert_eq!(
            parse_cli(&args(&[
                "prepare",
                "my-model",
                "--optimize",
                "auto",
                "--weights",
                "w"
            ])),
            Ok(Command::Prepare(PrepareArgs {
                dir: PathBuf::from("my-model"),
                weights: Some(PathBuf::from("w")),
                optimize: Some(Optimize::Auto),
            }))
        );
        assert!(parse_cli(&args(&["prepare", "--optimize", "auto"])).is_err());
        assert!(parse_cli(&args(&["prepare", "my-model", "--optimize", "max"])).is_err());
    }

    #[test]
    fn parses_serve() {
        let serve = |extra: &[&str]| parse_cli(&args(&[&["serve", "my-model"], extra].concat()));
        assert_eq!(
            serve(&[]),
            Ok(Command::Serve(ServeArgs {
                dir: PathBuf::from("my-model"),
                weights: None,
                optimize: None,
                bind: SocketAddr::from(([127, 0, 0, 1], 8080)),
                max_request_rows: 10_000,
                max_body: 16 << 20,
                max_batch_rows: None,
                max_queue: Duration::from_secs(1),
            }))
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
            ]),
            Ok(Command::Serve(ServeArgs {
                dir: PathBuf::from("my-model"),
                weights: Some(PathBuf::from("w")),
                optimize: Some(Optimize::Fast),
                bind: SocketAddr::from(([0, 0, 0, 0], 9000)),
                max_request_rows: 5,
                max_body: 1 << 20,
                max_batch_rows: Some(64),
                max_queue: Duration::from_millis(10),
            }))
        );
        for bad in [
            &["--max-request-rows", "0"][..],
            &["--max-batch-rows", "0"],
            &["--max-queue", "0"],
            &["--bind", "nowhere"],
            &["--max-request-rows", "1", "--max-request-rows", "2"],
            &["--max-rows", "5"],
            &["--port", "1"],
        ] {
            assert!(serve(bad).is_err(), "{bad:?}");
        }
        assert!(parse_cli(&args(&["serve", "--bind", "0.0.0.0:9000"])).is_err());
    }

    #[test]
    fn parses_command_help() {
        assert_eq!(
            parse_cli(&args(&["inspect", "--help"])),
            Ok(Command::Help(Some(HelpTopic::Inspect)))
        );
        assert_eq!(
            parse_cli(&args(&["help", "import"])),
            Ok(Command::Help(Some(HelpTopic::Import)))
        );
    }
}
