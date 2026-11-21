use std::{path::PathBuf, str::FromStr};

use crate::{
    CacheMode, ExecutionConfig, IsaPreference, StorageDType, TabIclV2Model, TaskKind, Workload,
    error::{Error, Result},
};

/// A parsed Ironicl command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Help(Option<HelpTopic>),
    Platform,
    Plan(PlanArgs),
    Inspect(InspectArgs),
    Import(ImportArgs),
}

/// Command-specific help pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTopic {
    Platform,
    Plan,
    Inspect,
    Import,
}

/// A model source accepted by commands that can read checkpoints or artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    Checkpoint(PathBuf),
    Model(PathBuf),
}

impl ModelSource {
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        match self {
            Self::Checkpoint(path) | Self::Model(path) => path,
        }
    }
}

/// Arguments for the memory and execution planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanArgs {
    pub config: ExecutionConfig,
    pub bandwidth_gbps: Option<f64>,
}

/// Arguments for model inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectArgs {
    pub source: ModelSource,
}

/// Arguments for converting a `PyTorch` checkpoint into an Ironicl artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportArgs {
    pub checkpoint: PathBuf,
    pub output: PathBuf,
}

/// Parse command-line arguments excluding the executable name.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for unknown commands or options,
/// duplicate options, missing values, and invalid source combinations.
pub fn parse_cli(args: &[String]) -> Result<Command> {
    let Some((command, rest)) = args.split_first() else {
        return Ok(Command::Help(None));
    };

    match command.as_str() {
        "help" | "--help" | "-h" => parse_help(rest),
        "platform" => parse_no_args(rest, HelpTopic::Platform, Command::Platform),
        "plan" => parse_plan(rest),
        "inspect" => parse_inspect(rest),
        "import" => parse_import(rest),
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
        "plan" => Ok(HelpTopic::Plan),
        "inspect" => Ok(HelpTopic::Inspect),
        "import" => Ok(HelpTopic::Import),
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

fn parse_plan(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Plan)));
    }

    let mut task = None;
    let mut training_rows = None;
    let mut features = None;
    let mut estimators = None;
    let mut query_microbatch = None;
    let mut cache_mode = None;
    let mut weight_dtype = None;
    let mut kv_dtype = None;
    let mut isa = None;
    let mut bandwidth_gbps = None;

    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let value = option_value(args, &mut index)?;
        match flag.as_str() {
            "--task" => set_once(&mut task, TaskKind::from_str(value)?, flag)?,
            "--rows" => set_once(&mut training_rows, parse_u64(flag, value)?, flag)?,
            "--features" => set_once(&mut features, parse_u64(flag, value)?, flag)?,
            "--estimators" => set_once(&mut estimators, parse_u64(flag, value)?, flag)?,
            "--microbatch" => set_once(&mut query_microbatch, parse_u32(flag, value)?, flag)?,
            "--cache" => set_once(&mut cache_mode, CacheMode::from_str(value)?, flag)?,
            "--weights" => set_once(&mut weight_dtype, StorageDType::from_str(value)?, flag)?,
            "--kv-dtype" => set_once(&mut kv_dtype, StorageDType::from_str(value)?, flag)?,
            "--isa" => set_once(&mut isa, IsaPreference::from_str(value)?, flag)?,
            "--bandwidth-gbps" => {
                let bandwidth = value
                    .parse::<f64>()
                    .map_err(|error| Error::InvalidArgument(format!("invalid value for {flag}: {error}")))?;
                if !bandwidth.is_finite() || bandwidth <= 0.0 {
                    return Err(Error::InvalidArgument(format!(
                        "{flag} must be finite and greater than zero"
                    )));
                }
                set_once(&mut bandwidth_gbps, bandwidth, flag)?;
            }
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "unknown option '{flag}' for plan"
                )));
            }
        }
    }

    let defaults = ExecutionConfig::default();
    let workload_defaults = defaults.workload;
    let task = task.unwrap_or(defaults.model.task);
    let config = ExecutionConfig {
        model: TabIclV2Model::new(task),
        workload: Workload {
            training_rows: training_rows.unwrap_or(workload_defaults.training_rows),
            features: features.unwrap_or(workload_defaults.features),
            estimators: estimators.unwrap_or(workload_defaults.estimators),
            query_microbatch: query_microbatch.unwrap_or(workload_defaults.query_microbatch),
        },
        cache_mode: cache_mode.unwrap_or(defaults.cache_mode),
        weight_dtype: weight_dtype.unwrap_or(defaults.weight_dtype),
        kv_dtype: kv_dtype.unwrap_or(defaults.kv_dtype),
        isa: isa.unwrap_or(defaults.isa),
    };
    config.validate()?;

    Ok(Command::Plan(PlanArgs {
        config,
        bandwidth_gbps,
    }))
}

fn parse_inspect(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Inspect)));
    }
    let source = parse_model_source(args, "inspect")?;
    Ok(Command::Inspect(InspectArgs { source }))
}

fn parse_import(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Import)));
    }

    let mut checkpoint = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let value = option_value(args, &mut index)?;
        match flag.as_str() {
            "--checkpoint" => set_once(&mut checkpoint, PathBuf::from(value), flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value), flag)?,
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "unknown option '{flag}' for import"
                )));
            }
        }
    }

    Ok(Command::Import(ImportArgs {
        checkpoint: required(checkpoint, "--checkpoint", "import")?,
        output: required(output, "--output", "import")?,
    }))
}

fn parse_model_source(args: &[String], command: &str) -> Result<ModelSource> {
    let mut checkpoint = None;
    let mut model = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let value = option_value(args, &mut index)?;
        match flag.as_str() {
            "--checkpoint" => set_once(&mut checkpoint, PathBuf::from(value), flag)?,
            "--model" => set_once(&mut model, PathBuf::from(value), flag)?,
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "unknown option '{flag}' for {command}"
                )));
            }
        }
    }

    resolve_model_source(checkpoint, model, command)
}

fn resolve_model_source(
    checkpoint: Option<PathBuf>,
    model: Option<PathBuf>,
    command: &str,
) -> Result<ModelSource> {
    match (checkpoint, model) {
        (Some(path), None) => Ok(ModelSource::Checkpoint(path)),
        (None, Some(path)) => Ok(ModelSource::Model(path)),
        (None, None) => Err(Error::InvalidArgument(format!(
            "{command} requires exactly one of --checkpoint or --model"
        ))),
        (Some(_), Some(_)) => Err(Error::InvalidArgument(format!(
            "{command} accepts only one of --checkpoint or --model"
        ))),
    }
}

fn option_value<'a>(args: &'a [String], index: &mut usize) -> Result<&'a str> {
    let flag = args
        .get(*index)
        .ok_or_else(|| Error::InvalidArgument("missing option".into()))?;
    if !flag.starts_with("--") {
        return Err(Error::InvalidArgument(format!(
            "unexpected positional argument '{flag}'"
        )));
    }
    *index += 1;
    let value = args
        .get(*index)
        .ok_or_else(|| Error::InvalidArgument(format!("missing value for {flag}")))?;
    *index += 1;
    Ok(value)
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(Error::InvalidArgument(format!("duplicate option '{flag}'")));
    }
    Ok(())
}

fn required<T>(value: Option<T>, flag: &str, command: &str) -> Result<T> {
    value.ok_or_else(|| Error::InvalidArgument(format!("{command} requires {flag}")))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|error| Error::InvalidArgument(format!("invalid value for {flag}: {error}")))
}

fn parse_u32(flag: &str, value: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .map_err(|error| Error::InvalidArgument(format!("invalid value for {flag}: {error}")))
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
    fn parses_checkpoint_inspection() {
        assert_eq!(
            parse_cli(&args(&["inspect", "--checkpoint", "model.ckpt"])),
            Ok(Command::Inspect(InspectArgs {
                source: ModelSource::Checkpoint(PathBuf::from("model.ckpt")),
            }))
        );
    }

    #[test]
    fn parses_checkpoint_import() {
        assert_eq!(
            parse_cli(&args(&[
                "import",
                "--checkpoint",
                "model.ckpt",
                "--output",
                "model.irmodel",
            ])),
            Ok(Command::Import(ImportArgs {
                checkpoint: PathBuf::from("model.ckpt"),
                output: PathBuf::from("model.irmodel"),
            }))
        );
    }

    #[test]
    fn rejects_ambiguous_model_source() {
        let error = parse_cli(&args(&[
            "inspect",
            "--checkpoint",
            "model.ckpt",
            "--model",
            "model.irmodel",
        ]))
        .expect_err("ambiguous source must fail");
        assert!(error.to_string().contains("only one"));
    }

    #[test]
    fn rejects_duplicate_options() {
        let error = parse_cli(&args(&["plan", "--rows", "100", "--rows", "200"]))
            .expect_err("duplicate option must fail");
        assert!(error.to_string().contains("duplicate option"));
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
