use std::{path::PathBuf, str::FromStr};

use crate::{
    BackendPreference, CacheMode, ExecutionConfig, StorageDType, TabIclV2Model, TaskKind, Workload,
    error::{Error, Result},
};

/// A parsed `IronICL` command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Help(Option<HelpTopic>),
    Platform,
    Plan(PlanArgs),
    Inspect(InspectArgs),
    Import(ImportArgs),
    Run(RunArgs),
    Benchmark(BenchmarkArgs),
    BuildContext(BuildContextArgs),
}

/// Command-specific help pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpTopic {
    Platform,
    Plan,
    Inspect,
    Import,
    Run,
    Benchmark,
    BuildContext,
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

/// Arguments for native static-context construction from an `IRINPUT` bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildContextArgs {
    pub model: PathBuf,
    pub input: PathBuf,
    pub output: PathBuf,
    pub kv_dtype: crate::KvDtype,
    pub pack_keys: bool,
    pub index_clusters: Option<usize>,
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

/// Arguments for converting a `PyTorch` checkpoint into an `IronICL` artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportArgs {
    pub checkpoint: PathBuf,
    pub output: PathBuf,
}

/// Arguments for direct development execution or optimized artifact execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunArgs {
    pub source: ModelSource,
    pub input: PathBuf,
    pub context: Option<PathBuf>,
    pub cache: CheckpointCacheOptions,
    pub trace_stages: bool,
    pub probabilities: bool,
}

/// Arguments for a resident-model inference benchmark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkArgs {
    pub model: PathBuf,
    pub input: PathBuf,
    pub context: Option<PathBuf>,
    pub probabilities: bool,
    pub warmup: u32,
    pub iterations: u32,
}

/// Controls automatic checkpoint import for `run`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CheckpointCacheOptions {
    pub no_cache: bool,
    pub rebuild_cache: bool,
    pub cache_dir: Option<PathBuf>,
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
        "build-context" => parse_build_context(rest),
        "plan" => parse_plan(rest),
        "inspect" => parse_inspect(rest),
        "import" => parse_import(rest),
        "run" => parse_run(rest),
        "benchmark" => parse_benchmark(rest),
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
        "run" => Ok(HelpTopic::Run),
        "benchmark" => Ok(HelpTopic::Benchmark),
        "build-context" => Ok(HelpTopic::BuildContext),
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

#[allow(clippy::too_many_lines)]
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
    let mut backend = None;
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
            "--backend" => set_once(&mut backend, BackendPreference::from_str(value)?, flag)?,
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
        backend: backend.unwrap_or(defaults.backend),
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

fn parse_build_context(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::BuildContext)));
    }

    let mut model = None;
    let mut input = None;
    let mut output = None;
    let mut kv_dtype = None;
    let mut pack_keys = false;
    let mut index_clusters = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        if flag == "--pack-keys" {
            pack_keys = true;
            index += 1;
            continue;
        }
        let value = option_value(args, &mut index)?;
        match flag.as_str() {
            "--model" => set_once(&mut model, PathBuf::from(value), flag)?,
            "--input" => set_once(&mut input, PathBuf::from(value), flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value), flag)?,
            "--kv-dtype" => set_once(&mut kv_dtype, value.parse::<crate::KvDtype>()?, flag)?,
            "--index-clusters" => {
                let clusters = usize::try_from(parse_u64(flag, value)?)
                    .map_err(|_| Error::InvalidArgument("--index-clusters exceeds usize".into()))?;
                if clusters < 2 {
                    return Err(Error::InvalidArgument(
                        "--index-clusters must be at least 2".into(),
                    ));
                }
                set_once(&mut index_clusters, clusters, flag)?;
            }
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "unknown option '{flag}' for build-context"
                )));
            }
        }
    }

    Ok(Command::BuildContext(BuildContextArgs {
        model: required(model, "--model", "build-context")?,
        input: required(input, "--input", "build-context")?,
        output: required(output, "--output", "build-context")?,
        kv_dtype: kv_dtype.unwrap_or_default(),
        pack_keys,
        index_clusters,
    }))
}

fn parse_run(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Run)));
    }
    let mut checkpoint = None;
    let mut model = None;
    let mut input = None;
    let mut context = None;
    let mut no_cache = false;
    let mut rebuild_cache = false;
    let mut cache_dir = None;
    let mut trace_stages = false;
    let mut probabilities = false;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        match flag.as_str() {
            "--no-cache" => set_flag(&mut no_cache, flag, &mut index)?,
            "--rebuild-cache" => set_flag(&mut rebuild_cache, flag, &mut index)?,
            "--trace-stages" => set_flag(&mut trace_stages, flag, &mut index)?,
            "--probabilities" => set_flag(&mut probabilities, flag, &mut index)?,
            "--checkpoint" | "--model" | "--input" | "--context" | "--cache-dir" => {
                let value = option_value(args, &mut index)?;
                match flag.as_str() {
                    "--checkpoint" => set_once(&mut checkpoint, PathBuf::from(value), flag)?,
                    "--model" => set_once(&mut model, PathBuf::from(value), flag)?,
                    "--input" => set_once(&mut input, PathBuf::from(value), flag)?,
                    "--context" => set_once(&mut context, PathBuf::from(value), flag)?,
                    "--cache-dir" => set_once(&mut cache_dir, PathBuf::from(value), flag)?,
                    _ => unreachable!(),
                }
            }
            _ => {
                return Err(Error::InvalidArgument(format!("unknown option '{flag}' for run")));
            }
        }
    }
    let source = resolve_model_source(checkpoint, model, "run")?;
    if no_cache && rebuild_cache {
        return Err(Error::InvalidArgument(
            "--no-cache and --rebuild-cache cannot be used together".into(),
        ));
    }
    if no_cache && cache_dir.is_some() {
        return Err(Error::InvalidArgument(
            "--cache-dir cannot be used with --no-cache".into(),
        ));
    }
    let input = required(input, "--input", "run")?;
    if no_cache {
        return Err(Error::InvalidArgument(
            "run cannot use --no-cache with --input because execution requires an imported model artifact"
                .into(),
        ));
    }
    if matches!(source, ModelSource::Model(_)) && (no_cache || rebuild_cache || cache_dir.is_some()) {
        return Err(Error::InvalidArgument(
            "checkpoint cache options require --checkpoint".into(),
        ));
    }
    if trace_stages && context.is_some() {
        return Err(Error::InvalidArgument(
            "--trace-stages is unavailable with --context".into(),
        ));
    }
    if probabilities && context.is_none() {
        return Err(Error::InvalidArgument(
            "--probabilities requires --context".into(),
        ));
    }
    Ok(Command::Run(RunArgs {
        source,
        input,
        context,
        cache: CheckpointCacheOptions {
            no_cache,
            rebuild_cache,
            cache_dir,
        },
        trace_stages,
        probabilities,
    }))
}

fn parse_benchmark(args: &[String]) -> Result<Command> {
    if is_help(args) {
        return Ok(Command::Help(Some(HelpTopic::Benchmark)));
    }
    let mut model = None;
    let mut input = None;
    let mut context = None;
    let mut probabilities = false;
    let mut warmup = None;
    let mut iterations = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        match flag.as_str() {
            "--probabilities" => set_flag(&mut probabilities, flag, &mut index)?,
            "--model" | "--input" | "--context" | "--warmup" | "--iterations" => {
                let value = option_value(args, &mut index)?;
                match flag.as_str() {
                    "--model" => set_once(&mut model, PathBuf::from(value), flag)?,
                    "--input" => set_once(&mut input, PathBuf::from(value), flag)?,
                    "--context" => set_once(&mut context, PathBuf::from(value), flag)?,
                    "--warmup" => set_once(&mut warmup, parse_u32(flag, value)?, flag)?,
                    "--iterations" => set_once(&mut iterations, parse_u32(flag, value)?, flag)?,
                    _ => unreachable!(),
                }
            }
            _ => {
                return Err(Error::InvalidArgument(format!(
                    "unknown option '{flag}' for benchmark"
                )));
            }
        }
    }
    let warmup = warmup.unwrap_or(2);
    if warmup > 1_000 {
        return Err(Error::InvalidArgument("--warmup must be at most 1000".into()));
    }
    let iterations = iterations.unwrap_or(10);
    if !(1..=10_000).contains(&iterations) {
        return Err(Error::InvalidArgument("--iterations must be in 1..=10000".into()));
    }
    if probabilities && context.is_none() {
        return Err(Error::InvalidArgument(
            "benchmark --probabilities requires --context".into(),
        ));
    }
    Ok(Command::Benchmark(BenchmarkArgs {
        model: required(model, "--model", "benchmark")?,
        input: required(input, "--input", "benchmark")?,
        context,
        probabilities,
        warmup,
        iterations,
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

fn set_flag(slot: &mut bool, flag: &str, index: &mut usize) -> Result<()> {
    if *slot {
        return Err(Error::InvalidArgument(format!("duplicate option '{flag}'")));
    }
    *slot = true;
    *index += 1;
    Ok(())
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
    fn parses_build_context() {
        assert_eq!(
            parse_cli(&args(&[
                "build-context",
                "--model",
                "m.ironicl",
                "--input",
                "train.irinput",
                "--output",
                "ctx.ironiclctx",
                "--kv-dtype",
                "bf16-v",
                "--pack-keys",
                "--index-clusters",
                "16",
            ])),
            Ok(Command::BuildContext(BuildContextArgs {
                model: PathBuf::from("m.ironicl"),
                input: PathBuf::from("train.irinput"),
                output: PathBuf::from("ctx.ironiclctx"),
                kv_dtype: crate::KvDtype::Bf16Value,
                pack_keys: true,
                index_clusters: Some(16),
            }))
        );
        assert_eq!(
            parse_cli(&args(&[
                "build-context",
                "--model",
                "m.ironicl",
                "--input",
                "train.irinput",
                "--output",
                "ctx.ironiclctx",
            ])),
            Ok(Command::BuildContext(BuildContextArgs {
                model: PathBuf::from("m.ironicl"),
                input: PathBuf::from("train.irinput"),
                output: PathBuf::from("ctx.ironiclctx"),
                kv_dtype: crate::KvDtype::Fp32,
                pack_keys: false,
                index_clusters: None,
            }))
        );
        assert!(parse_cli(&args(&["build-context", "--model", "m.ironicl"])).is_err());
        assert!(
            parse_cli(&args(&[
                "build-context",
                "--model",
                "m.ironicl",
                "--input",
                "a",
                "--output",
                "b",
                "--kv-dtype",
                "bf16-k",
            ]))
            .is_err()
        );
        assert!(
            parse_cli(&args(&[
                "build-context",
                "--model",
                "m.ironicl",
                "--input",
                "a",
                "--output",
                "b",
                "--index-clusters",
                "1",
            ]))
            .is_err()
        );
        assert_eq!(
            parse_cli(&args(&["build-context", "--help"])),
            Ok(Command::Help(Some(HelpTopic::BuildContext)))
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
                "model.ironicl",
            ])),
            Ok(Command::Import(ImportArgs {
                checkpoint: PathBuf::from("model.ckpt"),
                output: PathBuf::from("model.ironicl"),
            }))
        );
    }

    #[test]
    fn parses_both_run_sources() {
        assert_eq!(
            parse_cli(&args(&[
                "run",
                "--checkpoint",
                "model.ckpt",
                "--input",
                "request.irinput"
            ])),
            Ok(Command::Run(RunArgs {
                source: ModelSource::Checkpoint(PathBuf::from("model.ckpt")),
                input: PathBuf::from("request.irinput"),
                context: None,
                cache: CheckpointCacheOptions::default(),
                trace_stages: false,
                probabilities: false,
            }))
        );
        assert_eq!(
            parse_cli(&args(&[
                "run",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput"
            ])),
            Ok(Command::Run(RunArgs {
                source: ModelSource::Model(PathBuf::from("model.ironicl")),
                input: PathBuf::from("request.irinput"),
                context: None,
                cache: CheckpointCacheOptions::default(),
                trace_stages: false,
                probabilities: false,
            }))
        );
    }

    #[test]
    fn rejects_ambiguous_run_source() {
        let error = parse_cli(&args(&[
            "run",
            "--checkpoint",
            "model.ckpt",
            "--model",
            "model.ironicl",
            "--input",
            "request.irinput",
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
    fn parses_checkpoint_cache_options() {
        assert_eq!(
            parse_cli(&args(&[
                "run",
                "--checkpoint",
                "model.ckpt",
                "--input",
                "request.irinput",
                "--rebuild-cache",
                "--cache-dir",
                "cache",
            ])),
            Ok(Command::Run(RunArgs {
                source: ModelSource::Checkpoint(PathBuf::from("model.ckpt")),
                input: PathBuf::from("request.irinput"),
                context: None,
                cache: CheckpointCacheOptions {
                    no_cache: false,
                    rebuild_cache: true,
                    cache_dir: Some(PathBuf::from("cache")),
                },
                trace_stages: false,
                probabilities: false,
            }))
        );
        let error = parse_cli(&args(&[
            "run",
            "--checkpoint",
            "model.ckpt",
            "--input",
            "request.irinput",
            "--no-cache",
        ]))
        .expect_err("execution cannot use a direct checkpoint");
        assert!(error.to_string().contains("cannot use --no-cache"));
    }

    #[test]
    fn parses_trace_stages() {
        assert_eq!(
            parse_cli(&args(&[
                "run",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput",
                "--trace-stages",
            ])),
            Ok(Command::Run(RunArgs {
                source: ModelSource::Model(PathBuf::from("model.ironicl")),
                input: PathBuf::from("request.irinput"),
                context: None,
                cache: CheckpointCacheOptions::default(),
                trace_stages: true,
                probabilities: false,
            }))
        );
    }

    #[test]
    fn parses_cached_probabilities() {
        assert_eq!(
            parse_cli(&args(&[
                "run",
                "--model",
                "model.ironicl",
                "--context",
                "context.ironiclctx",
                "--input",
                "request.irquery",
                "--probabilities",
            ])),
            Ok(Command::Run(RunArgs {
                source: ModelSource::Model(PathBuf::from("model.ironicl")),
                input: PathBuf::from("request.irquery"),
                context: Some(PathBuf::from("context.ironiclctx")),
                cache: CheckpointCacheOptions::default(),
                trace_stages: false,
                probabilities: true,
            }))
        );
    }

    #[test]
    fn rejects_incompatible_cache_options() {
        for values in [
            &[
                "run",
                "--checkpoint",
                "model.ckpt",
                "--input",
                "request.irinput",
                "--no-cache",
                "--rebuild-cache",
            ][..],
            &[
                "run",
                "--checkpoint",
                "model.ckpt",
                "--input",
                "request.irinput",
                "--no-cache",
                "--cache-dir",
                "cache",
            ][..],
            &[
                "run",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput",
                "--rebuild-cache",
            ][..],
        ] {
            assert!(parse_cli(&args(values)).is_err());
        }
    }

    #[test]
    fn parses_command_help() {
        assert_eq!(
            parse_cli(&args(&["inspect", "--help"])),
            Ok(Command::Help(Some(HelpTopic::Inspect)))
        );
        assert_eq!(
            parse_cli(&args(&["help", "run"])),
            Ok(Command::Help(Some(HelpTopic::Run)))
        );
    }

    #[test]
    fn run_requires_input() {
        let error =
            parse_cli(&args(&["run", "--model", "model.ironicl"])).expect_err("input must be required");
        assert!(error.to_string().contains("run requires --input"));
    }

    #[test]
    fn parses_model_only_benchmark_defaults_and_options() {
        assert_eq!(
            parse_cli(&args(&[
                "benchmark",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput"
            ])),
            Ok(Command::Benchmark(BenchmarkArgs {
                model: PathBuf::from("model.ironicl"),
                input: PathBuf::from("request.irinput"),
                context: None,
                probabilities: false,
                warmup: 2,
                iterations: 10,
            }))
        );
        assert_eq!(
            parse_cli(&args(&[
                "benchmark",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput",
                "--warmup",
                "0",
                "--iterations",
                "10000",
            ])),
            Ok(Command::Benchmark(BenchmarkArgs {
                model: PathBuf::from("model.ironicl"),
                input: PathBuf::from("request.irinput"),
                context: None,
                probabilities: false,
                warmup: 0,
                iterations: 10_000,
            }))
        );
        assert_eq!(
            parse_cli(&args(&[
                "benchmark",
                "--model",
                "model.ironicl",
                "--input",
                "request.irquery",
                "--context",
                "context.ironiclctx",
                "--probabilities",
            ])),
            Ok(Command::Benchmark(BenchmarkArgs {
                model: PathBuf::from("model.ironicl"),
                input: PathBuf::from("request.irquery"),
                context: Some(PathBuf::from("context.ironiclctx")),
                probabilities: true,
                warmup: 2,
                iterations: 10,
            }))
        );
    }

    #[test]
    fn benchmark_rejects_checkpoints_and_out_of_range_counts() {
        for values in [
            &[
                "benchmark",
                "--checkpoint",
                "model.ckpt",
                "--input",
                "request.irinput",
            ][..],
            &[
                "benchmark",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput",
                "--warmup",
                "1001",
            ][..],
            &[
                "benchmark",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput",
                "--iterations",
                "0",
            ][..],
            &[
                "benchmark",
                "--model",
                "model.ironicl",
                "--input",
                "request.irinput",
                "--iterations",
                "10001",
            ][..],
        ] {
            assert!(parse_cli(&args(values)).is_err());
        }
    }
}
