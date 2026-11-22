use std::{env, process, time::Instant};

use std::path::{Path, PathBuf};

use ironicl::{
    CheckpointCacheOptions, Command, CpuCapabilities, HelpTopic, ImportArgs, InspectArgs, Isa, PrepareArgs,
    error::Result,
    memory::{ByteSize, MemoryLimits},
    models::tabicl_v2::directory,
    parse_cli,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}

fn run() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match parse_cli(&args)? {
        Command::Help(topic) => print_help(topic),
        Command::Platform => print_platform(),
        Command::Inspect(args) => inspect(&args)?,
        Command::Import(args) => import(&args)?,
        Command::Prepare(args) => prepare(&args)?,
        #[cfg(feature = "server")]
        Command::Serve(args) => ironicl::server::serve(&args)?,
        #[cfg(not(feature = "server"))]
        Command::Serve(_) => {
            return Err(ironicl::Error::NotImplemented(
                "this build has no server; build with --features server".into(),
            ));
        }
    }
    Ok(())
}

fn print_platform() {
    let host = CpuCapabilities::detect();
    println!("Ironicl host capabilities\n{host}");
    println!("isa:             {}", Isa::for_host(&host));
}

/// The weights folder `--weights` names, else the shared one.
fn weights_dir(weights: Option<&PathBuf>) -> Result<PathBuf> {
    weights.map_or_else(directory::default_weights_dir, |path| Ok(path.clone()))
}

/// Inspect a model directory, Ironicl weights, or a checkpoint, by what `path` is.
fn inspect(args: &InspectArgs) -> Result<()> {
    let path = &args.path;
    if path.is_dir() {
        return inspect_directory(path, &weights_dir(args.weights.as_ref())?);
    }
    let mut magic = [0; 8];
    let is_model = std::fs::File::open(path)
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
        .is_ok()
        && magic == ironicl::format::model::MODEL_ARTIFACT_MAGIC;
    if is_model {
        inspect_model_file(path)
    } else {
        inspect_checkpoint_file(path)
    }
}

fn inspect_directory(dir: &Path, weights: &Path) -> Result<()> {
    let summary = directory::inspect(dir, weights)?;
    let metadata = &summary.metadata;
    let field = |key: &str| match &metadata[key] {
        serde_json::Value::Null => "none".to_owned(),
        serde_json::Value::String(text) => text.clone(),
        value => value.to_string(),
    };
    println!("Ironicl model directory");
    println!("path:          {}", dir.display());
    println!("task:          {}", field("task"));
    println!("features:      {}", field("n_features"));
    println!("feature names: {}", field("feature_names"));
    println!("classes:       {}", field("classes"));
    println!("exported by:   Ironicl {}", field("ironicl"));
    let (path, found) = &summary.weights;
    let found = if *found { "" } else { " (missing)" };
    println!("weights:       {}{found}", path.display());
    match summary.cache {
        Some((optimize, bytes)) => println!("KV cache:      {optimize}, {}", ByteSize(bytes)),
        None => println!("KV cache:      none; ironicl prepare builds it"),
    }
    Ok(())
}

fn inspect_model_file(path: &Path) -> Result<()> {
    let summary = ironicl::inspect_model_artifact(path)?;
    println!("Ironicl model inspection");
    println!("path:          {}", path.display());
    println!("source sha256: {}", summary.source_sha256);
    println!("profile:       {}", summary.profile);
    println!("tensors:       {}", summary.tensor_count);
    println!("parameters:    {}", summary.parameter_count);
    println!("tensor bytes:  {}", ByteSize(summary.tensor_bytes));
    let dtypes = summary
        .dtype_counts
        .iter()
        .map(|(dtype, count)| format!("{dtype}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("dtypes:        {dtypes}");
    println!("task:          {}", optional_display(summary.config.task()));
    println!("max classes:   {}", optional_display(summary.config.max_classes));
    println!(
        "quantiles:     {}",
        optional_display(summary.config.num_quantiles)
    );
    println!("embedding dim: {}", optional_display(summary.config.embed_dim));
    println!(
        "column blocks: {}",
        optional_display(summary.config.column_blocks)
    );
    println!("row blocks:    {}", optional_display(summary.config.row_blocks));
    println!(
        "row CLS:       {}",
        optional_display(summary.config.row_cls_tokens)
    );
    println!("ICL blocks:    {}", optional_display(summary.config.icl_blocks));
    Ok(())
}

fn inspect_checkpoint_file(path: &Path) -> Result<()> {
    let summary = ironicl::inspect_checkpoint(path)?;
    println!("Ironicl checkpoint inspection");
    println!("path:          {}", path.display());
    println!("sha256:        {}", summary.source_sha256);
    println!("profile:       {}", summary.profile);
    println!("tensors:       {}", summary.tensor_count);
    println!("parameters:    {}", summary.parameter_count);
    println!("tensor bytes:  {}", ByteSize(summary.tensor_bytes));
    let dtypes = summary
        .dtype_counts
        .iter()
        .map(|(dtype, count)| format!("{dtype}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("dtypes:        {dtypes}");
    if let Some(config) = summary.config {
        println!("task:          {}", optional_display(config.task()));
        println!("max classes:   {}", optional_display(config.max_classes));
        println!("quantiles:     {}", optional_display(config.num_quantiles));
        println!("embedding dim: {}", optional_display(config.embed_dim));
        println!("column blocks: {}", optional_display(config.column_blocks));
        println!("row blocks:    {}", optional_display(config.row_blocks));
        println!("row CLS:       {}", optional_display(config.row_cls_tokens));
        println!("ICL blocks:    {}", optional_display(config.icl_blocks));
    }
    for warning in summary.warnings {
        println!("warning:       {warning}");
    }
    Ok(())
}

fn optional_display<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "unknown".into(), |value| value.to_string())
}

fn import(args: &ImportArgs) -> Result<()> {
    let (path, summary) = if let Some(output) = &args.output {
        (
            output.clone(),
            ironicl::import_checkpoint(&args.checkpoint, output)?,
        )
    } else {
        let prepared = ironicl::prepare_checkpoint(&args.checkpoint, &CheckpointCacheOptions::default())?;
        (prepared.path, prepared.summary)
    };
    println!("Imported the weights");
    println!("checkpoint:    {}", args.checkpoint.display());
    println!("weights:       {}", path.display());
    println!("source sha256: {}", summary.source_sha256);
    println!("profile:       {}", summary.profile);
    println!("tensors:       {}", summary.tensor_count);
    println!("parameters:    {}", summary.parameter_count);
    println!("tensor bytes:  {}", ByteSize(summary.tensor_bytes));
    Ok(())
}

fn prepare(args: &PrepareArgs) -> Result<()> {
    let start = Instant::now();
    let weights = weights_dir(args.weights.as_ref())?;
    let prepared = directory::prepare(
        &args.dir,
        &weights,
        args.optimize,
        None,
        MemoryLimits::resolve(None, None)?,
    )?;
    let path = args.dir.join(directory::CONTEXT);
    let bytes = std::fs::metadata(&path).map_or(0, |metadata| metadata.len());
    let action = if prepared.built { "Built" } else { "Reused" };
    println!("{action} the KV cache");
    println!("path:          {}", path.display());
    println!("device:        {}", prepared.model.device());
    println!("optimize:      {}", prepared.optimize);
    println!("size:          {}", ByteSize(bytes));
    println!("time:          {:.2} s", start.elapsed().as_secs_f64());
    Ok(())
}

fn print_help(topic: Option<HelpTopic>) {
    let text = match topic {
        None => {
            "Ironicl\n\n\
             Usage:\n\
               ironicl platform\n\
               ironicl inspect PATH [--weights DIR]\n\
               ironicl import CHECKPOINT [-o FILE]\n\
               ironicl prepare DIR [--weights DIR] [--optimize safe|auto|fast]\n\
               ironicl serve DIR [--weights DIR] [--optimize safe|auto|fast] [--bind ADDR]\n\
             \x20 [--max-request-rows N] [--max-body SIZE] [--max-batch-rows N] [--max-queue MS]\n\n\
             Use 'ironicl help COMMAND' for command-specific help."
        }
        Some(HelpTopic::Platform) => "Usage: ironicl platform",
        Some(HelpTopic::Inspect) => {
            "Usage: ironicl inspect PATH [--weights DIR]\n\n\
             Inspect a model directory, Ironicl weights (.irmodel), or a PyTorch\n\
             checkpoint. For a model directory, --weights is the folder its weights\n\
             are looked for in, by default the shared one."
        }
        Some(HelpTopic::Import) => {
            "Usage: ironicl import CHECKPOINT [-o FILE]\n\n\
             Convert a PyTorch checkpoint to Ironicl weights: by default into the\n\
             shared weights folder (~/.cache/ironicl/models/v1, under\n\
             XDG_CACHE_HOME when set), where prepare and serve look for them."
        }
        Some(HelpTopic::Prepare) => {
            "Usage: ironicl prepare DIR [--weights DIR] [--optimize safe|auto|fast]\n\n\
             Build the KV cache of a model directory on this machine's device and\n\
             write it to DIR/context.ircntxt, unless the saved cache is already the\n\
             one this model builds at the level. The level defaults to the saved\n\
             cache's, else safe. --weights is the folder of weights, by default the\n\
             shared one."
        }
        Some(HelpTopic::Serve) => {
            "Usage: ironicl serve DIR [--weights DIR] [--optimize safe|auto|fast] [--bind ADDR]\n\
             \x20      [--max-request-rows N] [--max-body SIZE] [--max-batch-rows N] [--max-queue MS]\n\n\
             Serve a model directory over HTTP: POST /v1/predict, GET /v1/health. It\n\
             first prepares the KV cache as ironicl prepare does. Concurrent requests\n\
             are predicted together, up to --max-batch-rows rows at once; a request\n\
             that would wait longer than --max-queue is refused with 503. Defaults:\n\
             --bind 127.0.0.1:8080, --max-request-rows 10000, --max-body 16M,\n\
             --max-batch-rows 1024 on a GPU and 32 on the CPU, --max-queue 1000."
        }
    };
    println!("{text}");
}
