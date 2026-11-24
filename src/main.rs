use std::{process, time::Instant};

use std::path::{Path, PathBuf};

use clap::Parser;
use ironicl::{
    CheckpointCacheOptions, Cli, Command, CpuCapabilities, ImportArgs, InspectArgs, Isa, PrepareArgs,
    error::Result,
    memory::{ByteSize, MemoryLimits},
    models::tabicl_v2::directory,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(2);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
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
    let names = metadata
        .feature_names
        .as_ref()
        .map_or("none".to_owned(), |names| names.join(", "));
    let classes = metadata.classes.as_ref().map_or("none".to_owned(), |classes| {
        serde_json::to_string(classes).unwrap_or_default()
    });
    println!("Ironicl model directory");
    println!("path:          {}", dir.display());
    println!("task:          {}", metadata.task);
    println!("features:      {}", metadata.n_features);
    println!("feature names: {names}");
    println!("classes:       {classes}");
    println!("exported by:   Ironicl {}", metadata.ironicl);
    let (path, found) = &summary.weights;
    let found = if *found { "" } else { " (missing)" };
    println!("weights:       {}{found}", path.display());
    match summary.cache {
        Some(Ok((optimize, bytes))) => println!("KV cache:      {optimize}, {}", ByteSize(bytes)),
        Some(Err(error)) => println!("KV cache:      unreadable ({error}); ironicl prepare rebuilds it"),
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
    match summary.config {
        ironicl::format::model::ModelConfig::TabIclV2(config) => {
            println!("task:          {}", optional_display(config.task()));
            println!("max classes:   {}", optional_display(config.max_classes));
            println!("quantiles:     {}", optional_display(config.num_quantiles));
            println!("embedding dim: {}", optional_display(config.embed_dim));
            println!("column blocks: {}", optional_display(config.column_blocks));
            println!("row blocks:    {}", optional_display(config.row_blocks));
            println!("row CLS:       {}", optional_display(config.row_cls_tokens));
            println!("ICL blocks:    {}", optional_display(config.icl_blocks));
        }
        ironicl::format::model::ModelConfig::KumoTabular(release) => print_kumo_tabular(release),
    }
    Ok(())
}

fn print_kumo_tabular(release: ironicl::models::kumo_tabular::Release) {
    let config = release.size.config();
    println!("task:          {}", release.task);
    println!("size:          {}", release.size);
    println!(
        "cell layers:   {} x {} channels, {} inducing points",
        config.embedding_layers, config.cell_channels, config.inducing_points
    );
    println!(
        "ICL layers:    {} x {} channels, {} heads",
        config.icl_layers, config.icl_channels, config.icl_heads
    );
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
    if let ironicl::CheckpointProfile::KumoTabular(release) = summary.profile {
        print_kumo_tabular(release);
    }
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
    if let Some(warning) = &prepared.warning {
        eprintln!("warning: {warning}");
    }
    let action = if prepared.built { "Built" } else { "Reused" };
    println!("{action} the KV cache");
    println!("path:          {}", path.display());
    println!("device:        {}", prepared.model.device());
    println!("optimize:      {}", prepared.optimize);
    println!("size:          {}", ByteSize(bytes));
    println!("time:          {:.2} s", start.elapsed().as_secs_f64());
    Ok(())
}
