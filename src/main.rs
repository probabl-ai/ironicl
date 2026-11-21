use std::{env, process, time::Instant};

use ironicl::{
    Command, CpuCapabilities, HelpTopic, ImportArgs, InspectArgs, Isa, ModelSource, PrepareArgs,
    error::Result, memory::ByteSize, models::tabicl_v2::saved, parse_cli,
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
        Command::Inspect(args) => inspect(args)?,
        Command::Import(args) => import(&args)?,
        Command::Prepare(args) => prepare(&args)?,
    }
    Ok(())
}

fn print_platform() {
    let host = CpuCapabilities::detect();
    println!("Ironicl host capabilities\n{host}");
    println!("isa:             {}", Isa::for_host(&host));
}

fn inspect(args: InspectArgs) -> Result<()> {
    match args.source {
        ModelSource::Checkpoint(path) => inspect_checkpoint_file(&path),
        ModelSource::Model(path) => inspect_model_file(&path),
    }
}

fn inspect_model_file(path: &std::path::Path) -> Result<()> {
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

fn inspect_checkpoint_file(path: &std::path::Path) -> Result<()> {
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
    let summary = ironicl::import_checkpoint(&args.checkpoint, &args.output)?;
    println!("Imported Ironicl model");
    println!("checkpoint:    {}", args.checkpoint.display());
    println!("model:         {}", args.output.display());
    println!("source sha256: {}", summary.source_sha256);
    println!("profile:       {}", summary.profile);
    println!("tensors:       {}", summary.tensor_count);
    println!("parameters:    {}", summary.parameter_count);
    println!("tensor bytes:  {}", ByteSize(summary.tensor_bytes));
    Ok(())
}

fn prepare(args: &PrepareArgs) -> Result<()> {
    let start = Instant::now();
    let (fitted, path) = saved::prepare(&args.dir, args.optimize)?;
    let seconds = start.elapsed().as_secs_f64();
    let bytes = std::fs::metadata(&path).map_or(0, |metadata| metadata.len());
    println!("Prepared the KV cache");
    println!("path:          {}", path.display());
    println!("device:        {}", fitted.device());
    println!("optimize:      {}", args.optimize);
    println!("size:          {}", ByteSize(bytes));
    println!("time:          {seconds:.2} s");
    Ok(())
}

fn print_help(topic: Option<HelpTopic>) {
    let text = match topic {
        None => {
            "Ironicl\n\n\
             Usage:\n\
               ironicl platform\n\
               ironicl inspect (--checkpoint FILE | --model FILE)\n\
               ironicl import --checkpoint FILE --output FILE\n\
               ironicl prepare DIR [--optimize safe|auto|fast]\n\n\
             Use 'ironicl help COMMAND' for command-specific help."
        }
        Some(HelpTopic::Platform) => "Usage: ironicl platform",
        Some(HelpTopic::Inspect) => {
            "Usage: ironicl inspect (--checkpoint FILE | --model FILE)\n\n\
             Inspect a PyTorch checkpoint or an imported Ironicl model."
        }
        Some(HelpTopic::Import) => {
            "Usage: ironicl import --checkpoint FILE --output FILE\n\n\
             Convert a PyTorch checkpoint to a portable Ironicl model."
        }
        Some(HelpTopic::Prepare) => {
            "Usage: ironicl prepare DIR [--optimize safe|auto|fast]\n\n\
             Build the KV cache of a model exported with export(), on this machine's\n\
             device, from its training data. The level defaults to safe and is\n\
             recorded in DIR/context.ircntxt, which ironicl serve reads."
        }
    };
    println!("{text}");
}
