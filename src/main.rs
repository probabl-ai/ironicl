use std::{env, process};

use ironicl::{
    Command, CpuCapabilities, HelpTopic, ImportArgs, InspectArgs, Isa, ModelSource, error::Result,
    memory::ByteSize, parse_cli,
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

fn print_help(topic: Option<HelpTopic>) {
    let text = match topic {
        None => {
            "Ironicl\n\n\
             Usage:\n\
               ironicl platform\n\
               ironicl inspect (--checkpoint FILE | --model FILE)\n\
               ironicl import --checkpoint FILE --output FILE\n\n\
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
    };
    println!("{text}");
}
