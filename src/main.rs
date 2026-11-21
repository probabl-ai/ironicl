use std::{env, process};

use ironicl::{
    Command, CpuCapabilities, HelpTopic, ImportArgs, InspectArgs, Isa, ModelSource, PlanArgs, StorageDType,
    error::{Error, Result},
    estimate_memory, memory,
    memory::ByteSize,
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
        Command::Plan(args) => print_plan(args)?,
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

fn print_plan(args: PlanArgs) -> Result<()> {
    let config = args.config;
    let workload = config.workload;
    let host = CpuCapabilities::detect();
    let isa = Isa::resolve(config, &host)?;
    let memory = estimate_memory(config)?;

    println!("Ironicl execution plan");
    println!("task:             {}", config.model.task);
    println!("training rows:    {}", workload.training_rows);
    println!("features:         {}", workload.features);
    println!("estimators:       {}", workload.estimators);
    println!("query microbatch: {}", workload.query_microbatch);
    println!("cache mode:       {}", config.cache_mode);
    println!("weight storage:   {}", config.weight_dtype);
    println!("cache storage:    {}", config.kv_dtype);
    println!("isa:              {isa}");
    println!();
    println!("weights:          {}", ByteSize(memory.weight_bytes));
    println!("column cache:     {}", ByteSize(memory.cache.column_bytes));
    println!("context cache:    {}", ByteSize(memory.cache.context_bytes));
    println!("cache total:      {}", ByteSize(memory.cache.total_bytes()));
    println!("persistent total: {}", ByteSize(memory.persistent_bytes()));

    // Whether it fits this machine, with the context built on the CPU.
    let storage = match config.kv_dtype {
        StorageDType::Fp32 => memory::Storage::FP32,
        dtype => memory::Storage {
            name: match dtype {
                StorageDType::Bf16 => "BF16",
                StorageDType::F16 => "FP16",
                _ => "INT8",
            },
            key_bytes: dtype.bytes_per_element(),
            value_bytes: dtype.bytes_per_element(),
        },
    };
    let shape = memory::ContextShape {
        task: config.model.task,
        estimators: workload.estimators,
        rows: workload.training_rows,
        features: workload.features,
    };
    let pools = memory::Pools::detect().with_limits(memory::MemoryLimits::resolve(None, None)?);
    let plan = memory::plan(shape, storage, memory::Builder::Cpu, pools, &[])?;
    println!();
    println!("{plan}");

    if let Some(bandwidth_gbps) = args.bandwidth_gbps {
        let seconds = memory
            .cache_scan_seconds(bandwidth_gbps)
            .ok_or_else(|| Error::InvalidArgument("bandwidth must be finite and greater than zero".into()))?;
        println!();
        println!("bandwidth input:  {bandwidth_gbps:.1} GB/s");
        println!("one scan floor:   {:.2} ms", seconds * 1_000.0);
        println!(
            "amortized floor:  {:.2} ms/query at microbatch {}",
            seconds * 1_000.0 / f64::from(workload.query_microbatch),
            workload.query_microbatch
        );
        println!("note: floors exclude projections, softmax, cache misses, and synchronization");
    }
    Ok(())
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
               ironicl plan [options]\n\
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
        Some(HelpTopic::Plan) => {
            "Usage: ironicl plan [options]\n\n\
             Options:\n\
               --task classifier|regressor\n\
               --rows N\n\
               --features N\n\
               --estimators N\n\
               --microbatch N\n\
               --cache kv|repr\n\
               --weights fp32|bf16|int8\n\
               --kv-dtype fp32|bf16|int8\n\
               --isa auto|portable|neon|avx512\n\
               --bandwidth-gbps N"
        }
    };
    println!("{text}");
}
