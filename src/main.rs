use std::{env, process, time::Instant};

use ironicl::{
    BenchmarkArgs, BenchmarkStats, BuildContextArgs, ClassifierExecutor, ClassifierInputBundle, Command,
    CpuCapabilities, HelpTopic, ImportArgs, InspectArgs, ModelSource, PlanArgs, RunArgs,
    StaticClassifierExecutor, StaticContextArtifact, StaticQueryBundle,
    cache::ByteSize,
    error::{Error, Result},
    estimate_memory, parse_cli, resolve_backend,
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
        Command::Run(args) => run_model(args)?,
        Command::Benchmark(args) => benchmark(&args)?,
        Command::BuildContext(args) => build_context(&args)?,
    }
    Ok(())
}

fn print_platform() {
    let host = CpuCapabilities::detect();
    println!("IronICL host capabilities\n{host}");
    println!(
        "zen5 base:       {}",
        if host.supports_zen5_base() { "yes" } else { "no" }
    );
}

fn print_plan(args: PlanArgs) -> Result<()> {
    let config = args.config;
    let workload = config.workload;
    let host = CpuCapabilities::detect();
    let selected_backend = resolve_backend(config, &host)?;
    let memory = estimate_memory(config)?;

    println!("IronICL execution plan");
    println!("task:             {}", config.model.task);
    println!("training rows:    {}", workload.training_rows);
    println!("features:         {}", workload.features);
    println!("estimators:       {}", workload.estimators);
    println!("query microbatch: {}", workload.query_microbatch);
    println!("cache mode:       {}", config.cache_mode);
    println!("weight storage:   {}", config.weight_dtype);
    println!("cache storage:    {}", config.kv_dtype);
    println!("backend:          {selected_backend}");
    println!();
    println!("weights:          {}", ByteSize(memory.weight_bytes));
    println!("column cache:     {}", ByteSize(memory.cache.column_bytes));
    println!("context cache:    {}", ByteSize(memory.cache.context_bytes));
    println!("cache total:      {}", ByteSize(memory.cache.total_bytes()));
    println!("persistent total: {}", ByteSize(memory.persistent_bytes()));

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
    println!("IronICL model inspection");
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

#[cfg(feature = "checkpoint")]
fn inspect_checkpoint_file(path: &std::path::Path) -> Result<()> {
    let summary = ironicl::inspect_checkpoint(path)?;
    println!("IronICL checkpoint inspection");
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

#[cfg(not(feature = "checkpoint"))]
fn inspect_checkpoint_file(_path: &std::path::Path) -> Result<()> {
    Err(Error::NotImplemented(
        "checkpoint inspection is disabled; rebuild with the 'checkpoint' feature".into(),
    ))
}

fn optional_display<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "unknown".into(), |value| value.to_string())
}

#[cfg(feature = "checkpoint")]
fn import(args: &ImportArgs) -> Result<()> {
    let summary = ironicl::import_checkpoint(&args.checkpoint, &args.output)?;
    println!("Imported IronICL model");
    println!("checkpoint:    {}", args.checkpoint.display());
    println!("model:         {}", args.output.display());
    println!("source sha256: {}", summary.source_sha256);
    println!("profile:       {}", summary.profile);
    println!("tensors:       {}", summary.tensor_count);
    println!("parameters:    {}", summary.parameter_count);
    println!("tensor bytes:  {}", ByteSize(summary.tensor_bytes));
    Ok(())
}

#[cfg(not(feature = "checkpoint"))]
fn import(_args: &ImportArgs) -> Result<()> {
    Err(Error::NotImplemented(
        "checkpoint import is disabled; rebuild with the 'checkpoint' feature".into(),
    ))
}

fn build_context(args: &BuildContextArgs) -> Result<()> {
    let input = ClassifierInputBundle::read(&args.input)?;
    let request = input.request();
    let (b, t, h, n) = (
        request.ensemble_count(),
        request.total_rows(),
        request.feature_count(),
        request.training_rows(),
    );
    let class_count = ironicl::StaticContextRequest::class_count_of(request.y_train(), b, n)?;
    let class_permutations = (0..b).flat_map(|_| 0..class_count as u64).collect::<Vec<_>>();
    let feature_permutations = (0..b).flat_map(|_| 0..h as u64).collect::<Vec<_>>();
    let context_sha256 = ironicl::decode_sha256(&input.sha256())?;
    let context_request = ironicl::StaticContextRequest::new(
        b,
        n,
        t,
        h,
        request.x(),
        request.y_train(),
        &class_permutations,
        &feature_permutations,
        context_sha256,
    )?;
    let options = ironicl::StaticContextOptions {
        kv_dtype: args.kv_dtype,
        pack_keys: args.pack_keys,
        sparse_index_clusters: args.index_clusters,
    };
    let start = Instant::now();
    let executor = ClassifierExecutor::new(ironicl::load_model_artifact(&args.model)?);
    let summary = ironicl::build_static_context_file(&executor, &context_request, &options, &args.output)?;
    println!("Built IronICL static context");
    println!("model:          {}", args.model.display());
    println!("input:          {}", args.input.display());
    println!("context:        {}", args.output.display());
    println!("estimators:     {}", summary.ensemble_count);
    println!("training rows:  {}", summary.training_rows);
    println!("features:       {}", summary.feature_count);
    println!("classes:        {}", summary.class_count);
    println!("version:        {}", summary.version);
    println!("kv dtype:       {}", summary.kv_dtype.name());
    println!("packed keys:    {}", summary.packed_icl_keys);
    println!(
        "index clusters: {}",
        optional_display(summary.sparse_index_clusters)
    );
    println!("file bytes:     {}", ByteSize(summary.file_bytes));
    println!("file sha256:    {}", ironicl::encode_sha256(&summary.file_sha256));
    println!("build seconds:  {:.3}", start.elapsed().as_secs_f64());
    Ok(())
}

fn run_model(args: RunArgs) -> Result<()> {
    let model_path = match args.source {
        ModelSource::Checkpoint(path) => match ironicl::prepare_checkpoint(&path, &args.cache)? {
            ironicl::PreparedCheckpoint::Direct { .. } => {
                return Err(Error::InvalidArgument(
                    "run cannot execute a direct checkpoint; remove --no-cache".into(),
                ));
            }
            ironicl::PreparedCheckpoint::Cached { path: model, .. } => model,
        },
        ModelSource::Model(path) => path,
    };
    let model = ironicl::load_model_artifact(&model_path)?;
    if let Some(context_path) = args.context {
        let query = StaticQueryBundle::read(&args.input)?;
        let context = StaticContextArtifact::read(&context_path)?;
        let executor = StaticClassifierExecutor::new(model, context)?;
        if args.probabilities {
            print_probabilities(&executor.predict_bundle_probabilities(&query)?);
        } else {
            print_logits(&executor.predict_bundle(&query)?);
        }
        return Ok(());
    }
    let input = ClassifierInputBundle::read(&args.input)?;
    let executor = ClassifierExecutor::new(model);
    if args.trace_stages {
        let trace = executor.predict_logits_with_trace(input.request())?;
        print_trace_stage("column_embeddings", trace.column_embeddings());
        print_trace_stage("row_representations", trace.row_representations());
        print_trace_stage("decoder_logits", trace.decoder_logits());
        print_logits(trace.logits());
    } else {
        print_logits(&executor.predict_logits(input.request())?);
    }
    Ok(())
}

fn benchmark(args: &BenchmarkArgs) -> Result<()> {
    let input_start = Instant::now();
    let cached = args.context.is_some();
    let input = (!cached)
        .then(|| ClassifierInputBundle::read(&args.input))
        .transpose()?;
    let query = cached.then(|| StaticQueryBundle::read(&args.input)).transpose()?;
    let input_load_ms = input_start.elapsed().as_secs_f64() * 1_000.0;
    let model_start = Instant::now();
    let model = ironicl::load_model_artifact(&args.model)?;
    let model_load_ms = model_start.elapsed().as_secs_f64() * 1_000.0;
    let executor = (!cached).then(|| ClassifierExecutor::new(model.clone()));
    let static_executor = args
        .context
        .as_ref()
        .map(|path| {
            StaticContextArtifact::read(path)
                .and_then(|context| StaticClassifierExecutor::new(model, context))
        })
        .transpose()?;

    let mut durations = Vec::with_capacity(args.iterations as usize);
    let mut shape = None;
    let mut checksum = 0.0_f64;
    for _ in 0..args.warmup {
        let output = benchmark_prediction(
            args.probabilities,
            executor.as_ref(),
            input.as_ref(),
            static_executor.as_ref(),
            query.as_ref(),
        )?;
        check_benchmark_prediction(&output, shape)?;
        shape = Some(output.shape());
    }
    ironicl::profile::reset();
    for _ in 0..args.iterations {
        let start = Instant::now();
        let output = benchmark_prediction(
            args.probabilities,
            executor.as_ref(),
            input.as_ref(),
            static_executor.as_ref(),
            query.as_ref(),
        )?;
        durations.push(start.elapsed());
        checksum = check_benchmark_prediction(&output, shape)?;
        shape = Some(output.shape());
    }
    let shape = shape.expect("benchmark has at least one iteration");
    let stats = BenchmarkStats::from_durations(&durations);
    let elapsed_seconds = durations
        .iter()
        .map(std::time::Duration::as_secs_f64)
        .sum::<f64>();
    let inferences_per_second = f64::from(args.iterations) / elapsed_seconds;
    let query_rows_per_second =
        inferences_per_second * f64::from(u32::try_from(shape[1]).expect("input bundle query rows fit u32"));

    println!("input: {}", args.input.display());
    println!("model: {}", args.model.display());
    if let Some(context) = &args.context {
        println!("context: {}", context.display());
    }
    println!("input_load_ms: {input_load_ms:.6}");
    println!("model_load_ms: {model_load_ms:.6}");
    println!("warmup: {}", args.warmup);
    println!("iterations: {}", args.iterations);
    println!(
        "output: {}",
        if args.probabilities {
            "probabilities"
        } else {
            "estimator_logits"
        }
    );
    println!("shape: {} {} {}", shape[0], shape[1], shape[2]);
    println!("min_ms: {:.6}", stats.min_ms);
    println!("p50_ms: {:.6}", stats.p50_ms);
    println!("p95_ms: {:.6}", stats.p95_ms);
    println!("p99_ms: {:.6}", stats.p99_ms);
    println!("max_ms: {:.6}", stats.max_ms);
    println!("mean_ms: {:.6}", stats.mean_ms);
    println!("inferences_per_second: {inferences_per_second:.6}");
    println!("query_rows_per_second: {query_rows_per_second:.6}");
    println!("output_checksum: {checksum:.9e}");
    if !args.probabilities {
        println!("logits_checksum: {checksum:.9e}");
    }
    let samples = durations
        .iter()
        .map(|duration| duration.as_nanos().to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("samples_ns: {samples}");
    ironicl::profile::report();
    Ok(())
}

enum BenchmarkPrediction {
    Logits(ironicl::ClassifierLogits),
    Probabilities(ironicl::ClassifierProbabilities),
}

impl BenchmarkPrediction {
    fn shape(&self) -> [usize; 3] {
        match self {
            Self::Logits(logits) => logits.shape(),
            Self::Probabilities(probabilities) => {
                let [rows, classes] = probabilities.shape();
                [1, rows, classes]
            }
        }
    }

    fn values(&self) -> &[f32] {
        match self {
            Self::Logits(logits) => logits.values(),
            Self::Probabilities(probabilities) => probabilities.values(),
        }
    }
}

fn benchmark_prediction(
    probabilities: bool,
    executor: Option<&ClassifierExecutor>,
    input: Option<&ClassifierInputBundle>,
    static_executor: Option<&StaticClassifierExecutor>,
    query: Option<&StaticQueryBundle>,
) -> Result<BenchmarkPrediction> {
    if probabilities {
        return Ok(BenchmarkPrediction::Probabilities(
            static_executor
                .expect("cached executor")
                .predict_bundle_probabilities(query.expect("cached query"))?,
        ));
    }
    let logits = if let (Some(executor), Some(input)) = (executor, input) {
        executor.predict_logits(input.request())?
    } else {
        static_executor
            .expect("cached executor")
            .predict_bundle(query.expect("cached query"))?
    };
    Ok(BenchmarkPrediction::Logits(logits))
}

fn check_benchmark_prediction(
    output: &BenchmarkPrediction,
    expected_shape: Option<[usize; 3]>,
) -> Result<f64> {
    if expected_shape.is_some_and(|shape| shape != output.shape()) {
        return Err(Error::InvalidArgument(
            "benchmark output shape changed between inferences".into(),
        ));
    }
    if output.values().iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument(
            "benchmark output contains a non-finite value".into(),
        ));
    }
    Ok(output.values().iter().map(|value| f64::from(*value)).sum())
}

fn print_trace_stage<const D: usize>(name: &str, stage: &ironicl::ClassifierTraceStage<D>) {
    let values = stage.values();
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = values.iter().sum::<f32>();
    let first = values
        .iter()
        .take(8)
        .map(|value| format!("{value:.9e}"))
        .collect::<Vec<_>>()
        .join(" ");
    let shape = stage
        .shape()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    println!("trace {name}: shape: {shape}; min: {min:.9e}; max: {max:.9e}; sum: {sum:.9e}; first8: {first}");
}

fn print_logits(logits: &ironicl::ClassifierLogits) {
    let [ensembles, test_rows, classes] = logits.shape();
    println!("shape: {ensembles} {test_rows} {classes}");
    for ensemble in 0..ensembles {
        for row in 0..test_rows {
            let offset = (ensemble * test_rows + row) * classes;
            let values = logits.values()[offset..][..classes]
                .iter()
                .map(|value| format!("{value:.9e}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!("logits[{ensemble},{row}]: {values}");
        }
    }
}

fn print_probabilities(probabilities: &ironicl::ClassifierProbabilities) {
    let [rows, classes] = probabilities.shape();
    println!("shape: {rows} {classes}");
    for row in 0..rows {
        let values = probabilities.values()[row * classes..][..classes]
            .iter()
            .map(|value| format!("{value:.9e}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("probabilities[{row}]: {values}");
    }
}

fn print_help(topic: Option<HelpTopic>) {
    let text = match topic {
        None => {
            "IronICL\n\n\
             Usage:\n\
               ironicl platform\n\
               ironicl plan [options]\n\
               ironicl inspect (--checkpoint FILE | --model FILE)\n\
               ironicl import --checkpoint FILE --output FILE\n\
               ironicl run (--checkpoint FILE | --model FILE) --input FILE [options]\n\
               ironicl benchmark --model FILE --input FILE [--context FILE] [--warmup N] [--iterations N]\n\
               ironicl build-context --model FILE --input FILE --output FILE [options]\n\n\
             Use 'ironicl help COMMAND' for command-specific help."
        }
        Some(HelpTopic::Platform) => "Usage: ironicl platform",
        Some(HelpTopic::Inspect) => {
            "Usage: ironicl inspect (--checkpoint FILE | --model FILE)\n\n\
             Inspect a PyTorch checkpoint or an imported IronICL model."
        }
        Some(HelpTopic::Import) => {
            "Usage: ironicl import --checkpoint FILE --output FILE\n\n\
             Convert a PyTorch checkpoint to a portable IronICL model."
        }
        Some(HelpTopic::Run) => {
            "Usage: ironicl run (--checkpoint FILE | --model FILE) --input FILE [--context FILE] [options]\n\n\
              Input:\n\
                 --input FILE    IRINPUT, or IRQUERY when --context is set\n\
                 --context FILE  full-K/V static-context artifact\n\n\
             Options for --checkpoint:\n\
               --no-cache\n\
               --rebuild-cache\n\
               --cache-dir DIR\n\n\
             --trace-stages   print retained coarse stage outputs; use small requests\n\n\
             --probabilities  aggregate cached estimator logits into predict_proba output\n\n\
              Checkpoints are imported to a content-addressed cache automatically.\n\
              --no-cache is not available for execution because it does not create an artifact."
        }
        Some(HelpTopic::Benchmark) => {
            "Usage: ironicl benchmark --model FILE --input FILE [--context FILE] [--probabilities] [--warmup N] [--iterations N]\n\n\
             Benchmark resident model-only inference. Checkpoints are not accepted, so import and conversion remain outside comparable timing.\n\n\
             Options:\n\
               --model FILE      imported IronICL model artifact\n\
               --input FILE      IRINPUT, or IRQUERY when --context is set\n\
                --context FILE    full-K/V static-context artifact\n\
                --probabilities   time final cached predict_proba output\n\
                --warmup N        untimed inferences, 0..=1000 (default: 2)\n\
               --iterations N    timed inferences, 1..=10000 (default: 10)\n\n\
             Latency percentiles use nearest-rank: ceil(percentile * iterations)."
        }
        Some(HelpTopic::BuildContext) => {
            "Usage: ironicl build-context --model FILE --input FILE --output FILE [options]\n\n\
             Build a full-K/V static context from the training rows of an IRINPUT bundle, natively.\n\
             The bundle's query rows are ignored; class and feature permutations are the identity.\n\n\
             Options:\n\
               --model FILE          imported IronICL model artifact\n\
               --input FILE          IRINPUT bundle with the preprocessed training table\n\
               --output FILE         static-context artifact to write (.ironiclctx)\n\
               --kv-dtype DTYPE      fp32|bf16|bf16-v ICL K/V storage (default: fp32)\n\
               --pack-keys           store ICL keys in source-tile packed order (format version 3)\n\
               --index-clusters N    add a coarse ICL key index with N clusters per head"
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
               --backend auto|portable|zen5\n\
               --bandwidth-gbps N"
        }
    };
    println!("{text}");
}
