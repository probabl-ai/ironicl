use std::process::Command;

#[test]
fn run_help_describes_required_input_and_cache_constraint() {
    let output = Command::new(env!("CARGO_BIN_EXE_ironicl"))
        .args(["run", "--help"])
        .output()
        .expect("run CLI help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 CLI output");
    assert!(stdout.contains("--input FILE"));
    assert!(stdout.contains("--no-cache is not available for execution"));
    assert!(stdout.contains("--trace-stages"));
}

#[test]
fn benchmark_help_describes_model_only_timing() {
    let output = Command::new(env!("CARGO_BIN_EXE_ironicl"))
        .args(["benchmark", "--help"])
        .output()
        .expect("benchmark CLI help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 CLI output");
    assert!(stdout.contains("benchmark --model FILE --input FILE"));
    assert!(stdout.contains("Checkpoints are not accepted"));
    assert!(stdout.contains("nearest-rank"));
}
