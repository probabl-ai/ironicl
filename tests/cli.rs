use std::process::Command;

#[test]
fn help_lists_the_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_ironicl"))
        .arg("help")
        .output()
        .expect("run CLI help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 CLI output");
    for command in ["platform", "inspect", "import", "prepare", "serve"] {
        assert!(
            stdout.contains(&format!("\n  {command} ")),
            "{command} is missing"
        );
    }
}

#[test]
fn plan_is_no_longer_a_command() {
    let output = Command::new(env!("CARGO_BIN_EXE_ironicl"))
        .arg("plan")
        .output()
        .expect("run CLI plan");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 CLI output");
    assert!(stderr.contains("unrecognized subcommand 'plan'"), "{stderr}");
}
