use std::process::Command;

#[test]
fn help_lists_the_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_ironicl"))
        .arg("help")
        .output()
        .expect("run CLI help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 CLI output");
    for command in ["platform", "plan", "inspect", "import"] {
        assert!(
            stdout.contains(&format!("ironicl {command}")),
            "{command} is missing"
        );
    }
}
