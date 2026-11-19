//! Model-independent preprocessing that must reproduce Python exactly.
//!
//! Models' own pipelines build on these pieces. This code handles untrusted
//! input, so it may not use `unsafe`.
#![forbid(unsafe_code)]

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "TabICL preprocessing uses it in a following change")
)]
pub(crate) mod python_random;

/// A golden fixture from `tools/export_preprocess_fixtures.py`.
#[cfg(test)]
pub(crate) fn fixture(name: &str) -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture JSON")
}

/// A fixture's unsigned integers, exactly.
#[cfg(test)]
pub(crate) fn integers(value: &serde_json::Value) -> Vec<u64> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|item| item.as_u64().expect("an unsigned integer"))
        .collect()
}
