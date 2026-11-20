//! Model-independent preprocessing: `CPython`'s random module, reproduced
//! exactly; class labels; and `f64` column statistics that follow scikit-learn
//! and `SciPy`. Models' own pipelines, such as `models::tabicl_v2::preprocess`,
//! build on these pieces. This code handles untrusted input, so it may not use `unsafe`.
#![forbid(unsafe_code)]

#[allow(dead_code, reason = "the fitted model uses it in the following change")]
pub mod labels;
pub(crate) mod python_random;
pub(crate) mod scale;
pub(crate) mod yeo_johnson;

/// A golden fixture from `tools/export_preprocess_fixtures.py`.
#[cfg(test)]
pub(crate) fn fixture(name: &str) -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture JSON")
}

/// A fixture's numbers.
#[cfg(test)]
pub(crate) fn numbers(value: &serde_json::Value) -> Vec<f64> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .map(|item| item.as_f64().expect("a number"))
        .collect()
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
