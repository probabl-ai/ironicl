//! Fitted preprocessing metadata and optional training tensors. Kumo keeps
//! the encoded input table (including missing numeric values) to refit on load.

use super::{
    artifact_error,
    container::{self, DType, Header, Kind, MEMORY_ALIGNMENT, Writer, decode_le, native_bytes},
};
use crate::{
    Labels, Preprocessed, Result,
    models::{
        ModelFamily, kumo_tabular,
        tabicl_v2::{
            fitted::Training,
            preprocess::{MAX_MEMBERS, Preprocessor},
        },
    },
};
use serde::{Deserialize, Serialize, de::IgnoredAny};

pub use container::{MAGIC as PREPROCESSOR_MAGIC, VERSION as PREPROCESSOR_VERSION};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PreprocessingIdentity {
    pub family: ModelFamily,
    pub checkpoint_sha256: [u8; 32],
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata<S> {
    model: ModelFamily,
    #[serde(with = "container::digest")]
    checkpoint_sha256: [u8; 32],
    /// Without training rows, this is the only way to identify the matching context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context_identity: Option<String>,
    state: S,
}

impl Metadata<IgnoredAny> {
    fn identity(&self) -> PreprocessingIdentity {
        PreprocessingIdentity {
            family: self.model,
            checkpoint_sha256: self.checkpoint_sha256,
        }
    }
}

pub(crate) fn identity(bytes: &[u8]) -> Result<PreprocessingIdentity> {
    Ok(
        Header::<Metadata<IgnoredAny>>::from_bytes(bytes, Kind::Preprocessing)?
            .metadata
            .identity(),
    )
}
pub(crate) fn file_identity(path: &std::path::Path, max: u64) -> Result<PreprocessingIdentity> {
    Ok(
        Header::<Metadata<IgnoredAny>>::open(path, Kind::Preprocessing, max)?
            .metadata
            .identity(),
    )
}

pub(crate) fn encode(
    preprocessor: &Preprocessor,
    training: Option<&Training>,
    checkpoint_sha256: &[u8; 32],
    context_sha256: &[u8; 32],
) -> Result<Vec<u8>> {
    preprocessor.validate()?;
    let meta = Metadata {
        model: ModelFamily::TabIclV2,
        checkpoint_sha256: *checkpoint_sha256,
        context_identity: training.is_none().then(|| super::encode_sha256(context_sha256)),
        state: preprocessor,
    };
    let mut tensors = Vec::new();
    if let Some(training) = training {
        let rows = training.targets.len() as u64;
        for (group, x) in training.group_rows.iter().enumerate() {
            tensors.push((
                format!("group.{group}.x"),
                DType::F32,
                vec![rows, preprocessor.n_features_out() as u64],
                native_bytes(x)?,
            ));
        }
        tensors.push((
            "y".into(),
            DType::F32,
            vec![rows],
            native_bytes(&training.targets)?,
        ));
    }
    Writer::new(Kind::Preprocessing, MEMORY_ALIGNMENT, meta, tensors)?.bytes()
}

fn finite_f32s(
    h: &Header<Metadata<Preprocessor>>,
    bytes: &[u8],
    index: usize,
    name: &str,
    shape: &[u64],
) -> Result<Vec<f32>> {
    let data = h.check_bytes(index, name, DType::F32, shape, bytes)?;
    if !DType::F32.all_finite(data)? {
        return Err(artifact_error("non-finite training value"));
    }
    Ok(decode_le(data, f32::from_le_bytes))
}

pub(crate) fn decode(bytes: &[u8]) -> Result<([u8; 32], Preprocessed, Option<String>)> {
    let h = Header::<Metadata<Preprocessor>>::from_bytes(bytes, Kind::Preprocessing)?;
    let p = &h.metadata.state;
    if h.metadata.model != ModelFamily::TabIclV2 {
        return Err(artifact_error("wrong preprocessing model"));
    }
    p.validate()?;
    let training = if h.tensors.is_empty() {
        None
    } else {
        let groups = p.normalizations.len();
        let rows = h.tensors[0].shape.first().copied().unwrap_or(0);
        if rows == 0 || h.tensors.len() != groups + 1 {
            return Err(artifact_error("invalid training tensor count or rows"));
        }
        let features = p.n_features_out() as u64;
        let group_rows = (0..groups)
            .map(|i| finite_f32s(&h, bytes, i, &format!("group.{i}.x"), &[rows, features]))
            .collect::<Result<_>>()?;
        let targets = finite_f32s(&h, bytes, groups, "y", &[rows])?;
        #[expect(clippy::cast_precision_loss, reason = "TabICL has at most ten classes")]
        if let Some(classes) = &p.classes
            && targets
                .iter()
                .any(|t| t.fract() != 0.0 || !(0.0..classes.len() as f32).contains(t))
        {
            return Err(artifact_error("invalid training class index"));
        }
        Some(Training { group_rows, targets })
    };
    Ok((
        h.metadata.checkpoint_sha256,
        Preprocessed {
            preprocessor: h.metadata.state,
            training,
        },
        h.metadata.context_identity,
    ))
}

// Unlike the human-readable directory summary, a fitted category dictionary must
// retain its type even when empty (all training values missing).
pub(crate) mod categories {
    use crate::Labels;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    #[derive(Serialize, Deserialize)]
    #[serde(remote = "Labels", rename_all = "lowercase")]
    enum Typed {
        Int(Vec<i64>),
        Float(Vec<f64>),
        Bool(Vec<bool>),
        Text(Vec<String>),
    }
    #[derive(Serialize)]
    struct Borrowed<'a>(#[serde(with = "Typed")] &'a Labels);
    #[derive(Deserialize)]
    struct Owned(#[serde(with = "Typed")] Labels);
    pub fn serialize<S: Serializer>(values: &[Option<Labels>], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(values.iter().map(|v| v.as_ref().map(Borrowed)))
    }
    pub fn deserialize<'de, D: Deserializer<'de>, T: From<Vec<Option<Labels>>>>(d: D) -> Result<T, D::Error> {
        Ok(Vec::<Option<Owned>>::deserialize(d)?
            .into_iter()
            .map(|v| v.map(|x| x.0))
            .collect::<Vec<_>>()
            .into())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KumoState<'a> {
    #[serde(with = "categories")]
    categories: std::borrow::Cow<'a, [Option<Labels>]>,
    feature_names: Option<std::borrow::Cow<'a, [String]>>,
    members: usize,
    seed: i128,
}

pub(crate) fn encode_kumo_tabular(t: &kumo_tabular::fitted::Training, sha: &[u8; 32]) -> Result<Vec<u8>> {
    let rows = t.labels.len() as u64;
    let meta = Metadata {
        model: ModelFamily::KumoTabular,
        checkpoint_sha256: *sha,
        context_identity: None,
        state: KumoState {
            categories: t.categories.as_slice().into(),
            feature_names: t.feature_names.as_deref().map(Into::into),
            members: t.members,
            seed: t.seed,
        },
    };
    let mut tensors = vec![(
        "x".into(),
        DType::F64,
        vec![rows, t.categories.len() as u64],
        native_bytes(&t.x)?,
    )];
    let (flags, lengths, text);
    match &t.labels {
        Labels::Int(v) => tensors.push(("y".into(), DType::I64, vec![rows], native_bytes(v)?)),
        Labels::Float(v) => tensors.push(("y".into(), DType::F64, vec![rows], native_bytes(v)?)),
        Labels::Bool(v) => {
            flags = v.iter().copied().map(u8::from).collect::<Vec<_>>();
            tensors.push(("y".into(), DType::U8, vec![rows], &flags));
        }
        Labels::Text(v) => {
            lengths = v
                .iter()
                .map(|s| i64::try_from(s.len()).map_err(artifact_error))
                .collect::<Result<Vec<_>>>()?;
            text = v.concat();
            tensors.push((
                "y.lengths".into(),
                DType::I64,
                vec![rows],
                native_bytes(&lengths)?,
            ));
            tensors.push((
                "y.utf8".into(),
                DType::U8,
                vec![text.len() as u64],
                text.as_bytes(),
            ));
        }
    }
    Writer::new(Kind::Preprocessing, MEMORY_ALIGNMENT, meta, tensors)?.bytes()
}

pub(crate) fn decode_kumo_tabular(bytes: &[u8]) -> Result<([u8; 32], kumo_tabular::fitted::Training)> {
    let h = Header::<Metadata<KumoState<'_>>>::from_bytes(bytes, Kind::Preprocessing)?;
    let state = &h.metadata.state;
    let features = state.categories.len();
    let rows = h
        .tensors
        .first()
        .and_then(|t| t.shape.first())
        .copied()
        .unwrap_or(0);
    if h.metadata.model != ModelFamily::KumoTabular
        || rows == 0
        || features == 0
        || !(1..=MAX_MEMBERS).contains(&state.members)
        || state.feature_names.as_ref().is_some_and(|v| v.len() != features)
        || !state.categories.iter().flatten().all(Labels::are_classes)
    {
        return Err(artifact_error("invalid Kumo preprocessing metadata"));
    }
    let x = decode_le(
        h.check_bytes(0, "x", DType::F64, &[rows, features as u64], bytes)?,
        f64::from_le_bytes,
    );
    #[expect(
        clippy::cast_precision_loss,
        clippy::float_cmp,
        reason = "category codes are exact integers"
    )]
    if x.chunks_exact(features).any(|row| {
        row.iter().zip(state.categories.iter()).any(|(&v, c)| {
            // Kumo's numeric preprocessing defines clipping for +/-infinity
            // and missing-value handling for NaN. Preserve those raw inputs;
            // only encoded categorical codes have a finite integer contract.
            c.as_ref()
                .is_some_and(|cats| v != -1.0 && (v.fract() != 0.0 || !(0.0..cats.len() as f64).contains(&v)))
        })
    }) {
        return Err(artifact_error("invalid Kumo training feature"));
    }
    let labels = kumo_labels(&h, bytes, rows)?;
    let KumoState {
        categories,
        feature_names,
        members,
        seed,
    } = h.metadata.state;
    Ok((
        h.metadata.checkpoint_sha256,
        kumo_tabular::fitted::Training {
            x,
            categories: categories.into_owned(),
            labels,
            members,
            seed,
            feature_names: feature_names.map(std::borrow::Cow::into_owned),
        },
    ))
}

fn kumo_labels(h: &Header<Metadata<KumoState<'_>>>, bytes: &[u8], rows: u64) -> Result<Labels> {
    match &h.tensors[..] {
        [_, y] => {
            let data = h.check_bytes(1, "y", y.dtype, &[rows], bytes)?;
            match y.dtype {
                DType::I64 => Ok(Labels::Int(decode_le(data, i64::from_le_bytes))),
                DType::F64 => {
                    if !DType::F64.all_finite(data)? {
                        return Err(artifact_error("non-finite Kumo target"));
                    }
                    Ok(Labels::Float(decode_le(data, f64::from_le_bytes)))
                }
                DType::U8 if data.iter().all(|&v| v <= 1) => {
                    Ok(Labels::Bool(data.iter().map(|&v| v == 1).collect()))
                }
                _ => Err(artifact_error("invalid target dtype or boolean")),
            }
        }
        [_, _, text] => {
            let lengths = decode_le(
                h.check_bytes(1, "y.lengths", DType::I64, &[rows], bytes)?,
                i64::from_le_bytes,
            );
            let mut data = h.check_bytes(2, "y.utf8", DType::U8, &[text.bytes as u64], bytes)?;
            let mut labels = Vec::with_capacity(lengths.len());
            for len in lengths {
                let n = usize::try_from(len).map_err(artifact_error)?;
                let s = data
                    .split_off(..n)
                    .ok_or_else(|| artifact_error("invalid label length"))?;
                labels.push(std::str::from_utf8(s).map_err(artifact_error)?.to_owned());
            }
            if !data.is_empty() {
                return Err(artifact_error("extra text label bytes"));
            }
            Ok(Labels::Text(labels))
        }
        _ => Err(artifact_error("invalid Kumo training tensors")),
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};
    use crate::{Features, Labels, NormMethod, PreprocessOptions, Preprocessed, Targets};

    fn fit(targets: &Targets) -> Preprocessed {
        let options = PreprocessOptions {
            n_estimators: 4,
            random_state: 3,
            norm_methods: vec![NormMethod::None, NormMethod::Power],
            outlier_threshold: 4.0,
        };
        let x: Vec<f64> = (0..60).map(|i| f64::from((i * 37 % 23) - 11) / 3.0).collect();
        Preprocessed::fit(&options, Features::F64(&x), 3, vec![None; 3], targets).unwrap()
    }

    #[test]
    fn unknown_model_family_is_not_routed_to_a_default_loader() {
        let meta = serde_json::json!({"model":"other", "checkpoint_sha256":"01".repeat(32), "state":{}});
        let bytes = super::Writer::new(super::Kind::Preprocessing, 64, meta, [])
            .unwrap()
            .bytes()
            .unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), &bytes).unwrap();
        for result in [
            super::identity(&bytes),
            super::file_identity(file.path(), u64::MAX),
        ] {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("unknown variant `other`")
            );
        }
    }

    #[test]
    fn corrupt_or_inconsistent_bytes_are_rejected() {
        let sha = [9; 32];
        let regressor = fit(&Targets::Values((0..20_u8).map(f32::from).collect()));
        let mut classifier = fit(&Targets::Labels(Labels::Int((0..20).map(|i| i % 3).collect())));
        classifier.preprocessor.feature_names = Some(["a", "b", "c"].map(String::from).to_vec());
        for fitted in [regressor, classifier.clone()] {
            let (preprocessor, training) = (&fitted.preprocessor, fitted.training.as_ref());
            assert!(training.is_some());
            for training in [training, None] {
                let bytes = encode(preprocessor, training, &sha, &[4; 32]).unwrap();
                let id = super::identity(&bytes).unwrap();
                assert_eq!(
                    (id.family, id.checkpoint_sha256),
                    (super::ModelFamily::TabIclV2, sha)
                );
                assert_eq!(
                    decode(&bytes).unwrap(),
                    (
                        sha,
                        Preprocessed {
                            preprocessor: preprocessor.clone(),
                            training: training.cloned()
                        },
                        training.is_none().then(|| crate::format::encode_sha256(&[4; 32]))
                    )
                );
                for index in [0, 9, 16] {
                    let mut corrupt = bytes.clone();
                    corrupt[index] ^= 1;
                    assert!(decode(&corrupt).is_err(), "byte {index}");
                }
                assert!(decode(&bytes[..bytes.len() - 1]).is_err());
            }
        }
        // A classifier cannot also carry a regression target scale.
        let mut inconsistent = classifier.preprocessor.clone();
        inconsistent.target_scale = Some((0.0, 1.0));
        assert!(
            encode(&inconsistent, None, &sha, &[4; 32])
                .and_then(|bytes| decode(&bytes))
                .is_err()
        );
        // Zero scale, inverted bounds, and a single-element NaN category must all fail validation.
        let mut zero_scale = classifier.preprocessor.clone();
        zero_scale.normalizations[0].scale[0] = 0.0;
        assert!(
            encode(&zero_scale, None, &sha, &[4; 32])
                .and_then(|bytes| decode(&bytes))
                .is_err()
        );
        let mut inverted = classifier.preprocessor.clone();
        inverted.normalizations[0].lower[0] = inverted.normalizations[0].upper[0] + 1.0;
        assert!(
            encode(&inverted, None, &sha, &[4; 32])
                .and_then(|bytes| decode(&bytes))
                .is_err()
        );
        let mut nan_category = classifier.preprocessor.clone();
        nan_category.categories[0] = Some(Labels::Float(vec![f64::NAN]));
        assert!(
            encode(&nan_category, None, &sha, &[4; 32])
                .and_then(|bytes| decode(&bytes))
                .is_err()
        );
        // Training targets that are not classes, or values that are not finite.
        let mut training = classifier.training.clone().unwrap();
        training.targets[0] = 3.0;
        assert!(decode(&encode(&classifier.preprocessor, Some(&training), &sha, &[4; 32]).unwrap()).is_err());
        let mut training = classifier.training.unwrap();
        training.group_rows[1][0] = f32::NAN;
        assert!(decode(&encode(&classifier.preprocessor, Some(&training), &sha, &[4; 32]).unwrap()).is_err());
    }

    #[test]
    fn kumo_keeps_typed_labels_empty_categories_and_large_seed() {
        for labels in [
            Labels::Bool(vec![true, false]),
            Labels::Text(vec![String::new(), "é猫".into()]),
            Labels::Float(vec![1.0, -0.5]),
            Labels::Int(vec![i64::MIN, i64::MAX]),
        ] {
            let training = crate::models::kumo_tabular::fitted::Training {
                x: vec![-1.0, -1.0],
                categories: vec![Some(Labels::Text(vec![]))],
                labels,
                members: 1,
                seed: i128::MIN,
                feature_names: None,
            };
            let bytes = super::encode_kumo_tabular(&training, &[5; 32]).unwrap();
            assert_eq!(super::decode_kumo_tabular(&bytes).unwrap().1, training);
        }
    }

    #[test]
    fn kumo_numeric_nonfinite_inputs_survive_serialization_but_category_codes_do_not() {
        let mut training = crate::models::kumo_tabular::fitted::Training {
            x: vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            categories: vec![None],
            labels: Labels::Int(vec![0, 1, 0]),
            members: 1,
            seed: 0,
            feature_names: None,
        };
        let bytes = super::encode_kumo_tabular(&training, &[5; 32]).unwrap();
        let (_, decoded) = super::decode_kumo_tabular(&bytes).unwrap();
        assert!(decoded.x[0].is_nan());
        assert_eq!(decoded.x[1].to_bits(), f64::INFINITY.to_bits());
        assert_eq!(decoded.x[2].to_bits(), f64::NEG_INFINITY.to_bits());
        training.categories = vec![Some(Labels::Text(vec!["a".into()]))];
        assert!(
            super::decode_kumo_tabular(&super::encode_kumo_tabular(&training, &[5; 32]).unwrap()).is_err()
        );
    }

    #[test]
    fn kumo_tabular_categories_must_be_sorted_and_unique() {
        let training = |categories: Labels| crate::models::kumo_tabular::fitted::Training {
            x: vec![0.0, 1.0, 1.0, 0.0],
            categories: vec![Some(categories)],
            labels: Labels::Int(vec![0, 1, 0, 1]),
            members: 2,
            seed: 7,
            feature_names: None,
        };
        let sha = [5; 32];
        let sorted = training(Labels::Text(vec!["a".into(), "b".into()]));
        let bytes = super::encode_kumo_tabular(&sorted, &sha).unwrap();
        let id = super::identity(&bytes).unwrap();
        assert_eq!(
            (id.family, id.checkpoint_sha256),
            (super::ModelFamily::KumoTabular, sha)
        );
        assert_eq!(super::decode_kumo_tabular(&bytes).unwrap().1, sorted);
        for bad in [
            Labels::Text(vec!["b".into(), "a".into()]),
            Labels::Text(vec!["a".into(), "a".into()]),
            Labels::Float(vec![f64::NAN]),
        ] {
            let bytes = super::encode_kumo_tabular(&training(bad), &sha).unwrap();
            assert!(super::decode_kumo_tabular(&bytes).is_err());
        }
    }
}
