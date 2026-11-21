//! Fitted `TabICL` preprocessing as `IRPREP` bytes: what pickling an estimator
//! saves beside its context, and what the server loads. The bytes may come
//! from outside, so decoding checks the digest, checks every length against the
//! bytes left before it allocates, and then checks the preprocessing's
//! invariants.
//!
//! Little-endian, after the magic and version:
//! - the checkpoint's SHA-256 (32 bytes, as `IRCNTXT` stores it);
//! - input features (u64) and their mask (u8 each);
//! - groups (u8) and their methods (u8: 0 none, 1 power);
//! - members (u64), each one's group (u8), feature order (u64 each), and class
//!   order (u64 each, for the class count below);
//! - classes (u64) and their labels: a kind (u8: 0 none, 1 int, 2 float,
//!   3 bool, 4 text), then i64, f64, u8, or u32-length UTF-8 values;
//! - for a regressor (no labels), the target's mean and scale (f64 each);
//!   for a classifier, its softmax temperature (f32) and whether it averages
//!   logits (u8);
//! - feature names (u64: 0, or one per input feature), each a u32-length UTF-8
//!   string;
//! - per group, f64 per kept feature: mean, scale, minimum, maximum, lower and
//!   upper bounds, and for a power group its lambdas, mean, and scale;
//! - the SHA-256 of everything before it.

use super::{
    ByteReader, CHECKSUM_BYTES, artifact_error, sha256, u64_to_usize, usize_to_u32, usize_to_u64, write_u32,
    write_u64,
};
use crate::models::tabicl_v2::preprocess::{MAX_MEMBERS, NormMethod, Normalization, Preprocessor};
use crate::preprocess::labels::Labels;
use crate::{ClassifierAggregation, Result};

pub const PREPROCESSOR_MAGIC: [u8; 8] = *b"IRPREP\0\0";
pub const PREPROCESSOR_VERSION: u32 = 2;
/// More classes than any classifier fit makes, so a corrupt count fails early.
const MAX_CLASSES: usize = 10;

fn write_count(bytes: &mut Vec<u8>, count: usize) -> Result<()> {
    write_u64(bytes, usize_to_u64(count, "a count")?);
    Ok(())
}

/// `preprocessor` as `IRPREP` bytes, for the checkpoint whose SHA-256 is
/// `checkpoint_sha256`.
pub(crate) fn encode(preprocessor: &Preprocessor, checkpoint_sha256: &[u8; 32]) -> Result<Vec<u8>> {
    let mut bytes = PREPROCESSOR_MAGIC.to_vec();
    write_u32(&mut bytes, PREPROCESSOR_VERSION);
    bytes.extend(checkpoint_sha256);
    write_count(&mut bytes, preprocessor.feature_mask.len())?;
    bytes.extend(preprocessor.feature_mask.iter().map(|&kept| u8::from(kept)));
    // A fit makes at most two groups, so each count and index fits in a byte.
    bytes.push(
        u8::try_from(preprocessor.normalizations.len()).map_err(|_| artifact_error("too many groups"))?,
    );
    bytes.extend(
        preprocessor
            .normalizations
            .iter()
            .map(|normalization| match normalization.method() {
                NormMethod::None => 0,
                NormMethod::Power => 1,
            }),
    );
    write_count(&mut bytes, preprocessor.member_groups.len())?;
    for &group in &preprocessor.member_groups {
        bytes.push(u8::try_from(group).map_err(|_| artifact_error("too many groups"))?);
    }
    for &index in preprocessor.feature_permutations.iter().flatten() {
        write_count(&mut bytes, index)?;
    }
    write_count(&mut bytes, preprocessor.classes.as_ref().map_or(0, Labels::len))?;
    for &index in preprocessor.class_permutations.iter().flatten() {
        write_count(&mut bytes, index)?;
    }
    match &preprocessor.classes {
        None => bytes.push(0),
        Some(Labels::Int(values)) => {
            bytes.push(1);
            for value in values {
                bytes.extend(value.to_le_bytes());
            }
        }
        Some(Labels::Float(values)) => {
            bytes.push(2);
            for value in values {
                bytes.extend(value.to_le_bytes());
            }
        }
        Some(Labels::Bool(values)) => {
            bytes.push(3);
            bytes.extend(values.iter().map(|&value| u8::from(value)));
        }
        Some(Labels::Text(values)) => {
            bytes.push(4);
            write_strings(&mut bytes, values)?;
        }
    }
    if let Some((mean, scale)) = preprocessor.target_scale {
        bytes.extend(mean.to_le_bytes());
        bytes.extend(scale.to_le_bytes());
    }
    if let Some(aggregation) = preprocessor.aggregation {
        bytes.extend(aggregation.softmax_temperature.to_le_bytes());
        bytes.push(u8::from(aggregation.average_logits));
    }
    let names = preprocessor.feature_names.as_deref().unwrap_or_default();
    write_count(&mut bytes, names.len())?;
    write_strings(&mut bytes, names)?;
    for normalization in &preprocessor.normalizations {
        for value in normalization.statistics().into_iter().flatten() {
            bytes.extend(value.to_le_bytes());
        }
    }
    let digest = sha256(&bytes);
    bytes.extend(digest);
    Ok(bytes)
}

fn write_strings(bytes: &mut Vec<u8>, values: &[String]) -> Result<()> {
    for value in values {
        write_u32(bytes, usize_to_u32(value.len(), "a string's length")?);
        bytes.extend(value.as_bytes());
    }
    Ok(())
}

fn read_strings(reader: &mut ByteReader<'_>, count: usize) -> Result<Vec<String>> {
    (0..count)
        .map(|_| {
            let length = reader.read_u32()? as usize;
            String::from_utf8(reader.read_bytes(length)?.to_vec())
                .map_err(|_| artifact_error("IRPREP has a string that is not UTF-8"))
        })
        .collect()
}

fn read_flags(reader: &mut ByteReader<'_>, count: usize) -> Result<Vec<bool>> {
    reader
        .read_chunks(count, 1)?
        .map(|byte| match byte[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(artifact_error("IRPREP has an invalid flag")),
        })
        .collect()
}

fn read_orders(reader: &mut ByteReader<'_>, members: usize, length: usize) -> Result<Vec<Vec<usize>>> {
    let total = members
        .checked_mul(length)
        .ok_or_else(|| artifact_error("IRPREP length overflows usize"))?;
    let indices = reader
        .read_chunks(total, 8)?
        .map(|bytes| {
            u64_to_usize(
                u64::from_le_bytes(bytes.try_into().expect("eight bytes")),
                "an index",
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(if length == 0 {
        vec![Vec::new(); members]
    } else {
        indices.chunks_exact(length).map(<[usize]>::to_vec).collect()
    })
}

/// The checkpoint's SHA-256 and the preprocessor that `encode` wrote.
pub(crate) fn decode(bytes: &[u8]) -> Result<([u8; 32], Preprocessor)> {
    let header = PREPROCESSOR_MAGIC.len() + 4 + 32;
    if bytes.len() < header + CHECKSUM_BYTES {
        return Err(artifact_error("IRPREP is truncated"));
    }
    let (body, digest) = bytes.split_at(bytes.len() - CHECKSUM_BYTES);
    if sha256(body) != digest {
        return Err(artifact_error("IRPREP checksum mismatch"));
    }
    let mut reader = ByteReader::new(body);
    if reader.read_array::<8>()? != PREPROCESSOR_MAGIC {
        return Err(artifact_error("not fitted preprocessing (IRPREP)"));
    }
    let version = reader.read_u32()?;
    if version != PREPROCESSOR_VERSION {
        return Err(artifact_error(format!(
            "unsupported IRPREP version {version}; expected {PREPROCESSOR_VERSION}"
        )));
    }
    let checkpoint_sha256 = reader.read_array::<32>()?;
    let features_in = reader.read_count("input features")?;
    let feature_mask = read_flags(&mut reader, features_in)?;
    let features = feature_mask.iter().filter(|kept| **kept).count();
    let group_count = usize::from(reader.read_u8()?);
    let groups = reader
        .read_chunks(group_count, 1)?
        .map(|byte| match byte[0] {
            0 => Ok(NormMethod::None),
            1 => Ok(NormMethod::Power),
            _ => Err(artifact_error("IRPREP has an invalid method")),
        })
        .collect::<Result<Vec<_>>>()?;
    let members = reader.read_count("members")?;
    if members > MAX_MEMBERS {
        return Err(artifact_error("IRPREP has too many members"));
    }
    let member_groups = reader
        .read_chunks(members, 1)?
        .map(|byte| usize::from(byte[0]))
        .collect();
    let feature_permutations = read_orders(&mut reader, members, features)?;
    let classes = reader.read_count("classes")?;
    if classes > MAX_CLASSES {
        return Err(artifact_error("IRPREP has too many classes"));
    }
    let class_permutations = read_orders(&mut reader, members, classes)?;
    let labels = read_labels(&mut reader, classes)?;
    let (target_scale, aggregation) = if labels.is_none() {
        let values = reader.read_f64s(2)?;
        (Some((values[0], values[1])), None)
    } else {
        let aggregation = ClassifierAggregation {
            softmax_temperature: f32::from_le_bytes(reader.read_array()?),
            average_logits: read_flags(&mut reader, 1)?[0],
        };
        (None, Some(aggregation))
    };
    let names = reader.read_count("feature names")?;
    if names != 0 && names != features_in {
        return Err(artifact_error("IRPREP has a name for some features only"));
    }
    let feature_names = (names != 0)
        .then(|| read_strings(&mut reader, names))
        .transpose()?;
    let normalizations = groups
        .iter()
        .map(|&method| read_normalization(&mut reader, method, features))
        .collect::<Result<_>>()?;
    if !reader.is_empty() {
        return Err(artifact_error("IRPREP has trailing bytes"));
    }
    let preprocessor = Preprocessor {
        feature_mask,
        normalizations,
        member_groups,
        feature_permutations,
        class_permutations,
        classes: labels,
        target_scale,
        feature_names,
        aggregation,
    };
    preprocessor.validate()?;
    Ok((checkpoint_sha256, preprocessor))
}

fn read_labels(reader: &mut ByteReader<'_>, classes: usize) -> Result<Option<Labels>> {
    let labels = match reader.read_u8()? {
        0 => None,
        1 => Some(Labels::Int(
            reader
                .read_chunks(classes, 8)?
                .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight bytes")))
                .collect(),
        )),
        2 => Some(Labels::Float(reader.read_f64s(classes)?)),
        3 => Some(Labels::Bool(read_flags(reader, classes)?)),
        4 => Some(Labels::Text(read_strings(reader, classes)?)),
        _ => return Err(artifact_error("IRPREP has an invalid label kind")),
    };
    if labels.as_ref().map_or(0, Labels::len) != classes {
        return Err(artifact_error("IRPREP has classes without labels"));
    }
    Ok(labels)
}

fn read_normalization(
    reader: &mut ByteReader<'_>,
    method: NormMethod,
    features: usize,
) -> Result<Normalization> {
    let count = match method {
        NormMethod::None => 6,
        NormMethod::Power => 9,
    };
    let statistics = (0..count)
        .map(|_| reader.read_f64s(features))
        .collect::<Result<_>>()?;
    Ok(Normalization::from_statistics(method, statistics))
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};
    use crate::{Features, Labels, NormMethod, PreprocessOptions, Preprocessor, Targets};

    fn fit(targets: &Targets) -> Preprocessor {
        let options = PreprocessOptions {
            n_estimators: 4,
            random_state: 3,
            norm_methods: vec![NormMethod::None, NormMethod::Power],
            outlier_threshold: 4.0,
        };
        let x: Vec<f64> = (0..60).map(|i| f64::from((i * 37 % 23) - 11) / 3.0).collect();
        Preprocessor::fit(&options, Features::F64(&x), 3, targets).unwrap()
    }

    #[test]
    fn corrupt_or_inconsistent_bytes_are_rejected() {
        let sha = [9; 32];
        let regressor = fit(&Targets::Values((0..20_u8).map(f32::from).collect()));
        let mut classifier = fit(&Targets::Labels(Labels::Int((0..20).map(|i| i % 3).collect())));
        classifier.feature_names = Some(["a", "b", "c"].map(String::from).to_vec());
        for preprocessor in [regressor, classifier.clone()] {
            let bytes = encode(&preprocessor, &sha).unwrap();
            assert_eq!(decode(&bytes).unwrap(), (sha, preprocessor));
            for index in [0, 9, 20, bytes.len() / 2, bytes.len() - 1] {
                let mut corrupt = bytes.clone();
                corrupt[index] ^= 1;
                assert!(decode(&corrupt).is_err(), "byte {index}");
            }
            assert!(decode(&bytes[..bytes.len() - 1]).is_err());
        }
        // A classifier with a target scale, digest and all.
        let mut inconsistent = classifier;
        inconsistent.target_scale = Some((0.0, 1.0));
        assert!(decode(&encode(&inconsistent, &sha).unwrap()).is_err());
    }
}
