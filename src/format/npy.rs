//! `NumPy`'s `.npy` arrays, as `np.save` writes the training data of a saved
//! model: little-endian `float32`, `float64`, or `int64`, in C order.
//!
//! Format: <https://numpy.org/doc/2.3/reference/generated/numpy.lib.format.html>.

use std::path::Path;

use super::artifact_error;
use crate::Result;

#[derive(Debug, PartialEq)]
pub enum Npy {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

/// The array at `path` and its shape.
///
/// # Errors
///
/// When the file cannot be read or holds another kind of array.
pub fn read(path: &Path) -> Result<(Npy, Vec<usize>)> {
    let bytes =
        std::fs::read(path).map_err(|error| artifact_error(format!("{}: {error}", path.display())))?;
    parse(&bytes).ok_or_else(|| {
        artifact_error(format!(
            "{} is not a C-ordered float32, float64, or int64 .npy array",
            path.display()
        ))
    })
}

fn parse(bytes: &[u8]) -> Option<(Npy, Vec<usize>)> {
    let rest = bytes.strip_prefix(b"\x93NUMPY")?;
    let (length, rest) = match rest {
        [1, _, a, b, rest @ ..] => (usize::from(u16::from_le_bytes([*a, *b])), rest),
        [2 | 3, _, a, b, c, d, rest @ ..] => {
            (usize::try_from(u32::from_le_bytes([*a, *b, *c, *d])).ok()?, rest)
        }
        _ => return None,
    };
    let header = std::str::from_utf8(rest.get(..length)?).ok()?;
    let data = &rest[length..];
    let field = |key: &str| header.split_once(&format!("'{key}': ")).map(|(_, value)| value);
    let descr = field("descr")?.strip_prefix('\'')?.split('\'').next()?;
    if !field("fortran_order")?.starts_with("False") {
        return None;
    }
    let shape = field("shape")?.strip_prefix('(')?.split(')').next()?;
    let shape: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|size| !size.is_empty())
        .map(|size| size.parse().ok())
        .collect::<Option<_>>()?;
    let count = shape
        .iter()
        .try_fold(1_usize, |count, &size| count.checked_mul(size))?;
    let array = match descr {
        "<f4" => Npy::F32(values(data, count, f32::from_le_bytes)?),
        "<f8" => Npy::F64(values(data, count, f64::from_le_bytes)?),
        "<i8" => Npy::I64(values(data, count, i64::from_le_bytes)?),
        _ => return None,
    };
    Some((array, shape))
}

fn values<T, const N: usize>(data: &[u8], count: usize, from: fn([u8; N]) -> T) -> Option<Vec<T>> {
    if data.len() != count.checked_mul(N)? {
        return None;
    }
    Some(data.as_chunks::<N>().0.iter().map(|&chunk| from(chunk)).collect())
}

#[cfg(test)]
mod tests {
    use super::{Npy, parse};

    /// `np.save` of `[[1.5, -2.0]]`, with `dict` as its header.
    fn npy(dict: &str) -> Vec<u8> {
        let header = format!("{dict:<117}\n");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend(u16::try_from(header.len()).unwrap().to_le_bytes());
        bytes.extend(header.as_bytes());
        for value in [1.5_f64, -2.0] {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn reads_what_numpy_writes() {
        let bytes = npy("{'descr': '<f8', 'fortran_order': False, 'shape': (1, 2), }");
        assert_eq!(parse(&bytes), Some((Npy::F64(vec![1.5, -2.0]), vec![1, 2])));
        assert_eq!(parse(&bytes[..bytes.len() - 1]), None);
        assert_eq!(
            parse(&npy("{'descr': '<f8', 'fortran_order': True, 'shape': (1, 2), }")),
            None
        );
        assert_eq!(
            parse(&npy(
                "{'descr': '>f8', 'fortran_order': False, 'shape': (1, 2), }"
            )),
            None
        );
    }
}
