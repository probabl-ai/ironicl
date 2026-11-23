//! Typed feature columns, and categorical features encoded as `TabICL` encodes
//! them with scikit-learn's `OrdinalEncoder`: a feature's categories are the
//! sorted distinct values its training rows use, and a value's code is its
//! category's index, or -1 when it is missing or not a category.

use super::labels::Labels;
use crate::{Error, Result};

/// The code of a missing value or of one that is not a category:
/// `OrdinalEncoder`'s `encoded_missing_value` and `unknown_value` in `TabICL`.
pub const MISSING: f64 = -1.0;

/// One feature's values, one per row.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// Numbers; NaN marks a missing value.
    Numeric(Vec<f64>),
    /// Each row's index in `categories`, or a negative code for a missing
    /// value. The categories need not be sorted, distinct, or all used.
    Categorical { categories: Labels, codes: Vec<i64> },
}

impl Column {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Categorical { codes, .. } => codes.len(),
        }
    }

    /// The index in `categories` of each row's value, `None` when it is missing.
    fn indices(categories: &Labels, codes: &[i64]) -> Result<impl Iterator<Item = Option<usize>>> {
        let count = categories.len();
        if codes
            .iter()
            .any(|&code| usize::try_from(code).is_ok_and(|code| code >= count))
        {
            return Err(Error::InvalidArgument(
                "a categorical code is not an index of its categories".into(),
            ));
        }
        Ok(codes.iter().map(|&code| usize::try_from(code).ok()))
    }
}

/// Check that `columns` is a nonempty table of equal-length columns.
fn check(columns: &[Column]) -> Result<usize> {
    let rows = columns.first().map_or(0, Column::len);
    if rows == 0 || columns.iter().any(|column| column.len() != rows) {
        return Err(Error::InvalidArgument(
            "X must be a nonempty table of equal-length columns".into(),
        ));
    }
    Ok(rows)
}

/// Each column's categories: the sorted distinct values its rows use, or
/// `None` for a numeric column.
///
/// Port of <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_encoders.py#L103-L112>.
///
/// # Errors
///
/// When the columns differ in length, a code is out of range, or a category is NaN.
pub fn categories(columns: &[Column]) -> Result<Vec<Option<Labels>>> {
    check(columns)?;
    columns
        .iter()
        .map(|column| match column {
            Column::Numeric(_) => Ok(None),
            Column::Categorical { categories, codes } => {
                let mut used = vec![false; categories.len()];
                for index in Column::indices(categories, codes)?.flatten() {
                    used[index] = true;
                }
                let indices: Vec<usize> = (0..used.len()).filter(|&index| used[index]).collect();
                categories
                    .select(&indices)
                    .classes()
                    .map(Some)
                    .map_err(|_| Error::InvalidArgument("a category must not be NaN".into()))
            }
        })
        .collect()
}

/// `columns` as a C-ordered `[rows, columns]` matrix: numbers as they are, and
/// each categorical value as its index in that column's `categories`, or
/// [`MISSING`].
///
/// Port of <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_encoders.py#L1598-L1615>.
///
/// # Errors
///
/// When the columns differ in length or in kind from `categories`, a code is
/// out of range, or a category's type differs from its feature's categories.
pub fn encode(columns: &[Column], categories: &[Option<Labels>]) -> Result<Vec<f64>> {
    let rows = check(columns)?;
    if columns.len() != categories.len() {
        return Err(Error::InvalidArgument(format!(
            "X has {} features; expected {}",
            columns.len(),
            categories.len()
        )));
    }
    let width = columns.len();
    let mut matrix = vec![0.0; rows * width];
    for (j, (column, fitted)) in columns.iter().zip(categories).enumerate() {
        let slots = matrix.iter_mut().skip(j).step_by(width);
        match (column, fitted) {
            (Column::Numeric(values), None) => slots.zip(values).for_each(|(slot, &value)| *slot = value),
            (Column::Categorical { categories, codes }, Some(fitted)) => {
                let positions = (0..categories.len())
                    .map(|index| fitted.position(categories.get(index)))
                    .collect::<Result<Vec<_>>>()
                    .map_err(|_| {
                        Error::InvalidArgument(format!(
                            "feature {j} has categories of a different type than in training"
                        ))
                    })?;
                for (slot, index) in slots.zip(Column::indices(categories, codes)?) {
                    #[expect(clippy::cast_precision_loss, reason = "Category counts stay far below 2^53")]
                    let code = index
                        .and_then(|index| positions[index])
                        .map_or(MISSING, |code| code as f64);
                    *slot = code;
                }
            }
            (Column::Numeric(_), Some(_)) => {
                return Err(Error::InvalidArgument(format!(
                    "feature {j} is numeric but was categorical in training"
                )));
            }
            (Column::Categorical { .. }, None) => {
                return Err(Error::InvalidArgument(format!(
                    "feature {j} is categorical but was numeric in training"
                )));
            }
        }
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::{Column, MISSING, categories, encode};
    use crate::preprocess::labels::Labels;

    fn text(values: &[&str]) -> Labels {
        Labels::Text(values.iter().map(|&value| value.into()).collect())
    }

    #[test]
    fn categories_are_sorted_and_others_are_missing() {
        let training = [
            Column::Numeric(vec![1.5, f64::NAN, 3.0]),
            // "unused" is a category no row has, as in a pandas Categorical.
            Column::Categorical {
                categories: text(&["y", "unused", "x"]),
                codes: vec![0, -1, 2],
            },
            Column::Categorical {
                categories: Labels::Int(vec![10, 2]),
                codes: vec![0, 1, 0],
            },
        ];
        let fitted = categories(&training).unwrap();
        assert_eq!(
            fitted,
            [None, Some(text(&["x", "y"])), Some(Labels::Int(vec![2, 10]))]
        );
        let matrix = encode(&training, &fitted).unwrap();
        let expected = [1.5, 1.0, 1.0, f64::NAN, MISSING, 0.0, 3.0, 0.0, 1.0];
        assert!(matrix.iter().zip(expected).all(|(a, b)| a.total_cmp(&b).is_eq()));

        // A category training never saw is missing; floats find integers.
        let query = [
            Column::Numeric(vec![0.0, 0.0]),
            Column::Categorical {
                categories: text(&["z", "y"]),
                codes: vec![0, 1],
            },
            Column::Categorical {
                categories: Labels::Float(vec![10.0, 2.5]),
                codes: vec![0, 1],
            },
        ];
        assert_eq!(
            encode(&query, &fitted).unwrap(),
            [0.0, MISSING, 1.0, 0.0, 1.0, MISSING]
        );

        let numbers_for_text = [
            Column::Numeric(vec![0.0]),
            Column::Categorical {
                categories: Labels::Int(vec![1]),
                codes: vec![0],
            },
            Column::Categorical {
                categories: Labels::Int(vec![2]),
                codes: vec![0],
            },
        ];
        assert!(encode(&numbers_for_text, &fitted).is_err());
        assert!(encode(&query[..2], &fitted).is_err());
        let out_of_range = [Column::Categorical {
            categories: text(&["a"]),
            codes: vec![1],
        }];
        assert!(categories(&out_of_range).is_err());
        assert!(categories(&[Column::Numeric(vec![1.0]), Column::Numeric(vec![])]).is_err());
        assert!(categories(&[]).is_err());
    }
}
