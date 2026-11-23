//! Class labels, encoded as scikit-learn's `LabelEncoder` does: the classes are
//! the sorted distinct labels, and a label's code is its class's index. A
//! categorical feature's categories are labels too.

use std::cmp::Ordering;

use crate::{Error, Result};

/// One label per row, all of one type. Text sorts by Unicode code point, as
/// `NumPy` sorts strings.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Labels {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
    Text(Vec<String>),
}

fn distinct<T: Clone>(values: &[T], order: impl Fn(&T, &T) -> Ordering) -> Vec<T> {
    let mut sorted = values.to_vec();
    sorted.sort_by(&order);
    sorted.dedup_by(|a, b| order(a, b) == Ordering::Equal);
    sorted
}

fn codes<T>(values: &[T], classes: &[T], order: impl Fn(&T, &T) -> Ordering) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|value| {
            classes
                .binary_search_by(|class| order(class, value))
                .map_err(|_| Error::InvalidArgument("y contains labels the training labels did not".into()))
        })
        .collect()
}

fn float_order(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).expect("labels are not NaN")
}

/// One label, to look up among classes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Label<'a> {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(&'a str),
}

impl Labels {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Int(values) => values.len(),
            Self::Float(values) => values.len(),
            Self::Bool(values) => values.len(),
            Self::Text(values) => values.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The label at `index`, below [`Self::len`].
    pub(crate) fn get(&self, index: usize) -> Label<'_> {
        match self {
            Self::Int(values) => Label::Int(values[index]),
            Self::Float(values) => Label::Float(values[index]),
            Self::Bool(values) => Label::Bool(values[index]),
            Self::Text(values) => Label::Text(&values[index]),
        }
    }

    /// The index of `label` in these classes, `None` when it is not one of
    /// them or is NaN. Integers and floats compare as numbers, as `NumPy`
    /// compares them.
    ///
    /// # Errors
    ///
    /// When `label` is a boolean or text and these classes are not, or the
    /// reverse.
    pub(crate) fn position(&self, label: Label<'_>) -> Result<Option<usize>> {
        let found = |result: std::result::Result<usize, usize>| Ok(result.ok());
        match (self, label) {
            (Self::Int(classes), Label::Int(value)) => found(classes.binary_search(&value)),
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Only integral values in range are cast"
            )]
            (Self::Int(classes), Label::Float(value)) => {
                // 2^63 is exact in f64; values at or above it are no i64.
                if value.fract() == 0.0 && (-(2.0_f64.powi(63))..2.0_f64.powi(63)).contains(&value) {
                    found(classes.binary_search(&(value as i64)))
                } else {
                    Ok(None)
                }
            }
            #[expect(clippy::cast_precision_loss, reason = "NumPy compares them as float64 too")]
            (Self::Float(_), Label::Int(value)) => self.position(Label::Float(value as f64)),
            (Self::Float(_), Label::Float(value)) if value.is_nan() => Ok(None),
            (Self::Float(classes), Label::Float(value)) => {
                found(classes.binary_search_by(|class| float_order(*class, value)))
            }
            (Self::Bool(classes), Label::Bool(value)) => found(classes.binary_search(&value)),
            (Self::Text(classes), Label::Text(value)) => {
                found(classes.binary_search_by(|class| class.as_str().cmp(value)))
            }
            _ => Err(Error::InvalidArgument(
                "a value has a different type than the training values".into(),
            )),
        }
    }

    /// The labels at `indices`, each below [`Self::len`].
    pub(crate) fn select(&self, indices: &[usize]) -> Self {
        fn pick<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
            indices.iter().map(|&index| values[index].clone()).collect()
        }
        match self {
            Self::Int(values) => Self::Int(pick(values, indices)),
            Self::Float(values) => Self::Float(pick(values, indices)),
            Self::Bool(values) => Self::Bool(pick(values, indices)),
            Self::Text(values) => Self::Text(pick(values, indices)),
        }
    }

    /// The sorted distinct labels.
    ///
    /// Port of <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_label.py#L90-L105>.
    pub(crate) fn classes(&self) -> Result<Self> {
        Ok(match self {
            Self::Int(values) => Self::Int(distinct(values, Ord::cmp)),
            Self::Float(values) => {
                if values.iter().any(|value| value.is_nan()) {
                    return Err(Error::InvalidArgument("labels must not be NaN".into()));
                }
                Self::Float(distinct(values, |a, b| float_order(*a, *b)))
            }
            Self::Bool(values) => Self::Bool(distinct(values, Ord::cmp)),
            Self::Text(values) => Self::Text(distinct(values, Ord::cmp)),
        })
    }

    /// Whether these labels are sorted and distinct, as classes are.
    pub(crate) fn are_classes(&self) -> bool {
        match self {
            Self::Int(values) => values.is_sorted_by(|a, b| a < b),
            Self::Float(values) => values.is_sorted_by(|a, b| a < b),
            Self::Bool(values) => values.is_sorted_by(|a, b| a < b),
            Self::Text(values) => values.is_sorted_by(|a, b| a < b),
        }
    }

    /// Each label's index in `classes`.
    ///
    /// Port of <https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/preprocessing/_label.py#L124-L144>.
    pub(crate) fn encode(&self, classes: &Self) -> Result<Vec<usize>> {
        match (self, classes) {
            (Self::Int(values), Self::Int(classes)) => codes(values, classes, Ord::cmp),
            (Self::Float(values), Self::Float(classes)) => {
                if values.iter().any(|value| value.is_nan()) {
                    return Err(Error::InvalidArgument("labels must not be NaN".into()));
                }
                codes(values, classes, |a, b| float_order(*a, *b))
            }
            (Self::Bool(values), Self::Bool(classes)) => codes(values, classes, Ord::cmp),
            (Self::Text(values), Self::Text(classes)) => codes(values, classes, Ord::cmp),
            _ => Err(Error::InvalidArgument(
                "y has a different label type than the training labels".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Label, Labels};

    #[test]
    fn positions_compare_numbers_as_numbers() {
        let ints = Labels::Int(vec![-3, 2, 10]);
        let floats = Labels::Float(vec![-0.5, 2.0]);
        for (classes, label, position) in [
            (&ints, Label::Int(10), Some(2)),
            (&ints, Label::Int(5), None),
            (&ints, Label::Float(2.0), Some(1)),
            (&ints, Label::Float(2.5), None),
            (&ints, Label::Float(1e300), None),
            (&ints, Label::Float(f64::NAN), None),
            (&floats, Label::Int(2), Some(1)),
            (&floats, Label::Float(-0.5), Some(0)),
            (&floats, Label::Float(f64::NAN), None),
        ] {
            assert_eq!(classes.position(label).unwrap(), position, "{label:?}");
        }
        let text = Labels::Text(vec!["a".into(), "b".into()]);
        assert_eq!(text.position(Label::Text("b")).unwrap(), Some(1));
        assert_eq!(
            Labels::Bool(vec![false, true])
                .position(Label::Bool(true))
                .unwrap(),
            Some(1)
        );
        assert!(text.position(Label::Int(1)).is_err());
        assert!(ints.position(Label::Text("1")).is_err());
        assert!(ints.position(Label::Bool(true)).is_err());
    }

    #[test]
    fn classes_are_sorted_and_codes_index_them() {
        let labels = Labels::Text(["zebra", "ant", "yak", "ant", "Émile"].map(String::from).to_vec());
        let classes = labels.classes().unwrap();
        assert_eq!(
            classes,
            Labels::Text(["ant", "yak", "zebra", "Émile"].map(String::from).to_vec())
        );
        assert!(classes.are_classes());
        assert_eq!(labels.encode(&classes).unwrap(), [2, 0, 1, 0, 3]);
        let floats = Labels::Float(vec![0.5, -0.0, 0.0, 2.0]);
        assert_eq!(floats.classes().unwrap(), Labels::Float(vec![-0.0, 0.5, 2.0]));
        assert!(Labels::Text(vec!["cat".into()]).encode(&classes).is_err());
        assert!(Labels::Int(vec![1]).encode(&classes).is_err());
        assert!(Labels::Float(vec![f64::NAN]).classes().is_err());
    }
}
