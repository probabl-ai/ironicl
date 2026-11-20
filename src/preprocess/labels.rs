//! Class labels, encoded as scikit-learn's `LabelEncoder` does: the classes are
//! the sorted distinct labels, and a label's code is its class's index.

use std::cmp::Ordering;

use crate::{Error, Result};

/// One label per row, all of one type. Text sorts by Unicode code point, as
/// `NumPy` sorts strings.
#[derive(Clone, Debug, PartialEq)]
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
    use super::Labels;

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
