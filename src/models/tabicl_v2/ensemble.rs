//! `TabICL`'s ensemble: each member's normalization group, feature order, and
//! class order, drawn from `CPython`'s `random.Random(random_state)` as
//! `TabICL` draws them, so it matches member for member.
//! Copyright (c) 2025, Soda team @ Inria; see `LICENSES/TabICL-BSD-3-Clause.txt`.

use super::preprocess::{NormMethod, PreprocessOptions};
use crate::preprocess::python_random::PythonRandom;

/// Above this many features, feature orders are random samples instead of a
/// Latin square's rows.
const LATIN_FEATURES: usize = 4000;

/// The feature orders of `TabICL`'s `Shuffler(method="latin")`.
///
/// For up to 4000 features they are the rows of a Latin square whose rows and
/// columns are shuffled. The square `TabICL` builds has a closed form: its first
/// row lists the features in the order they are drawn, and row `r` is that row
/// rotated right by `r`. Both shuffles draw only by the square's size, so the
/// orders come from two shuffled index lists without building the square.
///
/// Equivalent of
/// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L811-L868>
/// and <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L870-L903>.
pub(crate) enum FeatureShuffles {
    Identity(usize),
    Latin {
        first_row: Vec<usize>,
        rows: Vec<usize>,
        columns: Vec<usize>,
    },
    Sampled(Vec<Vec<usize>>),
}

impl FeatureShuffles {
    pub(crate) fn new(features: usize, members: usize, seed: i128) -> Self {
        if members == 1 {
            return Self::Identity(features);
        }
        let mut rng = PythonRandom::new(seed);
        let mut indices: Vec<usize> = (0..features).collect();
        if features > LATIN_FEATURES {
            return Self::Sampled((0..members).map(|_| rng.sample(&indices, features)).collect());
        }
        let mut first_row = Vec::with_capacity(features);
        while indices.len() > 1 {
            first_row.push(indices.remove(rng.index_below(indices.len())));
        }
        first_row.extend(indices);
        let mut rows: Vec<usize> = (0..features).collect();
        rng.shuffle(&mut rows);
        let mut columns: Vec<usize> = (0..features).collect();
        rng.shuffle(&mut columns);
        Self::Latin {
            first_row,
            rows,
            columns,
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Identity(_) => 1,
            Self::Latin { rows, .. } => rows.len(),
            Self::Sampled(orders) => orders.len(),
        }
    }

    pub(crate) fn get(&self, index: usize) -> Vec<usize> {
        match self {
            Self::Identity(features) => (0..*features).collect(),
            Self::Latin {
                first_row,
                rows,
                columns,
            } => {
                let n = first_row.len();
                rows.iter()
                    .map(|&row| first_row[(columns[index] + n - row) % n])
                    .collect()
            }
            Self::Sampled(orders) => orders[index].clone(),
        }
    }
}

/// `TabICL`'s `EnsembleGenerator`: every (feature order, class order) pair,
/// shuffled, then paired with each method in turn up to `n_estimators`. Returns
/// the methods that make groups, in option order, and each member's group,
/// feature order, and class order (empty for a regressor, `classes` `None`).
///
/// Equivalent of
/// <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L1077-L1128>
/// and <https://github.com/soda-inria/tabicl/blob/bd030d47ff7d9f987f0f33431d743a3003bfe0d3/src/tabicl/_sklearn/preprocessing.py#L811-L868>.
#[expect(clippy::type_complexity, reason = "the ensemble's four parts")]
pub(crate) fn ensemble(
    options: &PreprocessOptions,
    features: usize,
    classes: Option<usize>,
) -> (Vec<NormMethod>, Vec<usize>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let members = options.n_estimators;
    let shuffles = FeatureShuffles::new(features, members, options.random_state);
    let class_orders: Vec<Vec<usize>> = match classes {
        None => vec![Vec::new()],
        Some(classes) if members == 1 => vec![(0..classes).collect()],
        Some(classes) => (0..classes)
            .map(|shift| {
                let mut order: Vec<usize> = (0..classes).collect();
                order.rotate_right(shift);
                order
            })
            .collect(),
    };
    let mut combinations: Vec<usize> = (0..shuffles.len() * class_orders.len()).collect();
    PythonRandom::new(options.random_state).shuffle(&mut combinations);
    let methods = &options.norm_methods;
    let selected: Vec<(usize, NormMethod)> = (0..members.min(combinations.len() * methods.len()))
        .map(|k| (combinations[k / methods.len()], methods[k % methods.len()]))
        .collect();
    let group_methods: Vec<NormMethod> = methods
        .iter()
        .copied()
        .filter(|method| selected.iter().any(|(_, name)| name == method))
        .collect();
    let (mut member_groups, mut feature_permutations, mut class_permutations) = (vec![], vec![], vec![]);
    for (group, &method) in group_methods.iter().enumerate() {
        for &(combination, _) in selected.iter().filter(|(_, name)| *name == method) {
            member_groups.push(group);
            feature_permutations.push(shuffles.get(combination / class_orders.len()));
            class_permutations.push(class_orders[combination % class_orders.len()].clone());
        }
    }
    (
        group_methods,
        member_groups,
        feature_permutations,
        class_permutations,
    )
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::FeatureShuffles;
    use crate::preprocess::{fixture, python_random::PythonRandom, rows};

    /// Every feature order of `FeatureShuffles`.
    fn feature_shuffles(features: usize, members: usize, seed: i128) -> Vec<Vec<usize>> {
        let shuffles = FeatureShuffles::new(features, members, seed);
        (0..shuffles.len()).map(|index| shuffles.get(index)).collect()
    }

    fn count(value: &Value) -> usize {
        usize::try_from(value.as_u64().unwrap()).unwrap()
    }

    #[test]
    fn feature_shuffles_match_cpython() {
        for case in fixture("python_random.json")["feature_shuffles"]
            .as_array()
            .unwrap()
        {
            let (features, members) = (count(&case["features"]), count(&case["estimators"]));
            let seed: i128 = case["seed"].as_str().unwrap().parse().unwrap();
            let orders = feature_shuffles(features, members, seed);
            assert_eq!(orders.len(), count(&case["count"]), "{features} features");
            if let Some(expected) = case.get("orders") {
                assert_eq!(orders, rows(expected), "{features} features");
            } else {
                let digest = orders
                    .iter()
                    .flatten()
                    .fold(0xcbf2_9ce4_8422_2325_u64, |state, &value| {
                        (state ^ value as u64).wrapping_mul(0x0100_0000_01b3)
                    });
                assert_eq!(
                    format!("{digest:016x}"),
                    case["digest"].as_str().unwrap(),
                    "{features} features"
                );
            }
        }
    }

    /// `TabICL`'s Latin square, built as it builds it.
    fn literal_latin_square(features: usize, seed: i128) -> Vec<Vec<usize>> {
        let mut rng = PythonRandom::new(seed);
        let mut indices: Vec<usize> = (0..features).collect();
        let mut removed = Vec::new();
        while indices.len() > 1 {
            removed.push(indices.remove(rng.index_below(indices.len())));
        }
        let mut square = vec![indices];
        for &symbol in removed.iter().rev() {
            square.push(square[0].clone());
            for (index, row) in square.iter_mut().enumerate() {
                row.insert(index, symbol);
            }
        }
        rng.shuffle(&mut square);
        let mut transposed: Vec<Vec<usize>> = (0..features)
            .map(|column| square.iter().map(|row| row[column]).collect())
            .collect();
        rng.shuffle(&mut transposed);
        transposed
    }

    #[test]
    fn the_closed_form_is_the_square_tabicl_builds() {
        for features in 1..40 {
            for seed in [0, 1, 42, -3, 1 << 40] {
                assert_eq!(
                    feature_shuffles(features, 8, seed),
                    literal_latin_square(features, seed)
                );
            }
        }
    }

    #[test]
    fn one_member_keeps_the_feature_order() {
        assert_eq!(feature_shuffles(5, 1, 7), vec![vec![0, 1, 2, 3, 4]]);
        let sampled = FeatureShuffles::new(4001, 3, 0);
        assert_eq!(sampled.len(), 3);
        let mut order = sampled.get(2);
        order.sort_unstable();
        assert_eq!(order, (0..4001).collect::<Vec<_>>());
    }
}
