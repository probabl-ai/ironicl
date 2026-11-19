//! `CPython`'s `random.Random`, for the draws `TabICL`'s preprocessing makes.
//!
//! Seeding and the Mersenne Twister core are `CPython`'s `_random` module:
//! `init_by_array` over the seed's absolute value, in 32-bit words from the
//! least significant. `randbelow`, `choice`, `shuffle`, and `sample` follow
//! `Lib/random.py`. `CPython` promises only `random()` across versions; the golden
//! draws in `tests/fixtures/python_random.json` pin the rest.
//!
//! The methods follow `CPython` 3.12's `Lib/random.py`. Copyright (c) 2001-2023
//! Python Software Foundation; All Rights Reserved; see `LICENSES/Python-PSF-2.0.txt`.

use rand_mt::Mt;

pub(crate) struct PythonRandom(Mt);

impl PythonRandom {
    /// `random.Random(seed)` for an integer `seed`.
    ///
    /// Port of
    /// <https://github.com/python/cpython/blob/v3.12.13/Lib/random.py#L135-L171>
    /// and <https://github.com/python/cpython/blob/v3.12.13/Modules/_randommodule.c#L282-L363>.
    #[expect(clippy::cast_possible_truncation, reason = "Takes the low 32-bit word")]
    pub(crate) fn new(seed: i128) -> Self {
        let mut magnitude = seed.unsigned_abs();
        let mut key = Vec::new();
        while magnitude > 0 {
            key.push(magnitude as u32);
            magnitude >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        Self(Mt::new_with_key(key))
    }

    /// `getrandbits(bits)` for `1 <= bits <= 64`: words from the least
    /// significant, the last one shifted down to its bits.
    ///
    /// Port of <https://github.com/python/cpython/blob/v3.12.13/Modules/_randommodule.c#L488-L533>.
    fn getrandbits(&mut self, bits: u32) -> u64 {
        debug_assert!((1..=64).contains(&bits));
        if bits <= 32 {
            return u64::from(self.0.next_u32() >> (32 - bits));
        }
        let low = u64::from(self.0.next_u32());
        let high = u64::from(self.0.next_u32() >> (64 - bits));
        (high << 32) | low
    }

    /// `_randbelow(n)`: a uniform integer in `0..n`, by rejection over
    /// `n.bit_length()` bits.
    ///
    /// Port of <https://github.com/python/cpython/blob/v3.12.13/Lib/random.py#L242-L250>.
    pub(crate) fn randbelow(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randbelow needs a nonempty range");
        let bits = n.bit_width();
        loop {
            let value = self.getrandbits(bits);
            if value < n {
                return value;
            }
        }
    }

    /// `_randbelow(n)` as an index: `choice(seq)` picks `seq[index_below(len(seq))]`.
    ///
    /// Port of <https://github.com/python/cpython/blob/v3.12.13/Lib/random.py#L341-L348>.
    #[expect(clippy::cast_possible_truncation, reason = "The index is below n, a usize")]
    pub(crate) fn index_below(&mut self, n: usize) -> usize {
        self.randbelow(n as u64) as usize
    }

    /// `shuffle(items)`: Fisher–Yates from the last item down.
    ///
    /// Port of <https://github.com/python/cpython/blob/v3.12.13/Lib/random.py#L350-L357>.
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.index_below(i + 1);
            items.swap(i, j);
        }
    }

    /// `sample(population, k)`: a pool for small populations, otherwise
    /// rejection against the indices already drawn.
    ///
    /// Port of <https://github.com/python/cpython/blob/v3.12.13/Lib/random.py#L359-L452>.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "CPython's float threshold: a small positive exponent of an exact count"
    )]
    pub(crate) fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let n = population.len();
        assert!(k <= n, "sample larger than population");
        let mut setsize = 21_u128;
        if k > 5 {
            // CPython: 4 ** ceil(log(k * 3, 4)).
            let exponent = (libm::log(k as f64 * 3.0) / libm::log(4.0)).ceil() as u32;
            setsize += 4_u128.saturating_pow(exponent);
        }
        if n as u128 <= setsize {
            let mut pool = population.to_vec();
            return (0..k)
                .map(|i| {
                    let j = self.index_below(n - i);
                    let value = pool[j].clone();
                    pool[j] = pool[n - i - 1].clone();
                    value
                })
                .collect();
        }
        let mut selected = std::collections::HashSet::with_capacity(k);
        (0..k)
            .map(|_| {
                loop {
                    let j = self.index_below(n);
                    if selected.insert(j) {
                        break population[j].clone();
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::PythonRandom;
    use crate::preprocess::integers as numbers;

    #[test]
    fn draws_match_cpython() {
        let fixture = crate::preprocess::fixture("python_random.json");
        let streams = fixture["streams"].as_array().unwrap();
        assert_eq!(streams.len(), 8);
        for case in streams {
            let seed: i128 = case["seed"].as_str().unwrap().parse().unwrap();
            let mut rng = PythonRandom::new(seed);
            let below = [
                1,
                2,
                3,
                7,
                8,
                1000,
                1 << 31,
                (1 << 32) + 1,
                (1 << 40) + 3,
                1 << 63,
            ]
            .map(|n| rng.randbelow(n));
            assert_eq!(below.to_vec(), numbers(&case["randbelow"]), "seed {seed}");
            let choices = [1, 2, 5, 100].map(|n| rng.index_below(n) as u64);
            assert_eq!(choices.to_vec(), numbers(&case["choice"]), "seed {seed}");
            for (n, expected) in [0, 1, 2, 10, 100]
                .into_iter()
                .zip(case["shuffle"].as_array().unwrap())
            {
                let mut values: Vec<u64> = (0..n).collect();
                rng.shuffle(&mut values);
                assert_eq!(values, numbers(expected), "seed {seed}, shuffle of {n}");
            }
            let samples = case["sample"].as_array().unwrap();
            for ((n, k), expected) in [(1, 1), (6, 6), (50, 50), (1000, 3), (5000, 40)]
                .into_iter()
                .zip(samples)
            {
                let population: Vec<u64> = (0..n).collect();
                assert_eq!(
                    rng.sample(&population, k),
                    numbers(expected),
                    "seed {seed}, sample {k} of {n}"
                );
            }
        }
    }
}
