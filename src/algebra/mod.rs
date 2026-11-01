pub mod embedding;
pub mod fields;
pub mod linear_form;
mod multilinear;
mod multilinear_point;
pub mod ntt;
pub mod sumcheck;

use ark_ff::{AdditiveGroup, Field};
pub use multilinear::{eval_eq, mixed_multilinear_extend, multilinear_extend};
pub use multilinear_point::MultilinearPoint;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use self::embedding::Embedding;
#[cfg(feature = "parallel")]
use crate::utils::workload_size;

pub fn geometric_sequence<F: Field>(base: F, length: usize) -> Vec<F> {
    let mut result = Vec::with_capacity(length);
    let mut current = F::ONE;
    for _ in 0..length {
        result.push(current);
        current *= base;
    }
    result
}

pub fn dot<F: Field>(a: &[F], b: &[F]) -> F {
    mixed_dot(&embedding::Identity::new(), a, b)
}

pub fn tensor_product<F: Field>(a: &[F], b: &[F]) -> Vec<F> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for &x in a {
        for &y in b {
            result.push(x * y);
        }
    }
    result
}

/// Lift a vector to an embedding.
pub fn lift<M: Embedding>(embedding: &M, source: &[M::Source]) -> Vec<M::Target> {
    #[cfg(not(feature = "parallel"))]
    let result = source.iter().map(|c| embedding.map(*c)).collect();

    #[cfg(feature = "parallel")]
    let result = source.par_iter().map(|c| embedding.map(*c)).collect();

    result
}

pub fn scalar_mul_add<F: Field>(accumulator: &mut [F], weight: F, vector: &[F]) {
    mixed_scalar_mul_add(
        &embedding::Identity::<F>::new(),
        accumulator,
        weight,
        vector,
    );
}

/// Mixed scalar-mul add
///
/// `accumulator[i] += weight * vector[i]`
pub fn mixed_scalar_mul_add<M: Embedding>(
    embedding: &M,
    accumulator: &mut [M::Target],
    weight: M::Target,
    vector: &[M::Source],
) {
    assert_eq!(accumulator.len(), vector.len());
    for (accumulator, value) in accumulator.iter_mut().zip(vector) {
        *accumulator += embedding.mixed_mul(weight, *value);
    }
}

pub fn univariate_evaluate<F: Field>(coefficients: &[F], point: F) -> F {
    mixed_univariate_evaluate(&embedding::Identity::new(), coefficients, point)
}

/// Mixed field univariate Horner evaluation.
pub fn mixed_univariate_evaluate<M: Embedding>(
    embedding: &M,
    coefficients: &[M::Source],
    point: M::Target,
) -> M::Target {
    #[cfg(feature = "parallel")]
    if coefficients.len() > workload_size::<M::Source>() {
        let half = coefficients.len() / 2;
        let (low, high) = coefficients.split_at(half);
        let (low, high) = rayon::join(
            || mixed_univariate_evaluate(embedding, low, point),
            || mixed_univariate_evaluate(embedding, high, point),
        );
        return low + high * point.pow([half as u64]);
    }

    let Some(mut acc) = coefficients.last().map(|c| embedding.map(*c)) else {
        return M::Target::ZERO;
    };
    for &c in coefficients.iter().rev().skip(1) {
        acc *= point;
        acc = embedding.mixed_add(acc, c);
    }
    acc
}

pub fn mixed_dot<F: Field, G: Field>(
    embedding: &impl Embedding<Source = F, Target = G>,
    a: &[G],
    b: &[F],
) -> G {
    assert_eq!(a.len(), b.len());

    #[cfg(feature = "parallel")]
    if a.len() > workload_size::<G>() {
        return a
            .par_iter()
            .zip(b)
            .map(|(a, b)| embedding.mixed_mul(*a, *b))
            .sum();
    }

    a.iter()
        .zip(b)
        .map(|(a, b)| embedding.mixed_mul(*a, *b))
        .sum()
}

/// Shortest half [`geometric_accumulate`] splits an accumulator into; below
/// this, raising every point to the half's offset costs a large share of the
/// half's own work.
#[cfg(feature = "parallel")]
const GEOMETRIC_MIN_SPLIT: usize = 64;

/// Compute `accumulator[i] += sum_j scalars[j] * points[j]^i`
///
/// With the `parallel` feature and more than one thread, splits by total
/// work, one multiplication per entry and point, so a short accumulator with
/// many points is still spread over threads. The high half starts from
/// `scalars[j] * points[j]^half`, so every entry receives the same values as
/// in one sequential pass. On a single thread it never splits, since each
/// split only adds the cost of those powers.
pub fn geometric_accumulate<F: Field>(accumulator: &mut [F], mut scalars: Vec<F>, points: &[F]) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1
        && accumulator.len() >= 2 * GEOMETRIC_MIN_SPLIT
        && accumulator.len().saturating_mul(points.len()) > workload_size::<F>()
    {
        let half = accumulator.len() / 2;
        let (low, high) = accumulator.split_at_mut(half);
        let scalars_high = scalars
            .iter()
            .zip(points)
            .map(|(s, x)| *s * x.pow([half as u64]))
            .collect();
        rayon::join(
            || geometric_accumulate(low, scalars, points),
            || geometric_accumulate(high, scalars_high, points),
        );
        return;
    }

    for entry in accumulator {
        for (scalar, point) in scalars.iter_mut().zip(points) {
            *entry += *scalar;
            *scalar *= *point; // TODO: Skip on last
        }
    }
}

#[cfg(test)]
mod tests {
    use super::geometric_accumulate;
    use crate::algebra::fields::Field64;

    /// SplitMix64 over Field64, so the inputs are reproducible.
    fn random_elements(state: &mut u64, count: usize) -> Vec<Field64> {
        (0..count)
            .map(|_| {
                *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = *state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                Field64::from(z ^ (z >> 31))
            })
            .collect()
    }

    /// However it splits the accumulator, geometric_accumulate must give every
    /// entry exactly the values of one sequential pass.
    #[test]
    fn geometric_accumulate_matches_sequential_pass() {
        let mut state = 0x6e0_u64;
        for length in [0, 1, 2, 63, 64, 127, 128, 129, 1000, 5000, 9000] {
            for num_points in [0, 1, 3, 190] {
                let initial = random_elements(&mut state, length);
                let scalars = random_elements(&mut state, num_points);
                let points = random_elements(&mut state, num_points);

                let mut expected = initial.clone();
                let mut running = scalars.clone();
                for entry in &mut expected {
                    for (scalar, point) in running.iter_mut().zip(&points) {
                        *entry += *scalar;
                        *scalar *= *point;
                    }
                }

                #[cfg(feature = "parallel")]
                {
                    let mut single_thread = initial.clone();
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .expect("a one-thread pool must build")
                        .install(|| {
                            geometric_accumulate(&mut single_thread, scalars.clone(), &points);
                        });
                    assert_eq!(
                        single_thread, expected,
                        "single-threaded geometric_accumulate differs from a sequential pass \
                         for {length} entries and {num_points} points"
                    );
                }

                let mut actual = initial;
                geometric_accumulate(&mut actual, scalars, &points);
                assert_eq!(
                    actual, expected,
                    "geometric_accumulate differs from a sequential pass for {length} entries \
                     and {num_points} points"
                );
            }
        }
    }
}
