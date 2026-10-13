//! Feasibility tests for one extra quadratic selector round.
//! These tests do not implement the joint commitment or mask-authentication protocol.
use ark_ff::{AdditiveGroup, Field};
use whir::{
    algebra::{
        dot,
        embedding::{Basefield, Embedding, Identity},
        fields::{Field64, Field64_3},
        mixed_dot, univariate_evaluate,
    },
    buffer::{Buffer, BufferMath, BufferOps},
    protocols::{
        proof_of_work,
        sumcheck::{Config, SumcheckMaskLen, SumcheckMode},
    },
    transcript::{codecs::Empty, Codec, DomainSeparator, ProverState, VerifierState},
};

#[test]
fn selector_polynomial_is_quadratic_and_starts_from_166() {
    type F = Field64;
    let left = [1u64, 2, 3, 0].map(F::from);
    let right = [5u64, 7, 0, 0].map(F::from);
    let left_form = [2u64, 3, 4, 0].map(F::from);
    let right_form = [11u64, 13, 0, 0].map(F::from);
    let q = |s: F| {
        let witness: Vec<_> = left
            .iter()
            .zip(right)
            .map(|(&a, b)| (F::ONE - s) * a + s * b)
            .collect();
        let form: Vec<_> = left_form
            .iter()
            .zip(right_form)
            .map(|(&a, b)| (F::ONE - s) * a + s * b)
            .collect();
        dot(&witness, &form)
    };
    assert_eq!(q(F::ZERO), F::from(20));
    assert_eq!(q(F::ONE), F::from(146));
    assert_eq!(q(F::ZERO) + q(F::ONE), F::from(166));
    for s in [F::ZERO, F::ONE, F::from(2), F::from(7)] {
        assert_eq!(q(s), F::from(20) + F::from(28) * s + F::from(98) * s * s);
    }
    assert_eq!(q(F::from(2)), F::from(468));
}

fn existing_sumcheck_handles_pair<M>(embedding: M, mode: SumcheckMode, extra_coefficient: M::Target)
where
    M: Embedding<Source = Field64>,
    M::Target: Codec<[u8]>,
{
    type B = Field64;
    let witness: Vec<_> = [1u64, 2, 3, 0, 5, 7, 0, 0]
        .into_iter()
        .map(B::from)
        .collect();
    let mut form: Vec<_> = [2u64, 3, 4, 0, 11, 13, 0, 0]
        .into_iter()
        .map(M::Target::from)
        .collect();
    form[0] += extra_coefficient;
    let claim = mixed_dot(&embedding, &form, &witness);
    // The source first round folds one variable. The joint round folds two.
    let cfg = Config::new(8, proof_of_work::Config::none(), 2, mode.clone());
    assert_eq!(cfg.final_size(), 2);
    let masks: Vec<_> = match mode {
        SumcheckMode::Standard => vec![],
        // Deterministic test masks only. Production commits fresh random masks.
        SumcheckMode::ZeroKnowledge { .. } => [11u64, 17, 23, 31, 37, 41]
            .into_iter()
            .map(M::Target::from)
            .collect(),
    };
    let ds = DomainSeparator::protocol(&cfg).instance(&Empty);
    let mut ps = ProverState::new_std(&ds);
    let mut folded_form = Buffer::from(form.clone());
    let mut sum = claim;
    let (folded_witness, opening) = cfg.prove(
        &mut ps,
        &embedding,
        Buffer::from(witness.clone()),
        &mut folded_form,
        &mut sum,
        &masks,
    );
    assert_eq!(opening.round_challenges.len(), 2);
    let selector = opening.round_challenges[0];
    let next_coordinate = opening.round_challenges[1];
    let selected_witness: Vec<_> = (0..4)
        .map(|i| {
            embedding.map(witness[i]) * (M::Target::ONE - selector)
                + embedding.map(witness[i + 4]) * selector
        })
        .collect();
    let selected_form: Vec<_> = (0..4)
        .map(|i| form[i] * (M::Target::ONE - selector) + form[i + 4] * selector)
        .collect();
    for i in 0..2 {
        assert_eq!(
            folded_witness.to_slice()[i],
            selected_witness[i] * (M::Target::ONE - next_coordinate)
                + selected_witness[i + 2] * next_coordinate
        );
        assert_eq!(
            folded_form.to_slice()[i],
            selected_form[i] * (M::Target::ONE - next_coordinate)
                + selected_form[i + 2] * next_coordinate
        );
    }
    let mask_sum: M::Target = masks
        .chunks_exact(3)
        .zip(&opening.round_challenges)
        .map(|(mask, &r)| univariate_evaluate(mask, r))
        .sum();
    assert_eq!(
        sum,
        mask_sum + opening.mask_rlc * folded_witness.dot(&folded_form)
    );
    let proof = ps.proof();
    let mut vs = VerifierState::new_std(&ds, &proof);
    let mut verifier_sum = claim;
    let verifier_opening = cfg.verify(&mut vs, &mut verifier_sum).unwrap();
    assert_eq!(verifier_sum, sum);
    assert_eq!(verifier_opening.round_challenges, opening.round_challenges);
    assert_eq!(verifier_opening.mask_rlc, opening.mask_rlc);
    vs.check_eof().unwrap();
}

#[test]
fn existing_sumcheck_supports_selector_with_identity() {
    for mode in [
        SumcheckMode::Standard,
        SumcheckMode::ZeroKnowledge {
            mask_length: SumcheckMaskLen::new(3),
        },
    ] {
        existing_sumcheck_handles_pair(Identity::<Field64>::default(), mode, Field64::ZERO);
    }
}

#[test]
fn existing_sumcheck_supports_selector_with_genuine_extension_coefficients() {
    let extension = Field64_3::new(Field64::ZERO, Field64::ONE, Field64::ZERO);
    for mode in [
        SumcheckMode::Standard,
        SumcheckMode::ZeroKnowledge {
            mask_length: SumcheckMaskLen::new(3),
        },
    ] {
        existing_sumcheck_handles_pair(Basefield::<Field64_3>::default(), mode, extension);
    }
}
