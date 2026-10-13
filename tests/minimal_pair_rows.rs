//! Feasibility tests for existing IRS APIs. These are not a joint proof protocol.

use std::num::NonZeroUsize;

use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        eq_weights,
        fields::{Field64, Field64_3},
        univariate_evaluate,
    },
    buffer::{Buffer, BufferOps},
    hash,
    protocols::{
        code_switch::mixed_fold_chunks,
        irs_commit::{Commitment, Config, IrsMode, IrsParams},
        params::DecodingRegime,
    },
    transcript::{
        codecs::Empty, Codec, DomainSeparator, ProverState, VerifierMessage, VerifierState,
    },
};

fn check_pair<M: Embedding<Source = Field64> + Default>(mode: IrsMode, require_extension: bool)
where
    M::Target: Codec + Field<BasePrimeField = Field64>,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let embedding = M::default();
    let n = 32;
    let interleaving = 4;
    let chunk_len = n / interleaving;
    let config = Config::<M>::new(IrsParams {
        security_target: 128.0,
        decoding_regime: DecodingRegime::Johnson,
        hash_id: hash::BLAKE3,
        num_vectors: 1,
        vector_size: n,
        interleaving_depth: interleaving,
        rate: 0.25,
        mode,
    });
    let ds = DomainSeparator::protocol(&config)
        .session(&"minimal-pair-rows")
        .instance(&Empty);
    let left = Buffer::from(
        (0..n)
            .map(|i| Field64::from((i + 1) as u64))
            .collect::<Vec<_>>(),
    );
    let right = Buffer::from(
        (0..n)
            .map(|i| Field64::from((2 * i + 11) as u64))
            .collect::<Vec<_>>(),
    );

    // The second commitment follows the challenge from the first commitment.
    let mut prover = ProverState::new_std(&ds);
    let left_witness = config.commit(&mut prover, &[&left]);
    let stage_challenge: M::Target = prover.verifier_message();
    let right_witness = config.commit(&mut prover, &[&right]);
    let folding_point: Vec<M::Target> = prover.verifier_message_vec(3);
    if require_extension {
        assert!(folding_point.iter().any(|x| x
            .to_base_prime_field_elements()
            .skip(1)
            .any(|c| c != Field64::ZERO)));
    }
    let weights = eq_weights(&folding_point);
    assert_eq!(weights.len(), 2 * interleaving);

    // Reuse the existing multiple-commitment opening operation.
    let opened = config.open(&mut prover, &[&left_witness, &right_witness]);
    assert_eq!(opened.num_columns(), 2 * interleaving);
    let collapsed = opened.values_buffer(&embedding, &Buffer::from(weights.clone()));

    // Direct reference: concatenate both messages and fold all eight chunks.
    let both_messages = left.concat(&right);
    let folded_message = mixed_fold_chunks(
        &embedding,
        both_messages.to_slice(),
        chunk_len,
        &folding_point,
    );
    let both_masks = left_witness.masks.concat(&right_witness.masks);
    let folded_mask = mixed_fold_chunks(
        &embedding,
        both_masks.to_slice(),
        config.mask_length(),
        &folding_point,
    );

    for ((&point, row), &actual) in opened
        .points
        .iter()
        .zip(opened.rows())
        .zip(collapsed.to_slice())
    {
        let point = embedding.map(point);
        let message_value = univariate_evaluate(&folded_message, point);
        let mask_value = univariate_evaluate(&folded_mask, point);
        let expected = message_value + point.pow([chunk_len as u64]) * mask_value;
        assert_eq!(
            actual, expected,
            "opened rows must encode the folded message and both masks"
        );
        let direct: M::Target = row
            .iter()
            .zip(&weights)
            .map(|(&x, &weight)| embedding.mixed_mul(weight, x))
            .sum();
        assert_eq!(actual, direct);
    }
    let proof = prover.proof();

    // Verify both original roots at the same row challenges.
    let mut verifier = VerifierState::new_std(&ds, &proof);
    let left_commitment = config.receive_commitment(&mut verifier).unwrap();
    assert_eq!(verifier.verifier_message::<M::Target>(), stage_challenge);
    let right_commitment = config.receive_commitment(&mut verifier).unwrap();
    assert_eq!(verifier.verifier_message_vec::<M::Target>(3), folding_point);
    let checked = config
        .verify(&mut verifier, &[&left_commitment, &right_commitment])
        .unwrap();
    assert_eq!(checked, opened);
    verifier.check_eof().unwrap();

    // Swapping roots or replacing either root must reject the same opening.
    for mutation in 0..3 {
        let mut verifier = VerifierState::new_std(&ds, &proof);
        let left_commitment = config.receive_commitment(&mut verifier).unwrap();
        let _: M::Target = verifier.verifier_message();
        let right_commitment = config.receive_commitment(&mut verifier).unwrap();
        let _: Vec<M::Target> = verifier.verifier_message_vec(3);
        let commitments = match mutation {
            0 => [right_commitment, left_commitment],
            1 => [Commitment::default(), right_commitment],
            _ => [left_commitment, Commitment::default()],
        };
        assert!(config
            .verify(&mut verifier, &[&commitments[0], &commitments[1]])
            .is_err());
    }
}

#[test]
fn identity_standard_pair_rows_equal_folded_encoding() {
    check_pair::<Identity<Field64>>(IrsMode::Standard, false);
}

#[test]
fn identity_zk_pair_rows_include_both_source_masks() {
    check_pair::<Identity<Field64>>(
        IrsMode::ZeroKnowledge {
            mask_length: NonZeroUsize::new(48).unwrap(),
        },
        false,
    );
}

#[test]
fn extension_standard_pair_rows_equal_folded_encoding() {
    check_pair::<Basefield<Field64_3>>(IrsMode::Standard, true);
}

#[test]
fn extension_zk_pair_rows_include_both_source_masks() {
    check_pair::<Basefield<Field64_3>>(
        IrsMode::ZeroKnowledge {
            mask_length: NonZeroUsize::new(48).unwrap(),
        },
        true,
    );
}
