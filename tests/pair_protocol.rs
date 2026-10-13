//! Small examples for the complete two-witness opening.
use ark_ff::{AdditiveGroup, Field};
use ark_std::rand::{distributions::Standard, prelude::Distribution};
use whir::{
    algebra::{
        embedding::{Basefield, Embedding, Identity},
        fields::{Field256, Field64_3},
        linear_form::{Covector, LinearForm},
    },
    buffer::Buffer,
    hash,
    protocols::{
        params::{
            joint::JointConfig, DecodingRegime, FoldingFactor, Mode, PowBudget, RateSchedule,
            SecuritySpec, TuningSpec,
        },
        zook::{JointOpeningError, ProtocolConfig, ProverClaim},
    },
    transcript::{
        codecs::Empty, Codec, DomainSeparator, Proof, ProverState, VerifierMessage, VerifierState,
    },
};
use zeroize::Zeroize;

fn source<M: Embedding + Default>(mode: Mode) -> ProtocolConfig<M> {
    ProtocolConfig::derive(
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: 128,
            pow_budget: PowBudget::per_slot(10),
            hash_id: hash::BLAKE3,
        },
        TuningSpec {
            vector_size: 256,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Stepping,
        },
    )
    .unwrap()
}

struct Example<M: Embedding> {
    config: JointConfig<M>,
    proof: Proof,
    forms: Vec<Covector<M::Target>>,
    values: Vec<M::Target>,
    result: ProverClaim<M::Target>,
    stage_challenge: M::Target,
}

fn prove_example<M>(extension_coefficient: M::Target) -> Example<M>
where
    M: Embedding + Default,
    M::Target: Default + Zeroize + Codec,
    Standard: Distribution<M::Source> + Distribution<M::Target>,
{
    let config = JointConfig::derive(&source::<M>(Mode::Standard)).unwrap();
    let n = config.source().tuning().vector_size;
    let mut left = vec![M::Source::ZERO; n];
    let mut right = left.clone();
    // Four entries and three entries are independently padded to N.
    left[..4].copy_from_slice(&[1u64, 2, 3, 12].map(M::Source::from));
    right[..3].copy_from_slice(&[5u64, 7, 4].map(M::Source::from));
    let form = |entries: &[(usize, M::Target)]| {
        let mut weights = vec![M::Target::ZERO; 2 * n];
        for &(i, weight) in entries {
            weights[i] = weight;
        }
        Covector::new(weights)
    };
    let e = extension_coefficient;
    let forms = vec![
        form(&[
            (0, 2u64.into()),
            (1, 3u64.into()),
            (2, 4u64.into()),
            (n, 11u64.into()),
            (n + 1, 13u64.into()),
        ]),
        form(&[(1, e + M::Target::ONE), (n, e)]),
        form(&[(2, M::Target::ONE)]),
        form(&[(n + 2, M::Target::ONE)]),
        form(&[(3, M::Target::ONE)]),
    ];
    // 2*1 + 3*2 + 4*3 = 20; 11*5 + 13*7 = 146; total = 166.
    // The extension-valued claim is (e+1)*2 + e*5 = 7e+2.
    // The final three claims show the human-readable relation 3*4 = 12.
    let values = vec![
        166u64.into(),
        e * M::Target::from(7) + M::Target::from(2),
        3u64.into(),
        4u64.into(),
        12u64.into(),
    ];
    assert_eq!(values[2] * values[3], values[4]);
    let refs: Vec<&dyn LinearForm<M::Target>> = forms.iter().map(|f| f as _).collect();
    let ds = DomainSeparator::protocol(&"two-witness-examples").instance(&Empty);
    let mut ps = ProverState::new_std(&ds);
    let left = config.source().commit(&mut ps, Buffer::from(left));
    let stage_challenge = ps.verifier_message();
    let right = config.source().commit(&mut ps, Buffer::from(right));
    let result = config
        .prove_pair(&mut ps, [left, right], &refs, &values)
        .unwrap();
    // A separate protocol can continue after the pair opening.
    ps.prover_message(&M::Target::from(987));
    Example {
        config,
        proof: ps.proof(),
        forms,
        values,
        result,
        stage_challenge,
    }
}

fn verify_example<M>(example: &Example<M>, mutation: u8) -> Result<(), ()>
where
    M: Embedding + Default,
    M::Target: Default + Codec,
    Standard: Distribution<M::Target>,
{
    let ds = DomainSeparator::protocol(&"two-witness-examples").instance(&Empty);
    let mut vs = VerifierState::new_std(&ds, &example.proof);
    let left = example
        .config
        .source()
        .receive_commitment(&mut vs)
        .map_err(|_| ())?;
    let challenge: M::Target = vs.verifier_message();
    assert_eq!(challenge, example.stage_challenge);
    let right = example
        .config
        .source()
        .receive_commitment(&mut vs)
        .map_err(|_| ())?;
    let commitments = match mutation {
        1 => [right, left],
        2 => [left.clone(), left],
        _ => [left, right],
    };
    let refs: Vec<&dyn LinearForm<M::Target>> = example.forms.iter().map(|f| f as _).collect();
    let mut values = example.values.clone();
    if mutation == 3 {
        values[0] += M::Target::ONE;
    }
    let result = example
        .config
        .verify_pair(&mut vs, commitments, &refs, &values)
        .map_err(|_| ())?;
    assert_eq!(
        result.evaluation_point.len(),
        9,
        "selector plus eight original variables"
    );
    assert_eq!(result.evaluation_point, example.result.evaluation_point);
    assert_eq!(result.rlc_coefficients, example.result.rlc_coefficients);
    let mut final_forms = refs;
    if mutation == 4 {
        final_forms.swap(0, 1);
    }
    result.verify(&final_forms).map_err(|_| ())?;
    let sentinel: M::Target = vs.prover_message().map_err(|_| ())?;
    assert_eq!(sentinel, M::Target::from(987));
    vs.check_eof().map_err(|_| ())
}

#[test]
fn hand_calculated_claims_use_both_existing_commitments_at_128_bits() {
    let example = prove_example::<Identity<Field256>>(Field256::from(19));
    verify_example(&example, 0).unwrap();
}

#[test]
fn genuine_extension_coefficients_open_base_field_witnesses_at_128_bits() {
    let e = Field64_3::new(2u64.into(), 3u64.into(), 5u64.into());
    assert_ne!(e, Field64_3::from(2));
    let example = prove_example::<Basefield<Field64_3>>(e);
    assert_ne!(example.values[1], Field64_3::from(16));
    verify_example(&example, 0).unwrap();
}

#[test]
fn changed_roots_claim_values_and_final_forms_are_rejected() {
    let example = prove_example::<Identity<Field256>>(Field256::from(19));
    for mutation in 1..=4 {
        assert!(
            verify_example(&example, mutation).is_err(),
            "mutation {mutation} must fail"
        );
    }
}

#[test]
fn truncated_joint_proofs_are_rejected() {
    let mut example = prove_example::<Identity<Field256>>(Field256::from(19));
    example
        .proof
        .narg_string
        .truncate(example.proof.narg_string.len() / 2);
    assert!(verify_example(&example, 0).is_err());
}

#[test]
fn zero_knowledge_execution_fails_before_transcript_mutation() {
    type M = Identity<Field256>;
    let standard = JointConfig::derive(&source::<M>(Mode::Standard)).unwrap();
    let zk = source::<M>(Mode::ZeroKnowledge);
    // A deserialized object must not bypass the public derivation restriction.
    let mut encoded = serde_json::to_value(&standard).unwrap();
    encoded["source"] = serde_json::to_value(&zk).unwrap();
    let config: JointConfig<M> = serde_json::from_value(encoded).unwrap();
    let ds = DomainSeparator::protocol(&"zero-knowledge-closed").instance(&Empty);
    let mut ps = ProverState::new_std(&ds);
    let left = zk.commit(&mut ps, Buffer::from(vec![Field256::ZERO; 256]));
    let right = zk.commit(&mut ps, Buffer::from(vec![Field256::ZERO; 256]));
    let form = Covector::new(vec![Field256::ZERO; 512]);
    assert_eq!(
        config
            .prove_pair(&mut ps, [left, right], &[&form], &[Field256::ZERO])
            .unwrap_err(),
        JointOpeningError::UnsupportedZeroKnowledge
    );
    ps.prover_message(&Field256::from(987));
    let proof = ps.proof();
    let mut vs = VerifierState::new_std(&ds, &proof);
    let left = zk.receive_commitment(&mut vs).unwrap();
    let right = zk.receive_commitment(&mut vs).unwrap();
    assert_eq!(
        config
            .verify_pair(&mut vs, [left, right], &[&form], &[Field256::ZERO])
            .unwrap_err(),
        JointOpeningError::UnsupportedZeroKnowledge
    );
    let sentinel: Field256 = vs.prover_message().unwrap();
    assert_eq!(sentinel, Field256::from(987));
    vs.check_eof().unwrap();
}

#[test]
fn all_claim_values_precede_the_shared_batching_challenge() {
    type F = Field256;
    let example = prove_example::<Identity<F>>(F::from(19));
    let ds = DomainSeparator::protocol(&"two-witness-examples").instance(&Empty);
    let mut observer = VerifierState::new_std(&ds, &example.proof);
    let _ = example
        .config
        .source()
        .receive_commitment(&mut observer)
        .unwrap();
    let _: F = observer.verifier_message();
    let _ = example
        .config
        .source()
        .receive_commitment(&mut observer)
        .unwrap();
    let _: hash::Hash = observer.prover_message().unwrap();
    let count: whir::transcript::codecs::U64 = observer.prover_message().unwrap();
    assert_eq!(count.0 as usize, example.values.len());
    for expected in &example.values {
        let bound_value: F = observer.prover_message().unwrap();
        assert_eq!(&bound_value, expected);
    }
    let gamma: F = observer.verifier_message();
    assert_eq!(example.result.rlc_coefficients[0], F::ONE);
    assert_eq!(example.result.rlc_coefficients[1], gamma);
    assert_eq!(example.result.rlc_coefficients[2], gamma * gamma);
}

#[test]
fn malformed_claim_dimensions_fail_before_transcript_mutation() {
    type F = Field256;
    type ClaimInput<'a> = (&'a [&'a dyn LinearForm<F>], &'a [F]);
    let config = JointConfig::derive(&source::<Identity<F>>(Mode::Standard)).unwrap();
    let ds = DomainSeparator::protocol(&"pair-shape-checks").instance(&Empty);
    let mut ps = ProverState::new_std(&ds);
    let committed = [
        config
            .source()
            .commit(&mut ps, Buffer::from(vec![F::ONE; 256])),
        config
            .source()
            .commit(&mut ps, Buffer::from(vec![F::ONE; 256])),
    ];
    let form = Covector::new(vec![F::ZERO; 512]);
    let short = Covector::new(vec![F::ZERO; 256]);
    let malformed: [ClaimInput<'_>; 3] =
        [(&[], &[]), (&[&form], &[]), (&[&short], &[F::ZERO])];
    for (forms, values) in malformed {
        assert_eq!(
            config
                .prove_pair(&mut ps, committed.clone(), forms, values)
                .unwrap_err(),
            JointOpeningError::InvalidClaims
        );
    }
    config
        .prove_pair(&mut ps, committed, &[&form], &[F::ZERO])
        .unwrap();
    let proof = ps.proof();
    let mut vs = VerifierState::new_std(&ds, &proof);
    let commitments = [
        config.source().receive_commitment(&mut vs).unwrap(),
        config.source().receive_commitment(&mut vs).unwrap(),
    ];
    for (forms, values) in malformed {
        assert_eq!(
            config
                .verify_pair(&mut vs, commitments.clone(), forms, values)
                .unwrap_err(),
            JointOpeningError::InvalidClaims
        );
    }
    config
        .verify_pair(&mut vs, commitments, &[&form], &[F::ZERO])
        .unwrap()
        .verify(&[&form])
        .unwrap();
    vs.check_eof().unwrap();
}
