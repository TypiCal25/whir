//! One selector fold joins two existing, equally sized commitments.
//!
//! The caller commits the first witness, samples its stage challenges, and then
//! commits the second witness through `JointConfig::source`. Both commitments
//! must use the same transcript as this opening. Forms describe `[left | right]`.
//! Their meaning must be fixed by the public statement or earlier transcript.
//! This entry point binds the configuration and every value before sampling the
//! shared claim coefficients. The caller must finish with `FinalClaim::verify`.

use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
use thiserror::Error;
use zeroize::Zeroize;

use super::{commit::CommittedState, Commitment, CommittedWitness, FinalClaim, ProverClaim};
use crate::{
    algebra::{embedding::Embedding, linear_form::LinearForm},
    buffer::BufferOps,
    hash::Hash,
    protocols::params::{joint::JointConfig, spec::Mode},
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierState,
    },
};

/// Failure before or during the paired opening.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum JointOpeningError {
    #[error("invalid joint configuration")]
    InvalidConfiguration,
    #[error("joint zero knowledge is unavailable: the existing mask reconciliation reveals an unclaimed witness evaluation")]
    UnsupportedZeroKnowledge,
    #[error("joint opening requires nonempty matching claims with size 2N")]
    InvalidClaims,
    #[error("joint opening requires two N-entry round commitments")]
    InvalidWitness,
    #[error("invalid joint opening proof")]
    InvalidProof,
}

impl<M: Embedding + Default> JointConfig<M> {
    fn check_opening(
        &self,
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> Result<(), JointOpeningError> {
        // Fail closed before writing or reading any transcript data.
        if self.source().security().mode == Mode::ZeroKnowledge {
            return Err(JointOpeningError::UnsupportedZeroKnowledge);
        }
        self.validate()
            .map_err(|_| JointOpeningError::InvalidConfiguration)?;
        if forms.is_empty()
            || forms.len() != values.len()
            || forms
                .iter()
                .any(|form| form.size() != self.joint().tuning().vector_size)
        {
            return Err(JointOpeningError::InvalidClaims);
        }
        Ok(())
    }

    fn opening_binding(&self) -> Hash {
        let mut digest = blake3::Hasher::new();
        digest.update(b"whir-joint-two-witness-v1");
        digest.update(&serde_json::to_vec(self).expect("joint configuration serializes"));
        Hash(*digest.finalize().as_bytes())
    }

    /// Prove combined claims for two existing source commitments.
    ///
    /// The first result coordinate is the selector. Remaining coordinates belong
    /// to each original witness. This method leaves later transcript messages alone.
    /// ZeroKnowledge mode fails before any transcript mutation.
    pub fn prove_pair<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        committed: [CommittedWitness<M>; 2],
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> Result<ProverClaim<M::Target>, JointOpeningError>
    where
        M::Target: Default + Zeroize + Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.check_opening(forms, values)?;
        let n = self.source().tuning().vector_size;
        if committed.iter().any(|w| {
            !matches!(&w.state,
            CommittedState::Round { message, .. } if message.len() == n)
        }) {
            return Err(JointOpeningError::InvalidWitness);
        }
        // Values precede gamma even when the caller has not sent them already.
        ps.prover_message(&self.opening_binding());
        ps.prover_message(&U64(values.len() as u64));
        for value in values {
            ps.prover_message(value);
        }
        let [left, right] = committed;
        let source = self
            .source()
            .first_round()
            .expect("validated first round")
            .code_switch()
            .source();
        Ok(self
            .joint()
            .prove_with_second(ps, left, Some((source, right)), forms, values))
    }

    /// Verify the joint transcript and return the final form check.
    ///
    /// The caller must call `FinalClaim::verify(forms)` before accepting the proof.
    pub fn verify_pair<H>(
        &self,
        vs: &mut VerifierState<H>,
        commitments: [Commitment; 2],
        forms: &[&dyn LinearForm<M::Target>],
        values: &[M::Target],
    ) -> Result<FinalClaim<M::Target>, JointOpeningError>
    where
        M::Target: Default + Codec<[H::U]>,
        Standard: Distribution<M::Target>,
        H: DuplexSpongeInterface,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.check_opening(forms, values)?;
        let binding: Hash = vs
            .prover_message()
            .map_err(|_| JointOpeningError::InvalidProof)?;
        let count: U64 = vs
            .prover_message()
            .map_err(|_| JointOpeningError::InvalidProof)?;
        if binding != self.opening_binding() || count.0 != values.len() as u64 {
            return Err(JointOpeningError::InvalidProof);
        }
        for expected in values {
            let actual: M::Target = vs
                .prover_message()
                .map_err(|_| JointOpeningError::InvalidProof)?;
            if actual != *expected {
                return Err(JointOpeningError::InvalidProof);
            }
        }
        let [left, right] = commitments;
        let source = self
            .source()
            .first_round()
            .expect("validated first round")
            .code_switch()
            .source();
        self.joint()
            .verify_with_second(vs, left, Some((source, &right)), forms, values)
            .map_err(|_| JointOpeningError::InvalidProof)
    }
}
