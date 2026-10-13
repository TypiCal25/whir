//! Parameters for one opening of two equal, separately committed witnesses.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::algebra::embedding::Embedding;

use super::{
    branch::SolveMode,
    code_switch,
    error::{DeriveError, Pow},
    protocol_config::{ProtocolConfig, RoundConfig, RoundMode},
    spec::{FoldingFactor, Mode, RateSchedule, RoundContext},
    sumcheck,
};

/// The original commitment configuration and its derived joint opening plan.
///
/// Both inputs use `source`. The first joint round folds the selector before
/// the original variables. Its output and all later rounds remain unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JointConfig<M: Embedding> {
    source: ProtocolConfig<M>,
    joint: ProtocolConfig<M>,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum JointConfigError {
    #[error(
        "joint zero-knowledge opening is unavailable until the code-switch privacy leak is fixed"
    )]
    ZeroKnowledgeUnavailable,
    #[error("joint opening requires a first code-switch round")]
    NoFirstRound,
    #[error("joint opening requires a canonical source configuration")]
    InvalidSource,
    #[error("joint opening source dimensions are invalid or overflow")]
    InvalidDimensions,
    #[error("joint opening configuration differs from its derived plan")]
    InvalidJoint,
    #[error(transparent)]
    Derive(#[from] DeriveError),
}

impl<M: Embedding + Default> JointConfig<M> {
    /// Preserve both commitment encodings and rebuild only the first round.
    ///
    /// Source configurations must equal `ProtocolConfig::derive` for their
    /// specification. This restriction also rejects malformed deserialized plans.
    /// ZeroKnowledge mode fails until the underlying code-switch privacy leak is fixed.
    pub fn derive(source: &ProtocolConfig<M>) -> Result<Self, JointConfigError> {
        if source.security().mode == Mode::ZeroKnowledge {
            return Err(JointConfigError::ZeroKnowledgeUnavailable);
        }
        // Bound deserialized inputs before the existing solver computes rates and sizes.
        let tuning = source.tuning();
        if tuning.vector_size.checked_mul(2).is_none()
            || tuning.starting_log_inv_rate >= usize::BITS
            || matches!(
                tuning.rate_schedule,
                RateSchedule::Capped {
                    max_log_inv_rate: 0
                }
            )
        {
            return Err(JointConfigError::InvalidSource);
        }
        let canonical =
            ProtocolConfig::<M>::derive(source.security().clone(), source.tuning().clone())?;
        if source != &canonical {
            return Err(JointConfigError::InvalidSource);
        }
        let first = source.first_round().ok_or(JointConfigError::NoFirstRound)?;
        let virtual_source = first
            .code_switch()
            .source()
            .joint_pair_source()
            .ok_or(JointConfigError::InvalidDimensions)?;
        let context = RoundContext {
            vector_size: virtual_source.vector_size(),
            folding_factor: virtual_source.interleaving_depth().ilog2(),
            log_inv_rate: source.tuning().starting_log_inv_rate,
        };
        let security = source.security();
        let t_ood = first.code_switch().out_domain_samples();
        let sumcheck = sumcheck::solve(
            security,
            &context,
            &virtual_source,
            SolveMode::Standard,
            Pow::RoundSumcheck {
                index: first.round_index(),
            },
        )?;
        let code_switch = code_switch::solve(
            security,
            virtual_source,
            first.code_switch().target().clone(),
            t_ood,
            SolveMode::Standard,
            first.round_index(),
        )?;
        let mut tuning = source.tuning().clone();
        tuning.vector_size = context.vector_size;
        tuning.folding_factor = FoldingFactor::ConstantFromSecondRound {
            initial: context.folding_factor as usize,
            rest: tuning.folding_factor.at_round(1),
        };
        let joint = ProtocolConfig::new(
            security.clone(),
            tuning,
            Some(RoundConfig::new(
                first.round_index(),
                sumcheck,
                code_switch,
                RoundMode::Standard,
            )),
            source.tail_rounds().to_vec(),
            source.basecase_plan().clone(),
        );
        joint.validate()?;
        Ok(Self {
            source: source.clone(),
            joint,
        })
    }

    /// Check source integrity, complete joint derivation, security, and PoW limits.
    pub fn validate(&self) -> Result<(), JointConfigError> {
        let expected = Self::derive(&self.source)?;
        if self.joint != expected.joint {
            return Err(JointConfigError::InvalidJoint);
        }
        Ok(())
    }

    pub const fn source(&self) -> &ProtocolConfig<M> {
        &self.source
    }

    pub const fn joint(&self) -> &ProtocolConfig<M> {
        &self.joint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{
            embedding::{Basefield, Identity},
            fields::{Field256, Field64_3},
        },
        bits::Bits,
        protocols::params::{SecuritySpec, TuningSpec},
    };

    fn source<M: Embedding + Default>(mode: Mode, vector_size: usize) -> ProtocolConfig<M> {
        ProtocolConfig::derive(
            SecuritySpec::new(128).with_mode(mode),
            TuningSpec {
                vector_size,
                starting_log_inv_rate: 2,
                folding_factor: FoldingFactor::Constant(2),
                rate_schedule: RateSchedule::Stepping,
            },
        )
        .unwrap()
    }

    fn check_pair<M: Embedding + Default>() {
        let source = source::<M>(Mode::Standard, 256);
        let config = JointConfig::derive(&source).unwrap();
        config.validate().unwrap();
        let original = source.first_round().unwrap();
        let first = config.joint().first_round().unwrap();
        let original_irs = original.code_switch().source();
        let joint_irs = first.code_switch().source();

        // Two 256-entry witnesses become one 512-entry virtual witness.
        assert_eq!(joint_irs.vector_size(), 512);
        assert_eq!(original_irs.interleaving_depth(), 4);
        assert_eq!(joint_irs.interleaving_depth(), 8);
        assert_eq!(first.sumcheck().num_rounds(), 3);
        assert_eq!(
            first.sumcheck().num_rounds(),
            original.sumcheck().num_rounds() + 1
        );

        // The encoded polynomials and query positions remain the same.
        assert_eq!(joint_irs.message_length(), original_irs.message_length());
        assert_eq!(joint_irs.codeword_length(), original_irs.codeword_length());
        assert_eq!(
            joint_irs.in_domain_samples(),
            original_irs.in_domain_samples()
        );
        assert_eq!(joint_irs.mask_length(), original_irs.mask_length());
        assert_eq!(joint_irs.rate(), original_irs.rate());
        assert_eq!(joint_irs.num_cols(), 2 * original_irs.num_cols());

        // The selector disappears before the unchanged target and remaining rounds.
        assert_eq!(
            first.code_switch().target(),
            original.code_switch().target()
        );
        assert_eq!(config.joint().tail_rounds(), source.tail_rounds());
        assert_eq!(config.joint().basecase_plan(), source.basecase_plan());
        assert_eq!(config.joint().security(), source.security());
        assert_eq!(config.joint().security().target_security_bits, 128);
        config.joint().validate_security_target_met().unwrap();
        config.joint().validate_pow_budget().unwrap();
    }

    #[test]
    fn bn254_standard_keeps_commitments_and_adds_one_selector() {
        check_pair::<Identity<Field256>>();
    }

    #[test]
    fn extension_standard_keeps_base_field_commitment_encoding() {
        check_pair::<Basefield<Field64_3>>();
    }

    #[test]
    fn basecase_only_source_fails_preflight() {
        let source = source::<Identity<Field256>>(Mode::Standard, 8);
        assert_eq!(
            JointConfig::derive(&source),
            Err(JointConfigError::NoFirstRound)
        );
    }

    #[test]
    fn changed_source_security_data_fails_preflight() {
        let mut source = source::<Basefield<Field64_3>>(Mode::Standard, 256);
        source.corrupt_round_sumcheck_analytic_for_test(0, Bits::new(1.0));
        assert_eq!(
            JointConfig::derive(&source),
            Err(JointConfigError::InvalidSource)
        );
    }

    #[test]
    fn deserialized_joint_plan_must_match_its_complete_derivation() {
        let source = source::<Basefield<Field64_3>>(Mode::Standard, 256);
        let config = JointConfig::derive(&source).unwrap();
        let encoded = serde_json::to_vec(&config).unwrap();
        let mut decoded: JointConfig<Basefield<Field64_3>> =
            serde_json::from_slice(&encoded).unwrap();
        decoded.validate().unwrap();
        decoded
            .joint
            .corrupt_round_sumcheck_analytic_for_test(0, Bits::new(1.0));
        assert_eq!(decoded.validate(), Err(JointConfigError::InvalidJoint));
    }

    #[test]
    fn zero_knowledge_fails_preflight_until_privacy_is_fixed() {
        let source = source::<Basefield<Field64_3>>(Mode::ZeroKnowledge, 256);
        assert_eq!(
            JointConfig::derive(&source),
            Err(JointConfigError::ZeroKnowledgeUnavailable)
        );
    }

    #[test]
    fn deserialized_query_count_and_invalid_source_dimensions_are_rejected() {
        let source = source::<Identity<Field256>>(Mode::Standard, 256);
        let config = JointConfig::derive(&source).unwrap();

        // Reducing the joint query count must not preserve the declared security target.
        let mut encoded = serde_json::to_value(&config).unwrap();
        let query_count = &mut encoded["joint"]["first_round"]["code_switch"]["config"]["source"]
            ["in_domain_samples"];
        assert!(query_count.as_u64().unwrap() > 1);
        *query_count = serde_json::json!(1);
        let decoded: JointConfig<Identity<Field256>> = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.validate(), Err(JointConfigError::InvalidJoint));

        // Reject malformed source dimensions before security formulas divide by them.
        let mut encoded = serde_json::to_value(&config).unwrap();
        encoded["source"]["first_round"]["code_switch"]["config"]["source"]["interleaving_depth"] =
            serde_json::json!(0);
        let decoded: JointConfig<Identity<Field256>> = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.validate(), Err(JointConfigError::InvalidSource));

        let mut encoded = serde_json::to_value(&config).unwrap();
        encoded["source"]["tuning"]["starting_log_inv_rate"] = serde_json::json!(u32::MAX);
        let decoded: JointConfig<Identity<Field256>> = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.validate(), Err(JointConfigError::InvalidSource));
    }
}
