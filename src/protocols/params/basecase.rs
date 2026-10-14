//! Standard basecase selection and Zook final-opening bounds.

use ark_ff::Field;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::embedding::Identity,
    bits::Bits,
    protocols::{
        basecase::{BasecaseMode, Config as BasecaseConfig},
        irs_commit::Config as IrsConfig,
        params::{
            bounds::{error_bits, log2_sum},
            error::{grind_to_at, DeriveError, Pow, ZookSecurityError},
            irs_commit::{solve as solve_irs, validate_single_polynomial},
            protocol_config::{BasecasePlan, WitnessMaskConfig},
            regime::constrained_agreement_bits,
            spec::{Mode, RoundContext, SecuritySpec},
            sumcheck::analytic_error_bits,
        },
        proof_of_work::Config as PowConfig,
        sumcheck::{Config as SumcheckConfig, SumcheckMode},
    },
};

pub fn solve<F: Field>(
    spec: &SecuritySpec,
    vector_size: usize,
    log_inv_rate: u32,
    incoming_deep: usize,
) -> Result<BasecasePlan<F>, DeriveError> {
    let commit = solve_irs(
        spec,
        &RoundContext {
            vector_size,
            log_inv_rate,
            folding_factor: 0,
        },
    )?;
    solve_with_commit(spec, commit, incoming_deep)
}

pub fn solve_with_commit<F: Field>(
    spec: &SecuritySpec,
    commit: IrsConfig<Identity<F>>,
    incoming_deep: usize,
) -> Result<BasecasePlan<F>, DeriveError> {
    if matches!(spec.mode, Mode::ZeroKnowledge) {
        validate_single_polynomial(&commit)?;
        let witness_mask = WitnessMaskConfig::for_code(&commit, incoming_deep)?;
        return Ok(BasecasePlan::ZeroKnowledge {
            commit,
            witness_mask,
        });
    }
    let analytic = analytic_error_bits(&commit, false)?;
    let pow = grind_to_at(spec, analytic, Pow::BasecaseSumcheck)?;
    let sumcheck = SumcheckConfig::new(
        commit.vector_size(),
        pow,
        commit.vector_size().trailing_zeros() as usize,
        SumcheckMode::Standard,
    );
    Ok(BasecasePlan::Standard {
        config: BasecaseConfig::new(commit, sumcheck, BasecaseMode::Standard, PowConfig::none()),
        sumcheck_analytic: analytic,
    })
}

/// Final-opening security conditional on joint-mask binding.
///
/// `epsilon_combination = max(A_witness, A_masks)`;
/// `epsilon_queries <= (1 - theta_witness)^s_witness + (1 - theta_masks)^s_masks`.
/// Each `A` bounds degree-one constrained correlated agreement for its code;
/// each `theta` is that code's proximity radius and each `s` its query count.
/// The implementation conservatively uses the degree-two agreement bound.
/// Taking the maximum of error probabilities gives the minimum of security bits.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, final-opening batching and query bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZookFinalOpeningSoundness {
    /// One common random combination for the witness and joint mask commitments.
    pub combination: Bits,
    /// Sum of the two independent code-query error bounds.
    pub queries: Bits,
}

impl ZookFinalOpeningSoundness {
    /// Evaluate the final witness code and joint mask code at their actual query counts.
    pub fn for_codes<F: Field>(
        witness: &IrsConfig<Identity<F>>,
        masks: &IrsConfig<Identity<F>>,
    ) -> Result<Self, ZookSecurityError> {
        validate_single_polynomial(witness)?;
        validate_single_polynomial(masks)?;
        Ok(Self {
            combination: constrained_agreement_bits(witness)?
                .min(constrained_agreement_bits(masks)?),
            queries: error_bits(log2_sum(-witness.rbr_queries(), -masks.rbr_queries())),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{algebra::fields::Field64_2, protocols::params::test_utils::johnson_code};

    #[test]
    fn final_queries_lose_one_bit_when_errors_are_equal() {
        let code = johnson_code::<Identity<Field64_2>>(128, 1, 0.25);
        let result = ZookFinalOpeningSoundness::for_codes(&code, &code).unwrap();
        assert!((f64::from(result.queries) - (code.rbr_queries() - 1.0)).abs() < 1e-10);
    }

    #[test]
    fn final_opening_uses_weaker_agreement_bound() {
        let witness = johnson_code::<Identity<Field64_2>>(64, 1, 0.25);
        let masks = johnson_code::<Identity<Field64_2>>(1024, 1, 0.25);
        let result = ZookFinalOpeningSoundness::for_codes(&witness, &masks).unwrap();
        assert_eq!(
            result.combination,
            constrained_agreement_bits(&masks).unwrap()
        );
    }
}
