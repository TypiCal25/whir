# Two-witness opening

Open two existing, ordered `N`-element commitments with one proof.
Both commitments use the same source configuration and transcript.
Stage challenges can occur between them.

```rust
source.validate_pair()?;
let claim = source.prove_pair(&mut prover, [left, right], forms, values)?;
let claim = source.verify_pair(&mut verifier, [left, right], forms, values)?;
claim.verify(forms)?;
```

These methods belong to the existing `ProtocolConfig`.
Forms describe `[left | right]` and have size `2N`.
Public data or earlier transcript messages must fix each form.
The API checks the source and derives the first joint round before changing the transcript.
It binds the source configuration and claim values before sampling shared coefficients.
The result point contains the selector first, followed by the original coordinates.
The caller must finish with `FinalClaim::verify`.

For `W = [w0 | w1]` and `L = [L0 | L1]`, the extra sumcheck round proves

```text
q(s) = <(1-s)w0 + s*w1, (1-s)L0 + s*L1>
q(0) + q(1) = <w0,L0> + <w1,L1>.
```

The original commitment trees and encoded domain stay unchanged.
Their shared query rows form one virtual source with twice the interleaving.
The selector fold then enters the existing code-switch and remaining rounds.
Parameter derivation checks the new first-round security bounds.

The API supports Standard and ZeroKnowledge modes with identity and base-to-extension embeddings.
ZeroKnowledge folds both source masks and adds one mask for the selector sumcheck round.
Raw witness lengths can differ, but their padded sizes must match.
Basecase-only source configurations are unsupported.

```sh
cargo test --locked protocols::params::joint::tests
cargo test --locked --test minimal_pair_rows --test minimal_pair_sumcheck --test pair_protocol
```

Tests include `20 + 146 = 166`, the selector polynomial, and genuine extension coefficients.
The opening tests run in both modes.
They reject changed roots, claims, forms, dimensions, and truncated proofs.
They also check claim binding and transcript continuation.
