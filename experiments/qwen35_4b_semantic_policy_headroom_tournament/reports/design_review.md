# Adversarial design review

## Verdict

Proceed after immutable design lock. The experiment measures headroom before
training and cannot consume benchmark seeds.

## Objections and resolutions

1. **Outcome-based substrate selection can overfit.** Qualification selects
   only semantic axes, not repository instances. Both blocks must independently
   pass frozen rules; any later training and transfer use new skins/seeds in a
   new experiment.
2. **Ambiguous contracts may create impossible tasks.** Explicit controls
   validate harness/edit competence. The lower .15 bound rejects axes that are
   effectively unlearnable/underspecified for this parent.
3. **One lucky representation could carry an axis.** Two of three shapes must
   be individually in band on each block, and axis macro must also be in band.
4. **Task IDs could fake freshness.** Content hashes exclude ID/split; each
   36-task block is unique and the intersection is empty.
5. **Malformed behavior may be the only defect by assertion, not reality.**
   Partial code is executable-tested on ordinary acceptance, insufficiency,
   unknown resources, copied state, atomicity, and input nonmutation. Only the
   axis exception differs from oracle.
6. **Rejected-patch and failed-test states answer different questions.** The
   target curriculum would supervise post-failure revision, so eligibility is
   frozen to failed-test. Rejected-patch rates remain fully reported as proposal
   diagnostics.
7. **The previous explicit tasks were saturated.** That is intentional as a
   control condition here. Inferred cells remove only the explicit exception
   sentence while visible tests remain public and inspectable.
8. **Repeated parent evaluation is just more sampling.** No pass-if-either or
   candidate selection occurs. A/B are independent measurement blocks used to
   establish distributional headroom, not to solve individual tasks.
9. **Interface failure could masquerade as semantic headroom.** Invalid and
   answer-cap rates have absolute ceilings; explicit controls must be ≥.85.
10. **Menagerie temptation after a good axis.** Config and result schema set
    authorization false; the runner contains no benchmark path. Qualification
    licenses only a new procedural experiment.

## Residual limitations

- Visible tests disclose expected exception types, so this measures use of
  verifier evidence rather than inference from prose alone.
- Three axes and representations do not span all validation semantics.
- Three tasks per family make shape rates coarse (.0/.33/.67/1.0); replication
  and axis aggregation reduce but do not eliminate uncertainty.
