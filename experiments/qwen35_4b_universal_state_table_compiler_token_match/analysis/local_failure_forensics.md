# Fresh Local Failure Forensics

## Scope

This audit reads only the experiment-owned procedural local receipt at seed 88,008.
It does not read or import `benchmarks/`, any benchmark item, transcript, family
source, result detail, or hidden label. The aggregate seed remained sealed.

## Registered outcome

| Arm | Correct | Parsed | Cap contacts | Mean tokens | Execute | Induct | Probe | Target total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `close_xi_parent` | 19/26 | 23/26 | 3 | 438.1 | 2/2 | 0/2 | 2/2 | 4/6 |
| `replay_after_close` | 16/26 | 21/26 | 5 | 508.1 | 1/2 | 0/2 | 1/2 | 2/6 |
| `state_table_after_close` | 16/26 | 22/26 | 5 | 522.5 | 0/2 | 0/2 | 1/2 | 1/6 |

The candidate failed accuracy ≥0.65, parse ≥0.90, cap contacts ≤2, execute ≥0.50,
and induction ≥0.50. It passed probe ≥0.50 and the route-abstention guard. It did not
strictly beat either control overall or on the six target cases. Promotion was empty.

Receipts:

- complete local: `027c0f631e85fb9b07d6e8e43d51665f32b02fa0a39e928aae564a7d521f2869`
- parent gate: `8af2f1712e291b1f7f90859de6007d58b79847987735557efeeb015e9b62c964`
- replay gate: `d767cf2b34c46beb978e4ad626f32e8bfaa2671306d85198207c1440732cdc0c`
- candidate gate: `76dcd96a1296b64b9aa77a7f5a91e2323ed93b5ea2f6f8cf8a9f3bc69d3b0957`
- promotion: `429770fd7e5929b8154e4df540b8325927fb822d421acccf57ccc17bfead70f5`

## Paired flips

Against the parent, the candidate gained one trace and one optimize case, but lost
one probe, the other trace, both execute cases, and one order case: two wins versus
five losses. Against replay, it gained trace, abstain, and optimize, but lost repair,
execute, and order: three wins versus three losses. The tied total therefore hides a
worse targeted subtotal and a redistribution of non-target wins.

## Failure anatomy

1. **Correct state, wrong serialization.** On one state case the candidate computed
   the exact register values that both controls got wrong, but inserted spaces after
   semicolons. The frozen exact-answer contract correctly scored it false.
2. **Correct execution, no commit.** On one execute case the visible thought derived
   the exact target sequence, then continued until the cap without emitting it. The
   parser captured a literal `<answer>` placeholder instead.
3. **Declaration/operator confusion.** On the other execute case the candidate
   applied the two requested operations correctly, then treated the supplied cycle
   order—a reference declaration—as a third transformation and returned a wrong,
   length-changing sequence.
4. **Unbounded induction.** Both induction cases repeated examples or extended search
   until 1,024 tokens with no parseable answer. The variable-depth tables did not
   install a bounded stop rule.
5. **Score-table arithmetic failure.** On the regressed probe case, H1 and H2 produced
   identical outputs and H3 produced one different output, but the candidate counted
   three distinct outputs instead of two and selected the wrong probe.
6. **Repair-state propagation failure.** Both repair answers were wrong. The candidate
   could identify suspicious transitions but did not consistently propagate the
   corrected first state through the remaining operations.

## Supported boundary

The intervention was fully truth-audited, but its traces were idealized and
off-policy. It improved some isolated computations without making the procedure
available at the model's actual deployment prefixes. Another hand-authored trace
format is therefore not warranted. This result does not show that state tables are
never useful, nor does it measure broad retention; it rejects this 80-row package at
the frozen interface and dose.

## Constraint for the successor

A result-separated successor should generate fresh procedural tasks and collect the
authenticated parent's own rollouts before training. It should classify the first
observable failure prefix and attach an executable-oracle corrective continuation:
bounded close-and-commit after a correct state, explicit declaration-versus-operation
repair, finite induction-loop termination, independently recomputed probe counts, and
exact output serialization. Gate instances from seed 88,008 must remain held out.
The successor still needs a same-parent exact-forward-token replay control, fresh
construction/training/local/conditional seeds, the unchanged absolute gate, strict
wins over both controls overall and on target cases, and no benchmark access before
promotion.
