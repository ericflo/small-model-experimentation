# Pre-registration: Context Composition

Logged 2026-07-02, before any data. The third capability-installation mechanism. C12/C13 measured TOOLS
(externalization works); C14 measured WEIGHTS (SFT is format-local; a fully repaired simulator does not
propagate to inverse tasks). This experiment measures CONTEXT: can explicit in-context orchestration or
few-shot demonstration compose capabilities that weight-training cannot?

## Setup

Reused assets: the keystone experiment's fresh verified ladder tasks (120, d {2,3,4} × k {0,2}) and its
SIM adapter (regenerated from the committed recipe: QLoRA on 1,500 state-chain traces; simulator verified
0.80–0.84 through depth 5). Thinking budget 1024 for orchestrated conditions (simulating two pipelines
needs tokens); parse instructions identical across conditions.

## Conditions

2AFC (true pipeline vs one-op decoy; same items/decoys as keystone via fixed seed):
  A1 base + plain think        [keystone measured: 0.47]
  A2 base + ORCHESTRATED       ("simulate pipeline A on example 1 step by step, writing 'Step i: [...]'
                                 lines; then pipeline B; compare with the shown output; answer A or B")
  A3 base + ICL                (2 worked examples demonstrating the full simulate-both-compare procedure)
  A4 SIM  + plain think        [keystone measured: 0.10 — format capture]
  A5 SIM  + ORCHESTRATED       (the star cell: explicit invocation of the trained chain format inside 2AFC)
Identification (bare, pass@4):
  B1 base + orchestrated generate-and-test  ("propose a candidate pipeline, simulate it step by step in
      'Step i: [...]' lines on example 1, check against the shown output, revise if wrong; after checking,
      output the final code")   [base bare: 0.08]
  B2 SIM + orchestrated generate-and-test   [SIM bare: 0.09]

## Predictions (locked)

- **P-C1 (star)**: IF context can invoke weight-installed capability, A5 ≥ 0.70 (the SIM simulator is
  0.8+ at these depths; structured single-pipeline simulation is within capacity). A5 ≤ A2 ⇒ the trained
  primitive is inaccessible even under explicit invocation → format-locality deepens to SEALED MODULES.
- **P-C2**: A2 > A1 at d2 (orchestration helps base where its own simulation is decent: 0.88 at d2) but
  the gain shrinks by d4 (base simulation 0.30). Predicted A2 overall ≈ 0.55–0.65.
- **P-C3**: ICL alone (A3) moves little: < +0.10 over A1 — demonstrations install formats, not
  compositions (C14's format-locality predicts this).
- **P-C4**: orchestrated identification (B1/B2) lifts bare by <2× — the binding constraint there is
  proposal quality (~2× over chance, C13), which orchestration does not fix; any lift comes from the
  test-and-revise loop catching wrong proposals.
- **P-C5 (interaction, the mechanism read)**: A5 − A2 > 0.15 ⇒ the weight-installed simulator adds
  accessible capability under explicit invocation (C14 refines to "format-local under IMPLICIT
  invocation"); |A5 − A2| ≤ 0.05 with both high ⇒ orchestration alone suffices (in-context simulation was
  never the bottleneck — the bottleneck was the *procedure*); A5 < A2 ⇒ format capture actively blocks
  composed use (sealed modules, strongest form).

## Decision mapping

- CONTEXT-COMPOSES (A5 high, P-C5 positive): elicitation recipe = install primitives by SFT, compose by
  prompt — a two-stage strategy weights-only and prompts-only both miss. Update C14.
- PROCEDURE-WAS-MISSING (A2 ≈ A5, both high): the model always had enough simulation in-context; C13's
  P12 read shifts from capacity to procedure — thinking fails without an explicit algorithm.
- SEALED-MODULES (A5 ≤ A2, both low-moderate): format-locality is invocation-independent; trained
  capability is usable only in its trained I/O shape. Strongest form of C14.
