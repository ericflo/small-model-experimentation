# Does the installable hypothesize-and-verify skill move the structure wall? Experiment Log

## Scaffold

Created as a new experiment scaffold.

## 2026-07-08 design + adversarial review (pre-GPU)

Selected from the post-C47 forest-review workflow (4 lenses, 13 candidates): the collision cell where
C36 (wall is un-proposable structure) and C45 (hypothesize-and-verify is installable serial compute)
make opposite predictions. Assets recon'd: C36 skeleton metric is behavioral (model_structure_correct),
C45 traces committed (train_general.jsonl, adapter regenerable), scaffold arm confirmed novel repo-wide.
Three-lens design review (reports/design_review.md): sound_with_fixes. Must-fixes applied to README:
(1) mimicry-robust primary metric -- lookup-table code passes the legacy C36 metric by construction;
probe-input extension + per-arm mimicry rate; (2) POOLED depth-3 primary contrast with Holm (per-cell
n=30 has power 0.35 at +0.10 -- vacuous as pre-registered); (3) factorized evidence-pruned procedure
(whole-pipeline enumeration covers <0.5% of skeleton space in 1024 tokens -- null by construction);
(4) traces train into the THINK channel (C45's no-think recipe would make a flat SFT arm format noise);
(5) truth-blind trace generator, byte-verified with oracle blinded; (6) c45_zero regen-sanity gate;
(7) window-overlap stratification of depth-3 (C28/C33 already installed structure FROM oracle depth-3
content -- the novel cell is PROCEDURE-only transfer across depth, C21 negative prior); (8) compute-
parity reporting + K' presentation; (9) scaffold text frozen verbatim (configs/scaffold_prompt.txt,
zero op vocabulary per C30/C31 hint discipline); (10) no-think base anchor + pre-registered branch if
base-think itself clears the wall (C44 dose result). Next: build scripts, smoke, run.

## 2026-07-08 full run, gate stop #1: c45 regen gate was miscalibrated, not the rebuild

Pipeline ran clean through trap gate (oracle-skelfill 1.0 all cells), trace-gen, dsl_sft train, and
c45 regen train, then hard-stopped at the regen-sanity gate: a7 induction-via-generation 0.920 vs
required 0.95. The 0.95 came from C44's SHIFT-experiment number (1.00); C45's actual historical
held-out-a7 headline is 0.905 (meta_induction/runs/verdict_general.json). The regenerated adapter
at 0.920 EXCEEDS its source. Corrected the gate to 0.87 (historical minus 2x binomial SE at n=200),
re-judged the stored measurement (no GPU re-burn), relaunched. Lesson for the playbook: calibrate
reproduction gates to the SOURCE experiment's committed artifact number, never to a neighboring
claim's headline from memory.

## 2026-07-08 full run, crash #2: empty_cache raised INSIDE the OOM-recovery path

The OOM_ERRORS batch-halving patch WORKED (caught the scaffold-arm OOM at batch 48), but
`torch.cuda.empty_cache()` on the recovery path itself raised torch.AcceleratorError and killed
eval_arms mid-sweep. Fixes: (1) `_safe_empty_cache()` in gen_lib -- cleanup is retried once after
5s and never allowed to raise; (2) think-eval batch 48 -> 32 (scaffold prompts add ~200 input
tokens x K=12 x budget-1024 KV). base + base_nothink evals survived on disk; pipeline resumes at
scaffold. Compute-doc lesson at landing: wrap the CLEANUP call in OOM recovery, not just the
generate call.

## 2026-07-08 RESULTS: the wall holds -- C36 hardens; the procedure installs in-depth but does not climb

PIPELINE COMPLETE (343.8 min + 2 recovered crashes). Pre-registered pooled d3 Holm contrasts: ALL ns
(scaffold +0.033, c45_zero -0.017, dsl_sft +0.033; base d3 probe-cov 0.05 pooled -- wall replicates
at think@1024, no C44 frame shift). THE RESULT: dsl_sft depth-2 probe-robust cov 0.37->0.70 (list) /
0.33->0.57 (string), parse 1.00 -- the taught factorized hypothesize-and-verify procedure ~doubles
in-depth structure proposal -- but depth-3 flat (0.10/0.07 vs base 0.03/0.07). Procedure banking does
not cross depth (C21's law extends from answers to PROCEDURES). c45_zero: zero substrate transfer,
active interference (list d2 0.37->0.00, parse 0.70). Scaffold: format-confounded null per
pre-registration (parse 0.44-0.67 vs base 0.89). Mimicry ~0.01 everywhere (metric guard held, unused).
Strata (characterization): d3 zero-windows +0.14 (n=21) vs one-window -0.06 (n=32) -- non-monotone,
noise. verdict.json + PNG written. Write-up next: claim ~C48 (wall is not a missing procedure;
serial-strategy installs are depth-local), playbook + compute-doc lessons (safe empty_cache, gate
calibration to source artifacts), adapters out, landing order.

## 2026-07-09 pre-committed budget-2048 contingency: c45_zero confirmed dead; scaffold flagged BUDGET-LIMITED

The forced-close contingency (>50% at d3 for null arms; actual ~0.83-1.00) fired for scaffold and
c45_zero. Results @2048, d3 subset, K=12: c45_zero 0.017 (was 0.033) -- null confirmed, closed.
scaffold 0.150 (was 0.083; base@1024 0.05), parse still 0.50, forced-close STILL 1.00 -- the
scaffold null is budget-limited, not settled, and 2048 is still not enough budget for the taught
loop. Confound: base was never run @2048 (serial-compute dose, C44). Launched the missing post-hoc
control (base@2048, d3 subset) before concluding Result 5. Branches: base@2048 ~0.15 -> pure dose
effect (wall softens with budget; scaffold adds nothing beyond dose); base@2048 ~0.05 -> the
procedure text does real d3 work given budget (softens the headline). Primary pre-registered
verdict (@1024) unaffected either way.

## 2026-07-09 base@2048 control: the scaffold's 2048 jump was DOSE, not procedure

base@2048 d3 = 0.100 (2x its 1024 value 0.05; forced-close 0.84->0.75). Paired matched-budget
scaffold-vs-base @2048: +0.083, one-sided CI-lo +0.000, p=0.058 -- suggestive, post-hoc, n.s.
[CORRECTED 2026-07-09: task_id pairing bug; true paired value +0.050, p 0.043-0.051 -- see the audit entry below]
Verdict unchanged; Result 5 finalized: serial-compute dose moves d3 coverage for everyone
(C44-consistent softening from a very low floor), the taught procedure adds no significant margin
beyond dose, and even 2048 truncates 75-100% of thinking. The d3 budget dose-curve (4096+) is now
the sharpest open edge on the wall. Write-up complete; adapters out; audit + landing next.

## 2026-07-09 numbers audit caught MY contingency-contrast bug; analyze.py now owns the computation

Three-agent adversarial audit (69+68+95 checks): every pre-registered @1024 number re-derives
exactly, but my ad-hoc paired contrast for the b2048 probes was WRONG (+0.083): task_ids collide
across families (30 unique ids for 60 d3 rows) and my dict pairing matched list-scaffold rows to
string-base rows. The main analysis was safe (keys on (family, depth, task_id)). Fix: analyze.py
now computes the contingency block itself with the correct key -- canonical: scaffold-vs-base@2048
diff +0.050, CI-lo +0.000, p 0.043-0.051 across seeds (borderline, post-hoc, not verdict-grade).
Also fixed per audit: rulebook d2 yield ~51% solved/45-46% kept (was "48-60%"); trigger sentence
(two of THREE null arms; d3 forced-close base 0.84, nulls 1.00, dsl_sft 0.32); leakage clause
(51/1,476 trace found-pipelines coincide with eval d2 skeletons, 0 with d3 -- primary clean); viz
chart-3 reference attribution (0.905 is the committed C45 headline, not base); gate_c45.json
threshold field 0.87; manifest full_command includes the base control probe. Lesson (playbook-
grade): never hand-compute a published statistic -- extend the committed analyzer; and never key
task pairing on bare task_id (ids are only unique within family).
