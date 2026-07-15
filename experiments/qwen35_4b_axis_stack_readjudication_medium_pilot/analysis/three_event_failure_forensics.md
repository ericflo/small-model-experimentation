# Three-Event Failure Forensics (seeds 88014/88015/88016)

Produced 2026-07-15 from this repository's own graded gate outputs across the
three axis-line experiments (1,296 graded completions; no benchmark content).
This document is the design basis for the queued axis corpus v2.

## u_tracefix — quantified taxonomy (classes: a wrong-step/right-instruction, b right-step/wrong-instruction, c format, d never-finished, e wholly wrong)

| Event | Arm | correct | a | b | c | d | e |
|---|---|---|---|---|---|---|---|
| 88014 | candidate | 4 | 0 | 2 | 0 | 3 | 1 |
| 88014 | parent | 3 | 1 | 2 | 0 | 1 | 3 |
| 88014 | control | 2 | 1 | 1 | 0 | 0 | 6 |
| 88015 | candidate | 2 | 2 | 2 | 0 | 2 | 2 |
| 88015 | parent | 1 | 3 | 1 | 0 | 4 | 1 |
| 88015 | control | 0 | 2 | 0 | 0 | 3 | 5 |
| 88016 | candidate | 1 | 0 | 2 | 0 | 4 | 3 |
| 88016 | parent | 0 | 4 | 2 | 0 | 1 | 3 |
| 88016 | control | 2 | 3 | 2 | 0 | 1 | 2 |
| Pooled (90 rows) | | 15 | 16 | 14 | 0 | 19 | 26 |

Verdict: a SEARCH problem, not format (0 format failures) or primarily repair.
All 16 class-a failures carry the exactly correct instruction with the wrong
step number; the model conflates first-differing-slot with wrong-step. Pooled
accuracy by bug position: early 5/42, mid 8/33, late 2/15. Seventeen of 19
class-d failures are 1024-token enumeration spirals. In 65/75 failures the
model's own first-pass trace exactly reproduces the given produced output —
execution is fine; there is no recovery routine when it is not (pile-top
orientation, line-trace garbling). 16/30 truth repairs change the op TYPE and
several change the write target; the model's candidate space is almost
exclusively same-op parameter tweaks, so the true repair is often never
proposed. Gauges: 0/12 pooled across all arms/events.

Training-side: the taught think has two beats — execute-and-compare, then a
single ASSERTION ("testing single-step corrections, only rewriting step k...")
— the search itself is never demonstrated. At eval the model performs beat 1
faithfully and improvises beat 2. The install is real but constant: candidate
= parent + 1 task in every event (4v3, 2v1, 1v0); the 4→2→1 slide is frozen-set
difficulty drift onto the weak axes (early bugs 4→4→6, pile+gauges 3→5→6,
op-type repairs 4→6→6).

## u_protocol — parent competent; failures are toggle-parity, never arithmetic

Of 21 wrong-parsed answers pooled: 20/20 word-bearing answers carry the exactly
correct final tally; the branch WORD is wrong; zero tally slips in 300 graded
rows. Transitions: a_only→both 8, both→a_only 6 (toggle-parity), a_only→neither
5 (threshold crossing missed), 1 malformed. Mechanism: dropped early toggle
events or flip-treated-as-set-lit; failures concentrate at steps 6-7 with ≥2
toggles. The lesson (full per-surge restatement of both marker states) did not
transfer; the candidate restates only changes — the style that drops flips.

## Installed lessons — what visibly changed

u_explore: candidate adopts the taught frontier idiom near-verbatim and solves
in ~90 think tokens; the parent runs ad-hoc DFS and violates step budgets.
Caveat: 4/17 parent+control explore failures are spacing-only (route correct,
exact-match rejects 'a > b' vs 'a>b') — part of the measured win is separator
normalization.

u_hygiene: the dominant change is DECISIVENESS. Parent failures are 14/15
rambling 1024-token listicle cap-outs that never answer (decoy taken only
1/15); the candidate answers in ~100-token prose with an explicit dismissal
sentence. Residual candidate failures are crisp decoy adoptions, mostly when
the injection is co-located inside the queried record (only 7/40 training rows
had co-location).

## Retention and caps

u_route at ceiling for every arm/event. Candidate caps 13→9→9 (parents
17→20→18); the reduction is mostly u_induct (6→4→1 caps) where correctness is
0/8 for every arm in every event — the candidate learned to terminate and
commit on an unsolvable kind (decisiveness transfer; C38/C39 wall unchanged).
u_execute keeps capping (2 every event) and u_tracefix 3→2→3.

## v2 recommendations

(a) Trace-repair redesign: stage the answer format (localize-only STEP k;
repair-given-step; compose later); single formalism first (line best at 8/24;
quarantine gauges 0/12 and pile); demonstrate the search with 2-3 REJECTED
candidate rewrites including op-TYPE changes and write-target changes; teach
two checkpoint rules ("my trace must end at the given produced output, else my
semantics are wrong — re-read them"; "the first differing slot is not the wrong
step — the bug can be earlier"); bound thinks ~300 tokens with a
test-k-candidates-then-commit skeleton.

(b) Protocol replacement: parent needs nothing here. Measured-headroom
replacements from the retention screen: u_execute-flavored block (parent 2/8
with 2-6 caps every event) or u_repair-flavored block (parent 2→0→0; same
substrate as tracefix, transfer affinity observed 3/8 vs 0/8 at 88016). If a
protocol block stays, rebuild around explicit running flip-count parity
("flips so far 2, even → unlit") plus threshold-crossing edge cases.

(c) Instrument: 21 correct-but-rejected whitespace rows across three events
('a > b' vs 'a>b'; 'x=1; y=2' vs 'x=1;y=2') — parents 9, candidates 4,
controls 8. v2's gate should normalize separator whitespace at grading (or
report the class separately). Hygiene v2 should raise injection/answer
co-location share well above 7/40.

## Surprises

- Protocol tally arithmetic: zero errors in 300 rows — the model is a perfect
  adder and a poor parity-tracker.
- Parent hygiene failure is a register problem (listicle style → cap), not
  gullibility.
- The tracefix decline is difficulty drift, not forgetting: install held at
  exactly +1 task over parent in all three events.
- The cap halving is mostly the candidate declining to burn 1,024 tokens on
  the unsolvable induction kind.
