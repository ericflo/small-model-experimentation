# Counterfactual Plan Reflection Transfer — Design-Hold Report

## Summary

The experiment remains without model forward passes under a full-implementation
adversarial HOLD. Full CPU construction succeeds and the historical tokenizer receipt
is invalid as a training prerequisite. Exact-SHA Review 12 accepted the venv/CUTLASS,
active-UUID, and initial-mapping closures but returned HOLD on pre-Python code trust,
subprocess dependency closure, and lease-denied read-only mounts. A model-free
lease-only remediation is now implemented, its detached audits have passed, and
exact-SHA Review 13 remains required. No Qwen generation, GPU, training, capability
measurement, or Jacobian event exists.

## Research Program Fit

This is a posttraining experiment motivated by a mechanistic claim: supervise an
answer to a hypothetical reflection question and test whether the untrained action
branch improves. It is not another inference-time materialization prompt and not a
generic correctness probe.

## Method

Fresh exact-depth list, string, and register machines provide seven demonstrations
and three query inputs. Correct and shuffled arms share the same common transcript
and reflection question; only the final reflection answer differs. The reflection
names the three primitives and omits the exact final answer. Proposed deployment
asks for the answer on a different next-turn branch.

## Results

The full configured construction creates 576 unique exact-depth-three tasks: 216
train, 72 calibration, 144 qualification, and 144 confirmation, plus 48 exact-depth
1/2 retention tasks. It has zero cross-split program or behavior
collisions, unique exact plans on the seven visible demonstrations, complete
operation-position coverage, and behaviorally wrong shuffled donors. The pinned
historical tokenizer receipt then establishes exact parity across correct reflection, shuffled
reflection, and auxiliary plan-label arms: 77,020 prompt tokens, 5,164 target tokens,
and 82,184 forward tokens each. All 12 correct/shuffled optimizer groups match. The
current implementation requires that evidence to be reissued with exact tokenizer,
worktree, and script commitments. This is construction/training-parity readiness
evidence only; it is not a model result.

## Controls

The repaired design now includes a rendered-token-matched non-reflective plan-label
arm, a direct action-branch positive control, real retention data, exact target-only
loss masks, within-optimizer-step derangement, a frozen QLoRA recipe, paired
qualification/confirmation gates, and a specified literal-reflection diagnostic.
The implementation binds exact base/tokenizer/runtime bytes across training, merge,
and protected load windows; enforces a detached execution worktree with no ignored
state; binds a hashed external isolated interpreter and exact GPU identity; reconstructs
generation compute from raw token arrays; and makes checkpoint-aware end-to-end
matched-compute sample-more a transitive final-stage gate. Model, GPU, training,
evaluation, and J-space execution remain unauthorized after Review 10 returned HOLD.

## Oracle Versus Deployable Evidence

Procedural targets are oracle labels. A correct-reflection training target is not
deployable evidence. Only answer accuracy on the unreflected held-out action branch,
against frozen sample-more, can become deployable evidence.

## Interpretation

No capability inference is licensed. The current implementation passes the local
pinned-environment model-free tests. It now authenticates content inside protected
load windows, reconstructs prompt and training spend from raw/sealed token evidence,
starts under `-I -B -S` with complete stage-specific environment-byte authentication,
documents the distinct training and vLLM runtimes, and binds the selected physical GPU
UUID. Those implementation claims were then subjected to independent adversarial
review.
Review 10 accepted those Review-9 closures but rejected the remaining auth→import
window, unpinned interpreter/stdlib/native boundary, broken `-S` venv/re-exec semantics,
and PATH-resolved selected-device query.

The implementation submitted to Review 11 attempted to close those paths by retaining an authenticated
inotify/lease/hash window across every artifact-relevant import, pinning the
interpreter and complete stdlib/system/CUDA/site surfaces, explicitly activating the
authenticated CUTLASS path under `-S`, removing adaptive Mamba re-exec, pinning the
absolute device-query binary, and binding its physical UUID row to CUDA's sole active
logical device after initialization. Real detached bootstrap/seal checks pass for
both stage-specific environments. Review 11 nevertheless showed that an allowed
unleased system file can change without an observed event; venv-bin derivation still
resolves to `/usr/bin`; the active CUDA UUID is neither compared nor recorded;
PATH-resolved Git, `uv`, and `nvcc` remain outside the authenticated executable
boundary; and replay does not require a nonempty loaded-native closure. This remains
implementation evidence only and all execution authorization stays unchanged.

The Review-12 candidate adds byte-reproducible static training/vLLM launchers whose
live parent executable and inherited proof descriptor must match the tracked launcher
inode and committed hash. The dispatcher admits only fixed experiment stages and
re-enters the pinned interpreter with `-I -B -S`; direct Python entry fails. Executable
subprocesses use pinned open inodes rather than PATH. Lease denial is accepted only for
an unchanged exact read-only file mount, vLLM retains the un-resolved venv bin path,
active CUDA UUID is compared and recorded, and replay requires the full initial native
mapping set. The 92-test/23-subtest model-free suite and a real 4,915-lease plus
34-read-only-mount guard audit pass. This remains non-authorizing implementation
evidence pending exact-SHA review and detached launcher/bootstrap seals.

The exact-SHA detached seals subsequently passed at
`da80b2b314b44140f305e3b84bf727583486e882`: training protected 33,178 leased files
and vLLM protected 73,330; each admitted only the 34 exact read-only NVIDIA mounts and
authenticated 17 loaded mappings. vLLM also proved CUTLASS discovery without model
import. Both exact-SHA CI workflows passed, and the clean detached worktree was
removed. Review 12 subsequently returned HOLD. It accepted venv/CUTLASS/geometry,
active-UUID, and initial-mapping replay closures, but found that the
dispatcher/runtime/config/stage execute or influence execution before descriptor
authentication and that read-only bind mounts do not exclude pre-existing writable
mappings of their backing inodes.

## Review-12 Remediation Candidate

The replacement static C launchers authenticate and retain a committed manifest before
forking. Mandatory read leases cover the exact snapshot interpreter and dynamic loader,
the standard library, initial native closure, Git plus its dependencies and helpers,
the runtime contract, load guard, lock files, and selected stage. The child executes the
stage directly through inherited descriptors; there is no Python dispatcher. A fixed
worktree code/config guard starts before authenticated Git preflight, and the complete
runtime/site/native guard admits no unleased files. A pre-existing shared writable
mapping is an explicit kernel-level rejection test. Explicit Git, `uv`, and `nvcc`
metadata calls complete before guard seal. The external runtime snapshot contains only
byte copies whose exact tree surfaces are pinned in the repository. This is implementation
evidence, not authorization; Review 13 must independently attack an exact clean commit.

At exact commit `c8ff609ba9c0abb8eaa9be1775ec39e61f2a4f59`, training sealed
33,344 files and vLLM sealed 73,496, each with 46 preflight files, 16 initial native
mappings, and zero unleased files. Before that pass, the audit correctly rejected two
omitted preflight-native mappings and one copied runtime file with unleaseable source
ownership. The manifest probe and snapshot procedure were repaired, all 5,033 files
in the configured external roots accepted a read lease, and that property is now an
exhaustive regression. Both exact-SHA CI workflows and the full 94-test/23-subtest
model-free suite passed.

## Next Experiments

Obtain independent Review 13 of the exact committed SHA and remediate any reproduced
counterexample before changing authorization. Nothing beyond tokenizer-only work is
authorized yet.

## Artifact Manifest

See `artifact_manifest.yaml`.
