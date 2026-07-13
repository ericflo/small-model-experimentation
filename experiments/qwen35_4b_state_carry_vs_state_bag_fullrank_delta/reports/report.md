# Report

The canonical data and live feasibility gates pass; no scientific pilot result
exists yet. All 11 splits and 27,744 rows match the parent task exactly under
both frozen canonical hashes and direct artifact comparison, with no structural
duplicates or benchmark reads.

On the current 48GB RTX 6000 Ada, G0 discovered the preregistered 62 targets and
892,272,640 FP32 delta parameters. Carry and Bag had identical 892,840,988 total
trainable-parameter/value receipts; every delta tensor received nonzero
gradients in both arms; and Adam allocated 124 finite, shape-matched FP32 moment
tensors. Exact base/K=1 and Carry/Bag parity remained `0.0` before and after the
real optimizer step. K=12 was finite with 682 delta calls per arm. Peak
allocation was 24.49 GiB, peak reservation was 24.93 GiB, and reserved headroom
was 22.57 GiB. The 3.571 GB checkpoint round trip restored recurrent logits
with error `0.0`.

Verdict: `MODEL_SMOKE_PASS`. This proves that the full-rank capacity control is
mechanically executable on the registered hardware. It says nothing yet about
state formation or behavior; the paired seed-7401 pilot remains mandatory.
