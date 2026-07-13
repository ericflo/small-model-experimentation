# Qwen Program-Only Executable ABI

**Status:** finished

This standalone experiment tests whether a local 4B model can compile
deterministic office-style tasks into executable programs whose interpreter
result, not an emitted answer token, determines correctness.

The experiment compares final-answer supervision, trace-plus-final supervision,
and two program-only executable ABIs. Large adapter checkpoints are stored under:

`/workspace/large_artifacts/qwen_program_only_executable_abi`

