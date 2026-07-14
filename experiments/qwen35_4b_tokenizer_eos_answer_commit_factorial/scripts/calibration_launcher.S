/*
 * Static x86-64 calibration launcher.
 *
 * This deliberately uses only Linux syscalls: no ELF interpreter, libc,
 * shell, or inherited environment is involved before execve starts the
 * pinned Python interpreter.  Rebuild exactly with:
 *
 * /usr/bin/gcc -nostdlib -static -no-pie -s -Wl,--build-id=none \
 *   -o scripts/calibration_launcher scripts/calibration_launcher.S
 */

    .section .text
    .global _start
    .type _start, @function
_start:
    mov (%rsp), %r12                  /* original argc */
    lea 8(%rsp), %r13                 /* original argv */

    /* Four fixed arguments, original argv[1:], and a trailing NULL. */
    mov %r12, %rax
    add $4, %rax
    shl $3, %rax
    sub %rax, %rsp
    and $-16, %rsp
    mov %rsp, %rsi                    /* replacement argv */

    lea python_path(%rip), %rax
    mov %rax, 0(%rsi)
    lea isolated_flag(%rip), %rax
    mov %rax, 8(%rsi)
    lea no_bytecode_flag(%rip), %rax
    mov %rax, 16(%rsi)
    lea script_path(%rip), %rax
    mov %rax, 24(%rsi)

    mov $1, %rcx
.Lcopy_arguments:
    cmp %r12, %rcx
    jge .Larguments_done
    mov (%r13,%rcx,8), %rax
    lea 3(%rcx), %rdx
    mov %rax, (%rsi,%rdx,8)
    inc %rcx
    jmp .Lcopy_arguments

.Larguments_done:
    lea 3(%r12), %rax
    movq $0, (%rsi,%rax,8)

    lea python_path(%rip), %rdi
    lea launcher_environment(%rip), %rdx
    mov $59, %rax                     /* execve */
    syscall

    mov $127, %rdi
    mov $60, %rax                     /* _exit */
    syscall
    .size _start, .-_start

    .section .rodata
python_path:
    .asciz "/workspace/small-model-experimentation/.venv-vllm/bin/python"
isolated_flag:
    .asciz "-I"
no_bytecode_flag:
    .asciz "-B"
script_path:
    .asciz "/workspace/small-model-experimentation/experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/run_calibration.py"

env_launcher:
    .asciz "QWEN_CALIBRATION_STATIC_LAUNCHER=1"
env_home:
    .asciz "HOME=/root"
env_path:
    .asciz "PATH=/workspace/small-model-experimentation/.venv-vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin"
env_lang:
    .asciz "LANG=C.UTF-8"
env_lc_all:
    .asciz "LC_ALL=C.UTF-8"
env_tz:
    .asciz "TZ=Etc/UTC"
env_python_no_user_site:
    .asciz "PYTHONNOUSERSITE=1"
env_gh_config:
    .asciz "GH_CONFIG_DIR=/root/.config/gh"
env_gh_host:
    .asciz "GH_HOST=github.com"
env_no_color:
    .asciz "NO_COLOR=1"
env_git_no_system:
    .asciz "GIT_CONFIG_NOSYSTEM=1"
env_git_global:
    .asciz "GIT_CONFIG_GLOBAL=/dev/null"
env_git_prompt:
    .asciz "GIT_TERMINAL_PROMPT=0"
env_ld_library_path:
    .asciz "LD_LIBRARY_PATH=/usr/local/cuda/lib64"
env_cuda_order:
    .asciz "CUDA_DEVICE_ORDER=PCI_BUS_ID"
env_vllm_multiprocessing:
    .asciz "VLLM_ENABLE_V1_MULTIPROCESSING=0"
env_tokenizers_parallelism:
    .asciz "TOKENIZERS_PARALLELISM=false"

    .section .data.rel.ro
    .align 8
launcher_environment:
    .quad env_launcher
    .quad env_home
    .quad env_path
    .quad env_lang
    .quad env_lc_all
    .quad env_tz
    .quad env_python_no_user_site
    .quad env_gh_config
    .quad env_gh_host
    .quad env_no_color
    .quad env_git_no_system
    .quad env_git_global
    .quad env_git_prompt
    .quad env_ld_library_path
    .quad env_cuda_order
    .quad env_vllm_multiprocessing
    .quad env_tokenizers_parallelism
    .quad 0

    .section .note.GNU-stack,"",@progbits
