/*
 * Static x86-64 launcher for authenticated reflection-transfer stages.
 *
 * Build the two pinned variants exactly with:
 *
 * /usr/bin/gcc -nostdlib -static -no-pie -s -Wl,--build-id=none \
 *   -DTRAINING=1 -o scripts/training_launcher scripts/runtime_launcher.S
 * /usr/bin/gcc -nostdlib -static -no-pie -s -Wl,--build-id=none \
 *   -DVLLM=1 -o scripts/vllm_launcher scripts/runtime_launcher.S
 */

#if defined(TRAINING) == defined(VLLM)
#error "define exactly one of TRAINING or VLLM"
#endif

    .section .text
    .global _start
    .type _start, @function
_start:
    mov (%rsp), %r12                  /* original argc */
    lea 8(%rsp), %r13                 /* original argv */

    /* Keep the authenticated static image alive as Python's parent. */
    mov $57, %rax                     /* fork */
    syscall
    test %rax, %rax
    js .Lproof_failed
    jz .Lchild

    mov %rax, %r15                    /* child PID */
    sub $16, %rsp                     /* wait status */
.Lwait:
    mov %r15, %rdi
    mov %rsp, %rsi
    xor %rdx, %rdx
    xor %r10, %r10
    mov $61, %rax                     /* wait4 */
    syscall
    cmp $-4, %rax                     /* EINTR */
    je .Lwait
    test %rax, %rax
    js .Lproof_failed
    mov (%rsp), %eax
    mov %eax, %ecx
    and $0x7f, %ecx
    jnz .Lchild_signaled
    shr $8, %eax
    and $0xff, %eax
    mov %rax, %rdi
    mov $60, %rax
    syscall

.Lchild_signaled:
    add $128, %ecx
    mov %rcx, %rdi
    mov $60, %rax
    syscall

.Lchild:
    /* Kill the dynamic child if its authenticating parent disappears. */
    mov $157, %rax                    /* prctl */
    mov $1, %rdi                      /* PR_SET_PDEATHSIG */
    mov $9, %rsi                      /* SIGKILL */
    xor %rdx, %rdx
    xor %r10, %r10
    xor %r8, %r8
    syscall
    test %rax, %rax
    js .Lproof_failed
    mov $110, %rax                    /* getppid */
    syscall
    cmp $1, %rax
    je .Lproof_failed

    /* Carry the exact launcher inode through execve on descriptor 198. */
    mov $257, %rax                    /* openat */
    mov $-100, %rdi                   /* AT_FDCWD */
    lea self_executable(%rip), %rsi
    xor %rdx, %rdx                    /* O_RDONLY */
    xor %r10, %r10
    syscall
    test %rax, %rax
    js .Lproof_failed
    mov %rax, %r14
    cmp $198, %rax
    je .Lproof_ready
    mov %r14, %rdi
    mov $198, %rsi
    xor %rdx, %rdx
    mov $292, %rax                    /* dup3, flags=0 keeps fd across exec */
    syscall
    test %rax, %rax
    js .Lproof_failed
    mov %r14, %rdi
    mov $3, %rax                      /* close original descriptor */
    syscall

.Lproof_ready:
    /* Six fixed arguments, original argv[1:], and a trailing NULL. */
    mov %r12, %rax
    add $6, %rax
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
    lea no_site_flag(%rip), %rax
    mov %rax, 24(%rsi)
    lea dispatcher_path(%rip), %rax
    mov %rax, 32(%rsi)
    lea backend_name(%rip), %rax
    mov %rax, 40(%rsi)

    mov $1, %rcx
.Lcopy_arguments:
    cmp %r12, %rcx
    jge .Larguments_done
    mov (%r13,%rcx,8), %rax
    lea 5(%rcx), %rdx
    mov %rax, (%rsi,%rdx,8)
    inc %rcx
    jmp .Lcopy_arguments

.Larguments_done:
    lea 5(%r12), %rax
    movq $0, (%rsi,%rax,8)

    lea python_path(%rip), %rdi
    lea launcher_environment(%rip), %rdx
    mov $59, %rax                     /* execve */
    syscall

    mov $127, %rdi
    mov $60, %rax                     /* _exit */
    syscall

.Lproof_failed:
    mov $126, %rdi
    mov $60, %rax                     /* _exit */
    syscall
    .size _start, .-_start

    .section .rodata
#ifdef TRAINING
python_path:
    .asciz "/workspace/small-model-experimentation/.venv/bin/python"
backend_name:
    .asciz "training"
env_path:
    .asciz "PATH=/workspace/small-model-experimentation/.venv/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin"
#else
python_path:
    .asciz "/workspace/small-model-experimentation/.venv-vllm/bin/python"
backend_name:
    .asciz "vllm"
env_path:
    .asciz "PATH=/workspace/small-model-experimentation/.venv-vllm/bin:/usr/local/cuda-12.8/bin:/usr/bin:/bin"
#endif
isolated_flag:
    .asciz "-I"
no_bytecode_flag:
    .asciz "-B"
no_site_flag:
    .asciz "-S"
dispatcher_path:
    .asciz "/workspace/sme-reflection-exec/experiments/qwen35_4b_counterfactual_plan_reflection_transfer/scripts/runtime_entry.py"
self_executable:
    .asciz "/proc/self/exe"

env_home:
    .asciz "HOME=/root"
env_lang:
    .asciz "LANG=C.UTF-8"
env_lc_all:
    .asciz "LC_ALL=C.UTF-8"
env_tz:
    .asciz "TZ=Etc/UTC"
env_python_no_user_site:
    .asciz "PYTHONNOUSERSITE=1"
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
env_hf_offline:
    .asciz "HF_HUB_OFFLINE=1"
env_transformers_offline:
    .asciz "TRANSFORMERS_OFFLINE=1"

    .section .data.rel.ro
    .align 8
launcher_environment:
    .quad env_home
    .quad env_path
    .quad env_lang
    .quad env_lc_all
    .quad env_tz
    .quad env_python_no_user_site
    .quad env_git_no_system
    .quad env_git_global
    .quad env_git_prompt
    .quad env_ld_library_path
    .quad env_cuda_order
    .quad env_vllm_multiprocessing
    .quad env_tokenizers_parallelism
    .quad env_hf_offline
    .quad env_transformers_offline
    .quad 0

    .section .note.GNU-stack,"",@progbits
