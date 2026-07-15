#define _GNU_SOURCE
#define OPENSSL_SUPPRESS_DEPRECATED

#include <errno.h>
#include <fcntl.h>
#include <openssl/sha.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(TRAINING) == defined(VLLM)
#error "define exactly one of TRAINING or VLLM"
#endif

#ifndef MANIFEST_SHA256
#error "MANIFEST_SHA256 must pin the committed pre-Python manifest"
#endif

enum {
    GIT_FD = 192,
    LOADER_FD = 193,
    MANIFEST_FD = 194,
    RUNTIME_CONTRACT_FD = 195,
    LOAD_GUARD_FD = 196,
    STAGE_FD = 197,
    LAUNCHER_FD = 198,
    INTERPRETER_FD = 199,
};

static const char *manifest_path =
    "/workspace/sme-reflection-exec/experiments/"
    "qwen35_4b_counterfactual_plan_reflection_transfer/scripts/runtime_manifest.tsv";
static const char *library_path =
    "/workspace/sme-reflection-runtime/runtime-libs:/usr/local/cuda/lib64";
static volatile sig_atomic_t lease_break_seen = 0;

#ifdef TRAINING
static const char *backend = "training";
static const char *backend_path =
    "/workspace/small-model-experimentation/.venv/bin:"
    "/usr/local/cuda-12.8/bin:/workspace/sme-reflection-runtime/tools";
#else
static const char *backend = "vllm";
static const char *backend_path =
    "/workspace/small-model-experimentation/.venv-vllm/bin:"
    "/usr/local/cuda-12.8/bin:/workspace/sme-reflection-runtime/tools";
#endif

static void lease_break_handler(int signal_number) {
    (void)signal_number;
    lease_break_seen = 1;
}

static void die(const char *message) {
    if (message != NULL) {
        (void)write(STDERR_FILENO, message, strlen(message));
        (void)write(STDERR_FILENO, "\n", 1);
    }
    _exit(126);
}

static bool valid_hex_digest(const char *value) {
    if (value == NULL || strlen(value) != 64) {
        return false;
    }
    for (size_t index = 0; index < 64; ++index) {
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f'))) {
            return false;
        }
    }
    return true;
}

static void digest_fd(int descriptor, char output[65]) {
    SHA256_CTX context;
    unsigned char digest[SHA256_DIGEST_LENGTH];
    unsigned char buffer[1024 * 1024];
    if (lseek(descriptor, 0, SEEK_SET) < 0 || SHA256_Init(&context) != 1) {
        die("runtime launcher could not initialize file authentication");
    }
    for (;;) {
        ssize_t count = read(descriptor, buffer, sizeof(buffer));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            die("runtime launcher could not read an authenticated file");
        }
        if (count == 0) {
            break;
        }
        if (SHA256_Update(&context, buffer, (size_t)count) != 1) {
            die("runtime launcher could not update file authentication");
        }
    }
    if (SHA256_Final(digest, &context) != 1) {
        die("runtime launcher could not finish file authentication");
    }
    for (size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
        static const char hexadecimal[] = "0123456789abcdef";
        output[index * 2] = hexadecimal[digest[index] >> 4];
        output[index * 2 + 1] = hexadecimal[digest[index] & 15];
    }
    output[64] = '\0';
    if (lseek(descriptor, 0, SEEK_SET) < 0) {
        die("runtime launcher could not rewind an authenticated file");
    }
}

static int open_lease_hash(const char *path, const char *expected_digest) {
    struct stat descriptor_stat;
    struct stat path_stat;
    char observed_digest[65];
    if (path == NULL || path[0] != '/' || !valid_hex_digest(expected_digest)) {
        die("runtime launcher manifest contains an invalid file row");
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0 || fstat(descriptor, &descriptor_stat) < 0 ||
        lstat(path, &path_stat) < 0 || !S_ISREG(descriptor_stat.st_mode) ||
        !S_ISREG(path_stat.st_mode) || descriptor_stat.st_dev != path_stat.st_dev ||
        descriptor_stat.st_ino != path_stat.st_ino) {
        die("runtime launcher could not bind an exact regular manifest inode");
    }
    if (fcntl(descriptor, F_SETLEASE, F_RDLCK) < 0) {
        die("runtime launcher mandatory pre-Python read lease was denied");
    }
    digest_fd(descriptor, observed_digest);
    if (strcmp(observed_digest, expected_digest) != 0 ||
        fstat(descriptor, &descriptor_stat) < 0 || lstat(path, &path_stat) < 0 ||
        descriptor_stat.st_dev != path_stat.st_dev ||
        descriptor_stat.st_ino != path_stat.st_ino) {
        die("runtime launcher manifest inode differs from its pinned digest");
    }
    return descriptor;
}

static void inherit_as(int source, int target) {
    if (source != target && dup2(source, target) < 0) {
        die("runtime launcher could not install a proof descriptor");
    }
    int flags = fcntl(target, F_GETFD);
    if (flags < 0 || fcntl(target, F_SETFD, flags & ~FD_CLOEXEC) < 0) {
        die("runtime launcher could not make a proof descriptor inheritable");
    }
}

static bool valid_gpu_selector(const char *argument) {
    const char *prefix = "--cuda-visible-devices=GPU-";
    if (strncmp(argument, prefix, strlen(prefix)) != 0) {
        return false;
    }
    const char *cursor = argument + strlen(prefix);
    if (*cursor == '\0') {
        return false;
    }
    for (; *cursor != '\0'; ++cursor) {
        if (!((*cursor >= 'A' && *cursor <= 'Z') ||
              (*cursor >= 'a' && *cursor <= 'z') ||
              (*cursor >= '0' && *cursor <= '9') || *cursor == '-')) {
            return false;
        }
    }
    return true;
}

static int role_target(const char *role) {
    if (strcmp(role, "git") == 0) return GIT_FD;
    if (strcmp(role, "loader") == 0) return LOADER_FD;
    if (strcmp(role, "runtime_contract") == 0) return RUNTIME_CONTRACT_FD;
    if (strcmp(role, "load_window_guard") == 0) return LOAD_GUARD_FD;
    if (strcmp(role, "interpreter") == 0) return INTERPRETER_FD;
    return -1;
}

int main(int argc, char **argv) {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = lease_break_handler;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGIO, &action, NULL) < 0) {
        die("runtime launcher could not install its lease-break handler");
    }
    int argument_index = 1;
    const char *gpu_selector = NULL;
    if (argument_index < argc &&
        strncmp(argv[argument_index], "--cuda-visible-devices=", 23) == 0) {
        if (!valid_gpu_selector(argv[argument_index])) {
            die("runtime launcher received an invalid physical GPU UUID");
        }
        gpu_selector = argv[argument_index++] + 23;
    }
    if (argument_index >= argc || strchr(argv[argument_index], '\t') != NULL ||
        strchr(argv[argument_index], '\n') != NULL) {
        die("runtime launcher requires one fixed stage name");
    }
    const char *stage_name = argv[argument_index++];

    int manifest = open_lease_hash(manifest_path, MANIFEST_SHA256);
    FILE *stream = fdopen(dup(manifest), "r");
    if (stream == NULL) {
        die("runtime launcher could not parse its authenticated manifest");
    }
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length = getline(&line, &capacity, stream);
    if (length != 9 || strcmp(line, "schema\t1\n") != 0) {
        die("runtime launcher manifest schema changed");
    }
    bool role_seen[5] = {false, false, false, false, false};
    int stage_descriptor = -1;
    while ((length = getline(&line, &capacity, stream)) >= 0) {
        if (length < 2 || line[length - 1] != '\n') {
            die("runtime launcher manifest has an unterminated row");
        }
        line[length - 1] = '\0';
        char *save = NULL;
        char *kind = strtok_r(line, "\t", &save);
        if (kind != NULL && strcmp(kind, "file") == 0) {
            char *role = strtok_r(NULL, "\t", &save);
            char *path = strtok_r(NULL, "\t", &save);
            char *digest = strtok_r(NULL, "\t", &save);
            if (role == NULL || path == NULL || digest == NULL ||
                strtok_r(NULL, "\t", &save) != NULL) {
                die("runtime launcher manifest file row changed");
            }
            int descriptor = open_lease_hash(path, digest);
            int target = role_target(role);
            if (target >= 0) {
                int slot = target == GIT_FD ? 0 : target == LOADER_FD ? 1 :
                    target == RUNTIME_CONTRACT_FD ? 2 :
                    target == LOAD_GUARD_FD ? 3 : 4;
                if (role_seen[slot]) {
                    die("runtime launcher manifest duplicated a proof role");
                }
                role_seen[slot] = true;
                inherit_as(descriptor, target);
            }
        } else if (kind != NULL && strcmp(kind, "stage") == 0) {
            char *row_backend = strtok_r(NULL, "\t", &save);
            char *name = strtok_r(NULL, "\t", &save);
            char *path = strtok_r(NULL, "\t", &save);
            char *digest = strtok_r(NULL, "\t", &save);
            if (row_backend == NULL || name == NULL || path == NULL || digest == NULL ||
                strtok_r(NULL, "\t", &save) != NULL) {
                die("runtime launcher manifest stage row changed");
            }
            if (strcmp(row_backend, backend) == 0 && strcmp(name, stage_name) == 0) {
                if (stage_descriptor >= 0) {
                    die("runtime launcher manifest duplicated the selected stage");
                }
                stage_descriptor = open_lease_hash(path, digest);
                inherit_as(stage_descriptor, STAGE_FD);
            }
        } else {
            die("runtime launcher manifest row type changed");
        }
    }
    free(line);
    fclose(stream);
    for (size_t index = 0; index < 5; ++index) {
        if (!role_seen[index]) {
            die("runtime launcher manifest omitted a required proof role");
        }
    }
    if (stage_descriptor < 0 || lease_break_seen) {
        die("runtime launcher stage is absent or pre-Python bytes were mutable");
    }
    inherit_as(manifest, MANIFEST_FD);
    int self_descriptor = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    if (self_descriptor < 0 || fcntl(self_descriptor, F_SETLEASE, F_RDLCK) < 0) {
        die("runtime launcher could not lease its live executable inode");
    }
    inherit_as(self_descriptor, LAUNCHER_FD);

    pid_t child = fork();
    if (child < 0) {
        die("runtime launcher could not fork its guarded stage");
    }
    if (child == 0) {
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) < 0 || getppid() == 1) {
            die("runtime launcher could not bind child lifetime to its parent");
        }
        char path_environment[1024];
        char ld_environment[1024];
        char backend_environment[128];
        char stage_environment[512];
        char cuda_environment[256];
        snprintf(path_environment, sizeof(path_environment), "PATH=%s", backend_path);
        snprintf(ld_environment, sizeof(ld_environment), "LD_LIBRARY_PATH=%s", library_path);
        snprintf(backend_environment, sizeof(backend_environment),
                 "SME_RUNTIME_BACKEND=%s", backend);
        snprintf(stage_environment, sizeof(stage_environment),
                 "SME_RUNTIME_STAGE=%s", stage_name);
        char *environment[24] = {
            "HOME=/root", path_environment, "LANG=C.UTF-8", "LC_ALL=C.UTF-8",
            "TZ=Etc/UTC", "PYTHONNOUSERSITE=1", "GIT_CONFIG_NOSYSTEM=1",
            "GIT_CONFIG_GLOBAL=/dev/null", "GIT_TERMINAL_PROMPT=0",
            "GIT_OPTIONAL_LOCKS=0",
            "GIT_EXEC_PATH=/workspace/sme-reflection-runtime/lib/git-core",
            ld_environment,
            "LOCPATH=/workspace/sme-reflection-runtime/lib/locale",
            "GCONV_PATH=/workspace/sme-reflection-runtime/runtime-libs/gconv",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
            "TOKENIZERS_PARALLELISM=false", "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1", backend_environment, stage_environment,
            NULL, NULL, NULL,
        };
        if (gpu_selector != NULL) {
            snprintf(cuda_environment, sizeof(cuda_environment),
                     "CUDA_VISIBLE_DEVICES=%s", gpu_selector);
            environment[21] = cuda_environment;
        }
        size_t forwarded = (size_t)(argc - argument_index);
        char **child_argv = calloc(forwarded + 11, sizeof(char *));
        if (child_argv == NULL) {
            die("runtime launcher could not allocate its fixed argument vector");
        }
        child_argv[0] = "/proc/self/fd/193";
        child_argv[1] = "--library-path";
        child_argv[2] = (char *)library_path;
        child_argv[3] = "--argv0";
        child_argv[4] = "/workspace/sme-reflection-runtime/bin/python3.12";
        child_argv[5] = "/proc/self/fd/199";
        child_argv[6] = "-I";
        child_argv[7] = "-B";
        child_argv[8] = "-S";
        child_argv[9] = "/proc/self/fd/197";
        for (size_t index = 0; index < forwarded; ++index) {
            child_argv[10 + index] = argv[argument_index + (int)index];
        }
        execve("/proc/self/fd/193", child_argv, environment);
        die("runtime launcher could not execute its preauthenticated interpreter");
    }

    int status = 0;
    for (;;) {
        pid_t waited = waitpid(child, &status, 0);
        if (lease_break_seen) {
            (void)kill(child, SIGKILL);
            (void)waitpid(child, &status, 0);
            die("runtime launcher observed a pre-Python lease-break attempt");
        }
        if (waited == child) break;
        if (waited < 0 && errno == EINTR) continue;
        die("runtime launcher could not wait for its guarded stage");
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 126;
}
