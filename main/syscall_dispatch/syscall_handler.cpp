#include <asm-generic/errno-base.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <optional>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <asm/unistd_64.h>
#include <stdio.h>
#include "syscall_dispatch.h"
#include "orbis/freebsd_9.0_syscalls.hpp"
#include <sys/ucontext.h>
#include "ps4_sysctl.h"
#include <cassert>
#include <fcntl.h>
#include <filesystem>
namespace fs = std::filesystem;
#include "chroot/chroot.h"

#define DIRECT_SYSCALL_MAP(OP) \
    OP(SYS_getpid, __NR_getpid, ZERO_ARGS) \
    OP(SYS_write, __NR_write, THREE_ARGS)

//"system-call is done via the syscall instruction. The kernel destroys
// registers %rcx and %r11." - https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.99.pdf
// ^ need to mark %rcx and %r11 as clobbered

// "User-level applications use as integer registers for passing the sequence
// %rdi, %rsi, %rdx, %rcx, %r8 and %r9. The kernel interface uses 
// %rdi, %rsi, %rdx, %r10, %r8 and %r9"
// Judging by sysctl, the ps4 uses this convention also, moving %rcx to %r10 before
// the syscall instruction

// ZERO_ARGS ... SIX_ARGS
// Macros to invoke the native (linux) syscall, with the arguments directly mapped
// from the mcontext registers
// Puts the result in output_param
#define ZERO_ARGS(native_syscall_nr, mcontext, output_param) \
    asm volatile( \
        "mov %1, %%rax\n" \
        "syscall\n" \
        "mov %%rax, %0\n" \
        : "=r" (output_param) \
        : "i" (native_syscall_nr) \
        : "rcx", "r11", "rax" \
    );

#define ONE_ARG(native_syscall_nr, mcontext, output_param) \
    asm volatile( \
        "mov %1, %%rax\n" \
        "mov %2, %%rdi\n" \
        "syscall\n" \
        "mov %%rax, %0\n" \
        : "=r" (output_param) \
        : "i" (native_syscall_nr), \
            "g" (mcontext->gregs[REG_RDI]) \
        : "rcx", "r11", "rax", "rdi" \
    );

#define TWO_ARGS(native_syscall_nr, mcontext, output_param) \
    asm volatile( \
        "mov %1, %%rax\n" \
        "mov %2, %%rdi\n" \
        "mov %3, %%rsi\n" \
        "syscall\n" \
        "mov %%rax, %0\n" \
        : "=r" (output_param) \
        : "i" (native_syscall_nr), \
            "g" (mcontext->gregs[REG_RDI]), \
            "g" (mcontext->gregs[REG_RSI]) \
        : "rcx", "r11", "rax", "rdi", "rsi" \
    );

#define THREE_ARGS(native_syscall_nr, mcontext, output_param) \
    asm volatile( \
        "mov %1, %%rax\n" \
        "mov %2, %%rdi\n" \
        "mov %3, %%rsi\n" \
        "mov %4, %%rdx\n" \
        "syscall\n" \
        "mov %%rax, %0\n" \
        : "=r" (output_param) \
        : "i" (native_syscall_nr), \
            "g" (mcontext->gregs[REG_RDI]), \
            "g" (mcontext->gregs[REG_RSI]), \
            "g" (mcontext->gregs[REG_RDX]) \
        : "rcx", "r11", "rax", "rdi", "rsi", "rdx" \
    );

#define FOUR_ARGS(native_syscall_nr, mcontext, output_param) \
    asm volatile( \
        "mov %1, %%rax\n" \
        "mov %2, %%rdi\n" \
        "mov %3, %%rsi\n" \
        "mov %4, %%rdx\n" \
        "mov %5, %%r10\n" \
        "syscall\n" \
        "mov %%rax, %0\n" \
        : "=r" (output_param) \
        : "i" (native_syscall_nr), \
            "g" (mcontext->gregs[REG_RDI]), \
            "g" (mcontext->gregs[REG_RSI]), \
            "g" (mcontext->gregs[REG_RDX]), \
            "g" (mcontext->gregs[REG_R10]) \
        : "rcx", "r11", "rax", "rdi", "rsi", "rdx", "r10" \
    );

#define FIVE_ARGS(native_syscall_nr, mcontext, output_param) \
    asm volatile( \
        "mov %1, %%rax\n" \
        "mov %2, %%rdi\n" \
        "mov %3, %%rsi\n" \
        "mov %4, %%rdx\n" \
        "mov %5, %%r10\n" \
        "mov %6, %%r8\n" \
        "syscall\n" \
        "mov %%rax, %0\n" \
        : "=r" (output_param) \
        : "i" (native_syscall_nr), \
            "g" (mcontext->gregs[REG_RDI]), \
            "g" (mcontext->gregs[REG_RSI]), \
            "g" (mcontext->gregs[REG_RDX]), \
            "g" (mcontext->gregs[REG_R10]), \
            "g" (mcontext->gregs[REG_R8]) \
        : "rcx", "r11", "rax", "rdi", "rsi", "rdx", "r10", "r8" \
    );

#define SIX_ARGS(native_syscall_nr, mcontext, output_param) \
    asm volatile( \
        "mov %1, %%rax\n" \
        "mov %2, %%rdi\n" \
        "mov %3, %%rsi\n" \
        "mov %4, %%rdx\n" \
        "mov %5, %%r10\n" \
        "mov %6, %%r8\n" \
        "mov %6, %%r9\n" \
        "syscall\n" \
        "mov %%rax, %0\n" \
        : "=r" (output_param) \
        : "i" (native_syscall_nr), \
            "g" (mcontext->gregs[REG_RDI]), \
            "g" (mcontext->gregs[REG_RSI]), \
            "g" (mcontext->gregs[REG_RDX]), \
            "g" (mcontext->gregs[REG_R10]), \
            "g" (mcontext->gregs[REG_R8]), \
            "g" (mcontext->gregs[REG_R9]) \
        : "rcx", "r11", "rax", "rdi", "rsi", "rdx", "r10", "r8", "r9" \
    );


static greg_t handle_open(mcontext_t *mcontext) {
    int rv;
    fs::path modded_path;

    const char *name = *reinterpret_cast<char **>(&mcontext->gregs[REG_RDI]);
    int flags = *reinterpret_cast<int *>(&mcontext->gregs[REG_RSI]);
    mode_t mode = *reinterpret_cast<mode_t *>(&mcontext->gregs[REG_RDX]);

    fprintf(stderr, "\tname: %s\n"
        "\tflags: %d\n",
        name,
        flags
    );

    if (name[0] == '/' && is_chrooted()) {
        modded_path = get_chroot_path();
        modded_path += name;
        name = modded_path.c_str();
    };

    fprintf(stderr, "\tmodded_name: %s\n", name);

    rv = open(name, flags, mode);
    if (rv < 0) {
        rv = -errno;
    }
    return rv;
}

static greg_t handle_close(mcontext_t *mcontext) {
    int rv;

    int fd = *reinterpret_cast<int *>(&mcontext->gregs[REG_RDI]);
    rv = close(fd);
    if (rv) {
        rv = -errno;
    }
    return rv;
}

#define PS4_IOCPARM_SHIFT   13              /* number of bits for ioctl size */
#define PS4_IOCPARM_MASK    ((1 << PS4_IOCPARM_SHIFT) - 1) /* parameter length mask */
#define PS4_IOCPARM_LEN(x)  (((x) >> 16) & PS4_IOCPARM_MASK)
#define PS4_IOCBASECMD(x)   ((x) & ~(PS4_IOCPARM_MASK << 16))
#define PS4_IOCGROUP(x)     (((x) >> 8) & 0xff)

#define PS4_IOCPARM_MAX     (1 << PS4_IOCPARM_SHIFT) /* max size of ioctl */
#define PS4_IOC_VOID        0x20000000      /* no parameters */
#define PS4_IOC_OUT         0x40000000      /* copy out parameters */
#define PS4_IOC_IN          0x80000000      /* copy in parameters */
#define PS4_IOC_INOUT       (PS4_IOC_IN|PS4_IOC_OUT)
#define PS4_IOC_DIRMASK     (PS4_IOC_VOID|PS4_IOC_OUT|PS4_IOC_IN)

static greg_t handle_ioctl(mcontext_t *mcontext) {
    int rv;

    int fd = *reinterpret_cast<int *>(&mcontext->gregs[REG_RDI]);
    uint32_t request = *reinterpret_cast<uint32_t *>(&mcontext->gregs[REG_RSI]);
    void *argp = *reinterpret_cast<void **>(&mcontext->gregs[REG_RDX]);

    uint32_t group = PS4_IOCGROUP(request);
    uint32_t num = request & 0xff;
    uint32_t len = PS4_IOCPARM_LEN(request);

    fprintf(stderr,
        "\t%s\n"
        "\tgroup: %c (%d)\n"
        "\tnum: %d\n"
        "\tlen: %d\n",
        PS4_IOC_INOUT == (request & PS4_IOC_INOUT) ? "inout" 
            : (request & PS4_IOC_IN ? "in (write)"
                : (request & PS4_IOC_OUT ? "out (read)" : "void")),
        group, group,
        num,
        len
    );

    rv = -EINVAL;

    switch(group) {
        case 136:
        {
            switch(num) {
                case 6:
                {
                    memset(argp, 0, len);
                    rv = 0;
                }
                default:
                    break;
            }
        }
        default:
            break;
    }

    return rv;
}

static greg_t handle_sysctl(mcontext_t *mcontext) {
    int *name = *reinterpret_cast<int **>(&mcontext->gregs[REG_RDI]);
    uint namelen = *reinterpret_cast<uint *>(&mcontext->gregs[REG_RSI]);
    void *oldp = *reinterpret_cast<void **>(&mcontext->gregs[REG_RDX]);
    size_t *oldlenp = *reinterpret_cast<size_t **>(&mcontext->gregs[REG_R10]);
    void *newp = *reinterpret_cast<void **>(&mcontext->gregs[REG_R8]);
    bool is_write = newp != NULL;

    fprintf(stderr, "\tname: %d.%d.%d.%d\n"
        "\tnamelen: %d\n"
        "\toldlenp: %zu\n"
        "\twrite? : %s\n",
        name[0], name[1], name[2], name[3],
        namelen,
        *oldlenp,
        newp ? "yes" : "no"
    );

    // ??? = 1.37.64 (length 2)    - CTL_KERN.KERN_ARND
    // "kern.proc.ptc" = 0.3       - CTL_UNSPEC.?
    // ??? = 1.14.35.59262         - CTL_KERN.KERN_PROC.?.?
    //                fppcs4 says 1.14.35 is KERN_PROC_APPINFO
    // 1.33                        - CTL_KERN.KERN_USRSTACK
    //                from gnmdriver
    // 6.7                         - CTL_HW.HW_PAGESIZE

    greg_t rv = -ENOENT;
    switch(name[0]) {
        case CTL_KERN:
        {
            switch(name[1]) {
                case KERN_PROC:
                {
                    switch(name[2]) {
                        case KERN_PROC_APPINFO:
                        default:
                            break;
                    }
                    break;
                }
                case KERN_USRSTACK:
                {
                    // TODO:
                    // pthread_attr has lowest addressable byte.
                    // should this return highest addressable byte?
                    // during libSceGnmDriver init, it tries to mmap starting at stack_addr - 0x201000
                    // (diff is 2052 KB).
                    // The usrstack is allocated as 8192KB (for now)
                    // Maybe I should malloc with 2048KB.
                    // That means the red zone is 4KB, which matches the mmap'd size
                    assert(!is_write);
                    assert(*oldlenp == 8);
                    pthread_t pid = pthread_self();
                    pthread_attr_t attr;
                    assert(!pthread_getattr_np(pid, &attr));
                    void *stack_addr;
                    size_t stack_size;
                    assert(!pthread_attr_getstack(&attr, &stack_addr, &stack_size));
                    uint64_t stack_top = (uint64_t) stack_addr + stack_size;
                    *reinterpret_cast<void**>(oldp) = (void *)stack_top;
                    rv = 0;
                    break;
                }
                case KERN_ARND:
                    assert(!is_write);
                    assert(oldlenp);
                    arc4random_buf(oldp, *oldlenp);
                    rv = 0;
                    break;
                default:
                    break;
            }
            break;
        }
        case CTL_HW:
        {
            switch(name[1]) {
                case HW_PAGESIZE:
                {
                    assert(!is_write);
                    size_t pagesize = sysconf(_SC_PAGE_SIZE);
                    if (*oldlenp == 0 || *oldlenp == 4) {
                        *oldlenp = 4;
                        uint32_t pgsz_bits = pagesize;
                        memcpy(oldp, &pgsz_bits, sizeof(pgsz_bits));
                        rv = 0;
                    } else if (*oldlenp == 8) {
                        uint64_t pgsz_bits = pagesize;
                        memcpy(oldp, &pgsz_bits, sizeof(pgsz_bits));
                        rv = 0;
                    } else {
                        fprintf(stderr, "Unhandled oldlenp size\n");
                        rv = EINVAL;
                    }
                }
                default:
                    break;
            }
        }
        default:
            break;
    }

    if (rv) {
        fprintf(stderr, "sysctl: error: %s\n", strerror(-rv));
    }
    return rv;
}

static greg_t handle_mmap(mcontext_t *mcontext) {
    void *addr = *reinterpret_cast<void **>(&mcontext->gregs[REG_RDI]);
    size_t len = *reinterpret_cast<size_t *>(&mcontext->gregs[REG_RSI]);
    int prot = *reinterpret_cast<int *>(&mcontext->gregs[REG_RDX]);
    int flags = *reinterpret_cast<int *>(&mcontext->gregs[REG_R10]);
    int fd = *reinterpret_cast<int *>(&mcontext->gregs[REG_R8]);
    off_t offset = *reinterpret_cast<off_t *>(&mcontext->gregs[REG_R9]);

    greg_t rv;

    //pthread_t pid = pthread_self();
    //pthread_attr_t attr;
    //assert(!pthread_getattr_np(pid, &attr));
    //void *stack_addr;
    //size_t stack_size;
    //assert(!pthread_attr_getstack(&attr, &stack_addr, &stack_size));    

    // Call mmap wrapper to convert BSD flags to linux flags
    // TODO: define mmap_wrapper in helper librarie, to be called from here and by ps4 libs when
    // resolving mmap library call
    void *result_addr = mmap(addr, len, prot, flags, fd, offset);
    if (result_addr == MAP_FAILED) {
        rv = -errno;
    } else {
        rv = *reinterpret_cast<greg_t *>(&result_addr);
    }
    return rv;
}

extern "C" {

void freebsd_syscall_handler(int num, siginfo_t *info, void *ucontext_arg) {
    ucontext_t *ucontext = (ucontext_t *) ucontext_arg;
    mcontext_t *mcontext = &ucontext->uc_mcontext;

    int64_t rv = -EINVAL;

    greg_t ps4_syscall_nr = mcontext->gregs[REG_RAX];

    std::string bsdName = to_string((BsdSyscallNr) ps4_syscall_nr);
    printf("freebsd_syscall_handler: handling %s\n", bsdName.c_str());

    //greg_t arg1 = mcontext->gregs[REG_RDI];
    //greg_t arg2 = mcontext->gregs[REG_RSI];
    //greg_t arg3 = mcontext->gregs[REG_RDX];
    //greg_t arg4 = mcontext->gregs[REG_R10];
    //greg_t arg5 = mcontext->gregs[REG_R8];
    //greg_t arg6 = mcontext->gregs[REG_R9];

    switch (ps4_syscall_nr) {
// For some syscall that is 1:1 freebsd to native (linux), create a case statement
// for freebsd number bsd_nr, that invokes the native syscall with number native_nr.
// Pass N-Args directly from the mcontext to the native syscall.
#define DIRECT_SYSCALL_CASE(ps4_nr, native_nr, NUM_ARG_FUNCTION) \
        case ps4_nr: \
        { \
            NUM_ARG_FUNCTION(native_nr, mcontext, rv) \
            break; \
        }

// Declare case statements for each bsd -> native syscall mapping
// Only do this for the syscalls that map 1:1 with no intervention needed
DIRECT_SYSCALL_MAP(DIRECT_SYSCALL_CASE)

#undef DIRECT_SYSCALL_CASE

        case SYS_open:
        {
            rv = handle_open(mcontext);
            break;
        }

        case SYS_close:
        {
            rv = handle_close(mcontext);
            break;
        }

        case SYS_ioctl:
        {
            rv = handle_ioctl(mcontext);
            break;
        }

        case SYS___sysctl:
        {
            rv = handle_sysctl(mcontext);
            break;
        }
        case SYS_thr_self:
        {
            rv = gettid();
            break;
        }
        case SYS_mmap:
        {
            rv = handle_mmap(mcontext);
            break;
        }
        case 587:
            // get_authinfo
            rv = EINVAL;
            break;
        case 612:
            // nG-FYqFutUo
            rv = EINVAL;
            break;

        default:
        {
            std::string bsdName = to_string((BsdSyscallNr) ps4_syscall_nr);
            fprintf(stderr, "Unhandled syscall : %lli (%s)\n", ps4_syscall_nr, bsdName.c_str());
            break;
        }
    }

    // before this point, rv should be -errno if there was an error
    // change to bsd conventions here (rax holds result or positive errno, CF means error)
    if (rv < 0) {
        rv = -rv;
        // Set carry flag. ps4 libs and freebsd expect this
        mcontext->gregs[REG_EFL] |= 1;
    } else {
        // Clear carry flag (no error)
        mcontext->gregs[REG_EFL] &= (~1llu);
    }
    mcontext->gregs[REG_RAX] = rv;
}

}
