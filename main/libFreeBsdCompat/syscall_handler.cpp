#include <asm-generic/errno-base.h>
#include <cstdint>
#include <unistd.h>
#include <asm/unistd_64.h>
#include <stdio.h>
#include "freebsd_compat.h"
#include "orbis/freebsd_9.0_syscalls.hpp"
#include <sys/ucontext.h>

#define DIRECT_SYSCALL_MAP(OP) \
    OP(SYS_getpid, __NR_getpid, ZERO_ARGS)

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
            "g" (mcontext->gregs[REG_RSI]) \
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
            "g" (mcontext->gregs[REG_RSI]) \
            "g" (mcontext->gregs[REG_RDX]) \
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
            "g" (mcontext->gregs[REG_RSI]) \
            "g" (mcontext->gregs[REG_RDX]) \
            "g" (mcontext->gregs[REG_R10]) \
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
            "g" (mcontext->gregs[REG_RSI]) \
            "g" (mcontext->gregs[REG_RDX]) \
            "g" (mcontext->gregs[REG_R10]) \
            "g" (mcontext->gregs[REG_R8]) \
            "g" (mcontext->gregs[REG_R9]) \
        : "rcx", "r11", "rax", "rdi", "rsi", "rdx", "r10", "r8", "r9" \
    );

extern "C" {

void freebsd_syscall_handler(int num, siginfo_t *info, void *ucontext_arg) {
    ucontext_t *ucontext = (ucontext_t *) ucontext_arg;
    mcontext_t *mcontext = &ucontext->uc_mcontext;

    int64_t rv = EINVAL;

    greg_t bsd_syscall_nr = mcontext->gregs[REG_RAX];

    switch (bsd_syscall_nr) {
// For some syscall that is 1:1 freebsd to native (linux), create a case statement
// for freebsd number bsd_nr, that invokes the native syscall with number native_nr.
// Pass N-Args directly from the mcontext to the native syscall.
#define DIRECT_SYSCALL_CASE(bsd_nr, native_nr, NUM_ARG_FUNCTION) \
        case bsd_nr: \
        { \
            std::string bsdName = to_string(bsd_nr); \
            fprintf(stderr, "freebsd_syscall_handler: handling %s\n", bsdName.c_str()); \
            NUM_ARG_FUNCTION(native_nr, mcontext, rv) \
            break; \
        }

        // Declare case statements for each bsd -> native syscall mapping
        // Only do this for the syscalls that map 1:1 with no intervention needed
        DIRECT_SYSCALL_MAP(DIRECT_SYSCALL_CASE)

#undef DIRECT_SYSCALL_CASE

        case SYS___sysctl:
        {
            // ??? = 1.37.64 (length 2)
            // "kern.proc.ptc" = 0.3
            // ??? = 1.14.35.59262

            std::string bsdName = to_string((BsdSyscallNr) bsd_syscall_nr);
            fprintf(stderr, "freebsd_syscall_handler: handling %s\n", bsdName.c_str());
            uint namelen = mcontext->gregs[REG_RSI];
            int *name = (int *) mcontext->gregs[REG_RDI];
            void *newp = (void *) mcontext->gregs[REG_R8];
            fprintf(stderr, "name: %d.%d.%d.%d\n"
                "namelen: %d\n"
                "write? : %s\n",
                name[0], name[1], name[2], name[3],
                namelen,
                newp ? "yes" : "no"
            );
            break;
        }

        default:
        {
            std::string bsdName = to_string((BsdSyscallNr) bsd_syscall_nr);
            fprintf(stderr, "Unhandled syscall : %lli (%s)\n", bsd_syscall_nr, bsdName.c_str());
            break;
        }
    }

    if (rv < 0) {
        rv = -rv;
        // Set carry flag, ps4 and freebsd expects this
        mcontext->gregs[REG_EFL] |= 1;
    }
    mcontext->gregs[REG_RAX] = rv;
}

}