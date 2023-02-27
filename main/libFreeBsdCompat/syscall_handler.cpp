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

// TODO figure out if the ps4 passes %rdi, %rsi, %rdx, %r10, %r8, %r9 like amd64 linux

extern "C" {

void freebsd_syscall_handler(int num, siginfo_t *info, void *ucontext_arg) {
    ucontext_t *ucontext = (ucontext_t *) ucontext_arg;
    mcontext_t *mcontext = &ucontext->uc_mcontext;

    int64_t rv = EINVAL;

    greg_t bsd_syscall_nr = mcontext->gregs[REG_RAX];

    switch (bsd_syscall_nr) {
#define DIRECT_SYSCALL_CASE(bsd_nr, native_nr, NUM_ARG_FUNCTION) \
        case bsd_nr: \
        { \
            std::string bsdName = to_string(bsd_nr); \
            fprintf(stderr, "freebsd_syscall_handler: handling %s\n", bsdName.c_str()); \
            NUM_ARG_FUNCTION(native_nr, mcontext, rv) \
            break; \
        }

        DIRECT_SYSCALL_MAP(DIRECT_SYSCALL_CASE)

#undef DIRECT_SYSCALL_CASE

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