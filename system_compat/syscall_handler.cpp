#include <asm-generic/errno-base.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <optional>
#include <pthread.h>
#include <set>
#include <sys/mman.h>
#include <unistd.h>
#include <asm/unistd_64.h>
#include <stdio.h>
#include "syscall_handler.h"
#include "orbis/orbis_syscalls.h"
#include <sys/ucontext.h>
#include "ps4_sysctl.h"
#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include "system_compat/ps4_region.h"
#include "freebsd9.0/sys/rtprio.h"
#include "exported_wrappers.h"
#include "Common.h"

#define DIRECT_SYSCALL_MAP(OP) \
    OP(SYS_getpid, __NR_getpid, ZERO_ARGS) \
    OP(SYS_read, __NR_read, THREE_ARGS) \
    OP(SYS_write, __NR_write, THREE_ARGS) \
    OP(SYS_clock_gettime, __NR_clock_gettime, TWO_ARGS)

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

    const char *name = *reinterpret_cast<char **>(&mcontext->gregs[REG_RDI]);
    int flags = *reinterpret_cast<int *>(&mcontext->gregs[REG_RSI]);
    mode_t mode = *reinterpret_cast<mode_t *>(&mcontext->gregs[REG_RDX]);

    printf(CYN "open syscall: %s\n" RESET, name);
    rv = open_wrapper(name, flags, mode);
    
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

    //int fd = *reinterpret_cast<int *>(&mcontext->gregs[REG_RDI]);
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
        case 129:
        {
            // In libSceGnmDriver.prx.native init function
            switch (num) {
                case 16:
                    // under _ps4__sceGnmSetGsRingSizes
                    // *argp is 0x4000
                    // has to do with geom shaders.
                    // Notes in GPCS4
                    //rv = 0;
                    break;
                case 25:
                    // _ps4__sceGnmDisableMipStatsReport. called after 16
                    //rv = 0;
                    break;
                case 27:
                    // called after 25
                    //rv = 0;
                    break;
                default:
                    break;
            }
        }
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

    if (rv) {
        fprintf(stderr, RED "SYSCALL_HANDLER: ioctl error\n" RESET);
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

    printf("\tname: %d.%d.%d.%d\n"
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
                    // pthread_attr stack address has lowest addressable byte.
                    // should this return highest addressable byte? - do this for now
                    // during libSceGnmDriver init, it tries to mmap starting at stack_addr - 0x201000
                    // (diff is 2052 KB).
                    // That makes sense if the ps4 stack is 4096 KB, since the red zone would be the next 4KB underneath
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
        fprintf(stderr, RED "SYSCALL_HANDLER: sysctl error\n" RESET);
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

    // Call mmap wrapper to convert BSD flags to linux flags.
    void *result_addr = mmap_wrapper(addr, len, prot, flags, fd, offset);
    if (result_addr == MAP_FAILED) {
        rv = -errno;
    } else {
        rv = *reinterpret_cast<greg_t *>(&result_addr);
    }
    return rv;
}

static greg_t handle_rtprio_thread(mcontext_t *mcontext) {
    int function = *reinterpret_cast<int *>(&mcontext->gregs[REG_RDI]);
    // lwpid_t
    // int32_t lwpid = *reinterpret_cast<int32_t *>(&mcontext->gregs[REG_RSI]);
    struct rtprio *rtp = *reinterpret_cast<struct rtprio **>(&mcontext->gregs[REG_RDX]);

    greg_t rv = 0;

    if ( !rtp) {
        return -EINVAL;
    }

    switch (function) {
        case RTP_LOOKUP:
        {
            rtp->type = RTP_PRIO_NORMAL;
            rtp->prio = 0;
            break;
        }
        case RTP_SET:
        {
            break;
        }
        default:
            errno;
    }

    return rv;
}

static std::set<OrbisSyscallNr> green_syscalls = {
    //SYS_write,
    SYS_read,
    SYS_open,
    SYS_close,
    SYS_getpid,
    SYS_ioctl,
    SYS___sysctl,
    SYS_rtprio_thread,
    SYS_thr_self,
    SYS_mmap,
    SYS_clock_gettime,
};

static std::set<OrbisSyscallNr> red_syscalls = {
    SYS_netcontrol,
    SYS_evf_create,
	SYS_osem_create,
	SYS_osem_delete,
	SYS_dmem_container,
	SYS_get_authinfo,
	SYS_mname,
	SYS_dynlib_get_proc_param,
	SYS_mdbg_service,
    SYS_randomized_path,
	SYS_budget_get_ptype,
	SYS_get_proc_type_info,
};

extern "C" {

void freebsd_syscall_handler(int num, siginfo_t *info, void *ucontext_arg) {
    HostRegionScope __scope;

    ucontext_t *ucontext = (ucontext_t *) ucontext_arg;
    mcontext_t *mcontext = &ucontext->uc_mcontext;
    greg_t rv = -EINVAL;

    greg_t ps4_syscall_nr = mcontext->gregs[REG_RAX];

    std::string bsdName = to_string((OrbisSyscallNr) ps4_syscall_nr);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    greg_t arg1 = mcontext->gregs[REG_RDI];
    greg_t arg2 = mcontext->gregs[REG_RSI];
    greg_t arg3 = mcontext->gregs[REG_RDX];
    greg_t arg4 = mcontext->gregs[REG_R10];
    greg_t arg5 = mcontext->gregs[REG_R8];
    greg_t arg6 = mcontext->gregs[REG_R9];
#pragma GCC diagnostic pop

    if (red_syscalls.find((OrbisSyscallNr) ps4_syscall_nr) != red_syscalls.end()) {
        fprintf(stderr, RED "freebsd_syscall_handler: handling %s\n" RESET, bsdName.c_str());
    } else if (green_syscalls.find((OrbisSyscallNr) ps4_syscall_nr) != green_syscalls.end()) {
        printf(GRN "freebsd_syscall_handler: handling %s%s\n", bsdName.c_str(), RESET);
    }

    switch (ps4_syscall_nr) {
// For some syscall that is 1:1 orbis to native host (linux), create a case statement
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

        case SYS_netbsd_msync:
        case 277:
        {
            bsdName = to_string(SYS_netbsd_msync);
            rv = -EINVAL;
        }

        case 99:
        {
            // _ps4____sys_netcontrol
            // during libSceNet init
            //
            // then crash in _ps4____tls_get_addr
            rv = -EINVAL;
            break;
        }

        case SYS_shmctl:
        case 169:
        case 170:
        case 171:
        case 220:
        case 339:
        case 377:
        case 457:
        case 458:
        case 459:
        case 460:
        case 461:
        case 462:
        case 505:
        case 510:
        case 511:
        case 512:
        {
            bsdName = to_string(SYS_shmctl);
            rv = -EINVAL;
            break;
        }

        case SYS___sysctl:
        {
            rv = handle_sysctl(mcontext);
            break;
        }

        case SYS_lkmnosys:
        case 211:
        case 212:
        case 213:
        case 214:
        case 215:
        case 216:
        case 217:
        case 218:
        case 219:
        {
            bsdName = to_string(SYS_lkmnosys);
            // SYS_lkmnosys
            rv = -EINVAL;
            break;
        }

        case SYS_netbsd_lchown:
        case 275:
        {
            bsdName = to_string(SYS_netbsd_lchown);
            rv = -EINVAL;
            break;
        }

        case SYS_rtprio_thread:
        {
            rv = handle_rtprio_thread(mcontext);
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
        case SYS_osem_create:
            rv = -EINVAL;
            break;
        case SYS_osem_delete:
            rv = -EINVAL;
            break;
        case SYS_dmem_container:
            rv = -EINVAL;
            break;
        case SYS_get_authinfo:
            // get_authinfo
            // ScePthread: Fatal error 'Can't allocate initial thread' (errno = 22)            
            //rv = -EINVAL;
            rv = 0;
            break;
        case SYS_mname:
            // Can't allocate SceGnmGpuInfo memory - after this
            //rv = -EINVAL;
            rv = -EINVAL;
            //rv = 0;
            // If return 0,
            // gnmdriver init calls 588, mmap, 588, 601, 588
            break;
        case SYS_dynlib_get_proc_param:
            // somewhere under __ps4__malloc_init, libc startup
            rv = -EINVAL;
            break;
        case SYS_mdbg_service:
            // mdbg_service
            rv = -EINVAL;
            break;
        case SYS_randomized_path:
            // ... syscall handler
            // libkernel : _ps4____sys_randomized_path
            // libkernel : _ps4__sceKernelGetFsSandboxRandomWord
            // libSceDiscMap init
            //
            // prints "[DiskMap BitmapInfo] get path failed"
            rv = -EINVAL;
            break;
        case SYS_budget_get_ptype:
            rv = -EINVAL;
            break;
        case SYS_get_proc_type_info:
            // nG-FYqFutUo
            rv = -EINVAL;
            break;

        default:
        {
            std::string bsdName = to_string((OrbisSyscallNr) ps4_syscall_nr);
            fprintf(stderr, RED "Unhandled syscall : %lli (%s)\n" RESET, ps4_syscall_nr, bsdName.c_str());
            rv = -EINVAL;
            break;
        }
    }

    // before this point, rv should be -errno if there was an error
    // change to bsd conventions here (rax holds result or positive errno, CF means error)
    if (rv < 0) {
        rv = -rv;
        // Set carry flag. ps4/bsd code expects carry flag to be 1 on errors
        mcontext->gregs[REG_EFL] |= 1;
    } else {
        // Clear carry flag (no error)
        mcontext->gregs[REG_EFL] &= (~1llu);
    }
    mcontext->gregs[REG_RAX] = rv;
}

}
