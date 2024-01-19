#include <asm-generic/errno-base.h>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <set>
#include <sys/mman.h>
#include <stdio.h>
#include "syscall_handler.h"
#include "orbis/orbis_syscalls.h"
#include <sys/ucontext.h>
#include "ps4_sysctl.h"
#include <cassert>
#include <fcntl.h>
#include <filesystem>
#include "system_compat/globals.h"
#include "system_compat/ps4_region.h"
#include "freebsd9.0/sys/rtprio.h"
#include "exported_wrappers.h"
#include "Common.h"
#include <semaphore.h>
#include <sys/stat.h>
#include "orbis/sce_errors/sce_kernel_error.h"
#include "orbis/sce_file.h"

#define DIRECT_SYSCALL_MAP(OP) \
    OP(OrbisSyscallNr::ORB_SYS_getpid, __NR_getpid, ZERO_ARGS) \
    OP(OrbisSyscallNr::ORB_SYS_read, __NR_read, THREE_ARGS) \
    OP(OrbisSyscallNr::ORB_SYS_write, __NR_write, THREE_ARGS) \
    OP(OrbisSyscallNr::ORB_SYS_clock_gettime, __NR_clock_gettime, TWO_ARGS)

//"system-call is done via the syscall instruction. The kernel destroys
// registers %rcx and %r11." - https://refspecs.linuxfoundation.org/elf/x86_64-abi-0.99.pdf
// ^ need to mark %rcx and %r11 as clobbered

// "User-level applications use as integer registers for passing the sequence
// %rdi, %rsi, %rdx, %rcx, %r8 and %r9. The kernel interface uses 
// %rdi, %rsi, %rdx, %r10, %r8 and %r9"
// Judging by sysctl, the ps4 uses this convention also, moving %rcx to %r10 before
// the syscall instruction

template<typename T>
static inline T ARG1(mcontext_t *mcontext) { return *reinterpret_cast<T *>(&mcontext->gregs[REG_RDI]); }
template<typename T>
static inline T ARG2(mcontext_t *mcontext) { return *reinterpret_cast<T *>(&mcontext->gregs[REG_RSI]); }
template<typename T>
static inline T ARG3(mcontext_t *mcontext) { return *reinterpret_cast<T *>(&mcontext->gregs[REG_RDX]); }
template<typename T>
static inline T ARG4(mcontext_t *mcontext) { return *reinterpret_cast<T *>(&mcontext->gregs[REG_R10]); }
template<typename T>
static inline T ARG5(mcontext_t *mcontext) { return *reinterpret_cast<T *>(&mcontext->gregs[REG_R8]); }
template<typename T>
static inline T ARG6(mcontext_t *mcontext) { return *reinterpret_cast<T *>(&mcontext->gregs[REG_R9]); }

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
    auto *name = ARG1<const char *>(mcontext);
    auto sceFlags = ARG2<sce_file_flags_t>(mcontext);
    auto sceMode = ARG3<sce_mode_t>(mcontext);

    printf(CYN "open syscall: %s\n" RESET, name);
    rv = open_wrapper(name, sceFlags, sceMode);
    
    if (rv < 0) {
        rv = -errno;
    }
    return rv;
}

static greg_t handle_close(mcontext_t *mcontext) {
    int rv;
    auto fd = ARG1<int>(mcontext);

    rv = close(fd);
    if (rv) {
        rv = -errno;
    }
    return rv;
}

#define BSD_IOCPARM_SHIFT   13              /* number of bits for ioctl size */
#define BSD_IOCPARM_MASK    ((1 << BSD_IOCPARM_SHIFT) - 1) /* parameter length mask */
#define BSD_IOCPARM_LEN(x)  (((x) >> 16) & BSD_IOCPARM_MASK)
#define BSD_IOCBASECMD(x)   ((x) & ~(BSD_IOCPARM_MASK << 16))
#define BSD_IOCGROUP(x)     (((x) >> 8) & 0xff)

#define BSD_IOCPARM_MAX     (1 << BSD_IOCPARM_SHIFT) /* max size of ioctl */
#define BSD_IOC_VOID        0x20000000      /* no parameters */
#define BSD_IOC_OUT         0x40000000      /* copy out parameters */
#define BSD_IOC_IN          0x80000000      /* copy in parameters */
#define BSD_IOC_INOUT       (BSD_IOC_IN|BSD_IOC_OUT)
#define BSD_IOC_DIRMASK     (BSD_IOC_VOID|BSD_IOC_OUT|BSD_IOC_IN)

static greg_t handle_ioctl(mcontext_t *mcontext) {
    int rv;
    //auto fd = ARG1<int>(mcontext);
    auto request = ARG2<uint32_t>(mcontext);
    auto *argp = ARG3<void *>(mcontext);

    uint32_t group = BSD_IOCGROUP(request);
    uint32_t num = request & 0xff;
    uint32_t len = BSD_IOCPARM_LEN(request);

    fprintf(stderr,
        "\t%s\n"
        "\tgroup: %c (%d)\n"
        "\tnum: %d\n"
        "\tlen: %d\n",
        BSD_IOC_INOUT == (request & BSD_IOC_INOUT) ? "inout"
            : (request & BSD_IOC_IN ? "in (write)"
                : (request & BSD_IOC_OUT ? "out (read)" : "void")),
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
                    // TODO check this, dont remember
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
    auto *name = ARG1<int *>(mcontext);
    auto namelen = ARG2<uint>(mcontext);
    auto *oldp = ARG3<void *>(mcontext);
    auto *oldlenp = ARG4<size_t *>(mcontext);
    auto *newp = ARG5<void *>(mcontext);
    auto newlen = ARG6<size_t>(mcontext);
    bool is_write = newp != NULL;

    // ??? = 1.37.64 (length 2)    - CTL_KERN.KERN_ARND
    // "kern.proc.ptc" = 0.3       - CTL_UNSPEC.?
    // ??? = 1.14.35.59262         - CTL_KERN.KERN_PROC.?.?
    //                fppcs4 says 1.14.35 is KERN_PROC_APPINFO
    // 1.33                        - CTL_KERN.KERN_USRSTACK
    //                from gnmdriver
    // 6.7                         - CTL_HW.HW_PAGESIZE

    // "kern.smp.cpus" - 0.3.0.0, len 2 (not 3). also "kern.sched.cpusetsize"

    greg_t rv = -ENOENT;
    switch(name[0]) {
        case 0:
        {
            switch (name[1]) {
                case 3:
                {
                    rv = 0; // TODO
                    *reinterpret_cast<int *>(oldp) = 8;
                    *oldlenp = 1;
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
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
                    fprintf(stderr, GRN "sysctl name: CTL_HW.KERN_USRSTACK\n" RESET);
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
                    fprintf(stderr, GRN "sysctl name: CTL_KERN.KERN_ARND\n" RESET);
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
                    fprintf(stderr, GRN "sysctl name: CTL_KERN.HW_PAGESIZE\n" RESET);
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
        fprintf(stderr, "\tname: %d.%d.%d.%d\n"
            "\tnamelen: %d\n"
            "\toldlenp: %zu\n"
            "\twrite? : %s\n",
            name[0], name[1], name[2], name[3],
            namelen,
            *oldlenp,
            newp ? "yes" : "no"
        );        
    }
    return rv;
}

static greg_t handle_mmap(mcontext_t *mcontext) {
    auto *addr = ARG1<void *>(mcontext);
    auto len = ARG2<size_t>(mcontext);
    auto prot = ARG3<int>(mcontext);
    auto flags = ARG4<int>(mcontext);
    auto fd = ARG5<int>(mcontext);
    auto offset = ARG6<off_t>(mcontext);

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
    auto function = ARG1<int>(mcontext);
    // auto lwpid = ARG2<int32_t>(mcontext)
    auto *rtp = ARG3<struct rtprio *>(mcontext);

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

static greg_t handle_dynlib_get_proc_param(mcontext_t *mcontext) {
    greg_t rv;

    auto *pProcParam = ARG1<void **>(mcontext);
    auto *arg2 = ARG2<uint64_t *>(mcontext);

    // syscall seems to operate on two addresses.

    // one call stack:
    // syscall_handler
    // anon_fn1(arg1, arg2, arg3) <- arg3 not set by caller
    //              decompiler looks wrong: asm just seems to do syscall, ret if CF==0
    // _ps4__sceKernelGetProcParam() : returns anonfn1 ? 0 : *arg1
    //              returns 8 bytes - ptr?
    // _ps4___malloc_init (in libc.prx.so INI)

    // 2 stack variable pointers: must be return params, since they are uninitialized before syscall in caller.
    // arg2 possibly unused by _ps4__sceKernelGetProcParam? Does another caller variant exist which does the syscall
    // but actually uses *arg2?

    // global proc param should be set after loading eboot.bin
    // procparam contains other pointers, but there are relocations inside the procparam contents which
    // handle this. So we can load eboot.bin, if position-independent, at an arbitrary base address with dlopen().
    // getProcParam should return the correct address after main loads eboot.bin as a dynamic library.
    void *procParam = getProcParam();
    if (procParam) {
        rv = 0;
        *pProcParam = procParam;
    } else {
        // EINVAL?
        rv = -EINVAL;
    }

    return rv;
}

static greg_t handle_mname(mcontext_t *mcontext) {
    auto *addr = ARG1<void *>(mcontext);
    auto len = ARG2<size_t>(mcontext);
    auto *name = ARG3<const char *>(mcontext);

    greg_t rv;
    // Can't allocate SceGnmGpuInfo memory - appears after error here
    // return addr fixes this

    // under _ps4__sceKernelMapNamedFlexibleMemory
    // Usually Follows mmap
    // names memory?
    // arg1 and arg2 identical to previous mmaps

    //rv = -EINVAL;
    rv = 0;
    return rv;
}

template<uint NUM_SCE_BITS, typename V>
class Ps4HandleMap {
    // A class that manages some pool of opaque handles, e.g. sce semaphores.
    // Hand a simple integer key back to the ps4 application, when the ps4 app/lib tries to get a handle to some
    // resource.
    // We should allocate the appropriate linux resource and map the opaque handle, which we give back to the
    // app, to the linux resource.
    //
    // Useful to handle size differences in handles.
    // E.g. if a ps4 lib expects a 4 byte handle when the linux handle is an 8-byte pointer.
public:
    // give 8B size to the generic sce_type handle, but enforce that the actual used bits stays under SCE_NUM_BITS 
    typedef size_t sce_type;
    typedef V host_type;

    // check that size_t is wide enough to hold the opaque handle
    static_assert(sizeof(sce_type) * CHAR_BIT >= NUM_SCE_BITS);

    sce_type getNewHandle(host_type hostResourceHandle) {
        std::unique_lock<std::mutex> _lock(mutex);

        // Enforce that the MSB doesn't go 0 -> 1
        // A ps4 lib might check the sign of the handle for an error
        size_t msb = 1 << (NUM_SCE_BITS - 1);
        assert(maxSceHandle != ~msb); // Would overflow to signed with next increment, eg
        // maxSceHandle=0111 for SCE_NUM_BITS=4
        sce_type newId = ++maxSceHandle;
        ps4ToHostHandle.insert(std::make_pair(newId, hostResourceHandle));
        // TODO: make smarter data structure so we can reclaim freed slots
        // This will crash after 2^NUM_SCE_BITS handles
        return newId;
    }
    std::optional<host_type> find(sce_type key) {
        std::unique_lock<std::mutex> _lock(mutex);

        const auto& it = ps4ToHostHandle.find(key);
        if (it != ps4ToHostHandle.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    // Doesn't free the linux resource, only removes sce -> linux mapping
    bool remove(sce_type key) {
        std::unique_lock<std::mutex> _lock(mutex);

        return 1 == ps4ToHostHandle.erase(key);
    }

private:
    std::mutex mutex;
    std::map<sce_type, host_type> ps4ToHostHandle;
    size_t maxSceHandle = 0;
};

// The ps4 app/libs expect a OSem's to have a 32 bit handle.
// On linux, the handle is a sem_t *pointer.
// For ps4 functions that interact with OSem's, let the app pass around 32 bit handles,
// while the compatability layer maps these to linux sem_t's 
typedef Ps4HandleMap<32, sem_t *> OsemHandleMap;
static OsemHandleMap _OsemHandles;

static greg_t handle_osem_create(mcontext_t *mcontext) {
    greg_t rv;
    // libkernel wrapper: _ps4__sceKernelCreateSema
    // param_1: return param for sem type
    // param_2: name
    // param_3: int oflag or sce_mode_t mode?  (as in open, eg O_CREAT for oflag, user/group/others perms for mode)
    //      sce_mode_t would make more sense - "create" implies O_CREAT
    //      can probably ignore
    // param_4: init count?
    // param_5: GPCS4 says max count (I guess post() would error at the max)?
    // param_6: some kind of mode: if 0, makes syscall, if 1, sets return param to big value
    //      can probably ignore

    // 4 params to syscall: param_2, param_3, param_4, param_5
    auto *name = ARG1<const char *>(mcontext);
    auto sceMode = ARG2<sce_mode_t>(mcontext);
    auto init = ARG3<int>(mcontext);
    // TODO do anything with this?
    auto maxCount = ARG4<int>(mcontext);

    mode_t mode = sceModeToLinux(sceMode);

    // TODO: see verify sem_t is usable handle for callers
    // "sem.SceNpTusGame" created with SCE_S_IXOTH - might need to set USR R/W
    // if not set, next osem_create will fail with "permission denied"
    // Use O_EXCL? - EXCL returns error if exists
    sem_t *sem = sem_open(name, O_CREAT, mode, init);
    if (sem == SEM_FAILED) {
        fprintf(stderr, RED "sem_open failed: %s\n", strerror(errno));
        rv = -errno;
    } else {
        OsemHandleMap::sce_type sceHandle = _OsemHandles.getNewHandle(sem);
        // TODO handle error
        rv = sceHandle;
        printf(MAG "osem create: %lu -> %p\n", sceHandle, sem);
    }

    return rv;
}

static greg_t handle_osem_delete(mcontext_t *mcontext) {
    greg_t rv;
    auto sceHandle = ARG1<OsemHandleMap::sce_type>(mcontext);

    printf(MAG "osem delete: %lu\n", sceHandle);
    auto hostHandle = _OsemHandles.find(sceHandle);
    if ( !hostHandle) {
        fprintf(stderr, RED "sem_close: sce handle not found in OsemHandleMap\n");
        rv = -EINVAL;
    } else {
        rv = sem_close(hostHandle.value());
        if (rv < 0) {
            fprintf(stderr, RED "sem_close failed: %s\n", strerror(errno));
            rv = -errno;
        }
    }

    // I'd think osem_delete would be like sem_unlink
    // and osem_close would be like sem_close
    // But osem_delete takes a handle wihle sem_unlink uses a name

    return rv;
}

static greg_t handle_osem_open(mcontext_t *mcontext) {
    // Guessing that param_2 to _ps4__sceKernelOpenSema is the semaphore name, and that it
    // must already exist (since count and other init params aren't given)

    // _ps4__sceKernelOpenSema
    // param_1: return param
    // param_2: ??? arg to syscall

    // syscall: param_2
    const char *name = ARG1<const char *>(mcontext);

    return -EINVAL;
}

static greg_t handle_osem_close(mcontext_t *mcontext) {
    return -EINVAL;
}

static greg_t handle_osem_wait(mcontext_t *mcontext) {
    return -EINVAL;
}

static greg_t handle_osem_trywait(mcontext_t *mcontext) {
    return -EINVAL;
}

static greg_t handle_osem_post(mcontext_t *mcontext) {
    // "post" seems mean the same as "release"
    return -EINVAL;
}

static greg_t handle_osem_cancel(mcontext_t *mcontext) {
    return -EINVAL;
}

// Dunno what this is
struct authinfo_t {
    uint64_t utype;
    uint64_t flags;
};

static greg_t handle_get_authinfo(mcontext_t *mcontext) {
    auto pid = ARG1<pid_t>(mcontext);
    auto *authinfo = ARG2<authinfo_t *>(mcontext);

    /*
    uint64_t to_write[] = {
        0x3800000000000006,
        0x380000000000000f,
        0x3800000000000010,
        0x3800000000000015,
        0x3800000000000016,
        0x3800000000000017,
        0x3800000000000018,
        0x3800000000000033,
        0x3800000000000034,
        0x3800000000000035,
        0x3800000000000036,
        0x3800000000010003,
        0x3800000010000001,
        0x3800000010000002,
        0x3800000010000003,
        0x3800000010000004,
        0x3800000010000005,
        0x3800000010000009,
        0x380000001000000f,
    };
    */

    //authinfo->utype = 0x3800000010000009;
    authinfo->utype = 0;
    authinfo->flags = 0;

    return 0;
    //return -EINVAL;
}

static greg_t handle_namedobj_create(mcontext_t *mcontext) {
    auto *name = ARG1<const char *>(mcontext);
    auto *object = ARG2<void *>(mcontext);
    // arg3 is some kind of flags
    auto arg3 = ARG3<uint64_t>(mcontext);

    // libc init: arg2 is address of calloc'd sce_pthread_mutex
    greg_t rv = 0;
    // 0xffffffff means errror
    return rv;
}


static std::set<OrbisSyscallNr> green_syscalls = {
    //OrbisSyscallNr::ORB_SYS_write,

};

static void logSyscall(OrbisSyscallNr ps4_syscall_nr) {
    std::string bsdName = to_string((OrbisSyscallNr) ps4_syscall_nr);

    switch (ps4_syscall_nr) {
        case OrbisSyscallNr::ORB_SYS_read:
        case OrbisSyscallNr::ORB_SYS_open:
        case OrbisSyscallNr::ORB_SYS_close:
        case OrbisSyscallNr::ORB_SYS_getpid:
        case OrbisSyscallNr::ORB_SYS_ioctl:
        case OrbisSyscallNr::ORB_SYS_rtprio_thread:
        case OrbisSyscallNr::ORB_SYS_thr_self:
        case OrbisSyscallNr::ORB_SYS_mmap:
        case OrbisSyscallNr::ORB_SYS_clock_gettime:
        {
            // green
            printf(GRN "freebsd_syscall_handler: handling %s%s\n", bsdName.c_str(), RESET);            
            break;
        }
        case OrbisSyscallNr::ORB_SYS_mname:
	    case OrbisSyscallNr::ORB_SYS_osem_create:
	    case OrbisSyscallNr::ORB_SYS_osem_delete:
        case OrbisSyscallNr::ORB_SYS___sysctl:
        case OrbisSyscallNr::ORB_SYS_dynlib_get_proc_param:
        case OrbisSyscallNr::ORB_SYS_namedobj_create:
        {
            printf(YEL "freebsd_syscall_handler: handling %s%s\n", bsdName.c_str(), RESET);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_netcontrol:
        case OrbisSyscallNr::ORB_SYS_evf_create:
	    case OrbisSyscallNr::ORB_SYS_dmem_container:
	    case OrbisSyscallNr::ORB_SYS_get_authinfo:
	    case OrbisSyscallNr::ORB_SYS_mdbg_service:
        case OrbisSyscallNr::ORB_SYS_randomized_path:
	    case OrbisSyscallNr::ORB_SYS_budget_get_ptype:
	    case OrbisSyscallNr::ORB_SYS_get_proc_type_info:
        case OrbisSyscallNr::ORB_SYS_regmgr_call:
        case OrbisSyscallNr::ORB_SYS_osem_open:
        case OrbisSyscallNr::ORB_SYS_osem_close:
        case OrbisSyscallNr::ORB_SYS_osem_wait:
        case OrbisSyscallNr::ORB_SYS_osem_trywait:
        case OrbisSyscallNr::ORB_SYS_osem_post:
        case OrbisSyscallNr::ORB_SYS_osem_cancel:
        case OrbisSyscallNr::ORB_SYS_sysarch:
        {
            // red
            fprintf(stderr, RED "freebsd_syscall_handler: handling %s\n" RESET, bsdName.c_str());
            break;
        }

        default:
            break;
    }
}

extern "C" {

void ps4_syscall_handler(int num, siginfo_t *info, void *ucontext_arg) {
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

    logSyscall(OrbisSyscallNr(ps4_syscall_nr));

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

        case OrbisSyscallNr::ORB_SYS_open:
        {
            rv = handle_open(mcontext);
            break;
        }

        case OrbisSyscallNr::ORB_SYS_close:
        {
            rv = handle_close(mcontext);
            break;
        }

        case OrbisSyscallNr::ORB_SYS_ioctl:
        {
            rv = handle_ioctl(mcontext);
            break;
        }

        case OrbisSyscallNr::ORB_SYS_netbsd_msync:
        case 277:
        {
            bsdName = to_string(OrbisSyscallNr::ORB_SYS_netbsd_msync);
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

        case OrbisSyscallNr::ORB_SYS_shmctl:
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
            bsdName = to_string((OrbisSyscallNr) OrbisSyscallNr::ORB_SYS_shmctl);
            rv = -EINVAL;
            break;
        }

        case OrbisSyscallNr::ORB_SYS___sysctl:
        {
            rv = handle_sysctl(mcontext);
            break;
        }

        case OrbisSyscallNr::ORB_SYS_lkmnosys:
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
            bsdName = to_string(OrbisSyscallNr::ORB_SYS_lkmnosys);
            // OrbisSyscallNr::ORB_SYS_lkmnosys
            rv = -EINVAL;
            break;
        }

        case OrbisSyscallNr::ORB_SYS_netbsd_lchown:
        case 275:
        {
            bsdName = to_string(OrbisSyscallNr::ORB_SYS_netbsd_lchown);
            rv = -EINVAL;
            break;
        }

        case OrbisSyscallNr::ORB_SYS_rtprio_thread:
        {
            rv = handle_rtprio_thread(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_thr_self:
        {
            rv = gettid();
            break;
        }
        case OrbisSyscallNr::ORB_SYS_mmap:
        {
            rv = handle_mmap(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_create:
        {
            rv = handle_osem_create(mcontext);            
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_delete:
        {
            rv = handle_osem_delete(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_open:
        {
            rv = handle_osem_open(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_close:
        {
            rv = handle_osem_close(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_wait:
        {
            rv = handle_osem_wait(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_trywait:
        {
            rv = handle_osem_trywait(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_post:
        {
            rv = handle_osem_post(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_osem_cancel:
        {
            rv = handle_osem_cancel(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_namedobj_create:
        {
            rv = handle_namedobj_create(mcontext);
            break;
        }      
        case OrbisSyscallNr::ORB_SYS_dmem_container:
            rv = -EINVAL;
            break;
        case OrbisSyscallNr::ORB_SYS_get_authinfo:
            // get_authinfo
            // ScePthread: Fatal error 'Can't allocate initial thread' (errno = 22)            
            rv = handle_get_authinfo(mcontext);
            break;
        case OrbisSyscallNr::ORB_SYS_mname:
        {
            rv = handle_mname(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_dynlib_get_proc_param:
        {
            rv = handle_dynlib_get_proc_param(mcontext);
            break;
        }
        case OrbisSyscallNr::ORB_SYS_mdbg_service:
            // mdbg_service
            rv = -EINVAL;
            break;
        case OrbisSyscallNr::ORB_SYS_randomized_path:
            // ... syscall handler
            // libkernel : _ps4____OrbisSyscallNr::ORB_SYS_randomized_path
            // libkernel : _ps4__sceKernelGetFsSandboxRandomWord
            // libSceDiscMap init
            //
            // prints "[DiskMap BitmapInfo] get path failed"
            rv = -EINVAL;
            break;
        case OrbisSyscallNr::ORB_SYS_budget_get_ptype:
            rv = -EINVAL;
            break;
        case OrbisSyscallNr::ORB_SYS_get_proc_type_info:
            // nG-FYqFutUo
            rv = -EINVAL;
            break;

        default:
        {
            std::string ps4Name = to_string((OrbisSyscallNr) ps4_syscall_nr);
            fprintf(stderr, RED "Unhandled syscall : %lli (%s)\n" RESET, ps4_syscall_nr, ps4Name.c_str());
            rv = -EINVAL;
            break;
        }
    }

    // before this point, rv should be -errno if there was an error (linux convention)
    // change to bsd conventions here (rax holds result or positive errno, carry flag==1 means error)
    if (rv < 0) {
        // TODO? Convert linux errno to SCE_KERNEL_* errno?
        //rv = -rv;
        rv = errnoToSceError(-rv);
        // Set carry flag. ps4/bsd code expects carry flag to be 1 on errors
        mcontext->gregs[REG_EFL] |= 1;
    } else {
        // Clear carry flag (no error)
        mcontext->gregs[REG_EFL] &= (~1llu);
    }
    mcontext->gregs[REG_RAX] = rv;
}

}
