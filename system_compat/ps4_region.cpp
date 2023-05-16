#include <atomic>
#include <sys/mman.h>
#include <vector>
#include <algorithm>
#include <string>
#include <filesystem>
#include <sys/prctl.h>
#include <signal.h>
#include "Common.h"

extern "C" {
#include "pmparser.h"
}

#include "syscall_handler.h"

namespace fs = std::filesystem;

static thread_local char _syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_ALLOW;
static thread_local bool _in_ps4_region = false;

void enter_ps4_region() {
    _syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_BLOCK;
    _in_ps4_region = true;
}

void leave_ps4_region() {
    _in_ps4_region = false;
    _syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_ALLOW;
}

bool in_ps4_region() {
    return _in_ps4_region;
}

struct MappedRange {
    uint64_t base_addr;
    size_t n_pages;
};

// Syscalls from other memory regions will be handled by a translation layer
// This is to translate FreeBSD syscalls to linux, and handle differences in how the ps4 libs
// interpret syscall errors
static bool setupSyscallTrampoline(uint64_t &addr_start, uint64_t &addr_end) {
    // parse /proc/self/map to find the ranges to give to the SYSCALL_USER_DISPATCH prctl call
    int err;
    
    // Also record all the presently mapped pages
    // We will also mmap the gaps so future libraries (ps4 elfs) can't occupy them
    // We need to reserve a range that includes libc.so and the syscall trampoline code,
    // where syscalls can execute without being blocked by the user dispatch filter
    std::vector<MappedRange> occupiedRanges;

    procmaps_struct *pm;
    // errno is 2 on entry here, should check why
    procmaps_iterator *it = pmparser_parse(-1);

    addr_start = 0;
    addr_end = 0;
    bool found = false;
    while ((pm = pmparser_next(it))) {
        fs::path soPath(pm->pathname);
        if (soPath.has_filename()) {
            if ((soPath.has_stem() && soPath.stem() == "libc.so") 
                    ||  soPath.filename() == "libsystem_compat.so")
            {
                if ( !found) {
                    addr_start = (uint64_t) pm->addr_start;
                    addr_end = (uint64_t) pm->addr_end;
                } else {
                    addr_start = std::min(addr_start, (uint64_t) pm->addr_start);
                    addr_end = std::max(addr_end, (uint64_t) pm->addr_end);
                }
                found = true;
            }
        }

        assert(( (uint64_t)pm->addr_end - (uint64_t)pm->addr_start ) % PGSZ == 0);
        occupiedRanges.push_back({ (uint64_t) pm->addr_start, ((uint64_t)pm->addr_end - (uint64_t)pm->addr_start) / PGSZ});
    }

    if ( !found) {
        return false;
    }

    std::sort(occupiedRanges.begin(), occupiedRanges.end(), [](const auto &a, const auto &b) {
        return a.base_addr < b.base_addr;
    });

    uint pos = 0;
    for (; pos < occupiedRanges.size(); pos++) {
        if (occupiedRanges[pos].base_addr == addr_start) {
            break;
        }
    }
    assert(pos < occupiedRanges.size());

    // mmap the gaps between libc and our trampoline so ps4 libs can't be loaded there
    while (pos < occupiedRanges.size() - 1) {
        MappedRange &first = occupiedRanges[pos];
        MappedRange &snd = occupiedRanges[pos + 1];
        
        size_t page_gap = (snd.base_addr - first.base_addr) / PGSZ - first.n_pages;
        if (page_gap > 0) {
            uint64_t map_addr = first.base_addr + PGSZ * first.n_pages;
            uint64_t map_size = PGSZ * page_gap;
            assert(map_addr % PGSZ == 0);
            void * mmap_result;
            mmap_result = mmap((void *) map_addr, map_size, PROT_NONE, MAP_FIXED_NOREPLACE | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mmap_result == MAP_FAILED) {
                switch (errno) {
                    case EEXIST:
                        break;
                    default:
                        fprintf(stderr, "mmap failed: %s", strerror(errno));
                        return false;
                }
            }
        }
        // We've filled between libc and the trampoline, break
        if (snd.base_addr + PGSZ * snd.n_pages == addr_end) {
            break;   
        }

        ++pos;
    }

	struct sigaction action;
    (void)memset(&action, 0, sizeof action);
	action.sa_sigaction = freebsd_syscall_handler;
	(void)sigemptyset(&action.sa_mask);
	(void)sigfillset(&action.sa_mask);
	(void)sigdelset(&action.sa_mask, SIGSYS);
	action.sa_flags = SA_SIGINFO;

	err = sigaction(SIGSYS, &action, NULL);
    if (err) {
        printf("sigaction error: %s", strerror(errno));
        return false;
    }

    return true;
}

bool thread_init_syscall_user_dispatch() {
    static bool _has_set_sigsys_handler = false;
    static uint64_t _addr_start = 0;
    static uint64_t _addr_end = 0;

    if (!_has_set_sigsys_handler) {
        if (!setupSyscallTrampoline(_addr_start, _addr_end)) {
            return false;
        }
        _has_set_sigsys_handler = true;
    }

    int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, _addr_start, (_addr_end - _addr_start), &_syscall_dispatch_switch);
    leave_ps4_region();

    if (err) {
        printf("prctl error: %s", strerror(errno));
    }

    return !err;
}

static std::string chroot_path;

void set_chroot_path(const char *path) {
    chroot_path = path;
}

std::string get_chroot_path() {
    return chroot_path;
}