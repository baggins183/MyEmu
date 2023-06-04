#include <cstdio>
#include <dlfcn.h>
#include <sys/mman.h>
#include <cstring>
#include <stdio.h>
#include <dlfcn.h>
#include <filesystem>
#include <csignal>
#include "system_compat/ps4_region.h"
#include <cstdarg>
#include <freebsd/bsd_mman.h>

#include "Common.h"

void *mmap_wrapper(void *addr, size_t length, int prot, int flags,
                  int fd, off_t offset) {
    // Note: at some point mmap is called with addr 0x0000000fe0300000,
    // with MAP_ANONYMOUS | 0x2000
    // The linux mmap fails (even skipping the 0x2000)
    // Then prints:
    // "Can't allocate SceGnmGpuInfo memory"

    // Convert BSD flags to Linux flags
    int linux_flags = 0;

    if (flags & 0x2000) {
        fprintf(stderr, "mmap: warning: unhandled flag 0x2000 (unknown)\n");
    }

    if (flags & BSD_MAP_SHARED) {
        linux_flags |= MAP_SHARED;
    }
    if (flags & BSD_MAP_PRIVATE) {
        linux_flags |= MAP_PRIVATE;
    }
    if (flags & BSD_MAP_COPY) {
        fprintf(stderr, "mmap: warning: unhandled flag BSD_MAP_COPY\n");
    }
    if (flags & BSD_MAP_FIXED) {
        linux_flags |= MAP_FIXED;
    }
    if (flags & BSD_MAP_RENAME) {
        fprintf(stderr, "mmap: warning: unhandled flag BSD_MAP_RENAME\n");
    }
    if (flags & BSD_MAP_NORESERVE) {
        linux_flags |= MAP_NORESERVE;
    }
    if (flags & BSD_MAP_RESERVED0080) {
        fprintf(stderr, "mmap: warning: unhandled flag BSD_MAP_RESERVED0080\n");
    }
    if (flags & BSD_MAP_RESERVED0100) {
        fprintf(stderr, "mmap: warning: unhandled flag BSD_MAP_RESERVED0100\n");
    }
    if (flags & BSD_MAP_HASSEMAPHORE) {
        fprintf(stderr, "mmap: warning: unhandled flag BSD_MAP_HASSEMAPHORE\n");
    }
    if (flags & BSD_MAP_STACK) {
        linux_flags |= MAP_STACK;
    }
    if (flags & BSD_MAP_NOSYNC) {
        fprintf(stderr, "mmap: warning: unhandled flag BSD_MAP_NOSYNC\n");
    }
    if (flags & BSD_MAP_FILE) {
        linux_flags |= MAP_FILE;
    }
    // ANON should == ANONYMOUS
    if (flags & BSD_MAP_ANONYMOUS) {
        linux_flags |= MAP_ANONYMOUS;
    }

    if ((linux_flags & (MAP_PRIVATE| MAP_SHARED | MAP_SHARED_VALIDATE)) == 0) {
        linux_flags |= MAP_SHARED; // TODO
    }    

    void *rv = mmap(addr, length, prot, linux_flags, fd, offset);
    if (rv == MAP_FAILED) {
        fprintf(stderr, "(mmap wrapper) warning: map failed. errno = %s\n", strerror(errno));
    }
    return rv;
}

extern "C" {

// I forgot if this is necessary. I think just having the syscall handler is fine
void *PS4FUN(mmap)(void *addr, size_t length, int prot, int flags,
                  int fd, off_t offset)
{
    return mmap_wrapper(addr, length, prot, flags, fd, offset);
}

} // extern "C"