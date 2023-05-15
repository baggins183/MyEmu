#include <cstdio>
#include <dlfcn.h>
#include <sys/mman.h>

extern "C" {

#define	BSD_MAP_SHARED	0x0001		/* share changes */
#define	BSD_MAP_PRIVATE	0x0002		/* changes are private */
#define	BSD_MAP_COPY	BSD_MAP_PRIVATE	/* Obsolete */
#define	BSD_MAP_FIXED	 0x0010	/* map addr must be exactly as requested */
#define	BSD_MAP_RENAME	 0x0020	/* Sun: rename private pages to file */
#define	BSD_MAP_NORESERVE	 0x0040	/* Sun: don't reserve needed swap area */
#define	BSD_MAP_RESERVED0080 0x0080	/* previously misimplemented BSD_MAP_INHERIT */
#define	BSD_MAP_RESERVED0100 0x0100	/* previously unimplemented BSD_MAP_NOEXTEND */
#define	BSD_MAP_HASSEMAPHORE 0x0200	/* region may contain semaphores */
#define	BSD_MAP_STACK	 0x0400	/* region grows down, like a stack */
#define	BSD_MAP_NOSYNC	 0x0800 /* page to but do not sync underlying file */
#define	BSD_MAP_FILE	 0x0000	/* map from file (default) */
#define	BSD_MAP_ANON	 0x1000	/* allocated from memory, swap space */
#define	BSD_MAP_ANONYMOUS	 BSD_MAP_ANON /* For compatibility. */

void *mmap(void *addr, size_t length, int prot, int flags,
                  int fd, off_t offset)
{
    // Convert BSD flags to Linux flags
    typedef decltype(mmap)* PFN_mmap;
    static PFN_mmap impl = nullptr;
    if (!impl) {
        impl = (PFN_mmap) dlsym(RTLD_NEXT, "mmap");
    }

    int linux_flags = 0;

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
    if (flags & BSD_MAP_ANON) {
        linux_flags |= MAP_ANON;
    }
    if (flags & BSD_MAP_ANONYMOUS) {
        linux_flags |= MAP_ANONYMOUS;
    }

    return impl(addr, length, prot, linux_flags, fd, offset);
}

} // extern "C"