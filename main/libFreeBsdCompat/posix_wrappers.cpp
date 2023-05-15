#include <cstring>
#include <stdio.h>
#include <dlfcn.h>
#include <filesystem>
#include <csignal>
namespace fs = std::filesystem;
#include "../chroot.h"
#include <cstdarg>

#define CHROOT_WRAPPER(fn, filearg) \
    fs::path modded_path; \
    typedef decltype(fn)* PFN_##fn; \
    static PFN_##fn impl = nullptr; \
    if (!impl) { \
        impl = (PFN_##fn) dlsym(RTLD_NEXT, #fn); \
    } \
    if (filearg[0] == '/' && is_chrooted()) { \
        modded_path = get_chroot_path(); \
        modded_path += filearg; \
        filearg = modded_path.c_str(); \
    }

extern "C" {

FILE *fopen( const char * filename, const char * mode) {
    CHROOT_WRAPPER(fopen, filename)
    return impl(filename, mode);
}

FILE *fopen64(const char *__restrict filename, const char *__restrict modes) {
    CHROOT_WRAPPER(fopen64, filename)
    return impl(filename, modes);
}

int open(const char *pathname, int flags, mode_t mode) {
    CHROOT_WRAPPER(open, pathname)
    int res = impl(pathname, flags, mode);
    if (res < 0) {
        fprintf(stderr, "open failed for %s: %s\n", pathname, strerror(errno));
    }
    return res;
}

int creat(const char *pathname, mode_t mode) {
    CHROOT_WRAPPER(creat, pathname)
    return impl(pathname, mode);
}

int openat(int dirfd, const char *pathname, int flags, mode_t mode) {
    CHROOT_WRAPPER(openat, pathname)
    return impl(dirfd, pathname, flags, mode);
}

int openat2(int dirfd, const char *pathname, const struct open_how *how, size_t size) {
    CHROOT_WRAPPER(openat2, pathname)
    return impl(dirfd, pathname, how, size);
}

int fprintf (FILE *__restrict __stream, const char *__restrict __format, ...) {
    if ((uint64_t) __stream == 0x00007ffff7108f50
        || (uint64_t) __stream == 0x00007ffff7104f50) {
        __stream = stderr;
    }

    va_list ap;

    va_start(ap, __format);
    int rv = vfprintf(__stream, __format, ap);
    va_end(ap);

    return rv;
}

}