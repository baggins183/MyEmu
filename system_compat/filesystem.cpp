#include <cstring>
#include <stdio.h>
#include <dlfcn.h>
#include <filesystem>
#include <csignal>
namespace fs = std::filesystem;
#include "system_compat/ps4_region.h"
#include <cstdarg>

#include "wrappers.h"

static fs::path modPathForChroot(const char *path) {
    if (path[0] == '/') {
        fs::path moddedPath = get_chroot_path();
        moddedPath += path;
        return moddedPath;
    } else {
        return path;
    }
}

extern "C" {

FILE *fopen( const char * filename, const char * mode) {
    SYSTEM_LIB_WRAPPER(fopen, filename, mode)

    fs::path moddedPath = modPathForChroot(filename);
    auto rv = fopen__impl(moddedPath.c_str(), mode);

    END_SYSTEM_LIB_WRAPPER
    return rv;
}

FILE *fopen64(const char *__restrict filename, const char *__restrict modes) {
    SYSTEM_LIB_WRAPPER(fopen64, filename, modes)

    fs::path moddedPath = modPathForChroot(filename);
    auto rv = fopen64__impl(moddedPath.c_str(), modes);

    END_SYSTEM_LIB_WRAPPER
    return rv;
}

int open(const char *pathname, int flags, mode_t mode) {
    SYSTEM_LIB_WRAPPER(open, pathname, flags, mode)

    fs::path moddedPath = modPathForChroot(pathname);
    auto rv = open__impl(moddedPath.c_str(), flags, mode);

    END_SYSTEM_LIB_WRAPPER
    return rv;
}

int creat(const char *pathname, mode_t mode) {
    SYSTEM_LIB_WRAPPER(creat, pathname, mode)

    fs::path moddedPath = modPathForChroot(pathname);
    auto rv = creat__impl(moddedPath.c_str(), mode);

    END_SYSTEM_LIB_WRAPPER
    return rv;
}

int openat(int dirfd, const char *pathname, int flags, mode_t mode) {
    SYSTEM_LIB_WRAPPER(openat, dirfd, pathname, flags, mode)

    fs::path moddedPath = modPathForChroot(pathname);
    auto rv = openat__impl(dirfd, moddedPath.c_str(), flags, mode);

    END_SYSTEM_LIB_WRAPPER
    return rv;
}

int openat2(int dirfd, const char *pathname, const struct open_how *how, size_t size) {
    SYSTEM_LIB_WRAPPER(openat2, dirfd, pathname, how, size)

    fs::path moddedPath = modPathForChroot(pathname);
    auto rv = openat2__impl(dirfd, moddedPath.c_str(), how, size);

    END_SYSTEM_LIB_WRAPPER
    return rv;
}

} // extern "C"