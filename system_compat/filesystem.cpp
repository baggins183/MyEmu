#include <cstring>
#include <stdio.h>
#include <dlfcn.h>
#include <filesystem>
#include <csignal>
namespace fs = std::filesystem;
#include "system_compat/ps4_region.h"
#include <cstdarg>
#include "Common.h"
#include <fcntl.h>

static fs::path modPathForChroot(const char *path) {
    if (path[0] == '/') {
        fs::path moddedPath = get_chroot_path();
        moddedPath += path;
        return moddedPath;
    } else {
        return path;
    }
}

int open_wrapper(const char *pathname, int flags, mode_t mode) {
    fs::path moddedPath = modPathForChroot(pathname);
    return open(moddedPath.c_str(), flags, mode);
}


extern "C" {

FILE *PS4FUN(fopen) ( const char * filename, const char * mode) {
    printf(CYN "fopen: %s\n" RESET, filename);
    fs::path moddedPath = modPathForChroot(filename);
    return fopen(moddedPath.c_str(), mode);
}

FILE *PS4FUN(fopen64)(const char *__restrict filename, const char *__restrict modes) {
    printf(CYN "fopen64: %s\n" RESET, filename);
    fs::path moddedPath = modPathForChroot(filename);
    return fopen64(moddedPath.c_str(), modes);
}

int PS4FUN(open)(const char *pathname, int flags, mode_t mode) {
    printf(CYN "open: %s\n" RESET, pathname);
    return open_wrapper(pathname, flags, mode);
}

int PS4FUN(creat)(const char *pathname, mode_t mode) {
    printf(CYN "creat: %s\n" RESET, pathname);
    fs::path moddedPath = modPathForChroot(pathname);
    return creat(moddedPath.c_str(), mode);
}

int PS4FUN(openat)(int dirfd, const char *pathname, int flags, mode_t mode) {
    printf(CYN "openat: %s\n" RESET, pathname);
    fs::path moddedPath = modPathForChroot(pathname);
    return openat(dirfd, moddedPath.c_str(), flags, mode);
}

//int PS4FUN(openat2)(int dirfd, const char *pathname, const struct open_how *how, size_t size) {
//    fs::path moddedPath = modPathForChroot(pathname);
//    return openat2(dirfd, moddedPath.c_str(), how, size);
//}

} // extern "C"