#define LOGGER_IMPL
#include "Common.h"
#include "Elf/elf-sce.h"
#include "nid_hash/nid_hash.h"

#include <elf.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <system_error>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <pthread.h>
#include <map>
#include <utility>
#include <array>
#include <limits.h>
#include <atomic>

#include <filesystem>
namespace fs = std::filesystem;
#include <algorithm>
#include <dlfcn.h>
#include <link.h>
#include <libgen.h>
#include <sqlite3.h>
#include <set>
#include <optional>

#include "elfpatcher/elfpatcher.h"

#include <signal.h>
#include <sys/prctl.h>
// parsing the /proc/<pid>/map files
extern "C" {
#include <pmparser.h>
}

const std::string ebootPath = "eboot.bin";
char syscall_dispatch_switch;

struct {
    std::string currentPs4Lib;
} TheDebugContext;

struct CmdConfig {
    //std::string pkgDumpPath;
    std::string dlPath;
    std::string hashdbPath;
    std::string nativeElfOutputDir;
    std::string pkgdumpPath;
    std::string ps4libsPath;
    std::string preloadDirPath;
    bool purgeElfs;

    CmdConfig(): nativeElfOutputDir("./elfdump/"), purgeElfs(false) {}
};

CmdConfig CmdArgs;

bool parseCmdArgs(int argc, char **argv) {
    if (argc < 2) {
        //fprintf(stderr, "usage: %s [options] <PATH TO PKG DUMP>\n", argv[0]);
        fprintf(stderr, "usage: %s [options] <PATH TO ELF>\n", argv[0]);
        exit(1);
    }

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--hashdb")) {
            CmdArgs.hashdbPath = argv[++i];
        } else if (!strcmp(argv[i], "--native_elf_output")) {
            CmdArgs.nativeElfOutputDir = argv[++i];
        } else if(!strcmp(argv[i], "--pkgdump")) {
            CmdArgs.pkgdumpPath = argv[++i];
        } else if(!strcmp(argv[i], "--ps4libs")) {
            CmdArgs.ps4libsPath = argv[++i];
        } else if (!strcmp(argv[i], "--preload_dir")) {
            CmdArgs.preloadDirPath = argv[++i];
        } else if (!strcmp(argv[i], "--purge_elfs")) {
            CmdArgs.purgeElfs = true;
        } else {
            fprintf(stderr, "Unrecognized cmd arg: %s\n", argv[i]);
            return false;
        }
    }

    if (CmdArgs.hashdbPath.empty()) {
        fprintf(stderr, "Warning: no symbol hash database given\n");
        exit(1); // For debugging
    }

    //CmdArgs.pkgDumpPath = argv[argc - 1];
    CmdArgs.dlPath = argv[argc - 1];
    return true;
}

struct MappedRange {
    uint64_t baseVA;
    uint64_t nPages;
};

bool mapEntryModuleIntoMemory(fs::path nativeExecutablePath) {
    FILE *elf = fopen(nativeExecutablePath.c_str(), "r+");
    if ( !elf) {
        return false;
    }

    return true;
}

typedef int (*PFN_PS4_INIT_FUNC) (int, const char **, char **);

struct Ps4InitFuncArgs {
    int argc;
    char **argv;
    char **environ;
};

static bool callInitFunctions(std::string ps4Name, InitFiniInfo &initFiniInfo) {
    bool success = true;
    const char *argv[] = {
        // TODO use .so name? Or entry name?
        "eboot.bin"
    };
    int argc = 1;

    std::string nativeName = getNativeLibName(fs::path(ps4Name).filename());

    void *dl = dlopen(nativeName.c_str(), RTLD_LAZY);
    if ( !dl) {
        success = false;
        goto error;
    }
    struct link_map *l;
    if ( dlinfo(dl, RTLD_DI_LINKMAP, &l)) {
        success = false;
        goto error;
    }
    dlclose(dl);
    dl = nullptr;
    // decrements ref count, but should stay loaded

    syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_BLOCK;
    std::atomic_thread_fence(std::memory_order_seq_cst);

    if ( !initFiniInfo.dt_preinit_array.empty()) {
        for (uint i = 0; i < initFiniInfo.dt_preinit_array.size(); i++) {
            ((PFN_PS4_INIT_FUNC) (l->l_addr + initFiniInfo.dt_preinit_array[i])) (argc, argv, NULL);
        }
    }

    if ( initFiniInfo.dt_init) {
        ((PFN_PS4_INIT_FUNC) (l->l_addr + initFiniInfo.dt_init.value())) (argc, argv, NULL);
    }

    if ( !initFiniInfo.dt_init_array.empty()) {
        for (uint i = 0; i < initFiniInfo.dt_init_array.size(); i++) {
            ((PFN_PS4_INIT_FUNC) (l->l_addr + initFiniInfo.dt_init_array[i])) (argc, argv, NULL);
        }
    }

    syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_ALLOW;
    std::atomic_thread_fence(std::memory_order_seq_cst);   

error:
    char *err;
    while ((err = dlerror())) {
        printf("doInit: %s\n", err);
    }

    if (dl) {
        dlclose(dl);
    }
    return success;
}

void oob_syscall_handler(int signum) {
    int a = 5;
    int b = a * 10 + 6;
    printf("in handler\n");
}

// Restrict syscalls to libc.so.x
// Syscalls from other memory regions will be handled by a translation layer
// This is to translate FreeBSD syscalls to linux, and handle differences in how the ps4 libs
// interpret syscall errors
static bool setupSyscallTrampoline() {
    // parse /proc/self/map to get the start and end address of libc.so.x
    procmaps_struct *pm;
    // errno is 2 on entry here, should check why
    procmaps_iterator *it = pmparser_parse(-1);

    uint64_t addr_start = 0;
    uint64_t addr_end = 0;
    bool found = false;
    while ((pm = pmparser_next(it))) {
        fs::path soPath(pm->pathname);
        if ((soPath.has_stem() && soPath.stem() == "libc.so")
//                || (soPath.has_stem() && soPath.filename().stem() == "ld-linux-x86-64.so)"
        ) {
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

    if ( !found) {
        return false;
    }

    // TODO register exception handler
    //signal(SIGSYS, oob_syscall_handler);
    syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_BLOCK;
    int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, addr_start, (addr_end - addr_start + 1), &syscall_dispatch_switch);
    //int err = 0;

    if (err) {
        printf("prctl error: %s", strerror(errno));
        return false;
    }

    return true;
}

int main(int argc, char **argv) {
    // Make stdout unbuffered so crashes don't hide writes when stdout is redirected to file
    setvbuf(stdout, NULL, _IONBF, 0);    
    setvbuf(stdout, NULL, _IONBF, 0);    

    //FILE *eboot;
    void *dl;

    parseCmdArgs(argc, argv);
    ElfPatcherContext Ctx(CmdArgs.ps4libsPath,CmdArgs.preloadDirPath, CmdArgs.hashdbPath, CmdArgs.nativeElfOutputDir, CmdArgs.pkgdumpPath, CmdArgs.purgeElfs);

    if ( !getenv("LD_LIBRARY_PATH")) {
        fprintf(stderr, "LD_LIBRARY_PATH unset\n");
        return -1;
    }

    std::error_code ec;
    fs::create_directory(CmdArgs.nativeElfOutputDir, ".", ec);
    if (ec) {
        std::string m = ec.message();
        fprintf(stderr, "Failed to create directory %s: %s\n", CmdArgs.nativeElfOutputDir.c_str(), m.c_str());
        return -1;
    }    

    std::vector<std::string> worklist = { CmdArgs.dlPath };
    std::set<std::string> inWorklist;
    inWorklist.insert(getNativeLibName(worklist[0]));

    std::map<std::string, std::set<std::string>> depGraph;
    std::map<std::string, InitFiniInfo> initFiniInfos;

    uint i = 0; 
    while (i < worklist.size()) {
        fs::path libName = worklist[i];
        ++i;
        auto oLibPath = findPathToSceLib(libName, Ctx);
        if ( !oLibPath) {
            fprintf(stderr, "Warning: unable to locate library %s\n", libName.c_str());
            continue;
        }

        fs::path libPath = oLibPath.value();

        fs::path nativePath = CmdArgs.nativeElfOutputDir; 
        nativePath /= getNativeLibName(libPath);

        struct stat buf;
        bool exists = !stat(nativePath.c_str(), &buf);
        if (CmdArgs.purgeElfs || !exists) {
            std::string cmd = "cp " + libPath.string() + " " + nativePath.string();
            system(cmd.c_str());

            if (chmod(nativePath.c_str(), S_IRWXU | (S_IRGRP | S_IXGRP) | (S_IROTH | S_IXOTH))) {
                fprintf(stderr, "chmod failed\n");
                return -1;
            }

            if ( !patchPs4Lib(Ctx, nativePath)) {
                return -1;
            }
        } else {
            // TODO read DT_NEEDED ents to build topological order of deps
            // So we can call DT_INIT-like functions in correct order
        }
        std::string nativeName = getNativeLibName(libName.filename());

        initFiniInfos[nativeName] = Ctx.init_fini_info;
        
        for (auto &dep: Ctx.deps) {
            depGraph[nativeName].insert(getNativeLibName(dep));
        }

        for (auto &dep: Ctx.deps) {
            if (inWorklist.find(getNativeLibName(dep)) == inWorklist.end()) {
                worklist.push_back(dep);
                inWorklist.insert(getNativeLibName(dep));
            }
        }
        Ctx.reset();
    }

    fs::path firstLib = CmdArgs.nativeElfOutputDir;
    firstLib /= getNativeLibName(CmdArgs.dlPath);

    fs::path entryModule = CmdArgs.nativeElfOutputDir;
    entryModule /= getNativeLibName("eboot.bin");
    //if ( !mapEntryModuleIntoMemory(entryModule)) {
    //    return -1;
    //}

    std::vector<void *> preloadHandles;
    LibSearcher preloadSearcher({{CmdArgs.preloadDirPath, true}});

    printf("A\n");

    setupSyscallTrampoline();
    //syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_BLOCK;
    // Leave syscall filter off for now because it seems to mess with dlopen, which must be making syscalls directly
    syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_ALLOW;
    // Don't think fence is necessary, but keep for now
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // This below errors when signal handler not provided (good).
    // Need to now place trampoline code in region with libc
    //asm volatile ("movq $39, %%rax; syscall" ::: "rax");
    //asm volatile ("movq $39, %%rax; syscall" ::: "rax", "memory");

    // These may depend on ps4 libs (for wrapping sce functions by doing1 dlsym(RTLD_NEXT, ...))
    // So preload these only after patching dependencies
    // Note RTLD_GLOBAL: these symbols will take precedence over ps4 symbols in lookups/relocs
    for (std::string preload: Ctx.preloadNames) {
        auto preloadPath = preloadSearcher.findLibrary(preload);
        assert(preloadPath);

        void *handle = dlopen(preloadPath->c_str(), RTLD_LAZY | RTLD_GLOBAL);
        if (!handle) {
            char *err;
            while ((err = dlerror())) {
                fprintf(stderr, "(dlopen) %s\n", err);
            }
            return -1;
        }        
        preloadHandles.push_back(handle);
    }

    printf("B\n");

    // Load ps4 module
    dl = dlopen(firstLib.c_str(), RTLD_LAZY);
    if (!dl) {
        char *err;
        while ((err = dlerror())) {
            printf("(dlopen) %s\n", err);
        }
        return -1;
    }

    // TODO need to actually topological sort the dependencies

    // Do initialization for all libs, but not the entry module yet
    // Do reverse order of deps
    // TODO does preinit go in forward order?
    //for (uint i = worklist.size() - 1; i > 0; i--) {
    for (uint i = 1; i >= 0; i--) {
        fs::path nativeLibName = getNativeLibName(worklist[i]);
        callInitFunctions(worklist[i], initFiniInfos[nativeLibName]);
    }

    for (auto handle: preloadHandles) {
        dlclose(handle);
    }

    dlclose(dl);
    printf("main: done\n");

    return 0;
}