#include <iterator>
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
#include <fstream>
#include <algorithm>
#include <dlfcn.h>
#include <link.h>
#include <libgen.h>
#include <sqlite3.h>
#include <set>
#include <optional>
#include <sys/capability.h>
#include <deque>

#include "elfpatcher/elfpatcher.h"
#include "system_compat/ps4_region.h"

namespace fs = std::filesystem;

const std::string ebootPath = "eboot.bin";

struct {
    std::string currentPs4Lib;
} TheDebugContext;

struct CmdConfig {
    std::string entryModule; // TODO remove, this implied pkgdumpPath/eboot.bin
    std::string hashdbPath;
    std::string patchedElfDir;
    std::string pkgdumpPath;
    std::string ps4libsPath;
    std::string preloadDirPath;
    bool genElfs;
    bool onlyPatch;

    CmdConfig(): patchedElfDir("./elfdump/"), genElfs(false), onlyPatch(false) {}
};

CmdConfig CmdArgs;

bool parseCmdArgs(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s [options] <PATH TO ELF>\n", argv[0]);
        exit(1);
    }

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--hashdb")) {
            CmdArgs.hashdbPath = argv[++i];
        } else if (!strcmp(argv[i], "--patched_elf_dir")) {
            CmdArgs.patchedElfDir = argv[++i];
        } else if(!strcmp(argv[i], "--pkgdump")) {
            CmdArgs.pkgdumpPath = argv[++i];
        } else if(!strcmp(argv[i], "--ps4libs")) {
            CmdArgs.ps4libsPath = argv[++i];
        } else if (!strcmp(argv[i], "--preload_dir")) {
            CmdArgs.preloadDirPath = argv[++i];
        } else if (!strcmp(argv[i], "--gen_elfs")) {
            CmdArgs.genElfs = true;
        } else if (!strcmp(argv[i], "--only_patch")) {
            CmdArgs.onlyPatch = true;
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
    CmdArgs.entryModule = argv[argc - 1];
    return true;
}

bool mapEntryModuleIntoMemory(fs::path nativeExecutablePath) {
    FILE *elf = fopen(nativeExecutablePath.c_str(), "r+");
    if ( !elf) {
        fprintf(stderr, "Couldn't open entry module: error %s\n", strerror(errno));
        return false;
    }

    // TODO

    fclose(elf);
    return true;
}

static bool callInitFunctions(std::string ps4Name, InitFiniInfo &initFiniInfo) {
    bool success = true;
    const char *argv[] = {
        // TODO use .so name? Or entry name?
        "eboot.bin"
    };
    int argc = 1;

    std::string nativeName = getNativeLibName(fs::path(ps4Name).filename());
    fprintf(stderr, "callInitFunctions: %s start\n", nativeName.c_str());

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

    {
        CodeRegionScope __scope(PS4_REGION);

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

    }

error:
    char *err;
    while ((err = dlerror())) {
        printf("doInit: %s\n", err);
    }

    if (dl) {
        dlclose(dl);
    }
    fprintf(stderr, "callInitFunctions: %s done\n", nativeName.c_str());
    return success;
}

static bool findCyclicDependency(const std::string &curLib, const std::string &origLib, std::set<std::string> &transitiveDeps, const std::map<std::string, std::set<std::string>> &dependsOn) {
    if (curLib == origLib) {
        return true;
    }

    if (transitiveDeps.find(curLib) != transitiveDeps.end()) {
        return false;
    }

    transitiveDeps.insert(curLib);

    auto &deps = dependsOn.find(curLib)->second;
    for (auto &dep: deps) {
        if (findCyclicDependency(dep, origLib, transitiveDeps, dependsOn)) {
            return true;
        }
    }
    return false;
}

static void visitToBuildTopoOrder(const std::string &lib, const std::map<std::string, std::set<std::string>> &dependsOnOneWay, std::vector<std::string> &sorted, std::set<std::string> &visited) {
    if (visited.find(lib) != visited.end()) {
        return;
    }
    sorted.push_back(lib);
    visited.insert(lib);
    auto pair = dependsOnOneWay.find(lib);
    for (const auto &dep: pair->second) {
        visitToBuildTopoOrder(dep, dependsOnOneWay, sorted, visited);
    }
}

// The sce libs can have cyclic dependencies, so return an order s.t. libA comes before libB if libA is a transitive dependency of libB but
// libB is not a transitive dependency of libA
static std::vector<std::string> findTopologicalLibOrder(const std::map<std::string, std::set<std::string>> &dependsOn) {
    // Only direct dependencies, not transitive
    std::map<std::string, std::set<std::string>> dependsOnOneWay;

    for (const auto &pair: dependsOn) {
        std::string lib = pair.first;        
        const auto &deps = pair.second;
        std::set<std::string> totalDeps;
        for (auto &dep: deps) {
            std::set<std::string> transitiveDeps;
            if ( !findCyclicDependency(dep, lib, transitiveDeps, dependsOn)) {
                std::copy(transitiveDeps.begin(), transitiveDeps.end(), std::inserter(totalDeps, totalDeps.begin()));
            }
        }
        dependsOnOneWay[lib] = std::move(totalDeps);
    }

    std::vector<std::string> sorted;
    std::set<std::string> visited;
    for (const auto &pair: dependsOn) {
        visitToBuildTopoOrder(pair.first, dependsOnOneWay, sorted, visited);
    }

    std::reverse(sorted.begin(), sorted.end());
    return sorted;
}

static bool create_dir_structure(fs::path &chroot_path) {
    std::error_code ec;
    fs::create_directories(chroot_path, ec);
    if (ec) {
        std::string msg = ec.message();
        fprintf(stderr, "create_filesystem: %s\n", msg.c_str());
        return false;
    }

    static const char *dirs[] = {
        "/dev",
        "/dev/dipsw",
    };

    static const char *files[] {
        // graphics card?
        // libSceGnmDriver.prx.native opens this with O_RDWR
        "/dev/gc"
    };

    for (const char *dir: dirs) {
        fs::path dir_path = chroot_path;
        dir_path += fs::path(dir);
        fs::create_directories(dir_path, ec);
        if (ec) {
            std::string msg = ec.message();
            fprintf(stderr, "create_filesystem: %s\n", msg.c_str());
            return false;
        }
    }

    for (const char *file: files) {
        fs::path file_path = chroot_path;
        file_path += fs::path(file);
        std::ofstream ofs(file_path);
        ofs.close();
    }

    return true;
}

static bool init_filesystem() {
    std::error_code ec;

    const char *home_dir = getenv("HOME");
    if ( !home_dir) {
        return false;
    }
    fs::path chroot_path = home_dir;
    chroot_path /= fs::path(".local/share/MyEmu/chroot");

    if ( !create_dir_structure(chroot_path)) {
        return false;
    }

    // Create symlink to pkg file dump
    // The name should be "app0", at the root of the chrooted ps4 directory structure
    fs::path pkg_dump_mount_link_name = chroot_path;
    pkg_dump_mount_link_name /= "app0";
    fs::path pkg_dump_mount_target = CmdArgs.pkgdumpPath;

    pkg_dump_mount_target = fs::absolute(pkg_dump_mount_target, ec);
    if (ec) {
        std::string msg = ec.message();
        fprintf(stderr, "Couldn't get absolute path to pkg dump directory: %s\n", msg.c_str());
        return false;
    }

    fs::remove(pkg_dump_mount_link_name, ec);
    if (ec) {
        std::string msg = ec.message();
        fprintf(stderr, "Couldn't remove existing symlink to pkg dump: %s\n", msg.c_str());
        return false;        
    }

    fs::create_symlink(pkg_dump_mount_target, pkg_dump_mount_link_name, ec);
    if (ec) {
        std::string msg = ec.message();
        fprintf(stderr, "Couldn't create symlink to pkg: %s\n", msg.c_str());
        return false;
    }

    set_chroot_path(chroot_path.c_str());

    return true;
}

struct Ps4EntryThreadArgs {
    std::map<std::string, InitFiniInfo> initFiniInfos;
    std::vector<std::string> initLibOrder;

    // eboot.bin entry point arguments
    PFUNC_GameTheadEntry game_entry;
    void *game_arg;
    PFUNC_ExitFunction game_exit;
};

void *ps4_entry_thread(void *entry_thread_arg) {
    // setup sysctl values (or in main thread)
    // Call init functions
    //
    Ps4EntryThreadArgs *entryThreadArgs = (Ps4EntryThreadArgs *) entry_thread_arg;

    assert(thread_init_syscall_user_dispatch());

    constexpr std::array initBlackList = {
        "libSceNet.prx.so",
    };

    for (uint i = 0; i < entryThreadArgs->initLibOrder.size(); i++) {
        fs::path nativeLibName = getNativeLibName(entryThreadArgs->initLibOrder[i]);
        if (std::find(begin(initBlackList), end(initBlackList), nativeLibName) != end(initBlackList)) {
            printf("module %s blacklisted\n", nativeLibName.c_str());
            continue;
        }

        callInitFunctions(entryThreadArgs->initLibOrder[i], entryThreadArgs->initFiniInfos[nativeLibName]);
    }

    return NULL;
}

static bool runPs4Thread(Ps4EntryThreadArgs &entryThreadArgs) {
    pthread_t ps4Thread;
    pthread_attr_t attr;

    pthread_attr_init(&attr);

    constexpr size_t stack_size = 2048 * 1024;
    assert(stack_size > (unsigned) __sysconf (__SC_THREAD_STACK_MIN_VALUE));
    void *stack = malloc(stack_size);

    if (!stack_size) {
        return false;
    }

    pthread_attr_setstack(&attr, stack, stack_size);

    pthread_create(&ps4Thread, &attr, ps4_entry_thread, (void *) &entryThreadArgs);
    pthread_join(ps4Thread, NULL);

    return true;
}

int main(int argc, char **argv) {
    //FILE *eboot;
    void *dl;

    parseCmdArgs(argc, argv);
    ElfPatcherContext Ctx(CmdArgs.ps4libsPath,CmdArgs.preloadDirPath, CmdArgs.hashdbPath, CmdArgs.patchedElfDir, CmdArgs.pkgdumpPath, CmdArgs.genElfs);

    if ( !getenv("LD_LIBRARY_PATH")) {
        fprintf(stderr, "LD_LIBRARY_PATH unset\n");
        return -1;
    }

    std::error_code ec;
    fs::create_directory(CmdArgs.patchedElfDir, ".", ec);
    if (ec) {
        std::string m = ec.message();
        fprintf(stderr, "Failed to create directory %s: %s\n", CmdArgs.patchedElfDir.c_str(), m.c_str());
        return -1;
    }    

    std::map<std::string, std::set<std::string>> dependsOn;
    std::map<std::string, InitFiniInfo> initFiniInfos;

    if (CmdArgs.genElfs) {
        // Patch ps4 modules, starting with the entry executable (eboot.bin) and working through
        // the dependencies.
        // Convert these to native ELF files (libs openable with dlopen)
        std::deque<std::string> worklist = { CmdArgs.entryModule };
        std::set<std::string> inWorklist;
        inWorklist.insert(getNativeLibName(worklist[0]));        
        while ( !worklist.empty()) {
            fs::path libName = worklist.front();
            worklist.pop_front();
            auto oLibPath = findPathToSceLib(libName, Ctx);
            if ( !oLibPath) {
                fprintf(stderr, "Warning: unable to locate library %s\n", libName.c_str());
                continue;
            }

            fs::path libPath = oLibPath.value();
            fs::path nativePath = CmdArgs.patchedElfDir; 
            nativePath /= getNativeLibName(libPath);

            if ( !fs::copy_file(libPath, nativePath, fs::copy_options::overwrite_existing)) {
                fprintf(stderr, "Couldn't copy %s to %s\n", libPath.c_str(), nativePath.c_str());
                return -1;
            }

            if (chmod(nativePath.c_str(), S_IRWXU | (S_IRGRP | S_IXGRP) | (S_IROTH | S_IXOTH))) {
                fprintf(stderr, "chmod failed\n");
                return -1;
            }

            if ( !patchPs4Lib(Ctx, nativePath)) {
                return -1;
            }

            std::string nativeName = nativePath.filename();
            initFiniInfos[nativeName] = Ctx.initFiniInfo;
            for (auto &dep: Ctx.deps) {
                dependsOn[nativeName].insert(getNativeLibName(dep));
            }

            for (auto &dep: Ctx.deps) {
                if (inWorklist.find(getNativeLibName(dep)) == inWorklist.end()) {
                    worklist.push_back(dep);
                    inWorklist.insert(getNativeLibName(dep));
                }
            }

            fs::path elfJsonPath = nativePath;
            elfJsonPath += ".json";
            if ( !dumpPatchedElfInfoToJson(elfJsonPath, nativePath, Ctx.initFiniInfo)) {
                return -1;
            }
            Ctx.reset();
        }
    }
    else {
        fs::path prefix = CmdArgs.patchedElfDir;
        std::deque<std::string> worklist = { getNativeLibName(CmdArgs.entryModule) };
        std::set<std::string> inWorklist;
        inWorklist.insert(worklist[0]);         
        while ( !worklist.empty()) {
            std::string nativeName = worklist.front();
            worklist.pop_front();
            fs::path nativePath = prefix;
            nativePath /= nativeName;
            std::vector<std::string> deps;
            findDependencies(nativePath, deps);

            // Get info about the lib (currently the addresses of init/finilization functions)
            fs::path elfJsonPath = nativePath;
            elfJsonPath += ".json";
            InitFiniInfo initFini;
            if ( !parsePatchedElfInfoFromJson(elfJsonPath, &initFini)) {
                return -1;
            }
            initFiniInfos[nativeName] = initFini;

            for (auto &dep: deps) {
                dependsOn[nativeName].insert(getNativeLibName(dep));
            }

            for (auto &dep: deps) {
                if (inWorklist.find(getNativeLibName(dep)) == inWorklist.end()) {
                    worklist.push_back(dep);
                    inWorklist.insert(getNativeLibName(dep));
                }
            }
        }
    }

    if (CmdArgs.onlyPatch) {
        return 0;
    }

    // Setup trampoline to handle syscalls coming from inside sce code
    // Needs to be before dlopen is called on any sce lib, because we need to
    // reserve the range between libc and the lib containing the handler function
    assert(thread_init_syscall_user_dispatch());

    // Preload compatability libs that override sce functions in symbol order
    std::vector<void *> preloadHandles;
    LibSearcher preloadSearcher({{CmdArgs.preloadDirPath, true}});
    for (std::string preload: Ctx.preloadNames) {
        auto preloadPath = preloadSearcher.findLibrary(preload);
        assert(preloadPath);
        fprintf(stderr, "main: preloading %s\n", preloadPath->c_str());
        // Note RTLD_GLOBAL: these symbols will take precedence over ps4 symbols in lookups/relocs
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

    // Load immediate shared library dependencies of the entry module (should be eboot.bin)
    std::vector<std::string> immediateDeps;
    fs::path patchedEntryModule = CmdArgs.patchedElfDir;
    patchedEntryModule /= getNativeLibName(CmdArgs.entryModule);
    if ( !findDependencies(patchedEntryModule, immediateDeps)) {
        return -1;
    }
    for (std::string depName: immediateDeps) {
        fs::path depPath = CmdArgs.patchedElfDir;
        depPath /= depName;
        dl = dlopen(depName.c_str(), RTLD_LAZY);
        if (!dl) {
            char *err;
            while ((err = dlerror())) {
                fprintf(stderr, "(dlopen) %s\n", err);
            }
            return -1;
        }
    }

    // Find initialization order for all sce libraries.
    // Do in order of dependencies (there can be circular deps)
    // TODO does preinit go in forward order?
    std::vector<std::string> topologicalLibOrder;
    topologicalLibOrder = findTopologicalLibOrder(dependsOn);
    assert(topologicalLibOrder.size() == dependsOn.size());
    if ( !init_filesystem()) {
        return 1;
    }

    Ps4EntryThreadArgs args;
    args.initFiniInfos = initFiniInfos;
    args.initLibOrder = topologicalLibOrder;
    runPs4Thread(args);

    // Shutdown
    for (auto handle: preloadHandles) {
        dlclose(handle);
    }

    dlclose(dl);
    printf("main: done\n");

    return 0;
}