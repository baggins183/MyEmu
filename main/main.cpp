#define LOGGER_IMPL
#include "Common.h"
#include "Elf/elf-sce.h"
#include "nid_hash/nid_hash.h"

#include <elf.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <system_error>
#ifdef __linux
#include <cstdint>
#endif
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

#include <filesystem>
namespace fs = std::filesystem;
#include <algorithm>
#include <dlfcn.h>
#include <libgen.h>
#include <sqlite3.h>
#include <set>
#include <optional>

#include "elfpatcher/elfpatcher.h"

const std::string ebootPath = "eboot.bin";

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

    std::set<std::string> dependencies = { CmdArgs.dlPath };
    std::set<std::string> satisfied;
    while (!dependencies.empty()) {
        auto it = dependencies.begin();
        fs::path libName = *it;
        dependencies.erase(it);

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

            if ( !patchPs4Lib(Ctx, nativePath, dependencies)) {
                return -1;
            }
        }
        satisfied.insert(libName.filename());
        for (auto &sat: satisfied) {
            dependencies.erase(sat);
        }
    }

    fs::path firstLib = CmdArgs.nativeElfOutputDir;
    firstLib /= getNativeLibName(CmdArgs.dlPath);

    fs::path entryModule = CmdArgs.nativeElfOutputDir;
    entryModule /= getNativeLibName("eboot.bin");
    //if ( !mapEntryModuleIntoMemory(entryModule)) {
    //    return -1;
    //}

    std::vector<void *> preloadHandles;
    LibSearcher preloadSearcher({CmdArgs.preloadDirPath});

    // These may depend on ps4 libs (for wrapping sce functions by doing1 dlsym(RTLD_NEXT, ...))
    // So preload these only after patching dependencies
    // Note RTLD_GLOBAL: these symbols will take precedence over ps4 symbols in relocations
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

    //void *testdl = dlmopen(LM_ID_NEWLM, "./build/ps4lib_overloads/libevery.so", RTLD_NOW | RTLD_GLOBAL);


    // Load ps4 module
    // dl = dlmopen(preloadNamespace, firstLib.c_str(), RTLD_LAZY);
    dl = dlopen(firstLib.c_str(), RTLD_LAZY);
    //dl = dlmopen(lmid, firstLib.c_str(), RTLD_LAZY);
    if (!dl) {
        char *err;
        while ((err = dlerror())) {
            printf("(dlopen) %s\n", err);
        }
        return -1;
    }

    for (auto handle: preloadHandles) {
        dlclose(handle);
    }

    dlclose(dl);
    printf("main: done\n");

    return 0;
}