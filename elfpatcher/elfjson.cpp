#include "Common.h"
#include "elfpatcher/elfpatcher.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <boost/iostreams/device/mapped_file.hpp>

static bool getFileHash(fs::path path, uint64_t &hash) {
    boost::iostreams::mapped_file_source file;
    file.open(path);
    if ( !file.is_open()) {
        fprintf(stderr, "Couldn't open %s\n", path.c_str());
        return false;
    }

    hash = simpleHash((unsigned char *) file.data(), file.size());
    return true;
}

bool dumpPatchedElfInfoToJson(fs::path jsonPath, fs::path elfPath, InitFiniInfo &initFiniInfo) {
    json elfJson;
    elfJson["filename"] = elfPath.filename();

    uint64_t hash;
    if ( !getFileHash(elfPath, hash)) {
        return false;
    }
    elfJson["hash"] = hash;

    if ( !initFiniInfo.dt_preinit_array.empty()) {
        elfJson["dt_preinit_array"] = initFiniInfo.dt_preinit_array;
    }
    if (initFiniInfo.dt_init) {
        elfJson["dt_init"] = initFiniInfo.dt_init.value();
    }
    if ( !initFiniInfo.dt_init_array.empty()) {
        elfJson["dt_init_array"] = initFiniInfo.dt_init_array;
    }
    if (initFiniInfo.dt_fini) {
        elfJson["dt_fini"] = initFiniInfo.dt_fini.value();
    }
    if ( !initFiniInfo.dt_fini_array.empty()) {
        elfJson["dt_fini_array"] = initFiniInfo.dt_fini_array;
    }

    std::ofstream of(jsonPath);
    if ( !of.is_open()) {
        return false;
    }
    of << std::setw(4) << elfJson << std::endl;

    return true;
}

std::optional<json> parsePatchedElfInfoFromJson(fs::path jsonPath, InitFiniInfo *initFiniInfo) {
    std::ifstream is(jsonPath);
    if ( !is.is_open()) {
        return false;
    }

    json elfJson;
    is >> elfJson;

    if (initFiniInfo) {
        initFiniInfo->dt_preinit_array = elfJson["dt_preinit_array"].get<std::vector<Elf64_Addr>>();
        initFiniInfo->dt_init = elfJson["dt_init"];
        initFiniInfo->dt_init_array = elfJson["dt_init_array"].get<std::vector<Elf64_Addr>>();
        initFiniInfo->dt_fini = elfJson["dt_fini"];
        initFiniInfo->dt_fini_array = elfJson["dt_fini_array"].get<std::vector<Elf64_Addr>>();
    }

    return elfJson;
}
