#include "Common.h"
#include "elfpatcher/elfpatcher.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <elf.h>
#include <fstream>
#include <fcntl.h>
#include <boost/iostreams/device/mapped_file.hpp>
#include <optional>
#include <iostream>

void to_json(json& j, const InitFiniInfo& info) {
    if ( !info.dt_preinit_array.empty()) {
        j["dt_preinit_array"] = info.dt_preinit_array;
    }
    if (info.dt_init) {
        j["dt_init"] = info.dt_init.value();
    }
    if ( !info.dt_init_array.empty()) {
        j["dt_init_array"] = info.dt_init_array;
    }
    if (info.dt_fini) {
        j["dt_fini"] = info.dt_fini.value();
    }
    if ( !info.dt_fini_array.empty()) {
        j["dt_fini_array"] = info.dt_fini_array;
    }
}

void from_json(const json& j, InitFiniInfo& info) {
    if (j.contains("dt_preinit_array")) {
        j["dt_preinit_array"].get_to<std::vector<Elf64_Addr>>(info.dt_preinit_array);
    }
    if (j.contains("dt_init")) {
        info.dt_init = j["dt_init"].get<Elf64_Addr>();
    }
    if (j.contains("dt_init_array")) {
        j["dt_init_array"].get_to<std::vector<Elf64_Addr>>(info.dt_init_array);
    }
    if (j.contains("dt_fini")) {
        info.dt_fini = j["dt_fini"].get<Elf64_Addr>();
    }
    if (j.contains("dt_fini_array")) {
        j["dt_fini_array"].get_to<std::vector<Elf64_Addr>>(info.dt_fini_array);
    }
}

bool dumpPatchedElfInfoToJson(fs::path jsonPath, const PatchedElfInfo &elfInfo) {
    std::ofstream of(jsonPath);
    if ( !of.is_open()) {
        return false;
    }
    of << std::setw(4) << json(elfInfo) << std::endl;

    return true;
}

std::optional<PatchedElfInfo> parsePatchedElfInfoFromJson(fs::path jsonPath) {
    std::ifstream is(jsonPath);
    if ( !is.is_open()) {
        return std::nullopt;
    }

    json elfJson;
    is >> elfJson;

    return elfJson;
}
