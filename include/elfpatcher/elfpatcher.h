#ifndef _ELF_PATCHER_H_
#define _ELF_PATCHER_H_

#include "Common.h"

#include <cstring>
#include <elf.h>
#include <filesystem>
namespace fs = std::filesystem;
#include <optional>
#include <vector>
#include <set>
#include <map>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class LibSearcher {
public:
    struct PathElt {
        fs::path path;
        bool recurse;
    };

    LibSearcher() {}

    LibSearcher(std::vector<fs::path> paths) {
        for (auto &path: paths) {
            m_paths.push_back({path, false});
        }
    }

    LibSearcher(std::vector<PathElt> paths): m_paths(paths) {}    

    std::optional<fs::path> findLibrary(fs::path name);
    std::optional<fs::path> findLibrary(fs::path stem, std::vector<fs::path> validExts);

private:
    std::vector<PathElt> m_paths;
};

struct Section {
public:
    Section(Elf64_Shdr sHdr): hdr(sHdr) {}
    Section() {
        hdr.sh_name = 0;
        hdr.sh_type = SHT_NULL;
        hdr.sh_flags = 0;
        hdr.sh_addr = 0;
        hdr.sh_offset = 0;
        hdr.sh_size = 0;
        hdr.sh_link = 0;
        hdr.sh_info = 0;
        hdr.sh_addralign = 0;
        hdr.sh_entsize = 0;
    }

    Elf64_Word getName() { return hdr.sh_name; }
    Elf64_Word getType() { return hdr.sh_type; }
    Elf64_Xword getFlags() { return hdr.sh_flags; }
    Elf64_Addr getAddr() { return hdr.sh_addr; }
    Elf64_Off getOffset() { return hdr.sh_offset; }
    Elf64_Xword getSize() { return hdr.sh_size; }
    Elf64_Word getLink() { return hdr.sh_link; }
    Elf64_Word getInfo() { return hdr.sh_info; }
    Elf64_Xword getAddrAlign() { return hdr.sh_addralign; }
    Elf64_Xword getEntSize() { return hdr.sh_entsize; }

    void setName(Elf64_Word name) { hdr.sh_name = name; }
    void setType(Elf64_Word type) { hdr.sh_type = type; }
    void setFlags(Elf64_Xword flags) { hdr.sh_flags = flags; }
    void setAddr(Elf64_Addr addr) { hdr.sh_addr = addr; }
    void setOffset(Elf64_Off offset) { hdr.sh_offset = offset; }
    void setSize(Elf64_Xword size) { hdr.sh_size = size; }
    void setLink(Elf64_Word link) { hdr.sh_link = link; }
    void setInfo(Elf64_Word info) { hdr.sh_info = info; }
    void setAddrAlign(Elf64_Xword addralign) { hdr.sh_addralign = addralign; }
    void setEntSize(Elf64_Xword entsize) { hdr.sh_entsize = entsize; }

    void appendContents(unsigned char *data, size_t len) {
        size_t off = contents.size();
        contents.resize(contents.size() + len);
        memcpy(&contents[off], data, len);
    }

    Elf64_Shdr hdr;
    std::vector<unsigned char> contents;
};

struct SectionMap {
    // Assume first section is null, so 0 index is taken and invalid
    uint shstrtabIdx;
    uint dynstrIdx;
    uint dynsymIdx;
    uint dynamicIdx;
    uint relroIdx;
    uint gotpltIdx;
    uint relaIdx;
    uint jmprelIdx;
    uint hashIdx;
    uint strtabIdx;
    uint symtabIdx;
};

struct DynamicTableInfo {
    // hashOff and hashSz can be ignored
    // The Elf-mandatory hash table can be produced without referring to the
    // hash table in PT_SCE_DYNAMIC/PT_SCE_DYNLIBDATA
    uint64_t hashOff;
    uint64_t hashSz;
    uint64_t strtabOff;
    uint64_t strtabSz;
    uint64_t symtabOff;
    uint64_t symtabSz;
    uint64_t symtabEntSz;
    uint64_t relaOff;
    uint64_t relaSz;
    uint64_t relaEntSz;
    uint64_t pltgotAddr;
    //uint64_t pltgotSz;  // should figure this one out
    uint64_t pltrel; // Type of reloc in PLT
    uint64_t jmprelOff;
    // pltrelsz aka jmprelSz
    // holds total size of all jmprel relocations
    // Is this also the size of the .got.plt section?
    // Wouldn't make much sense as all entries would be 24B, when 8B for an address in
    // the slot would make more sense
    uint64_t pltrelsz;
    // lower 32 bits of DT_SCE_MODULE_INFO. String table offset to *this* mod's "project name"
    uint32_t moduleInfoString;
    std::vector<uint64_t> neededLibs;
    std::vector<uint64_t> neededMods;
    // module id (from DT_SCE_NEEDED_MODULE/DT_SCE_IMPORT_LIB/DT_SCE_IMPORT_LIB_ATTR)
    // mapped to the name in the old dynlibdata's dynstr section
    // module id and idx are confusing. id (top 16 bits of d_val's +- 1) seem to be the consistent piece
    std::map<uint64_t, uint64_t> modIdToName;
};

struct Segment {
    Elf64_Phdr pHdr;
    std::vector<unsigned char> contents;

    Segment() {}

    Segment & operator=( Segment && rhs) {
        this->contents = std::move(rhs.contents);
        this->pHdr = rhs.pHdr;
        return *this;
    }

    Segment (Segment && other) {
        *this = std::move(other);
    }    
};

struct InitFiniInfo {
    std::optional<Elf64_Addr> preinit_array_base;
    std::vector<Elf64_Addr> dt_preinit_array;

    std::optional<Elf64_Addr> dt_init;

    std::optional<Elf64_Addr> init_array_base;
    std::vector<Elf64_Addr> dt_init_array;

    std::optional<Elf64_Addr> dt_fini;

    std::optional<Elf64_Addr> fini_array_base;
    std::vector<Elf64_Addr> dt_fini_array;
};

void to_json(json& j, const InitFiniInfo& p);
void from_json(const json& j, InitFiniInfo& p);

struct ElfPatcherContext {
    fs::path pkgdumpPath;
    LibSearcher ps4LibSearcher;
    
    fs::path nativeElfOutputDir;
    LibSearcher nativeLibSearcher;

    std::vector<fs::path> preloadNames;
    fs::path hashdbPath;
    bool purgeElfs;

    // Per-object state
    // dependencies (DT_NEEDED) on the processed ELF. Uses original names (non-native)
    std::vector<std::string> deps;
    // info about initialization and finalization functions
    InitFiniInfo initFiniInfo;

    ElfPatcherContext(std::string ps4libsPath, std::string preloadDir, std::string hashdbPath, std::string nativeElfOutputDir, std::string pkgdumpPath, bool purgeElfs):
            pkgdumpPath(pkgdumpPath),
            ps4LibSearcher({{pkgdumpPath, true}, {ps4libsPath, false}}),
            nativeElfOutputDir(nativeElfOutputDir),
            nativeLibSearcher({nativeElfOutputDir}),
            hashdbPath(hashdbPath),
            purgeElfs(purgeElfs)
    {
        fs::recursive_directory_iterator it(preloadDir);
        for (auto &dirent: it) {
            if (dirent.is_regular_file() && dirent.path().extension() == ".so") {
                preloadNames.push_back(dirent.path().filename());
            }
        }
    }

    // Reset per-object state
    void reset() {
        initFiniInfo = InitFiniInfo(); 
        deps.clear();
    }
};

/* EXPORTS */

// Get the canonical name to the patched ELF corresponding to name of the unpatched ELF given by ps4LibName
// This handles .sprx and .prx extension confusion
//
// This is used ahead of time, while patching a given ELF, to rename dependencies to other shared libraries (in DT_NEEDED tags).
// This is done before the patched ELF of the dependency is necessarily created

// For example:
// libA.sprx => libA.prx.native
// libA.prx => libA.prx.native
fs::path getNativeLibName(fs::path ps4LibName);

// Find the path to the given sce library, patched (native) or unpatched
// Native ELF's are found in the elfdump directory, and correspond to Ctx.nativeLibSearcher
// Unpatched ELF's are found in the pkg dump or the default system (ps4) libraries, and correspond
// to Ctx.ps4LibSearcher
// Given a name with the .prx or .sprx extension, this function looks for the best match
std::optional<fs::path> findPathToSceLib(fs::path ps4LibName, ElfPatcherContext &Ctx);

// Patch the ps4 ELF at "elfPath" to a legal ELF.
// Insert readable symbols for any symbol found that is a known hash
//
// This doesn't patch any dependencies of "inputElfPath".
// Instead it returns the names of dependencies in "dependencies"
// These are the ps4/sce ELF names and not the canonical patched names.
// To be able to dlopen() or exec() this, all the recursive dependencies should be patched beforehand 
bool patchPs4Lib(ElfPatcherContext &Ctx, std::string elfPath);

bool dumpPatchedElfInfoToJson(fs::path jsonPath, fs::path elfPath, InitFiniInfo &initFiniInfo);

std::optional<json> parsePatchedElfInfoFromJson(fs::path jsonPath, InitFiniInfo *initFiniInfo = nullptr);

bool findDependencies(fs::path patchedElf, std::vector<std::string> &deps);

#endif // _ELF_PATCHER_H_