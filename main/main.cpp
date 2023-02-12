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

const long PGSZ = sysconf(_SC_PAGE_SIZE);
const std::string ebootPath = "eboot.bin";

#define ROUND_DOWN(x, SZ) ((x) - (x) % (SZ))
#define ROUND_UP(x, SZ) ( (x) % (SZ) ? (x) - ((x) % (SZ)) + (SZ) : (x))

static fs::path getNativeLibPath(fs::path ps4LibName) {
    assert(ps4LibName.has_filename());

    auto ext = ps4LibName.extension();
    assert(ext != ".native");
    if (ext == ".sprx") {
        ext = ".prx";
    }
    ext += ".native";

    auto libStem = ps4LibName.stem();
    libStem += ext;

    return libStem;
}

// Store the names to compatibility libs to preload for each ps4 lib
// This will work like LD_PRELOAD for the ps4 libs themselves.
// Using LD_PRELOAD would cause ld-linux to lookup symbols in these compatibility libs
// when loading the emu executable itself, which we don't want
//
// These will be prepended to the list of DT_NEEDED for each patched ps4 libs
// The dir given to init should be set in LD_LIBRARY_PATH as well so dlopen can locate
// each DT_NEEDED entry
//
// Could also append to the DT_NEEDED's of the Emu executable
static struct {
    std::vector<std::string> soNames;

    void init(fs::path preloadDir) {
        std::string ldLibraryPath = getenv("LD_LIBRARY_PATH");
        assert( ldLibraryPath.find(preloadDir) != ldLibraryPath.npos);

        fs::directory_iterator it(preloadDir);
        for (auto &dirent: it) {
            if (dirent.is_regular_file() && dirent.path().extension() == ".so") {
                soNames.push_back(dirent.path().filename());
            }
        }
    }
} g_Preloads;

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

    std::optional<fs::path> findLibrary(fs::path name) {
        for (PathElt &elt: m_paths) {
            assert( !name.has_parent_path());
            if (elt.recurse) {
                fs::path filename = name.filename();
                fs::recursive_directory_iterator it(elt.path);
                for (const fs::directory_entry& dirent: it) {
                    if (dirent.is_regular_file()) {
                        auto entpath = dirent.path();
                        if (entpath.filename() == filename) {
                            return entpath;
                        }
                    }
                }
            } else {
                fs::path match = elt.path;
                match /= name;
                struct stat buf;
                if ( !stat(match.c_str(), &buf)) {
                    return match;
                }
            }
        }
        return std::nullopt;
    }

    std::optional<fs::path> findLibrary(fs::path stem, std::vector<fs::path> validExts) {
        for (auto &ext: validExts) {
            fs::path f = stem;
            f += ext;
            if (auto ret = findLibrary(f)) {
                return ret;
            }
        }
        return std::nullopt;
    }

private:
    std::vector<PathElt> m_paths;
};

static LibSearcher TheLibrarySearcher;

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

CmdConfig g_CmdArgs;

bool parseCmdArgs(int argc, char **argv) {
    if (argc < 2) {
        //fprintf(stderr, "usage: %s [options] <PATH TO PKG DUMP>\n", argv[0]);
        fprintf(stderr, "usage: %s [options] <PATH TO ELF>\n", argv[0]);
        exit(1);
    }

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--hashdb")) {
            g_CmdArgs.hashdbPath = argv[++i];
        } else if (!strcmp(argv[i], "--native_elf_output")) {
            g_CmdArgs.nativeElfOutputDir = argv[++i];
        } else if(!strcmp(argv[i], "--pkgdump")) {
            g_CmdArgs.pkgdumpPath = argv[++i];
        } else if(!strcmp(argv[i], "--ps4libs")) {
            g_CmdArgs.ps4libsPath = argv[++i];
        } else if (!strcmp(argv[i], "--preload_dir")) {
            g_CmdArgs.preloadDirPath = argv[++i];
        } else if (!strcmp(argv[i], "--purge_elfs")) {
            g_CmdArgs.purgeElfs = true;
        } else {
            fprintf(stderr, "Unrecognized cmd arg: %s\n", argv[i]);
            return false;
        }
    }

    if (g_CmdArgs.hashdbPath.empty()) {
        fprintf(stderr, "Warning: no symbol hash database given\n");
        exit(1); // For debugging
    }

    //CmdArgs.pkgDumpPath = argv[argc - 1];
    g_CmdArgs.dlPath = argv[argc - 1];
    return true;
}

static std::optional<fs::path> findPathToLibName(fs::path ps4LibName) {
    if (ps4LibName == g_CmdArgs.dlPath) {
        return ps4LibName;
    }

    if (!ps4LibName.has_extension()) {
        fprintf(stderr, "Warning, findPathToLibName: library %s has no extension\n", ps4LibName.c_str());
    }

    fs::path ext = ps4LibName.extension();
    std::vector<fs::path> validExts = { ext };
    if (ext == ".prx") {
        validExts.push_back(".sprx");
    } else if (ext == ".sprx") {
        validExts.push_back(".prx");
    }

    fs::path libStem = ps4LibName.stem();

    return TheLibrarySearcher.findLibrary(libStem, validExts);
}

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

void printSegmentRanges(std::vector<Elf64_Phdr>& progHdrs) {
    for (auto p: progHdrs) {
        printf("[%lx, %lx), size=%lx\n", p.p_offset, p.p_offset + p.p_filesz, p.p_filesz);
    }
}

int findPhdr(std::vector<Elf64_Phdr> &pHdrs, Elf64_Word type) {
    for (size_t i = 0; i < pHdrs.size(); i++) {
        if (pHdrs[i].p_type == type) {
            return i;
        }
    }
    return -1;
}

bool writePadding(FILE *f, size_t alignment, bool forceBump = false) {
    fseek(f, 0, SEEK_END);
    size_t flen = ftell(f);

    // force power of 2
    assert(alignment != 0 && (((alignment - 1) & alignment) == 0)); 
    size_t padlen = ROUND_UP(flen | (forceBump ? 1 : 0), alignment) - flen;
    std::vector<unsigned char> padding(padlen, 0);
    if (padlen > 0) {
        assert(1 == fwrite(padding.data(), padlen, 1, f));
    }

    return true;
}

// return index into strtab
static uint appendToStrtab(Section &strtab, const char *str) {
    uint off = strtab.contents.size();
    strtab.appendContents((unsigned char *) str, strlen(str) + 1);

    return off;
}

Segment CreateSegment(Elf64_Phdr pHdr, std::vector<Section> &sections, std::vector<uint> idxs) {
    Segment seg;
    seg.pHdr = pHdr;

    uint64_t segBeginVa = pHdr.p_vaddr;
    uint64_t segBeginFileOff = pHdr.p_offset;

    std::vector<unsigned char> &totalContents = seg.contents;

    for (uint idx: idxs) {
        Section &section = sections[idx];
        uint64_t segOff = ROUND_UP(totalContents.size(), section.getAddrAlign());
        std::vector<unsigned char> &sectionContents = section.contents;
        totalContents.resize(segOff + sectionContents.size());

        memcpy(&totalContents[segOff], &sectionContents[0], sectionContents.size());

        section.setAddr(segBeginVa + segOff);
        section.setOffset(segBeginFileOff + segOff);
        section.setSize(sectionContents.size());

        assert((seg.pHdr.p_vaddr + segOff) % section.getAddrAlign() == 0);
    }

    seg.pHdr.p_filesz = totalContents.size();
    seg.pHdr.p_memsz = totalContents.size(); // ??? TODO possibly

    return seg;
}

// find a new virtual and physical address for pHdr
// use the maximum offsets of other progHdrs so no overlapping happens
void rebaseSegment(Elf64_Phdr *pHdr, std::vector<Elf64_Phdr> &progHdrs) {
    uint64_t highestVAddr = 0;
    uint64_t highestPAddr = 0;    
    for (auto phdr: progHdrs) {
        highestVAddr = std::max(highestVAddr, phdr.p_vaddr + phdr.p_memsz);
        highestPAddr = std::max(highestPAddr, phdr.p_paddr + phdr.p_memsz);
    }
    pHdr->p_vaddr = ROUND_UP(highestVAddr, pHdr->p_align);
    pHdr->p_paddr = ROUND_UP(highestPAddr, pHdr->p_align);    
}

static int getLibraryOrModuleIndex(const char *str) {
    const char *libModIndex = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    const uint radix = strlen(libModIndex);
    const uint chars = strlen(str);

    // Keep assert until we find a case of this
    // If there are >radix modules/libs, we could need 2 digits
    assert(chars == 1);

    uint idx = 0;
    for (uint i = 0; i < chars; i++)
    {
        uint j = 0;
        for(; j < radix; j++) {
            if (libModIndex[j] == str[i])
                break;
        }
        assert((j < radix) && "Invalid character for library or module index");
        if (j >= radix) {
            return -1;
        }
        idx *= radix;
        idx += j;
    }
    return idx;
}

static bool isHashedSymbol(const char *str, int &libIdx, int &modIdx) {
    std::string sym(str);
    if (sym.length() > 12) {
        std::string suff = sym.substr(11);
        if (suff[0] == '#') {
            auto secondDelim = suff.find_first_of('#', 1);
            if (secondDelim != suff.npos) {
                assert(secondDelim != suff.size());
                if (secondDelim != suff.size()) {
                    std::string lib = suff.substr(1, secondDelim - 1);
                    std::string mod = suff.substr(secondDelim + 1);
                    libIdx = getLibraryOrModuleIndex(lib.c_str());
                    modIdx = getLibraryOrModuleIndex(mod.c_str());
                    if (libIdx >= 0 && modIdx >= 0) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

static sqlite3 *openHashDb(const char *path) {
    sqlite3 *db;
    int res = sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nullptr);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Couldn't open database %s\n", path);
        return nullptr;
    }
    return db;
}

static bool reverseKnownHashes(std::vector<const char *> &oldStrings, std::vector<std::string> &newStrings) {
    sqlite3 *db = openHashDb(g_CmdArgs.hashdbPath.c_str());
    if ( !db) {
        return false;
    }

    std::stringstream sql;
    sql << 
        "SELECT * FROM Hashes\n"
        "WHERE hash IN\n"
        "(";

    uint numHashes = 0;
    for (uint i = 0; i < oldStrings.size(); i++) {
        const char *old = oldStrings[i];
        int libIdx, modIdx;
        if (!isHashedSymbol(old, libIdx, modIdx)) {
            continue;
        }
        if (numHashes > 0) {
            sql << ",";
        }
        std::string hash(old, 11);
        sql << "'" << hash << "'";
        numHashes++;
    }

    sql << ");";

    auto ssql = sql.str();

    sqlite3_stmt *stmt;
    int res = sqlite3_prepare(db, ssql.c_str(), ssql.size(), &stmt, NULL);
    if (res != SQLITE_OK) {
        fprintf(stderr, "sqlite3_prepare errored\n");
        return false;
    }

    std::map<std::string, std::string> hashToSymbol;
    bool doMore;
    do {
        int res = sqlite3_step(stmt);
        switch (res) {
            case SQLITE_ROW:
                doMore = true;
                break;
            case SQLITE_DONE:
                doMore = false;
                break;
            case SQLITE_ERROR:
                fprintf(stderr, "sqlite3_step errored\n");
                return false;
            default:
                fprintf(stderr, "reverseKnownHashes: sqlite3_step unhandled code\n");
                return false;
        }
        if (!doMore) {
            break;
        }

        const char *symbol = (const char *) sqlite3_column_text(stmt, 0);
        const char *hash = (const char *) sqlite3_column_text(stmt, 1);
        hashToSymbol[hash] = symbol;
    } while (doMore);

    sqlite3_finalize(stmt);
    sqlite3_close_v2(db);

    for (const char *old: oldStrings) {
        int libIdx, modIdx;
        if (isHashedSymbol(old, libIdx, modIdx)) {
            std::string hash(old, 11);
            const auto &it = hashToSymbol.find(hash);
            if (it != hashToSymbol.end()) {
                const std::string& symbol = it->second; 
                newStrings.push_back(symbol);
            } else {
                // truncate #A#B lib/module id's
                newStrings.push_back(hash);
            }
        } else {
            newStrings.push_back(old);
        }
    }

    return true;
}

unsigned long elf_Hash(const unsigned char *name) {
    unsigned long h = 0, g;
 
	    while (*name)
	    {
		     h = (h << 4) + *name++;
		     if ((g = h & 0xf0000000))
			      h ^= g >> 24;
				   h &= ~g;
	    }
	    return h;
}

static void createSymbolHashTable(DynamicTableInfo &info, std::vector<Section> &sections, SectionMap &sMap) {
    Section &hashtable = sections[sMap.hashIdx];
    const Section &dynsym = sections[sMap.dynsymIdx];
    const Section &dynstr = sections[sMap.dynstrIdx];
    
    size_t numSyms = info.symtabSz / info.symtabEntSz;

    Elf64_Word nbucket = 200;
    Elf64_Word nchain = numSyms;

    const Elf64_Sym *syms = reinterpret_cast<const Elf64_Sym *>(dynsym.contents.data());    

    hashtable.appendContents((unsigned char *) &nbucket, sizeof(nbucket));
    hashtable.appendContents((unsigned char *) &nchain, sizeof(nchain));
    hashtable.contents.resize(sizeof(nbucket) + sizeof(nchain) + sizeof(Elf64_Word) * (nbucket + nchain));
    Elf64_Word *buckets = reinterpret_cast<Elf64_Word *>(&hashtable.contents[sizeof(nbucket) + sizeof(nchain)]);
    Elf64_Word *chains = buckets + nbucket;
    for (uint i = 0; i < nbucket + nchain; i++) {
        buckets[i] = STN_UNDEF;
    }    

    for (uint i = 0; i < numSyms; i++) {
        Elf64_Word strOff = syms[i].st_name;
        const unsigned char *name = &dynstr.contents[strOff];
        size_t bIdx = elf_Hash(name) % nbucket;
        if (buckets[bIdx] == STN_UNDEF) {
            buckets[bIdx] = i;
            continue;
        }
        size_t cIdx = buckets[bIdx];
        while (chains[cIdx] != STN_UNDEF) {
            cIdx = chains[cIdx];
        }
        chains[cIdx] = i;
    }
}

// Build new dynamic sections from the PT_SCE_DYNLIBDATA segment
static bool fixDynlibData(FILE *elf, std::vector<Elf64_Phdr> &progHdrs, DynamicTableInfo &dynInfo, std::vector<Section> &sections, SectionMap &sMap, std::vector<Elf64_Dyn> &newDynEnts, std::set<std::string> &dependencies) {
    uint phIdx = findPhdr(progHdrs, PT_SCE_DYNLIBDATA);
    std::vector<unsigned char> dynlibContents(progHdrs[phIdx].p_filesz);

    assert(sMap.dynstrIdx);
    assert(sMap.shstrtabIdx);
    assert(sMap.dynsymIdx);
    assert(sMap.relaIdx);
    assert(sMap.gotpltIdx);

    fseek(elf, progHdrs[phIdx].p_offset, SEEK_SET);
    if (1 != fread(dynlibContents.data(), progHdrs[phIdx].p_filesz, 1, elf)) {
        return false;
    }

    // Ps4 ELF's always seem to have plt rela entries (.rela.plt) followed immediately by normal rela entries (.rela)
    // Keep this until I find a counterexample
    assert(dynInfo.relaOff == dynInfo.jmprelOff + dynInfo.pltrelsz);

    uint numJmpRelas = dynInfo.pltrelsz / dynInfo.relaEntSz;
    Elf64_Rela *jmprelas = reinterpret_cast<Elf64_Rela *>(&dynlibContents[dynInfo.jmprelOff]);
    Section &jmprela = sections[sMap.jmprelIdx];
    for (uint i = 0; i < numJmpRelas; i++) {
        Elf64_Rela ent = jmprelas[i];
        jmprela.appendContents((unsigned char *) &ent, sizeof(ent));
    }

    uint numRelas = dynInfo.relaSz / dynInfo.relaEntSz;
    Elf64_Rela *relas = reinterpret_cast<Elf64_Rela *>(&dynlibContents[dynInfo.relaOff]);
    Section &rela = sections[sMap.relaIdx];
    for (uint i = 0; i < numRelas; i++) {
        Elf64_Rela ent = relas[i];
        rela.appendContents((unsigned char *) &ent, sizeof(ent));
    }

    std::vector<const char *> oldSymStrings;

    uint numSyms = dynInfo.symtabSz / dynInfo.symtabEntSz;
    Elf64_Sym *syms = reinterpret_cast<Elf64_Sym *>(&dynlibContents[dynInfo.symtabOff]);
    for (uint i = 0; i < numSyms; i++) {
        Elf64_Sym ent = syms[i];
        if (ent.st_name) {
            oldSymStrings.push_back((const char *) &dynlibContents[dynInfo.strtabOff+ ent.st_name]);
        } else {
            oldSymStrings.push_back("");
        }
    }

    std::vector<std::string> newStrings;
    if (!g_CmdArgs.hashdbPath.empty()) {
        if ( !reverseKnownHashes(oldSymStrings, newStrings)) {
            return false;
        }
    }

    int firstNonLocalIdx = -1;
    Section &dynsym = sections[sMap.dynsymIdx];
    for (uint i = 0; i < numSyms; i++) {
        Elf64_Sym ent = syms[i];
        if (ent.st_name) {
            ent.st_name = appendToStrtab(sections[sMap.dynstrIdx], newStrings[i].c_str());
        }
        //ent.st_info = ELF64_ST_INFO(0, 0);
        //ent.st_info;
        //ent.st_other;
        //ent.st_value;

        // I think shndx doesn't matter for dynamic linking, and that st_value is just used as a VA
        // I think it's only relevant for static linking, when sections from multiple objects are being rearranged
        // and stitched into linked objects. In that case, providing the shndx is indirection that lets you keep the location
        // section_base + st_value consistent with the real location, because the section contents are moved by the static linker
        //
        // Even though it shouldn't affect dynamic linking, it seems to mess with the debugger
        // Also 0 (SHT_NULL) does actually seem to mess with dlopen()
        if (ent.st_shndx >= SHN_LORESERVE && ent.st_shndx <= SHN_LORESERVE) {
            if (false) {
                printf("Symbol has reserved symbol shndx\n");
            }
        } else if (ent.st_shndx != SHN_UNDEF) {
            // Using a random shdnx, e.g. 777, messes with the debugger
            // For now, .dynsym has worked. Technically this should refer to the .text or .data section I think.
            // LLDB has shown function names from the converted elf's in the disas and callstack
            ent.st_shndx = sMap.dynsymIdx; // TODO
        }
        //ent.st_value; // TODO - ensure all pointer values are in LOAD segments that haven't been rebased or removed
        // For example, the got and plt
        //ent.st_size;

        if (firstNonLocalIdx < 0 && ELF64_ST_BIND(ent.st_info) != STB_LOCAL) {
            firstNonLocalIdx = i;
        }

        dynsym.appendContents((unsigned char *) &ent, sizeof(ent));
    }

    if (firstNonLocalIdx < 0) {
        firstNonLocalIdx = numSyms;
    }
    // the sh_info field for .dynsym should contain the first symbol that doesn't have STB_LOCAL
    sections[sMap.dynsymIdx].setInfo(firstNonLocalIdx);

    // Create Hash table
    createSymbolHashTable(dynInfo, sections, sMap);

    // Write needed libs and modules
    for (uint64_t libStrOff: dynInfo.neededLibs) {
        fs::path ps4Name = reinterpret_cast<char *>(&dynlibContents[dynInfo.strtabOff + libStrOff]);
        if ( findPathToLibName(ps4Name)) {
            dependencies.insert(ps4Name);
        } else {
            fprintf(stderr, "Warning: couldn't find dependency %s needed by %s. Skipping\n", ps4Name.c_str(), g_DebugContext.currentPs4Lib.c_str());
            continue;
        }

        fs::path nativeName = getNativeLibPath(ps4Name);

        uint64_t name = appendToStrtab(sections[sMap.dynstrIdx], nativeName.c_str());
        Elf64_Dyn needed;
        needed.d_tag = DT_NEEDED;
        needed.d_un.d_val = name;
        newDynEnts.push_back(needed);
    }

#if 0
    for (uint64_t modInfo: dynInfo.neededMods) {
        //uint64_t id = modInfo >> 48;
        // I think we should subtract 1 here.
        // Otherwise the DT_SCE_NEEDED_MODULE and DT_SCE_IMPORT_LIB tags seem to be off by 1
        uint64_t id = (modInfo >> 48) - 1;
        uint64_t minor = (modInfo >> 40) & 0xF;
        uint64_t major  = (modInfo >> 32) & 0xF;
        uint64_t index = modInfo & 0xFFF;

        if (dynInfo.modIdToName.find(id) == dynInfo.modIdToName.end()) {
            fprintf(stderr, "Warning: module with id %lu (idx %lu, version %lu.%lu), not found in previous DT_SCE_IMPORT_LIB\n", 
                id, index, major, minor);
        } else {
            uint64_t modStrOff = dynInfo.modIdToName[id];
            const char *name = reinterpret_cast<char *>(&dynlibContents[dynInfo.strtabOff + modStrOff]);

            printf("Module name for id %lu (idx %lu, version %lu.%lu) : %s\n", id, index, major, minor, name);
        }
    }
#endif

    // Create dynlibdata segment
    std::vector<uint> dynlibDataSections = {
        sMap.dynstrIdx,
        sMap.dynsymIdx,
        sMap.hashIdx,
        sMap.jmprelIdx,
        sMap.relaIdx,
    };
    Elf64_Phdr newDynlibDataSegmentHdr {
        .p_type = PT_LOAD,
        .p_flags = PF_R | PF_W | PF_X,
        .p_align = static_cast<Elf64_Xword>(PGSZ)
    };
    rebaseSegment(&newDynlibDataSegmentHdr, progHdrs);
    fseek(elf, 0, SEEK_END);
    writePadding(elf, newDynlibDataSegmentHdr.p_align, true);
    newDynlibDataSegmentHdr.p_offset = ftell(elf);
    Segment dynlibDataSegment = CreateSegment(newDynlibDataSegmentHdr, sections, dynlibDataSections);
    assert(1 == fwrite(dynlibDataSegment.contents.data(), dynlibDataSegment.contents.size(), 1, elf));
    progHdrs.push_back(dynlibDataSegment.pHdr);

    newDynEnts.push_back({
        DT_HASH,
        {sections[sMap.hashIdx].getAddr()} // TODO
    });

    newDynEnts.push_back({
        DT_PLTGOT,
        {sections[sMap.gotpltIdx].getAddr()}
    });

    newDynEnts.push_back({ 
        DT_JMPREL,
        {sections[sMap.jmprelIdx].getAddr()}
    });

    newDynEnts.push_back({
        DT_PLTRELSZ,
        {sections[sMap.jmprelIdx].getSize()}
    });

    newDynEnts.push_back({
        DT_PLTREL,
        {dynInfo.pltrel}
    });

    newDynEnts.push_back({ 
        DT_RELA,
        {sections[sMap.relaIdx].getAddr()}
    });

    // DT_RELASZ
    // TODO make sure this stays the same
    newDynEnts.push_back({
        DT_RELASZ,
        {sections[sMap.relaIdx].getSize()}
    });

    // DT_RELAENT
    newDynEnts.push_back({
        DT_RELAENT,
        {dynInfo.relaEntSz}
    });

    // DT_STRTAB
    newDynEnts.push_back({
        DT_STRTAB,
        {sections[sMap.dynstrIdx].getAddr()}
    });

    // DT_STRSZ
    newDynEnts.push_back({
        DT_STRSZ,
        {sections[sMap.dynstrIdx].getSize()}
    });

    // DT_SYMTAB
    newDynEnts.push_back({
        DT_SYMTAB,
        {sections[sMap.dynsymIdx].getAddr()}
    });

    // DT_SYMENT
    newDynEnts.push_back({
        DT_SYMENT,
        {dynInfo.symtabEntSz}
    });

    return true;
}

// write new Segment for PT_DYNAMIC based on SCE dynamic entries
// append to end of file
// modify the progHdrs array
// add new dynlibdata sections and update section map
static bool fixDynamicInfoForLinker(FILE *elf, std::vector<Elf64_Phdr> &progHdrs, std::vector<Section> &sections, SectionMap &sMap, std::set<std::string> &dependencies) {
    uint oldDynamicPIdx = findPhdr(progHdrs, PT_DYNAMIC);
    Elf64_Phdr DynPhdr = progHdrs[oldDynamicPIdx];
    Elf64_Shdr sHdr;
    std::vector<Elf64_Dyn> newDynEnts;
    DynamicTableInfo dynInfo;

    sMap.dynstrIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".dynstr"),
        .sh_type = SHT_STRTAB,
        .sh_flags = SHF_STRINGS | SHF_ALLOC, // TODO check this, confusingly ELFs I've seen don't set SHF_STRINGS
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
    };
    sections.emplace_back(sHdr);

    sMap.dynsymIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".dynsym"),
        .sh_type = SHT_SYMTAB,
        .sh_flags = SHF_ALLOC,
        .sh_link = sMap.dynstrIdx,
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
        .sh_entsize = sizeof(Elf64_Sym),
    };
    sections.emplace_back(sHdr);

    sMap.hashIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".hash"),
        .sh_type = SHT_HASH,
        .sh_flags = SHF_ALLOC,
        .sh_link = sMap.dynsymIdx,
        .sh_addralign = 8,
        .sh_entsize = 0, // TODO?
    };
    sections.emplace_back(sHdr);

    std::vector<unsigned char> buf(DynPhdr.p_filesz);
    fseek(elf, DynPhdr.p_offset, SEEK_SET);
    assert(1 == fread(buf.data(), DynPhdr.p_filesz, 1, elf));

    Elf64_Dyn *dyn = (Elf64_Dyn*) buf.data();
    
    int maxDynHeaders = DynPhdr.p_filesz / sizeof(Elf64_Dyn);
    for(int i = 0; i < maxDynHeaders; i++, dyn++) {
        DynamicTag tag = (DynamicTag) dyn->d_tag;
        if (tag == DT_NULL) {
            break;
        }

        //printf("%s\n", to_string(tag).c_str());
        switch(tag) {
            // DT_ tags
            case DT_NULL:
                assert(false && "shouldn't be here");
                break;
            case DT_NEEDED:
                // "Additionally, DT_NEEDED tags are created for each dynamic library used by the application.
                // The value is set to the offset of the library's name in the string table. Each DT_NEEDED tag 
                // should also have a corresponding DT_SCE_IMPORT_LIB and DT_SCE_IMPORT_LIB_ATTR tag."
                // Need to change strings to their legal ELF counterparts.
                // Find out which strtab this should use
                dynInfo.neededLibs.push_back(dyn->d_un.d_val);
                break;
            case DT_SONAME:
                // TODO
                //fprintf(stderr, "Warning: unhandled DynEnt DT_SONAME\n");
                break;

            case DT_REL:
            case DT_RELSZ:
            case DT_RELENT:
            {
                std::string name = to_string(tag);
                printf("Warning: unhandled DynEnt %s\n", name.c_str());
                assert(false); // We will add empty DT_REL* tags for executables,
                // so assume DT_RELA* are the only ones given until seeing otherwise
                break;
            }
                

            // https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter2-55859.html#chapter2-48195
            //case DT_INIT:
            //case DT_FINI:
            //case DT_INIT_ARRAY:
            case DT_FINI_ARRAY:
            case DT_INIT_ARRAYSZ:
            case DT_FINI_ARRAYSZ:
                break;

            case DT_INIT:
            //case DT_FINI:
            case DT_INIT_ARRAY:
            //case DT_FINI_ARRAY:
            //case DT_INIT_ARRAYSZ:
            //case DT_FINI_ARRAYSZ:

            case DT_RPATH:
            case DT_SYMBOLIC:
            case DT_DEBUG:
            case DT_TEXTREL:
            case DT_BIND_NOW:
            case DT_RUNPATH:
            case DT_FLAGS:
                //https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-42444.html#chapter7-tbl-5            
                // should be DF_TEXTREL
            case DT_PREINIT_ARRAY:
            case DT_PREINIT_ARRAYSZ:
            case DT_SYMTAB_SHNDX:
            case DT_RELRSZ:
            case DT_RELR:
            case DT_RELRENT:
            case DT_NUM:
            case DT_LOOS:
            case DT_HIOS:
            case DT_LOPROC:
            case DT_HIPROC:
            case DT_PROCNUM:
                newDynEnts.push_back(*dyn);
                //printf("unhandled tag: %s\n", to_string(tag).c_str());
                break;

            case DT_SCE_MODULE_INFO:
            {
                uint64_t upp = dyn->d_un.d_val >> 32;
                uint64_t low = dyn->d_un.d_val & 0xffffffff;
                //assert(upp == 0x101);
                if (upp != 0x101) {
                    fprintf(stderr, "Warning: upp != 0x101 in DT_SCE_MODULE_INFO\n");
                }
                dynInfo.moduleInfoString = low;
                break;
            }
            case DT_SCE_NEEDED_MODULE:
                // dunno the difference between libs and modules here
                // "The indexes into the library list start at index 0, and the indexes into the module
                // list start at index 1. Most of the time, the library list and module list are in 
                // the same order, so the module ID is usually the library ID + 1. 
                // This is not always the case however because some modules contain more than one library."
                dynInfo.neededMods.push_back(dyn->d_un.d_val);
                break;
            case DT_SCE_MODULE_ATTR:
                // ignore
                break;
            case DT_SCE_IMPORT_LIB:
            {
                uint64_t upp = dyn->d_un.d_val >> 32;
                uint64_t low = dyn->d_un.d_val & 0xffffffff;
                assert((upp - 1) % 0x10000 == 0);
                uint64_t id = (upp - 1) / 0x10000;
                dynInfo.modIdToName[id] = low;
                break;
            }
            case DT_SCE_IMPORT_LIB_ATTR:
            {
                uint64_t upp = dyn->d_un.d_val >> 32;
                uint64_t low = dyn->d_un.d_val & 0xffffffff;
                assert(upp % 0x10000 == 0);
                assert(low == 0x9);
                break;
            }
            case DT_SCE_HASH:
                // sym hashtable, ignore for now
                dynInfo.hashOff = dyn->d_un.d_val;
                break;
            case DT_SCE_PLTGOT:
                // Addr of table that contains .data.rel.ro and .got.plt
                // Can possibly just convert to DT_PLTGOT
                // (didn't need this in old attempt, relocs just affect this area)
                // Should add section headers
                dynInfo.pltgotAddr = dyn->d_un.d_val;

                break;
            case DT_SCE_JMPREL:
                // Offset of the table containing jump slot relocations
                // relative to dynlibdata segment start
                // At this offset, there are DT_SCE_PLTRELSZ:d_val jmp slot relocations,
                // then DT_SCE_RELASZ:d_val relas seem to follow
                dynInfo.jmprelOff = dyn->d_un.d_val;
                break;                 
            case DT_SCE_PLTREL:
                // Types of relocations (DT_RELA)
                assert(dyn->d_un.d_val == DT_RELA);
                dynInfo.pltrel = dyn->d_un.d_val;
                break;
            case DT_SCE_PLTRELSZ:
                // Seems to be the # of jmp slot relocations (* relaentsz).
                // See DT_SCE_JMPREL
                // point of confusion
                dynInfo.pltrelsz = dyn->d_un.d_val;
                break;                  
            case DT_SCE_RELA:
                // seems unecessary
                // maybe because redundancy with DT_SCE_JMPREL, DT_SCE_PLTRELSZ, DT_SCE_RELASZ
                // Most likely this is DT_SCE_JMPREL (jmprelOff) + DT_SCE_PLTRELSZ (jmprelSz), haven't verified
                dynInfo.relaOff = dyn->d_un.d_ptr;            
                break;
            case DT_SCE_RELASZ:
                // number of relas that follow PLT relas
                dynInfo.relaSz = dyn->d_un.d_val;
                break;
            case DT_SCE_RELAENT:
                // size of rela entries (0x18)
                assert(dyn->d_un.d_val == 0x18);
                dynInfo.relaEntSz = dyn->d_un.d_val;
                break;
            case DT_SCE_STRTAB:
                // Contains hashed sym names
                // Convert to DT_STRTAB
                // Unhash all known strings, put in .dynstr, map dynent to this
                dynInfo.strtabOff = dyn->d_un.d_val;
                break;
            case DT_SCE_STRSZ:
                // Convert to DT_STRSZ based on DT_SCE_STRTAB conversion
                dynInfo.strtabSz = dyn->d_un.d_val;
                break;
            case DT_SCE_SYMTAB:
                // Convert to DT_SYMTAB
                // find new string offsets based on DT_SCE_STRTAB
                // Make new SHT_DYNSYM section, link to .dynsym/whatever DT_SCE_STRTAB maps to
                // TODO, look at sh_info requirement for that section
                // Special Sections at https://docs.oracle.com/cd/E19683-01/817-3677/6mj8mbtc9/index.html#chapter6-79797
                dynInfo.symtabOff = dyn->d_un.d_val;
                break;
            case DT_SCE_SYMENT:
                // size of syments (0x18)
                assert(dyn->d_un.d_val == 0x18);
                dynInfo.symtabEntSz = dyn->d_un.d_val;
                break;                                                
            case DT_SCE_HASHSZ:
                // size of sym hashtable, ignore for now
                dynInfo.hashSz = dyn->d_un.d_val;
                break;
            case DT_SCE_SYMTABSZ:
                // write size of converted symtab (might be same sz)
                dynInfo.symtabSz = dyn->d_un.d_val;
                break;

            case DT_SCE_EXPORT_LIB_ATTR:
            case DT_SCE_EXPORT_LIB:
            case DT_SCE_FINGERPRINT:
            case DT_SCE_ORIGINAL_FILENAME:
                break;

            default:
            {
                std::string name = to_string(tag);
                printf("Warning: unhandled DynEnt %s\n", name.c_str());
                break;
            }
        }
    }
    
    Elf64_Phdr *relroHeader = &progHdrs[1];
    uint64_t relroVA = relroHeader->p_vaddr;
    uint64_t relroFileOff = relroHeader->p_offset;
    assert(relroVA <= dynInfo.pltgotAddr && dynInfo.pltgotAddr <= relroVA + relroHeader->p_filesz);

    sMap.gotpltIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".got.plt"),
        .sh_type = SHT_PROGBITS,
        .sh_flags = SHF_ALLOC | SHF_WRITE,
        .sh_addr = dynInfo.pltgotAddr,
        .sh_offset = relroFileOff + (dynInfo.pltgotAddr - relroVA),
        // TODO try (pltrelsz / relaentSz) x8, x16
        // In example ELF, entsize is 8
        // Can probably be conservative (too large)
        // dunno if .got.plt is just array of slots, or if theres metadata
        // should try to find out using example .got.plt
        .sh_size = (dynInfo.pltrelsz / dynInfo.relaEntSz) * 8,
        .sh_addralign = 8,
        .sh_entsize = 8,
    };
    sections.emplace_back(sHdr);

    sMap.relaIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".rela"),
        .sh_type = SHT_RELA,
        .sh_flags = SHF_ALLOC,
        .sh_link = sMap.dynsymIdx,
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
        .sh_entsize = dynInfo.relaEntSz
    };
    sections.emplace_back(sHdr);

    assert(sMap.gotpltIdx);
    sMap.jmprelIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".rela.plt"), // .rela.plt?
        .sh_type = SHT_RELA,
        .sh_flags = SHF_ALLOC | SHF_INFO_LINK,
        .sh_link = sMap.dynsymIdx,
        .sh_info = sMap.gotpltIdx, // In sample ELF, I see info for .rela.plt points to .got.plt
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
        .sh_entsize = dynInfo.relaEntSz,
    };
    sections.emplace_back(sHdr);    

    if ( !fixDynlibData(elf, progHdrs, dynInfo, sections, sMap, newDynEnts, dependencies)) {
        return false;
    };

    // End DynEnt array
    newDynEnts.push_back({
        DT_NULL,
        {0}
    });

    // Write dynents to elf
    writePadding(elf, PGSZ);
    uint64_t segmentFileOff = ftell(elf);
    assert(1 == fwrite(newDynEnts.data(), newDynEnts.size() * sizeof(Elf64_Dyn), 1, elf));

    Elf64_Phdr newDynPhdr = DynPhdr;
    newDynPhdr.p_align = PGSZ;
    newDynPhdr.p_filesz = newDynEnts.size() * sizeof(Elf64_Dyn);
    // calculate p_vaddr, p_paddr
    rebaseSegment(&newDynPhdr, progHdrs);
    newDynPhdr.p_flags = PF_X | PF_W | PF_R;
    newDynPhdr.p_offset = segmentFileOff;
    progHdrs.push_back(newDynPhdr);

    progHdrs[oldDynamicPIdx].p_type = PT_NULL;

    // Need to create load command for the DYNAMIC segment.
    // Just use the same ranges
    Elf64_Phdr loadPhdrForDynSegment = newDynPhdr;
    loadPhdrForDynSegment.p_type = PT_LOAD;
    progHdrs.push_back(loadPhdrForDynSegment);

    sMap.dynamicIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".dynamic"),
        .sh_type = SHT_DYNAMIC,
        .sh_flags = SHF_WRITE | SHF_ALLOC,
        .sh_addr = newDynPhdr.p_vaddr,
        .sh_offset = newDynPhdr.p_offset,
        .sh_size = newDynPhdr.p_filesz,
        .sh_link = sMap.dynstrIdx,
        .sh_addralign = newDynPhdr.p_align,
        .sh_entsize = sizeof(Elf64_Dyn),
    };
    sections.emplace_back(sHdr);



    // TODO add sections for gotplt, relro, other sections that aren't changing location but are needed

    // TODO place extra dynamic ents with tags
    // DT_RELA
    // etc
    // Add sections used in relocs, loading dynamic dependencies, etc
    // These are sections we've moved out of the original dynlibdata segment

    //Elf64_Dyn dyn;



    return true;
}

static bool isLoadableSegType(Elf64_Word p_type) {
    switch (p_type) {
        case PT_LOAD:
        //case PT_GNU_RELRO:
        //case PT_DYNAMIC:
        //case PT_TLS:
            return true;

        default:
            return false;
    }
}

static void finalizeProgramHeaders(std::vector<Elf64_Phdr> &progHdrs) {
    //printf("before:\n");
    //for (Elf64_Phdr &pHdr: progHdrs) {
        //printf("\t%s\n", to_string((ProgramSegmentType) pHdr.p_type).c_str());
    //}

    // TODO look into which SCE segments need to be kept or converted to PT_LOAD

    // Convert certain SCE Segments
    for (Elf64_Phdr &pHdr: progHdrs) {
        switch(pHdr.p_type) {
            case PT_SCE_RELRO:	
                //pHdr.p_type = PT_GNU_RELRO;
                pHdr.p_type = PT_LOAD;
                break;
            case PT_GNU_EH_FRAME:
            case PT_SCE_PROCPARAM:	
            case PT_SCE_MODULEPARAM:
            case PT_SCE_LIBVERSION:
            case PT_SCE_RELA:	
            case PT_SCE_COMMENT:
                /*
                if (pHdr.p_memsz) {
                    pHdr.p_type = PT_LOAD;
                } else {
                    pHdr.p_type = PT_NULL;
                }*/
                break;
            default:
                break;
        }
    }

    // Get rid of  certain Segments
    std::vector<Elf64_Phdr> finalSegments;
    for (Elf64_Phdr &pHdr: progHdrs) {
        switch(pHdr.p_type) {
            case PT_NULL:
            case PT_EMU_IGNORE:
            //case PT_GNU_EH_FRAME:
            case PT_SCE_PROCPARAM:	
            case PT_SCE_MODULEPARAM:
            case PT_SCE_LIBVERSION:
            case PT_SCE_DYNLIBDATA:
            case PT_SCE_COMMENT:
                break;
            default:
                finalSegments.push_back(pHdr);
        }
    }
    progHdrs = finalSegments;

    for (Elf64_Phdr &pHdr: progHdrs) {
        pHdr.p_flags = PF_X | PF_W | PF_R;
    }

    std::stable_sort(progHdrs.begin(), progHdrs.end(), [](const Elf64_Phdr& left, const Elf64_Phdr& right){
        if (isLoadableSegType(left.p_type) && isLoadableSegType(right.p_type)) {
            if (left.p_vaddr == right.p_vaddr) {
                return left.p_memsz >= right.p_memsz;
            }
            return left.p_vaddr <= right.p_vaddr;
        }
        return isLoadableSegType(left.p_type);
    });

    //printf("final headers:\n");
    //for (Elf64_Phdr &pHdr: progHdrs) {
        //printf("\t%s\n", to_string((ProgramSegmentType) pHdr.p_type).c_str());
    //}
}

bool patchPs4Lib(std::string nativePath, std::set<std::string> &dependencies) {
    std::vector<Elf64_Phdr> progHdrs;
    // in a Ps4 module, these Dyn Ents are in the DYNAMIC segment/.dynamic section,
    // and describe the DYNLIBDATA segment
    std::vector<Elf64_Dyn> oldPs4DynEnts;    
    std::vector<Elf64_Dyn> newElfDynEnts;
    // index of the strtab in the shEntries

    g_DebugContext.currentPs4Lib = nativePath;

    FILE *f = fopen(nativePath.c_str(), "r+");
    if (!f) {
        fprintf(stderr, "couldn't open %s\n", nativePath.c_str());
        return false;
    }

    Elf64_Ehdr elfHdr;
    fseek(f, 0, SEEK_SET);
    assert (1 == fread(&elfHdr, sizeof(elfHdr), 1, f));

    elfHdr.e_ident[EI_OSABI] = ELFOSABI_SYSV;
    elfHdr.e_type = ET_DYN;

    Elf64_Shdr sHdr;

    progHdrs.resize(elfHdr.e_phnum);
    fseek(f, elfHdr.e_phoff, SEEK_SET);
    assert(elfHdr.e_phnum == fread(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    // Section headers accumulate here
    std::vector<Section> sections;
    SectionMap sMap;

    sHdr = {
        .sh_name = 0,
        .sh_type = SHT_NULL
    };
    sections.emplace_back(sHdr);

    sMap.shstrtabIdx = sections.size();
    sHdr = {
        .sh_type = SHT_STRTAB,
        .sh_flags = 0,
        .sh_addralign = 1,
    };
    sections.emplace_back(sHdr);
    appendToStrtab(sections[sMap.shstrtabIdx], "\0");
    sections[sMap.shstrtabIdx].setName(appendToStrtab(sections[sMap.shstrtabIdx], ".shstrtab"));

    // Patch dynamic segment with new dynents
    // Copy parts of PT_SCE_DYNLIBDATA to new Segment, such as unhashed strings
    // Fix up syms, relas, strings
    // Crate relevant sections for dlopen
    if ( !fixDynamicInfoForLinker(f, progHdrs, sections, sMap, dependencies)) {
        return false;
    }
    // Change OS specific type
    // I don't think this should be loaded currently. p_vaddr and p_memsz are 0
    // I think plt/got are in different section which is already PT_LOAD
    //progHdrs[findPhdr(progHdrs, PT_SCE_DYNLIBDATA)].p_type = PT_NULL;

    // For now, pack the extra sections into their own segment
    // in order to append them. The data is all that matters.
    // Want to make sure they own their own parts of the ELF and
    // future segments see that their region is owned when using rebaseSegment
    std::vector<uint> extraSections {
        sMap.shstrtabIdx
    };
    Elf64_Phdr extraSegmentHdr {
        .p_type = PT_EMU_IGNORE,
        .p_flags = PF_R | PF_W | PF_X,
        .p_align = static_cast<Elf64_Xword>(PGSZ)
    };
    rebaseSegment(&extraSegmentHdr, progHdrs);
    fseek(f, 0, SEEK_END);
    writePadding(f, extraSegmentHdr.p_align, true);
    extraSegmentHdr.p_offset = ftell(f);
    Segment extraSegment = CreateSegment(extraSegmentHdr, sections, extraSections);
    assert(1 == fwrite(extraSegment.contents.data(), extraSegment.contents.size(), 1, f));
    progHdrs.push_back(extraSegment.pHdr);

    finalizeProgramHeaders(progHdrs);

    // write program headers
    fseek(f, 0, SEEK_END);
    writePadding(f, PGSZ, true);
    elfHdr.e_phoff = ftell(f);
    elfHdr.e_phnum = progHdrs.size();
    assert(elfHdr.e_phnum == fwrite(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    // write section headers
    fseek(f, 0, SEEK_END);
    writePadding(f, PGSZ, true);
    elfHdr.e_shoff = ftell(f);
    elfHdr.e_shstrndx = sMap.shstrtabIdx;
    std::vector<Elf64_Shdr> sectionHeaders;
    for (Section &section: sections) {
        sectionHeaders.push_back(section.hdr);
    }
    elfHdr.e_shnum = sectionHeaders.size();
    elfHdr.e_shentsize = sizeof(Elf64_Shdr);
    assert(elfHdr.e_shnum == fwrite(sectionHeaders.data(), sizeof(Elf64_Shdr), elfHdr.e_shnum, f));

    // write elf header, now referring to new program and section headers
    fseek(f, 0, SEEK_SET);
    assert(1 == fwrite(&elfHdr, sizeof(Elf64_Ehdr), 1, f));

    fsync(fileno(f));
    fclose(f);

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

    g_DebugContext.init("test.log");
    LOGFILE("main: start\n");

    if (!g_CmdArgs.preloadDirPath.empty()) {
        g_Preloads.init(g_CmdArgs.preloadDirPath);
    }

    TheLibrarySearcher = LibSearcher({{g_CmdArgs.pkgdumpPath, true}, {g_CmdArgs.ps4libsPath, false}});

    if ( !getenv("LD_LIBRARY_PATH")) {
        fprintf(stderr, "LD_LIBRARY_PATH unset\n");
        return -1;
    }

    std::error_code ec;
    fs::create_directory(g_CmdArgs.nativeElfOutputDir, ".", ec);
    if (ec) {
        std::string m = ec.message();
        fprintf(stderr, "Failed to create directory %s: %s\n", g_CmdArgs.nativeElfOutputDir.c_str(), m.c_str());
        return -1;
    }    

    std::set<std::string> dependencies = { g_CmdArgs.dlPath };
    std::set<std::string> satisfied;
    while (!dependencies.empty()) {
        auto it = dependencies.begin();
        fs::path libName = *it;
        dependencies.erase(it);

        auto oLibPath = findPathToLibName(libName);
        if ( !oLibPath) {
            fprintf(stderr, "Warning: unable to locate library %s\n", libName.c_str());
            continue;
        }

        fs::path libPath = oLibPath.value();

        fs::path nativePath = g_CmdArgs.nativeElfOutputDir; 
        nativePath /= getNativeLibPath(libPath);

        struct stat buf;
        bool exists = !stat(nativePath.c_str(), &buf);
        if (g_CmdArgs.purgeElfs || !exists) {
            std::string cmd = "cp " + libPath.string() + " " + nativePath.string();
            system(cmd.c_str());

            if (chmod(nativePath.c_str(), S_IRWXU | (S_IRGRP | S_IXGRP) | (S_IROTH | S_IXOTH))) {
                fprintf(stderr, "chmod failed\n");
                return -1;
            }

            if ( !patchPs4Lib(nativePath, dependencies)) {
                return -1;
            }
        }
        satisfied.insert(libName.filename());
        for (auto &sat: satisfied) {
            dependencies.erase(sat);
        }
    }

    LOGFILE("main: before dlopen\n");

    fs::path firstLib = g_CmdArgs.nativeElfOutputDir;
    firstLib /= getNativeLibPath(g_CmdArgs.dlPath);

    fs::path entryModule = g_CmdArgs.nativeElfOutputDir;
    entryModule /= getNativeLibPath("eboot.bin");
    //if ( !mapEntryModuleIntoMemory(entryModule)) {
    //    return -1;
    //}

    // Load ps4 module
   // dl = dlmopen(preloadNamespace, firstLib.c_str(), RTLD_LAZY);
    dl = dlopen(firstLib.c_str(), RTLD_LAZY);
    if (!dl) {
        char *err;
        while ((err = dlerror())) {
            printf("(dlopen) %s\n", err);
        }
        return -1;
    }

    dlclose(dl);
    printf("main: done\n");

    return 0;
}