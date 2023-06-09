#include "Elf/elf-sce.h"
#include <algorithm>
#include <elf.h>
#include <elfpatcher/elfpatcher.h>
#include <fcntl.h>
#include <vector>
#include <sys/stat.h>
#include <sqlite3.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <boost/iostreams/device/mapped_file.hpp>

std::optional<fs::path> LibSearcher::findLibrary(fs::path name) {
    for (PathElt &elt: m_paths) {
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

std::optional<fs::path> LibSearcher::findLibrary(fs::path stem, std::vector<fs::path> validExts) {
    for (auto &ext: validExts) {
        fs::path f = stem;
        f += ext;
        if (auto ret = findLibrary(f)) {
            return ret;
        }
    }
    return std::nullopt;
}

fs::path getNativeLibName(fs::path ps4LibName) {
    assert(ps4LibName.has_filename());

    auto ext = ps4LibName.extension();
    if (ext == ".sprx") {
        ext = ".prx";
    }
    
    if (ext != ".so") {
        ext += ".so";
    }

    auto libStem = ps4LibName.stem();
    libStem += ext;

    return libStem;
}

fs::path getPs4LibName(fs::path nativeLibName) {
    assert(nativeLibName.extension() == ".so");
    fs::path ps4Name = nativeLibName.stem();
    return ps4Name;
}

std::optional<fs::path> findPathToSceLib(fs::path libName, ElfPatcherContext &Ctx) {
    if (!libName.has_extension()) {
        fprintf(stderr, "Warning, findPathToLibName: library %s has no extension\n", libName.c_str());
    }

    fs::path ext = libName.extension();
    std::vector<fs::path> validExts = { ext };
    if (ext == ".prx") {
        validExts.push_back(".sprx");
    } else if (ext == ".sprx") {
        validExts.push_back(".prx");
    }

    fs::path libStem = libName.stem();

    return Ctx.ps4LibSearcher.findLibrary(libStem, validExts);
}

static bool getFileHash(fs::path path, uint64_t &hash) {
    boost::iostreams::mapped_file_source file;
    try {
        file.open(path);
    } catch (const std::ios_base::failure& e) {
        fprintf(stderr, "Couldn't open %s: %s\n", path.c_str(), e.what());
        return false;
    }

    hash = pjwHash((unsigned char *) file.data(), file.size());
    return true;
}

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

static bool reverseKnownHashes(ElfPatcherContext &Ctx, std::vector<const char *> &oldStrings, std::vector<std::string> &newStrings) {
    sqlite3 *db = openHashDb(Ctx.hashdbPath.c_str());
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

    // Prepend _ps4__ for reversed strings, _hash__ for hashes we couldn't reverse 
    for (const char *old: oldStrings) {
        int libIdx, modIdx;
        if (isHashedSymbol(old, libIdx, modIdx)) {
            // truncate #A#B lib/module id's suffix
            std::string hash(old, 11);
            const auto &it = hashToSymbol.find(hash);
            // Use a prefix "_ps4__" so the symbols in sce libraries don't collide with the host's dynamic libraries, like
            // libc.
            // So the hosts dynamic linker will treat "_ps4__printf" different than "printf" used in emulator code             
            if (it != hashToSymbol.end()) {
                std::string symbol = "_ps4__" + it->second; 
                newStrings.push_back(symbol);
            } else {
                std::string symbol = "_hash__" + hash; 
                newStrings.push_back(symbol);
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
static bool fixDynlibData(ElfPatcherContext &Ctx, FILE *elf, std::vector<Elf64_Phdr> &progHdrs, DynamicTableInfo &dynInfo, std::vector<Section> &sections, SectionMap &sMap, std::vector<Elf64_Dyn> &newDynEnts) {
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
    Elf64_Sym *dynsymSyms = reinterpret_cast<Elf64_Sym *>(&dynlibContents[dynInfo.symtabOff]);
    for (uint i = 0; i < numSyms; i++) {
        Elf64_Sym ent = dynsymSyms[i];
        if (ent.st_name) {
            oldSymStrings.push_back((const char *) &dynlibContents[dynInfo.strtabOff+ ent.st_name]);
        } else {
            oldSymStrings.push_back("");
        }
    }

    std::vector<std::string> newStrings;
    if (!Ctx.hashdbPath.empty()) {
        if ( !reverseKnownHashes(Ctx, oldSymStrings, newStrings)) {
            return false;
        }
    }

    int firstNonLocalIdx = -1;
    Section &dynsym = sections[sMap.dynsymIdx];
    for (uint i = 0; i < numSyms; i++) {
        Elf64_Sym ent = dynsymSyms[i];
        if (ent.st_name) {
            ent.st_name = appendToStrtab(sections[sMap.dynstrIdx], newStrings[i].c_str());
        }
        // Even though it shouldn't affect dynamic linking, it seems to mess with the debugger
        // Also 0 (SHT_NULL) does actually mess with dlopen()
        if (ent.st_shndx >= SHN_LORESERVE && ent.st_shndx <= SHN_LORESERVE) {
            if (false) {
                printf("Symbol has reserved symbol shndx\n");
            }
        } else if (ent.st_shndx != SHN_UNDEF) {
            // GDB needs shndx to point to .text to show function names correctly
            ent.st_shndx = sMap.textIdx;
        }

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
        if ( findPathToSceLib(ps4Name, Ctx)) {
            Ctx.deps.push_back(ps4Name);
        } else {
            fprintf(stderr, "Warning: couldn't find dependency %s needed by %s. Skipping\n", ps4Name.c_str(), Ctx.currentPs4Lib.c_str());
            continue;
        }

        fs::path nativeName = getNativeLibName(ps4Name);

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

    // Copy dynsym to symtab and dynstr to strtab for GDB
    Elf64_Shdr sHdr;
    sMap.strtabIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".strtab"),
        .sh_type = SHT_STRTAB,
        .sh_flags = SHF_STRINGS,
        .sh_link = 0,
        .sh_info = 0,
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
        .sh_entsize = 0
    };
    sections.emplace_back(sHdr);
    sections[sMap.strtabIdx].contents = sections[sMap.dynstrIdx].contents;

    sMap.symtabIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".symtab"),
        .sh_type = SHT_SYMTAB,
        .sh_flags = 0,
        .sh_link = sMap.strtabIdx,
        .sh_info = sections[sMap.dynsymIdx].getInfo(),
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
        .sh_entsize = sizeof(Elf64_Sym),
    };
    sections.emplace_back(sHdr);
    sections[sMap.symtabIdx].contents = sections[sMap.dynsymIdx].contents;

    Section &symtab = sections[sMap.symtabIdx];
    Elf64_Sym *symtabSyms = reinterpret_cast<Elf64_Sym *>(symtab.contents.data());
    Elf64_Addr textBaseAddr = sections[sMap.textIdx].getAddr();
    // Change ents in .symtab to have values as offsets relative to .text.
    // I think GDB expects this
    for (uint i = 0; i < symtab.contents.size() / sizeof(Elf64_Sym); i++) {
        Elf64_Sym *sym = &symtabSyms[i];
        if (sym->st_value > 0) {
            sym->st_value -= textBaseAddr;
        }
    }

    // Create dynlibdata segment
    std::vector<uint> dynlibDataSections = {
        sMap.dynstrIdx,
        sMap.dynsymIdx,
        sMap.hashIdx,
        sMap.jmprelIdx,
        sMap.relaIdx,
        sMap.strtabIdx, // strtab and symtab don't need VA addresses, so they could go somewhere else
        sMap.symtabIdx
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
        {sections[sMap.hashIdx].getAddr()}
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

// Address should have corresponding contents in the ELF
static Elf64_Off findFileOffForAddr(std::vector<Elf64_Phdr> &progHdrs, Elf64_Addr addr) {
    Elf64_Phdr *containing = nullptr;
    for (uint i = 0; i < progHdrs.size(); i++) {
        Elf64_Phdr *phdr = &progHdrs[i];
        if (phdr->p_vaddr <= addr 
                    && addr < phdr->p_vaddr + phdr->p_filesz) {
            containing = phdr;
            break;
        }
    }

    assert(containing);

    return containing->p_offset + (addr - containing->p_vaddr);
}

// write new Segment for PT_DYNAMIC based on SCE dynamic entries
// append to end of file
// modify the progHdrs array
// add new dynlibdata sections and update section map
static bool fixDynamicInfoForLinker(ElfPatcherContext &Ctx, FILE *elf, std::vector<Elf64_Phdr> &progHdrs, std::vector<Section> &sections, SectionMap &sMap) {
    uint oldDynamicPIdx = findPhdr(progHdrs, PT_DYNAMIC);
    Elf64_Phdr DynPhdr = progHdrs[oldDynamicPIdx];
    Elf64_Shdr sHdr;
    std::vector<Elf64_Dyn> newDynEnts;
    DynamicTableInfo dynInfo;

    sMap.dynstrIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".dynstr"),
        .sh_type = SHT_STRTAB,
        .sh_flags = SHF_STRINGS | SHF_ALLOC,
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
        .sh_entsize = 0,
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
                dynInfo.neededLibs.push_back(dyn->d_un.d_val);
                break;
            case DT_SONAME:
                break;

            case DT_REL:
            case DT_RELSZ:
            case DT_RELENT:
            {
                std::string name = to_string(tag);
                printf("Warning: unhandled DynEnt %s\n", name.c_str());
                break;
            }
                

            // https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter2-55859.html#chapter2-48195
            case DT_PREINIT_ARRAY:
                Ctx.initFiniInfo.preinit_array_base = dyn->d_un.d_ptr;
                break;
            case DT_PREINIT_ARRAYSZ:
                Ctx.initFiniInfo.dt_preinit_array.resize(dyn->d_un.d_val / sizeof(Elf64_Addr));
                break;
            case DT_INIT:
                Ctx.initFiniInfo.dt_init = dyn->d_un.d_ptr;
                break;
            case DT_INIT_ARRAY:
                Ctx.initFiniInfo.init_array_base = dyn->d_un.d_ptr;
                break;
            case DT_INIT_ARRAYSZ:
                Ctx.initFiniInfo.dt_init_array.resize(dyn->d_un.d_val / sizeof(Elf64_Addr));
                break;
            case DT_FINI:
                Ctx.initFiniInfo.dt_fini = dyn->d_un.d_ptr;
                break;
            case DT_FINI_ARRAY:
                Ctx.initFiniInfo.fini_array_base = dyn->d_un.d_ptr;
                break;
            case DT_FINI_ARRAYSZ:
                Ctx.initFiniInfo.dt_fini_array.resize(dyn->d_un.d_val / sizeof(Elf64_Addr));
                break;
            // Don't add dynents for these, since they take different parameters

            case DT_RPATH:
            case DT_SYMBOLIC:
            case DT_DEBUG:
            case DT_TEXTREL:
            case DT_BIND_NOW:
            case DT_RUNPATH:
            case DT_FLAGS:
                //https://docs.oracle.com/cd/E23824_01/html/819-0690/chapter6-42444.html#chapter7-tbl-5            
                // should be DF_TEXTREL
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
                //uint64_t low = dyn->d_un.d_val & 0xffffffff;
                assert(upp % 0x10000 == 0);
                //assert(low == 0x9);
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

    // Extract init, fini info from dynents
    if ( !Ctx.initFiniInfo.dt_preinit_array.empty()) {
        fseek(elf, findFileOffForAddr(progHdrs, Ctx.initFiniInfo.preinit_array_base.value()), SEEK_SET);
        fread(Ctx.initFiniInfo.dt_preinit_array.data(), sizeof(Elf64_Addr), Ctx.initFiniInfo.dt_preinit_array.size(), elf);
    }
    if ( !Ctx.initFiniInfo.dt_init_array.empty()) {
        fseek(elf, findFileOffForAddr(progHdrs, Ctx.initFiniInfo.init_array_base.value()), SEEK_SET);
        fread(Ctx.initFiniInfo.dt_init_array.data(), sizeof(Elf64_Addr), Ctx.initFiniInfo.dt_init_array.size(), elf);
    }
    if ( !Ctx.initFiniInfo.dt_fini_array.empty()) {
        fseek(elf, findFileOffForAddr(progHdrs, Ctx.initFiniInfo.fini_array_base.value()), SEEK_SET);
        fread(Ctx.initFiniInfo.dt_fini_array.data(), sizeof(Elf64_Addr), Ctx.initFiniInfo.dt_fini_array.size(), elf);
    }    

    //
    
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

    if ( !fixDynlibData(Ctx, elf, progHdrs, dynInfo, sections, sMap, newDynEnts)) {
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

    return true;
}

static void finalizeProgramHeaders(std::vector<Elf64_Phdr> &progHdrs) {
    for (Elf64_Phdr &pHdr: progHdrs) {
        switch(pHdr.p_type) {
            case PT_SCE_RELRO:	
                //pHdr.p_type = PT_GNU_RELRO;
                // TODO need this?
                pHdr.p_type = PT_LOAD;
                break;
            case PT_SCE_PROCPARAM:
                // TODO convert to LOAD?
            case PT_GNU_EH_FRAME:
            case PT_SCE_MODULEPARAM:
            case PT_SCE_LIBVERSION:
            case PT_SCE_RELA:	
            case PT_SCE_COMMENT:
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

    for (Elf64_Phdr &pHdr: finalSegments) {
        // TODO set permissions per segment, don't do this
        pHdr.p_flags = PF_X | PF_W | PF_R;
    }
    progHdrs = finalSegments;
}

bool patchPs4Lib(ElfPatcherContext &Ctx, const std::string elfPath) {
    std::vector<Elf64_Phdr> progHdrs;
    // in a Ps4 module, these Dyn Ents are in the DYNAMIC segment/.dynamic section,
    // and describe the DYNLIBDATA segment
    std::vector<Elf64_Dyn> oldPs4DynEnts;
    std::vector<Elf64_Dyn> newElfDynEnts;
    // index of the strtab in the shEntries
    PatchedElfInfo elfInfo;

    Ctx.currentPs4Lib = elfPath;

    FILE *f = fopen(elfPath.c_str(), "r+");
    if (!f) {
        fprintf(stderr, "couldn't open %s\n", elfPath.c_str());
        return false;
    }

    Elf64_Ehdr elfHdr;
    fseek(f, 0, SEEK_SET);
    assert (1 == fread(&elfHdr, sizeof(elfHdr), 1, f));

    assert(sizeof(Elf64_Phdr) == elfHdr.e_phentsize);
    // assert(sizeof(Elf64_Shdr) == elfHdr.e_shentsize); // this is 0

    elfHdr.e_ident[EI_OSABI] = ELFOSABI_SYSV;

    elfInfo.elfType = (EtType) elfHdr.e_type;
    switch(elfHdr.e_type) {
        case ET_SCE_EXEC:
            elfHdr.e_type = ET_EXEC;
            break;
        case ET_SCE_DYNAMIC:
        // Pretty sure DYNEXEC is PIE
        case ET_SCE_DYNEXEC:
            // technically eboot.bin (executable) should become ET_EXEC.
            // DF_1_PIE in DT_FLAGS_1 should indicate whether an EXEC is PIE.
            // We will dlopen eboot.bin from the Emu process (for simplicity), so just mark it
            // ET_DYN here
            // (if some game's exe isn't PIE, then we won't be able to do this, and will need to
            // manually do relocations and map its segments to the exact addresses)
            elfHdr.e_type = ET_DYN;
            break;
    }

    Elf64_Shdr sHdr;

    progHdrs.resize(elfHdr.e_phnum);
    fseek(f, elfHdr.e_phoff, SEEK_SET);
    assert(elfHdr.e_phnum == fread(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    for (unsigned int i = 0; i < elfHdr.e_phnum; i++) {
        Elf64_Phdr *phdr = &progHdrs[i];
        switch (phdr->p_type) {
            case PT_SCE_PROCPARAM:
                elfInfo.procParam = phdr->p_vaddr;
                break;
            default:
                break;
        }
    }

    // Section headers accumulate here.
    // Ps4 elfs seem to have their section table stripped, so create them so dlopen can work.
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

    Elf64_Phdr textSegment = progHdrs[0];
    Elf64_Phdr dataSegment = progHdrs[2];
    // Add .text, .data sections so hopefully gdb doesnt bug out
    // Probably not accurate, bigger than in actuality
    sMap.textIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".text"),
        .sh_type = SHT_PROGBITS,
        .sh_flags = SHF_ALLOC | SHF_EXECINSTR,
        .sh_addr = textSegment.p_vaddr,
        .sh_offset = textSegment.p_offset,
        .sh_size = textSegment.p_filesz,
        .sh_addralign = 16
    };
    sections.emplace_back(sHdr);

    sHdr = {
        .sh_name = appendToStrtab(sections[sMap.shstrtabIdx], ".data"),
        .sh_type = SHT_PROGBITS,
        .sh_flags = SHF_ALLOC | SHF_WRITE,
        .sh_addr = dataSegment.p_vaddr,
        .sh_offset = dataSegment.p_offset,
        .sh_size = dataSegment.p_filesz,
        .sh_addralign = 8
    };
    sections.emplace_back(sHdr);

    // Patch dynamic segment with new dynents
    // Copy parts of PT_SCE_DYNLIBDATA to new Segment, such as unhashed strings
    // Fix up syms, relas, strings
    // Crate relevant sections for dlopen
    if ( !fixDynamicInfoForLinker(Ctx, f, progHdrs, sections, sMap)) {
        return false;
    }

    // Reserve space + fixup offset info for shstrtab section, with CreateSegment.
    // The segment will be deleted during finalizeProgramHeaders.
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

    // dump info about lib to json (including addresses of init/fini functions)
    elfInfo.path = elfPath;

    if ( !getFileHash(elfPath, elfInfo.hash)) {
        return false;
    }
    elfInfo.initFiniInfo = Ctx.initFiniInfo;
    // et type set above

    fs::path jsonPath = elfPath;
    jsonPath += ".json";
    if ( !dumpPatchedElfInfoToJson(jsonPath, elfInfo)) {
        return false;
    }

    return true;
}

bool findDependencies(fs::path patchedElf, std::vector<std::string> &deps) {
    const Elf64_Ehdr *elfHdr;
    const Elf64_Shdr *sHdrs;

    deps.clear();

    boost::iostreams::mapped_file_source file;
    try {
        file.open(patchedElf.c_str());
    } catch (const std::ios_base::failure& e) {
        fprintf(stderr, "Couldn't open %s: %s\n", patchedElf.c_str(), e.what());
        return false;
    }

    const unsigned char *data = (const unsigned char *) file.data();
    elfHdr = reinterpret_cast<const Elf64_Ehdr *>(data);
    sHdrs = reinterpret_cast<const Elf64_Shdr *>(data + elfHdr->e_shoff);

    const Elf64_Shdr *shstrtab = &sHdrs[elfHdr->e_shstrndx];
    const Elf64_Shdr *dynstr = nullptr;
    const Elf64_Shdr *dynamic = nullptr;

    const char *shStrtabData = reinterpret_cast<const char *>(data + shstrtab->sh_offset);
    for (uint i = 0; i < elfHdr->e_shnum; i++) {
        const Elf64_Shdr *curSection = &sHdrs[i];
        const char *shName = &shStrtabData[curSection->sh_name];
        if ( !strcmp(".dynstr", shName)) {
            dynstr = curSection;
        } else if (!strcmp(".dynamic", shName)) {
            dynamic = curSection;
        }
    }

    if ( !dynstr) {
        fprintf(stderr ,"Couldn't find .dynstr section in %s\n", patchedElf.c_str());
        return false;
    }
    if ( !dynamic) {
        fprintf(stderr ,"Couldn't find .dynamic section in %s\n", patchedElf.c_str());
        return false;
    }

    const char *dynstrData = reinterpret_cast<const char *>(data + dynstr->sh_offset);
    const Elf64_Dyn *dyn = reinterpret_cast<const Elf64_Dyn *>(data + dynamic->sh_offset);

    for (uint i = 0; i < dynamic->sh_size / dynamic->sh_entsize; i++) {
        if (dyn[i].d_tag == DT_NULL) {
            break;
        } else if (dyn[i].d_tag == DT_NEEDED) {
            const char *libName = &dynstrData[dyn[i].d_un.d_val];
            deps.emplace_back(libName);
        }
    }

    return true;
}

// Trying to get ghidra debugger to work
// This doesn't completely work yet. Seeing jumps to bad addresses. Jump slot solution might
// not be right.
bool rebasePieElf(fs::path patchedElf, Elf64_Addr baseVA) {
    const Elf64_Ehdr *elfHdr;
    Elf64_Phdr *pHdrs;
    Elf64_Shdr *sHdrs;
    Elf64_Off vaDiff; 

    // Rebase a ELF_DYN Elf so the lowest base VA corresponding to file contents (ie the text section)
    // begins at address baseVA

    // Can think of these things I need to do:
    //  -Add baseVA diff to segments with PT_LOAD
    //  -Add baseVA diff to sections with SHF_ALLOC
    //  -Add baseVA diff to symtab/dynsym entries
    //  -Add baseVA diff to relocations' r_offset
    //  -Add baseVA diff to some dynamic ents
    // (baseVA diff means difference between old baseVA and new baseVA)

    boost::iostreams::mapped_file file;
    try {
        file.open(patchedElf.c_str());
    } catch (const std::ios_base::failure& e) {
        fprintf(stderr, "Couldn't open %s: %s\n", patchedElf.c_str(), e.what());
        return false;
    }

    unsigned char *data = reinterpret_cast<unsigned char *>(file.data());
    elfHdr = reinterpret_cast<const Elf64_Ehdr *>(data);
    pHdrs = reinterpret_cast<Elf64_Phdr *>(data + elfHdr->e_phoff);
    sHdrs = reinterpret_cast<Elf64_Shdr *>(data + elfHdr->e_shoff);
    if (elfHdr->e_type != ET_DYN) {
        return false;
    }

    Elf64_Addr oldBaseVA = std::numeric_limits<Elf64_Addr>::max();
    if (elfHdr->e_phnum <= 0) {
        return false;
    }

    {
        const Elf64_Phdr *ph = pHdrs;
        for (auto i = 0; i < elfHdr->e_phnum; i++) {
            if (ph->p_flags & PT_LOAD) {
                oldBaseVA = std::min(oldBaseVA, ph->p_vaddr);
            }
            ph = reinterpret_cast<const Elf64_Phdr *>(reinterpret_cast<const unsigned char *>(ph) + elfHdr->e_phentsize);
        }
    }

    vaDiff = baseVA - oldBaseVA;

    std::vector<Elf64_Shdr *> symtabs;
    std::vector<Elf64_Shdr *> relSections;
    std::vector<Elf64_Shdr *> relaSections;
    std::vector<Elf64_Shdr *> relrSections;
    Elf64_Shdr *dynSHdr = nullptr;
    Elf64_Shdr *pltgotSHdr = nullptr;

    Elf64_Off shstrabShOff = elfHdr->e_shstrndx * elfHdr->e_shentsize;
    const Elf64_Shdr *shstrtabShdr = reinterpret_cast<Elf64_Shdr *>(reinterpret_cast<unsigned char *>(sHdrs) + shstrabShOff);
    const char *shstrtab = reinterpret_cast<char *>(data + shstrtabShdr->sh_offset);

    {
        Elf64_Shdr *sh = sHdrs;
        for (auto i = 0; i < elfHdr->e_shnum; i++) {
            if (sh->sh_type == SHT_SYMTAB) {
                symtabs.push_back(sh);
            } else if (sh->sh_type == SHT_REL) {
                relSections.push_back(sh);
            } else if (sh->sh_type == SHT_RELA) {
                relaSections.push_back(sh);
            } else if (sh->sh_type == SHT_RELR) {
                relrSections.push_back(sh);
            } else if (sh->sh_type == SHT_DYNAMIC) {
                if (dynSHdr) {
                    fprintf(stderr, "Multiple .dynamic sections in file %s\n", patchedElf.c_str());
                    return false;
                }
                dynSHdr = sh;
            } else if ( !strcmp(&shstrtab[sh->sh_name], ".got.plt")) {
                if (pltgotSHdr) {
                    fprintf(stderr, "Multiple .got.plt sections in file %s\n", patchedElf.c_str());
                    return false;
                }                
                pltgotSHdr = sh;
            }
            sh = reinterpret_cast<Elf64_Shdr *>(reinterpret_cast<unsigned char *>(sh) + elfHdr->e_shentsize);
        }
    }

    if ( !dynSHdr) {
        fprintf(stderr, "Couldn't find .dynamic section in file %s\n", patchedElf.c_str());
        return false;
    }
    if ( !dynSHdr) {
        fprintf(stderr, "Couldn't find .got.plt section in file %s\n", patchedElf.c_str());
        return false;
    }    

    for (Elf64_Shdr *sh: symtabs) {
        Elf64_Sym *sym = reinterpret_cast<Elf64_Sym *>(data + sh->sh_offset);
        for (auto i = 0; i < sh->sh_size / sh->sh_entsize; i++) {
            if (sym->st_shndx != SHN_UNDEF && sym->st_shndx != SHN_ABS) {
                // SHN_UNDEF means symbol is undefined
                // SHN_ABS means the symbol is not a virtual address
                sym->st_value += vaDiff;
            }
            sym = reinterpret_cast<Elf64_Sym *>(reinterpret_cast<unsigned char *>(sym) + sh->sh_entsize);
        }
    }

    for (Elf64_Shdr *sh: relSections) {
        Elf64_Rel *rel = reinterpret_cast<Elf64_Rel *>(data + sh->sh_offset);
        for (auto i = 0; i < sh->sh_size / sh->sh_entsize; i++) {
            rel->r_offset += vaDiff;
            rel = reinterpret_cast<Elf64_Rel *>(reinterpret_cast<unsigned char *>(rel) + sh->sh_entsize);
        }
    }

    for (Elf64_Shdr *sh: relaSections) {
        Elf64_Rela *rela = reinterpret_cast<Elf64_Rela *>(data + sh->sh_offset);
        for (auto i = 0; i < sh->sh_size / sh->sh_entsize; i++) {
            if (ELF64_R_TYPE(rela->r_info) == R_X86_64_JUMP_SLOT) {
                Elf64_Off slot_offset = rela->r_offset - pltgotSHdr->sh_addr;
                Elf64_Addr *slot = reinterpret_cast<Elf64_Addr *>(data + pltgotSHdr->sh_offset + slot_offset);
                *slot += vaDiff;
                // Before the dynamic loader locates the function, the pltgot slot holds the address of the
                // dynamic linker stub which locates the function.
                // The file contents of the slot should be the address of the stub, relative to base VA 0x0.
                // The dynamic loader initializes the pltgot by adding the runtime load address of the library to each
                // slot to create the absolute address of the stub.
                // The stub happens to be the instruction after "jmp [SLOT_ADDR]" in rela.plt. ie:

                // 7FFFF64C4318: FF 25 7A BA 0D 00          jmpq   *0xdba7a(%rip)  <- indirect jump into pltgot slot
                // 7FFFF64C431E: 68 25 00 00 00             pushq  $0x25           <- slot initially points here with lazy binding.
                //                                                                    0x25 is probably the symbol index into .dynsym?
                //                                                                    doesn't seem to match, idk.
                // 7FFFF64C4323: E9 90 FD FF FF             jmp    0x7ffff64c40b8  <- symbol lookup handler
                // 7FFFF64C4328: FF 25 72 BA 0D 00          jmpq   *0xdba72(%rip)  <- start of next thunk
                // 7FFFF64C432E: 68 26 00 00 00             pushq  $0x26
                // 7FFFF64C4333: E9 80 FD FF FF             jmp    0x7ffff64c40b8
            }
            rela->r_offset += vaDiff;
            rela = reinterpret_cast<Elf64_Rela *>(reinterpret_cast<unsigned char *>(rela) + sh->sh_entsize);
        }
    }

    for (Elf64_Shdr *sh: relrSections) {
        Elf64_Relr *relr = reinterpret_cast<Elf64_Relr *>(data + sh->sh_offset);
        for (auto i = 0; i < sh->sh_size / sh->sh_entsize; i++) {
            // relr ents contain an address for the linker to use.
            // The linker adds the addend, stored at that address, to the module's load address in memory
            *relr += vaDiff;
            relr = reinterpret_cast<Elf64_Relr *>(reinterpret_cast<unsigned char *>(relr) + sh->sh_entsize);
        }
    }

    if (dynSHdr) {
        Elf64_Dyn *dyn = reinterpret_cast<Elf64_Dyn *>(data + dynSHdr->sh_offset);
        for (auto i = 0; i < dynSHdr->sh_size / dynSHdr->sh_entsize; i++) {
            bool end = false;
            switch (dyn->d_tag) {
                case DT_NULL:
                    end = true;
                    break;

                case DT_PLTGOT:
                case DT_HASH:
                case DT_STRTAB:
                case DT_SYMTAB:
                case DT_RELA:
                case DT_INIT:
                case DT_FINI:
                case DT_REL:
                case DT_DEBUG:
                case DT_JMPREL:
                case DT_INIT_ARRAY:
                case DT_FINI_ARRAY:
                case DT_PREINIT_ARRAY:
                case DT_CONFIG:
                case DT_DEPAUDIT:
                case DT_AUDIT:
                case DT_PLTPAD:
                case DT_MOVETAB:
                case DT_SYMINFO:
                case DT_VERDEF:
                case DT_VERNEED:
                    dyn->d_un.d_ptr += vaDiff;
                    break;

                default:
                    break;
            }
            if (end) {
                break;
            }
            dyn = reinterpret_cast<Elf64_Dyn *>(reinterpret_cast<unsigned char *>(dyn) + dynSHdr->sh_entsize);
        }        
    }

    {
        Elf64_Phdr *ph = pHdrs;
        for (auto i = 0; i < elfHdr->e_phnum; i++) {
            if ((ph->p_flags & PT_LOAD) || ph->p_vaddr > 0) {
                ph->p_vaddr += vaDiff;
                ph->p_paddr += vaDiff;
            }
            ph = reinterpret_cast<Elf64_Phdr *>(reinterpret_cast<unsigned char *>(ph) + elfHdr->e_phentsize);
        }
    }

    {
        Elf64_Shdr *sh = sHdrs;
        for (auto i = 0; i < elfHdr->e_shnum; i++) {
            if ((sh->sh_flags & SHF_ALLOC) || sh->sh_addr > 0) {
                sh->sh_addr += vaDiff;
            }
            sh = reinterpret_cast<Elf64_Shdr *>(reinterpret_cast<unsigned char *>(sh) + elfHdr->e_shentsize);
        }
    }

    fs::path elfJson = patchedElf;
    elfJson += ".json";
    auto elfInfo = parsePatchedElfInfoFromJson(elfJson);
    if ( !elfInfo) {
        return false;
    }

    for (Elf64_Addr &fn: elfInfo->initFiniInfo.dt_preinit_array) {
        fn += vaDiff;
    }

    if (elfInfo->initFiniInfo.dt_init) {
        elfInfo->initFiniInfo.dt_init.value() += vaDiff;
    }

    for (Elf64_Addr &fn: elfInfo->initFiniInfo.dt_init_array) {
        fn += vaDiff;
    }

    if (elfInfo->initFiniInfo.dt_fini) {
        elfInfo->initFiniInfo.dt_fini.value() += vaDiff;
    }

    for (Elf64_Addr &fn: elfInfo->initFiniInfo.dt_fini_array) {
        fn += vaDiff;
    }

    if ( !dumpPatchedElfInfoToJson(elfJson, elfInfo.value())) {
        return false;
    }

    return true;
}