#include "Common.h"
#include "Elf/elf-sce.h"
#include "Elf/elf_amd64.h"
#ifdef __linux
#include <cstdint>
#endif
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <elf.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <pthread.h>
#include <map>
#include <utility>

#include <algorithm>
#include "nid_hash/nid_hash.h"

#define ROUND_DOWN(x, SZ) ((x) - (x) % (SZ))
#define ROUND_UP(x, SZ) ( (x) % (SZ) ? (x) - ((x) % (SZ)) + (SZ) : (x))

static void dumpElfHdr(const char *name, Elf64_Ehdr *elfHdr) {
    printf("ELF HEADER DUMP:\n"
        "name: %s\n"
        "\te_ident: %s\n"
        "\te_type: %d\n"
        "\te_machine: %d\n"
        "\te_version: %d\n"
        "\te_entry: 0x%lx\n"
        "\te_phoff: 0x%lx\n"
        "\te_shoff: 0x%lx\n"
        "\te_flags: %x\n"
        "\te_phentsize: %d\n"
        "\te_phnum: %d\n"
        "\te_shentsize: %d\n"
        "\te_shnum: %d\n"
        "\te_shstrndx: %d\n",
        name,
        elfHdr->e_ident,
        elfHdr->e_type,
        elfHdr->e_machine,
        elfHdr->e_version,
        elfHdr->e_entry,
        elfHdr->e_phoff,
        elfHdr->e_shoff,
        elfHdr->e_flags,
        elfHdr->e_phentsize,
        elfHdr->e_phnum,
        elfHdr->e_shentsize,
        elfHdr->e_shnum,
        elfHdr->e_shstrndx      
    );
}

const long pgsz = sysconf(_SC_PAGE_SIZE);

struct CmdConfig {
    bool dumpElfHeader;
    bool dumpRelocs;
    bool dumpModuleInfo;
    bool dumpSymbols;
    std::string pkgDumpPath;

};

CmdConfig CmdArgs;

struct EntryPointWrapperArg {
    Elf64_Addr entryPointAddr;
    Ps4EntryArg entryArg;
};

static void ps4ThreadExitFn(void) {
    printf("ps4 thread exited\n");
}

static void *ps4EntryWrapper(void *arg) {
    EntryPointWrapperArg *wrapperArg = (EntryPointWrapperArg *) arg;
    Elf64_Addr entryPoint = wrapperArg->entryPointAddr;
    //void (*entryFn)(void) = (void (*)(void)) entryPoint;
    PFUNC_GameTheadEntry entryFn = (PFUNC_GameTheadEntry) entryPoint;

    printf("in ps4EntryWrapper()\n");

    entryFn((void *) &wrapperArg->entryArg, ps4ThreadExitFn);
    return NULL;
}

struct MappedSegment {
    Elf64_Addr mStart;
    Elf64_Off mSize;
};

struct DynamicTableInfo {
    Elf64_Phdr dynamicPHdr;
    Elf64_Phdr dynlibdataPHdr;
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
    uint64_t pltgotOff;
    //uint64_t pltgotSz;  // should figure this one out
    uint64_t pltrel;  /* Type of reloc in PLT */
    uint64_t jmprelOff;
    uint64_t jmprelSz;
    std::vector<uint64_t> neededLibs; // offsets into strtab for filenames
    std::vector<uint64_t> neededModules;
};

struct Module {
    std::string name;
    std::string path; // location in filesystem
    uint64_t baseVA;
    // total memory this takes from baseVA onwards
    // should be a multiple of pagesz, s.t. next module can start at this->baseVA + this->memSz
    uint64_t memSz;
    // VA ranges (without base offset of module)
    // mmap will be called on ranges with baseVA added
    std::vector<MappedSegment> mappedSegments;
    std::vector<Elf64_Phdr> pHeaders;
    DynamicTableInfo dynTableInfo;
    std::vector<std::string> neededLibsStrings;

    std::vector<char> strtab;
    std::vector<Elf64_Rela> relocs;
    std::vector<Elf64_Sym> symbols;
    bool isNative;
};

static bool processDynamicSegment(Elf64_Phdr *phdr, FILE *elf, DynamicTableInfo &info) {
    std::vector<unsigned char> buf;
    buf.resize(phdr->p_filesz);

    info.dynamicPHdr = *phdr;

    fseek(elf, phdr->p_offset, SEEK_SET);
    if (1 != fread(buf.data(), phdr->p_filesz, 1, elf)) {
        return false;
    }

    Elf64_Dyn entry;
    for(int i = 0; ; i++) {
        assert((i + 1) * sizeof(Elf64_Dyn) <= phdr->p_filesz);
        unsigned char *off = &buf[i * sizeof(Elf64_Dyn)];
        memcpy(&entry, off, sizeof(entry));

        if (entry.d_tag == DT_NULL) {
            break;
        }

        switch(entry.d_tag) {
            case DT_NULL:
                break;
            case DT_NEEDED:
                info.neededLibs.push_back(entry.d_un.d_val);
                break;
            case DT_PLTRELSZ:
                info.jmprelSz = entry.d_un.d_val; // TODO check this
                break;
            case DT_PLTGOT:
                info.pltgotOff = entry.d_un.d_ptr;
                break;
            case DT_HASH:
                info.hashOff = entry.d_un.d_ptr;
                break;
            case DT_STRTAB:
                info.strtabOff = entry.d_un.d_ptr;
                break;
            case DT_SYMTAB:
                info.symtabOff = entry.d_un.d_ptr;
                break;
            case DT_RELA:
                info.relaOff = entry.d_un.d_ptr;            
                break;
            case DT_RELASZ:
                info.relaSz = entry.d_un.d_val;
                break;
            case DT_RELAENT:
                info.relaEntSz = entry.d_un.d_val;
                break;
            case DT_STRSZ:
                info.strtabSz = entry.d_un.d_val;
                break;
            case DT_SYMENT:
                info.symtabEntSz = entry.d_un.d_val;
                break;
            case DT_PLTREL:
                assert(entry.d_un.d_val == DT_RELA);
                info.pltrel = entry.d_un.d_val;            
                break;
            case DT_JMPREL:
                info.jmprelOff = entry.d_un.d_ptr;
                break;                
            case DT_INIT:
            case DT_FINI:
            case DT_SONAME:
            case DT_RPATH:
            case DT_SYMBOLIC:
            case DT_REL:
            case DT_RELSZ:
            case DT_RELENT:
            case DT_DEBUG:
            case DT_TEXTREL:
            case DT_BIND_NOW:
            case DT_INIT_ARRAY:
            case DT_FINI_ARRAY:
            case DT_INIT_ARRAYSZ:
            case DT_FINI_ARRAYSZ:
            case DT_RUNPATH:
            case DT_FLAGS:
            case DT_ENCODING:
            case DT_PREINIT_ARRAYSZ:
            case DT_SYMTAB_SHNDX:
            case DT_NUM:
            case DT_LOOS:
            case DT_HIOS:
            default:
                break;
        }

        //uint64_t val = entry.d_un.d_val;
        switch(entry.d_tag) {
            case DT_SCE_NEEDED_MODULE:
                info.neededModules.push_back(entry.d_un.d_val);
                //printf("dyn case DT_SCE_NEEDED_MODULE: %lx\n", val);
                break;
            case DT_SCE_IMPORT_LIB:
                //printf("dyn case DT_SCE_IMPORT_LIB: %lx\n", val);
                break;
            case DT_SCE_IMPORT_LIB_ATTR:
                //printf("dyn case DT_SCE_IMPORT_LIB_ATTR: %lx\n", val);
                break;                       
            case DT_SCE_HASH:
                info.hashOff = entry.d_un.d_ptr;
                break;
            case DT_SCE_PLTGOT:
                info.pltgotOff = entry.d_un.d_ptr;
                break;
            case DT_SCE_JMPREL:
                info.jmprelOff = entry.d_un.d_ptr;
                break;                 
            case DT_SCE_PLTREL:
                assert(entry.d_un.d_val == DT_RELA);
                info.pltrel = entry.d_un.d_val;
                break;
            case DT_SCE_PLTRELSZ:
                info.jmprelSz = entry.d_un.d_val;
                break;                  
            case DT_SCE_RELA:
                info.relaOff = entry.d_un.d_ptr;
                break;
            case DT_SCE_RELASZ:
                info.relaSz = entry.d_un.d_val;
                break;
            case DT_SCE_RELAENT:
                info.relaEntSz = entry.d_un.d_val;
                break;
            case DT_SCE_STRTAB:
                info.strtabOff = entry.d_un.d_ptr;
                break;
            case DT_SCE_STRSZ:
                info.strtabSz = entry.d_un.d_val;
                break;
            case DT_SCE_SYMTAB:
                info.symtabOff = entry.d_un.d_ptr;
                break;
            case DT_SCE_SYMENT:
                info.symtabEntSz = entry.d_un.d_val;
                break;                                                
            case DT_SCE_HASHSZ:
                info.hashSz = entry.d_un.d_val;
                break;
            case DT_SCE_SYMTABSZ:
                info.symtabSz = entry.d_un.d_val;
                break;
            default:
                break;
        }
    }

    return true;
}

static bool isHashedSymbol(const std::string &sym) {
    if (sym.length() == 15) {
        std::string suffix = sym.substr(11);
        return suffix[0] == '#' && suffix[2] == '#' 
            /* && suffix[1] >= 'A' && suffix[1] <= 'Z' && suffix[3] >= 'A' && suffix[3] <= 'Z' */;
    }
    return false;
}

// if ref is hash:
//      if def is hash:
//          compare
//      else:
//          hash def
//          compare
// else:
//      look for exact match (not hash)
bool findSymbolDef(Elf64_Sym sym, Module &referenceLocation, std::map<std::string, Module> &modules,  
                /* ret */Module &defLocation, Elf64_Sym &symDef) {
    std::string refName(&referenceLocation.strtab[sym.st_name]);
    std::string origName = refName;

    std::vector<std::string> moduleOverrides;
    moduleOverrides.push_back("all.sprx");
    std::vector<std::string> searchPaths;

    searchPaths.insert(searchPaths.end(), moduleOverrides.begin(), moduleOverrides.end());
    searchPaths.insert(searchPaths.end(), referenceLocation.neededLibsStrings.begin(), referenceLocation.neededLibsStrings.end());

    bool found = false;
    for (uint i = 0; i < searchPaths.size(); i++) {
        std::string modName = searchPaths[i];
        if (modules.find(modName) == modules.end()) {
            continue;
        }
        Module &mod = modules[modName];

        for (uint i = 0; i < mod.symbols.size(); i++) {
            Elf64_Sym modSym = mod.symbols[i];

            if (modSym.st_value == STN_UNDEF) {
                continue;
            }

            std::string defName(&mod.strtab[modSym.st_name]);

            if (isHashedSymbol(refName)) {
                std::string hashedDef;
                if (isHashedSymbol(defName)) {
                    hashedDef = defName;
                } else {
                    char hash[12];
                    calculateNid(defName.c_str(), defName.size(), hash);           
                    hashedDef = std::string(hash);
                } 

                if (refName.substr(0, 11) != hashedDef.substr(0, 11)) {
                    continue;
                }
            } else if (refName != defName) {
                continue;
            }

            uint8_t bind_type = ELF64_ST_BIND(sym.st_info);
            switch(bind_type) {
                case STB_LOCAL:
                    //fprintf(stderr, "found local symbol matching name: %s\n", origName.c_str());
                    continue;
                case STB_GLOBAL:
                    assert(ELF64_ST_TYPE(modSym.st_info) == ELF64_ST_TYPE(sym.st_info));
                    symDef = modSym;
                    defLocation = mod;
                    printf("resolved GLOBAL symbol: %s, needed in module %s, found in %s\n", origName.c_str(), referenceLocation.name.c_str(), mod.name.c_str());                    
                    return true;                
                case STB_WEAK:
                    // TODO just skip if mismatch, this is legal
                    assert(ELF64_ST_TYPE(modSym.st_info) == ELF64_ST_TYPE(sym.st_info));
                    symDef = modSym;
                    defLocation = mod;
                    found = true;
                    break;                
                //case STB_NUM: // not defined in FreeBSD 9.0
                case STB_LOOS:
                case STB_HIOS:
                case STB_LOPROC:
                case STB_HIPROC:
                default:
                    //fprintf(stderr, "unhandled binding type: %d\n", bind_type);
                    break;
            }

            return true;
        }
    }

    if (found) {
        //printf("resolved WEAK symbol: %s, needed in module %s, found in %s\n", origName.c_str(), referenceLocation.name.c_str(), defLocation.name.c_str());
    }

    return found;
}

void printReloc(Elf64_Rela reloc, Module &mod, FILE *stream = stdout) {
    Elf64_Sym sym = mod.symbols[ELF64_R_SYM(reloc.r_info)];
    fprintf(stream, "RELOC DUMP: %s -> 0x%lx\n"
                "\tsym: %s\n"
                "\tmodule: %s\n"
                "\tsym idx: %lu\n"                
                "\tr_type: %lu\n"
                "\tr_offset: 0x%lx\n"
                "\tr_addend: 0x%lx\n"
                "\tMapped Addr (mod.baseVA + r_offset): 0x%lx\n",
        &mod.strtab[sym.st_name], mod.baseVA + reloc.r_offset,
        &mod.strtab[sym.st_name],
        mod.name.c_str(),
        ELF64_R_SYM(reloc.r_info),
        ELF64_R_TYPE(reloc.r_info),
        reloc.r_offset,
        reloc.r_addend,
        mod.baseVA + reloc.r_offset
    );
}

void printModuleInfo(Module &mod, FILE *stream = stdout) {
    printf("MODULE %s, path: %s ***************************\n", mod.name.c_str(), mod.path.c_str());
    printf("baseVA: 0x%lx, endVA: 0x%lx, memSz: 0x%lx\n", mod.baseVA, mod.baseVA + mod.memSz, mod.memSz);
}

bool handleRelocations(Module &mod, std::map<std::string, Module> &modules) {
    //std::map<std::basic_string<char>, Module>::iterator it = modules.begin();
    for (uint j = 0; j < mod.relocs.size(); j++) {
        Elf64_Rela reloc = mod.relocs[j];
        uint64_t r_type = ELF64_R_TYPE(reloc.r_info);
        uint64_t symIdx = ELF64_R_SYM(reloc.r_info);

        if (symIdx == STN_UNDEF) {
            // "If the index is STN_UNDEF, the undefined symbol index, the relocation uses 0 as the symbol value"
            // TODO write 0
        } else {
            Elf64_Sym sym = mod.symbols[ELF64_R_SYM(reloc.r_info)];
            const char *symName = &mod.strtab[sym.st_name];            
            //printReloc(reloc, mod);

            Module defLocation;
            Elf64_Sym symDef;
            bool res = findSymbolDef(sym, mod, modules, defLocation, symDef);
            if (!res) {
                fprintf(stderr, "couldn't find definition for relocation sym %s in module %s, path %s\n", symName, mod.name.c_str(), mod.path.c_str());
                //printReloc(reloc, mod, stderr);
                continue;
            }

            switch (r_type) {
                case R_AMD64_NONE:
                    break;
                case R_AMD64_64:
                {
                    uint64_t addr = mod.baseVA + reloc.r_offset;
                    uint64_t value = defLocation.baseVA + symDef.st_value + reloc.r_addend;
                    //uint64_t value = symDef.st_value + reloc.r_addend;
                    *((uint64_t *) addr) = value;

                    printf("symbol: %s, type: R_AMD64_64, addr: 0x%lx, value: 0x%lx, addend %lx, \n", symName, addr, value, reloc.r_addend);
                    break;
                }
                case R_AMD64_JUMP_SLOT:
                {
                    uint64_t addr = mod.baseVA + reloc.r_offset;
                    uint64_t value = defLocation.baseVA + symDef.st_value;
                    //uint64_t value = symDef.st_value;
                    *((uint64_t *) addr) = value;
                    printf("symbol: %s, type: R_AMD64_JUMP_SLOT, addr: 0x%lx, value: 0x%lx, \n", symName, addr, value);
                    break;
                }
                case R_AMD64_DTPMOD64:
                {
                    // "https://chao-tic.github.io/blog/2018/12/25/tls"
                    //uint64_t addr = mod.baseVA + reloc.r_offset;
                    //fprintf(stderr, "mod %s, symbol: %s, type: R_AMD64_JUMP_SLOT, addr: 0x%lx\n", mod.name.c_str(), symName, addr);    
                }                
                case R_AMD64_PC32:
                case R_AMD64_GOT32:
                case R_AMD64_PLT32:
                case R_AMD64_COPY:
                case R_AMD64_GLOB_DAT:                
                case R_AMD64_RELATIVE:
                case R_AMD64_GOTPCREL:
                case R_AMD64_32:
                case R_AMD64_32S:
                case R_AMD64_16:
                case R_AMD64_PC16:
                case R_AMD64_8:
                case R_AMD64_PC8:
                case R_AMD64_DTPOFF64:
                case R_AMD64_TPOFF64:
                case R_AMD64_TLSGD:
                case R_AMD64_TLSLD:
                case R_AMD64_DTPOFF32:
                case R_AMD64_GOTTPOFF:
                case R_AMD64_TPOFF32:
                case R_AMD64_PC64:
                case R_AMD64_GOTOFF64:
                case R_AMD64_GOTPC32:
                default:
                {
                    // See the R_X86_64* types if the type is > R_AMD64_GOTPC32
                    //fprintf(stderr, "unhandled relocation type: %lu\n", r_type);
                    //assert(false && "unhandled relocation type");        
                }
            }            
        }
    }

    return true;
}

FILE *openWithSearchPaths(std::string name, std::string mode, 
        /* ret */ std::string *path) {

    std::vector<std::string> paths;

    paths.push_back("./");
    paths.push_back(CmdArgs.pkgDumpPath);
    paths.push_back(CmdArgs.pkgDumpPath + "/sce_sys/");
    paths.push_back(CmdArgs.pkgDumpPath + "/sce_module/");
    paths.push_back("../../dynlibs/lib/");
    paths.push_back("../library_overloads/");

    std::vector<std::string> basenamePermutations;
    basenamePermutations.push_back(name);

    // TODO replace .prx extension with .sprx
    size_t extPos = name.find_last_of('.');
    if (extPos != std::string::npos && name.substr(extPos) == ".prx") {
        std::string sprx = name.substr(0, extPos) + ".sprx";
        basenamePermutations.push_back(sprx);
    }

    FILE *file;
    for (uint i = 0; i < paths.size(); i++) {
        for (uint j = 0; j < basenamePermutations.size(); j++) {
            std::string basename = basenamePermutations[j];
            std::string fpath = std::string(paths[i]) + std::string("/") + basename;
            file = fopen(fpath.c_str(), mode.c_str());
            if (file) {
                if (path) {
                    *path = fpath;
                }
                return file;
            }
        }
    }
    return NULL;
}

bool parseNativeDynamicContents(DynamicTableInfo dynTableInfo, FILE *elf, /* ret */
                    std::vector<char> &strtab, 
                    std::vector<std::string> &dynLibStrings,
                    std::vector<Elf64_Rela> &relocs,
                    std::vector<Elf64_Sym> &symbols) {
    
    
    std::vector<unsigned char> dynLibDataContents(dynTableInfo.dynlibdataPHdr.p_filesz);
    fseek(elf, dynTableInfo.dynlibdataPHdr.p_offset, SEEK_SET);
    if (1 != fread(dynLibDataContents.data(), dynTableInfo.dynlibdataPHdr.p_filesz, 1, elf)) {
        return false;
    }

    for (uint i = 0; i < dynTableInfo.neededLibs.size(); i++) {
        uint64_t strOff = dynTableInfo.strtabOff + dynTableInfo.neededLibs[i];
        std::string path((char *) &dynLibDataContents[strOff]);
        dynLibStrings.push_back(path);
    }

    for (uint i = 0; i < (dynTableInfo.jmprelSz + dynTableInfo.relaSz) / dynTableInfo.relaEntSz; i++) {
        // handle endianess TODO
        uint64_t relaOff = dynTableInfo.jmprelOff + i * dynTableInfo.relaEntSz;
        Elf64_Rela *ent = (Elf64_Rela *) &dynLibDataContents[relaOff];
        relocs.push_back(*ent);
    }

    for (uint i = 0; i < dynTableInfo.symtabSz / dynTableInfo.symtabEntSz; i++) {
        // handle endianess TODO
        uint64_t symOff = dynTableInfo.symtabOff + i * dynTableInfo.symtabEntSz;
        Elf64_Sym *ent = (Elf64_Sym *) &dynLibDataContents[symOff];
        symbols.push_back(*ent);
    }

    strtab.resize(dynTableInfo.strtabSz);
    memcpy(strtab.data(), &dynLibDataContents[dynTableInfo.strtabOff], dynTableInfo.strtabSz);

    return true;
}

bool parseHostDynamicContents(FILE *elf, Elf64_Ehdr elfHdr, /* ret */
                    std::vector<char> &strtab, 
                    std::vector<std::string> &dynLibStrings,
                    std::vector<Elf64_Rela> &relocs,
                    std::vector<Elf64_Sym> &symbols) {


    // anything in dynTableInfo should look in .dyn* section of elf
    // e.g. neededLibStrings should use .dynstr.base + offset

    // otherwise look in symtab section, need to figure this out 
    return true;
}

// fill out Module struct with metadata for module at given path and put in Module table
// don't decide on baseVA yet
// recurse to other necessary modules

// TODO change or make different function for custom compiled modules that reads traditional section hdrs and symtab
bool getModuleInfo(std::string basename, bool isNative, std::map<std::string, Module> &modules, Elf64_Addr &lastModuleEndAddr) {
    FILE *elf;
    Module mod;
    DynamicTableInfo dynTableInfo;
    std::vector<Elf64_Phdr> progHdrs;

    if (modules.find(basename) != modules.end()) {
        return true;
    }

    //printf("getModuleInfo: %s\n", path.c_str());

    std::string fullModulePath;
    elf = openWithSearchPaths(basename, "r", &fullModulePath);
    if (!elf) {
        //fprintf(stderr, "couldn't open %s\n", basename.c_str());
        return false;
    }

    Elf64_Ehdr elfHdr;
    fseek(elf, 0, SEEK_SET);
    if (1 != fread(&elfHdr, sizeof(elfHdr), 1, elf)) {
        return false;
    }

    if (CmdArgs.dumpElfHeader) {
        dumpElfHdr(basename.c_str(), &elfHdr);
    }

    progHdrs.resize(elfHdr.e_phnum);
    fseek(elf, elfHdr.e_phoff, SEEK_SET);
    if (elfHdr.e_phnum != fread(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, elf)) {
        return false;
    }

    std::vector<MappedSegment> mappedSegments;

    for (uint i = 0; i < progHdrs.size(); i++) {
        Elf64_Phdr *phdr = &progHdrs[i];
        // guarantees that we can put dynamic libraries in memory at the start of the next page after the last module ended
        //assert(pgsz % phdr->p_align == 0);
        switch (progHdrs[i].p_type) {
            //case PT_PHDR:
            //case PT_INTERP:
            case PT_DYNAMIC:
                processDynamicSegment(phdr, elf, dynTableInfo);
                if (!isNative) {
                    dynTableInfo.dynlibdataPHdr = *phdr;
                }
                break;
            case PT_SCE_DYNLIBDATA:
                dynTableInfo.dynlibdataPHdr = *phdr;
                break;
            case PT_LOAD:
            //case PT_GNU_EH_FRAME:
            case PT_SCE_RELRO:
            //case PT_SCE_MODULEPARAM:
            //case PT_SCE_PROCPARAM:
            {
                // TODO make it so all PIC shared libs start at 0

                uint64_t mappingStart = ROUND_DOWN(phdr->p_vaddr, pgsz);
                // should be last byte on a page
                uint64_t mappingEnd = ROUND_UP(phdr->p_vaddr + phdr->p_filesz - 1, pgsz) - 1;
                uint64_t mappingSize = mappingEnd - mappingStart + 1;
                MappedSegment segA;
                segA.mStart = mappingStart;
                segA.mSize = mappingSize;

                bool shouldCreateSeg = true;
                for (uint j = 0; j < mappedSegments.size(); j++) {
                    MappedSegment segB = mappedSegments[j];
                    // see if  intervals overlap in some way
                    if (segA.mStart > segB.mStart) {
                        std::swap(segA, segB);
                    }

                    if (segB.mStart > segA.mStart + segA.mSize) {
                        // they don't overlap
                        continue;
                    }

                    // they overlap
                    // coalesce into new interval
                    MappedSegment replacedSeg;
                    replacedSeg.mStart = segA.mStart;
                    replacedSeg.mSize = (segB.mStart - segA.mStart) + segB.mSize;
                    mappedSegments[j] = replacedSeg;
                    shouldCreateSeg = false;
                    break;
                }

                if (shouldCreateSeg) {
                    MappedSegment newSeg;
                    newSeg.mStart = mappingStart;
                    newSeg.mSize = mappingSize;
                    mappedSegments.push_back(newSeg);                    
                }
            }
                break;
            default:
                break;
        }
    }

    Elf64_Off highestAddr = 0;
    for (uint i = 0; i < mappedSegments.size(); i++) {
        MappedSegment *seg = &mappedSegments[i];
        assert(seg->mSize % pgsz == 0);
        highestAddr = std::max(highestAddr, seg->mStart + seg->mSize);
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    // parse dynlibdata for relocations, symbols, strings

    std::vector<unsigned char> dynLibDataContents;
    std::vector<char> strtab;
    std::vector<std::string> dynLibStrings;
    std::vector<Elf64_Rela> relocs;
    std::vector<Elf64_Sym> symbols;

    //if (isNative) {
        if (!parseNativeDynamicContents(dynTableInfo, elf, strtab, dynLibStrings, relocs, symbols))
            return false;
    //} else {}

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    assert(lastModuleEndAddr % pgsz == 0);

    mod.name = basename;
    mod.path = fullModulePath;
    mod.baseVA = lastModuleEndAddr;
    mod.memSz = highestAddr;
    mod.mappedSegments = mappedSegments;
    mod.pHeaders = progHdrs;
    mod.dynTableInfo = dynTableInfo;
    mod.neededLibsStrings = dynLibStrings;

    mod.strtab = strtab;

    mod.relocs = relocs;
    mod.symbols = symbols;
    mod.isNative = isNative;

    modules[basename] = mod;

    lastModuleEndAddr += mod.memSz;
    for (uint i = 0; i < dynLibStrings.size(); i++) {
        std::string lib = dynLibStrings[i];
        if (!getModuleInfo(lib, true, modules, lastModuleEndAddr))
            ;//return false;
    }
    fclose(elf);

    return true;
}

// map one module into memory and do relocations
bool mapModule(Module &mod, std::map<std::string, Module> &modules) {
    FILE *elf = fopen(mod.path.c_str(), "r");
    if (!elf) {
        fprintf(stderr, "couldn't open %s to map module\n", mod.name.c_str());
        return 1;
    }

    for (uint i = 0; i < mod.mappedSegments.size(); i++) {
        MappedSegment seg = mod.mappedSegments[i];
        void *addr = mmap((void *) (mod.baseVA + seg.mStart), seg.mSize, 
                PROT_EXEC | PROT_READ | PROT_WRITE,
                MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0 );
        assert((uint64_t) addr == mod.baseVA + seg.mStart);
    }

    std::vector<unsigned char> buf;
    for (uint i = 0; i < mod.pHeaders.size(); i++) {
        Elf64_Phdr *phdr = &mod.pHeaders[i];
        switch (phdr->p_type) {
            case PT_LOAD:
            case PT_SCE_RELRO:
            {
                buf.resize(phdr->p_filesz);

                fseek(elf, phdr->p_offset, SEEK_SET);
                assert( 1 == fread(buf.data(), phdr->p_filesz, 1, elf) );
                memcpy((char *) (mod.baseVA + phdr->p_vaddr), buf.data(), phdr->p_filesz);                
            }
            default:
                break;
        }
    }

    fclose(elf);

    if (!handleRelocations(mod, modules))
        return false;  // TODO

    return true;
}

bool loadCustomModules(std::map<std::string, Module> &modules, Elf64_Addr &lastModuleEndAddr) {
    const char *overrides[] = {
        "all.so"
    };

    for (uint i = 0; i < ARRLEN(overrides); i++) {
        if (!getModuleInfo(std::string(overrides[i]), false, modules, lastModuleEndAddr)) {
            fprintf(stderr, "couldn't load custom module %s\n", overrides[i]);
            return false;
        }
    }

    return true;
}

// get info about all necessary dynamic libraries, map process memory, and do relocations
bool loadFirstModule(std::string name, std::map<std::string, Module> &modules) {
    Elf64_Addr lastModuleEndAddr = 0;
    if (!getModuleInfo(name, true, modules, lastModuleEndAddr))
        return false;

    // load our own modules that we want to take precedence during relocations
    // names of these modules should be different from any requested in ps4 ELF's
    if (!loadCustomModules(modules, lastModuleEndAddr))  {
        return false;
    }

    std::map<std::basic_string<char>, Module>::iterator it = modules.begin();
    for (; it != modules.end(); it++) {
#ifndef __linux
        Module &mod = it->second;
        if (!mapModule(mod, modules)) {
            fprintf(stderr, "Error while mapping %s\n", it->second.name.c_str());
            return false;
        }
#endif
    }
    
    return true;
}

bool parseCmdArgs(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s [options] <PATH TO PKG DUMP>\n", argv[0]);
        exit(1);
    }

    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--dump_elf_header")) {
            CmdArgs.dumpElfHeader = true;
        } else if (!strcmp(argv[i], "--dump_relocs")) {
            CmdArgs.dumpRelocs = true;
        } else if (!strcmp(argv[i], "--dump_module_info")) {
            CmdArgs.dumpModuleInfo = true;
        }else if (!strcmp(argv[i], "--dump_symbols")) {
            CmdArgs.dumpSymbols = true;
        } else {
            fprintf(stderr, "Unrecognized cmd arg: %s\n", argv[i]);
            return false;
        }
    }

    CmdArgs.pkgDumpPath = argv[argc - 1];
    return true;
}

bool modulelambda(Module &a, Module &b) {
    return a.baseVA < b.baseVA;    
}

int main(int argc, char **argv) {
    const std::string ebootPath = "eboot.bin";
    FILE *eboot;
    std::map<std::string, Module> modules;

    if (!parseCmdArgs(argc, argv)) {
        return 1;
    }
    
    eboot = openWithSearchPaths(ebootPath, "r", NULL);
    if (!eboot) {
        fprintf(stderr, "couldn't open eboot.bin\n");
        return 1;
    }

    Elf64_Ehdr elfHdr;
    fseek(eboot, 0, SEEK_SET);
    if (1 != fread(&elfHdr, sizeof(elfHdr), 1, eboot)) {
        return 1;
    }

    // loadEntryModule will open file again
    fclose(eboot);
    if ( !loadFirstModule(ebootPath, modules)) {
        return 1;
    }

    std::vector<Module> sortedModules;
    std::map<std::basic_string<char>, Module>::iterator it = modules.begin();
    for (; it != modules.end(); it++) {
        sortedModules.push_back(it->second);
    }

#ifdef __linux
    std::sort(sortedModules.begin(), sortedModules.end(), [](const Module &a, const Module &b) -> bool {
        return a.baseVA < b.baseVA;
    });
    for (Module &mod: sortedModules) {
        if (CmdArgs.dumpModuleInfo) {
            printModuleInfo(mod);
        }

        if (CmdArgs.dumpSymbols) {
            for (Elf64_Sym &sym: mod.symbols) {
                fprintf(stderr, "Module: %s\n", mod.name.c_str());
                fprintf(stderr, "SYMBOL DUMP: %s\n", &mod.strtab[sym.st_name]);
            }
        }

        if (CmdArgs.dumpRelocs) {
            for (Elf64_Rela &rela: mod.relocs) {
                printReloc(rela, mod, stderr);
            }
        }
    }

    return 0;
#endif

    pthread_t ps4Thread;
    EntryPointWrapperArg arg;
    arg.entryPointAddr = elfHdr.e_entry;

    Ps4EntryArg entryArg;
    entryArg.argc = 1;
    entryArg.argv[0] = "eboot.bin";
    arg.entryArg = entryArg;

    pthread_create(&ps4Thread, NULL, &ps4EntryWrapper, (void *) &arg);
    pthread_join(ps4Thread, NULL);

    printf("main: done\n");
    return 0;
}
