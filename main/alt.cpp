#include "Common.h"
#include "elf-sce.h"
#include "elf_amd64.h"
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
#include "../nid_hash/nid_hash.h"
#include <dlfcn.h>
#include <libgen.h>
#include <map>

const long PGSZ = sysconf(_SC_PAGE_SIZE);
const std::string ebootPath = "eboot.bin";

#define ROUND_DOWN(x, SZ) ((x) - (x) % (SZ))
#define ROUND_UP(x, SZ) ( (x) % (SZ) ? (x) - ((x) % (SZ)) + (SZ) : (x))

class Module {
public:
    Module() {};

    uint64_t baseVA;
    bool isEntryModule;
    std::vector<Elf64_Phdr> pHeaders;
    std::string path;
    std::vector<char> strtab;
};

class Ps4Module : public Module {
public:
    std::vector<Elf64_Rela> relocs;
    std::vector<Elf64_Sym> symbols;
};

class HostModule : public Module {

};

// RELA might not work, .rela + dstname might be necessary

#define SECTION_TABLE(OP) \
    OP(DYNSTR, ".dynstr") \
    OP(DYNSYM, ".dynsym") \
    OP(PLT, ".plt") \
    OP(GOT, ".got") \
    OP(PLTGOT, ".plt.got") \
    OP(RELA, ".rela" ) \
    OP(SCE_DYNSTR, ".dynstr") \
    OP(SCE_DYNSYM, ".dynsym") \
    OP(SCE_PLTGOT, ".plt.got") \

#define SECTION_ENUM_LIST(e, name) e,
#define SECTION_NAME_LIST(e, name) name,

enum section_type {
    SECTION_TABLE(SECTION_ENUM_LIST)
    NUM_SECTIONS,
};

const char *section_names[NUM_SECTIONS] {
    SECTION_TABLE(SECTION_NAME_LIST)
};

class SectionMap {
private:
    std::map<section_type, int> sectionToIndex;
    int numSections;
public:
    int getOrSetSectionIdx(section_type type) {
        if (sectionToIndex.find(type) != sectionToIndex.end()) {
            return sectionToIndex[type];
        }
        sectionToIndex[type] = ++numSections;
        return numSections;
    }
};

SectionMap section_map;

struct Section {
    std::vector<unsigned char> contents;
    Elf64_Shdr hdr;
    section_type type;
    bool finalized; // false if contents need to be written into segment contents

    // copy section contents into dst buffer if the section isn't already finalized
    // also complete the section header with vaddr, size, and offset info
    void finalize(Elf64_Phdr *parent, std::vector<unsigned char> &dst) {
        if ( !finalized) {
            // assume dst contents are already aligned (rounded up w/ padding)
            
            size_t sectionSizePadded = ROUND_UP(contents.size(), PGSZ);
            size_t segOff = dst.size();

            hdr.sh_addr = parent->p_vaddr + segOff;
            hdr.sh_offset = parent->p_offset + segOff;
            hdr.sh_size = contents.size();
            hdr.sh_addralign = PGSZ;

            dst.resize(dst.size() + sectionSizePadded);
            memcpy(&dst[segOff], contents.data(), contents.size());

            finalized = true;
        }
    }
};

struct Segment {
    std::vector<unsigned char> contents;
    std::vector<Section> sections;
    Elf64_Phdr hdr;
    bool finalized;

    Segment(Elf64_Phdr hdr, bool finalized): hdr(hdr), finalized(finalized) {}

    void finalize(Elf64_Off fileOffset, Elf64_Addr vAddr, Elf64_Addr pAddr) {
        if (!finalized) {
            hdr.p_offset = fileOffset;
            hdr.p_vaddr = vAddr;
            hdr.p_paddr = pAddr;

            for (auto &section: sections) {
                section.finalize(&hdr, contents);
            }

            // finalize phdr here TODO
            hdr.p_filesz = contents.size();
            // should probably give this up front
            hdr.p_memsz = std::max(hdr.p_memsz, hdr.p_filesz);
            hdr.p_align = PGSZ;

            finalized = true;
        }
    }
};



struct CmdConfig {
    bool dumpElfHeader;
    bool dumpRelocs;
    bool dumpModuleInfo;
    bool dumpSymbols;
    std::string pkgDumpPath;
    std::string dlPath;
};

CmdConfig CmdArgs;

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

    //CmdArgs.pkgDumpPath = argv[argc - 1];
    CmdArgs.dlPath = argv[argc - 1];
    return true;
}

static void dumpElfHdr(const char *name) {
    FILE *f = fopen(name, "r+");
    assert(f);

    Elf64_Ehdr elfHdr;
    fseek(f, 0, SEEK_SET);
    assert (1 == fread(&elfHdr, sizeof(elfHdr), 1, f));    

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
        "\te_shstrndx: %d\n"
        "\tABI: %hhu\n",
        name,
        elfHdr.e_ident,
        elfHdr.e_type,
        elfHdr.e_machine,
        elfHdr.e_version,
        elfHdr.e_entry,
        elfHdr.e_phoff,
        elfHdr.e_shoff,
        elfHdr.e_flags,
        elfHdr.e_phentsize,
        elfHdr.e_phnum,
        elfHdr.e_shentsize,
        elfHdr.e_shnum,
        elfHdr.e_shstrndx,
        elfHdr.e_ident[EI_OSABI]
    );

    fclose(f);
}

static void dumpShdr(Module &mod, Elf64_Shdr *sHdr) {
    printf("name: %s\n"
        "\tsh_type: %u\n",
        (char *) &mod.strtab[sHdr->sh_name],
        sHdr->sh_type
    );
}

int findPhdr(std::vector<Elf64_Phdr> &pHdrs, Elf64_Word type) {
    for (size_t i = 0; i < pHdrs.size(); i++) {
        if (pHdrs[i].p_type == type) {
            return i;
        }
    }
    return -1;
}

Section *findSection(Segment &seg, section_type type) {
    for (auto &sec: seg.sections) {
        if (sec.type == type) {
            return &sec;
        }
    }
    return nullptr;
}

Section *findSection(std::vector<Segment> segments, section_type type) {
    for (auto &seg: segments) {
        if (Section *section = findSection(seg, type)) {
            return section;
        }
    }
    return nullptr;
}

bool writePadding(FILE *f, size_t alignment) {
    fseek(f, 0, SEEK_END);
    size_t flen = ftell(f);

    size_t padlen = ROUND_UP(flen, alignment) - flen;
    std::vector<unsigned char> padding(padlen, 0);
    assert(1 == fwrite(padding.data(), padlen, 1, f));

    return true;
}

// return index into strtab
uint appendToStrtab(std::vector<char> &strtab, const char *str) {
    int off = strtab.size();
    int addedSz = strlen(str) + 1;
    strtab.resize(off + addedSz);
    memcpy(&strtab[off], str, addedSz);

    return off;
}

static bool patchDynamicSegment(std::vector<Elf64_Phdr> &progHdrs, FILE *elf) {
    int idx = findPhdr(progHdrs, PT_DYNAMIC);
    assert(idx >= 0);
    Elf64_Phdr *dynPhdr = &progHdrs[idx];

    std::vector<unsigned char> buf(dynPhdr->p_filesz);

    fseek(elf, dynPhdr->p_offset, SEEK_SET);
    assert(1 == fread(buf.data(), dynPhdr->p_filesz, 1, elf));

    std::vector<Elf64_Dyn> newEntries;

    Elf64_Dyn *dynEnt;
    for(int i = 0; ; i++) {
        assert((i + 1) * sizeof(Elf64_Dyn) <= dynPhdr->p_filesz);
        unsigned char *off = &buf[i * sizeof(Elf64_Dyn)];
        dynEnt = (Elf64_Dyn *) off;

        if (dynEnt->d_tag == DT_NULL) {
            newEntries.push_back(*dynEnt);
            break;
        }

        switch(dynEnt->d_tag) {
            // DT_ tags
            case DT_NULL:
            case DT_NEEDED:
            case DT_PLTRELSZ:
            case DT_PLTGOT:
            case DT_HASH:
            case DT_STRTAB:
            case DT_SYMTAB:
            case DT_RELA:
            case DT_RELASZ:
            case DT_RELAENT:
            case DT_STRSZ:
            case DT_SYMENT:
            case DT_INIT:
            case DT_FINI:
            case DT_SONAME:
            case DT_RPATH:
            case DT_SYMBOLIC:
            case DT_REL:
            case DT_RELSZ:
            case DT_RELENT:
            case DT_PLTREL:
            case DT_DEBUG:
            case DT_TEXTREL:
            case DT_JMPREL:
            case DT_BIND_NOW:
            case DT_INIT_ARRAY:
            case DT_FINI_ARRAY:
            case DT_INIT_ARRAYSZ:
            case DT_FINI_ARRAYSZ:
            case DT_RUNPATH:
            case DT_FLAGS:
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
                newEntries.push_back(*dynEnt);
                break;
        }

        //uint64_t val = entry.d_un.d_val;
        switch(dynEnt->d_tag) {
            // DT_SCE tags
            case DT_SCE_NEEDED_MODULE:
                break;
            case DT_SCE_IMPORT_LIB:
                break;
            case DT_SCE_IMPORT_LIB_ATTR:
                break;                       
            case DT_SCE_HASH:
                break;
            case DT_SCE_PLTGOT:
                break;
            case DT_SCE_JMPREL:
                break;                 
            case DT_SCE_PLTREL:
                break;
            case DT_SCE_PLTRELSZ:
                break;                  
            case DT_SCE_RELA:
                break;
            case DT_SCE_RELASZ:
                break;
            case DT_SCE_RELAENT:
                break;
            case DT_SCE_STRTAB:

                break;
            case DT_SCE_STRSZ:
                break;
            case DT_SCE_SYMTAB:
                break;
            case DT_SCE_SYMENT:
                break;                                                
            case DT_SCE_HASHSZ:
                break;
            case DT_SCE_SYMTABSZ:
                break;
            default:
                break;
        }
    }

    // TODO
    // There are DT_NEEDED's in the ents. Should put them in our new strtab
    // Should also replace DT_STRTAB address w/ new strtab address
    // ps4 libs have DT_SCE_STRTAB, convert this to DT_STRTAB
    //
    // We need to put the strtab in a new loadable segment
    // Should standardize the way we build and append new segments/sections
    // sections can have a parent segment for simplicity
    // each segment can be a list of sections, each section can be a list of
    // buffers 

    fseek(elf, dynPhdr->p_offset, SEEK_SET);
    assert(1 == fwrite(newEntries.data(), newEntries.size() * sizeof(Elf64_Dyn), 1, elf));

    uint64_t highestVAddr = 0;
    uint64_t highestPAddr = 0;
    for (auto phdr: progHdrs) {
        highestVAddr = std::max(highestVAddr, phdr.p_vaddr + phdr.p_memsz);
        highestPAddr = std::max(highestPAddr, phdr.p_paddr + phdr.p_memsz);
    }
    dynPhdr->p_vaddr = ROUND_UP(highestVAddr, PGSZ);
    dynPhdr->p_paddr = ROUND_UP(highestPAddr, PGSZ);
    //dynPhdr->p_align = PGSZ;
    dynPhdr->p_flags = PF_X | PF_W | PF_R;

    return true;
}

bool patchPs4Lib(Ps4Module &lib, /* ret */ std::string &newPath) {
    // alignment for new sections (data) we add
    const size_t section_alignment = 16;

    std::vector<Elf64_Phdr> progHdrs;
    //std::vector<Elf64_Phdr> newProgHdrs;
    std::vector<std::vector<unsigned char>> newSegments;
    std::vector<char> newStrtab;
    // in a Ps4 module, these Dyn Ents are in the DYNAMIC segment/.dynamic section,
    // and describe the DYNLIBDATA segment
    std::vector<Elf64_Dyn> oldPs4DynEnts;    
    std::vector<Elf64_Dyn> newElfDynEnts;
    // new section headers, to be appended to the new module file
    std::vector<Elf64_Shdr> newSectionHdrs;
    // index of the strtab in the shEntries
    size_t strtabIdx;

    newPath = lib.path + ".new";

    std::string cmd = "cp " + lib.path + " " + newPath;
    system(cmd.c_str());

    FILE *f = fopen(newPath.c_str(), "r+");
    if (!f) {
        fprintf(stderr, "couldn't open %s\n", newPath.c_str());
        return false;
    }

    Elf64_Ehdr elfHdr;
    fseek(f, 0, SEEK_SET);
    assert (1 == fread(&elfHdr, sizeof(elfHdr), 1, f));

    elfHdr.e_ident[EI_OSABI] = ELFOSABI_SYSV;
    elfHdr.e_type = ET_DYN;

    // patches: 
    // change ABI to SYSV (TODO may need to change for syscalls)
    // possibly delete PT_SCE_DYNLIBDATA if errors, probably just need to delete pHeader
    // remove SCE stuff from DynTable
    // add .dynamic section that points to top of PT_DYNAMIC segment
    // possibly add new PT_DYNAMIC segment, delete old one, and put both Dyn ents and dynamic data here
    //      probably dont need this ^

    progHdrs.resize(elfHdr.e_phnum);
    fseek(f, elfHdr.e_phoff, SEEK_SET);
    assert(elfHdr.e_phnum == fread(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    int idx = findPhdr(progHdrs, PT_DYNAMIC);
    assert(idx >= 0);
    Elf64_Phdr *dynamicPHdr = &progHdrs[idx];
    patchDynamicSegment(progHdrs, f);

    // Append new sections (data) to the EOF
    // then append the section headers array
    // keep updating the string table (strtab) as we go

    //typedef struct
    //{
    //    Elf64_Word	sh_name;		/* Section name (string tbl index) */
    //    Elf64_Word	sh_type;		/* Section type */
    //    Elf64_Xword	sh_flags;		/* Section flags */
    //    Elf64_Addr	sh_addr;		/* Section virtual addr at execution */
    //    Elf64_Off	sh_offset;		/* Section file offset */
    //    Elf64_Xword	sh_size;		/* Section size in bytes */
    //    Elf64_Word	sh_link;		/* Link to another section */
    //    Elf64_Word	sh_info;		/* Additional section information */
    //    Elf64_Xword	sh_addralign;		/* Section alignment */
    //    Elf64_Xword	sh_entsize;		/* Entry size if section holds table */
    //} Elf64_Shdr;

    newSectionHdrs.push_back({
        0,
        SHT_NULL, // type
        0, 0, 0, 0, 0, 0, 0, 0
    });

    newSectionHdrs.push_back({
        appendToStrtab(newStrtab, ".strtab"),
        SHT_STRTAB,
        SHF_STRINGS,
        0,
        0, // file offset to be updated before writing strtab data
        0, // size, same as above ^
        0,
        0,
        section_alignment,
        0
    });
    strtabIdx = newSectionHdrs.size() - 1;

    // TODO this assumes we are reusing the old phdr/dynamic section and overwriting that
    newSectionHdrs.push_back({
        appendToStrtab(newStrtab, ".dynamic"),
        SHT_DYNAMIC,
        0,
        0,
        dynamicPHdr->p_offset,
        dynamicPHdr->p_memsz,
        static_cast<Elf64_Word>(strtabIdx),
        0,// sh_info
        dynamicPHdr->p_align, // alignment
        sizeof(Elf64_Dyn)
    });

    // TODO look into SHT_DYNSYM, SHT_SYMTAB

    // append strtab to file
    fseek(f, 0, SEEK_END);    
    writePadding(f, section_alignment);
    size_t strtabOff = ftell(f);
    assert(1 == fwrite(newStrtab.data(), newStrtab.size(), 1, f));

    // append sections to file
    // correct strtab offset in strtab entry
    fseek(f, 0, SEEK_END);    
    writePadding(f, section_alignment);
    size_t shOff = ftell(f);
    newSectionHdrs[strtabIdx].sh_offset = strtabOff;
    newSectionHdrs[strtabIdx].sh_size = newStrtab.size();
    assert(newSectionHdrs.size() == fwrite(newSectionHdrs.data(), sizeof(Elf64_Shdr), newSectionHdrs.size(), f));

    // update program headers to file
    fseek(f, elfHdr.e_phoff, SEEK_SET);
    assert(elfHdr.e_phnum == fwrite(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    // update ELF header to file
    elfHdr.e_shoff = shOff;
    elfHdr.e_shnum = newSectionHdrs.size();
    elfHdr.e_shstrndx = strtabIdx;
    //elfHdr.e_ident[0] = 'X';
    // elfHdr.e_shentsize TODO
    fseek(f, 0, SEEK_SET);
    assert(1 == fwrite(&elfHdr, sizeof(elfHdr), 1, f));    

    lib.strtab = newStrtab;

    fsync(fileno(f));
    fclose(f);

    for (auto sHdr: newSectionHdrs) {
        dumpShdr(lib, &sHdr);
    }

    return true;
}

bool dumpHostElfSectionsHdrs(HostModule &mod) {
    FILE *f = fopen(mod.path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "couldn't open %s\n", mod.path.c_str());
        return false;
    }

    Elf64_Ehdr elfHdr;
    fseek(f, 0, SEEK_SET);
    if (1 != fread(&elfHdr, sizeof(elfHdr), 1, f)) {
        return false;
    }

    fseek(f, elfHdr.e_shoff, SEEK_SET);

    std::vector<Elf64_Shdr> sectionHdrs(elfHdr.e_shnum);
    assert(elfHdr.e_shnum == fread(sectionHdrs.data(), sizeof(Elf64_Shdr), elfHdr.e_shnum, f));

    fclose(f);

    return true;
}

int main(int argc, char **argv) {
    //FILE *eboot;
    void *dl;
    bool ret;
    
    parseCmdArgs(argc, argv);

    Ps4Module module;
    module.path = CmdArgs.dlPath;

    std::string newPath;
    ret = patchPs4Lib(module, newPath);

    if (!ret)
        return -1;

    if (CmdArgs.dumpElfHeader) {
        dumpElfHdr(module.path.c_str());
        dumpElfHdr(newPath.c_str());
    }

    dl = dlopen(newPath.c_str(), RTLD_LAZY);
    if (!dl) {
        char *err;
        while ((err = dlerror())) {
            printf("%s\n", err);
        }
        return -1;
    }

    return 0;
}