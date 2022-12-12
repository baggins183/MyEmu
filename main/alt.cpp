#include "Common.h"
#include "elf-sce.h"
#include "elf_amd64.h"
#include <sys/types.h>
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
    OP(NULL, "") \
    OP(STRTAB, ".strtab") \
    OP(SHSTRTAB, ".shstrtab") \
    OP(DYNAMIC, ".dynamic") \
    OP(DYNSTR, ".dynstr") \
    OP(DYNSYM, ".dynsym") \
    OP(INIT, ".init") \
    OP(FINI, ".fini") \
    OP(PLT, ".plt") \
    OP(GOT, ".got") \
    OP(PLTGOT, ".plt.got") \
    OP(RELA, ".rela" ) \
    OP(SCE_DYNSTR, ".dynstr") \
    OP(SCE_DYNSYM, ".dynsym") \
    OP(DT_SCE_PLTGOT, ".plt.got") \

#define SECTION_ENUM_LIST(e, name) STYPE_##e,
#define SECTION_NAME_LIST(e, name) name,

enum section_type {
    SECTION_TABLE(SECTION_ENUM_LIST)
    NUM_SECTIONS,
};

const char *section_names[] {
    SECTION_TABLE(SECTION_NAME_LIST)
};

struct Section {
    std::vector<unsigned char> contents;
    Elf64_Shdr hdr;
    section_type type;

    Section() {}
    Section(Elf64_Shdr hdr): hdr(hdr) {}
    Section(Elf64_Shdr hdr, section_type type): hdr(hdr), type(type) {}
};

class SectionMap {
private:
    std::map<section_type, int> typeToIndex;
    std::map<section_type, Section> typeToSection;

    int numSections;
public:
    SectionMap(): numSections(0) {}

    int getSectionIndex(section_type type) {
        if (typeToIndex.find(type) != typeToIndex.end()) {
            return typeToIndex[type];
        }
        assert("section doesn't exist" && false);
        return -1;
    }

    Section &getSection(section_type type) {
        assert(typeToSection.find(type) != typeToSection.end());
        return typeToSection[type];
    }    

    Section &addSection(section_type type, Elf64_Shdr hdr) {
        if (typeToSection.find(type) != typeToSection.end()) {
            return typeToSection[type];
        }

        typeToIndex[type] = numSections++;
        typeToSection[type] = Section(hdr, type);
        return typeToSection[type];
    }

    // collect headers in order
    std::vector<Elf64_Shdr> getHeaders() {
        size_t nSections = typeToSection.size();
        std::vector<Elf64_Shdr> ret(nSections);

        section_type type;
        int idx;
        for (auto kv : typeToIndex) {
            std::tie(type, idx) = kv;
            ret[idx] = typeToSection[type].hdr;
        }
        return ret;
    }
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
        "\tsh_type: %u\n"
        "\tsh_addr: %lu\n"
        "\tsh_offset: %lu\n"
        "\tsh_size: %lu\n"
        "\tsh_link: %i\n"
        "\tsh_info: %i\n"
        "\tsh_entsize: %lu\n",
        (char *) &mod.strtab[sHdr->sh_name],
        sHdr->sh_type,
        sHdr->sh_addr,
        sHdr->sh_offset,
        sHdr->sh_size,
        sHdr->sh_link,
        sHdr->sh_info,
        sHdr->sh_entsize
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

bool writePadding(FILE *f, size_t alignment) {
    fseek(f, 0, SEEK_END);
    size_t flen = ftell(f);

    size_t padlen = ROUND_UP(flen, alignment) - flen;
    std::vector<unsigned char> padding(padlen, 0);
    assert(1 == fwrite(padding.data(), padlen, 1, f));

    return true;
}

// return index into strtab
uint appendToStrtab(Section &strtab, const char *str) {
    int off = strtab.contents.size();
    int addedSz = strlen(str) + 1;
    strtab.contents.resize(off + addedSz);
    memcpy(&strtab.contents[off], str, addedSz);

    return off;
}

Segment CreateSegment(Elf64_Phdr pHdr, std::vector<section_type> &sections, SectionMap &sectionMap) {
    Segment seg;
    seg.pHdr = pHdr;

    uint64_t segBeginVa = pHdr.p_vaddr;
    uint64_t segBeginFileOff = pHdr.p_offset;
    for (section_type stype: sections) {
        Section &section = sectionMap.getSection(stype);
        uint64_t segOff = seg.contents.size();
        if (section.contents.size() > 0) {
            seg.contents.resize(segOff + ROUND_UP(section.contents.size(), PGSZ));
            memcpy(&seg.contents[segOff], &section.contents[0], section.contents.size());
        }
        section.hdr.sh_addr = segBeginVa + segOff;
        section.hdr.sh_offset = segBeginFileOff + segOff;
        section.hdr.sh_flags |= PROT_EXEC | PROT_WRITE | PROT_READ; // TODO remove
    }

    seg.pHdr.p_filesz = seg.contents.size();
    seg.pHdr.p_memsz = seg.contents.size(); // ???

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
    pHdr->p_vaddr = ROUND_UP(highestVAddr, PGSZ);
    pHdr->p_paddr = ROUND_UP(highestPAddr, PGSZ);    
}

// write new Segment for PT_DYNAMIC based on SCE dynamic entries
// append to end of file
// modify the progHdrs array
// also add existing SCE sections to oldSectionMap
static bool patchDynamicSegment(FILE *elf, std::vector<Elf64_Phdr> &progHdrs, SectionMap &newSectionMap, SectionMap &oldSectionMap) {
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

    writePadding(elf, PGSZ);
    uint64_t segmentFileOff = ftell(elf);
    assert(1 == fwrite(newEntries.data(), newEntries.size() * sizeof(Elf64_Dyn), 1, elf));

    // calculate p_vaddr, p_paddr
    rebaseSegment(dynPhdr, progHdrs);
    //dynPhdr->p_align = PGSZ;
    dynPhdr->p_flags = PF_X | PF_W | PF_R;
    dynPhdr->p_offset = segmentFileOff;
    dynPhdr->p_filesz = newEntries.size() * sizeof(Elf64_Dyn);

    return true;
}

bool patchPs4Lib(Ps4Module &lib, /* ret */ std::string &newPath) {
    std::vector<Elf64_Phdr> progHdrs;
    // in a Ps4 module, these Dyn Ents are in the DYNAMIC segment/.dynamic section,
    // and describe the DYNLIBDATA segment
    std::vector<Elf64_Dyn> oldPs4DynEnts;    
    std::vector<Elf64_Dyn> newElfDynEnts;
    // index of the strtab in the shEntries

    SectionMap newSectionMap;
    SectionMap oldSections;

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
    patchDynamicSegment(f, progHdrs, newSectionMap, oldSections);

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

    /*Elf64_Shdr firstHdr = {
        .sh_type = SHT_NULL
    };
    newSectionMap.addSection(STYPE_NULL, firstHdr);*/

    Elf64_Shdr shstrtabSHdr = {
        .sh_name = 0,
        .sh_type = SHT_STRTAB,
        .sh_flags = SHF_STRINGS
    };
    Section &shstrtab = newSectionMap.addSection(STYPE_SHSTRTAB, shstrtabSHdr);
    appendToStrtab(shstrtab, section_names[STYPE_SHSTRTAB]);

    Elf64_Shdr dynstrSHdr = {
        .sh_name = appendToStrtab(shstrtab, section_names[STYPE_DYNSTR]),
        .sh_type = SHT_STRTAB,
        .sh_flags = SHF_STRINGS | SHF_ALLOC
    };
    Section &dynstr = newSectionMap.addSection(STYPE_DYNSTR, dynstrSHdr);

    Elf64_Shdr dynsymSHdr = {
        .sh_name = appendToStrtab(shstrtab, section_names[STYPE_DYNSYM]),
        .sh_type = SHT_SYMTAB,
        .sh_flags = SHF_WRITE | SHF_ALLOC,
        .sh_link = static_cast<Elf64_Word>(newSectionMap.getSectionIndex(STYPE_DYNSTR)),
        .sh_entsize = sizeof(Elf64_Sym)
    };
    Section &dynsym = newSectionMap.addSection(STYPE_DYNSYM, dynsymSHdr);

    Elf64_Shdr dynamicSHdr = {
        .sh_name = appendToStrtab(shstrtab, section_names[STYPE_DYNAMIC]),
        .sh_type = SHT_DYNAMIC,
        .sh_flags = SHF_WRITE | SHF_ALLOC,
        .sh_addr = dynamicPHdr->p_vaddr,
        .sh_offset = dynamicPHdr->p_offset,
        .sh_size = dynamicPHdr->p_memsz,
        .sh_link = static_cast<Elf64_Word>(newSectionMap.getSectionIndex(STYPE_DYNSTR)),
        .sh_addralign = dynamicPHdr->p_align,
        .sh_entsize = sizeof(Elf64_Dyn)        
    };
    newSectionMap.addSection(STYPE_DYNAMIC, dynamicSHdr);

    std::vector<section_type> extraDynSections = {
        STYPE_DYNSTR,
        STYPE_DYNSYM,
    };
    Elf64_Phdr extraDynSegmentHdr {
        .p_type = PT_LOAD,
        .p_flags = PF_R
    };
    rebaseSegment(&extraDynSegmentHdr, progHdrs);
    writePadding(f, PGSZ);
    extraDynSegmentHdr.p_offset = ftell(f);
    Segment miscLoadSegment = CreateSegment(extraDynSegmentHdr, extraDynSections, newSectionMap);
    fseek(f, 0, SEEK_END);
    assert(1 == fwrite(miscLoadSegment.contents.data(), miscLoadSegment.contents.size(), 1, f));

    // update program headers to file
    fseek(f, 0, SEEK_END);
    elfHdr.e_phoff = ftell(f);
    elfHdr.e_phnum = progHdrs.size();
    assert(elfHdr.e_phnum == fwrite(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    fseek(f, 0, SEEK_END);
    elfHdr.e_shoff = ftell(f);
    elfHdr.e_shstrndx = newSectionMap.getSectionIndex(STYPE_SHSTRTAB);
    //elfHdr.e_shnum = 
    std::vector<Elf64_Shdr> sectionHeaders = newSectionMap.getHeaders();
    elfHdr.e_shnum = sectionHeaders.size();
    assert(elfHdr.e_shnum == fwrite(progHdrs.data(), sizeof(Elf64_Shdr), elfHdr.e_shnum, f));
    
    fseek(f, 0, SEEK_SET);
    assert(1 == fwrite(&elfHdr, sizeof(Elf64_Ehdr), 1, f));

    // TODO add misc static sections (shstrtab, etc)
    assert(false);

    /*lib.strtab.resize(shstrtab.contents.size());
    memcpy(lib.strtab.data(), shstrtab.contents.data(), shstrtab.contents.size());
    for (int i = 0; i < sectionHeaders.size(); i++) {
        Elf64_Shdr sHdr = sectionHeaders[i];
        printf("index: %i\n", i);
        dumpShdr(lib, &sHdr);
    }*/

    fsync(fileno(f));
    fclose(f);

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