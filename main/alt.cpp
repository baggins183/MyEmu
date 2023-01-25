#include "Common.h"
#include "nid_hash/nid_hash.h"

#include <elf.h>
#include <sys/types.h>
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

#include <algorithm>
#include <dlfcn.h>
#include <libgen.h>

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

    Elf32_Word getName() { return hdr.sh_name; }
    Elf32_Word getType() { return hdr.sh_type; }
    Elf32_Word getFlags() { return hdr.sh_flags; }
    Elf32_Addr getAddr() { return hdr.sh_addr; }
    Elf32_Off getOffset() { return hdr.sh_offset; }
    Elf32_Word getSize() { return hdr.sh_size; }
    Elf32_Word getLink() { return hdr.sh_link; }
    Elf32_Word getInfo() { return hdr.sh_info; }
    Elf32_Word getAddrAlign() { return hdr.sh_addralign; }
    Elf32_Word getEntSize() { return hdr.sh_entsize; }

    void setName(Elf32_Word name) { hdr.sh_name = name; }
    void setType(Elf32_Word type) { hdr.sh_type = type; }
    void setFlags(Elf32_Word flags) { hdr.sh_flags = flags; }
    void setAddr(Elf32_Addr addr) { hdr.sh_addr = addr; }
    void setOffset(Elf32_Off offset) { hdr.sh_offset = offset; }
    void setSize(Elf32_Word size) { hdr.sh_size = size; }
    void setLink(Elf32_Word link) { hdr.sh_link = link; }
    void setInfo(Elf32_Word info) { hdr.sh_info = info; }
    void setAddrAlign(Elf32_Word addralign) { hdr.sh_addralign = addralign; }
    void setEntSize(Elf32_Word entsize) { hdr.sh_entsize = entsize; }

    Elf64_Shdr hdr;
    std::vector<unsigned char> contents;
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
uint appendToStrtab(Section &strtab, const char *str) {
    int off = strtab.contents.size();
    int addedSz = strlen(str) + 1;
    strtab.contents.resize(off + addedSz);
    memcpy(&strtab.contents[off], str, addedSz);

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

// write new Segment for PT_DYNAMIC based on SCE dynamic entries
// append to end of file
// modify the progHdrs array
static bool patchDynamicSegment(FILE *elf, std::vector<Elf64_Phdr> &progHdrs) {
    int idx = findPhdr(progHdrs, PT_DYNAMIC);
    assert(idx >= 0);
    Elf64_Phdr *dynPhdr = &progHdrs[idx];

    std::vector<unsigned char> buf(dynPhdr->p_filesz);

    fseek(elf, dynPhdr->p_offset, SEEK_SET);
    assert(1 == fread(buf.data(), dynPhdr->p_filesz, 1, elf));

    std::vector<Elf64_Dyn> newEntries;

    Elf64_Dyn *dynEnt = (Elf64_Dyn*) buf.data();
    
    int maxDynHeaders = dynPhdr->p_filesz / sizeof(Elf64_Dyn);
    for(int i = 0; i < maxDynHeaders; i++, dynEnt++) {
        DynamicTag tag = (DynamicTag) dynEnt->d_tag;
        if (tag == DT_NULL) {
            newEntries.push_back(*dynEnt);
            break;
        }

        switch(tag) {
            // DT_ tags
            case DT_NULL:
                assert(false && "shouldn't be here");
                break;
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
            default:
                break;
        }

        //uint64_t val = entry.d_un.d_val;
        switch(tag) {
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

    Elf64_Shdr sHdr;

    progHdrs.resize(elfHdr.e_phnum);
    fseek(f, elfHdr.e_phoff, SEEK_SET);
    assert(elfHdr.e_phnum == fread(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    int idx = findPhdr(progHdrs, PT_DYNAMIC);
    assert(idx >= 0);
    Elf64_Phdr *dynamicPHdr = &progHdrs[idx];
    patchDynamicSegment(f, progHdrs);

    std::vector<Section> sections;

    sHdr = {
        .sh_name = 0,
        .sh_type = SHT_NULL
    };
    sections.emplace_back(sHdr);

    uint shstrtabIdx = sections.size();
    sHdr = {
        .sh_type = SHT_STRTAB,
        .sh_flags = 0,
        .sh_addralign = 1,
    };
    sections.emplace_back(sHdr);
    appendToStrtab(sections[shstrtabIdx], "\0");
    sections[shstrtabIdx].setName(appendToStrtab(sections[shstrtabIdx], ".shstrtab"));

    uint dynstrIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[shstrtabIdx], ".dynstr"),
        .sh_type = SHT_STRTAB,
        .sh_flags = SHF_STRINGS | SHF_ALLOC,
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
    };
    sections.emplace_back(sHdr);

    uint dynsymIdx = sections.size();
    sHdr = {
        .sh_name = appendToStrtab(sections[shstrtabIdx], ".dynsym"),
        .sh_type = SHT_SYMTAB,
        .sh_flags = SHF_WRITE | SHF_ALLOC,
        .sh_link = dynstrIdx,
        .sh_addralign = static_cast<Elf64_Xword>(PGSZ),
        .sh_entsize = sizeof(Elf64_Sym),
    };
    sections.emplace_back(sHdr);

    sHdr = {
        .sh_name = appendToStrtab(sections[shstrtabIdx], ".dynamic"),
        .sh_type = SHT_DYNAMIC,
        .sh_flags = SHF_WRITE | SHF_ALLOC,
        .sh_addr = dynamicPHdr->p_vaddr,
        .sh_offset = dynamicPHdr->p_offset,
        .sh_size = dynamicPHdr->p_memsz,
        .sh_link = dynstrIdx,
        .sh_addralign = dynamicPHdr->p_align,
        .sh_entsize = sizeof(Elf64_Dyn),
    };
    sections.emplace_back(sHdr);
    
    // Create dynstr, dynsym sections


    // Add sections used in relocs, loading dynamic dependencies, etc
    // Use a new segment
    std::vector<uint> dynlibDataSections = {
        dynstrIdx,
        dynsymIdx,
    };
    Elf64_Phdr dynlibDataSegmentHdr {
        .p_type = PT_LOAD,
        .p_flags = PF_R | PF_W | PF_X,
        .p_align = static_cast<Elf64_Xword>(PGSZ)
    };
    rebaseSegment(&dynlibDataSegmentHdr, progHdrs);
    fseek(f, 0, SEEK_END);
    writePadding(f, dynlibDataSegmentHdr.p_align, true);
    dynlibDataSegmentHdr.p_offset = ftell(f);
    Segment dynlibDataSegment = CreateSegment(dynlibDataSegmentHdr, sections, dynlibDataSections);
    if (dynlibDataSegment.pHdr.p_filesz > 0) {
        assert(1 == fwrite(dynlibDataSegment.contents.data(), dynlibDataSegment.contents.size(), 1, f));
    }
    progHdrs.push_back(dynlibDataSegment.pHdr); // TODO 

    // For now, pack the extra sections into their own segment
    // in order to append them. The data is all that matters.
    // Want to make sure they own their own parts of the ELF and
    // future segments see that their region is owned when using rebaseSegment
    std::vector<uint> extraSections {
        shstrtabIdx
    };
    Elf64_Phdr extraSegmentHdr {
        .p_type = PT_NULL,
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

    // write program headers
    fseek(f, 0, SEEK_END);
    elfHdr.e_phoff = ftell(f);
    elfHdr.e_phnum = progHdrs.size();
    assert(elfHdr.e_phnum == fwrite(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f));

    // write section headers
    fseek(f, 0, SEEK_END);
    writePadding(f, PGSZ, true);
    elfHdr.e_shoff = ftell(f);
    elfHdr.e_shstrndx = shstrtabIdx;
    std::vector<Elf64_Shdr> sectionHeaders;
    for (Section &section: sections) {
        sectionHeaders.push_back(section.hdr);
    }
    elfHdr.e_shnum = sectionHeaders.size();
    elfHdr.e_shentsize = sizeof(Elf64_Shdr);
    assert(elfHdr.e_shnum == fwrite(sectionHeaders.data(), sizeof(Elf64_Shdr), elfHdr.e_shnum, f));

    // write elf headers, now referring to new program and section headers
    fseek(f, 0, SEEK_SET);
    assert(1 == fwrite(&elfHdr, sizeof(Elf64_Ehdr), 1, f));

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
    bool res = true;
    
    parseCmdArgs(argc, argv);

    Ps4Module module;
    module.path = CmdArgs.dlPath;

    std::string newPath;
    res = patchPs4Lib(module, newPath);

    if (!res)
        return -1;

    if (CmdArgs.dumpElfHeader) {
        dumpElfHdr(module.path.c_str());
        dumpElfHdr(newPath.c_str());
    }

    printf("hi from main\n");

    //setenv("LD_DEBUG", "all", 1);

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