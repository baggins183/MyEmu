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

const long pgsz = sysconf(_SC_PAGE_SIZE);
const std::string ebootPath = "eboot.bin";

#define ROUND_DOWN(x, SZ) ((x) - (x) % (SZ))
#define ROUND_UP(x, SZ) ( (x) % (SZ) ? (x) - ((x) % (SZ)) + (SZ) : (x))

struct MappedSegment {
    uint64_t firstPage;
    uint64_t nPages;
};

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
    std::vector<MappedSegment> mappedSegments;
    std::vector<Elf64_Rela> relocs;
    std::vector<Elf64_Sym> symbols;
};

class HostModule : public Module {

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
        "\te_shstrndx: %d\n"
        "\tABI: %hhu\n",
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
        elfHdr->e_shstrndx,
        elfHdr->e_ident[EI_OSABI]
    );
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

bool patchPs4Lib(Ps4Module &lib, /* ret */ std::string &newPath) {
    // alignment for new sections (data) we add
    const size_t section_alignment = 16;

    std::vector<Elf64_Phdr> progHdrs;
    std::vector<Elf64_Phdr> newProgHdrs;
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
    Elf64_Phdr dynamicPHdr = progHdrs[idx];
    size_t numDynEntries = 0;

    std::vector<unsigned char> buf;
    buf.resize(dynamicPHdr.p_filesz);

    Elf64_Dyn *dynEnt = (Elf64_Dyn *) buf.data();
    while (numDynEntries * sizeof(Elf64_Dyn) < buf.size()) {
        numDynEntries++;
        if (dynEnt->d_tag == DT_NULL) {
            break;
        }
        dynEnt++;
    }

    oldPs4DynEnts.resize(numDynEntries);
    memcpy(oldPs4DynEnts.data(), buf.data(), numDynEntries * sizeof(Elf64_Dyn));

    for (auto dynEnt: oldPs4DynEnts) {
        switch (dynEnt.d_tag) {
            case DT_NULL:
                newElfDynEnts.push_back(dynEnt);
                break;
            default:
                break;
        }
    }

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
        appendToStrtab(newStrtab, "strtab"),
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
        appendToStrtab(newStrtab, "dynamic"),
        SHT_DYNAMIC,
        0,
        0,
        dynamicPHdr.p_offset,
        dynamicPHdr.p_memsz,
        static_cast<Elf64_Word>(strtabIdx),
        0,// sh_info
        dynamicPHdr.p_align, // alignment
        sizeof(Elf64_Dyn)
    });

    // TODO look into SHT_DYNSYM, SHT_SYMTAB

    // append strtab to file
    writePadding(f, section_alignment);
    size_t strtabOff = ftell(f);
    assert(1 == fwrite(newStrtab.data(), newStrtab.size(), 1, f));

    // append sections to file
    // correct strtab offset in strtab entry
    writePadding(f, section_alignment);
    size_t shOff = ftell(f);
    newSectionHdrs[strtabIdx].sh_offset = strtabOff;
    newSectionHdrs[strtabIdx].sh_size = newStrtab.size();
    assert(newSectionHdrs.size() == fwrite(newSectionHdrs.data(), sizeof(Elf64_Shdr), newSectionHdrs.size(), f));

    // update ELF header to file
    elfHdr.e_shoff = shOff;
    elfHdr.e_shnum = newSectionHdrs.size();
    elfHdr.e_shstrndx = strtabIdx;
    //elfHdr.e_ident[0] = 'X';
    // elfHdr.e_shentsize TODO
    fseek(f, 0, SEEK_SET);
    assert(1 == fwrite(&elfHdr, sizeof(elfHdr), 1, f));    

    lib.strtab = newStrtab;
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

    dl = dlopen(newPath.c_str(), RTLD_LAZY);
    if (!dl) {
        printf("%s\n", dlerror());
        return -1;
    }

    return 0;
}