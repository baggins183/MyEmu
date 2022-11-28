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

bool patchPs4Lib(Ps4Module &lib, /* ret */ std::string &newPath) {
    std::vector<Elf64_Phdr> newPhdhrs;
    std::vector<std::vector<unsigned char>> newSegments;

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
    if (1 != fread(&elfHdr, sizeof(elfHdr), 1, f)) {
        return false;
    }

    elfHdr.e_ident[EI_OSABI] = ELFOSABI_SYSV;
    elfHdr.e_type = ET_DYN;

    fseek(f, 0, SEEK_SET);
    if (1 != fwrite(&elfHdr, sizeof(elfHdr), 1, f)) {
        return false;
    }

    // patches: 
    // change ABI to SYSV (TODO may need to change for syscalls)
    // insert dynamic segment pheader
    // append dynamic segment
    // possibly delete PT_SCE_DYNLIBDATA if errors, probably just need to delete pHeader


    std::vector<Elf64_Phdr> progHdrs;
    progHdrs.resize(elfHdr.e_phnum);
    fseek(f, elfHdr.e_phoff, SEEK_SET);
    if (elfHdr.e_phnum != fread(progHdrs.data(), sizeof(Elf64_Phdr), elfHdr.e_phnum, f)) {
        return false;
    }

    dumpElfHdr(newPath.c_str(), &elfHdr);

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
}