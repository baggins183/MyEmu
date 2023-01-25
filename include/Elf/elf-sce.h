#pragma once

// Sony SCE ELF specified types

// define some enums for debugging convenience


// ELF Types
#define ET_SCE_EXEC	0xFE00
#define ET_SCE_REPLAY_EXEC	0xFE01
#define ET_SCE_RELEXEC	0XFE04
#define ET_SCE_STUBLIB	0xFE0C
#define ET_SCE_DYNEXEC	0xFE10
#define ET_SCE_DYNAMIC	0xFE18

#undef PT_NULL
#undef PT_LOAD
#undef PT_DYNAMIC
#undef PT_INTERP
#undef PT_NOTE
#undef PT_SHLIB
#undef PT_PHDR
#undef PT_TLS
#undef PT_NUM
#undef PT_LOOS
#undef PT_GNU_EH_FRAME
#undef PT_GNU_STACK
#undef PT_GNU_RELRO
#undef PT_GNU_PROPERTY
#undef PT_LOSUNW
#undef PT_SUNWBSS
#undef PT_SUNWSTACK
#undef PT_HISUNW
#undef PT_HIOS
#undef PT_LOPROC
#undef PT_HIPROC

#define PT_TABLE(OP) \
    OP(PT_NULL,		0) \
    OP(PT_LOAD,		1) \
    OP(PT_DYNAMIC,	2) \
    OP(PT_INTERP,	3) \
    OP(PT_NOTE,		4) \
    OP(PT_SHLIB,	5) \
    OP(PT_PHDR,		6) \
    OP(PT_TLS,		7) \
    OP(PT_NUM,		8) \
    OP(PT_LOOS,		0x60000000) \
    OP(PT_GNU_EH_FRAME,	0x6474e550) \
    OP(PT_GNU_STACK,	0x6474e551) \
    OP(PT_GNU_RELRO,	0x6474e552) \
    OP(PT_GNU_PROPERTY,	0x6474e553) \
    OP(PT_LOSUNW,	0x6ffffffa) \
    OP(PT_SUNWBSS,	0x6ffffffa) \
    OP(PT_SUNWSTACK,	0x6ffffffb) \
    OP(PT_HISUNW,	0x6fffffff) \
    OP(PT_HIOS,		0x6fffffff) \
    OP(PT_LOPROC,	0x70000000) \
    OP(PT_HIPROC,	0x7fffffff)

// Program Segment Type
#define PT_SCE_TABLE(OP) \
    OP(PT_SCE_RELA,	0x60000000) \
    OP(PT_SCE_DYNLIBDATA,	0x61000000) \
    OP(PT_SCE_PROCPARAM,	0x61000001) \
    OP(PT_SCE_MODULEPARAM,	0x61000002) \
    OP(PT_SCE_RELRO,	0x61000010) \
    OP(PT_SCE_COMMENT,	0X6FFFFF00) \
    OP(PT_SCE_LIBVERSION,	0X6FFFFF01)

#define PT_ENUM_OP(name, value) \
    name = value,

enum ProgramSegmentType {
    PT_TABLE(PT_ENUM_OP)
    PT_SCE_TABLE(PT_ENUM_OP)
};

#undef SHT_NULL
#undef SHT_PROGBITS
#undef SHT_SYMTAB
#undef SHT_STRTAB
#undef SHT_RELA
#undef SHT_HASH
#undef SHT_DYNAMIC
#undef SHT_NOTE
#undef SHT_NOBITS
#undef SHT_REL
#undef SHT_SHLIB
#undef SHT_DYNSYM
#undef SHT_INIT_ARRAY
#undef SHT_FINI_ARRAY
#undef SHT_PREINIT_ARRAY
#undef SHT_GROUP
#undef SHT_SYMTAB_SHNDX
#undef SHT_RELR
#undef SHT_NUM
#undef SHT_LOOS
#undef SHT_GNU_ATTRIBUTES
#undef SHT_GNU_HASH
#undef SHT_GNU_LIBLIST
#undef SHT_CHECKSUM
#undef SHT_LOSUNW
#undef SHT_SUNW_move
#undef SHT_SUNW_COMDAT
#undef SHT_SUNW_syminfo
#undef SHT_GNU_verdef
#undef SHT_GNU_verneed
#undef SHT_GNU_versym
#undef SHT_HISUNW
#undef SHT_HIOS
#undef SHT_LOPROC
#undef SHT_HIPROC
#undef SHT_LOUSER
#undef SHT_HIUSER

/* Legal values for sh_type (section type).  */
#define SHT_TABLE(OP) \
    OP(SHT_NULL,	  0) \
    OP(SHT_PROGBITS,	  1) \
    OP(SHT_SYMTAB,	  2) \
    OP(SHT_STRTAB,	  3) \
    OP(SHT_RELA,	  4) \
    OP(SHT_HASH,	  5) \
    OP(SHT_DYNAMIC,	  6) \
    OP(SHT_NOTE,	  7) \
    OP(SHT_NOBITS,	  8) \
    OP(SHT_REL,		  9) \
    OP(SHT_SHLIB,	  10) \
    OP(SHT_DYNSYM,	  11) \
    OP(SHT_INIT_ARRAY,	  14) \
    OP(SHT_FINI_ARRAY,	  15) \
    OP(SHT_PREINIT_ARRAY, 16) \
    OP(SHT_GROUP,	  17) \
    OP(SHT_SYMTAB_SHNDX,  18) \
    OP(SHT_RELR,	  19) \
    OP(SHT_NUM,		  20) \
    OP(SHT_LOOS,	  0x60000000) \
    OP(SHT_GNU_ATTRIBUTES, 0x6ffffff5) \
    OP(SHT_GNU_HASH,	  0x6ffffff6) \
    OP(SHT_GNU_LIBLIST,	  0x6ffffff7) \
    OP(SHT_CHECKSUM,	  0x6ffffff8) \
    OP(SHT_LOSUNW,	  0x6ffffffa) \
    OP(SHT_SUNW_move,	  0x6ffffffa) \
    OP(SHT_SUNW_COMDAT,   0x6ffffffb) \
    OP(SHT_SUNW_syminfo,  0x6ffffffc) \
    OP(SHT_GNU_verdef,	  0x6ffffffd) \
    OP(SHT_GNU_verneed,	  0x6ffffffe) \
    OP(SHT_GNU_versym,	  0x6fffffff) \
    OP(SHT_HISUNW,	  0x6fffffff) \
    OP(SHT_HIOS,	  0x6fffffff) \
    OP(SHT_LOPROC,	  0x70000000) \
    OP(SHT_HIPROC,	  0x7fffffff) \
    OP(SHT_LOUSER,	  0x80000000) \
    OP(SHT_HIUSER,	  0x8fffffff)

#define SHT_ENUM_OP(name, value) \
    name = value,

enum ShtType {
    SHT_TABLE(SHT_ENUM_OP)
};


#undef DT_NULL
#undef DT_NEEDED
#undef DT_PLTRELSZ
#undef DT_PLTGOT
#undef DT_HASH
#undef DT_STRTAB
#undef DT_SYMTAB
#undef DT_RELA
#undef DT_RELASZ
#undef DT_RELAENT
#undef DT_STRSZ
#undef DT_SYMENT
#undef DT_INIT
#undef DT_FINI
#undef DT_SONAME
#undef DT_RPATH
#undef DT_SYMBOLIC
#undef DT_REL
#undef DT_RELSZ
#undef DT_RELENT
#undef DT_PLTREL
#undef DT_DEBUG
#undef DT_TEXTREL
#undef DT_JMPREL
#undef DT_BIND_NOW
#undef DT_INIT_ARRAY
#undef DT_FINI_ARRAY
#undef DT_INIT_ARRAYSZ
#undef DT_FINI_ARRAYSZ
#undef DT_RUNPATH
#undef DT_FLAGS
#undef DT_PREINIT_ARRAY
#undef DT_PREINIT_ARRAYSZ
#undef DT_SYMTAB_SHNDX
#undef DT_RELRSZ
#undef DT_RELR
#undef DT_RELRENT
#undef DT_NUM
#undef DT_LOOS
#undef DT_HIOS
#undef DT_LOPROC
#undef DT_HIPROC
#undef DT_PROCNUM

#define DT_TABLE(OP) \
    OP(DT_NULL,		0) \
    OP(DT_NEEDED,	1) \
    OP(DT_PLTRELSZ,	2) \
    OP(DT_PLTGOT,	3) \
    OP(DT_HASH,		4) \
    OP(DT_STRTAB,	5) \
    OP(DT_SYMTAB,	6) \
    OP(DT_RELA,		7) \
    OP(DT_RELASZ,	8) \
    OP(DT_RELAENT,	9) \
    OP(DT_STRSZ,	10) \
    OP(DT_SYMENT,	11) \
    OP(DT_INIT,		12) \
    OP(DT_FINI,		13) \
    OP(DT_SONAME,	14) \
    OP(DT_RPATH,	15) \
    OP(DT_SYMBOLIC,	16) \
    OP(DT_REL,		17) \
    OP(DT_RELSZ,	18) \
    OP(DT_RELENT,	19) \
    OP(DT_PLTREL,	20) \
    OP(DT_DEBUG,	21) \
    OP(DT_TEXTREL,	22) \
    OP(DT_JMPREL,	23) \
    OP(DT_BIND_NOW,	24) \
    OP(DT_INIT_ARRAY,	25) \
    OP(DT_FINI_ARRAY,	26) \
    OP(DT_INIT_ARRAYSZ,	27) \
    OP(DT_FINI_ARRAYSZ,	28) \
    OP(DT_RUNPATH,	29) \
    OP(DT_FLAGS,	30) \
    OP(DT_PREINIT_ARRAY, 32) \
    OP(DT_PREINIT_ARRAYSZ, 33) \
    OP(DT_SYMTAB_SHNDX,	34) \
    OP(DT_RELRSZ,	35) \
    OP(DT_RELR,		36) \
    OP(DT_RELRENT,	37) \
    OP(DT_NUM,		38) \
    OP(DT_LOOS,		0x6000000d) \
    OP(DT_HIOS,		0x6ffff000) \
    OP(DT_LOPROC,	0x70000000) \
    OP(DT_HIPROC,	0x7fffffff) \
    OP(DT_PROCNUM,	0x37)

#define DT_SCE_TABLE(OP) \
    OP(DT_SCE_IDTABENTSZ,		0x61000005) \
    OP(DT_SCE_FINGERPRINT,		0x61000007) \
    OP(DT_SCE_ORIGINAL_FILENAME,		0x61000009) \
    OP(DT_SCE_MODULE_INFO,		0x6100000D) \
    OP(DT_SCE_NEEDED_MODULE,		0x6100000F) \
    OP(DT_SCE_MODULE_ATTR,		0x61000011) \
    OP(DT_SCE_EXPORT_LIB,		0x61000013) \
    OP(DT_SCE_IMPORT_LIB,		0x61000015) \
    OP(DT_SCE_EXPORT_LIB_ATTR,		0x61000017) \
    OP(DT_SCE_IMPORT_LIB_ATTR,		0x61000019) \
    OP(DT_SCE_STUB_MODULE_NAME,		0x6100001D) \
    OP(DT_SCE_STUB_MODULE_VERSION,		0x6100001F) \
    OP(DT_SCE_STUB_LIBRARY_NAME,		0x61000021) \
    OP(DT_SCE_STUB_LIBRARY_VERSION,		0x61000023) \
    OP(DT_SCE_HASH,		0x61000025) \
    OP(DT_SCE_PLTGOT,		0x61000027) \
    OP(DT_SCE_JMPREL,		0x61000029) \
    OP(DT_SCE_PLTREL,		0x6100002B) \
    OP(DT_SCE_PLTRELSZ,		0x6100002D) \
    OP(DT_SCE_RELA,		0x6100002F) \
    OP(DT_SCE_RELASZ,		0x61000031) \
    OP(DT_SCE_RELAENT,		0x61000033) \
    OP(DT_SCE_STRTAB,		0x61000035) \
    OP(DT_SCE_STRSZ,		0x61000037) \
    OP(DT_SCE_SYMTAB,		0x61000039) \
    OP(DT_SCE_SYMENT,		0x6100003B) \
    OP(DT_SCE_HASHSZ,		0x6100003D) \
    OP(DT_SCE_SYMTABSZ,		0x6100003F) \
    OP(DT_SCE_HIOS,		0X6FFFF000)

#define DT_ENUM_OP(name, value) \
    name = value,

enum DynamicTag {
    DT_TABLE(DT_ENUM_OP)
    DT_SCE_TABLE(DT_ENUM_OP)
};