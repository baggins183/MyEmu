#ifndef _COMMON_H_
#define _COMMON_H_

#include <elf.h>
#include "Elf/elf_amd64.h"

// make sure this gets included after elf.h to undef some stuff (mostly for debugging)
#include "Elf/elf-sce.h"

#ifdef __linux
#include <cstdint>
#endif
#include <stdint.h>
#include "DebugContext.h"

#define ARRLEN(arr) (sizeof(arr) / sizeof(arr[0]))

typedef void (*PFUNC_ExitFunction)();
typedef void* (*PFUNC_GameTheadEntry)(void* pArg, PFUNC_ExitFunction pExit);

typedef struct {
    uint64_t argc;
    const char *argv[1];
} Ps4EntryArg;

#endif