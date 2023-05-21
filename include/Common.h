#ifndef _COMMON_H_
#define _COMMON_H_

#include <elf.h>
#include "Elf/elf_amd64.h"

// make sure this gets included after elf.h to undef some stuff (mostly for debugging)
#include "Elf/elf-sce.h"

#include <stdint.h>
#include <cassert>
#include <unistd.h>

#define ARRLEN(arr) (sizeof(arr) / sizeof(arr[0]))

typedef void (*PFUNC_ExitFunction)();
typedef void* (*PFUNC_GameTheadEntry)(void* pArg, PFUNC_ExitFunction pExit);

typedef struct {
    uint64_t argc;
    const char *argv[1];
} Ps4EntryArg;

// functions in DT_PREINIT, DT_INIT, DT_INIT_ARRAY
typedef int (*PFN_PS4_INIT_FUNC) (int, const char **, char **);

struct Ps4InitFuncArgs {
    int argc;
    char **argv;
    char **environ;
};

static const long PGSZ = sysconf(_SC_PAGE_SIZE);

#define ROUND_DOWN(x, SZ) ((x) - (x) % (SZ))
#define ROUND_UP(x, SZ) ( (x) % (SZ) ? (x) - ((x) % (SZ)) + (SZ) : (x))

#define PS4API __attribute__((sysv_abi))
#define PS4FUN(fn) PS4API _ps4__##fn

#endif