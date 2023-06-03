#include "sce_errors/sce_errors.h"
#include "Common.h"

#include <dlfcn.h>
#include <signal.h>

struct  sce_module_handle_t {
    // Fairly confident
    // code uses 4B regs to pass around
    uint32_t handle;
    // probably analagous to module id from TLS
};

// I'm not sure what bits are part of a request if any, and what parts of this struct are for filling in
struct sce_module_info_request_t {
    // Guessing
    // movq used to setup contents, so assume 8B
    uint64_t bits;

    // some bits are probably for the compiled SDK version
};

extern "C" {

// Note: I saw this crash on a code path through libSceSysModule's DT_INIT function
// Implementing this leaves a crash in sceKernelGetModuleInfo
sce_module_handle_t PS4FUN(sceKernelGetExecutableModuleHandle)(void) {
    return { 0 };
}

// Defintely takes 2 args, not more.
// Not sure what $2 is exactly.
// Returns error code
// TODO check where errors are coming from: break on printf, go down callstack to see what condition lead to error
int PS4FUN(sceKernelGetModuleInfo)(sce_module_handle_t mod, sce_module_info_request_t *request) {
    raise(SIGTRAP);
    switch(request->bits) {
        case 0x160:
            // something to do with the SDK version
            // TODO figure out where to write SDK version
            
            // Trace when returning non-zero:
            //[Sysmodule(177726) 3607] sceKernelGetModuleInfo() fails (ffffffff)
            //[Sysmodule(177726) 3122] get_process_info() fails (ffffffff)            

            // Trace when returning 0:
            // [Sysmodule(178214) 3133] Cannot get SDK version (0x80020016)
            // dipsw [0x06]: open failed, err = 0x8001ffea            

            return 0;
        default:
            return 0;
    }
}

int PS4FUN(sceKernelGetModuleInfo2)(void) {
    raise(SIGTRAP);
    return 0;
}

int PS4FUN(sceKernelGetCompiledSdkVersion)(void) {
    return 0;
}

}