#include "system_compat/ps4_region.h"
#include <cassert>

// If the caller is native (non-ps4 code), then call fn, resolved from
// the host libraries, on the args.
// If the caller is ps4 code, move on to the handler that
// should follow this macro.
// Construct a scope that enters host code on construction and
// returns to the previous state (ps4 or host) on destruction.
//
// TODO maybe do call_once for getting function address to be safe
#define SYSTEM_LIB_WRAPPER(fn, ...) \
    CodeRegionScope __region_scope(CodeRegionType::HOST_REGION); \
    typedef decltype(fn)* PFN_##fn; \
    static PFN_##fn fn##__impl = nullptr; \
    if (!fn##__impl) { \
        fn##__impl = (PFN_##fn) dlsym(RTLD_NEXT, #fn); \
        assert(fn##__impl); \
    } \
    if (__region_scope.lastScopeWasHost()) { \
        return fn##__impl(__VA_ARGS__); \
    }

