#include <cassert>

// If the caller is native (non-ps4 code), then call fn, resolved from
// the host libraries, on the args.
// If the caller is ps4 code, move on to the handler that
// should follow.
// Call leave_ps4_region(), since the handler will invoke
// native code, and if that code includes another wrapper, we
// don't want that wrapper to think the caller is ps4 code.
#define SYSTEM_LIB_WRAPPER(fn, ...) \
    typedef decltype(fn)* PFN_##fn; \
    static PFN_##fn fn##__impl = nullptr; \
    if (!fn##__impl) { \
        fn##__impl = (PFN_##fn) dlsym(RTLD_NEXT, #fn); \
        assert(fn##__impl); \
    } \
    if (!in_ps4_region()) { \
        return fn##__impl(__VA_ARGS__); \
    } \
    \
    leave_ps4_region(); \

// We are returning to ps4 code
#define END_SYSTEM_LIB_WRAPPER \
    enter_ps4_region();

// TODO cause a compile time error if END_SYSTEM_LIB_WRAPPER doesn't follow SYSTEM_LIB_WRAPPER
