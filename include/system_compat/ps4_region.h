#ifndef _PS4_REGION_H_
#define _PS4_REGION_H_

#include <string>

void enter_ps4_region();

void leave_ps4_region();

bool in_ps4_region();

bool thread_init_syscall_user_dispatch();

void set_chroot_path(const char *path);

std::string get_chroot_path();

enum CodeRegionType {
    PS4_REGION,
    HOST_REGION
};

// Convenience for capturing scopes for host code calling ps4 code, or vice versa.
// The constructor sets the thread local variable for the saying whether the
// thread is currently in ps4 code or host code.
// The destructor goes back to the previous state - ps4 or host code
//
// Basically works like a push and pop.
class CodeRegionScope {
public:
    CodeRegionScope(CodeRegionType new_region_type) {
        m_lastRegionType = in_ps4_region() ? PS4_REGION : HOST_REGION;
        if (new_region_type != m_lastRegionType) {
            switch (new_region_type) {
                case PS4_REGION:
                    enter_ps4_region();
                    break;
                case HOST_REGION:
                    leave_ps4_region();
                    break;
            }
        }
    }

    CodeRegionScope(): CodeRegionScope(HOST_REGION) {}

    ~CodeRegionScope() {
        CodeRegionType current_region_type = in_ps4_region() ? PS4_REGION : HOST_REGION;
        if (current_region_type != m_lastRegionType) {
            switch (m_lastRegionType) {
                case PS4_REGION:
                    enter_ps4_region();
                    break;
                case HOST_REGION:
                    leave_ps4_region();
                    break;
            }
        }
    }

    bool lastScopeWasPs4() {
        return m_lastRegionType == PS4_REGION;
    }
    bool lastScopeWasHost() {
        return m_lastRegionType == HOST_REGION;
    }
private:
    CodeRegionType m_lastRegionType;
};

// If the caller is native (non-ps4 code), then call fn, resolved from
// the host libraries, on the args.
// If the caller is ps4 code, move on to the handler that
// should follow this macro.
// Construct a scope that enters host code on construction and
// returns to the previous state (ps4 or host) on destruction.
//
// TODO maybe do call_once for getting function address to be safe
/*
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
*/

#endif // _PS4_REGION_H_