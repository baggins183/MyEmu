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

#endif // _PS4_REGION_H_