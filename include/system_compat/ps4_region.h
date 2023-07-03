#ifndef _PS4_REGION_H_
#define _PS4_REGION_H_

#include <string>

void enter_ps4_region();

void leave_ps4_region();

bool thread_init_syscall_user_dispatch();

// Careful way to flip the syscall intercept switch off on entry to the handler,
// and to make sure the switch goes on when exiting.
struct HostRegionScope {
    HostRegionScope()  { leave_ps4_region(); }
    ~HostRegionScope() { enter_ps4_region(); }
};

struct Ps4RegionScope {
    Ps4RegionScope()  { enter_ps4_region(); }
    ~Ps4RegionScope() { leave_ps4_region(); }
};

void set_chroot_path(const char *path);

std::string get_chroot_path();

#endif // _PS4_REGION_H_