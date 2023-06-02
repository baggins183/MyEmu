#ifndef _PS4_REGION_H_
#define _PS4_REGION_H_

#include <string>

void enter_ps4_region();

void leave_ps4_region();

bool thread_init_syscall_user_dispatch();

void set_chroot_path(const char *path);

std::string get_chroot_path();

#endif // _PS4_REGION_H_