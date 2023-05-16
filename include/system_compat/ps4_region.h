#include <string>

void enter_ps4_region();

void leave_ps4_region();

bool in_ps4_region();

bool thread_init_syscall_user_dispatch();

void set_chroot_path(const char *path);

std::string get_chroot_path();