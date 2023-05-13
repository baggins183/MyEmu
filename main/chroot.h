#include <string>

void enter_chroot();

void leave_chroot();

bool is_chrooted();

void set_chroot_path(const char *path);

std::string get_chroot_path();