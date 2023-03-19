#include <string>

void enable_chroot();

void disable_chroot();

bool is_chrooted();

void set_chroot_path(const char *path);

std::string get_chroot_path();