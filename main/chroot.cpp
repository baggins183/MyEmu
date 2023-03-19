#include <string>

bool gChrootBit = false;
std::string gChrootPath;

void enable_chroot() {
    gChrootBit = true;
}

void disable_chroot() {
    gChrootBit = false;
}

bool is_chrooted() {
    return gChrootBit;
}

void set_chroot_path(const char *path) {
    gChrootPath = path;
}

std::string get_chroot_path() {
    return gChrootPath;
}