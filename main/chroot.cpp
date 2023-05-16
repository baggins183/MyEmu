#include <string>

bool gChrootBit = false;
std::string gChrootPath;

void enter_chroot() {
    gChrootBit = true;
}

void leave_chroot() {
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