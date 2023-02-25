#include <stdio.h>
#include "freebsd_compat.h"

extern "C" {

void freebsd_syscall_handler(int num, siginfo_t *info, void *ucontext) {
    printf("In Syscall Handler !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
}

}