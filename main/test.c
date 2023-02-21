#include <link.h>
#include <sys/prctl.h>
#include <linux/prctl.h>
#include <errno.h>
#include <stdio.h>
#include <dlfcn.h>
//#include <asm/unistd_32.h>
#include <asm/unistd_64.h>
#include <unistd.h>

int main() {
    //void *libc = dlopen("libc.so", RTLD_LAZY);
    //struct link_map l;
    //dlinfo(libc, int request, void *__restrict arg)

    printf("start\n");

    // 32 bit
    // libc
    //int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, 0xf7c1e000, (0xf8000000 - 0xf7c1e000 + 1));
    // exe
    //int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, 0x56000000, (0x58000000 - 0x56000000 + 1));

    // 64 bit (during gdb)
    // libc
    //int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, 0x00007ffff7dd8440, (0x00007ffff7f3125d - 0x00007ffff7dd8440 + 1));
    // ld-linux
    //int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, 0x00007ffff7fca000, (0x00007ffff7ff00a5 - 0x00007ffff7fca000 + 1));
    // exe
    int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, 0x0000555550000000, (0x0000555560000000 - 0x0000555550000000 + 1));

    asm volatile ("movq $39, %%rax; syscall" ::: "rax");
    //syscall(39);
    //getpid();

    // 64

    //asm volatile ("movl $20, %%eax;\nint $0x80;\n" ::: "eax");
    //if (err) {
      //  printf("error: %i\n", errno);
    //}
    int a = 2 + 2;
    printf("end\n");
    int b = 4 + a * 98;
}