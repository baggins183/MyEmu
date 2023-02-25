#include <link.h>
#include <sys/prctl.h>
#include <linux/prctl.h>
#include <errno.h>
#include <stdio.h>
#include <dlfcn.h>
//#include <asm/unistd_32.h>
#include <asm/unistd_64.h>
#include <unistd.h>
#include <signal.h>

#include "../external/proc_maps_parser/proc_maps_parser/include/pmparser.h"

extern void (*_syscall_handler) (int);

char syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_ALLOW;

int N = 5;

void c_handler(int num,
	siginfo_t *info,
	void *ucontext)
{
    N = 77;
}

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
    procmaps_struct *pm;
    procmaps_iterator *it = pmparser_parse(-1);
    uint64_t addr_start = 0;
    uint64_t addr_end = 0;

    int found = 0;
    while ((pm = pmparser_next(it))) {
        uint len = strlen(pm->pathname);
        const char *filename = NULL;
        int i = len - 1;
        for (; i > 0; i--) {
            if (pm->pathname[i - 1] == '/') {
                break;
            }
        }
        if (i <= 0) {
            continue;
        }
        filename = &pm->pathname[i];

        if (strcmp(filename, "test64")) {
            continue;
        }

        if ( !found) {
            addr_start = (uint64_t) pm->addr_start;
            addr_end = (uint64_t) pm->addr_end;
        } else {
            addr_start = (uint64_t) pm->addr_start < addr_start ? (uint64_t) pm->addr_start : addr_start;
            addr_end = (uint64_t) pm->addr_end > addr_end ? (uint64_t) pm->addr_end : addr_end;
        }
        found = 1;
    }

    if ( !found) {
        return -1;
    }    

	int ret, rc;
	struct sigaction action;

	(void)memset(&action, 0, sizeof action);
	action.sa_sigaction = c_handler;
	(void)sigemptyset(&action.sa_mask);
	/*
	 *  We need to block a range of signals from occurring
	 *  while we are handling the SIGSYS to avoid any
	 *  system calls that would cause a nested SIGSYS.
	 */
	(void)sigfillset(&action.sa_mask);
	(void)sigdelset(&action.sa_mask, SIGSYS);
	action.sa_flags = SA_SIGINFO;

	ret = sigaction(SIGSYS, &action, NULL);

    int err = prctl(PR_SET_SYSCALL_USER_DISPATCH, PR_SYS_DISPATCH_ON, addr_start, (addr_end - addr_start + 1), &syscall_dispatch_switch);

    //signal(SIGSYS, _syscall_handler);
    syscall_dispatch_switch = SYSCALL_DISPATCH_FILTER_BLOCK;

    int64_t pid = -1;
    asm volatile ("movq $39, %%rax; syscall; movq %%rax, %0"
        : "=r" (pid) 
        :: "rax");
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