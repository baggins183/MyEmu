#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

void freebsd_syscall_handler(int num, siginfo_t *info, void *ucontext);

#ifdef __cplusplus
}
#endif