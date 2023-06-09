#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

void ps4_syscall_handler(int num, siginfo_t *info, void *ucontext);

#ifdef __cplusplus
}
#endif