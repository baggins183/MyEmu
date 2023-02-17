#include "sce_errors/sce_errors.h"
#include "ps4lib_overrides/sce_kernel_scepthread.h"
#include "Logger.h"
#include "Common.h"

#include <assert.h>
#include <bits/types/struct_sched_param.h>
#include <dlfcn.h>
#include <sched.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <asm/unistd_64.h>
#include <map>
#include <string>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <limits.h>

extern "C" {

pid_t getpid(void) {
    return syscall(__NR_getpid);
}

struct tls_index
{
	unsigned long ti_module;
	unsigned long ti_offset;
};

void*  __tls_get_addr(tls_index *ti) {
	LOG("IN __tls_get_addr!!!!!!!!!!!!!!!!!!!!!!!!!!");
	raise(SIGTRAP);

	return nullptr;
}

void _Times(void) {
	LOG("IN _Times!!!!!!!!!!!!!!!!!!!!!!!!!!");
	raise(SIGTRAP);	
}

} // extern "C"