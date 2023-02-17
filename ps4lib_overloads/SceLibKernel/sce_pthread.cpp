#include "sce_pthread_common.h"

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

ScePthread scePthreadSelf(void) {
	return pthread_self();
}

int scePthreadSetaffinity(ScePthread thread, const SceKernelCpumask mask) {
	// TODO
	return SCE_OK;
}

int scePthreadCondInit(ScePthreadCond *cond, const ScePthreadCondattr *attr, const char *name) {
	// TODO name
	int err = pthread_cond_init(cond, attr);
	return pthreadErrorToSceError(err);
}

int scePthreadAttrGet(ScePthread thread, ScePthreadAttr *attr) {
	int err = pthread_getattr_np(thread, &(*attr)->handle);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}

int scePthreadCreate(ScePthread *thread, const ScePthreadAttr *attr, void *(PS4API *entry) (void *), void *arg, const char *name) {
	raise(SIGTRAP);

	return 0;
}

}