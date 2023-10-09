#if 1

#include "ps4lib_overloads/SceLibKernel/sce_pthread_common.h"

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

int PS4FUN(scePthreadAttrGet)(ScePthread thread, ScePthreadAttr *attr) {
	int err = pthread_getattr_np(thread, &(*attr)->handle);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadAttrInit)(ScePthreadAttr *attr) {
	sce_pthread_attr_t *object = (sce_pthread_attr_t *) calloc(1, sizeof(sce_pthread_attr_t));
	int err = pthread_attr_init(&object->handle);

	*attr = object;
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadAttrDestroy)(ScePthreadAttr *attr) {
	int err = pthread_attr_destroy(&(*attr)->handle);
	free(*attr);

	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadAttrSetschedparam)(ScePthreadAttr *attr, const SceKernelSchedParam *param) {
	raise(SIGTRAP);
	sched_param native_param;
	int current_policy;
	pthread_attr_getschedpolicy(&(*attr)->handle, &current_policy);
	int min_prio = sched_get_priority_min(current_policy);
	int max_prio = sched_get_priority_max(current_policy);

	assert(max_prio > min_prio);

	// For now just map the min:max interval proportionally to SCE_KERNEL_PRIO_FIFO_HIGHEST:SCE_KERNEL_PRIO_FIFO_LOWEST
	// also flip
	// TODO figure out way to preserve native priority so _Getschedparam can be accurate
	// Going from ps4 prio <-> native prio must be lossy if one range is bigger, so we'd have to store ps4 prio somewhere
	float r = ((float) param->sched_priority - SCE_KERNEL_PRIO_FIFO_HIGHEST) / (SCE_KERNEL_PRIO_FIFO_LOWEST - SCE_KERNEL_PRIO_FIFO_HIGHEST);
	r = 1.0 - r;

	native_param.sched_priority = min_prio + r * (max_prio - min_prio);

	int err = pthread_attr_setschedparam(&(*attr)->handle, &native_param);

	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadAttrGetschedparam)(ScePthreadAttr *attr, SceKernelSchedParam *param) {
	raise(SIGTRAP);
	sched_param native_param;
	int err = pthread_attr_getschedparam(&(*attr)->handle, &native_param);
	if (err) {
		raise(SIGTRAP);
		return pthreadErrorToSceError(err);
	}

	int current_policy;
	pthread_attr_getschedpolicy(&(*attr)->handle, &current_policy);
	int min_prio = sched_get_priority_min(current_policy);
	int max_prio = sched_get_priority_max(current_policy);

	assert(max_prio > min_prio);

	float r = ((float) native_param.sched_priority - min_prio) * (max_prio - min_prio);
	r = 1.0 - r;

	param->sched_priority = SCE_KERNEL_PRIO_FIFO_HIGHEST + r * (SCE_KERNEL_PRIO_FIFO_LOWEST - SCE_KERNEL_PRIO_FIFO_HIGHEST);

	return SCE_OK;
}

int PS4FUN(scePthreadAttrSetschedpolicy)(ScePthreadAttr *attr, int policy) {
	SCHED_RR;
	int err = pthread_attr_setschedpolicy(&(*attr)->handle, policy);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadAttrGetschedpolicy)(ScePthreadAttr *attr, int *policy) {
	int err = pthread_attr_getschedpolicy(&(*attr)->handle, policy);
	if (err) {
		raise(SIGTRAP);
	}	
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadAttrSetstacksize)(ScePthreadAttr *attr, size_t stackSize) {
	int err = pthread_attr_setstacksize(&(*attr)->handle, stackSize);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);	
}

int PS4FUN(scePthreadAttrGetstacksize)(ScePthreadAttr *attr, size_t *stackSize) {
	int err = pthread_attr_getstacksize(&(*attr)->handle, stackSize);
	if (err) {
		raise(SIGTRAP);
	}	
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadAttrGetaffinity)(ScePthreadAttr *attr, SceKernelCpumask* mask) {
	cpu_set_t native_mask;
	SceKernelCpumask rv = 0;
	pthread_attr_getaffinity_np(&(*attr)->handle, sizeof(rv), &native_mask);
	for (uint i = 0; i < sizeof(rv) * CHAR_BIT; i++) {
		if (CPU_ISSET(i, &native_mask)) {
			rv |= 1 << i;
		}
	}
	*mask = rv;
	return SCE_OK;
}

int PS4FUN(scePthreadAttrSetaffinity)(ScePthreadAttr *attr, SceKernelCpumask* mask) {
	raise(SIGTRAP);
    // TODO possibly limit to 8 bits
	cpu_set_t native_mask;
	CPU_ZERO(&native_mask);
	for (uint i = 0; i < sizeof(*mask) * CHAR_BIT; i++) {
		if (*mask & (1 << i)) {
			CPU_SET(i, &native_mask);
		}
	}
	int err = pthread_attr_setaffinity_np(&(*attr)->handle, sizeof(mask), &native_mask);
	return pthreadErrorToSceError(err);
}

#if 1
//int PS4FUN(scePthreadAttrDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrGet)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrGetaffinity)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrGetdetachstate)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrGetguardsize)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrGetinheritsched)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrGetschedparam)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrGetschedpolicy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrGetscope)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrGetstack)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrGetstackaddr)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrGetstacksize)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrInit)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrSetaffinity)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrSetcreatesuspend)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrSetdetachstate)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrSetguardsize)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrSetinheritsched)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrSetschedparam)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrSetschedpolicy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrSetscope)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrSetstack)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadAttrSetstackaddr)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadAttrSetstacksize)(void) { raise(SIGTRAP); return SCE_OK; }
#endif

} // extern "C"

#endif // #if 0