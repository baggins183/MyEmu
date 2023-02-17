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

int scePthreadMutexattrInit(ScePthreadMutexattr *attr)
{
	int err = pthread_mutexattr_init(attr);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}


int scePthreadMutexattrDestroy(ScePthreadMutexattr *attr)
{
	int err = pthread_mutexattr_destroy(attr);
	if (err) {
		raise(SIGTRAP);
	}	
	return pthreadErrorToSceError(err);
}

static int sceMutexAttrProtocolToPthreadType(int protocol)
{
	int pthreadType = 0;
	switch (protocol)
	{
		case SCE_PTHREAD_PRIO_NONE: pthreadType = PTHREAD_PRIO_NONE; break;
		case SCE_PTHREAD_PRIO_INHERIT: pthreadType = PTHREAD_PRIO_INHERIT; break;
		case SCE_PTHREAD_PRIO_PROTECT: pthreadType = PTHREAD_PRIO_PROTECT; break;
	}
	return pthreadType;
}

int scePthreadMutexattrSetprotocol(ScePthreadMutexattr *attr, int protocol)
{
	int type = sceMutexAttrProtocolToPthreadType(protocol);
	int err  = pthread_mutexattr_setprotocol(attr, type);
	if (err) {
		raise(SIGTRAP);
	}	

	return pthreadErrorToSceError(err);
}

static int sceMutexAttrTypeToPthreadType(int sceType)
{
	int pthreadType = -1;
	switch (sceType)
	{
	case SCE_PTHREAD_MUTEX_ERRORCHECK:
		pthreadType = PTHREAD_MUTEX_ERRORCHECK;
		break;
	case SCE_PTHREAD_MUTEX_RECURSIVE:
		pthreadType = PTHREAD_MUTEX_RECURSIVE;
		break;
	case SCE_PTHREAD_MUTEX_NORMAL:
		pthreadType = PTHREAD_MUTEX_NORMAL;
		break;
	case SCE_PTHREAD_MUTEX_ADAPTIVE_NP:
		pthreadType = PTHREAD_MUTEX_ADAPTIVE_NP;
		break;
	default:
		break;
	}
	return pthreadType;
}

int scePthreadMutexattrSettype(ScePthreadMutexattr *attr, int type)
{
	int ptype = sceMutexAttrTypeToPthreadType(type);
	int err   = pthread_mutexattr_settype(attr, ptype);
	return pthreadErrorToSceError(err);
}


/**********************************************************/
// PTHREAD Attr

int scePthreadAttrInit(ScePthreadAttr *attr) {
	sce_pthread_attr_t *object = (sce_pthread_attr_t *) calloc(1, sizeof(sce_pthread_attr_t));
	int err = pthread_attr_init(&object->handle);

	*attr = object;
	return pthreadErrorToSceError(err);
}

int scePthreadAttrDestroy(ScePthreadAttr *attr) {
	int err = pthread_attr_destroy(&(*attr)->handle);
	free(*attr);

	return pthreadErrorToSceError(err);
}

int scePthreadAttrSetschedparam(ScePthreadAttr *attr, const SceKernelSchedParam *param) {
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

int scePthreadAttrGetschedparam(ScePthreadAttr *attr, SceKernelSchedParam *param) {
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

int scePthreadAttrSetschedpolicy(ScePthreadAttr *attr, int policy) {
	SCHED_RR;
	int err = pthread_attr_setschedpolicy(&(*attr)->handle, policy);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}

int scePthreadAttrGetschedpolicy(ScePthreadAttr *attr, int *policy) {
	int err = pthread_attr_getschedpolicy(&(*attr)->handle, policy);
	if (err) {
		raise(SIGTRAP);
	}	
	return pthreadErrorToSceError(err);
}

int scePthreadAttrSetstacksize(ScePthreadAttr *attr, size_t stackSize) {
	int err = pthread_attr_setstacksize(&(*attr)->handle, stackSize);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);	
}

int scePthreadAttrGetstacksize(ScePthreadAttr *attr, size_t *stackSize) {
	int err = pthread_attr_getstacksize(&(*attr)->handle, stackSize);
	if (err) {
		raise(SIGTRAP);
	}	
	return pthreadErrorToSceError(err);
}

int scePthreadAttrGetaffinity(ScePthreadAttr *attr, SceKernelCpumask* mask) {
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

int scePthreadAttrSetaffinity(ScePthreadAttr *attr, SceKernelCpumask* mask) {
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

//int scePthreadAttrDestroy() {}
//int scePthreadAttrGet() {}
//int scePthreadAttrGetaffinity() {}
//int scePthreadAttrGetdetachstate() {}
//int scePthreadAttrGetguardsize() {}
//int scePthreadAttrGetinheritsched() {}
//int scePthreadAttrGetschedparam() {}
//int scePthreadAttrGetschedpolicy() {}
//int scePthreadAttrGetscope() {}
//int scePthreadAttrGetstack() {}
//int scePthreadAttrGetstackaddr() {}
//int scePthreadAttrGetstacksize() {}
//int scePthreadAttrInit() {}
//int scePthreadAttrSetaffinity() {}
//int scePthreadAttrSetcreatesuspend() {}
//int scePthreadAttrSetdetachstate() {}
//int scePthreadAttrSetguardsize() {}
//int scePthreadAttrSetinheritsched() {}
//int scePthreadAttrSetschedparam() {}
//int scePthreadAttrSetschedpolicy() {}
//int scePthreadAttrSetscope() {}
//int scePthreadAttrSetstack() {}
//int scePthreadAttrSetstackaddr() {}
//int scePthreadAttrSetstacksize() {}

} // extern "C"