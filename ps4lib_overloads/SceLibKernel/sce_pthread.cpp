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

int pthreadErrorToSceError(int perror)
{
	int sceError = SCE_KERNEL_ERROR_UNKNOWN;

	switch (perror)
	{
	case 0:
		sceError = SCE_OK;
		break;
	case EPERM:
		sceError = SCE_KERNEL_ERROR_EPERM;
		break;
	case ENOENT:
		sceError = SCE_KERNEL_ERROR_ENOENT;
		break;
	case ESRCH:
		sceError = SCE_KERNEL_ERROR_ESRCH;
		break;
	case EINTR:
		sceError = SCE_KERNEL_ERROR_EINTR;
		break;
	case EIO:
		sceError = SCE_KERNEL_ERROR_EIO;
		break;
	case ENXIO:
		sceError = SCE_KERNEL_ERROR_ENXIO;
		break;
	case E2BIG:
		sceError = SCE_KERNEL_ERROR_E2BIG;
		break;
	case ENOEXEC:
		sceError = SCE_KERNEL_ERROR_ENOEXEC;
		break;
	case EBADF:
		sceError = SCE_KERNEL_ERROR_EBADF;
		break;
	case ECHILD:
		sceError = SCE_KERNEL_ERROR_ECHILD;
		break;
	case EAGAIN:
		sceError = SCE_KERNEL_ERROR_EAGAIN;
		break;
	case ENOMEM:
		sceError = SCE_KERNEL_ERROR_ENOMEM;
		break;
	case EACCES:
		sceError = SCE_KERNEL_ERROR_EACCES;
		break;
	case EFAULT:
		sceError = SCE_KERNEL_ERROR_EFAULT;
		break;
	case EBUSY:
		sceError = SCE_KERNEL_ERROR_EBUSY;
		break;
	case EEXIST:
		sceError = SCE_KERNEL_ERROR_EEXIST;
		break;
	case EXDEV:
		sceError = SCE_KERNEL_ERROR_EXDEV;
		break;
	case ENODEV:
		sceError = SCE_KERNEL_ERROR_ENODEV;
		break;
	case ENOTDIR:
		sceError = SCE_KERNEL_ERROR_ENOTDIR;
		break;
	case EISDIR:
		sceError = SCE_KERNEL_ERROR_EISDIR;
		break;
	case EINVAL:
		sceError = SCE_KERNEL_ERROR_EINVAL;
		break;
	case ENFILE:
		sceError = SCE_KERNEL_ERROR_ENFILE;
		break;
	case EMFILE:
		sceError = SCE_KERNEL_ERROR_EMFILE;
		break;
	case ENOTTY:
		sceError = SCE_KERNEL_ERROR_ENOTTY;
		break;
	case EFBIG:
		sceError = SCE_KERNEL_ERROR_EFBIG;
		break;
	case ENOSPC:
		sceError = SCE_KERNEL_ERROR_ENOSPC;
		break;
	case ESPIPE:
		sceError = SCE_KERNEL_ERROR_ESPIPE;
		break;
	case EROFS:
		sceError = SCE_KERNEL_ERROR_EROFS;
		break;
	case EMLINK:
		sceError = SCE_KERNEL_ERROR_EMLINK;
		break;
	case EPIPE:
		sceError = SCE_KERNEL_ERROR_EPIPE;
		break;
	case EDOM:
		sceError = SCE_KERNEL_ERROR_EDOM;
		break;
	case EDEADLK:
		sceError = SCE_KERNEL_ERROR_EDEADLK;
		break;
	case ENAMETOOLONG:
		sceError = SCE_KERNEL_ERROR_ENAMETOOLONG;
		break;
	case ENOLCK:
		sceError = SCE_KERNEL_ERROR_ENOLCK;
		break;
	case ENOSYS:
		sceError = SCE_KERNEL_ERROR_ENOSYS;
		break;
	case ENOTEMPTY:
		sceError = SCE_KERNEL_ERROR_ENOTEMPTY;
		break;
	default:
	{
    	LOG("Warning: unkown errcode %lu\n")
		sceError = perror;
	}
		break;
	}

	return sceError;
}

extern "C" {

ScePthread scePthreadSelf(void) {
	return pthread_self();
}

int scePthreadSetaffinity(ScePthread thread, const SceKernelCpumask mask) {
	// TODO
	return SCE_OK;
}

int scePthreadCreate(ScePthread *thread, const ScePthreadAttr *attr, void *(PS4API *entry) (void *), void *arg, const char *name) {
	raise(SIGTRAP);

	return 0;
}

#if 1
int scePthreadAtfork(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadBarrierattrDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadBarrierattrGetpshared(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadBarrierattrInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadBarrierattrSetpshared(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadBarrierDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadBarrierInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadBarrierWait(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCancel(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadCreate(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadDetach(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadEqual(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadExit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetaffinity(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetconcurrency(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetcpuclockid(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetname(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetprio(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetschedparam(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetspecific(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadGetthreadid(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadJoin(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadKeyCreate(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadKeyDelete(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMain(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMulti(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadOnce(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRename(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadResume(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadResumeAll(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockattrDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockattrGetpshared(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockattrGettype(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockattrInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockattrSetpshared(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockattrSettype(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockRdlock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockTimedrdlock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockTimedwrlock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockTryrdlock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockTrywrlock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockUnlock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadRwlockWrlock(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadSelf(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSemDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSemGetvalue(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSemInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSemPost(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSemTimedwait(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSemTrywait(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSemWait(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadSetaffinity(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetBesteffort(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetcancelstate(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetcanceltype(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetconcurrency(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetDefaultstacksize(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetName(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetprio(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetschedparam(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSetspecific(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSingle(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSuspend(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadSuspendAll(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadTestcancel(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadTimedjoin(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadYield(void) { raise(SIGTRAP); return SCE_OK; }
#endif

}