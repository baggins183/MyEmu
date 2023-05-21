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
int PS4FUN(scePthreadAtfork)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadBarrierattrDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadBarrierattrGetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadBarrierattrInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadBarrierattrSetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadBarrierDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadBarrierInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadBarrierWait)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCancel)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadCreate)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadDetach)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadEqual)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadExit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetaffinity)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetconcurrency)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetcpuclockid)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetname)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetprio)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetschedparam)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetspecific)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadGetthreadid)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadJoin)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadKeyCreate)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadKeyDelete)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMain)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMulti)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadOnce)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRename)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadResume)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadResumeAll)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockattrDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockattrGetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockattrGettype)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockattrInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockattrSetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockattrSettype)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockRdlock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockTimedrdlock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockTimedwrlock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockTryrdlock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockTrywrlock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockUnlock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadRwlockWrlock)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadSelf)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSemDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSemGetvalue)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSemInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSemPost)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSemTimedwait)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSemTrywait)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSemWait)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadSetaffinity)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetBesteffort)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetcancelstate)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetcanceltype)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetconcurrency)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetDefaultstacksize)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetName)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetprio)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetschedparam)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSetspecific)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSingle)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSuspend)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadSuspendAll)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadTestcancel)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadTimedjoin)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadYield)(void) { raise(SIGTRAP); return SCE_OK; }
#endif

}