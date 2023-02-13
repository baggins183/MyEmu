#include "sce_errors/sce_errors.h"
#include "ps4lib_overrides/sce_kernel_scepthread.h"

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <asm/unistd_64.h>
#include <map>
#include <string>
#include "errno.h"

#include "Logger.h"

#define PS4API __attribute__((sysv_abi))

static std::map<pthread_mutex_t *, std::string> *MutexNames = nullptr;
static std::map<pthread_mutex_t *, std::string>& initMutexNames() {
	if (!MutexNames) 
		MutexNames = new std::map<pthread_mutex_t *, std::string>;

	return *MutexNames;
}
static bool mutexHasName(ScePthreadMutex *mutex) {
	auto &map = initMutexNames();
	return map.find(&(*mutex)->handle) != map.end();
}
static const char *getMutexName(ScePthreadMutex *mutex) {
	auto &map = initMutexNames();

	if (map.find(&(*mutex)->handle) == map.end()) {
		return nullptr;
	}
	std::string &name = map[&(*mutex)->handle];
	return name.c_str();
}
static void addMutexName(ScePthreadMutex *mutex, const char *name) {
	auto &map = initMutexNames();

	map[&(*mutex)->handle] = name;
}

static int pthreadErrorToSceError(int perror)
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

int scePthreadMutexLock(ScePthreadMutex *mutex) {
    LOG("scePthreadMutexLock\n")
    LOG("    sceMut: %llx\n", mutex)
    LOG("    *sceMut: %llx\n", *mutex)
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex))
	}
    int err = pthread_mutex_lock(&((*mutex)->handle));
    //fprintf(stderr, "In scePthreadMutexLock\n");
    //fprintf(stderr, "In scePthreadMutexLock\n");
    return pthreadErrorToSceError(err);
}

int scePthreadMutexUnlock(ScePthreadMutex *mutex) {
    LOG("scePthreadMutexUnlock\n")
    LOG("    sceMut: %llx\n", mutex)
    LOG("    *sceMut: %llx\n", *mutex)
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex))
	}

    int err = pthread_mutex_unlock(&((*mutex)->handle));
    //fprintf(stderr, "In scePthreadMutexLock\n");
    //fprintf(stderr, "In scePthreadMutexLock\n");
    return pthreadErrorToSceError(err);
}

//typedef int (*PFN_SCEPTHREADMUTEXINIT)(ScePthreadMutex *mutex, const ScePthreadMutexattr *attr, const char *name);

int scePthreadMutexInit(ScePthreadMutex *mutex, const ScePthreadMutexattr *attr, const char *name) {
    //printf("Warning: passthrough scePthreadMutexInit\n");
    LOG("scePthreadMutexInit\n")
    LOG("    sceMut: %llx\n", mutex)
	if (name) {
    	LOG("    name: %s\n", name)
	}

    //PFN_SCEPTHREADMUTEXINIT scePthreadMutexInit_impl = (PFN_SCEPTHREADMUTEXINIT) dlsym(RTLD_NEXT, "scePthreadMutexInit");
    //return scePthreadMutexInit_impl(mutex, attr, name);

    sce_pthread_mutex_t *object = (sce_pthread_mutex_t *) calloc(1, sizeof(sce_pthread_mutex_t));
    assert(object);

	int err = SCE_OK;
	if ( !attr) {
		pthread_mutexattr_t defaultAttr;
		pthread_mutexattr_init(&defaultAttr);
		pthread_mutexattr_settype(&defaultAttr, PTHREAD_MUTEX_ERRORCHECK);
		err = pthread_mutex_init(&object->handle, &defaultAttr);
		pthread_mutexattr_destroy(&defaultAttr);
	} else {
		pthread_mutex_init(&object->handle, attr);
	}

	*mutex = object;
	if (name) {
		addMutexName(mutex, name);
	}
    LOG("    *sceMut (result): %llx\n", *mutex)

	return pthreadErrorToSceError(err);
}

int scePthreadMutexDestroy(ScePthreadMutex *mutex) {
    LOG("scePthreadMutexDestroy\n")
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex))
	}

	int err = pthread_mutex_destroy(&(*mutex)->handle);
	free(*mutex);
	return pthreadErrorToSceError(err);
}

int PS4API scePthreadMutexattrInit(ScePthreadMutexattr *attr)
{
	int err = pthread_mutexattr_init(attr);
	return pthreadErrorToSceError(err);
}


int PS4API scePthreadMutexattrDestroy(ScePthreadMutexattr *attr)
{
	int err = pthread_mutexattr_destroy(attr);
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

int PS4API scePthreadMutexattrSetprotocol(ScePthreadMutexattr *attr, int protocol)
{
	// TODO:
	// winpthreads' implementation has limit,
	// it only support INHERIT & PROTECT protocol, others will fail.
	// 
	int type = sceMutexAttrProtocolToPthreadType(protocol);
	int err  = pthread_mutexattr_setprotocol(attr, type);

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

int PS4API scePthreadMutexattrSettype(ScePthreadMutexattr *attr, int type)
{
	int ptype = sceMutexAttrTypeToPthreadType(type);
	int err   = pthread_mutexattr_settype(attr, ptype);
	return pthreadErrorToSceError(err);
}








/***********************************************************/

pid_t getpid(void) {
    return syscall(__NR_getpid);
}

} // extern "C"