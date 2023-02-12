#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <asm/unistd_64.h>
#include "Logger.h"

typedef struct
{
	uint32_t        dummy[256];
	pthread_mutex_t handle;
} sce_pthread_mutex_t;

typedef pthread_mutexattr_t   ScePthreadMutexattr;

int scePthreadMutexLock(sce_pthread_mutex_t **mutex) {
    LOG("In scePthreadMutexLock\n")
    LOG("    sceMut: %llx\n", mutex)
    LOG("    *sceMut: %llx\n", *mutex)
    exit(1);
    int err = pthread_mutex_lock(&(*mutex)->handle);
    //fprintf(stderr, "In scePthreadMutexLock\n");
    //fprintf(stderr, "In scePthreadMutexLock\n");
    return err;
}

typedef int (*PFN_SCEPTHREADMUTEXINIT)(sce_pthread_mutex_t **mutex, const ScePthreadMutexattr *attr, const char *name);
//static PFN_SCEPTHREADMUTEXINIT scePthreadMutexInit_impl;

int scePthreadMutexInit(sce_pthread_mutex_t **mutex, const ScePthreadMutexattr *attr, const char *name) {
    //printf("Warning: passthrough scePthreadMutexInit\n");
    LOG("Warning: passthrough scePthreadMutexInit\n")
    LOG("    sceMut: %llx\n", mutex)
    LOG("    *sceMut: %llx\n", *mutex)    
    // TODO see if we can use RTLD_NEXT to search an entire namespace for a symbol, instead of search order being confined to dependecies of *this*
    // object (libevery.so)
    // Currently we have to add DT_NEEDED entry for libkernel.prx to libevery.so so dlsym can find the next definition
    PFN_SCEPTHREADMUTEXINIT scePthreadMutexInit_impl = (PFN_SCEPTHREADMUTEXINIT) dlsym(RTLD_NEXT, "scePthreadMutexInit");
    return scePthreadMutexInit_impl(mutex, attr, name);
}

pid_t getpid(void) {
    return syscall(__NR_getpid);
}