#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <asm/unistd_64.h>

typedef struct
{
	uint32_t        dummy[256];
	pthread_mutex_t handle;
} sce_pthread_mutex_t;

int scePthreadMutexLock(sce_pthread_mutex_t **mutex) {
    int err = pthread_mutex_lock(&(*mutex)->handle);
    fprintf(stderr, "In scePthreadMutexLock\n");
    exit(1);
    return err;
}

pid_t getpid(void) {
    return syscall(__NR_getpid);
}