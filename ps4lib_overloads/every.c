#include <stdint.h>
#include <pthread.h>

typedef struct
{
	uint32_t        dummy[256];
	pthread_mutex_t handle;
} sce_pthread_mutex_t;

int scePthreadMutexLock(sce_pthread_mutex_t **mutex) {
    int err = pthread_mutex_lock(&(*mutex)->handle);
    return err;
}

