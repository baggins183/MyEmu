#ifndef _SCE_PTHREAD_COMMON_H_
#define _SCE_PTHREAD_COMMON_H_

#include <orbis/sce_errors/sce_errors.h>
#include "ps4lib_overrides/sce_kernel_scepthread.h"
#include "Logger.h"
#include "Common.h"

#include <signal.h>

int pthreadErrorToSceError(int perror);

#endif // _SCE_PTHREAD_COMMON_H_