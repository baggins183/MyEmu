#include "sce_pthread_common.h"

extern "C" {

#if 1
int scePthreadCondattrDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondattrGetclock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondattrGetpshared(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondattrInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondattrSetclock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondattrSetpshared(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondBroadcast(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondSignal(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondSignalto(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondTimedwait(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadCondWait(void) { raise(SIGTRAP); return SCE_OK; }
#endif

}