#if 0

#include "sce_pthread_common

extern "C" {

#if 1
int PS4FUN(scePthreadCondattrDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondattrGetclock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondattrGetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondattrInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondattrSetclock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondattrSetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondBroadcast)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondSignal)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondSignalto)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondTimedwait)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadCondWait)(void) { raise(SIGTRAP); return SCE_OK; }
#endif

}

#endif // #if 0