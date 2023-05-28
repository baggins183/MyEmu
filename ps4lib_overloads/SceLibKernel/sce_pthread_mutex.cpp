#include "sce_pthread_common.h"

static std::map<pthread_mutex_t *, std::string> MutexNames;

static bool mutexHasName(ScePthreadMutex *mutex) {
	return MutexNames.find(&(*mutex)->handle) != MutexNames.end();
}
static inline const char *getMutexName(ScePthreadMutex *mutex) {
	if (MutexNames.find(&(*mutex)->handle) == MutexNames.end()) {
		return nullptr;
	}
	std::string &name = MutexNames[&(*mutex)->handle];
	return name.c_str();
}
static void addMutexName(ScePthreadMutex *mutex, const char *name) {
	MutexNames[&(*mutex)->handle] = name;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" {

int PS4FUN(scePthreadMutexattrInit)(ScePthreadMutexattr *attr)
{
	int err = pthread_mutexattr_init(attr);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}


int PS4FUN(scePthreadMutexattrDestroy)(ScePthreadMutexattr *attr)
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

int PS4FUN(scePthreadMutexattrSetprotocol)(ScePthreadMutexattr *attr, int protocol)
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

int PS4FUN(scePthreadMutexattrSettype)(ScePthreadMutexattr *attr, int type)
{
	int ptype = sceMutexAttrTypeToPthreadType(type);
	int err   = pthread_mutexattr_settype(attr, ptype);
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadMutexLock)(ScePthreadMutex *mutex) {
    LOG("scePthreadMutexLock\n")
    LOG("    sceMut: %llx\n", mutex)
    LOG("    *sceMut: %llx\n", *mutex)
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex))
	}
    int err = pthread_mutex_lock(&((*mutex)->handle));
	if (err) {
		raise(SIGTRAP);
	}	
    //fprintf(stderr, "In scePthreadMutexLock\n");
    //fprintf(stderr, "In scePthreadMutexLock\n");
    return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadMutexUnlock)(ScePthreadMutex *mutex) {
    LOG("scePthreadMutexUnlock\n")
    LOG("    sceMut: %llx\n", mutex)
    LOG("    *sceMut: %llx\n", *mutex)
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex))
	}

    int err = pthread_mutex_unlock(&((*mutex)->handle));
	if (err) {
		raise(SIGTRAP);
	}		
    //fprintf(stderr, "In scePthreadMutexLock\n");
    //fprintf(stderr, "In scePthreadMutexLock\n");
    return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadMutexInit)(ScePthreadMutex *mutex, const ScePthreadMutexattr *attr, const char *name) {
    //printf("Warning: passthrough scePthreadMutexInit\n");
    LOG("scePthreadMutexInit\n")
    LOG("    sceMut: %llx\n", mutex)
	if (name) {
    	LOG("    name: %s\n", name)
	}

    sce_pthread_mutex_t *object = (sce_pthread_mutex_t *) calloc(1, sizeof(sce_pthread_mutex_t));
    assert(object);

	int err;
	if ( !attr) {
		pthread_mutexattr_t defaultAttr;
		pthread_mutexattr_init(&defaultAttr);
		pthread_mutexattr_settype(&defaultAttr, PTHREAD_MUTEX_ERRORCHECK);
		err = pthread_mutex_init(&object->handle, &defaultAttr);
		pthread_mutexattr_destroy(&defaultAttr);
	} else {
		err = pthread_mutex_init(&object->handle, attr);
	}

	if (err) {
		raise(SIGTRAP);
	}	

	*mutex = object;
	if (name) {
		addMutexName(mutex, name);
	}
    LOG("    *sceMut (result): %llx\n", *mutex)

	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadMutexDestroy)(ScePthreadMutex *mutex) {
    LOG("scePthreadMutexDestroy\n")
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex))
	}

	int err = pthread_mutex_destroy(&(*mutex)->handle);
	free(*mutex);
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadMutexGetprioceiling)(ScePthreadMutex *mutex, int prioceiling, int *old_ceiling)
{
	raise(SIGTRAP);
	LOG("scePthreadMutexGetprioceiling\n")
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex));
	}
	int err = pthread_mutex_setprioceiling(&(*mutex)->handle, prioceiling, old_ceiling);
	raise(SIGTRAP);
	return pthreadErrorToSceError(err);
}


int PS4FUN(scePthreadMutexTimedlock)(ScePthreadMutex *mutex, const struct timespec * abs_timeout)
{
	raise(SIGTRAP);
	LOG("scePthreadMutexTimedlock\n")
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex));
	}	
	int err = pthread_mutex_timedlock(&(*mutex)->handle, abs_timeout);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}

int PS4FUN(scePthreadMutexTrylock)(ScePthreadMutex *mutex)
{
	int err = pthread_mutex_trylock(&((*mutex)->handle));
	return pthreadErrorToSceError(err);
}

#if 1
//int PS4FUN(scePthreadMutexattrDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrGetkind)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrGetprioceiling)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrGetprotocol)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrGetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrGettype)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexattrInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrInitForInternalLibc)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrSetkind)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrSetprioceiling)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexattrSetprotocol)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexattrSetpshared)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexattrSettype)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexDestroy)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexGetprioceiling)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexGetspinloops)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexGetyieldloops)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexInit)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexInitForInternalLibc)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexIsowned)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexLock)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexSetprioceiling)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexSetspinloops)(void) { raise(SIGTRAP); return SCE_OK; }
int PS4FUN(scePthreadMutexSetyieldloops)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexTimedlock)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexTrylock)(void) { raise(SIGTRAP); return SCE_OK; }
//int PS4FUN(scePthreadMutexUnlock)(void) { raise(SIGTRAP); return SCE_OK; }
#endif

}