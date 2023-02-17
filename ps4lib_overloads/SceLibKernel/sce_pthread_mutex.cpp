#include "sce_pthread_common.h"

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" {

int scePthreadMutexattrInit(ScePthreadMutexattr *attr)
{
	int err = pthread_mutexattr_init(attr);
	if (err) {
		raise(SIGTRAP);
	}
	return pthreadErrorToSceError(err);
}


int scePthreadMutexattrDestroy(ScePthreadMutexattr *attr)
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

int scePthreadMutexattrSetprotocol(ScePthreadMutexattr *attr, int protocol)
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

int scePthreadMutexattrSettype(ScePthreadMutexattr *attr, int type)
{
	int ptype = sceMutexAttrTypeToPthreadType(type);
	int err   = pthread_mutexattr_settype(attr, ptype);
	return pthreadErrorToSceError(err);
}

int scePthreadMutexLock(ScePthreadMutex *mutex) {
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

int scePthreadMutexUnlock(ScePthreadMutex *mutex) {
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

int scePthreadMutexInit(ScePthreadMutex *mutex, const ScePthreadMutexattr *attr, const char *name) {
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

int scePthreadMutexDestroy(ScePthreadMutex *mutex) {
    LOG("scePthreadMutexDestroy\n")
	if (mutexHasName(mutex)) {
		LOG("    name: %s\n", getMutexName(mutex))
	}

	int err = pthread_mutex_destroy(&(*mutex)->handle);
	free(*mutex);
	return pthreadErrorToSceError(err);
}

int scePthreadMutexGetprioceiling(ScePthreadMutex *mutex, int prioceiling, int *old_ceiling)
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


int scePthreadMutexTimedlock(ScePthreadMutex *mutex, const struct timespec * abs_timeout)
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

int scePthreadMutexTrylock(ScePthreadMutex *mutex)
{
	int err = pthread_mutex_trylock(&((*mutex)->handle));
	return pthreadErrorToSceError(err);
}

#if 1
//int scePthreadMutexattrDestroy(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrGetkind(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrGetprioceiling(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrGetprotocol(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrGetpshared(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrGettype(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexattrInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrInitForInternalLibc(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrSetkind(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrSetprioceiling(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexattrSetprotocol(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexattrSetpshared(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexattrSettype(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexDestroy(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexGetprioceiling(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexGetspinloops(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexGetyieldloops(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexInit(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexInitForInternalLibc(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexIsowned(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexLock(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexSetprioceiling(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexSetspinloops(void) { raise(SIGTRAP); return SCE_OK; }
int scePthreadMutexSetyieldloops(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexTimedlock(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexTrylock(void) { raise(SIGTRAP); return SCE_OK; }
//int scePthreadMutexUnlock(void) { raise(SIGTRAP); return SCE_OK; }
#endif

}