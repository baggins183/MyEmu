#include "Common.h"
#include <signal.h>

extern "C" {

void *PS4FUN(malloc)(size_t size) {
    auto *rv = malloc(size);
    return rv;
}

void PS4FUN(free)( void *ptr ) {
    free(ptr);
}

void *PS4FUN(calloc)(size_t nitems, size_t size) {
    //raise(SIGTRAP);
    auto *rv = calloc(nitems, size);
    return rv;
}

}