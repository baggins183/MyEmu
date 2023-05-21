#ifndef _EXPORTED_WRAPPERS_H_
#define _EXPORTED_WRAPPERS_H_

// Include wrappers that are called from multiple places.
// Like mmap_wrapper, which is called by the mmap syscall handler and
// the _ps4__mmap libc wrapper. 

#include <sys/types.h>

void *mmap_wrapper(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int open_wrapper(const char *pathname, int flags, mode_t mode);

#endif // _EXPORTED_WRAPPERS_H_