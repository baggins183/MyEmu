#ifndef _NID_HASH_H_
#define _NID_HASH_H_

#include <stddef.h>
void calculateNid(const char *sym, size_t sym_len, /* ret */ char *hash);

#endif // _NID_HASH_H