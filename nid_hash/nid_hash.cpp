#include <assert.h>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include <vector>
#include <sys/mman.h>
#include <string>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#ifdef __FreeBSD__
#include <sys/endian.h>
#else
#include <byteswap.h>
#endif

const char b64chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static int b64_encode_nid(unsigned char *in, char *out)
{
    const size_t len = 8;
	const size_t  elen = 12;
	size_t  i;
	size_t  j;
	size_t  v;

	if (in == NULL || len == 0)
		return 0;

	out[elen - 1] = '\0';

	for (i=0, j=0; i<len; i+=3, j+=4) {
		v = in[i];
		v = i+1 < len ? v << 8 | in[i+1] : v << 8;
		v = i+2 < len ? v << 8 | in[i+2] : v << 8;

		out[j]   = b64chars[(v >> 18) & 0x3F];
		out[j+1] = b64chars[(v >> 12) & 0x3F];
		if (i+1 < len) {
			out[j+2] = b64chars[(v >> 6) & 0x3F];
		} else {
			out[j+2] = '=';
		}
		if (i+2 < len) {
			out[j+3] = b64chars[v & 0x3F];
		} else {
			out[j+3] = '=';
		}
	}

    out[elen - 1] = '\0';

	return 1;
}


void calculateNid(const char *sym, size_t sym_len, /* ret */ char *hash) {
    const unsigned char nid_suffix_key[16] = {
        0x51, 0x8D, 0x64, 0xA6,
        0x35, 0xDE, 0xD8, 0xC1,
        0xE6, 0xB0, 0x39, 0xB1,
        0xC3, 0xE5, 0x52, 0x30
    };

    int mbuf_sz = sym_len + sizeof(nid_suffix_key);
    unsigned char *mbuf = (unsigned char *) malloc(mbuf_sz);
    memcpy(mbuf, sym, sym_len);
    memcpy(&mbuf[sym_len], nid_suffix_key, sizeof(nid_suffix_key));

    unsigned char digest[SHA_DIGEST_LENGTH];

    assert(SHA1(mbuf, mbuf_sz, digest));
    unsigned char base64_in[8];

#ifdef __FreeBSD__
    *((uint64_t *) base64_in) = bswap64(*(uint64_t *) digest);
#else
    *((uint64_t *) base64_in) = bswap_64(*(uint64_t *) digest);
#endif

    b64_encode_nid(base64_in, hash);
}