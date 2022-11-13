#include <assert.h>
#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include <byteswap.h>
#include <vector>
#include <sys/mman.h>
#include <string>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <unordered_set>

#include "nid_hash.h"

void checkHash(const char *sym, size_t sym_len, std::unordered_set<std::string> *known_hashes) {
    char hash[12];
    calculateNid(sym, sym_len, hash);
    
    if (known_hashes) {
        if (known_hashes->find(std::string(hash)) != known_hashes->end()) {
            printf("found match: %.*s -> %s\n", (int) sym_len, sym, hash);
        }
    } else {
        printf("%.*s -> %s\n", (int) sym_len, sym, hash);
    }
}

bool readKnownHashes(const char *dict_path, std::unordered_set<std::string> &known_hashes) {
    FILE *fdict;
    void *fdict_mapping = nullptr;
    size_t fdict_len;
    
    fdict = fopen(dict_path, "r");
    if (!fdict) {
        return false;
    }
    fseek(fdict, 0, SEEK_END);
    fdict_len = ftell(fdict);
    fseek(fdict, 0, SEEK_SET);
    int fd = fileno(fdict);
    fdict_mapping = mmap(NULL, fdict_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (fdict_mapping == MAP_FAILED) {
        printf("mmap failed, error %d\n", errno);
        return false;
    }

    unsigned char *hash = (unsigned char *) fdict_mapping;
    int len;
    int total_read = 0;
    for (;;) {
        while (total_read < fdict_len && isspace(*hash)) {
            hash++;
            total_read++;
        }
        len = 0;
        while (total_read < fdict_len && !isspace(hash[len])) {
            len++;
            total_read++;
        }
        if (len == 0) {
            break;
        }
        known_hashes.insert(std::string((const char *) hash, len));
        hash += len;
    }

    munmap(fdict_mapping, fdict_len);
    fclose(fdict);

    return true;
}

bool processSymbolFile(const char *symbols_path, std::unordered_set<std::string> *known_hashes) {
    FILE *fsym;
    void *fsym_mapping = nullptr;
    size_t fsym_len;
    
    fsym = fopen(symbols_path, "r");
    if (!fsym) {
        return false;
    }
    fseek(fsym, 0, SEEK_END);
    fsym_len = ftell(fsym);
    fseek(fsym, 0, SEEK_SET);
    int fd = fileno(fsym);
    fsym_mapping = mmap(NULL, fsym_len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (fsym_mapping == MAP_FAILED) {
        printf("mmap failed, error %d\n", errno);
        return false;
    }

    unsigned char *sym = (unsigned char *) fsym_mapping;
    int len;
    int total_read = 0;
    for (;;) {
        while (total_read < fsym_len && isspace(*sym)) {
            sym++;
            total_read++;
        }
        len = 0;
        while (total_read < fsym_len && !isspace(sym[len])) {
            len++;
            total_read++;
        }
        if (len == 0) {
            break;
        }
        checkHash((char *) sym, len, known_hashes);
        sym += len;
    }

    munmap(fsym_mapping, fsym_len);
    fclose(fsym);

    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        exit(1);
    }

    std::unordered_set<std::string> known_hashes;

    const char *dict_path = NULL;
    const char *symbols_path = NULL;

    std::vector<const char *> symbol_args;

    for (int i = 1; i < argc; i++) {
        if (!strcmp("--dict", argv[i])) {
            dict_path = argv[++i];
        } else if (!strcmp("--symbols", argv[i])) {
            symbols_path = argv[++i];
        }  else {
            symbol_args.push_back(argv[i]);
        }
    }

    if (symbol_args.size() == 0 && !symbols_path) {
        fprintf(stderr, "no symbols or symbol file given\n");
        return 1;
    }

    if (dict_path) {
        readKnownHashes(dict_path, known_hashes);
    }

    if (symbols_path) {
        if (!processSymbolFile(symbols_path, dict_path ? &known_hashes : nullptr))
            return 1;
    }

    for (auto sym: symbol_args) {
        checkHash(sym, strlen(sym), dict_path ? &known_hashes : nullptr);
    }
}