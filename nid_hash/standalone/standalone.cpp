#include <assert.h>
#include <cstddef>
#include <sstream>
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

#include <sqlite3.h>

#include "nid_hash/nid_hash.h"

struct DbRow {
    std::string symbol;
    std::string hash;
};

static void checkHash(const char *sym, size_t sym_len, std::vector<DbRow> *rows) {
    char hash[12];
    calculateNid(sym, sym_len, hash);

    if (rows) {
        rows->push_back({
            std::string(sym, sym_len),
            std::string(hash, 11)
        });
    } else {
        printf("%.*s -> %s\n", (int) sym_len, sym, hash);
    }
}

static bool processSymbolFile(const char *symbols_path, std::vector<DbRow> *rows) {
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
    uint len;
    uint total_read = 0;
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
        checkHash((char *) sym, len, rows);
        sym += len;
    }

    munmap(fsym_mapping, fsym_len);
    fclose(fsym);

    return true;
}

static bool writeDbRows(sqlite3 *db, std::vector<DbRow> rows) {
    std::stringstream tx;
    tx << "BEGIN TRANSACTION;\n";
    for (DbRow &row: rows) {
        tx <<
             "INSERT OR IGNORE INTO Hashes VALUES " <<
             "('" << row.symbol << "', '" << row.hash << "');";
    }
    tx << "COMMIT;";

    char *err;
    int res = sqlite3_exec(db, tx.str().c_str(), NULL, NULL, &err);
    if (res) {
        printf("Sqlite error inserting rows.\nError: %s\n", err);
        return false;
    }

    return true;
}

// open db and create if doesn't exist
static sqlite3 *openOrCreateDb(const char *path) {
    sqlite3 *db;
    int res = sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (res != SQLITE_OK) {
        fprintf(stderr, "Couldn't open database %s\n", path);
        return nullptr;
    }

    std::stringstream ss;
    ss << 
        "CREATE TABLE IF NOT EXISTS Hashes (symbol TEXT UNIQUE, hash TEXT);"
        "\n"
        "CREATE UNIQUE INDEX IF NOT EXISTS symbol_index\n"
        "   ON Hashes (symbol ASC);\n"
        "\n"
        "CREATE INDEX IF NOT EXISTS hash_index\n"
        "   ON Hashes (hash ASC);\n";

    char *err;
    auto sql = ss.str();
    res = sqlite3_exec(db, sql.c_str(), NULL, NULL, &err);
    if (res) {
        printf("Error creating sqlite table\nMessage: %s\n", err);
        return nullptr;
    }
    return db;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s [--symbols <>] [--hashdb]", argv[0]);
        exit(1);
    }

    const char *symbols_path = NULL;
    const char *hashdb = NULL;

    std::vector<const char *> symbol_args;

    for (int i = 1; i < argc; i++) {
        if (!strcmp("--symbols", argv[i])) {
            symbols_path = argv[++i];
        } else if(!strcmp("--hashdb", argv[i])) {
            hashdb = argv[++i];
        } else {
            symbol_args.push_back(argv[i]);
        }
    }

    if (symbol_args.size() == 0 && !symbols_path) {
        fprintf(stderr, "no symbols or symbol file given\n");
        return 1;
    }

    std::vector<DbRow> rows;

    if (symbols_path) {
        if (!processSymbolFile(symbols_path, hashdb ? &rows : nullptr))
            return 1;
    }

    for (auto sym: symbol_args) {
        checkHash(sym, strlen(sym), hashdb ? &rows : nullptr);
    }

    sqlite3 *db;
    if (hashdb) {
        db = openOrCreateDb(hashdb);
        if (!db) {
            return 1;
        }
        writeDbRows(db, rows);
    }
}