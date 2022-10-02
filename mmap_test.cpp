#include <elf.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

int main() {
    void *addr;
    Elf64_Addr target = 0x10000;
    addr = mmap((void *) target, sysconf(_SC_PAGE_SIZE), 
            PROT_EXEC | PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    printf("%llx\n", addr);

    if ((Elf64_Addr) addr != target) {
        return 1;
    }

    char data[4] = "hi";
    memcpy(addr, data, sizeof(data));

    printf("%s\n", (char *) addr);
/*
    addr = mmap((void *) 0x10000, sysconf(_SC_PAGE_SIZE),
            PROT_EXEC | PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    printf("%llx\n", addr);
*/
    return 0;
}