CC := g++
CARGS := -Wall
UNAME := $(shell uname)
INC := $(INC) include/ ./

ifeq ($(UNAME), Linux)
	LD_ARGS := -Wl,-T,link_linux.ld
else ifeq ($(UNAME), FreeBSD)
	LD_ARGS := -Wl,-T,link_bsd.ld
endif

HEADERS := $(foreach dir, $(INC), $(wildcard $(dir)/*.h))

main: main.cpp $(HEADERS) ../nid_hash/nid_hash.h ../nid_hash/nid_hash.cpp
	$(CC) $(CARGS) $(addprefix -I, $(INC)) -g -fpie $(LD_ARGS) -o $@ main.cpp ../nid_hash/nid_hash.cpp -lpthread -lcrypto

alt: alt.cpp $(HEADERS) ../nid_hash/nid_hash.h ../nid_hash/nid_hash.cpp
	$(CC) $(CARGS) $(addprefix -I, $(INC)) -g -fpie $(LD_ARGS) -o $@ alt.cpp ../nid_hash/nid_hash.cpp -lpthread -lcrypto

clean:
	rm -f main alt *.bin *.o