CC := g++
UNAME := $(shell uname)
INC := $(INC) include/

HEADERS := $(foreach dir, $(INC), $(wildcard $(dir)/*.h))

standalone: nid_hash.cpp nid_hash.h $(HEADERS) standalone.cpp
	$(CC) $(addprefix -I, $(INC)) -g -o $@ standalone.cpp nid_hash.cpp -lcrypto

clean:
	rm -f nid_hash *.bin *.o