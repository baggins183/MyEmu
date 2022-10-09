CC := g++
INC := $(INC) include/

HEADERS := $(foreach dir, $(INC), $(wildcard $(dir)/*.h))

nid_hash: nid_hash.cpp $(HEADERS)
	$(CC) $(addprefix -I, $(INC)) -g nid_hash.cpp -o $@ -lcrypto

clean:
	rm -f nid_hash *.bin *.o