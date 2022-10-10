CC := g++
UNAME := $(shell uname)
INC := $(INC) include/

HEADERS := $(foreach dir, $(INC), $(wildcard $(dir)/*.h))

nid_hash: nid_hash.cpp $(HEADERS)
	@if [ "$$(uname)" != "Linux" ]; then\
		echo "nid_hash not supported"; \
		exit 1; \
    fi
	$(CC) $(addprefix -I, $(INC)) -g nid_hash.cpp -o $@ -lcrypto

clean:
	rm -f nid_hash *.bin *.o