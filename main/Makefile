CC := g++
UNAME := $(shell uname)
INC := $(INC) include/ ./

ifeq ($(UNAME), Linux)
	LD_ARGS := -Wl,-T,link_linux.ld
else ifeq ($(UNAME), FreeBSD)
	LD_ARGS := -Wl,-T,link_bsd.ld
endif

HEADERS := $(foreach dir, $(INC), $(wildcard $(dir)/*.h))

main: main.cpp $(HEADERS)
	$(CC) $(addprefix -I, $(INC)) -g -fpie $(LD_ARGS) main.cpp -o $@ -lpthread

clean:
	rm -f main *.bin *.o