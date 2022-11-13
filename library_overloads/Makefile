CC := g++
CARGS := -Wall
UNAME := $(shell uname)
CLIBS := -lpthread

all: $(wildcard *.c)
	@for SHARED_LIB in $(addsuffix .so, $(basename $^)); do \
	$(MAKE) $$SHARED_LIB ; \
	done;

%.so: %.c
	$(CC) $(CARGS) -g -fpic -shared -o $@ $< $(CLIBS)

clean:
	rm -f *.so