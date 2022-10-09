export CC := g++
# don't prepend -I yet
export INC := $(realpath Common/include)

#all: main nid_hash

all: main nid_hash

main:
	$(MAKE) -C main/

nid_hash:
	$(MAKE) -C nid_hash/ nid_hash

clean:
	$(MAKE) -C main/ clean
	$(MAKE) -C nid_hash/ clean


.PHONY: main nid_hash clean