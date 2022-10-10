#!/bin/sh
# search a file or directory for all possible C identifiers according to the definition of a legal C identifier.

if [ $# -lt 1 ]; then
	echo "usage: $0 [files/dirs]+"
	exit 1
fi

grep -ohRE "[_a-zA-Z][_a-zA-Z0-9]*" $@ | sort | uniq