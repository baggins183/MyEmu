#!/bin/sh
# search a file or directory for all possible C identifiers according to the definition of a legal C identifier.

if [ $# -ne 1 ]; then
	echo "usage: $0 <file or directory>"
	exit 1
fi

grep -ohRE "[_a-zA-Z][_a-zA-Z0-9]*" $1 | sort | uniq