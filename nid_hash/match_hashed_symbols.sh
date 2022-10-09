#!/bin/sh

if [ $# -ne 1 ]; then
	echo "usage: $0 <log_file>"
	exit 1
fi

#sed -E -n "s/SYMBOL DUMP: (.{11})#.#./\1/p" $1
sed -E -n "s/.*(.{11})#.#./\1/p" $1
