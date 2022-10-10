#!/bin/sh
# print all occurances of nid hashes in the input
# nid hashes are 11 characters followed by #, a captitol letter, #, and another capitol letter
# the last 4 characters are a suffix that somehow encode the module to find the symbol in (I think)
# exclude the suffix from output

if [ $# -ne 1 ]; then
	echo "usage: $0 <log_file>"
	exit 1
fi

#sed -E -n "s/SYMBOL DUMP: (.{11})#.#./\1/p" $1
sed -E -n "s/.*(.{11})#.#./\1/p" $1
