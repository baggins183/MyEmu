#!/bin/sh

#USAGE="Usage: $0 lib*.spv"
#
#if [ "$#" == "0" ]; then
#	echo "$USAGE"
#	exit 1
#fi
#
#FILE=shift
#
#while (( "$#" )); do
#    HEADER="$1.h"
#    hexdump -ve '1/1 "0x%.2x,"' $1 > $HEADER
#
#    shift
#done

dxc -spirv libMiscHelpers.hlsl -T lib_6_7 -fspv-target-env=vulkan1.3 -Fh libMiscHelpers.spv.h -Vn g_miscHelpersBytes