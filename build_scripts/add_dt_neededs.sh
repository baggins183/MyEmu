#/bin/sh

# add DT_NEEDED tags to the main executable.
# This has an effect like running the main executable with LD_PRELOAD. Symbols defined in
# the compatibility libs will take precedence over definitions in the ps4 elf's, but not over definitions in
# the preexisting DT_NEEDEDs in the main exe, such as glibc

if [ $# -lt 2 ]; then
    echo "usage: $0 <elf to patch> [needed_1] ... [needed_n]"
    exit -1
fi

ELF=$1
shift

ALREADY_NEEDED=($(patchelf --print-needed $ELF))
for N in "${ALREADY_NEEDED[@]}"; do
    patchelf --remove-needed $N $ELF
done

for N in $@; do
    BASE="$(basename $N)"
    patchelf --add-needed $BASE $ELF
done

# These will prepend the preexisting dependencies back into the main ELF
for ((i=${#ALREADY_NEEDED[@]}-1; i>=0; i--)); do
    N="${ALREADY_NEEDED[$i]}"

    patchelf --add-needed $N $ELF
done