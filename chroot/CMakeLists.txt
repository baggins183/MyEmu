FILE(GLOB SOURCES *.cpp *.c)

# TODO probably get rid of this chroot library all together
# Use shared library so only one variable in process determines
# whether the program is "chrooted". Could make thread_local in addition
add_library(ps4_chroot SHARED ${SOURCES})