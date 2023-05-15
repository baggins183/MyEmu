FILE(GLOB SOURCES *.cpp *.c)

add_library(ps4_stdlib_wrappers SHARED ${SOURCES})
target_link_libraries(ps4_stdlib_wrappers ps4_chroot)