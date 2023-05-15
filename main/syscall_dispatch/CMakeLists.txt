FILE(GLOB SOURCES *.cpp *.c)

add_library(syscall_dispatch SHARED ${SOURCES})