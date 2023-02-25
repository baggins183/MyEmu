FILE(GLOB SOURCES *.cpp *.c)

add_library(FreeBsdCompat SHARED ${SOURCES})