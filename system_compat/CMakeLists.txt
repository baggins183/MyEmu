FILE(GLOB SOURCES *.cpp *.c)

add_library(system_compat SHARED ${SOURCES})
