FILE(GLOB SOURCES *.cpp *.c)

add_library(compat_misc SHARED ${SOURCES})
target_link_libraries(compat_misc)