FILE(GLOB SOURCES *.cpp *.c)

add_library(compat_libc_internal SHARED ${SOURCES})
target_link_libraries(compat_libc_internal)

#add_custom_command(TARGET compat_libkernel POST_BUILD COMMAND
#    ${PROJECT_SOURCE_DIR}/build_scripts/add_dt_neededs.sh $<TARGET_FILE:compat_libkernel> "libkernel.prx.native"
#)
