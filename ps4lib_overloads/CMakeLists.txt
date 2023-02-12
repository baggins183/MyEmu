#foreach(lib ${ALL_LIBS})
#    get_filename_component(basename ${lib} NAME_WE)
#    add_library(${basename} SHARED ${lib})
#    message(STATUS "${basename}")
#endforeach()

add_library(every SHARED every.c)
target_link_libraries(every pthread)
# TODO get rid of "every" and do 1:1 compat lib to existing sys module
# example libkernel.so : libkernel.prx
add_custom_command(TARGET every POST_BUILD COMMAND
    ${PROJECT_SOURCE_DIR}/build_scripts/add_dt_neededs.sh $<TARGET_FILE:every> "libkernel.prx.native"
)

add_library(scepthread SHARED scepthread.c)
target_link_libraries(scepthread pthread)

set(LIBRARY_PRELOAD_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
message(STATUS ${LIBRARY_PRELOAD_DIR})