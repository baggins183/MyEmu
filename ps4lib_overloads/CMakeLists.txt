#foreach(lib ${ALL_LIBS})
#    get_filename_component(basename ${lib} NAME_WE)
#    add_library(${basename} SHARED ${lib})
#    message(STATUS "${basename}")
#endforeach()

add_library(every SHARED every.c)
target_link_libraries(every pthread)
# temp solution: add dependency to libevery.so so it can find functions to wrap in base module
#get_property(LIBNAME TARGET LIBRARY_OUTPUT_NAME every)
add_custom_command(TARGET every POST_BUILD COMMAND
    ${PROJECT_SOURCE_DIR}/build_scripts/add_dt_neededs.sh $<TARGET_FILE:every> "libkernel.prx.native"
)

add_library(scepthread SHARED scepthread.c)
target_link_libraries(scepthread pthread)

set(LIBRARY_PRELOAD_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
message(STATUS ${LIBRARY_PRELOAD_DIR})