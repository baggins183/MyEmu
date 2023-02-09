#foreach(lib ${ALL_LIBS})
#    get_filename_component(basename ${lib} NAME_WE)
#    add_library(${basename} SHARED ${lib})
#    message(STATUS "${basename}")
#endforeach()

add_library(every SHARED every.c)
target_link_libraries(every pthread)

add_library(scepthread SHARED scepthread.c)
target_link_libraries(scepthread pthread)

set(LIBRARY_PRELOAD_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
message(STATUS ${LIBRARY_PRELOAD_DIR})