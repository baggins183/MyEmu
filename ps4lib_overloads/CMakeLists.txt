file(GLOB ALL_LIBS "*.c" ".cpp")
message(STATUS "LIBRARY OVERLOADS" "${ALL_LIBS}")

#foreach(lib ${ALL_LIBS})
#    get_filename_component(basename ${lib} NAME_WE)
#    add_library(${basename} SHARED ${lib})
#    message(STATUS "${basename}")
#endforeach()

add_library(every SHARED every.c)
target_link_libraries(every pthread)

add_library(scepthread SHARED scepthread.c)
target_link_libraries(scepthread pthread)