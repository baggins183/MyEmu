# Specify how to build the result of the library
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libpmparser.a
    COMMAND ${CMAKE_COMMAND} -E env BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR} CFLAGS="-g -O0" make
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/proc_maps_parser"
)

add_custom_target(interm_target
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libpmparser.a
)

add_library(proc_maps_parser STATIC IMPORTED GLOBAL)
add_dependencies(proc_maps_parser interm_target)

set_target_properties(proc_maps_parser
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libpmparser.a
)