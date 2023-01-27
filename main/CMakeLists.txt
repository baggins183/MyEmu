add_executable(main main.cpp)
set_property(TARGET main PROPERTY POSITION_INDEPENDENT_CODE ON)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LINKER_ARGS -T "${CMAKE_CURRENT_SOURCE_DIR}/link_linux.ld")
elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
    set(LINKER_ARGS -T "${CMAKE_CURRENT_SOURCE_DIR}/link_bsd.ld")
endif()
target_link_libraries(main pthread nid_hash)
target_link_options(main PRIVATE "${LINKER_ARGS}")

add_executable(alt alt.cpp)
set_property(TARGET alt PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(alt pthread dl nid_hash sqlite3)

