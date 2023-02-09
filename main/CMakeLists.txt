add_executable(old old.cpp)
set_property(TARGET old PROPERTY POSITION_INDEPENDENT_CODE ON)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(LINKER_ARGS -T "${CMAKE_CURRENT_SOURCE_DIR}/link_linux.ld")
elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
    set(LINKER_ARGS -T "${CMAKE_CURRENT_SOURCE_DIR}/link_bsd.ld")
endif()
target_link_libraries(old pthread nid_hash)
target_link_options(old PRIVATE "${LINKER_ARGS}")

add_executable(main main.cpp)
set_property(TARGET main PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(main pthread dl nid_hash sqlite3)

message(STATUS testing: ${LIBRARY_PRELOAD_DIR})

file(GLOB PRELOAD_LIBS "${LIBRARY_PRELOAD_DIR}/*.so")
message(STATUS pre ${PRELOAD_LIBS})
add_custom_command(TARGET main POST_BUILD COMMAND
    ${PROJECT_SOURCE_DIR}/build_scripts/add_dt_neededs.sh main ${PRELOAD_LIBS}
)
#target_link_options(main PRIVATE --as-needed libevery.so)

