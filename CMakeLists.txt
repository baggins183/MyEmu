cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -g")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

project(MyEmu)

include_directories("include/")

add_subdirectory(ps4lib_overloads)
add_subdirectory(nid_hash)
add_subdirectory(main)
