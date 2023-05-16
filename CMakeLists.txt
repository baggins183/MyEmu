cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
            "Default build type: Debug" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

project(MyEmu C CXX ASM)

include_directories("include/")

add_subdirectory(system_compat)
add_subdirectory(ps4lib_overloads)
add_subdirectory(nid_hash)
add_subdirectory(elfpatcher)
add_subdirectory(main)
