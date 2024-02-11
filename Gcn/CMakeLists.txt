#find_package(LLVM PATHS /opt/llvm-git/lib/cmake/llvm/ NO_DEFAULT_PATH REQUIRED CONFIG)
#find_package(LLVM PATHS "$ENV{HOME}/programming/llvm-project/build/lib/cmake/llvm/" NO_DEFAULT_PATH REQUIRED CONFIG)

find_package(LLVM PATHS "$ENV{HOME}/programming/llvm-project/build/install/lib/cmake/llvm/" NO_DEFAULT_PATH REQUIRED CONFIG)
find_package(MLIR PATHS "$ENV{HOME}/programming/llvm-project/build/install/lib/cmake/mlir/" NO_DEFAULT_PATH REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

include_directories(${CMAKE_CURRENT_BINARY_DIR}/GcnDialect)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_subdirectory(GcnDialect)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRGcn
        LLVM
)
message(STATUS "LIBS!!!!!!!!!!!!!!" ${LIBS})

#FILE(GLOB SOURCES *.cpp *.c)
#add_llvm_executable(gcn-dis ${SOURCES})
add_executable(gcn-dis disassembler.cpp)
target_compile_options(gcn-dis PUBLIC -fno-rtti)
target_link_libraries(gcn-dis PRIVATE ${LIBS})
