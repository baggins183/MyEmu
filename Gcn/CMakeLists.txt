#find_package(LLVM PATHS /opt/llvm-git/lib/cmake/llvm/ NO_DEFAULT_PATH REQUIRED CONFIG)
#find_package(LLVM PATHS "$ENV{HOME}/programming/llvm-project/build/lib/cmake/llvm/" NO_DEFAULT_PATH REQUIRED CONFIG)
find_package(LLVM PATHS "$ENV{HOME}/programming/llvm-project/build/install/lib/cmake/llvm/" NO_DEFAULT_PATH REQUIRED CONFIG)
include_directories("${LLVM_INCLUDE_DIR}")

include_directories("${SPIRV-Headers_SOURCE_DIR}/include/spirv/unified1")

add_executable(gcn-dis disassembler.cpp SpirvVisitor.cpp)
target_link_libraries(gcn-dis LLVM)