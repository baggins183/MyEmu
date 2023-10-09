find_package(LLVM REQUIRED CONFIG)

add_executable(gcn-dis disassembler.cpp)
target_link_libraries(gcn-dis LLVM)