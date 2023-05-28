find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams)

add_library(elfpatcher STATIC elfpatcher.cpp elfjson.cpp)
target_link_libraries(elfpatcher pthread dl nid_hash sqlite3 nlohmann_json::nlohmann_json ${Boost_IOSTREAMS_LIBRARY})