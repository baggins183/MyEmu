add_library(elfpatcher STATIC elfpatcher.cpp)
target_link_libraries(elfpatcher pthread dl nid_hash sqlite3)