add_library(nid_hash STATIC nid_hash.cpp)
target_link_libraries(nid_hash crypto sqlite3)