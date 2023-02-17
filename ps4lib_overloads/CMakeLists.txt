add_subdirectory(SceLibKernel)

add_library(every SHARED every.cpp)
target_link_libraries(every pthread)
# TODO get rid of "every" and do 1:1 compat lib to existing sys module
# example libkernel.so : libkernel.prx