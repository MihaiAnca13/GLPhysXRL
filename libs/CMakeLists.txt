add_subdirectory(PhysX)

add_library(third_party INTERFACE)

target_link_libraries(third_party INTERFACE PhysX)
target_include_directories(third_party INTERFACE PhysX)