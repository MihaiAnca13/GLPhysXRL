add_subdirectory(PhysX)
add_subdirectory(stb)

add_library(third_party INTERFACE)

target_link_libraries(third_party INTERFACE PhysX stb)
target_include_directories(third_party INTERFACE PhysX stb)