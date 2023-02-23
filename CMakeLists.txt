cmake_minimum_required(VERSION 3.24)
project(c_ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(Torch REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(c_ml main.cpp)
#add_executable(c_ml draw_stuff.cpp glad.c)
#add_executable(c_ml trying_physics.cpp ${SOURCES})

# Add third party directory
add_subdirectory(libs)

# Include directories
include_directories(
        ./include
)

set(ALL_LIBS
        third_party
        )

target_include_directories(c_ml PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(c_ml PUBLIC "${TORCH_LIBRARIES}" glfw OpenGL::GL ${CMAKE_DL_LIBS} ${ALL_LIBS})