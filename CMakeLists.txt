cmake_minimum_required(VERSION 3.24)
project(c_ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Os")
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(Torch_DIR "/home/mihai/CLionProjects/c_ml/libs/libtorch/share/cmake/Torch")

find_package(Threads REQUIRED)
find_package(Torch REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(imgui REQUIRED)
find_package(assimp REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(c_ml
        main.cpp
        glad.c
        Camera.cpp
        shaderClass.cpp
        ${SOURCES}
        Shadow.cpp
        objects/Sphere.cpp
        objects/MyObjects.cpp
        objects/Table.cpp
        Skybox.cpp
        SpringArmCamera.cpp
        UtilsAngles.cpp
        objects/Mesh.cpp objects/Model.cpp)

# Add third party directory
add_subdirectory(libs)

# Include directories
include_directories(
        ./include
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/default.frag ${CMAKE_CURRENT_BINARY_DIR}/default.frag COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/default.vert ${CMAKE_CURRENT_BINARY_DIR}/default.vert COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/skybox.frag ${CMAKE_CURRENT_BINARY_DIR}/skybox.frag COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/skybox.vert ${CMAKE_CURRENT_BINARY_DIR}/skybox.vert COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shadowMap.frag ${CMAKE_CURRENT_BINARY_DIR}/shadowMap.frag COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shadowMap.vert ${CMAKE_CURRENT_BINARY_DIR}/shadowMap.vert COPYONLY)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(ALL_LIBS
        third_party
        )

target_include_directories(c_ml PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(c_ml PRIVATE ${CMAKE_DL_LIBS} Threads::Threads ssl crypto imgui::imgui assimp::assimp)
target_link_libraries(c_ml PUBLIC "${TORCH_LIBRARIES}" glfw OpenGL::GL ${ALL_LIBS})