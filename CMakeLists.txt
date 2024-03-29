cmake_minimum_required(VERSION 3.24)
project(c_ml)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Os")
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(Torch_DIR "/home/mihai/CLionProjects/c_ml/libs/libtorch/share/cmake/Torch")

find_package(Threads REQUIRED)
find_package(Torch REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(imgui REQUIRED)
find_package(assimp REQUIRED)
find_package(tensorboard_logger REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(c_ml
        main.cpp
#        playground.cpp
        rendering/glad.c
        rendering/Camera.cpp
        rendering/shaderClass.cpp
        ${SOURCES}
        rendering/Shadow.cpp
        objects/Sphere.cpp
        objects/MyObjects.cpp
        objects/Table.cpp
        rendering/Skybox.cpp
        rendering/SpringArmCamera.cpp
        utils/UtilsAngles.cpp
        objects/Mesh.cpp
        objects/Model.cpp
        Environment.cpp
        Agent.cpp LinearNetwork.cpp RunningMean.cpp Train.cpp Test.cpp Play.cpp include/Modes.h)

# Add third party directory
add_subdirectory(libs)

# Include directories
include_directories(
        ./include
)

configure_file(shaders/default.frag ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.frag COPYONLY)
configure_file(shaders/default.vert ${CMAKE_CURRENT_BINARY_DIR}/shaders/default.vert COPYONLY)
configure_file(shaders/skybox.frag ${CMAKE_CURRENT_BINARY_DIR}/shaders/skybox.frag COPYONLY)
configure_file(shaders/skybox.vert ${CMAKE_CURRENT_BINARY_DIR}/shaders/skybox.vert COPYONLY)
configure_file(shaders/shadowMap.frag ${CMAKE_CURRENT_BINARY_DIR}/shaders/shadowMap.frag COPYONLY)
configure_file(shaders/shadowMap.vert ${CMAKE_CURRENT_BINARY_DIR}/shaders/shadowMap.vert COPYONLY)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(ALL_LIBS
        third_party
        )

target_include_directories(c_ml PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(c_ml PRIVATE ${CMAKE_DL_LIBS} Threads::Threads ssl crypto imgui::imgui assimp::assimp)
target_link_libraries(c_ml PUBLIC "${TORCH_LIBRARIES}" glfw OpenGL::GL ${ALL_LIBS} tensorboard_logger::tensorboard_logger)