# Exploring OpenGL + PhysX + PyTorch, all in C++

## Sources
https://polyhaven.com
https://matheowis.github.io/HDRI-to-CubeMap/
https://learnopengl.com
https://registry.khronos.org
https://www.youtube.com/@VictorGordan
https://nvidia-omniverse.github.io/PhysX/physx/5.1.3

## Setup
Place libtorch and PhysX under the libs folder.  
Install OpenGL, GLAD and glfw3.  
Add the following to CMake options `-DCMAKE_TOOLCHAIN_FILE=/home/mihai/CLionProjects/vcpkg/scripts/buildsystems/vcpkg.cmake`  

## TODO
- camera on ball toggle
- controls for ball
- create scene in Blender
- load model in OpenGl
- load mesh in Physx
- wrap environment in class
- headless mode
- write RL agent/model
- train loop!