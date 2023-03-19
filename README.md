# Exploring OpenGL + PhysX + PyTorch, all in C++

## Sources
https://polyhaven.com
https://matheowis.github.io/HDRI-to-CubeMap/
https://learnopengl.com
http://www.opengl-tutorial.org
https://registry.khronos.org
https://www.youtube.com/@VictorGordan
https://nvidia-omniverse.github.io/PhysX/physx/5.1.3
https://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
https://academo.org/demos/rotation-about-point/
https://stackoverflow.com/questions/20923232/how-to-rotate-a-vector-by-a-given-direction

## Setup
Place libtorch and PhysX under the libs folder.  
Install OpenGL, GLAD and glfw3.  
Add the following to CMake options `-DCMAKE_TOOLCHAIN_FILE=/home/mihai/CLionProjects/vcpkg/scripts/buildsystems/vcpkg.cmake`  

## TODO
- load mesh in Physx
- wrap environment in class
- headless mode
- write RL agent/model
- train loop!