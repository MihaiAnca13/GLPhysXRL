#include "Environment.h"
#include <iostream>


using std::cout, std::endl;


int main() {
    Environment environment(800, 600, 100.0f, 1.0f);

    while (!glfwWindowShouldClose(environment.window)) {
        environment.Step();
    }

    // Clean up
    environment.CleanUp();

    return 0;
}