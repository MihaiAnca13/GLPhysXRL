#include "Environment.h"
#include <iostream>


using std::cout, std::endl;


int main() {
    Environment environment(800, 600, 100.0f, 1.0f, 5, false, false);
    environment.Reset();

    int i = 0;

    while (!glfwWindowShouldClose(environment.window)) {
        auto obs = environment.Step(1.0f, 0.0f);

        i++;
        if (i > 350) {
            environment.Reset();
            i = 0;
        }
    }

    // Clean up
    environment.CleanUp();

    return 0;
}