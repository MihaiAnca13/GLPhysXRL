#include "Environment.h"
#include <iostream>


using std::cout, std::endl;


int main() {
    Config config{.width = 800,
                  .height = 600,
                  .bounds = 100.0f,
                  .ballDensity = 1.0f,
                  .numSubsteps =5 ,
                  .manualControl = true,
                  .headless = false,
                  .maxSteps = 1000,
                  .threshold = 0.03f,
                  .bonusAchievedReward = 10.0f,
    };

    Environment environment(config);
    environment.Reset();

    while (!glfwWindowShouldClose(environment.window)) {
        auto stepRes = environment.Step(1.0f, 0.0f);

        cout << "reward: " << stepRes.reward << endl;
//        auto obs = stepRes.observation;
//        cout << obs.ballPosition.x << " " << obs.ballPosition.y << " " << obs.ballPosition.z << endl;

        if (stepRes.done) {
            environment.Reset();
        }
    }

    // Clean up
    environment.CleanUp();

    return 0;
}