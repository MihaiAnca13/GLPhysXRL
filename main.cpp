#include "Agent.h"
#include "Environment.h"
#include <iostream>
#include <cstdlib>


using std::cout, std::endl;


int main() {
    srand(time(NULL));

    Config config{.width = 800,
                  .height = 600,
                  .bounds = 100.0f,
                  .ballDensity = 1.0f,
                  .numSubsteps =5 ,
                  .manualControl = false,
                  .headless = true,
                  .maxSteps = 1024,
                  .threshold = 0.03f,
                  .bonusAchievedReward = 10.0f,
    };

    Agent agent = Agent();

    Environment environment(config);
    environment.Reset();

    agent.Train(&environment);

//    while (!glfwWindowShouldClose(environment.window)) {
//        auto stepRes = environment.Step({1.0f, 0.0f});
//
//        cout << "reward: " << stepRes.reward << endl;
////        auto obs = stepRes.observation;
////        cout << obs.ballPosition.x << " " << obs.ballPosition.y << " " << obs.ballPosition.z << endl;
//
//        if (stepRes.done) {
//            environment.Reset();
//        }
//    }

    // Clean up
    environment.CleanUp();

    return 0;
}