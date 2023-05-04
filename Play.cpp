#include "Agent.h"
#include "Environment.h"
#include <iostream>
#include <cuda_runtime.h>
#include "Modes.h"

using std::cout, std::endl;

void play() {
    cudaSetDevice(0);

    EnvConfig envConfig{
            .width = 800,
            .height = 600,
            .bounds = 50.0f,
            .ballDensity = 1.0f,
            .numSubsteps = 5,
            .manualControl = true,
            .headless = false,
            .maxSteps = 256,  // 1024
            .threshold = 0.1f,
            .bonusAchievedReward = 1.0f,
            .num_envs = 12,
    };

    Environment environment(envConfig);
    environment.Reset();

    auto action = torch::zeros({envConfig.num_envs, 2}, torch::TensorOptions().dtype(torch::kFloat32));

    while (!glfwWindowShouldClose(environment.window)) {
        auto stepRes = environment.Step(action, nullptr);

        cout << "reward: " << stepRes.reward[0].item<float>() << endl;
//        auto obs = stepRes.observation;
//        cout << "ball pos: " << obs[0][0].item<float>() << ", " << obs[0][1].item<float>() << ", " << obs[0][2].item<float>() << endl;

        if (stepRes.done[0].item<bool>()) {
            environment.Reset();
        }
    }

    // Clean up
    environment.CleanUp();
}