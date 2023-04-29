#include "Agent.h"
#include "Environment.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>


using std::cout, std::endl;


int main() {
    // check if file exists
    if (std::filesystem::exists("../runs/run_name/summaries/events.out.tfevents.mihai-desktop"))
        // remove file
        std::filesystem::remove("../runs/run_name/summaries/events.out.tfevents.mihai-desktop");

    cudaSetDevice(0);

    srand(time(NULL));

    EnvConfig envConfig{.width = 800,
                  .height = 600,
                  .bounds = 100.0f,
                  .ballDensity = 1.0f,
                  .numSubsteps = 5,
                  .manualControl = false,
                  .headless = false,
                  .maxSteps = 256,  // 1024
                  .threshold = 0.03f,
                  .bonusAchievedReward = 10.0f,
                  .num_envs = 256,
    };

    AgentConfig agentConfig{.num_epochs = 1000,
                            .horizon_length = 32,
                            .mini_batch_size = 2048,
                            .learning_rate = 1e-4,
                            .clip_param = 0.2,
                            .value_loss_coef = 0.5,
                            .gamma = 0.9,
                            .tau = 0.95,
                            .reward_multiplier = 1.0,
    };

    Environment environment(envConfig);
    environment.Reset();

    Agent agent = Agent(agentConfig, &environment);

    agent.Train();

//    auto action = torch::zeros({envConfig.num_envs, 2}, torch::TensorOptions().dtype(torch::kFloat32));
//    while (!glfwWindowShouldClose(environment.window)) {
//        auto stepRes = environment.Step(action);
//
//        cout << "reward: " << stepRes.reward << endl;
////        auto obs = stepRes.observation;
////        cout << "ball pos: " << obs[0][0] << ", " << obs[0][1] << ", " << obs[0][2] << endl;
//
//        if (stepRes.done[0].item<bool>()) {
//            environment.Reset();
//        }
//    }

    // Clean up
    environment.CleanUp();

    return 0;
}