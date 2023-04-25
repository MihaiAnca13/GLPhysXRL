#include "Agent.h"
#include "Environment.h"
#include <iostream>
#include <cstdlib>


using std::cout, std::endl;


int main() {
    srand(time(NULL));

    EnvConfig envConfig{.width = 800,
                  .height = 600,
                  .bounds = 100.0f,
                  .ballDensity = 1.0f,
                  .numSubsteps =5 ,
                  .manualControl = false,
                  .headless = true,
                  .maxSteps = 1024,
                  .threshold = 0.03f,
                  .bonusAchievedReward = 10.0f,
                  .num_envs = 2
    };

    AgentConfig agentConfig{.num_epochs = 10,
                            .horizon_length = 16,
                            .mini_batch_size = 2,
                            .learning_rate = 1e-3,
                            .clip_param = 0.2,
                            .value_loss_coef = 0.5,
                            .gamma = 0.9,
                            .tau = 0.95,
                            .reward_multiplier = 0.01,
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
////        cout << obs.ballPosition.x << " " << obs.ballPosition.y << " " << obs.ballPosition.z << endl;
//
//        if (stepRes.done[0].item<bool>()) {
//            environment.Reset();
//        }
//    }

    // Clean up
    environment.CleanUp();

    return 0;
}