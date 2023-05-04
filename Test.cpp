#include "Agent.h"
#include "Environment.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <ctime>
#include "Modes.h"

using std::cout, std::endl;


void test(std::string load_path) {
    // check if load_path exists
    if (!std::filesystem::exists(load_path)) {
        cout << "Load path does not exist!" << endl;
        return;
    }

    cudaSetDevice(0);

    srand(time(nullptr));
    int seed = rand() % 100000;

    cout << "Setting seed to " << seed << endl;

    srand(seed);
    torch::manual_seed(seed);

    EnvConfig envConfig{
            .width = 800,
            .height = 600,
            .bounds = 50.0f,
            .ballDensity = 1.0f,
            .numSubsteps = 5,
            .manualControl = false,
            .headless = false,
            .maxSteps = 256,  // 1024
            .threshold = 0.1f,
            .bonusAchievedReward = 1.0f,
            .num_envs = 12,
    };

    AgentConfig agentConfig{
            .num_epochs = 1000,
            .horizon_length = 256,
            .mini_batch_size = 8192,
            .mini_epochs = 8,
            .learning_rate = 1e-4,
            .clip_param = 0.2,
            .value_loss_coef = 0.5,
            .bound_loss_coef = 0.0001,
            .gamma = 0.99,
            .tau = 0.95,
            .reward_multiplier = 1.0,
    };

    Environment environment(envConfig);
    environment.Reset();

    // load_path contains 'weights' so we need to remove it to get the log_path
    std::string log_path = load_path.substr(0, load_path.find_last_of('/') - 8) + "_test/";
    // create the missing folders
    std::filesystem::create_directory(log_path);
    std::filesystem::create_directory(log_path + "summaries");
    std::filesystem::create_directory(log_path + "weights");

    Agent agent = Agent(agentConfig, &environment, log_path);

    agent.Test(load_path);

    // Clean up
    environment.CleanUp();
}