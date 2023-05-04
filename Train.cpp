#include "Agent.h"
#include "Environment.h"
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <ctime>
#include "Modes.h"

using std::cout, std::endl;

std::string handle_path(std::string path);


void train(std::string run_path) {
    run_path = handle_path(run_path);

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
            .headless = true,
            .maxSteps = 256,  // 1024
            .threshold = 0.1f,
            .bonusAchievedReward = 1.0f,
            .num_envs = 256,
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

    Agent agent = Agent(agentConfig, &environment, run_path);

    agent.Train();

    // Clean up
    environment.CleanUp();
}


std::string handle_path(std::string path) {
    // check if run_path ends in /
    if (path.back() != '/') {
        path += '/';
    }

    // check if run_path exists
    if (std::filesystem::exists(path)) {
        // add timestamp to run_path
        std::time_t t = std::time(nullptr);
        char mbstr[100];
        std::strftime(mbstr, sizeof(mbstr), "%Y-%m-%d-%H-%M-%S", std::localtime(&t));
        // add timestamp to run_path before the last /
        path.insert(path.find_last_of('/'), mbstr);
        path += '/';
    }

    // create the missing folders
    std::filesystem::create_directory(path);
    std::filesystem::create_directory(path + "summaries");
    std::filesystem::create_directory(path + "weights");

    return path;
}