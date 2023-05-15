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

void save_config(const std::string &run_path, AgentConfig agentConfig, const int seed);


void train(std::string run_path) {
    run_path = handle_path(run_path);

    cudaSetDevice(0);

    srand(time(nullptr));
    int seed = rand() % 100000;
    seed = 71687;

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
            .actionPenalty = 0.001f,
    };

    AgentConfig agentConfig{
            .num_epochs = 200,
            .horizon_length = 256,
            .mini_batch_size = 8192,
            .mini_epochs = 16,
            .learning_rate = 1e-4,
            .clip_param = 0.2,
            .value_loss_coef = 0.25,
            .bound_loss_coef = 0.0001,
            .gamma = 0.9,
            .tau = 0.95,
            .reward_multiplier = 1.0,
    };

    save_config(run_path, agentConfig, seed);

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


void save_config(const std::string &run_path, AgentConfig agentConfig, const int seed) {
    // save agent config to a file
    std::ofstream agent_config_file(run_path + "agent_config.txt");
    agent_config_file << "seed: " << seed << std::endl;
    agent_config_file << "num_epochs: " << agentConfig.num_epochs << std::endl;
    agent_config_file << "horizon_length: " << agentConfig.horizon_length << std::endl;
    agent_config_file << "mini_batch_size: " << agentConfig.mini_batch_size << std::endl;
    agent_config_file << "mini_epochs: " << agentConfig.mini_epochs << std::endl;
    agent_config_file << "learning_rate: " << agentConfig.learning_rate << std::endl;
    agent_config_file << "clip_param: " << agentConfig.clip_param << std::endl;
    agent_config_file << "value_loss_coef: " << agentConfig.value_loss_coef << std::endl;
    agent_config_file << "bound_loss_coef: " << agentConfig.bound_loss_coef << std::endl;
    agent_config_file << "gamma: " << agentConfig.gamma << std::endl;
    agent_config_file << "tau: " << agentConfig.tau << std::endl;
    agent_config_file << "reward_multiplier: " << agentConfig.reward_multiplier << std::endl;
    agent_config_file.close();
}
