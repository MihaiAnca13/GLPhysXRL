//
// Created by mihai on 05/04/23.
//

#ifndef C_ML_AGENT_H
#define C_ML_AGENT_H

#include <vector>
#include "LinearNetwork.h"
#include "Environment.h"
#include <torch/torch.h>
#include <torch/optim/adamw.h>
#include <tensorboard_logger.h>
#include "RunningMean.h"
#include "TimeMe.h"


using namespace torch;


typedef struct {
    Tensor obs;
    Tensor action;
    Tensor reward;
    Tensor next_obs;
    Tensor done;
    Tensor value;
    Tensor old_log_prob;
    Tensor returns;
    Tensor advantages;
} Transition;


typedef struct {
    int num_epochs;
    int horizon_length;
    int mini_batch_size;
    float learning_rate;
    float clip_param;
    float value_loss_coef;
    float bound_loss_coef;
    float gamma;
    float tau;
    float reward_multiplier;
} AgentConfig;


class Agent {
public:
    // config parameters
    int num_epochs = 10;
    int horizon_length = 16;
    int mini_batch_size = 2;
    float learning_rate = 1e-3;
    float clip_param = 0.2;
    float value_loss_coef = 0.5;
    float bound_loss_coef = 0.0001;
    float gamma = 0.9;
    float tau = 0.95;
    float reward_multiplier = 0.01;

    float last_reward = 0.0f;

    int obs_size = 7;
    int action_size = 2;

    TensorBoardLoggerOptions options{1000000, 5, false};
    TensorBoardLogger logger = TensorBoardLogger("../runs/run_name/summaries/events.out.tfevents.mihai-desktop", options);

    std::vector<Transition> memory;
    DeviceType device = torch::kCUDA;

    TensorOptions floatOptions = torch::TensorOptions().dtype(torch::kFloat32).device(device).layout(torch::kStrided).requires_grad(false);
    TensorOptions boolOptions = torch::TensorOptions().dtype(torch::kBool).device(device).layout(torch::kStrided).requires_grad(false);

    RunningMeanStd value_mean_std = RunningMeanStd(1, 1e-5, false, false);

    Agent(AgentConfig config, Environment *env);

    Network net;
    optim::AdamW optimizer;

    Environment* env;
    int num_envs = 1;

    void Train();

    Tensor get_value(Tensor observation);

    void SetTrain();
    void SetEval();

    // GAE function for calculating the advantage
    [[nodiscard]] Tensor compute_GAE(const Tensor &returns, const Tensor &values, const Tensor &dones, const Tensor& last_values, const Tensor& last_dones) const;

    static Tensor log_prob(const Tensor &action, const Tensor &mu, const Tensor &sigma);

    int PlayOne();
    void PrepareBatch();

private:
    Tensor _obs;
};

#endif //C_ML_AGENT_H
