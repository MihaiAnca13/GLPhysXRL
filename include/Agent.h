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


class Agent {
public:
    int num_epochs = 10;
    int horizon_length = 16;
    int mini_batch_size = 2;
    float learning_rate = 1e-3;
    float clip_param = 0.2;
    float value_loss_coef = 0.5;
    float entropy_coef = 0.01;
    float gamma = 0.9;
    float tau = 0.95;
    float reward_multiplier = 0.01;
    float epsilon = 1e-5;
    float last_loss = 0.0f;

    std::vector<Transition> memory;
    DeviceType device = torch::kCPU;

    TensorOptions lossOptions = torch::TensorOptions().dtype(torch::kFloat32).device(device).layout(torch::kStrided).requires_grad(true);
    TensorOptions floatOptions = torch::TensorOptions().dtype(torch::kFloat32).device(device).layout(torch::kStrided).requires_grad(false);
    TensorOptions boolOptions = torch::TensorOptions().dtype(torch::kBool).device(device).layout(torch::kStrided).requires_grad(false);

    Agent();

    Network net;
    optim::AdamW optimizer;

    void Train(Environment *env);

    Tensor get_value(Tensor observation);

    // GAE function for calculating the advantage
    [[nodiscard]] Tensor get_advantage(const Tensor &returns, const Tensor &values, const Tensor &dones) const;

    static Tensor log_prob(const Tensor &action, const Tensor &mu, const Tensor &sigma);

//    Action act(Observation observation);
//
//    void save_model();
//
//    void load_model();

};

#endif //C_ML_AGENT_H
