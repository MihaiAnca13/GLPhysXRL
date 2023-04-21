//
// Created by mihai on 05/04/23.
//

#ifndef C_ML_AGENT_H
#define C_ML_AGENT_H

#include <vector>
#include "LinearNetwork.h"
#include "Environment.h"


typedef struct {
    Tensor obs;
    Tensor action;
    Tensor reward;
    Tensor next_obs;
    Tensor done;
    Tensor value;
} Transition;


class Agent {
public:
    int num_epochs = 10;
    int horizon_length = 16;
    int mini_batch_size = 64;
    float clip_param = 0.2;
    float value_loss_coef = 0.5;
    float entropy_coef = 0.01;
    float gamma = 0.9;
    float tau = 0.95;
    float reward_multiplier = 0.01;
    std::vector<Transition> memory;

    TensorOptions floatOptions = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).layout(torch::kStrided).requires_grad(false);
    TensorOptions boolOptions = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).layout(torch::kStrided).requires_grad(false);

    Agent();

    Network net;

    void Train(Environment *env);

    Tensor get_value(Tensor observation);

    // GAE function for calculating the advantage
    [[nodiscard]] Tensor get_advantage(const Tensor& returns, const Tensor& values, const Tensor& dones) const;

//    Action act(Observation observation);
//
//    void save_model();
//
//    void load_model();

};

#endif //C_ML_AGENT_H
