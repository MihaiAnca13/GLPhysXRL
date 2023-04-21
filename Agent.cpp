//
// Created by mihai on 05/04/23.
//
#include "Agent.h"
#include <utility>


Agent::Agent() : net(Network(7, 2)) {
//    net->to(torch::kCUDA);
    net->train();
    memory.reserve(horizon_length);
}


void Agent::Train(Environment *env) {
    Tensor obs = env->Reset().toTensor();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        memory.clear();
        // Collect data from the environment
        for (int step = 0; step < horizon_length; ++step) {
            auto net_output = net->forward(obs);
            // convert from mu and sigma to action
            Tensor action = at::normal(net_output.mu, torch::ones_like(net_output.mu));

            // clamping takes place in the environment
            auto envStep = env->Step(action);
            Tensor reward = torch::tensor(envStep.reward * reward_multiplier, floatOptions);
            Tensor next_obs = envStep.observation.toTensor();
            Tensor done = torch::tensor(envStep.done, floatOptions);

            Transition transition = {obs, action, reward, next_obs, done, get_value(next_obs)};
            memory.push_back(transition);
            obs = next_obs;
            if (envStep.done) {
                obs = env->Reset().toTensor();
            }
        }

        // Compute returns and advantages
        Tensor returns = torch::zeros({horizon_length}, floatOptions);
        Tensor values = torch::zeros({horizon_length}, floatOptions);
        Tensor dones = torch::zeros({horizon_length}, floatOptions);
        Tensor advantages = torch::zeros({horizon_length}, floatOptions);
        Tensor R = torch::zeros({1}, floatOptions);
        for (int i = memory.size() - 1; i >= 0; --i) {
            values[i] = memory[i].value.squeeze(-1);
            dones[i] = memory[i].done;
            R = memory[i].reward + gamma * R;
            returns[i] = R[0];
        }
        advantages = get_advantage(returns, values, dones);

//        // Update the agent using PPO
//        for (int i = 0; i < num_steps / mini_batch_size; ++i) {
//            // Sample a mini-batch of transitions
//            std::vector<Transition> mini_batch(mini_batch_size);
//            for (int j = 0; j < mini_batch_size; j++) {
//                int idx = rand() % memory.size();
//                mini_batch[j] = memory[idx];
//            }
//
//            // Compute the loss
//            torch::Tensor loss = torch::zeros({1});
//
//            torch::Tensor old_log_prob = torch::zeros({mini_batch_size});
//            torch::Tensor ratio = torch::zeros({mini_batch_size});
//            torch::Tensor surrogate_loss = torch::zeros({mini_batch_size});
//            torch::Tensor value_loss = torch::zeros({mini_batch_size});
//
//            for (int j = 0; j < mini_batch_size; j++) {
//                old_log_prob[j] = agent.get_log_prob(mini_batch[j].state, mini_batch[j].action);
//                ratio[j] = torch::exp(agent.get_log_prob(mini_batch[j].state, mini_batch[j].action) - old_log_prob[j]);
//                surrogate_loss[j] = torch::min(ratio * advantages[i], torch::clamp(ratio, {1 - clip_param}, {1 + clip_param}) * advantages[i]);
//                value_loss[j] = (agent.get_value(mini_batch[j].state) - returns[i]).pow(2);
//            }
//
//            loss = -surrogate_loss.mean() + value_loss_coef * value_loss.mean() - entropy_coef * agent.get_entropy(mini_batch[0].state);
//
//
//            // Optimize the agent
//            agent.optimize(loss);
//        }
    }
}


// GAE
Tensor Agent::get_advantage(const Tensor& returns, const Tensor& values, const Tensor& dones) const {
    Tensor advantages = torch::zeros({horizon_length}, floatOptions);
    Tensor delta = torch::zeros({horizon_length}, floatOptions);
    Tensor last_gae = torch::zeros({1}, floatOptions);

    for (int i = horizon_length - 1; i >= 0; --i) {
        delta[i] = returns[i] + gamma * values[i] * (1 - dones[i]) - values[i];
        advantages[i] = delta[i] + gamma * tau * (1 - dones[i]) * last_gae[0];
        last_gae[0] = advantages[i];
    }
    return advantages;
}


Tensor Agent::get_value(Tensor observation) {
    return net->forward(std::move(observation)).value;
}