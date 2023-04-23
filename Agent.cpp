//
// Created by mihai on 05/04/23.
//
#include "Agent.h"
#include <utility>


Agent::Agent() : net(Network(7, 2)), optimizer(net->parameters(), torch::optim::AdamWOptions(learning_rate)) {
    net->to(device);
    net->train();
    memory.reserve(horizon_length);
}


void Agent::Train(Environment *env) {
    Tensor obs = env->Reset().toTensor();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        memory.clear();
        int num_steps = 0;
        // Collect data from the environment
        for (int step = 0; step < horizon_length; ++step) {
            // Disable gradient calculations
            torch::NoGradGuard no_grad;

            auto net_output = net->forward(obs);
            // convert from mu and sigma to action
            Tensor action = at::normal(net_output.mu, torch::ones_like(net_output.mu));

            Tensor old_log_prob = log_prob(action, net_output.mu, torch::ones_like(net_output.mu));

            // clamping takes place in the environment
            auto envStep = env->Step(action);
            Tensor reward = torch::tensor(envStep.reward * reward_multiplier, floatOptions);
            Tensor next_obs = envStep.observation.toTensor();
            Tensor done = torch::tensor(envStep.done, floatOptions);

            Transition transition = {obs,
                                     action,
                                     reward,
                                     next_obs,
                                     done,
                                     get_value(next_obs),
                                     old_log_prob,
                                     torch::zeros({1}, floatOptions),
                                     torch::zeros({1}, floatOptions)};
            memory.push_back(transition);
            obs = next_obs;
            if (envStep.done) {
                obs = env->Reset().toTensor();
            }
            num_steps++;
        }

        {
            // Disable gradient calculations
            torch::NoGradGuard no_grad;
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
            // update memory with advantages, returns
            for (int i = 0; i < memory.size(); ++i) {
                memory[i].returns = returns[i];
                memory[i].advantages = advantages[i];
            }
        }

        // Update the agent using PPO
        for (int i = 0; i < num_steps / mini_batch_size; ++i) {
            // Sample a mini-batch of transitions
            std::vector<Transition> mini_batch(mini_batch_size);
            for (int j = 0; j < mini_batch_size; j++) {
                int idx = rand() % memory.size();
                mini_batch[j] = memory[idx];
            }

            // Compute the loss
            torch::Tensor surrogate_loss = torch::zeros({1}, lossOptions);
            torch::Tensor value_loss = torch::zeros({1}, lossOptions);

            for (int j = 0; j < mini_batch_size; ++j) {
                auto net_output = net->forward(mini_batch[j].obs);
                auto new_log_prob = log_prob(mini_batch[j].action, net_output.mu, torch::ones_like(net_output.mu));
                auto ratio = torch::exp(new_log_prob - mini_batch[j].old_log_prob);
                auto clipped_ratio = torch::clamp(ratio, 1 - epsilon, 1 + epsilon);
                auto min_ratio_adv = torch::min(ratio * mini_batch[j].advantages, clipped_ratio * mini_batch[j].advantages);
                surrogate_loss = surrogate_loss + min_ratio_adv;
                value_loss = value_loss + torch::mse_loss(net_output.value, mini_batch[j].returns);
            }
            surrogate_loss /= mini_batch_size;
            value_loss /= mini_batch_size;

            auto loss = -surrogate_loss + value_loss;

            // Update the network
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // Print the loss
            std::cout << "Epoch: " << epoch << " " << i + 1 << "/" << (int)(num_steps / mini_batch_size) << " Loss: " << loss.item<float>() << std::endl;

            if (i == 0) {
                last_loss = loss.item<float>();
            }
            else if (loss.item<float>() < last_loss) {
                cout << "Saving model new best with loss " << loss.item<float>() << endl;
                last_loss = loss.item<float>();
                torch::save(net, "model.pt");
            }
        }
    }
}


// GAE
Tensor Agent::get_advantage(const Tensor &returns, const Tensor &values, const Tensor &dones) const {
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


// Compute the log probability of an action given the mean and standard deviation
Tensor Agent::log_prob(const Tensor &action, const Tensor &mu, const Tensor &sigma) {
    auto log_prob = -0.5 * torch::pow((action - mu) / sigma, 2) - torch::log(sigma) - 0.5 * log(2 * M_PI);
    return torch::sum(log_prob, -1);
}