//
// Created by mihai on 05/04/23.
//
#include "Agent.h"
#include <utility>


void sleep(int seconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(seconds * 1000));
}


Agent::Agent(AgentConfig config, Environment *environment) : net(Network(environment->observation_size, environment->action_size)), optimizer(net->parameters(), torch::optim::AdamWOptions(learning_rate)) {
    //loading the config
    num_epochs = config.num_epochs;
    horizon_length = config.horizon_length;
    mini_batch_size = config.mini_batch_size;
    learning_rate = config.learning_rate;
    clip_param = config.clip_param;
    value_loss_coef = config.value_loss_coef;
    gamma = config.gamma;
    tau = config.tau;
    reward_multiplier = config.reward_multiplier;
    env = environment;
    num_envs = env->num_envs;

    obs_size = env->observation_size;
    action_size = env->action_size;

    net->to(device);
    net->train();
    memory.reserve(horizon_length);
}


void Agent::Train() {
    _obs = env->Reset();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        int num_steps = PlayOne(env);
        ComputeAdvantage();

        // assert that num_envs * horizon_length is divisible by mini_batch_size
        assert(num_envs * horizon_length % mini_batch_size == 0);

        // Update the agent using PPO
        for (int i = 0; i < num_steps / mini_batch_size; ++i) {
            // Sample a mini-batch of transitions and convert the required samples to tensors
            Tensor obs = torch::zeros({mini_batch_size, obs_size}, floatOptions);
            Tensor action = torch::zeros({mini_batch_size, action_size}, floatOptions);
            Tensor old_log_prob = torch::zeros({mini_batch_size, 1}, floatOptions);
            Tensor advantages = torch::zeros({mini_batch_size, 1}, floatOptions);
            Tensor returns = torch::zeros({mini_batch_size, 1}, floatOptions);

            for (int j = 0; j < mini_batch_size; j++) {
                int idx1 = rand() % memory.size();
                int idx2 = rand() % num_envs;
                obs[j] = memory[idx1].obs[idx2];
                action[j] = memory[idx1].action[idx2];
                old_log_prob[j] = memory[idx1].old_log_prob[idx2];
                advantages[j] = memory[idx1].advantages[idx2];
                returns[j] = memory[idx1].returns[idx2];
            }

            // Compute the surrogate loss and the value loss
            auto net_output = net->forward(obs);
            auto new_log_prob = log_prob(action, net_output.mu, torch::ones_like(net_output.mu));
            auto ratio = (old_log_prob - new_log_prob).exp();
            auto surr1 = ratio * advantages;
            auto surr2 = torch::clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages;
            auto surrogate_loss = -torch::min(surr1, surr2);
            auto value_loss = torch::mse_loss(net_output.value, returns);

            // calculate mean of surrogate_loss
            surrogate_loss = surrogate_loss.mean();

            auto loss = surrogate_loss + value_loss_coef * value_loss;

            // Update the network
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // Print the loss
            std::cout << "Epoch: " << epoch << " " << i + 1 << "/" << (int) (num_steps / mini_batch_size) << " Loss: " << loss.item<float>() << std::endl;

            if (i == 0 && epoch == 0) {
                last_loss = loss.item<float>();
            } else if (loss.item<float>() < last_loss) {
                cout << "Saving model new best with loss " << loss.item<float>() << endl;
                last_loss = loss.item<float>();
                torch::save(net, "model.pt");
            }
        }
    }
}


int Agent::PlayOne(Environment *env) {
    memory.clear();
    int num_steps = 0;
    // Collect data from the environment
    for (int step = 0; step < horizon_length; ++step) {
        // Disable gradient calculations
        torch::NoGradGuard no_grad;

        auto net_output = net->forward(_obs);
        // convert from mu and sigma to action
        Tensor action = at::normal(net_output.mu, torch::ones_like(net_output.mu));
        // clamp action between -1 and 1
        action = torch::clamp(action, -1.0, 1.0);

        Tensor old_log_prob = log_prob(action, net_output.mu, torch::ones_like(net_output.mu));

        // clamping takes place in the environment
        auto envStep = env->Step(action);
        Tensor reward = envStep.reward * reward_multiplier;
        Tensor next_obs = envStep.observation;

        Transition transition = {_obs,
                                 action,
                                 reward.to(device),
                                 next_obs,
                                 envStep.done.to(device),
                                 get_value(next_obs),
                                 old_log_prob,
                                 torch::zeros({1}, floatOptions),
                                 torch::zeros({1}, floatOptions)};
        memory.push_back(transition);
        _obs = next_obs;
        if (envStep.done[0].item<float>() == 1.0f) {
            _obs = env->Reset();
        }
        num_steps += num_envs;
    }
    return num_steps;
}


void Agent::ComputeAdvantage() {
    // Disable gradient calculations
    torch::NoGradGuard no_grad;
    // Compute returns and advantages
    Tensor returns = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor values = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor dones = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor advantages = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor R = torch::zeros({num_envs}, floatOptions);

    // compute returns and save values and dones for each step
    for (int step = memory.size() - 1; step >= 0; step--) {
        R = memory[step].reward + gamma * R;
        returns.narrow(1, step, 1) = R.unsqueeze(1);

        values.narrow(1, step, 1) = memory[step].value;
        dones.narrow(1, step, 1) = memory[step].done.unsqueeze(1);
    }

    auto last_values = get_value(memory[memory.size() - 1].next_obs);
    advantages = compute_GAE(returns, values, dones, last_values);

    // update memory with advantages, returns
    for (int step = 0; step < memory.size(); step++) {
        memory[step].advantages = advantages.narrow(1, step, 1);
        memory[step].returns = returns.narrow(1, step, 1);
    }
}


// GAE
Tensor Agent::compute_GAE(const Tensor &returns, const Tensor &values, const Tensor &dones, const Tensor &last_values) const {
    Tensor advantages = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor delta = torch::zeros({num_envs}, floatOptions);
    Tensor last_gae = torch::zeros({num_envs, 1}, floatOptions);
    Tensor nextvalues = torch::zeros({num_envs}, floatOptions);

    for (int step = horizon_length - 1; step >= 0; step--) {
        if (step == horizon_length - 1) {
            nextvalues = last_values;
        } else {
            nextvalues = values.narrow(1, step + 1, 1);
        }

        delta = returns.narrow(1, step, 1) + gamma * nextvalues * (1 - dones.narrow(1, step, 1)) - values.narrow(1, step, 1);
        advantages.narrow(1, step, 1) = delta + gamma * tau * (1 - dones.narrow(1, step, 1)) * last_gae;
        last_gae = advantages.narrow(1, step, 1);
    }

    return advantages;
}


Tensor Agent::get_value(Tensor observation) {
    return net->forward(std::move(observation)).value;
}


// Compute the log probability of an action given the mean and standard deviation
Tensor Agent::log_prob(const Tensor &action, const Tensor &mu, const Tensor &sigma) {
    auto log_prob = -0.5 * (action - mu).pow(2) / sigma.pow(2) - 0.5 * log(2 * M_PI) - torch::log(sigma);
    return log_prob.sum(1, true);
}
