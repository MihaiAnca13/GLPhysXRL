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
    bound_loss_coef = config.bound_loss_coef;
    gamma = config.gamma;
    tau = config.tau;
    reward_multiplier = config.reward_multiplier;
    env = environment;
    num_envs = env->num_envs;

    obs_size = env->observation_size;
    action_size = env->action_size;

    net->to(device);
    net->train();
    value_mean_std->to(device);
    value_mean_std->train();
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

            {
                // no grads
                torch::NoGradGuard no_grad;
                // normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
            }

            // Compute the surrogate loss and the value loss
            auto net_output = net->forward(obs);
            auto new_log_prob = log_prob(action, net_output.mu, torch::ones_like(net_output.mu));
            auto ratio = (old_log_prob - new_log_prob).exp();
            auto surr1 = ratio * advantages;
            auto surr2 = torch::clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages;
            auto actor_loss = -torch::min(surr1, surr2);

            auto value_loss = torch::mse_loss(net_output.value, returns);

            // calculate mean of surrogate_loss
            actor_loss = actor_loss.mean();

            // calculate bound loss
            auto mu_loss_high = torch::pow(torch::clamp_min(net_output.mu - 1.1, 0.0), 2);
            auto mu_loss_low = torch::pow(torch::clamp_max(net_output.mu + 1.1, 0.0), 2);
            auto b_loss = (mu_loss_low + mu_loss_high).sum(-1).mean();

            // log the loss
            logger.add_scalar("Loss/actor_loss", epoch * num_steps / mini_batch_size + i, actor_loss.item<float>());
            logger.add_scalar("Loss/critic_loss", epoch * num_steps / mini_batch_size + i, value_loss.item<float>());
            logger.add_scalar("Loss/bound_loss", epoch * num_steps / mini_batch_size + i, b_loss.item<float>());

            auto loss = actor_loss + value_loss_coef * value_loss + b_loss * bound_loss_coef;

            // Update the network
            optimizer.zero_grad();
            loss.backward();

            // truncate gradients and step
            torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
            optimizer.step();

            // Print the loss
            std::cout << "Epoch: " << epoch << " " << i + 1 << "/" << (int) (num_steps / mini_batch_size) << " Actor Loss: " << actor_loss.item<float>() << " Critic Loss: " << value_loss.item<float>() << std::endl;
            logger.add_scalar("Info/epoch", epoch * num_steps / mini_batch_size + i, (float) epoch);

            if (i == 0 && epoch == 0) {
                last_reward = env->last_reward_mean;
            } else if (env->last_reward_mean > last_reward) {
                cout << "Saving model with new best reward " << env->last_reward_mean << endl;
                last_reward = env->last_reward_mean;
                torch::save(net, "model.pt");
            }

            if (epoch % 100 == 0 && i == 0) {
                cout << "Saving model with current reward: " << env->last_reward_mean << endl;
                torch::save(net, "auto_model.pt");
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
        auto envStep = env->Step(action, &logger);
        Tensor reward = envStep.reward * reward_multiplier;
        Tensor next_obs = envStep.observation;

        Transition transition = {_obs,
                                 action,
                                 reward,
                                 next_obs,
                                 envStep.done,
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
    Tensor reward = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor values = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor dones = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor advantages = torch::zeros({num_envs, horizon_length}, floatOptions);

    // save reward, values and dones for each step
    for (int step = memory.size() - 1; step >= 0; step--) {
        reward.narrow(1, step, 1) = memory[step].reward.unsqueeze(1);
        values.narrow(1, step, 1) = memory[step].value;
        dones.narrow(1, step, 1) = memory[step].done.unsqueeze(1);
    }

    auto last_values = get_value(memory[memory.size() - 1].next_obs);
    auto last_dones = memory[memory.size() - 1].done.unsqueeze(1);
    advantages = compute_GAE(reward, values, dones, last_values, last_dones);

    auto returns = advantages + values;

    // flatten returns, pass through value_mean_std and then reshape back
    returns = returns.view({num_envs * horizon_length, 1});
    returns = value_mean_std->forward(returns);
    returns = returns.view({num_envs, horizon_length});

    // update memory with advantages, returns
    for (int step = 0; step < memory.size(); step++) {
        memory[step].advantages = advantages.narrow(1, step, 1);
        memory[step].returns = returns.narrow(1, step, 1);
    }
}


// GAE
Tensor Agent::compute_GAE(const Tensor &rewards, const Tensor &values, const Tensor &dones, const Tensor &last_values, const Tensor& last_dones) const {
    Tensor advantages = torch::zeros({num_envs, horizon_length}, floatOptions);
    Tensor delta = torch::zeros({num_envs}, floatOptions);
    Tensor last_gae = torch::zeros({num_envs, 1}, floatOptions);
    Tensor nextvalues = torch::zeros({num_envs}, floatOptions);
    Tensor nextnonterminal = torch::zeros({num_envs}, floatOptions);

    for (int step = horizon_length - 1; step >= 0; step--) {
        if (step == horizon_length - 1) {
            nextvalues = last_values;
            nextnonterminal = last_dones;
        } else {
            nextvalues = values.narrow(1, step + 1, 1);
            nextnonterminal = 1 - dones.narrow(1, step + 1, 1);
        }

        delta = rewards.narrow(1, step, 1) + gamma * nextvalues * nextnonterminal - values.narrow(1, step, 1);
        advantages.narrow(1, step, 1) = delta + gamma * tau * nextnonterminal * last_gae;
        last_gae = advantages.narrow(1, step, 1);
    }

    return advantages;
}


Tensor Agent::get_value(Tensor observation) {
    return value_mean_std->forward(net->forward(std::move(observation)).value, true);
}


// Compute the log probability of an action given the mean and standard deviation
Tensor Agent::log_prob(const Tensor &action, const Tensor &mu, const Tensor &sigma) {
    auto log_prob = -0.5 * (action - mu).pow(2) / sigma.pow(2) - 0.5 * log(2 * M_PI) - torch::log(sigma);
    return log_prob.sum(1, true);
}
