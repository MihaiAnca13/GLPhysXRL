//
// Created by mihai on 05/04/23.
//
#include "Agent.h"
#include <utility>


void sleep(int seconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(seconds * 1000));
}


Agent::Agent(AgentConfig config, Environment *environment) : net(Network(environment->observation_size, environment->action_size)),
                                                             optimizer(net->parameters(), torch::optim::AdamWOptions(learning_rate)) {
    //loading the config
    num_epochs = config.num_epochs;
    horizon_length = config.horizon_length;
    mini_batch_size = config.mini_batch_size;
    mini_epochs = config.mini_epochs;
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
    value_mean_std->to(device);

    InitMemory();
}


void Agent::InitMemory() {
    memory.obs = torch::zeros({horizon_length, num_envs, obs_size}, floatOptions);
    memory.action = torch::zeros({horizon_length, num_envs, action_size}, floatOptions);
    memory.reward = torch::zeros({horizon_length, num_envs, 1}, floatOptions);
    memory.next_obs = torch::zeros({horizon_length, num_envs, obs_size}, floatOptions);
    memory.done = torch::zeros({horizon_length, num_envs, 1}, floatOptions);
    memory.value = torch::zeros({horizon_length, num_envs, 1}, floatOptions);
    memory.old_log_prob = torch::zeros({horizon_length, num_envs, 1}, floatOptions);
    memory.returns = torch::zeros({horizon_length, num_envs, 1}, floatOptions);
    memory.advantages = torch::zeros({horizon_length, num_envs, 1}, floatOptions);
    memory.mu = torch::zeros({horizon_length, num_envs, action_size}, floatOptions);
}


void Agent::SetTrain() {
    net->train();
    value_mean_std->train();
}

void Agent::SetEval() {
    net->eval();
    value_mean_std->eval();
}


void Agent::Train() {
    _obs = env->Reset();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        TimeMe t("Epoch");
        InitMemory();
        SetEval();
        int num_steps = PlayOne();
        PrepareBatch();

        // assert that num_envs * horizon_length is divisible by mini_batch_size
        assert(num_envs * horizon_length % mini_batch_size == 0);

        float last_critic_loss, last_actor_loss;

        SetTrain();
        for (int mini_e = 0; mini_e < mini_epochs; mini_e++) {
            Tensor batch_idx = torch::randperm(horizon_length * num_envs, longOptions);
            long last_idx = 0;

            // Update the agent using PPO
            for (int i = 0; i < num_steps / mini_batch_size; i++) {
                // Sample a mini-batch of transitions and convert the required samples to tensors
                Tensor obs = memory.obs.index({batch_idx.slice(0, last_idx, last_idx + mini_batch_size)});
                Tensor action = memory.action.index({batch_idx.slice(0, last_idx, last_idx + mini_batch_size)});
                Tensor old_log_prob = memory.old_log_prob.index({batch_idx.slice(0, last_idx, last_idx + mini_batch_size)});
                Tensor advantages = memory.advantages.index({batch_idx.slice(0, last_idx, last_idx + mini_batch_size)});
                Tensor returns = memory.returns.index({batch_idx.slice(0, last_idx, last_idx + mini_batch_size)});
                Tensor old_mu = memory.mu.index({batch_idx.slice(0, last_idx, last_idx + mini_batch_size)});

                last_idx += mini_batch_size;

                // Compute the surrogate loss and the value loss
                auto net_output = net->forward(obs);

                auto new_log_prob = neg_log_prob(action, net_output.mu, torch::ones_like(net_output.mu));
                auto ratio = (old_log_prob - new_log_prob).exp();
                auto surr1 = ratio * advantages;
                auto surr2 = torch::clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages;
                auto actor_loss = -torch::min(surr1, surr2).mean();

                auto value_loss = torch::mse_loss(net_output.value, returns);

                // calculate bound loss
                auto mu_loss_high = torch::pow(torch::clamp_min(net_output.mu - 1.1, 0.0), 2);
                auto mu_loss_low = torch::pow(torch::clamp_max(net_output.mu + 1.1, 0.0), 2);
                auto b_loss = (mu_loss_low + mu_loss_high).sum(-1).mean();

                // log the loss
                logger.add_scalar("Loss/actor_loss", _steps, actor_loss.item<float>());
                logger.add_scalar("Loss/critic_loss", _steps, value_loss.item<float>());
                logger.add_scalar("Loss/bound_loss", _steps, b_loss.item<float>());
                _steps++;

                auto loss = actor_loss + value_loss_coef * value_loss + b_loss * bound_loss_coef;

                // Update the network
                optimizer.zero_grad();
                loss.backward();

                // truncate gradients and step
                torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
                optimizer.step();

                auto kl = policy_kl(net_output.mu, torch::ones_like(net_output.mu), old_mu, torch::ones_like(old_mu));
                double new_lr = update_lr(kl.item<double>());
                logger.add_scalar("Info/lr", _steps, new_lr);

                last_actor_loss = actor_loss.item<float>();
                last_critic_loss = value_loss.item<float>();

                if (i == 0 && epoch == 0 && mini_e == 0) {
                    last_reward = env->last_reward_mean;
                } else if (env->last_reward_mean > last_reward) {
                    cout << "Saving model with new best reward " << env->last_reward_mean << endl;
                    last_reward = env->last_reward_mean;
                    torch::save(net, "model.pt");
                }

                if (epoch % 100 == 0 && i == 0 && mini_e == 0) {
                    cout << "Saving model with current reward: " << env->last_reward_mean << endl;
                    torch::save(net, "auto_model.pt");
                }
            }
        }

        // Print the loss
        std::cout << "Epoch: " << epoch << " Actor Loss: " << last_actor_loss << " Critic Loss: " << last_critic_loss << std::endl;
        logger.add_scalar("Info/epoch", _steps, (float) epoch);
    }
}


int Agent::PlayOne() {
    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    int num_steps = 0;
    // Collect data from the environment
    for (int step = 0; step < horizon_length; step++) {
        auto net_output = net->forward(_obs);
        // convert from mu and sigma to action
        Tensor action = at::normal(net_output.mu, torch::ones_like(net_output.mu));
        // clamp action between -1 and 1
        action = torch::clamp(action, -1.0, 1.0);

        Tensor old_log_prob = neg_log_prob(action, net_output.mu, torch::ones_like(net_output.mu));

        // clamping takes place in the environment
        auto envStep = env->Step(action, &logger);
        Tensor reward = envStep.reward * reward_multiplier;
        Tensor next_obs = envStep.observation;

        memory.obs[step] = _obs;
        memory.action[step] = action;
        memory.reward[step] = reward.unsqueeze(-1);
        memory.next_obs[step] = next_obs;
        memory.done[step] = torch::zeros_like(envStep.done).unsqueeze(-1);  // done is always false
        memory.value[step] = value_mean_std->forward(net_output.value, true);
        memory.old_log_prob[step] = old_log_prob;
        memory.mu[step] = net_output.mu;

        _obs = next_obs;
        if (envStep.done[0].item<float>() == 1.0f) {
            _obs = env->Reset();
        }
        num_steps += num_envs;
    }
    return num_steps;
}


void Agent::PrepareBatch() {
    // Disable gradient calculations
    torch::NoGradGuard no_grad;

    auto last_values = get_value(memory.next_obs[horizon_length - 1]);
    auto last_dones = memory.done[horizon_length - 1];
    memory.advantages = compute_GAE(memory.reward, memory.value, memory.done, last_values, last_dones);

    auto returns = memory.advantages + memory.value;

    // flatten returns, pass through value_mean_std and then reshape back
    SetTrain();
    returns = returns.flatten(0, 1);
    returns = value_mean_std->forward(returns);
    SetEval();

    memory.returns = returns;
    memory.advantages = memory.advantages.flatten(0, 1);

    // normalize advantage
    memory.advantages = (memory.advantages - memory.advantages.mean()) / (memory.advantages.std() + 1e-8);

    // flatten everything in memory that's used for training
    memory.obs = memory.obs.flatten(0, 1);
    memory.action = memory.action.flatten(0, 1);
    memory.old_log_prob = memory.old_log_prob.flatten(0, 1);
    memory.mu = memory.mu.flatten(0, 1);
}


// GAE
Tensor Agent::compute_GAE(const Tensor &rewards, const Tensor &values, const Tensor &dones, const Tensor &last_values, const Tensor &last_dones) const {
    Tensor advantages = torch::zeros({horizon_length, num_envs, 1}, floatOptions);
    Tensor last_gae = torch::zeros({num_envs, 1}, floatOptions);
    Tensor nextvalues = torch::zeros({num_envs}, floatOptions);
    Tensor nextnonterminal = torch::zeros({num_envs}, floatOptions);

    for (int step = horizon_length - 1; step >= 0; step--) {
        if (step == horizon_length - 1) {
            nextvalues = last_values;
            nextnonterminal = last_dones;
        } else {
            nextvalues = values[step + 1];
            nextnonterminal = 1 - dones[step + 1];
        }

        auto delta = rewards[step] + gamma * nextvalues * nextnonterminal - values[step];
        last_gae = delta + gamma * tau * nextnonterminal * last_gae;
        advantages[step] = last_gae;
    }

    return advantages;
}


Tensor Agent::get_value(Tensor observation) {
    return value_mean_std->forward(net->forward(std::move(observation)).value, true);
}


// Compute the neg log probability of an action given the mean and standard deviation
Tensor Agent::neg_log_prob(const Tensor &action, const Tensor &mu, const Tensor &sigma) {
    return 0.5 * (((action - mu) / sigma).pow(2)).sum(1, true) + 0.5 * log(2 * M_PI) * action.size(1) + torch::log(sigma).sum(1, true);
}


Tensor Agent::policy_kl(const Tensor &mu, const Tensor &sigma, const Tensor &mu_old, const Tensor &sigma_old) {
    auto sigma_ratio = (sigma / sigma_old).log();
    auto mu_diff = (mu_old - mu).pow(2) / (2 * sigma_old.pow(2));
    auto kl = (sigma_ratio + mu_diff + 0.5 * log(2 * M_PI)).sum(1, true);
    return kl.mean();
}


double Agent::update_lr(const double& kl) {
    if (kl > (2.0f * kl_threshold)) {
         learning_rate = max(min_lr, learning_rate / learning_rate_decay);
    }
    else if (kl < (0.5f * kl_threshold)) {
        learning_rate = min(max_lr, learning_rate * learning_rate_decay);
    }

    // update lr in optimizer
    for (auto& param_group : optimizer.param_groups()) {
        param_group.options().set_lr(learning_rate);
    }

    return learning_rate;
}