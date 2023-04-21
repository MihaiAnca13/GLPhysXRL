#include <vector>
#include <tuple>
#include <torch/torch.h>

struct Transition {
    // Transition variables
    State state;
    Action action;
    float reward;
};


// GAE function for calculating the advantage
std::vector<float> get_advantage(std::vector<float> rewards, std::vector<float> values, float gamma, float lambda) {
    std::vector<float> advantages(rewards.size());
    float prev_advantage = 0;
    for (int i = rewards.size() - 1; i >= 0; --i) {
        float delta = rewards[i] + gamma * values[i + 1] - values[i];
        advantages[i] = delta + gamma * lambda * prev_advantage;
        prev_advantage = advantages[i];
    }
    return advantages;
}


void TrainAgent() {
    // Define the PPO hyperparameters
    int num_epochs = 10;
    int num_steps = 2048;
    int mini_batch_size = 64;
    float clip_param = 0.2;
    float value_loss_coef = 0.5;
    float entropy_coef = 0.01;
    float trainGamma = 0.9;

    // Initialize the environment and agent
    Environment env;
    Agent agent;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Collect data from the environment
        std::vector<Transition> memory;
        for (int step = 0; step < num_steps; ++step) {
            Action action = agent.act(env.get_state());
            State next_state;
            float reward;
            bool done;
            std::tie(next_state, reward, done) = env.step(action);
            memory.push_back({env.get_state(), action, reward});
            if (done) {
                env.reset();
            }
        }

        // Compute returns and advantages
        std::vector<float> returns(memory.size());
        std::vector<float> advantages(memory.size());
        float R = 0.0f;
        for (int i = memory.size() - 1; i >= 0; --i) {
            R = memory[i].reward + trainGamma * R;
            returns[i] = R;
        }
        advantages = get_advantage(returns, agent.get_values(memory), trainGamma, 0.95);

        // Update the agent using PPO
        for (int i = 0; i < num_steps / mini_batch_size; ++i) {
            // Sample a mini-batch of transitions
            std::vector<Transition> mini_batch(mini_batch_size);
            for (int j = 0; j < mini_batch_size; j++) {
                int idx = rand() % memory.size();
                mini_batch[j] = memory[idx];
            }

            // Compute the loss
            torch::Tensor loss = torch::zeros({1});

            torch::Tensor old_log_prob = torch::zeros({mini_batch_size});
            torch::Tensor ratio = torch::zeros({mini_batch_size});
            torch::Tensor surrogate_loss = torch::zeros({mini_batch_size});
            torch::Tensor value_loss = torch::zeros({mini_batch_size});

            for (int j = 0; j < mini_batch_size; j++) {
                old_log_prob[j] = agent.get_log_prob(mini_batch[j].state, mini_batch[j].action);
                ratio[j] = torch::exp(agent.get_log_prob(mini_batch[j].state, mini_batch[j].action) - old_log_prob[j]);
                surrogate_loss[j] = torch::min(ratio * advantages[i], torch::clamp(ratio, {1 - clip_param}, {1 + clip_param}) * advantages[i]);
                value_loss[j] = (agent.get_value(mini_batch[j].state) - returns[i]).pow(2);
            }

            loss = -surrogate_loss.mean() + value_loss_coef * value_loss.mean() - entropy_coef * agent.get_entropy(mini_batch[0].state);


            // Optimize the agent
            agent.optimize(loss);
        }
    }
}