//
// Created by mihai on 29/04/23.
//

#include "RunningMean.h"


RunningMeanStdImpl::RunningMeanStdImpl(IntArrayRef insize, double epsilon, bool per_channel, bool norm_only)
        : insize(insize), epsilon(epsilon), norm_only(norm_only), per_channel(per_channel) {
    if (per_channel) {
        if (insize.size() == 3) {
            axis = {0, 2, 3};
        }
        if (insize.size() == 2) {
            axis = {0, 2};
        }
        if (insize.size() == 1) {
            axis = {0};
        }
        in_size = insize[0];
    } else {
        axis = {0};
        in_size = insize.size();
    }

    running_mean = register_buffer("running_mean", torch::zeros({in_size}, kFloat64));
    running_var = register_buffer("running_var", torch::ones({in_size}, kFloat64));
    count = register_buffer("count", torch::ones({}, kFloat64));
}

MeanVarCount RunningMeanStdImpl::_update_mean_var_count_from_moments(const Tensor& mean, const Tensor& var, const Tensor& batch_mean, const Tensor& batch_var,
                                                                      const Tensor& batch_count) {
    Tensor delta = batch_mean - mean;
    Tensor tot_count = count + batch_count;

    Tensor new_mean = mean + delta * batch_count / tot_count;

    Tensor m_a = var * count;
    Tensor m_b = batch_var * batch_count;
    Tensor M2 = m_a + m_b + delta.pow(2) * count * batch_count / tot_count;

    Tensor new_var = M2 / tot_count;

    return {new_mean, new_var, tot_count};
}

Tensor RunningMeanStdImpl::forward(Tensor input, bool unnorm) {
    if (is_training()) {
        auto mean = input.mean(axis);
        auto var = input.var(axis);

        auto update = _update_mean_var_count_from_moments(running_mean, running_var, mean, var, torch::tensor(input.size(0)));
        running_mean = update.mean;
        running_var = update.var;
        count = update.count;
    }

    Tensor current_mean, current_var;

    if (per_channel) {
        if (insize.size() == 3) {
            current_mean = running_mean.view({1, insize[0], 1, 1}).expand_as(input);
            current_var = running_var.view({1, insize[0], 1, 1}).expand_as(input);
        }
        if (insize.size() == 2) {
            current_mean = running_mean.view({1, insize[0], 1}).expand_as(input);
            current_var = running_var.view({1, insize[0], 1}).expand_as(input);
        }
        if (insize.size() == 1) {
            current_mean = running_mean.view({1, insize[0]}).expand_as(input);
            current_var = running_var.view({1, insize[0]}).expand_as(input);
        }
    } else {
        current_mean = running_mean;
        current_var = running_var;
    }

    Tensor y;
    if (unnorm) {
        y = input.clamp(-5.0, 5.0);
        y = sqrt(current_var.to(kFloat32) + epsilon) * y.to(kFloat32) + current_mean.to(kFloat32);
    } else {
        if (norm_only) {
            y = input / sqrt(current_var.to(kFloat32) + epsilon);
        } else {
            y = (input - current_mean.to(kFloat32)) / sqrt(current_var.to(kFloat32) + epsilon);
            y = y.clamp(-5.0, 5.0);
        }
    }
    return y;
}