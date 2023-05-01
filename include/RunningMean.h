//
// Created by mihai on 29/04/23.
//

#ifndef C_ML_RUNNINGMEAN_H
#define C_ML_RUNNINGMEAN_H

#include <torch/torch.h>


using namespace torch;


class RunningMeanStdImpl : public nn::Module {
public:
    RunningMeanStdImpl(IntArrayRef insize, double epsilon, bool per_channel, bool norm_only);

    Tensor forward(Tensor input, bool unnorm = false);

    Tensor running_mean;
    Tensor running_var;

private:
    IntArrayRef insize;
    double epsilon = 1e-5;
    bool per_channel = false;
    bool norm_only = false;
    std::vector<int64_t> axis;
    int in_size;
    Tensor count;

    void _update_mean_var_count_from_moments(const Tensor &batch_mean, const Tensor &batch_var, const Tensor &batch_count);
};

TORCH_MODULE(RunningMeanStd);


#endif //C_ML_RUNNINGMEAN_H
