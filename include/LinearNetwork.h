//
// Created by mihai on 05/04/23.
//

#ifndef C_ML_LINEARNETWORK_H
#define C_ML_LINEARNETWORK_H

#include <vector>
#include <torch/torch.h>

using namespace torch;


typedef struct {
    Tensor mu;
    Tensor value;
} NetworkOutput;


struct NetworkImpl : nn::Module {
    NetworkImpl(int64_t in, int64_t out);
    NetworkOutput forward(Tensor x);

    nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, mu{nullptr}, value{nullptr};
};
TORCH_MODULE(Network);

#endif //C_ML_LINEARNETWORK_H
