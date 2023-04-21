//
// Created by mihai on 05/04/23.
//
#include "LinearNetwork.h"


NetworkImpl::NetworkImpl(int64_t in, int64_t out) {
    fc1 = register_module("fc1", nn::Linear(in, 128));
    fc2 = register_module("fc2", nn::Linear(128, 128));
    fc3 = register_module("fc3", nn::Linear(128, 128));
    mu = register_module("mu", nn::Linear(128, out));
    sigma = register_module("sigma", nn::Linear(128, 1));
    value = register_module("value", nn::Linear(128, 1));
}


NetworkOutput NetworkImpl::forward(Tensor x) {
    x = elu(fc1->forward(x));
    x = elu(fc2->forward(x));
    x = elu(fc3->forward(x));
    return {mu->forward(x), sigma->forward(x), value->forward(x)};
}