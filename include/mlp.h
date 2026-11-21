#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "tensor.h"

namespace fv {

enum class Act { Linear, Sigmoid, Tanh, ReLU };

enum class Init {
    Xavier,
    He,
    Random
};

std::string act_name(Act a);

// Fully connected feed-forward network. Input X has shape (N, features).
class Mlp {
public:
    // Mlp(std::vector<size_t> sizes, Act hidden, Act output, uint64_t seed = 1);
Mlp(std::vector<size_t> sizes, Act hidden, Act output,
    uint64_t seed = 1, Init init = Init::Xavier);
    Tensor forward(const Tensor& X);
    void backward(const Tensor& dOut);
    void sgd_step(double lr);

    size_t num_params() const;
    Tensor get_params() const;
    void set_params(const Tensor& flat);
    Tensor get_grads() const;

    const std::vector<size_t>& sizes() const { return sizes_; }
    Act hidden_act() const { return hidden_; }
    Act output_act() const { return output_; }

private:
    struct Layer {
        Tensor W, b, dW, db;
        Tensor in, z, a;
        Act act = Act::Linear;
    };
    std::vector<Layer> layers_;
    std::vector<size_t> sizes_;
    Act hidden_, output_;
};

double mse_loss(const Tensor& P, const Tensor& Y);
Tensor mse_grad(const Tensor& P, const Tensor& Y);

Tensor softmax(const Tensor& logits);
double softmax_ce_loss(const Tensor& logits, const Tensor& Y);
Tensor softmax_ce_grad(const Tensor& logits, const Tensor& Y);

Tensor one_hot(const std::vector<int>& labels, size_t classes);
double accuracy(const Tensor& P, const Tensor& Y);

}  // namespace fv
