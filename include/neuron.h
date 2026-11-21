#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fv {

double sigmoid(double z);

// McCulloch-Pitts: binary inputs, fixed weights, fires (1) if sum(w*x) >= threshold.
class McCullochPitts {
public:
    McCullochPitts(std::vector<double> weights, double threshold);
    int fire(const std::vector<int>& inputs) const;

    static McCullochPitts AND2();
    static McCullochPitts OR2();
    static McCullochPitts NAND2();
    static McCullochPitts NOR2();
    static McCullochPitts NOT1();

private:
    std::vector<double> w_;
    double th_;
};

// XOR(a,b) = AND( OR(a,b), NAND(a,b) )  (needs two layers)
int mp_xor(int a, int b);

// Sigmoid neuron: y = sigmoid(w.x + b), trained by gradient descent on log loss.
class SigmoidNeuron {
public:
    explicit SigmoidNeuron(size_t n_inputs, uint64_t seed = 1);
    double forward(const std::vector<double>& x) const;
    double train_epoch(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& targets, double lr);
    const std::vector<double>& weights() const { return w_; }
    double bias() const { return b_; }

private:
    std::vector<double> w_;
    double b_;
};

}  // namespace fv
