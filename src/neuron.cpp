#include "neuron.h"

#include <cmath>
#include <stdexcept>
#include "linalg.h"

namespace fv {

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

McCullochPitts::McCullochPitts(std::vector<double> weights, double threshold)
    : w_(std::move(weights)), th_(threshold) {}

int McCullochPitts::fire(const std::vector<int>& x) const {
    if (x.size() != w_.size()) throw std::invalid_argument("MP: wrong number of inputs");
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) s += w_[i] * x[i];
    return s >= th_ ? 1 : 0;
}

McCullochPitts McCullochPitts::AND2()  { return McCullochPitts({1, 1}, 2); }
McCullochPitts McCullochPitts::OR2()   { return McCullochPitts({1, 1}, 1); }
McCullochPitts McCullochPitts::NAND2() { return McCullochPitts({-1, -1}, -1); }
McCullochPitts McCullochPitts::NOR2()  { return McCullochPitts({-1, -1}, 0); }
McCullochPitts McCullochPitts::NOT1()  { return McCullochPitts({-1}, 0); }

int mp_xor(int a, int b) {
    int h1 = McCullochPitts::OR2().fire({a, b});
    int h2 = McCullochPitts::NAND2().fire({a, b});
    return McCullochPitts::AND2().fire({h1, h2});
}

SigmoidNeuron::SigmoidNeuron(size_t n_inputs, uint64_t seed) : w_(n_inputs), b_(0.0) {
    Tensor r = random_uniform(std::vector<size_t>{n_inputs}, seed, -0.5, 0.5);
    for (size_t i = 0; i < n_inputs; ++i) w_[i] = r[i];
}

double SigmoidNeuron::forward(const std::vector<double>& x) const {
    if (x.size() != w_.size()) throw std::invalid_argument("SigmoidNeuron: wrong input size");
    double z = b_;
    for (size_t i = 0; i < x.size(); ++i) z += w_[i] * x[i];
    return sigmoid(z);
}

double SigmoidNeuron::train_epoch(const std::vector<std::vector<double>>& X,
                                  const std::vector<double>& t, double lr) {
    const double eps = 1e-12;
    std::vector<double> gw(w_.size(), 0.0);
    double gb = 0.0, loss = 0.0;
    for (size_t s = 0; s < X.size(); ++s) {
        double y = forward(X[s]);
        double err = y - t[s];
        for (size_t j = 0; j < w_.size(); ++j) gw[j] += err * X[s][j];
        gb += err;
        loss += -(t[s] * std::log(y + eps) + (1.0 - t[s]) * std::log(1.0 - y + eps));
    }
    const double n = static_cast<double>(X.size());
    for (size_t j = 0; j < w_.size(); ++j) w_[j] -= lr * gw[j] / n;
    b_ -= lr * gb / n;
    return loss / n;
}

}  // namespace fv
