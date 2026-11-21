#include "optimizer.h"

#include <stdexcept>

namespace fv {

static void check(const Tensor& p, const Tensor& g) {
    if (p.size() != g.size()) throw std::invalid_argument("optimizer: params and grads differ in size");
}

void Sgd::step(Tensor& p, const Tensor& g) {
    check(p, g);
    for (size_t i = 0; i < p.size(); ++i) p[i] -= lr_ * g[i];
}

void Momentum::step(Tensor& p, const Tensor& g) {
    check(p, g);
    if (v_.size() != p.size()) v_ = Tensor(p.shape());
    for (size_t i = 0; i < p.size(); ++i) {
        v_[i] = mu_ * v_[i] - lr_ * g[i];
        p[i] += v_[i];
    }
}

void Nesterov::step(Tensor& p, const Tensor& g) {
    check(p, g);
    if (v_.size() != p.size()) v_ = Tensor(p.shape());
    for (size_t i = 0; i < p.size(); ++i) {
        double v_old = v_[i];
        v_[i] = mu_ * v_old - lr_ * g[i];
        p[i] += -mu_ * v_old + (1.0 + mu_) * v_[i];
    }
}

std::unique_ptr<Optimizer> make_optimizer(const std::string& name, double lr, double mu) {
    if (name == "sgd") return std::make_unique<Sgd>(lr);
    if (name == "momentum") return std::make_unique<Momentum>(lr, mu);
    if (name == "nesterov") return std::make_unique<Nesterov>(lr, mu);
    throw std::invalid_argument("unknown optimizer: " + name);
}

}  // namespace fv
