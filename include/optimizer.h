#pragma once
#include <memory>
#include <string>
#include "tensor.h"

namespace fv {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Tensor& params, const Tensor& grads) = 0;
    virtual std::string name() const = 0;
};

// p <- p - lr*g (batch GD with full batch, SGD with mini-batches)
class Sgd : public Optimizer {
public:
    explicit Sgd(double lr) : lr_(lr) {}
    void step(Tensor& params, const Tensor& grads) override;
    std::string name() const override { return "sgd"; }
private:
    double lr_;
};

// v <- mu*v - lr*g;  p <- p + v
class Momentum : public Optimizer {
public:
    Momentum(double lr, double mu) : lr_(lr), mu_(mu) {}
    void step(Tensor& params, const Tensor& grads) override;
    std::string name() const override { return "momentum"; }
private:
    double lr_, mu_;
    Tensor v_;
};

// v_new = mu*v - lr*g;  p <- p - mu*v_old + (1+mu)*v_new
class Nesterov : public Optimizer {
public:
    Nesterov(double lr, double mu) : lr_(lr), mu_(mu) {}
    void step(Tensor& params, const Tensor& grads) override;
    std::string name() const override { return "nesterov"; }
private:
    double lr_, mu_;
    Tensor v_;
};

std::unique_ptr<Optimizer> make_optimizer(const std::string& name, double lr, double mu = 0.9);

}  // namespace fv
