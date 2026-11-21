#include <cmath>
#include <iostream>
#include "gradcheck.h"
#include "linalg.h"
#include "mlp.h"
#include "neuron.h"

using namespace fv;

static int failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cout << "FAIL line " << __LINE__ << ": " #cond << "\n";   \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

static double grad_error(Act h, Act o, bool ce) {
    Mlp net({3, 5, 4, 2}, h, o, 3);
    Tensor X = random_uniform({6, 3}, 21);
    Tensor Y = ce ? one_hot({0, 1, 0, 1, 1, 0}, 2) : random_uniform({6, 2}, 22, 0.0, 1.0);
    auto loss = [&](const Tensor& p) {
        net.set_params(p);
        Tensor out = net.forward(X);
        return ce ? softmax_ce_loss(out, Y) : mse_loss(out, Y);
    };
    Tensor p0 = net.get_params();
    net.set_params(p0);
    Tensor out = net.forward(X);
    net.backward(ce ? softmax_ce_grad(out, Y) : mse_grad(out, Y));
    Tensor analytic = net.get_grads();
    return gradient_check(loss, p0, analytic);
}

int main() {
    for (int a = 0; a <= 1; ++a)
        for (int b = 0; b <= 1; ++b) {
            CHECK(McCullochPitts::AND2().fire({a, b}) == (a && b));
            CHECK(McCullochPitts::OR2().fire({a, b}) == (a || b));
            CHECK(McCullochPitts::NAND2().fire({a, b}) == !(a && b));
            CHECK(McCullochPitts::NOR2().fire({a, b}) == !(a || b));
            CHECK(mp_xor(a, b) == (a != b));
        }
    CHECK(McCullochPitts::NOT1().fire({0}) == 1 && McCullochPitts::NOT1().fire({1}) == 0);

    {
        std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        SigmoidNeuron n(2, 1);
        std::vector<double> t = {0, 0, 0, 1};
        double first = n.train_epoch(X, t, 0.5), last = first;
        for (int e = 0; e < 5000; ++e) last = n.train_epoch(X, t, 0.5);
        CHECK(last < first);
        for (size_t i = 0; i < 4; ++i) CHECK((n.forward(X[i]) >= 0.5) == (t[i] >= 0.5));

        SigmoidNeuron nx(2, 1);
        std::vector<double> tx = {0, 1, 1, 0};
        for (int e = 0; e < 5000; ++e) nx.train_epoch(X, tx, 0.5);
        int correct = 0;
        for (size_t i = 0; i < 4; ++i) correct += ((nx.forward(X[i]) >= 0.5) == (tx[i] >= 0.5));
        CHECK(correct < 4);
    }

    CHECK(grad_error(Act::Tanh, Act::Sigmoid, false) < 1e-6);
    CHECK(grad_error(Act::Sigmoid, Act::Linear, false) < 1e-6);
    CHECK(grad_error(Act::Tanh, Act::Linear, true) < 1e-6);
    CHECK(grad_error(Act::ReLU, Act::Linear, true) < 1e-6);

    {
        Mlp a({2, 4, 1}, Act::Tanh, Act::Sigmoid, 5);
        CHECK(a.num_params() == 2 * 4 + 4 + 4 * 1 + 1);
        Mlp b({2, 4, 1}, Act::Tanh, Act::Sigmoid, 5);
        Mlp c({2, 4, 1}, Act::Tanh, Act::Sigmoid, 6);
        CHECK(relative_error(a.get_params(), b.get_params()) == 0.0);
        CHECK(relative_error(a.get_params(), c.get_params()) > 0.0);
    }

    {
        Tensor p = softmax(Tensor({2, 3}, {1, 2, 3, 1000, 1000, 1000}));
        CHECK(std::abs(p[0] + p[1] + p[2] - 1.0) < 1e-12);
        CHECK(std::abs(p[3] - 1.0 / 3.0) < 1e-12);
    }

    {
        Tensor X({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
        Tensor Y({4, 1}, {0, 1, 1, 0});
        Mlp net({2, 6, 1}, Act::Tanh, Act::Sigmoid, 1);
        double first = 0.0, last = 0.0;
        for (int e = 0; e < 10000; ++e) {
            Tensor out = net.forward(X);
            last = mse_loss(out, Y);
            if (e == 0) first = last;
            net.backward(mse_grad(out, Y));
            net.sgd_step(0.5);
        }
        CHECK(last < first);
        CHECK(last < 0.05);
        CHECK(accuracy(net.forward(X), Y) == 1.0);
    }

    {
        const double centers[3][2] = {{-2, 0}, {2, 0}, {0, 3}};
        Tensor noise = random_uniform({60, 2}, 5, -0.6, 0.6);
        Tensor X({60, 2});
        std::vector<int> labels;
        for (int i = 0; i < 60; ++i) {
            int c = i % 3;
            labels.push_back(c);
            X[i * 2] = centers[c][0] + noise[i * 2];
            X[i * 2 + 1] = centers[c][1] + noise[i * 2 + 1];
        }
        Tensor Y = one_hot(labels, 3);
        Mlp net({2, 8, 3}, Act::Tanh, Act::Linear, 2);
        for (int e = 0; e < 500; ++e) {
            Tensor out = net.forward(X);
            net.backward(softmax_ce_grad(out, Y));
            net.sgd_step(0.5);
        }
        CHECK(accuracy(net.forward(X), Y) >= 0.95);
    }

    if (failures == 0) std::cout << "All neuron and MLP tests passed\n";
    return failures == 0 ? 0 : 1;
}
