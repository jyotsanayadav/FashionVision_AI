#include <iomanip>
#include <iostream>
#include <string>
#include "experiment.h"
#include "gradcheck.h"
#include "linalg.h"
#include "mlp.h"
#include "neuron.h"
#include "viz.h"

using namespace fv;

static void mp_demo() {
    std::cout << "== 1. McCulloch-Pitts neurons (fixed weights, threshold) ==\n";
    std::cout << " a b | AND OR NAND NOR | XOR (2 layers)\n";
    auto AND = McCullochPitts::AND2(), OR = McCullochPitts::OR2();
    auto NAND = McCullochPitts::NAND2(), NOR = McCullochPitts::NOR2();
    for (int a = 0; a <= 1; ++a)
        for (int b = 0; b <= 1; ++b)
            std::cout << " " << a << " " << b << " |  " << AND.fire({a, b}) << "   " << OR.fire({a, b})
                      << "    " << NAND.fire({a, b}) << "    " << NOR.fire({a, b}) << " |  "
                      << mp_xor(a, b) << "\n";
    std::cout << "NOT(0)=" << McCullochPitts::NOT1().fire({0}) << "  NOT(1)=" << McCullochPitts::NOT1().fire({1}) << "\n\n";
}

static void sigmoid_demo() {
    std::cout << "== 2. Single sigmoid neuron trained with gradient descent ==\n";
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    struct Task { const char* name; std::vector<double> t; };
    std::vector<Task> tasks = {{"AND", {0, 0, 0, 1}}, {"OR", {0, 1, 1, 1}}, {"XOR", {0, 1, 1, 0}}};
    for (const auto& task : tasks) {
        SigmoidNeuron n(2, 1);
        double loss = 0.0;
        for (int e = 0; e < 5000; ++e) loss = n.train_epoch(X, task.t, 0.5);
        int correct = 0;
        for (size_t i = 0; i < X.size(); ++i) correct += ((n.forward(X[i]) >= 0.5) == (task.t[i] >= 0.5));
        std::cout << "  " << task.name << ": " << correct << "/4 correct, final loss " << loss
                  << (std::string(task.name) == "XOR" ? "   <- a single neuron cannot learn XOR" : "") << "\n";
    }
    std::cout << "\n";
}

static void xor_mlp_demo(const std::string& dir) {
    std::cout << "== 3. MLP (2-6-1) learns XOR with backprop ==\n";
    Tensor X({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1});
    Tensor Y({4, 1}, {0, 1, 1, 0});

    const double lr = 0.5;
    const int epochs = 10000;
    const uint64_t seed = 1;
    Mlp net({2, 6, 1}, Act::Tanh, Act::Sigmoid, seed);

    ExperimentLog log("xor_mlp");
    log.hparam("layers", "2-6-1");
    log.hparam("hidden_activation", act_name(net.hidden_act()));
    log.hparam("output_activation", act_name(net.output_act()));
    log.hparam("loss", "mse");
    log.hparam("learning_rate", lr);
    log.hparam("epochs", epochs);
    log.hparam("seed", static_cast<double>(seed));
    log.hparam("initialization", "xavier_uniform");
    log.param_count(net.num_params());

    std::cout << "  hyperparameters: lr=" << lr << ", epochs=" << epochs << ", layers=2-6-1, seed=" << seed << "\n";
    std::cout << "  learned parameters: " << net.num_params() << " (weights + biases)\n";

    std::vector<double> curve;
    for (int e = 0; e <= epochs; ++e) {
        Tensor out = net.forward(X);
        double loss = mse_loss(out, Y);
        if (e % 50 == 0) {
            log.epoch(e, loss, accuracy(out, Y));
            curve.push_back(loss);
        }
        if (e % 2000 == 0) std::cout << "  epoch " << std::setw(5) << e << "  loss " << loss << "\n";
        net.backward(mse_grad(out, Y));
        net.sgd_step(lr);
    }
    Tensor out = net.forward(X);
    std::cout << "  predictions: ";
    for (size_t i = 0; i < 4; ++i) std::cout << out[i] << " ";
    std::cout << "(targets 0 1 1 0), accuracy " << accuracy(out, Y) * 100 << "%\n";

    if (log.save(dir) && write_line_svg(curve, dir + "/xor_loss_curve.svg", "XOR training loss",
                                        "epoch / 50", "MSE loss"))
        std::cout << "  wrote " << dir << "/xor_mlp_config.json, xor_mlp_history.csv, xor_loss_curve.svg\n";
    else
        std::cout << "  ERROR: could not write into " << dir << "\n";
    std::cout << "\n";
}

static double check_mlp(Act h, Act o, bool ce) {
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

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "results";
    std::cout << std::fixed << std::setprecision(4);
    mp_demo();
    sigmoid_demo();
    xor_mlp_demo(dir);

    std::cout << "== 4. Backprop verified against numerical gradients ==\n";
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "  tanh    -> sigmoid + MSE      : " << check_mlp(Act::Tanh, Act::Sigmoid, false) << "\n";
    std::cout << "  sigmoid -> linear  + MSE      : " << check_mlp(Act::Sigmoid, Act::Linear, false) << "\n";
    std::cout << "  tanh    -> linear  + softmax CE: " << check_mlp(Act::Tanh, Act::Linear, true) << "\n";
    std::cout << "  relu    -> linear  + softmax CE: " << check_mlp(Act::ReLU, Act::Linear, true) << "\n";
    std::cout << "  (below 1e-5 is good)\n";
    return 0;
}
