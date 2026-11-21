#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

namespace fv {

// 1. Hebbian Learning
// Δw = learning_rate * input * output
void hebbian_learning(
    std::vector<double>& weights,
    const std::vector<double>& inputs,
    double output,
    double lr
) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += lr * inputs[i] * output;
    }
}

// 2. Modified Hebbian Learning (Oja's Rule)
// Δw = lr * y * (x - y*w)
void oja_learning(
    std::vector<double>& weights,
    const std::vector<double>& inputs,
    double output,
    double lr
) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += lr * output *
            (inputs[i] - output * weights[i]);
    }
}

// 3. Competitive Learning
// Only the winning neuron updates its weights.
void competitive_learning(
    std::vector<std::vector<double>>& weights,
    const std::vector<double>& inputs,
    double lr
) {
    size_t winner = 0;
    double best_distance = 1e100;

    for (size_t i = 0; i < weights.size(); ++i) {
        double distance = 0.0;

        for (size_t j = 0; j < inputs.size(); ++j) {
            double diff = inputs[j] - weights[i][j];
            distance += diff * diff;
        }

        if (distance < best_distance) {
            best_distance = distance;
            winner = i;
        }
    }

    for (size_t j = 0; j < inputs.size(); ++j) {
        weights[winner][j] +=
            lr * (inputs[j] - weights[winner][j]);
    }
}

// 4. Error-Correction Learning (Delta Rule)
// Δw = lr * error * input
void error_correction_learning(
    std::vector<double>& weights,
    double& bias,
    const std::vector<double>& inputs,
    double target,
    double prediction,
    double lr
) {
    double error = target - prediction;

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] += lr * error * inputs[i];
    }

    bias += lr * error;
}

} // namespace fv

int main() {
    using namespace fv;

    std::cout << std::fixed << std::setprecision(4);

    // Same input for the first two learning rules.
    std::vector<double> inputs = {1.0, 0.5};
    double output = 0.8;
    double lr = 0.1;

    std::cout << "HEBBIAN LEARNING\n";

    std::vector<double> hebb_weights = {0.2, 0.3};
    hebbian_learning(hebb_weights, inputs, output, lr);

    for (double w : hebb_weights)
        std::cout << w << " ";
    std::cout << "\n\n";

    std::cout << "MODIFIED HEBBIAN LEARNING (OJA)\n";

    std::vector<double> oja_weights = {0.2, 0.3};
    oja_learning(oja_weights, inputs, output, lr);

    for (double w : oja_weights)
        std::cout << w << " ";
    std::cout << "\n\n";

    std::cout << "COMPETITIVE LEARNING\n";

    std::vector<std::vector<double>> competitive_weights = {
        {0.1, 0.2},
        {0.8, 0.9},
        {0.4, 0.5}
    };

    competitive_learning(
        competitive_weights, inputs, lr
    );

    for (size_t i = 0; i < competitive_weights.size(); ++i) {
        std::cout << "Neuron " << i + 1 << ": ";

        for (double w : competitive_weights[i])
            std::cout << w << " ";

        std::cout << "\n";
    }

    std::cout << "\nERROR-CORRECTION LEARNING\n";

    std::vector<double> error_weights = {0.2, 0.3};
    double bias = 0.1;
    double target = 1.0;
    double prediction = 0.6;

    error_correction_learning(
        error_weights, bias, inputs,
        target, prediction, lr
    );

    std::cout << "Weights: ";

    for (double w : error_weights)
        std::cout << w << " ";

    std::cout << "\nBias: " << bias << "\n";

    return 0;
}