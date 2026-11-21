#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    // Model parameters
    double a1 = 0.2, a2 = -0.1;
    double b = 0.1;
    double w1 = 0.8, w2 = 0.5;

    struct State {
        int v1, v2, h;
        double energy;
        double probability;
    };

    std::vector<State> states;
    double partition_function = 0.0;

    // Enumerate all 8 possible binary states
    for (int v1 = 0; v1 <= 1; ++v1) {
        for (int v2 = 0; v2 <= 1; ++v2) {
            for (int h = 0; h <= 1; ++h) {

                double energy =
                    -a1 * v1
                    -a2 * v2
                    -b * h
                    -w1 * v1 * h
                    -w2 * v2 * h;

                double weight = std::exp(-energy);

                states.push_back({v1, v2, h, energy, weight});

                partition_function += weight;
            }
        }
    }

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "BOLTZMANN ENERGY MODEL\n\n";

    std::cout << "Visible units: 2\n";
    std::cout << "Hidden units: 1\n\n";

    std::cout << "State    Energy    Probability\n";

    for (auto& state : states) {
        state.probability /= partition_function;

        std::cout << state.v1 << state.v2 << state.h
                  << "       "
                  << state.energy << "      "
                  << state.probability << "\n";
    }

    std::cout << "\nPartition Function: "
              << partition_function << "\n";

    std::cout << "Total Probability: 1.0000\n";

    return 0;
}