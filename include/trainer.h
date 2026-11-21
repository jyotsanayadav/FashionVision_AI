#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "dataset.h"
#include "mlp.h"

namespace fv {

// struct TrainConfig {
//     std::string optimizer = "sgd";   // sgd | momentum | nesterov
//     int epochs = 10;
//     size_t batch_size = 64;          // = train.size() for batch gradient descent
//     double lr = 0.05;
//     double momentum = 0.9;
//     uint64_t seed = 1;
// };
struct TrainConfig {
    std::string optimizer = "sgd";
    int epochs = 10;
    size_t batch_size = 64;
    double lr = 0.05;
    double momentum = 0.9;
    uint64_t seed = 1;
    double l2_lambda = 0.0;
};
struct EpochStats {
    int epoch;
    double train_loss;
    double val_loss;
    double val_acc;
    double seconds;   // cumulative training time
};

// std::vector<EpochStats> train_classifier(Mlp& net, const Dataset& train, const Dataset& val,
//                                          size_t classes, const TrainConfig& cfg);
std::vector<EpochStats> train_classifier(
    Mlp& net,
    const Dataset& train,
    const Dataset& val,
    size_t classes,
    const TrainConfig& cfg,
    Tensor* best_params = nullptr,
    int* best_epoch = nullptr
);
double evaluate(Mlp& net, const Dataset& d, size_t classes, double* loss = nullptr);

}  // namespace fv
