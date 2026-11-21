#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "dataset.h"
#include "mlp.h"
#include "trainer.h"

using namespace fv;

void scale_pixels(Dataset& d, double factor) {
    for (size_t i = 0; i < d.X.size(); ++i) {
        d.X[i] *= factor;
    }
}

int main(int argc, char** argv) {
    const std::string data_dir =
        (argc > 1) ? argv[1] : "dataset/fashion-mnist";

    const size_t n_load =
        (argc > 2) ? std::stoul(argv[2]) : 12000;

    try {
        Dataset all = load_idx(
            data_dir + "/train-images-idx3-ubyte",
            data_dir + "/train-labels-idx1-ubyte",
            n_load
        );

        Dataset test = load_idx(
            data_dir + "/t10k-images-idx3-ubyte",
            data_dir + "/t10k-labels-idx1-ubyte"
        );

        remove_duplicates(all);
        remove_overlap(all, test);

        Dataset train, val;
        split_stratified(all, 1.0 / 6.0, 42, train, val);

        Dataset raw_train = train;
        Dataset raw_val = val;
        Dataset raw_test = test;

        scale_pixels(raw_train, 255.0);
        scale_pixels(raw_val, 255.0);
        scale_pixels(raw_test, 255.0);

        TrainConfig normalized_cfg;
        normalized_cfg.optimizer = "momentum";
        normalized_cfg.epochs = 15;
        normalized_cfg.batch_size = 64;
        normalized_cfg.lr = 0.01;
        normalized_cfg.momentum = 0.9;
        normalized_cfg.seed = 1;

        TrainConfig raw_cfg = normalized_cfg;
        raw_cfg.lr = 0.00001;

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "Training images: " << train.size()
                  << "\nValidation images: " << val.size()
                  << "\nTest images: " << test.size() << "\n\n";

        std::cout << std::left << std::setw(18) << "Input"
                  << std::right << std::setw(12) << "Val Loss"
                  << std::setw(12) << "Val Acc"
                  << std::setw(12) << "Test Loss"
                  << std::setw(12) << "Test Acc"
                  << "\n";

        // Experiment 1: normalized pixels [0,1]
        Mlp normalized_net(
            {784, 128, 10}, Act::ReLU, Act::Linear, 1
        );

        Tensor best_normalized;
        int normalized_epoch = 0;

        train_classifier(
            normalized_net, train, val, 10,
            normalized_cfg, &best_normalized, &normalized_epoch
        );

        normalized_net.set_params(best_normalized);

        double normalized_val_loss = 0.0;
        double normalized_val_acc =
            evaluate(normalized_net, val, 10, &normalized_val_loss);

        double normalized_test_loss = 0.0;
        double normalized_test_acc =
            evaluate(normalized_net, test, 10, &normalized_test_loss);

        std::cout << std::left << std::setw(18) << "Normalized [0,1]"
                  << std::right << std::setw(12) << normalized_val_loss
                  << std::setw(12) << normalized_val_acc
                  << std::setw(12) << normalized_test_loss
                  << std::setw(12) << normalized_test_acc
                  << "\n";

        // Experiment 2: raw pixels [0,255]
        Mlp raw_net(
            {784, 128, 10}, Act::ReLU, Act::Linear, 1
        );

        Tensor best_raw;
        int raw_epoch = 0;

        train_classifier(
            raw_net, raw_train, raw_val, 10,
            raw_cfg, &best_raw, &raw_epoch
        );

        raw_net.set_params(best_raw);

        double raw_val_loss = 0.0;
        double raw_val_acc =
            evaluate(raw_net, raw_val, 10, &raw_val_loss);

        double raw_test_loss = 0.0;
        double raw_test_acc =
            evaluate(raw_net, raw_test, 10, &raw_test_loss);

        std::cout << std::left << std::setw(18) << "Raw [0,255]"
                  << std::right << std::setw(12) << raw_val_loss
                  << std::setw(12) << raw_val_acc
                  << std::setw(12) << raw_test_loss
                  << std::setw(12) << raw_test_acc
                  << "\n";

        std::cout << "\nBest normalized epoch: " << normalized_epoch
                  << "\nBest raw epoch: " << raw_epoch << "\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}