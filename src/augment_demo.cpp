#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "dataset.h"
#include "mlp.h"
#include "trainer.h"

using namespace fv;

// Create an augmented copy using small random translations.
Dataset augment_dataset(const Dataset& original, uint64_t seed) {
    const size_t n = original.size();
    const size_t rows = 28;
    const size_t cols = 28;

    Dataset result;
    result.X = Tensor({n * 2, rows * cols});
    result.y.resize(n * 2);

    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::uniform_int_distribution<int> shift(-2, 2);

    for (size_t i = 0; i < n; ++i) {
        int dx = shift(rng);
        int dy = shift(rng);

        for (size_t y = 0; y < rows; ++y) {
            for (size_t x = 0; x < cols; ++x) {
                // Keep the original image.
                result.X[i * rows * cols + y * cols + x] =
                    original.X[i * rows * cols + y * cols + x];

                // Create translated image.
                int src_y = static_cast<int>(y) - dy;
                int src_x = static_cast<int>(x) - dx;

                double pixel = 0.0;

                if (src_y >= 0 && src_y < 28 &&
                    src_x >= 0 && src_x < 28) {
                    pixel = original.X[
                        i * rows * cols +
                        static_cast<size_t>(src_y) * cols +
                        static_cast<size_t>(src_x)
                    ];
                }

                result.X[
                    (n + i) * rows * cols + y * cols + x
                ] = pixel;
            }
        }

        result.y[i] = original.y[i];
        result.y[n + i] = original.y[i];
    }

    return result;
}

struct Result {
    std::string method;
    double val_loss;
    double val_acc;
    double test_loss;
    double test_acc;
    int best_epoch;
};

int main(int argc, char** argv) {
    const std::string data_dir =
        argc > 1 ? argv[1] : "dataset/fashion-mnist";

    const std::string out_dir =
        argc > 2 ? argv[2] : "results";

    const size_t n_load =
        argc > 3 ? std::stoul(argv[3]) : 12000;

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

        std::cout << "Original training images: "
                  << train.size()
                  << "\nValidation images: " << val.size()
                  << "\nTest images: " << test.size()
                  << "\n\n";

        Dataset augmented_train = augment_dataset(train, 42);

        std::cout << "Augmented training images: "
                  << augmented_train.size() << "\n\n";

        std::vector<Result> results;

        for (int mode = 0; mode < 2; ++mode) {
            const bool use_augmentation = (mode == 1);

            const Dataset& current_train =
                use_augmentation ? augmented_train : train;

            Mlp net({784, 128, 10}, Act::ReLU, Act::Linear, 1);

            TrainConfig cfg;
            cfg.optimizer = "momentum";
            cfg.epochs = 15;
            cfg.batch_size = 64;
            cfg.lr = 0.01;
            cfg.momentum = 0.9;
            cfg.seed = 1;

            Tensor best_params;
            int best_epoch = 0;

            train_classifier(
                net, current_train, val, 10, cfg,
                &best_params, &best_epoch
            );

            net.set_params(best_params);

            Result r;
            r.method = use_augmentation
                ? "With Augmentation"
                : "Without Augmentation";

            r.val_acc = evaluate(net, val, 10, &r.val_loss);
            r.test_acc = evaluate(net, test, 10, &r.test_loss);
            r.best_epoch = best_epoch;

            results.push_back(r);
        }

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "Method              Val Loss   Val Acc"
                     "   Test Loss   Test Acc   Best Epoch\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(20) << r.method
                      << std::right << std::setw(10) << r.val_loss
                      << std::setw(10) << r.val_acc
                      << std::setw(12) << r.test_loss
                      << std::setw(11) << r.test_acc
                      << std::setw(11) << r.best_epoch
                      << '\n';
        }

        std::ofstream csv(
            out_dir + "/augmentation_compare.csv"
        );

        if (!csv) {
            std::cerr << "Cannot open results CSV file.\n";
            return 1;
        }

        csv << "method,val_loss,val_accuracy,"
               "test_loss,test_accuracy,best_epoch\n";

        for (const auto& r : results) {
            csv << r.method << ','
                << r.val_loss << ','
                << r.val_acc << ','
                << r.test_loss << ','
                << r.test_acc << ','
                << r.best_epoch << '\n';
        }

        std::cout << "\nResults saved to "
                  << out_dir << "/augmentation_compare.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}