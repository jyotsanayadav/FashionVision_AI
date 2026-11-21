#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "dataset.h"
#include "mlp.h"
#include "trainer.h"
using namespace fv;
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

        // Use a smaller training subset to make overfitting easier to observe.
        const size_t small_n = std::min<size_t>(1000, train.size());

        Dataset small_train;
        small_train.X = Tensor({small_n, 784});
        small_train.y.resize(small_n);

        for (size_t i = 0; i < small_n; ++i) {
            for (size_t j = 0; j < 784; ++j) {
                small_train.X[i * 784 + j] = train.X[i * 784 + j];
            }
            small_train.y[i] = train.y[i];
        }

        std::cout << "Training images: " << small_train.size()
                  << "\nValidation images: " << val.size()
                  << "\nTest images: " << test.size() << "\n\n";

        Mlp net({784, 256, 128, 10}, Act::ReLU, Act::Linear, 1);

        TrainConfig cfg;
        cfg.optimizer = "momentum";
        cfg.epochs = 40;
        cfg.batch_size = 64;
        cfg.lr = 0.01;
        cfg.momentum = 0.9;
        cfg.seed = 1;
        cfg.l2_lambda = 0.0;

        std::vector<EpochStats> history =
            train_classifier(net, small_train, val, 10, cfg);

        std::ofstream csv(out_dir + "/overfitting_history.csv");

        if (!csv) {
            std::cerr << "Cannot open results CSV file.\n";
            return 1;
        }

        csv << "epoch,train_loss,val_loss,val_accuracy,seconds\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Epoch   Train Loss   Val Loss   Val Accuracy\n";

        for (const auto& e : history) {
            std::cout << std::setw(5) << e.epoch
                      << std::setw(13) << e.train_loss
                      << std::setw(11) << e.val_loss
                      << std::setw(14) << e.val_acc << '\n';

            csv << e.epoch << ','
                << e.train_loss << ','
                << e.val_loss << ','
                << e.val_acc << ','
                << e.seconds << '\n';
        }

        std::cout << "\nResults saved to "
                  << out_dir << "/overfitting_history.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}