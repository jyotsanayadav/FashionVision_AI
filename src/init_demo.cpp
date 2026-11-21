#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "dataset.h"
#include "mlp.h"
#include "trainer.h"

using namespace fv;

struct InitRun {
    std::string name;
    Init method;
};

int main(int argc, char** argv) {
    const std::string data_dir =
        (argc > 1) ? argv[1] : "dataset/fashion-mnist";

    const std::string out_dir =
        (argc > 2) ? argv[2] : "results";

    const size_t n_load =
        (argc > 3) ? std::stoul(argv[3]) : 12000;

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

        std::cout << "Training images: " << train.size()
                  << "\nValidation images: " << val.size()
                  << "\nTest images: " << test.size() << "\n\n";

        std::vector<InitRun> runs = {
            {"Xavier", Init::Xavier},
            {"He", Init::He},
            {"Random", Init::Random}
        };

        TrainConfig cfg;
        cfg.optimizer = "momentum";
        cfg.epochs = 15;
        cfg.batch_size = 64;
        cfg.lr = 0.01;
        cfg.momentum = 0.9;
        cfg.seed = 1;

        std::ofstream csv(out_dir + "/initialization_compare.csv");

        if (!csv) {
            std::cerr << "Cannot open results CSV file.\n";
            return 1;
        }

        csv << "initialization,val_loss,val_accuracy,"
               "test_loss,test_accuracy,best_epoch\n";

        std::cout << std::fixed << std::setprecision(4);

        std::cout << std::left << std::setw(12) << "Method"
                  << std::right << std::setw(12) << "Val Loss"
                  << std::setw(12) << "Val Acc"
                  << std::setw(12) << "Test Loss"
                  << std::setw(12) << "Test Acc"
                  << std::setw(12) << "Best Epoch"
                  << "\n";

        for (const auto& r : runs) {
            Mlp net(
                {784, 128, 10},
                Act::ReLU,
                Act::Linear,
                1,
                r.method
            );

            Tensor best_params;
            int best_epoch = 0;

            train_classifier(
                net, train, val, 10, cfg,
                &best_params, &best_epoch
            );

            net.set_params(best_params);

            double val_loss = 0.0;
            double val_acc = evaluate(net, val, 10, &val_loss);

            double test_loss = 0.0;
            double test_acc = evaluate(net, test, 10, &test_loss);

            csv << r.name << ","
                << val_loss << ","
                << val_acc << ","
                << test_loss << ","
                << test_acc << ","
                << best_epoch << "\n";

            std::cout << std::left << std::setw(12) << r.name
                      << std::right << std::setw(12) << val_loss
                      << std::setw(12) << val_acc
                      << std::setw(12) << test_loss
                      << std::setw(12) << test_acc
                      << std::setw(12) << best_epoch
                      << "\n";
        }

        std::cout << "\nResults saved to "
                  << out_dir << "/initialization_compare.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}