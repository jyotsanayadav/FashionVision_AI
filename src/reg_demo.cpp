#include "dataset.h"
#include "trainer.h"
#include "mlp.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace fv;

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : "dataset/fashion-mnist";
    const std::string out_dir = argc > 2 ? argv[2] : "results";
    const size_t n_load = argc > 3 ? std::stoul(argv[3]) : 12000;


    Dataset all = load_idx(
        data_dir + "/train-images-idx3-ubyte",
        data_dir + "/train-labels-idx1-ubyte",
        n_load
    );

    Dataset test = load_idx(
        data_dir + "/t10k-images-idx3-ubyte",
        data_dir + "/t10k-labels-idx1-ubyte"
    );

    // remove_duplicates_and_overlap(all, test);

    // auto split = stratified_split(all, 1.0 / 6.0, 42);
    // Dataset train = split.first;
    // Dataset val = split.second;
    remove_duplicates(all);
remove_overlap(all, test);

Dataset train, val;
split_stratified(all, 1.0 / 6.0, 42, train, val);

    std::cout << "Training images: " << train.size() << '\n';
    std::cout << "Validation images: " << val.size() << '\n';
    std::cout << "Test images: " << test.size() << "\n\n";

    struct Result {
        double lambda;
        double val_loss;
        double val_acc;
        double test_loss;
        double test_acc;
        int best_epoch;
    };

    std::vector<Result> results;

    for (double lambda : {0.0, 0.0001}) {
        Mlp net({784, 128, 10}, Act::ReLU, Act::Linear, 1);

        TrainConfig cfg;
        cfg.optimizer = "momentum";
        cfg.epochs = 15;
        cfg.batch_size = 64;
        cfg.lr = 0.01;
        cfg.momentum = 0.9;
        cfg.seed = 1;
        cfg.l2_lambda = lambda;

        Tensor best_params;
        int best_epoch = 0;

        auto history = train_classifier(
            net, train, val, 10, cfg, &best_params, &best_epoch
        );

        if (best_params.size() > 0) {
            net.set_params(best_params);
        }

        Result r;
        r.lambda = lambda;
        r.val_acc = evaluate(net, val, 10, &r.val_loss);
        r.test_acc = evaluate(net, test, 10, &r.test_loss);
        r.best_epoch = best_epoch;

        results.push_back(r);
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Lambda       Val Loss   Val Acc   Test Loss   Test Acc   Best Epoch\n";

    for (const auto& r : results) {
        std::cout << std::setw(8) << r.lambda << "   "
                  << std::setw(8) << r.val_loss << "   "
                  << std::setw(7) << r.val_acc << "   "
                  << std::setw(9) << r.test_loss << "   "
                  << std::setw(8) << r.test_acc << "   "
                  << r.best_epoch << '\n';
    }

    std::ofstream csv(out_dir + "/regularization_compare.csv");

    csv << "lambda,val_loss,val_accuracy,test_loss,test_accuracy,best_epoch\n";

    for (const auto& r : results) {
        csv << r.lambda << ','
            << r.val_loss << ','
            << r.val_acc << ','
            << r.test_loss << ','
            << r.test_acc << ','
            << r.best_epoch << '\n';
    }

    std::cout << "\nSaved results to "
              << out_dir << "/regularization_compare.csv\n";

    return 0;
}