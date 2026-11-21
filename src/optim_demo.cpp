#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "dataset.h"
#include "experiment.h"
#include "mlp.h"
#include "trainer.h"
#include "viz.h"

using namespace fv;

struct Run {
    std::string label;
    TrainConfig cfg;
};

static TrainConfig make_cfg(const std::string& opt, int epochs, size_t bs,
                            double lr, double mu) {
    TrainConfig c;
    c.optimizer = opt;
    c.epochs = epochs;
    c.batch_size = bs;
    c.lr = lr;
    c.momentum = mu;
    c.seed = 1;
    return c;
}

int main(int argc, char** argv) {
    const std::string data_dir =
        (argc > 1) ? argv[1] : "dataset/fashion-mnist";

    const std::string out_dir =
        (argc > 2) ? argv[2] : "results";

    const size_t n_load =
        (argc > 3) ? std::stoul(argv[3]) : 60000;

    std::cout << std::fixed << std::setprecision(4);

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

        size_t dups = remove_duplicates(all);
        size_t leaks = remove_overlap(all, test);

        Dataset train, val;

        split_stratified(all, 1.0 / 6.0, 42, train, val);

        std::cout << "Loaded " << n_load
                  << " training images, removed " << dups
                  << " duplicates and " << leaks
                  << " train/test overlaps\n";

        std::cout << "Split: train " << train.size()
                  << ", validation " << val.size()
                  << " (test set of " << test.size()
                  << " untouched)\n\n";

        const int epochs = 15;

        std::vector<Run> runs;

        // runs.push_back({
        //     "Batch_GD",
        //     make_cfg("sgd", epochs, train.size(), 0.5, 0.0)
        // });
        runs.push_back({
    "LargeBatch",
    make_cfg("sgd", epochs, 4096, 0.01, 0.0)
});

        runs.push_back({
            "SGD",
            make_cfg("sgd", epochs, 64, 0.05, 0.0)
        });

        runs.push_back({
            "Momentum",
            make_cfg("momentum", epochs, 64, 0.01, 0.9)
        });

        runs.push_back({
            "Nesterov",
            make_cfg("nesterov", epochs, 64, 0.01, 0.9)
        });

        std::ofstream csv(out_dir + "/optimizer_compare.csv");

        if (!csv) {
            std::cerr << "ERROR: cannot write into "
                      << out_dir << "\n";
            return 1;
        }

        csv << "optimizer,epoch,train_loss,val_loss,val_acc,seconds\n";

        std::vector<std::string> names;

        std::vector<std::vector<double>> loss_curves, acc_curves;

        std::cout << std::left << std::setw(10) << "method"
          << std::right << std::setw(10) << "updates"
          << std::setw(12) << "val loss"
          << std::setw(10) << "val acc"
          << std::setw(12) << "test loss"
          << std::setw(10) << "test acc"
          << std::setw(10) << "time(s)"
          << "\n";

        for (const auto& r : runs) {

            Mlp net({784, 128, 10}, Act::ReLU, Act::Linear, 1);

            // Store the best model weights
            Tensor best_params;
            int best_epoch = 0;

            auto hist = train_classifier(
                net,
                train,
                val,
                10,
                r.cfg,
                &best_params,
                &best_epoch
            );

            // Restore the weights from the best validation epoch
            net.set_params(best_params);
            double best_val_loss = 0.0;
double best_val_acc = evaluate(net, val, 10, &best_val_loss);

double test_loss = 0.0;
double test_acc = evaluate(net, test, 10, &test_loss);
            ExperimentLog log("optim_" + r.label);

            log.hparam("optimizer", r.cfg.optimizer);
            log.hparam("batch_size", static_cast<double>(r.cfg.batch_size));
            log.hparam("learning_rate", r.cfg.lr);
            log.hparam("momentum", r.cfg.momentum);
            log.hparam("epochs", static_cast<double>(r.cfg.epochs));
            log.hparam("best_epoch", static_cast<double>(best_epoch));
            log.hparam("architecture", "784-128-10 relu");

            log.param_count(net.num_params());

            std::vector<double> lc, ac;

            for (const auto& h : hist) {

                csv << r.label << ","
                    << h.epoch << ","
                    << h.train_loss << ","
                    << h.val_loss << ","
                    << h.val_acc << ","
                    << h.seconds << "\n";

                lc.push_back(h.val_loss);
                ac.push_back(h.val_acc);

                log.epoch(
                    h.epoch,
                    h.val_loss,
                    h.val_acc
                );
            }

            log.save(out_dir);

            names.push_back(r.label);
            loss_curves.push_back(lc);
            acc_curves.push_back(ac);

            size_t per_epoch =
                (train.size() + r.cfg.batch_size - 1) /
                r.cfg.batch_size;

            std::cout << std::left << std::setw(10)
          << r.label
          << std::right << std::setw(10)
          << per_epoch * static_cast<size_t>(epochs)
          << std::setw(12)
          << best_val_loss
          << std::setw(10)
          << best_val_acc
          << std::setw(12)
          << test_loss
          << std::setw(10)
          << test_acc
          << std::setw(10)
          << hist.back().seconds
          << "\n";
        }

        write_multiline_svg(
            names,
            loss_curves,
            out_dir + "/optimizer_val_loss.svg",
            "Validation loss by optimizer",
            "epoch",
            "cross-entropy"
        );

        write_multiline_svg(
            names,
            acc_curves,
            out_dir + "/optimizer_val_acc.svg",
            "Validation accuracy by optimizer",
            "epoch",
            "accuracy"
        );

        std::cout << "\nWrote optimizer_compare.csv, "
                  << "two SVG charts and per-run configs in "
                  << out_dir << "\n";

    } catch (const std::exception& e) {

        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}