#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "dataset.h"
#include "linalg.h"
#include "optimizer.h"
#include "trainer.h"

using namespace fv;

static int failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cout << "FAIL line " << __LINE__ << ": " #cond << "\n";   \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

static void put_be32(std::ofstream& f, uint32_t v) {
    unsigned char b[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16),
                          (unsigned char)(v >> 8), (unsigned char)v};
    f.write(reinterpret_cast<char*>(b), 4);
}

static double run_quadratic(const std::string& name, double lr, int steps) {
    auto opt = make_optimizer(name, lr, 0.9);
    Tensor x({3}, {1.0, -2.0, 0.5});
    for (int i = 0; i < steps; ++i) {
        Tensor g = x;
        opt->step(x, g);
    }
    return frobenius_norm(x);
}

int main() {
    {
        Tensor p({1}, {1.0}), g({1}, {1.0});
        Momentum m(0.1, 0.9);
        m.step(p, g);
        CHECK(std::abs(p[0] - 0.9) < 1e-12);
        m.step(p, g);
        CHECK(std::abs(p[0] - 0.71) < 1e-12);

        Tensor q({1}, {1.0});
        Nesterov n(0.1, 0.9);
        n.step(q, g);
        CHECK(std::abs(q[0] - 0.81) < 1e-12);

        Tensor s({1}, {1.0});
        Sgd sg(0.1);
        sg.step(s, g);
        CHECK(std::abs(s[0] - 0.9) < 1e-12);
    }

    CHECK(run_quadratic("sgd", 0.1, 200) < 1e-6);
    CHECK(run_quadratic("momentum", 0.05, 300) < 1e-4);
    CHECK(run_quadratic("nesterov", 0.05, 300) < 1e-4);
    {
        bool threw = false;
        try { make_optimizer("adamx", 0.1); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }

    {
        {
            std::ofstream fi("tmp_images.idx", std::ios::binary), fl("tmp_labels.idx", std::ios::binary);
            put_be32(fi, 2051); put_be32(fi, 3); put_be32(fi, 2); put_be32(fi, 2);
            const unsigned char px[12] = {0, 255, 51, 102, 0, 255, 51, 102, 9, 9, 9, 9};
            fi.write(reinterpret_cast<const char*>(px), 12);
            put_be32(fl, 2049); put_be32(fl, 3);
            const char lab[3] = {4, 4, 7};
            fl.write(lab, 3);
        }
        Dataset d = load_idx("tmp_images.idx", "tmp_labels.idx");
        CHECK(d.size() == 3 && d.features() == 4);
        CHECK(std::abs(d.X[1] - 1.0) < 1e-12 && std::abs(d.X[2] - 0.2) < 1e-12);
        CHECK(d.y[2] == 7);
        CHECK(remove_duplicates(d) == 1);
        CHECK(d.size() == 2);

        Dataset other = subset(d, std::vector<size_t>{1});
        CHECK(remove_overlap(d, other) == 1 && d.size() == 1);
        std::remove("tmp_images.idx");
        std::remove("tmp_labels.idx");

        bool threw = false;
        try { load_idx("does_not_exist.idx", "nope.idx"); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    {
        Dataset d;
        d.X = random_uniform({100, 2}, 3);
        for (int i = 0; i < 100; ++i) d.y.push_back(i % 4);
        Dataset tr, va;
        split_stratified(d, 0.2, 7, tr, va);
        CHECK(tr.size() + va.size() == 100);
        auto cv = class_counts(va, 4);
        for (size_t c = 0; c < 4; ++c) CHECK(cv[c] == 5);
    }

    {
        auto a = shuffled_indices(50, 9), b = shuffled_indices(50, 9);
        CHECK(a == b);
        std::vector<bool> seen(50, false);
        for (size_t i : a) seen[i] = true;
        bool all = true;
        for (bool s : seen) all = all && s;
        CHECK(all);
    }

    {
        const double centers[3][2] = {{-2, 0}, {2, 0}, {0, 3}};
        Tensor noise = random_uniform({90, 2}, 5, -0.6, 0.6);
        Dataset d;
        d.X = Tensor({90, 2});
        for (int i = 0; i < 90; ++i) {
            int c = i % 3;
            d.y.push_back(c);
            d.X[i * 2] = centers[c][0] + noise[i * 2];
            d.X[i * 2 + 1] = centers[c][1] + noise[i * 2 + 1];
        }
        struct Case { const char* opt; double lr; };
        const Case cases[3] = {{"sgd", 0.1}, {"momentum", 0.02}, {"nesterov", 0.02}};
        for (const Case& cs : cases) {
            Mlp net({2, 8, 3}, Act::Tanh, Act::Linear, 2);
            TrainConfig cfg;
            cfg.optimizer = cs.opt;
            cfg.epochs = 60;
            cfg.batch_size = 30;
            cfg.lr = cs.lr;
            cfg.momentum = 0.9;
            auto h = train_classifier(net, d, d, 3, cfg);
            CHECK(h.size() == 60);
            CHECK(h.back().train_loss < h.front().train_loss);
            CHECK(h.back().val_acc >= 0.95);
        }
    }

    if (failures == 0) std::cout << "All optimizer and dataset tests passed\n";
    return failures == 0 ? 0 : 1;
}
