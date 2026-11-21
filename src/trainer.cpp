#include "trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include "optimizer.h"

namespace fv {

static Tensor rows_of(const Dataset& d, const std::vector<size_t>& order, size_t from, size_t to,
                      std::vector<int>& labels) {
    const size_t f = d.features();
    Tensor X({to - from, f});
    labels.assign(to - from, 0);
    for (size_t k = from; k < to; ++k) {
        const size_t src = order[k];
        for (size_t j = 0; j < f; ++j) X[(k - from) * f + j] = d.X[src * f + j];
        labels[k - from] = d.y[src];
    }
    return X;
}

double evaluate(Mlp& net, const Dataset& d, size_t classes, double* loss) {
    std::vector<size_t> order(d.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    const size_t chunk = 1000;
    double loss_sum = 0.0;
    size_t correct = 0;
    for (size_t s = 0; s < d.size(); s += chunk) {
        size_t e = std::min(d.size(), s + chunk);
        std::vector<int> labels;
        Tensor X = rows_of(d, order, s, e, labels);
        Tensor Y = one_hot(labels, classes);
        Tensor out = net.forward(X);
        loss_sum += softmax_ce_loss(out, Y) * static_cast<double>(e - s);
        correct += static_cast<size_t>(std::llround(accuracy(out, Y) * static_cast<double>(e - s)));
    }
    if (loss) *loss = loss_sum / static_cast<double>(d.size());
    return static_cast<double>(correct) / static_cast<double>(d.size());
}

// std::vector<EpochStats> train_classifier(Mlp& net, const Dataset& train, const Dataset& val,
//                                          size_t classes, const TrainConfig& cfg) {
std::vector<EpochStats> train_classifier(
    Mlp& net,
    const Dataset& train,
    const Dataset& val,
    size_t classes,
    const TrainConfig& cfg,
    Tensor* best_params,
    int* best_epoch
) {
    auto opt = make_optimizer(cfg.optimizer, cfg.lr, cfg.momentum);
    std::vector<EpochStats> history;
    double best_val_loss = INFINITY;

if (best_epoch) {
    *best_epoch = 0;
}
    const auto t0 = std::chrono::steady_clock::now();
    const size_t bs = std::min(cfg.batch_size, train.size());

    for (int e = 1; e <= cfg.epochs; ++e) {
        auto order = shuffled_indices(train.size(), cfg.seed + static_cast<uint64_t>(e));
        double loss_sum = 0.0;
        size_t batches = 0;
        for (size_t s = 0; s < train.size(); s += bs) {
            size_t end = std::min(train.size(), s + bs);
            std::vector<int> labels;
            Tensor X = rows_of(train, order, s, end, labels);
            Tensor Y = one_hot(labels, classes);

            Tensor out = net.forward(X);
            loss_sum += softmax_ce_loss(out, Y);
            ++batches;
            // net.backward(softmax_ce_grad(out, Y));

            // Tensor params = net.get_params();
            // opt->step(params, net.get_grads());
            // net.set_params(params);
            net.backward(softmax_ce_grad(out, Y));

Tensor params = net.get_params();
Tensor grads = net.get_grads();

// Apply L2 regularization to weights only (not biases)
size_t offset = 0;
const auto& sizes = net.sizes();

for (size_t layer = 0; layer + 1 < sizes.size(); ++layer) {
    size_t weight_count = sizes[layer] * sizes[layer + 1];

    for (size_t j = 0; j < weight_count; ++j) {
        grads[offset + j] += cfg.l2_lambda * params[offset + j];
    }

    // Move past weights and biases
    offset += weight_count + sizes[layer + 1];
}

opt->step(params, grads);
net.set_params(params);
        }
        EpochStats st;
        st.epoch = e;
        st.train_loss = loss_sum / static_cast<double>(batches);
        st.val_acc = evaluate(net, val, classes, &st.val_loss);
        if (st.val_loss < best_val_loss) {
    best_val_loss = st.val_loss;

    if (best_params) {
        *best_params = net.get_params();
    }

    if (best_epoch) {
        *best_epoch = e;
    }
}
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        history.push_back(st);
    }
    return history;
}

}  // namespace fv
