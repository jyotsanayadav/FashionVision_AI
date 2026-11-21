#include "mlp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "linalg.h"
#include "neuron.h"

namespace fv {

std::string act_name(Act a) {
    switch (a) {
        case Act::Linear:  return "linear";
        case Act::Sigmoid: return "sigmoid";
        case Act::Tanh:    return "tanh";
        case Act::ReLU:    return "relu";
    }
    return "?";
}

static double act_fn(Act a, double z) {
    switch (a) {
        case Act::Sigmoid: return sigmoid(z);
        case Act::Tanh:    return std::tanh(z);
        case Act::ReLU:    return z > 0.0 ? z : 0.0;
        default:           return z;
    }
}

static double act_deriv(Act a, double z, double out) {
    switch (a) {
        case Act::Sigmoid: return out * (1.0 - out);
        case Act::Tanh:    return 1.0 - out * out;
        case Act::ReLU:    return z > 0.0 ? 1.0 : 0.0;
        default:           return 1.0;
    }
}

// Mlp::Mlp(std::vector<size_t> sizes, Act hidden, Act output, uint64_t seed)
  Mlp::Mlp(std::vector<size_t> sizes, Act hidden, Act output,
         uint64_t seed, Init init) 
: sizes_(std::move(sizes)), hidden_(hidden), output_(output) {
    if (sizes_.size() < 2) throw std::invalid_argument("Mlp: need at least input and output sizes");
    for (size_t l = 0; l + 1 < sizes_.size(); ++l) {
        size_t in = sizes_[l], out = sizes_[l + 1];
        Layer L;
        double limit = 0.0;
        switch (init) {
            case Init::Xavier:
                limit = std::sqrt(6.0 / static_cast<double>(in + out));
                break;
            case Init::He:
limit = std::sqrt(6.0 / static_cast<double>(in));
                break;
            case Init::Random:
                limit = 0.1;
                break;
        }
        L.W = random_uniform({in, out}, seed + 1000 * l + 1, -limit, limit);
        L.b = Tensor({1, out});
        L.dW = Tensor({in, out});
        L.db = Tensor({1, out});
        L.act = (l + 2 == sizes_.size()) ? output_ : hidden_;
        layers_.push_back(std::move(L));
    }
}

Tensor Mlp::forward(const Tensor& X) {
    Tensor cur = X;
    for (auto& L : layers_) {
        if (cur.ndim() != 2 || cur.shape()[1] != L.W.shape()[0])
            throw std::invalid_argument("Mlp::forward: input width does not match the network");
        L.in = cur;
        L.z = cur.matmul(L.W) + L.b;
        L.a = L.z;
        for (size_t i = 0; i < L.a.size(); ++i) L.a[i] = act_fn(L.act, L.z[i]);
        cur = L.a;
    }
    return cur;
}

void Mlp::backward(const Tensor& dOut) {
    Tensor d = dOut;
    for (size_t l = layers_.size(); l-- > 0;) {
        Layer& L = layers_[l];
        Tensor dZ = d;
        for (size_t i = 0; i < dZ.size(); ++i) dZ[i] = d[i] * act_deriv(L.act, L.z[i], L.a[i]);

        L.dW = L.in.transpose().matmul(dZ);
        const size_t N = dZ.shape()[0], out = dZ.shape()[1];
        L.db = Tensor({1, out});
        for (size_t r = 0; r < N; ++r)
            for (size_t c = 0; c < out; ++c) L.db[c] += dZ[r * out + c];

        d = dZ.matmul(L.W.transpose());
    }
}

void Mlp::sgd_step(double lr) {
    for (auto& L : layers_) {
        for (size_t i = 0; i < L.W.size(); ++i) L.W[i] -= lr * L.dW[i];
        for (size_t i = 0; i < L.b.size(); ++i) L.b[i] -= lr * L.db[i];
    }
}

size_t Mlp::num_params() const {
    size_t n = 0;
    for (const auto& L : layers_) n += L.W.size() + L.b.size();
    return n;
}

Tensor Mlp::get_params() const {
    Tensor out(std::vector<size_t>{num_params()});
    size_t k = 0;
    for (const auto& L : layers_) {
        for (size_t i = 0; i < L.W.size(); ++i) out[k++] = L.W[i];
        for (size_t i = 0; i < L.b.size(); ++i) out[k++] = L.b[i];
    }
    return out;
}

void Mlp::set_params(const Tensor& flat) {
    if (flat.size() != num_params()) throw std::invalid_argument("set_params: wrong size");
    size_t k = 0;
    for (auto& L : layers_) {
        for (size_t i = 0; i < L.W.size(); ++i) L.W[i] = flat[k++];
        for (size_t i = 0; i < L.b.size(); ++i) L.b[i] = flat[k++];
    }
}

Tensor Mlp::get_grads() const {
    Tensor out(std::vector<size_t>{num_params()});
    size_t k = 0;
    for (const auto& L : layers_) {
        for (size_t i = 0; i < L.dW.size(); ++i) out[k++] = L.dW[i];
        for (size_t i = 0; i < L.db.size(); ++i) out[k++] = L.db[i];
    }
    return out;
}

double mse_loss(const Tensor& P, const Tensor& Y) {
    if (P.shape() != Y.shape()) throw std::invalid_argument("mse_loss: shapes differ");
    double s = 0.0;
    for (size_t i = 0; i < P.size(); ++i) s += (P[i] - Y[i]) * (P[i] - Y[i]);
    return s / static_cast<double>(P.shape()[0]);
}

Tensor mse_grad(const Tensor& P, const Tensor& Y) {
    Tensor g(P.shape());
    const double n = static_cast<double>(P.shape()[0]);
    for (size_t i = 0; i < P.size(); ++i) g[i] = 2.0 * (P[i] - Y[i]) / n;
    return g;
}

Tensor softmax(const Tensor& z) {
    const size_t N = z.shape()[0], C = z.shape()[1];
    Tensor p(z.shape());
    for (size_t r = 0; r < N; ++r) {
        double mx = z[r * C];
        for (size_t c = 1; c < C; ++c) mx = std::max(mx, z[r * C + c]);
        double sum = 0.0;
        for (size_t c = 0; c < C; ++c) {
            p[r * C + c] = std::exp(z[r * C + c] - mx);
            sum += p[r * C + c];
        }
        for (size_t c = 0; c < C; ++c) p[r * C + c] /= sum;
    }
    return p;
}

double softmax_ce_loss(const Tensor& logits, const Tensor& Y) {
    Tensor p = softmax(logits);
    double s = 0.0;
    for (size_t i = 0; i < p.size(); ++i) s -= Y[i] * std::log(std::max(p[i], 1e-12));
    return s / static_cast<double>(p.shape()[0]);
}

Tensor softmax_ce_grad(const Tensor& logits, const Tensor& Y) {
    Tensor g = softmax(logits) - Y;
    return g * (1.0 / static_cast<double>(logits.shape()[0]));
}

Tensor one_hot(const std::vector<int>& labels, size_t classes) {
    Tensor t({labels.size(), classes});
    for (size_t i = 0; i < labels.size(); ++i) t[i * classes + static_cast<size_t>(labels[i])] = 1.0;
    return t;
}

double accuracy(const Tensor& P, const Tensor& Y) {
    const size_t N = P.shape()[0], C = P.shape()[1];
    size_t ok = 0;
    for (size_t r = 0; r < N; ++r) {
        if (C == 1) {
            ok += ((P[r] >= 0.5) == (Y[r] >= 0.5));
        } else {
            size_t bp = 0, by = 0;
            for (size_t c = 1; c < C; ++c) {
                if (P[r * C + c] > P[r * C + bp]) bp = c;
                if (Y[r * C + c] > Y[r * C + by]) by = c;
            }
            ok += (bp == by);
        }
    }
    return static_cast<double>(ok) / static_cast<double>(N);
}

}  // namespace fv
