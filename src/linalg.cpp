#include "linalg.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fv {

Tensor identity(size_t n) {
    Tensor I({n, n});
    for (size_t i = 0; i < n; ++i) I[i * n + i] = 1.0;
    return I;
}

double frobenius_norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.size(); ++i) s += t[i] * t[i];
    return std::sqrt(s);
}

Tensor random_uniform(std::vector<size_t> shape, uint64_t seed, double lo, double hi) {
    Tensor t(std::move(shape));
    uint64_t state = seed * 2862933555777941757ULL + 3037000493ULL;
    for (size_t i = 0; i < t.size(); ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u = static_cast<double>(state >> 11) / 9007199254740992.0;
        t[i] = lo + (hi - lo) * u;
    }
    return t;
}

// One-sided Jacobi (Hestenes), m >= n.
static SVDResult svd_tall(const Tensor& A, double tol, int max_sweeps) {
    const size_t m = A.shape()[0], n = A.shape()[1];
    Tensor U = A;
    Tensor V = identity(n);

    int sweeps = 0;
    for (; sweeps < max_sweeps; ++sweeps) {
        bool rotated = false;
        for (size_t p = 0; p + 1 < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (size_t i = 0; i < m; ++i) {
                    double up = U[i * n + p], uq = U[i * n + q];
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if (gamma == 0.0 || std::abs(gamma) <= tol * std::sqrt(alpha * beta)) continue;
                rotated = true;

                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = c * t;

                for (size_t i = 0; i < m; ++i) {
                    double up = U[i * n + p], uq = U[i * n + q];
                    U[i * n + p] = c * up - s * uq;
                    U[i * n + q] = s * up + c * uq;
                }
                for (size_t i = 0; i < n; ++i) {
                    double vp = V[i * n + p], vq = V[i * n + q];
                    V[i * n + p] = c * vp - s * vq;
                    V[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if (!rotated) break;
    }

    std::vector<double> S(n);
    double smax = 0.0;
    for (size_t j = 0; j < n; ++j) {
        double s2 = 0.0;
        for (size_t i = 0; i < m; ++i) s2 += U[i * n + j] * U[i * n + j];
        S[j] = std::sqrt(s2);
        smax = std::max(smax, S[j]);
    }
    for (size_t j = 0; j < n; ++j) {
        if (S[j] > 0.0 && S[j] > 1e-13 * smax) {
            for (size_t i = 0; i < m; ++i) U[i * n + j] /= S[j];
        } else {
            S[j] = 0.0;
            for (size_t i = 0; i < m; ++i) U[i * n + j] = 0.0;
        }
    }

    std::vector<size_t> order(n);
    for (size_t j = 0; j < n; ++j) order[j] = j;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return S[a] > S[b]; });

    SVDResult res;
    res.U = Tensor({m, n});
    res.V = Tensor({n, n});
    res.S = Tensor(std::vector<size_t>{n});
    res.sweeps = sweeps;
    for (size_t k = 0; k < n; ++k) {
        size_t j = order[k];
        res.S[k] = S[j];
        for (size_t i = 0; i < m; ++i) res.U[i * n + k] = U[i * n + j];
        for (size_t i = 0; i < n; ++i) res.V[i * n + k] = V[i * n + j];
    }
    return res;
}

SVDResult svd(const Tensor& A, double tol, int max_sweeps) {
    if (A.ndim() != 2) throw std::invalid_argument("svd: need a 2D tensor");
    if (A.shape()[0] >= A.shape()[1]) return svd_tall(A, tol, max_sweeps);
    SVDResult t = svd_tall(A.transpose(), tol, max_sweeps);
    SVDResult r;
    r.U = t.V;
    r.S = t.S;
    r.V = t.U;
    r.sweeps = t.sweeps;
    return r;
}

Tensor reconstruct(const SVDResult& r, size_t rank) {
    const size_t m = r.U.shape()[0], ku = r.U.shape()[1];
    const size_t n = r.V.shape()[0], kv = r.V.shape()[1];
    size_t k = r.S.size();
    if (rank > 0 && rank < k) k = rank;

    Tensor out({m, n});
    for (size_t p = 0; p < k; ++p)
        for (size_t i = 0; i < m; ++i) {
            double us = r.S[p] * r.U[i * ku + p];
            for (size_t j = 0; j < n; ++j) out[i * n + j] += us * r.V[j * kv + p];
        }
    return out;
}

}  // namespace fv
