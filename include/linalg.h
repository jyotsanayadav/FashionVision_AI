#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "tensor.h"

namespace fv {

// Thin SVD: A (m x n) = U * diag(S) * V^T, r = min(m, n)
struct SVDResult {
    Tensor U;
    Tensor S;
    Tensor V;
    int sweeps = 0;
};

SVDResult svd(const Tensor& A, double tol = 1e-12, int max_sweeps = 60);
Tensor reconstruct(const SVDResult& r, size_t rank = 0);
Tensor identity(size_t n);
double frobenius_norm(const Tensor& t);
Tensor random_uniform(std::vector<size_t> shape, uint64_t seed, double lo = -1.0, double hi = 1.0);

}  // namespace fv
