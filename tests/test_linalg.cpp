#include <algorithm>
#include <cmath>
#include <iostream>
#include "gradcheck.h"
#include "linalg.h"

using namespace fv;

static int failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cout << "FAIL line " << __LINE__ << ": " #cond << "\n";   \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static void check_svd(const Tensor& A, bool full_rank = true) {
    SVDResult r = svd(A);
    CHECK(max_abs_diff(reconstruct(r), A) < 1e-10);

    double sumsq = 0.0;
    for (size_t i = 0; i < r.S.size(); ++i) {
        CHECK(r.S[i] >= 0.0);
        if (i > 0) CHECK(r.S[i - 1] >= r.S[i]);
        sumsq += r.S[i] * r.S[i];
    }
    CHECK(std::abs(std::sqrt(sumsq) - frobenius_norm(A)) < 1e-10);

    if (full_rank) {
        Tensor GU = r.U.transpose().matmul(r.U);
        Tensor GV = r.V.transpose().matmul(r.V);
        CHECK(max_abs_diff(GU, identity(GU.shape()[0])) < 1e-10);
        CHECK(max_abs_diff(GV, identity(GV.shape()[0])) < 1e-10);
    }
}

int main() {
    check_svd(random_uniform({5, 3}, 1));
    check_svd(random_uniform({3, 5}, 2));
    check_svd(random_uniform({6, 6}, 3));
    check_svd(random_uniform({10, 4}, 4));

    {
        SVDResult r = svd(Tensor({2, 2}, {1, 2, 3, 4}));
        CHECK(std::abs(r.S[0] - 5.464985704) < 1e-8);
        CHECK(std::abs(r.S[1] - 0.365966191) < 1e-8);
    }
    {
        SVDResult r = svd(Tensor({2, 2}, {3, 0, 0, 2}));
        CHECK(std::abs(r.S[0] - 3.0) < 1e-12 && std::abs(r.S[1] - 2.0) < 1e-12);
    }
    {
        Tensor A({3, 2}, {1, 2, 2, 4, 3, 6});
        check_svd(A, false);
        SVDResult r = svd(A);
        CHECK(r.S[1] < 1e-10);
        CHECK(std::abs(r.S[0] - std::sqrt(70.0)) < 1e-10);
    }
    {
        SVDResult r = svd(Tensor({3, 3}));
        CHECK(r.S[0] == 0.0);
    }
    {
        Tensor A = random_uniform({8, 8}, 5);
        SVDResult r = svd(A);
        double prev = frobenius_norm(A);
        for (size_t k = 1; k <= 8; ++k) {
            double e = frobenius_norm(A - reconstruct(r, k));
            CHECK(e <= prev + 1e-12);
            prev = e;
        }
        CHECK(prev < 1e-10);
    }

    {
        Tensor x({3}, {1.0, -2.0, 0.5});
        auto f = [](const Tensor& t) { return t.dot(t); };
        Tensor g = numerical_gradient(f, x);
        CHECK(relative_error(g, x * 2.0) < 1e-8);
    }
    {
        Tensor W = random_uniform({2, 3}, 7);
        Tensor x = random_uniform({3, 1}, 8);
        Tensor y = random_uniform({2, 1}, 9);
        auto loss = [&](const Tensor& Wt) {
            Tensor res = Wt.matmul(x) - y;
            return res.dot(res);
        };
        Tensor analytic = (W.matmul(x) - y).matmul(x.transpose()) * 2.0;
        CHECK(gradient_check(loss, W, analytic) < 1e-7);
        CHECK(gradient_check(loss, W, analytic * 1.1) > 1e-3);
    }

    CHECK(max_abs_diff(random_uniform({4, 4}, 11), random_uniform({4, 4}, 11)) == 0.0);

    if (failures == 0) std::cout << "All linalg and gradient tests passed\n";
    return failures == 0 ? 0 : 1;
}
