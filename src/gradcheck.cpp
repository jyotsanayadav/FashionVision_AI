#include "gradcheck.h"

#include <algorithm>
#include <stdexcept>
#include "linalg.h"

namespace fv {

Tensor numerical_gradient(const ScalarFn& f, Tensor x, double h) {
    Tensor g(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
        double orig = x[i];
        x[i] = orig + h;
        double fp = f(x);
        x[i] = orig - h;
        double fm = f(x);
        x[i] = orig;
        g[i] = (fp - fm) / (2.0 * h);
    }
    return g;
}

double relative_error(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::invalid_argument("relative_error: shapes differ");
    double den = frobenius_norm(a) + frobenius_norm(b);
    return frobenius_norm(a - b) / std::max(den, 1e-12);
}

double gradient_check(const ScalarFn& f, const Tensor& x, const Tensor& analytic, double h) {
    return relative_error(numerical_gradient(f, x, h), analytic);
}

}  // namespace fv
