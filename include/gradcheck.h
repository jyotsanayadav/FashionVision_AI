#pragma once
#include <functional>
#include "tensor.h"

namespace fv {

using ScalarFn = std::function<double(const Tensor&)>;

Tensor numerical_gradient(const ScalarFn& f, Tensor x, double h = 1e-5);
double relative_error(const Tensor& a, const Tensor& b);
double gradient_check(const ScalarFn& f, const Tensor& x, const Tensor& analytic, double h = 1e-5);

}  // namespace fv
