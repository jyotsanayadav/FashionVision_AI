#include "tensor.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace fv {

static size_t product(const std::vector<size_t>& s) {
    size_t p = 1;
    for (size_t d : s) p *= d;
    return p;
}

static std::vector<size_t> strides_of(const std::vector<size_t>& s) {
    std::vector<size_t> st(s.size(), 1);
    for (size_t i = s.size(); i-- > 1;) st[i - 1] = st[i] * s[i];
    return st;
}

Tensor::Tensor(std::vector<size_t> shape, double fill)
    : shape_(std::move(shape)), data_(product(shape_), fill) {}

Tensor::Tensor(std::vector<size_t> shape, std::vector<double> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
    if (data_.size() != product(shape_))
        throw std::invalid_argument("Tensor: data size does not match shape");
}

size_t Tensor::flat_index(const std::vector<size_t>& idx) const {
    if (idx.size() != shape_.size())
        throw std::out_of_range("Tensor: wrong number of indices");
    auto st = strides_of(shape_);
    size_t flat = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= shape_[i]) throw std::out_of_range("Tensor: index out of range");
        flat += idx[i] * st[i];
    }
    return flat;
}

double& Tensor::at(const std::vector<size_t>& idx) { return data_[flat_index(idx)]; }
double Tensor::at(const std::vector<size_t>& idx) const { return data_[flat_index(idx)]; }

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
    if (product(new_shape) != size())
        throw std::invalid_argument("reshape: element count must stay the same");
    return Tensor(std::move(new_shape), data_);
}

Tensor Tensor::transpose() const {
    if (ndim() != 2) throw std::invalid_argument("transpose: only 2D supported");
    size_t r = shape_[0], c = shape_[1];
    Tensor out({c, r});
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j) out.data_[j * r + i] = data_[i * c + j];
    return out;
}

std::vector<size_t> Tensor::broadcast_shape(const std::vector<size_t>& a,
                                            const std::vector<size_t>& b) {
    size_t n = std::max(a.size(), b.size());
    std::vector<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        size_t da = (i < n - a.size()) ? 1 : a[i - (n - a.size())];
        size_t db = (i < n - b.size()) ? 1 : b[i - (n - b.size())];
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("broadcast: incompatible shapes");
        out[i] = (da == 1) ? db : da;
    }
    return out;
}

template <typename F>
Tensor Tensor::binary_op(const Tensor& o, F f) const {
    auto out_shape = broadcast_shape(shape_, o.shape_);
    Tensor out(out_shape);
    size_t nd = out_shape.size();

    auto eff_strides = [&](const Tensor& t) {
        std::vector<size_t> st(nd, 0);
        auto ts = strides_of(t.shape_);
        size_t off = nd - t.ndim();
        for (size_t i = 0; i < t.ndim(); ++i) st[off + i] = (t.shape_[i] == 1) ? 0 : ts[i];
        return st;
    };
    auto sa = eff_strides(*this);
    auto sb = eff_strides(o);

    std::vector<size_t> idx(nd, 0);
    for (size_t flat = 0; flat < out.size(); ++flat) {
        size_t ia = 0, ib = 0;
        for (size_t d = 0; d < nd; ++d) {
            ia += idx[d] * sa[d];
            ib += idx[d] * sb[d];
        }
        out.data_[flat] = f(data_[ia], o.data_[ib]);
        for (size_t d = nd; d-- > 0;) {
            if (++idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
    return out;
}

Tensor Tensor::operator+(const Tensor& o) const { return binary_op(o, [](double x, double y) { return x + y; }); }
Tensor Tensor::operator-(const Tensor& o) const { return binary_op(o, [](double x, double y) { return x - y; }); }
Tensor Tensor::operator*(const Tensor& o) const { return binary_op(o, [](double x, double y) { return x * y; }); }
Tensor Tensor::operator/(const Tensor& o) const { return binary_op(o, [](double x, double y) { return x / y; }); }
Tensor Tensor::operator*(double s) const {
    Tensor out = *this;
    for (double& v : out.data_) v *= s;
    return out;
}

double Tensor::sum() const {
    double s = 0.0;
    for (double v : data_) s += v;
    return s;
}

double Tensor::dot(const Tensor& o) const {
    if (size() != o.size()) throw std::invalid_argument("dot: sizes differ");
    double s = 0.0;
    for (size_t i = 0; i < data_.size(); ++i) s += data_[i] * o.data_[i];
    return s;
}

Tensor Tensor::matmul(const Tensor& o) const {
    if (ndim() != 2 || o.ndim() != 2 || shape_[1] != o.shape_[0])
        throw std::invalid_argument("matmul: need (m,k) x (k,n)");
    size_t m = shape_[0], k = shape_[1], n = o.shape_[1];
    Tensor out({m, n});
    for (size_t i = 0; i < m; ++i)
        for (size_t p = 0; p < k; ++p) {
            double a = data_[i * k + p];
            for (size_t j = 0; j < n; ++j) out.data_[i * n + j] += a * o.data_[p * n + j];
        }
    return out;
}

std::string Tensor::to_string() const {
    std::ostringstream ss;
    ss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) ss << (i ? "," : "") << shape_[i];
    ss << "], data=[";
    for (size_t i = 0; i < data_.size(); ++i) ss << (i ? ", " : "") << data_[i];
    ss << "])";
    return ss.str();
}

}  // namespace fv
