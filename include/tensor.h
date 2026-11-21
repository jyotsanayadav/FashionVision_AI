#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace fv {

class Tensor {
public:
    Tensor() : shape_{}, data_(1, 0.0) {}
    explicit Tensor(std::vector<size_t> shape, double fill = 0.0);
    Tensor(std::vector<size_t> shape, std::vector<double> data);

    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return data_.size(); }

    double& operator[](size_t flat) { return data_[flat]; }
    double operator[](size_t flat) const { return data_[flat]; }
    double& at(const std::vector<size_t>& idx);
    double at(const std::vector<size_t>& idx) const;

    Tensor reshape(std::vector<size_t> new_shape) const;
    Tensor transpose() const;  // 2D only

    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                               const std::vector<size_t>& b);
    Tensor operator+(const Tensor& o) const;
    Tensor operator-(const Tensor& o) const;
    Tensor operator*(const Tensor& o) const;  // element-wise
    Tensor operator/(const Tensor& o) const;
    Tensor operator*(double s) const;

    double sum() const;
    double dot(const Tensor& o) const;
    Tensor matmul(const Tensor& o) const;

    std::string to_string() const;

private:
    template <typename F>
    Tensor binary_op(const Tensor& o, F f) const;
    size_t flat_index(const std::vector<size_t>& idx) const;

    std::vector<size_t> shape_;
    std::vector<double> data_;
};

}  // namespace fv
