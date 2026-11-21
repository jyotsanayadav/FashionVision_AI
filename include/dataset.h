#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "tensor.h"

namespace fv {

struct Dataset {
    Tensor X;                // (N, features), pixel values scaled to [0,1]
    std::vector<int> y;      // class labels
    size_t size() const { return y.size(); }
    size_t features() const { return X.ndim() == 2 ? X.shape()[1] : 0; }
};

// Reads the IDX format used by MNIST / Fashion-MNIST. max_n = 0 means read everything.
Dataset load_idx(const std::string& images_path, const std::string& labels_path, size_t max_n = 0);

Dataset subset(const Dataset& d, const std::vector<size_t>& idx);
std::vector<size_t> shuffled_indices(size_t n, uint64_t seed);

void split_stratified(const Dataset& d, double val_fraction, uint64_t seed,
                      Dataset& train, Dataset& val);

size_t remove_duplicates(Dataset& d);
size_t remove_overlap(Dataset& d, const Dataset& other);

std::vector<size_t> class_counts(const Dataset& d, size_t classes);

}  // namespace fv
