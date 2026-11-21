#include "dataset.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace fv {

static uint32_t read_be32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("IDX: unexpected end of file");
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}

Dataset load_idx(const std::string& images_path, const std::string& labels_path, size_t max_n) {
    std::ifstream fi(images_path, std::ios::binary), fl(labels_path, std::ios::binary);
    if (!fi) throw std::runtime_error("cannot open " + images_path);
    if (!fl) throw std::runtime_error("cannot open " + labels_path);

    if (read_be32(fi) != 2051) throw std::runtime_error("bad magic number in " + images_path);
    size_t n = read_be32(fi);
    size_t rows = read_be32(fi), cols = read_be32(fi);
    if (read_be32(fl) != 2049) throw std::runtime_error("bad magic number in " + labels_path);
    size_t nl = read_be32(fl);
    if (n != nl) throw std::runtime_error("image and label counts differ");
    if (max_n > 0 && max_n < n) n = max_n;

    const size_t feat = rows * cols;
    Dataset d;
    d.X = Tensor({n, feat});
    d.y.resize(n);

    std::vector<unsigned char> buf(feat);
    for (size_t i = 0; i < n; ++i) {
        fi.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(feat));
        if (!fi) throw std::runtime_error("IDX: truncated image data");
        for (size_t j = 0; j < feat; ++j) d.X[i * feat + j] = buf[j] / 255.0;
        char lab;
        fl.read(&lab, 1);
        if (!fl) throw std::runtime_error("IDX: truncated label data");
        d.y[i] = static_cast<unsigned char>(lab);
    }
    return d;
}

Dataset subset(const Dataset& d, const std::vector<size_t>& idx) {
    const size_t f = d.features();
    Dataset out;
    out.X = Tensor({idx.size(), f});
    out.y.resize(idx.size());
    for (size_t k = 0; k < idx.size(); ++k) {
        for (size_t j = 0; j < f; ++j) out.X[k * f + j] = d.X[idx[k] * f + j];
        out.y[k] = d.y[idx[k]];
    }
    return out;
}

static uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::vector<size_t> shuffled_indices(size_t n, uint64_t seed) {
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    uint64_t s = seed;
    for (size_t i = n; i > 1; --i) std::swap(idx[i - 1], idx[splitmix64(s) % i]);  // Fisher-Yates
    return idx;
}

void split_stratified(const Dataset& d, double val_fraction, uint64_t seed,
                      Dataset& train, Dataset& val) {
    int max_label = 0;
    for (int l : d.y) max_label = std::max(max_label, l);
    std::vector<std::vector<size_t>> by_class(static_cast<size_t>(max_label) + 1);
    for (size_t i = 0; i < d.size(); ++i) by_class[static_cast<size_t>(d.y[i])].push_back(i);

    std::vector<size_t> tr, va;
    for (size_t c = 0; c < by_class.size(); ++c) {
        auto order = shuffled_indices(by_class[c].size(), seed + c);
        size_t nval = static_cast<size_t>(std::llround(val_fraction * static_cast<double>(order.size())));
        for (size_t k = 0; k < order.size(); ++k)
            (k < nval ? va : tr).push_back(by_class[c][order[k]]);
    }
    auto mix = [&](std::vector<size_t>& v, uint64_t sd) {
        auto p = shuffled_indices(v.size(), sd);
        std::vector<size_t> out(v.size());
        for (size_t i = 0; i < v.size(); ++i) out[i] = v[p[i]];
        v = out;
    };
    mix(tr, seed + 100);
    mix(va, seed + 200);
    train = subset(d, tr);
    val = subset(d, va);
}

static uint64_t row_hash(const Dataset& d, size_t i) {
    uint64_t h = 1469598103934665603ULL;
    const size_t f = d.features();
    for (size_t j = 0; j < f; ++j) {
        h ^= static_cast<uint64_t>(std::llround(d.X[i * f + j] * 255.0));
        h *= 1099511628211ULL;
    }
    return h;
}

size_t remove_duplicates(Dataset& d) {
    std::unordered_set<uint64_t> seen;
    std::vector<size_t> keep;
    for (size_t i = 0; i < d.size(); ++i)
        if (seen.insert(row_hash(d, i)).second) keep.push_back(i);
    size_t removed = d.size() - keep.size();
    if (removed > 0) d = subset(d, keep);
    return removed;
}

size_t remove_overlap(Dataset& d, const Dataset& other) {
    std::unordered_set<uint64_t> bad;
    for (size_t i = 0; i < other.size(); ++i) bad.insert(row_hash(other, i));
    std::vector<size_t> keep;
    for (size_t i = 0; i < d.size(); ++i)
        if (!bad.count(row_hash(d, i))) keep.push_back(i);
    size_t removed = d.size() - keep.size();
    if (removed > 0) d = subset(d, keep);
    return removed;
}

std::vector<size_t> class_counts(const Dataset& d, size_t classes) {
    std::vector<size_t> c(classes, 0);
    for (int l : d.y) ++c[static_cast<size_t>(l)];
    return c;
}

}  // namespace fv
