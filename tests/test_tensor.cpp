#include <cmath>
#include <iostream>
#include "tensor.h"

using fv::Tensor;

static int failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cout << "FAIL line " << __LINE__ << ": " #cond << "\n";   \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

static bool throws(void (*f)()) {
    try { f(); } catch (...) { return true; }
    return false;
}

int main() {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3}, {10, 20, 30});
    Tensor c = a + b;
    CHECK(c.shape() == std::vector<size_t>({2, 3}));
    CHECK(c.at({0, 0}) == 11 && c.at({1, 2}) == 36);

    Tensor col({3, 1}, {1, 2, 3});
    Tensor row({1, 4}, {1, 2, 3, 4});
    Tensor outer = col * row;
    CHECK(outer.shape() == std::vector<size_t>({3, 4}));
    CHECK(outer.at({2, 3}) == 12);

    Tensor s(std::vector<size_t>{}, std::vector<double>{2.0});
    CHECK((a * s).at({1, 1}) == 10);

    Tensor r = a.reshape({3, 2});
    CHECK(r.at({2, 1}) == 6);

    Tensor t = a.transpose();
    CHECK(t.shape() == std::vector<size_t>({3, 2}));
    CHECK(t.at({2, 0}) == 3 && t.at({0, 1}) == 4);

    Tensor u({3}, {1, 2, 3}), v({3}, {4, 5, 6});
    CHECK(std::abs(u.dot(v) - 32.0) < 1e-12);

    Tensor m = a.matmul(r);
    CHECK(m.shape() == std::vector<size_t>({2, 2}));
    CHECK(m.at({0, 0}) == 22 && m.at({1, 1}) == 64);

    CHECK(throws([] { Tensor({2}, {1, 2}) + Tensor({3}, {1, 2, 3}); }));
    CHECK(throws([] { Tensor({2, 3}).reshape({4, 2}); }));
    CHECK(throws([] { Tensor({2, 3}).matmul(Tensor({2, 3})); }));

    if (failures == 0) std::cout << "All tensor tests passed\n";
    return failures == 0 ? 0 : 1;
}
