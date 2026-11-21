#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include "gradcheck.h"
#include "linalg.h"
#include "viz.h"

using namespace fv;

static void save_heatmap(const Tensor& t, const std::string& dir, const std::string& name,
                         const std::string& title) {
    std::string path = dir + "/" + name;
    if (write_heatmap_svg(t, path, title)) std::cout << "  wrote " << path << "\n";
    else std::cout << "  ERROR: could not write " << path << " (does the folder exist?)\n";
}

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "results";
    std::cout << std::fixed << std::setprecision(4);

    const size_t n = 16;
    Tensor noise = random_uniform({n, n}, 42, -0.05, 0.05);
    Tensor A({n, n});
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double u1 = std::sin(0.4 * static_cast<double>(i)), v1 = std::cos(0.3 * static_cast<double>(j));
            double u2 = (i < n / 2) ? 1.0 : -1.0, v2 = (j < n / 2) ? 1.0 : -1.0;
            A[i * n + j] = 2.0 * u1 * v1 + 1.0 * u2 * v2 + noise[i * n + j];
        }

    SVDResult r = svd(A);
    std::cout << "SVD of a 16x16 matrix finished in " << r.sweeps << " Jacobi sweeps\n";

    double total = 0.0;
    std::vector<double> sv;
    for (size_t i = 0; i < r.S.size(); ++i) {
        sv.push_back(r.S[i]);
        total += r.S[i] * r.S[i];
    }

    std::cout << "\nrank k | Frobenius error | energy kept\n";
    double cum = 0.0;
    for (size_t k = 1; k <= 8; ++k) {
        cum += r.S[k - 1] * r.S[k - 1];
        if (k == 1 || k == 2 || k == 4 || k == 8) {
            double err = frobenius_norm(A - reconstruct(r, k));
            std::cout << std::setw(6) << k << " | " << std::setw(15) << err << " | "
                      << std::setw(9) << 100.0 * cum / total << "%\n";
        }
    }

    std::cout << "\nWriting images:\n";
    save_heatmap(A, dir, "tensor_original.svg", "Original matrix A");
    save_heatmap(reconstruct(r, 1), dir, "tensor_rank1.svg", "Rank-1 approximation");
    save_heatmap(reconstruct(r, 2), dir, "tensor_rank2.svg", "Rank-2 approximation");
    save_heatmap(reconstruct(r, 4), dir, "tensor_rank4.svg", "Rank-4 approximation");
    save_heatmap(A - reconstruct(r, 2), dir, "tensor_residual_rank2.svg", "Residual A - rank2");
    std::string bar = dir + "/tensor_singular_values.svg";
    if (write_bar_svg(sv, bar, "Singular values of A", "component"))
        std::cout << "  wrote " << bar << "\n";

    Tensor W = random_uniform({2, 3}, 7);
    Tensor x = random_uniform({3, 1}, 8);
    Tensor y = random_uniform({2, 1}, 9);
    auto loss = [&](const Tensor& Wt) {
        Tensor res = Wt.matmul(x) - y;
        return res.dot(res);
    };
    Tensor analytic = (W.matmul(x) - y).matmul(x.transpose()) * 2.0;
    Tensor numeric = numerical_gradient(loss, W);

    std::cout << "\nGradient check for L(W) = ||Wx - y||^2\n";
    std::cout << "  analytic[0..2] : " << analytic[0] << " " << analytic[1] << " " << analytic[2] << "\n";
    std::cout << "  numeric [0..2] : " << numeric[0] << " " << numeric[1] << " " << numeric[2] << "\n";
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "  relative error : " << relative_error(numeric, analytic) << "\n";
    return 0;
}
