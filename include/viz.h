#pragma once
#include <string>
#include <vector>
#include "tensor.h"

namespace fv {

// Blue = negative, white = 0, red = positive.
bool write_heatmap_svg(const Tensor& m, const std::string& path,
                       const std::string& title, bool show_values = false);

bool write_bar_svg(const std::vector<double>& values, const std::string& path,
                   const std::string& title, const std::string& xlabel = "index");

bool write_line_svg(const std::vector<double>& values, const std::string& path,
                    const std::string& title, const std::string& xlabel,
                    const std::string& ylabel);

bool write_multiline_svg(const std::vector<std::string>& names,
                         const std::vector<std::vector<double>>& series,
                         const std::string& path, const std::string& title,
                         const std::string& xlabel, const std::string& ylabel);

}  // namespace fv
