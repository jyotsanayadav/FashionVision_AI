#include "viz.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fv {

static std::string color_for(double t) {
    t = std::max(-1.0, std::min(1.0, t));
    int r = 255, g = 255, b = 255;
    if (t >= 0) g = b = static_cast<int>(255 * (1.0 - t));
    else        r = g = static_cast<int>(255 * (1.0 + t));
    std::ostringstream ss;
    ss << "rgb(" << r << "," << g << "," << b << ")";
    return ss.str();
}

bool write_heatmap_svg(const Tensor& m, const std::string& path,
                       const std::string& title, bool show_values) {
    if (m.ndim() != 2 || m.size() == 0) return false;
    const size_t rows = m.shape()[0], cols = m.shape()[1];

    double vmax = 0.0;
    for (size_t i = 0; i < m.size(); ++i) vmax = std::max(vmax, std::abs(m[i]));
    if (vmax == 0.0) vmax = 1.0;

    const int cell = static_cast<int>(std::min<size_t>(48, std::max<size_t>(6, 480 / std::max(rows, cols))));
    const int margin = 12, top = 36, bottom = 24;
    int w = static_cast<int>(cols) * cell + 2 * margin;
    int h = static_cast<int>(rows) * cell + top + bottom;
    if (w < 260) w = 260;

    std::ofstream out(path);
    if (!out) return false;
    out << std::fixed << std::setprecision(2);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << w << "\" height=\"" << h
        << "\" font-family=\"sans-serif\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<text x=\"" << margin << "\" y=\"22\" font-size=\"15\" font-weight=\"bold\">"
        << title << "</text>\n";

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double v = m[i * cols + j];
            int x = margin + static_cast<int>(j) * cell;
            int y = top + static_cast<int>(i) * cell;
            out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << cell
                << "\" height=\"" << cell << "\" fill=\"" << color_for(v / vmax) << "\"/>\n";
            if (show_values) {
                out << "<text x=\"" << x + cell / 2 << "\" y=\"" << y + cell / 2 + 4
                    << "\" font-size=\"" << cell * 0.3 << "\" text-anchor=\"middle\" fill=\"#222\">"
                    << v << "</text>\n";
            }
        }
    }
    out << "<text x=\"" << margin << "\" y=\"" << h - 7
        << "\" font-size=\"11\" fill=\"#555\">blue = negative, red = positive, scale +/-"
        << vmax << "</text>\n";
    out << "</svg>\n";
    return true;
}

bool write_bar_svg(const std::vector<double>& v, const std::string& path,
                   const std::string& title, const std::string& xlabel) {
    if (v.empty()) return false;
    double vmax = *std::max_element(v.begin(), v.end());
    if (vmax <= 0.0) vmax = 1.0;

    const int W = 520, H = 300, L = 55, R = 20, T = 40, B = 50;
    const double plot_w = W - L - R, plot_h = H - T - B;
    const double slot = plot_w / static_cast<double>(v.size());
    const double bw = slot * 0.7;

    std::ofstream out(path);
    if (!out) return false;
    out << std::fixed << std::setprecision(2);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H
        << "\" font-family=\"sans-serif\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<text x=\"" << L << "\" y=\"24\" font-size=\"15\" font-weight=\"bold\">" << title << "</text>\n";
    out << "<line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L << "\" y2=\"" << T + plot_h
        << "\" stroke=\"#333\"/>\n";
    out << "<line x1=\"" << L << "\" y1=\"" << T + plot_h << "\" x2=\"" << L + plot_w << "\" y2=\""
        << T + plot_h << "\" stroke=\"#333\"/>\n";
    out << "<text x=\"" << L - 6 << "\" y=\"" << T + plot_h + 4
        << "\" font-size=\"11\" text-anchor=\"end\">0</text>\n";
    out << "<text x=\"" << L - 6 << "\" y=\"" << T + 4
        << "\" font-size=\"11\" text-anchor=\"end\">" << vmax << "</text>\n";

    for (size_t i = 0; i < v.size(); ++i) {
        double bh = plot_h * std::max(v[i], 0.0) / vmax;
        double x = L + static_cast<double>(i) * slot + (slot - bw) / 2.0;
        double y = T + plot_h - bh;
        out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << bw << "\" height=\"" << bh
            << "\" fill=\"#3b82f6\"/>\n";
        if (v.size() <= 16)
            out << "<text x=\"" << x + bw / 2 << "\" y=\"" << y - 4
                << "\" font-size=\"10\" text-anchor=\"middle\">" << v[i] << "</text>\n";
        if (v.size() <= 24 || i % 5 == 0)
            out << "<text x=\"" << x + bw / 2 << "\" y=\"" << T + plot_h + 16
                << "\" font-size=\"11\" text-anchor=\"middle\">" << i + 1 << "</text>\n";
    }
    out << "<text x=\"" << W / 2 << "\" y=\"" << H - 10
        << "\" font-size=\"12\" text-anchor=\"middle\">" << xlabel << "</text>\n";
    out << "</svg>\n";
    return true;
}

bool write_line_svg(const std::vector<double>& v, const std::string& path,
                    const std::string& title, const std::string& xlabel,
                    const std::string& ylabel) {
    if (v.size() < 2) return false;
    double vmin = *std::min_element(v.begin(), v.end());
    double vmax = *std::max_element(v.begin(), v.end());
    if (vmax - vmin < 1e-12) vmax = vmin + 1.0;

    const int W = 560, H = 320, L = 60, R = 20, T = 40, B = 50;
    const double pw = W - L - R, ph = H - T - B;

    std::ofstream out(path);
    if (!out) return false;
    out << std::fixed << std::setprecision(3);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H
        << "\" font-family=\"sans-serif\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<text x=\"" << L << "\" y=\"22\" font-size=\"15\" font-weight=\"bold\">" << title << "</text>\n";
    out << "<text x=\"" << L << "\" y=\"" << T - 8 << "\" font-size=\"11\" fill=\"#555\">" << ylabel << "</text>\n";
    out << "<line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L << "\" y2=\"" << T + ph << "\" stroke=\"#333\"/>\n";
    out << "<line x1=\"" << L << "\" y1=\"" << T + ph << "\" x2=\"" << L + pw << "\" y2=\"" << T + ph << "\" stroke=\"#333\"/>\n";
    out << "<text x=\"" << L - 6 << "\" y=\"" << T + ph + 4 << "\" font-size=\"11\" text-anchor=\"end\">" << vmin << "</text>\n";
    out << "<text x=\"" << L - 6 << "\" y=\"" << T + 4 << "\" font-size=\"11\" text-anchor=\"end\">" << vmax << "</text>\n";

    out << "<polyline fill=\"none\" stroke=\"#ef4444\" stroke-width=\"2\" points=\"";
    for (size_t i = 0; i < v.size(); ++i) {
        double x = L + pw * static_cast<double>(i) / static_cast<double>(v.size() - 1);
        double y = T + ph - ph * (v[i] - vmin) / (vmax - vmin);
        out << x << "," << y << " ";
    }
    out << "\"/>\n";
    out << "<text x=\"" << W / 2 << "\" y=\"" << H - 10 << "\" font-size=\"12\" text-anchor=\"middle\">" << xlabel << "</text>\n";
    out << "</svg>\n";
    return true;
}

bool write_multiline_svg(const std::vector<std::string>& names,
                         const std::vector<std::vector<double>>& series,
                         const std::string& path, const std::string& title,
                         const std::string& xlabel, const std::string& ylabel) {
    if (series.empty() || names.size() != series.size()) return false;
    double vmin = 1e300, vmax = -1e300;
    size_t len = 0;
    for (const auto& s : series) {
        len = std::max(len, s.size());
        for (double x : s) { vmin = std::min(vmin, x); vmax = std::max(vmax, x); }
    }
    if (len < 2) return false;
    if (vmax - vmin < 1e-12) vmax = vmin + 1.0;

    static const char* colors[] = {"#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#6b7280"};
    const int W = 620, H = 340, L = 60, R = 20, T = 50, B = 50;
    const double pw = W - L - R, ph = H - T - B;

    std::ofstream out(path);
    if (!out) return false;
    out << std::fixed << std::setprecision(3);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H
        << "\" font-family=\"sans-serif\">\n<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<text x=\"" << L << "\" y=\"22\" font-size=\"15\" font-weight=\"bold\">" << title << "</text>\n";
    out << "<text x=\"" << L << "\" y=\"" << T - 8 << "\" font-size=\"11\" fill=\"#555\">" << ylabel << "</text>\n";
    out << "<line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L << "\" y2=\"" << T + ph << "\" stroke=\"#333\"/>\n";
    out << "<line x1=\"" << L << "\" y1=\"" << T + ph << "\" x2=\"" << L + pw << "\" y2=\"" << T + ph << "\" stroke=\"#333\"/>\n";
    out << "<text x=\"" << L - 6 << "\" y=\"" << T + ph + 4 << "\" font-size=\"11\" text-anchor=\"end\">" << vmin << "</text>\n";
    out << "<text x=\"" << L - 6 << "\" y=\"" << T + 4 << "\" font-size=\"11\" text-anchor=\"end\">" << vmax << "</text>\n";

    for (size_t k = 0; k < series.size(); ++k) {
        const char* col = colors[k % 6];
        out << "<polyline fill=\"none\" stroke=\"" << col << "\" stroke-width=\"2\" points=\"";
        for (size_t i = 0; i < series[k].size(); ++i) {
            double x = L + pw * static_cast<double>(i) / static_cast<double>(len - 1);
            double y = T + ph - ph * (series[k][i] - vmin) / (vmax - vmin);
            out << x << "," << y << " ";
        }
        out << "\"/>\n";
        const double lx = L + 10 + 140.0 * static_cast<double>(k % 4);
        const double row = (k >= 4) ? 14.0 : 0.0;
        out << "<rect x=\"" << lx << "\" y=\"" << T + ph + 14 + row
            << "\" width=\"12\" height=\"4\" fill=\"" << col << "\"/>\n";
        out << "<text x=\"" << lx + 16 << "\" y=\"" << T + ph + 19 + row
            << "\" font-size=\"11\">" << names[k] << "</text>\n";
    }
    out << "<text x=\"" << W - R << "\" y=\"" << T - 8 << "\" font-size=\"11\" text-anchor=\"end\" fill=\"#555\">"
        << xlabel << "</text>\n</svg>\n";
    return true;
}

}  // namespace fv
