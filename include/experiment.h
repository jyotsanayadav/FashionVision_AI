#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fv {

// Records hyperparameters (chosen by us) and the learned-parameter count,
// then saves <name>_config.json and <name>_history.csv.
class ExperimentLog {
public:
    explicit ExperimentLog(std::string name) : name_(std::move(name)) {}

    void hparam(const std::string& key, const std::string& value) { hparams_.emplace_back(key, value); }
    void hparam(const std::string& key, const char* value) { hparams_.emplace_back(key, std::string(value)); }
    void hparam(const std::string& key, double value) {
        std::ostringstream ss;
        ss << value;
        hparams_.emplace_back(key, ss.str());
    }
    void param_count(size_t n) { num_params_ = n; }
    void epoch(int e, double loss, double metric) { history_.push_back({e, loss, metric}); }

    bool save(const std::string& dir) const {
        std::ofstream j(dir + "/" + name_ + "_config.json");
        std::ofstream c(dir + "/" + name_ + "_history.csv");
        if (!j || !c) return false;

        j << "{\n  \"experiment\": \"" << name_ << "\",\n  \"hyperparameters\": {\n";
        for (size_t i = 0; i < hparams_.size(); ++i)
            j << "    \"" << hparams_[i].first << "\": \"" << hparams_[i].second << "\""
              << (i + 1 < hparams_.size() ? "," : "") << "\n";
        j << "  },\n  \"num_learned_parameters\": " << num_params_ << ",\n";
        if (!history_.empty())
            j << "  \"final_loss\": " << history_.back().loss << ",\n"
              << "  \"final_metric\": " << history_.back().metric << "\n";
        else
            j << "  \"final_loss\": null\n";
        j << "}\n";

        c << "epoch,loss,metric\n";
        for (const auto& h : history_) c << h.epoch << "," << h.loss << "," << h.metric << "\n";
        return true;
    }

private:
    struct Row { int epoch; double loss; double metric; };
    std::string name_;
    std::vector<std::pair<std::string, std::string>> hparams_;
    size_t num_params_ = 0;
    std::vector<Row> history_;
};

}  // namespace fv
