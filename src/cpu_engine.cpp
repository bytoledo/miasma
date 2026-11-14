#include "cpu_engine.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace seird {

CpuTelemetryEngine::CpuTelemetryEngine(std::size_t window_size)
    : window_size_(std::max<std::size_t>(window_size, 8)) {}

void CpuTelemetryEngine::reset() {
    stats_window_.clear();
    incidence_window_.clear();
    history_.clear();
}

TelemetryMetrics CpuTelemetryEngine::update(const SimStats& stats, const SimParams& params) {
    TelemetryMetrics m;
    m.stats = stats;

    // Calcular tasa de nuevos casos (expuestos + infectados respecto al paso previo)
    float incidence = 0.0f;
    if (!stats_window_.empty()) {
        const SimStats& prev = stats_window_.back();
        // Nuevos contagios aproximados: S_{t-1} - S_t (siempre >= 0 salvo reinicio)
        int64_t delta_s = static_cast<int64_t>(prev.count_s) - static_cast<int64_t>(stats.count_s);
        float elapsed_step = stats.sim_day - prev.sim_day;
        float dt_safe      = std::max(elapsed_step > 1e-4f ? elapsed_step : params.dt, 1e-4f);
        incidence          = static_cast<float>(std::max<int64_t>(delta_s, 0)) / dt_safe;
    }
    m.new_cases_rate = incidence;

    stats_window_.push_back(stats);
    incidence_window_.push_back(incidence);

    if (stats_window_.size() > window_size_) {
        stats_window_.erase(stats_window_.begin());
        incidence_window_.erase(incidence_window_.begin());
    }

    const int n = static_cast<int>(incidence_window_.size());

    // Cálculo de media y desviación estándar para Z-score (ventana <= 64 elementos)
    if (n >= 2) {
        double sum    = 0.0;
        double sum_sq = 0.0;

        for (int i = 0; i < n; ++i) {
            double val = static_cast<double>(incidence_window_[i]);
            sum    += val;
            sum_sq += val * val;
        }

        double mean     = sum / static_cast<double>(n);
        double variance = std::max(0.0, (sum_sq / static_cast<double>(n)) - (mean * mean));
        double stddev   = std::sqrt(variance);

        if (stddev > 1e-5) {
            m.z_score = static_cast<float>((static_cast<double>(incidence) - mean) / stddev);
        } else {
            m.z_score = 0.0f;
        }
    }

    // Estimación de R_t mediante tasa de crecimiento exponencial r y tiempo generacional T_g:
    // R_t = 1 + r * T_g, donde T_g ≈ incubation_days + 0.5 * infectious_days
    if (n >= 4) {
        int lag = std::min(n - 1, 10);
        const SimStats& past = stats_window_[n - 1 - lag];

        double active_now  = static_cast<double>(stats.count_e + stats.count_i);
        double active_past = static_cast<double>(past.count_e + past.count_i);
        double elapsed     = static_cast<double>(stats.sim_day - past.sim_day);
        if (elapsed <= 1e-4) {
            elapsed = static_cast<double>(lag) * static_cast<double>(std::max(params.dt, 1e-4f));
        }

        if (active_past > 0.0 && active_now > 0.0 && elapsed > 0.0) {
            double r   = (std::log(active_now) - std::log(active_past)) / elapsed;
            double t_g = static_cast<double>(params.incubation_days) +
                         0.5 * static_cast<double>(params.infectious_days);
            m.rt = static_cast<float>(std::max(0.0, 1.0 + r * t_g));
        } else if (active_now > 0.0 && active_past == 0.0) {
            m.rt = 1.5f;
        } else {
            m.rt = 0.0f;
        }
    }

    // Case Fatality Rate (CFR) acumulada
    uint64_t resolved = static_cast<uint64_t>(stats.count_r) + static_cast<uint64_t>(stats.count_d);
    if (resolved > 0) {
        m.cfr = static_cast<float>(static_cast<double>(stats.count_d) / static_cast<double>(resolved));
    }

    m.outbreak_alert = ((m.z_score > 2.0f) || (stats.max_cell_z > 2.0f)) && (m.rt > 1.0f);

    history_.push_back(m);
    if (history_.size() > 2048) {
        history_.erase(history_.begin());
    }

    return m;
}

} // namespace seird
