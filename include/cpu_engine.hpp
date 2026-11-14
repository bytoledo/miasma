#pragma once

#include "types.hpp"

#include <cstddef>
#include <vector>

namespace seird {

struct TelemetryMetrics {
    SimStats stats;
    float rt             = 0.0f;  // Número reproductivo efectivo R_t
    float z_score        = 0.0f;  // Z-score de anomalía / brote
    float new_cases_rate = 0.0f;  // Tasa de nuevos casos por día
    float cfr            = 0.0f;  // Case Fatality Rate: D / (R + D)
    bool  outbreak_alert = false; // true si z_score > 2.0 y rt > 1.0
};

class CpuTelemetryEngine {
public:
    explicit CpuTelemetryEngine(std::size_t window_size = 64);

    // Registra un nuevo SimStats y calcula R_t, Z-score y métricas con OpenMP
    TelemetryMetrics update(const SimStats& stats, const SimParams& params);

    // Reinicia el historial de telemetría
    void reset();

    const std::vector<TelemetryMetrics>& history() const { return history_; }

private:
    std::size_t window_size_;
    std::vector<SimStats>         stats_window_;
    std::vector<float>            incidence_window_;
    std::vector<TelemetryMetrics> history_;
};

} // namespace seird
