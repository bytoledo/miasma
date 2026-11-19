#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace seird {

struct TelemetryMetrics {
    SimStats stats;
    float rt             = 0.0f;  // Número reproductivo efectivo R_t (puntual)
    float rt_p05         = 0.0f;  // Percentil 5% bootstrap OpenMP de R_t
    float rt_p95         = 0.0f;  // Percentil 95% bootstrap OpenMP de R_t
    float prob_rt_gt_1   = 0.0f;  // Probabilidad P(R_t > 1) (estándar CDC CFA)
    float r0_ngm         = 0.0f;  // R_0 estructural vía Matriz de Siguiente Generación (NGM)
    float z_score        = 0.0f;  // Z-score de anomalía / brote
    float new_cases_rate = 0.0f;  // Tasa exacta de nuevos contagios (S -> E) por día
    float cfr            = 0.0f;  // Case Fatality Rate: D / (R + D)
    bool  outbreak_alert = false; // true si anomalía (>2 sigma) y rt > 1.0
};

struct OdeState {
    double s   = 0.0;
    double e   = 0.0;
    double i   = 0.0;
    double r   = 0.0;
    double d   = 0.0;
    double day = 0.0;
};

// Avanza un paso dt del sistema ODE SEIRDS homogéneo mediante Runge-Kutta de 4.º orden (RK4)
OdeState ode_seirds_step_rk4(const OdeState& st, double beta_eff, const SimParams& params, double dt);

// Integra el modelo ODE SEIRDS determinista de referencia durante total_days días
std::vector<OdeState> ode_seirds_simulate(
    double total_pop,
    double initial_infected,
    double beta_eff,
    const SimParams& params,
    double total_days,
    double dt = 1.0 / 48.0
);

// Estima la tasa de transmisión homogénea equivalente beta_eff a partir del disco de infección del ABM
double compute_abm_homogeneous_beta(int num_agents, const SimParams& params);

// Estima la tasa de transmisión cinética 2D considerando velocidad finita, cuarentena y superdispersión
double compute_abm_kinetic_beta(int num_agents, const SimParams& params);

// Calcula R_0 estructural mediante el radio espectral de la Matriz de Siguiente Generación (3x3 RiskGroup)
double compute_r0_ngm(int num_agents, const SimParams& params);

// Fracción de pico infectado I_max / N analítica del modelo SIR clásico para un R_0 dado
double sir_analytical_peak_fraction(double r0);

// Tasa de ataque final analítica 1 - S(inf)/N del modelo SIR clásico para un R_0 dado
double sir_analytical_final_attack_rate(double r0);

class CpuTelemetryEngine {
public:
    explicit CpuTelemetryEngine(std::size_t window_size = 64);

    // Registra un nuevo SimStats y calcula R_t (con bootstrap OpenMP), R_0 NGM, Z-score y métricas
    TelemetryMetrics update(const SimStats& stats, const SimParams& params);

    // Reinicia el historial de telemetría
    void reset();

    const std::vector<TelemetryMetrics>& history() const { return history_; }

private:
    std::size_t window_size_;
    uint64_t    update_count_ = 0;
    std::vector<SimStats>         stats_window_;
    std::vector<float>            incidence_window_;
    std::vector<TelemetryMetrics> history_;
};

struct EnsembleMetrics {
    int   num_replicas   = 1;
    float sim_day        = 0.0f;
    float i_p05          = 0.0f; // Fracción de infectados P5
    float i_p50          = 0.0f; // Fracción de infectados Mediana (P50)
    float i_p95          = 0.0f; // Fracción de infectados P95
    float rt_p05         = 0.0f;
    float rt_p50         = 0.0f;
    float rt_p95         = 0.0f;
    float prob_rt_gt_1   = 0.0f; // Fracción empírica de réplicas con R_t > 1
    float extinction_pct = 0.0f; // % de réplicas con extinción (E + I == 0)
};

class CpuEnsembleEngine {
public:
    explicit CpuEnsembleEngine(int num_replicas = 1, std::size_t window_size = 64);

    void reset(int num_replicas);

    // Actualiza las K réplicas con OpenMP y calcula percentiles P5/P50/P95, P(Rt>1) y extinción
    EnsembleMetrics update(const std::vector<SimStats>& rep_stats, const SimParams& params, int agents_per_replica);

    const std::vector<EnsembleMetrics>& history() const { return history_; }

private:
    std::size_t window_size_;
    std::vector<CpuTelemetryEngine> engines_;
    std::vector<EnsembleMetrics>    history_;
};

} // namespace seird
