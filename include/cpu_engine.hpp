#pragma once

#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace seird {

enum RtCategory : uint8_t {
    RT_DECLINING        = 0, // P(Rt > 1) < 10%
    RT_LIKELY_DECLINING = 1, // 10% <= P(Rt > 1) < 25%
    RT_UNCERTAIN        = 2, // 25% <= P(Rt > 1) <= 75%
    RT_LIKELY_GROWING   = 3, // 75% <  P(Rt > 1) <= 90%
    RT_GROWING          = 4  // P(Rt > 1) > 90%
};

const char* rt_category_label(RtCategory cat);

struct TelemetryMetrics {
    SimStats stats;
    float      rt              = 0.0f;  // Número reproductivo efectivo R_t (puntual)
    float      rt_p05          = 0.0f;  // Percentil 5% bootstrap OpenMP de R_t
    float      rt_p95          = 0.0f;  // Percentil 95% bootstrap OpenMP de R_t
    float      prob_rt_gt_1    = 0.0f;  // Probabilidad P(R_t > 1) (estándar CDC CFA)
    RtCategory rt_category     = RT_UNCERTAIN; // Categoría operativa CDC CFA
    float      r0_ngm          = 0.0f;  // R_0 estructural vía Matriz de Siguiente Generación (NGM)
    float      z_score         = 0.0f;  // Z-score de anomalía / brote
    float      new_cases_rate  = 0.0f;  // Tasa exacta de nuevos contagios (S -> E) por día
    float      cfr             = 0.0f;  // Case Fatality Rate: D / (R + D)
    float      ifr             = 0.0f;  // Infection Fatality Rate: D / total_infectados_acumulados
    bool       outbreak_alert  = false; // true si anomalía (>2 sigma) y rt > 1.0
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
    uint64_t    update_count_  = 0;
    uint64_t    cum_infections_ = 0;
    std::vector<SimStats>         stats_window_;
    std::vector<float>            incidence_window_;
    std::vector<TelemetryMetrics> history_;
};

struct EnsembleMetrics {
    int   num_replicas   = 1;
    float sim_day        = 0.0f;
    float i_p05          = 0.0f; // Fracción de infectados P5
    float i_p10          = 0.0f; // Fracción de infectados P10
    float i_p50          = 0.0f; // Fracción de infectados Mediana (P50)
    float i_p90          = 0.0f; // Fracción de infectados P90
    float i_p95          = 0.0f; // Fracción de infectados P95
    float rt_p05         = 0.0f;
    float rt_p50         = 0.0f;
    float rt_p95         = 0.0f;
    float prob_rt_gt_1   = 0.0f; // Fracción empírica de réplicas con R_t > 1
    float extinction_pct = 0.0f; // % de réplicas con extinción (E + I == 0)
};

struct ForecastScoreMetrics {
    float mae         = 0.0f; // Mean Absolute Error (P50 vs referencia)
    float rmse        = 0.0f; // Root Mean Squared Error (P50 vs referencia)
    float is80        = 0.0f; // Interval Score al 80% ([P10, P90])
    float wis         = 0.0f; // Weighted Interval Score (mediana + IC80 + IC90)
    float coverage_80 = 0.0f; // Cobertura empírica [P10, P90] en [0, 1]
    float coverage_90 = 0.0f; // Cobertura empírica [P5, P95] en [0, 1]
};

// Evalúa la precisión puntual y probabilística de un ensamble frente a una serie observada
ForecastScoreMetrics compute_ensemble_forecast_scores(
    const std::vector<EnsembleMetrics>& ens_traj,
    const std::vector<float>& obs_i_frac
);

struct SensitivityRow {
    float beta              = 0.0f;
    float quarantine_factor = 1.0f;
    float superspreader_frac = 0.0f;
    float r0_ngm            = 0.0f;
    float peak_i_frac       = 0.0f;
    float final_attack_rate = 0.0f;
};

// Ejecuta un barrido paramétrico global en paralelo (OpenMP) para análisis de sensibilidad e identificabilidad
std::vector<SensitivityRow> run_global_sensitivity_sweep(
    int num_agents,
    int initial_infected,
    const SimParams& base_params,
    const std::vector<float>& betas,
    const std::vector<float>& quar_factors,
    const std::vector<float>& super_fracs,
    double horizon_days = 60.0
);

struct AbcParticle {
    float beta            = 0.25f;
    float incubation_days = 5.0f;
    float infectious_days = 10.0f;
    float weight          = 1.0f;
    float distance        = 0.0f;
};

struct AbcSmcConfig {
    int      num_particles   = 64;
    int      num_generations = 4;
    float    beta_min        = 0.05f, beta_max  = 0.60f;
    float    incub_min       = 2.0f,  incub_max = 10.0f;
    float    infec_min       = 4.0f,  infec_max = 18.0f;
    float    obs_noise_cv    = 0.03f;
    uint64_t seed            = 2026ULL;
};

struct AbcSmcResult {
    std::vector<AbcParticle> posterior;
    float beta_p05         = 0.0f;
    float beta_p50         = 0.0f;
    float beta_p95         = 0.0f;
    float incub_p50        = 0.0f;
    float infec_p50        = 0.0f;
    float final_epsilon    = 0.0f;
    ForecastScoreMetrics posterior_scores{};
};

// Calibración estocástica mediante Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC)
AbcSmcResult calibrate_abc_smc(
    int num_agents,
    int initial_infected,
    const SimParams& base_params,
    const std::vector<float>& obs_days,
    const std::vector<float>& obs_i_frac,
    const AbcSmcConfig& cfg = AbcSmcConfig{}
);

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
