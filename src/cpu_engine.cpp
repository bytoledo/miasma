#include "cpu_engine.hpp"

#include <algorithm>
#include <cmath>
#include <omp.h>

namespace seird {

namespace {

constexpr double PI_D = 3.14159265358979323846;

struct OdeDeriv {
    double s = 0.0;
    double e = 0.0;
    double i = 0.0;
    double r = 0.0;
    double d = 0.0;
};

OdeDeriv eval_seirds_deriv(const OdeState& st, double beta_eff, const SimParams& params) {
    const double n = st.s + st.e + st.i + st.r + st.d;
    if (n <= 0.0) return {};

    const double sigma   = (params.incubation_days > 1e-4f) ? (1.0 / static_cast<double>(params.incubation_days)) : 1e4;
    const double inv_inf = (params.infectious_days > 1e-4f) ? (1.0 / static_cast<double>(params.infectious_days)) : 1e4;
    const double alpha   = std::clamp(static_cast<double>(params.fatality_rate), 0.0, 1.0);
    const double gamma   = (1.0 - alpha) * inv_inf;
    const double mu_d    = alpha * inv_inf;
    const double nu_v    = std::max(0.0, static_cast<double>(params.vaccination_rate));
    const double xi      = (params.immunity_days > 1e-4f) ? (1.0 / static_cast<double>(params.immunity_days)) : 0.0;

    const double s   = std::max(0.0, st.s);
    const double e   = std::max(0.0, st.e);
    const double i   = std::max(0.0, st.i);
    const double r   = std::max(0.0, st.r);
    const double inf = beta_eff * s * i / n;

    OdeDeriv k;
    k.s = -inf - nu_v * s + xi * r;
    k.e =  inf - sigma * e;
    k.i =  sigma * e - (gamma + mu_d) * i;
    k.r =  gamma * i + nu_v * s - xi * r;
    k.d =  mu_d * i;
    return k;
}

uint64_t splitmix64_next(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

} // namespace

OdeState ode_seirds_step_rk4(const OdeState& st, double beta_eff, const SimParams& params, double dt) {
    const double n = st.s + st.e + st.i + st.r + st.d;
    if (n <= 0.0 || dt <= 0.0) return st;

    const OdeDeriv k1 = eval_seirds_deriv(st, beta_eff, params);

    auto advance = [&](const OdeDeriv& k, double h) {
        return OdeState{
            std::max(0.0, st.s + h * k.s),
            std::max(0.0, st.e + h * k.e),
            std::max(0.0, st.i + h * k.i),
            std::max(0.0, st.r + h * k.r),
            std::max(0.0, st.d + h * k.d),
            st.day + h
        };
    };

    const OdeDeriv k2 = eval_seirds_deriv(advance(k1, 0.5 * dt), beta_eff, params);
    const OdeDeriv k3 = eval_seirds_deriv(advance(k2, 0.5 * dt), beta_eff, params);
    const OdeDeriv k4 = eval_seirds_deriv(advance(k3, dt),       beta_eff, params);

    OdeState out;
    out.s   = std::max(0.0, st.s + (dt / 6.0) * (k1.s + 2.0 * k2.s + 2.0 * k3.s + k4.s));
    out.e   = std::max(0.0, st.e + (dt / 6.0) * (k1.e + 2.0 * k2.e + 2.0 * k3.e + k4.e));
    out.i   = std::max(0.0, st.i + (dt / 6.0) * (k1.i + 2.0 * k2.i + 2.0 * k3.i + k4.i));
    out.r   = std::max(0.0, st.r + (dt / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r));
    out.d   = std::max(0.0, st.d + (dt / 6.0) * (k1.d + 2.0 * k2.d + 2.0 * k3.d + k4.d));
    out.day = st.day + dt;
    return out;
}

std::vector<OdeState> ode_seirds_simulate(
    double total_pop,
    double initial_infected,
    double beta_eff,
    const SimParams& params,
    double total_days,
    double dt
) {
    std::vector<OdeState> traj;
    if (total_pop <= 0.0 || dt <= 0.0 || total_days < 0.0) return traj;

    const double init_i = std::clamp(initial_infected, 0.0, total_pop);
    OdeState curr{total_pop - init_i, 0.0, init_i, 0.0, 0.0, 0.0};
    traj.push_back(curr);

    const int steps = static_cast<int>(std::ceil(total_days / dt));
    traj.reserve(static_cast<std::size_t>(steps) + 1);
    for (int s = 0; s < steps; ++s) {
        curr = ode_seirds_step_rk4(curr, beta_eff, params, dt);
        traj.push_back(curr);
    }
    return traj;
}

double compute_abm_homogeneous_beta(int num_agents, const SimParams& params) {
    if (num_agents <= 0) return 0.0;
    const double area = std::max(1.0, static_cast<double>(params.world_width) * static_cast<double>(params.world_height));
    const double r    = std::max(0.0, static_cast<double>(params.infection_radius));
    const double dt   = std::max(1e-5, static_cast<double>(params.dt));
    const double b    = std::clamp(static_cast<double>(params.beta), 0.0, 0.999);

    const double p_step = 1.0 - std::pow(1.0 - b, dt);
    return (static_cast<double>(num_agents) * (PI_D * r * r) / area) * (p_step / dt);
}

double compute_abm_kinetic_beta(int num_agents, const SimParams& params) {
    if (num_agents <= 0) return 0.0;
    const double area   = std::max(1.0, static_cast<double>(params.world_width) * static_cast<double>(params.world_height));
    const double r      = std::max(0.0, static_cast<double>(params.infection_radius));
    const double dt     = std::max(1e-5, static_cast<double>(params.dt));
    const double b      = std::clamp(static_cast<double>(params.beta), 0.0, 0.999);
    const double p_step = 1.0 - std::pow(1.0 - b, dt);

    const double v_eff     = std::max(0.01, static_cast<double>(params.base_speed) * static_cast<double>(params.quarantine_factor) * 1.03);
    const double v_rel     = (4.0 / PI_D) * v_eff;
    const double area_step = std::min(PI_D * r * r, 2.0 * r * v_rel * dt);
    const double k_dwell   = std::max(1.0, (PI_D * r) / std::max(2.0 * v_rel * dt, 1e-9));

    const double p_enc       = 1.0 - std::pow(1.0 - p_step, k_dwell);
    const double s_mult      = std::max(1.0, static_cast<double>(params.superspreader_mult));
    const double f_super     = std::clamp(static_cast<double>(params.superspreader_frac), 0.0, 1.0);
    const double p_enc_super = 1.0 - std::pow(1.0 - p_step, k_dwell * s_mult);
    const double p_eff       = (1.0 - f_super) * p_enc + f_super * p_enc_super;

    return (static_cast<double>(num_agents) * area_step / (area * dt)) * p_eff;
}

double compute_r0_ngm(int num_agents, const SimParams& params) {
    if (num_agents <= 0) return 0.0;

    // Estratos RiskGroup: RISK_LOW (50%), RISK_MEDIUM (35%), RISK_HIGH (15%)
    const double f[3] = {0.50, 0.35, 0.15};
    const double m[3] = {1.15, 1.00, 0.70};
    const double m_bar = f[0] * m[0] + f[1] * m[1] + f[2] * m[2]; // 1.03

    const double beta_eff = compute_abm_kinetic_beta(num_agents, params);
    const double d_inf    = std::max(0.0, static_cast<double>(params.infectious_days));

    double k_mat[3][3]{};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            k_mat[a][b] = beta_eff * d_inf * f[a] * (m[a] * m[b]) / (m_bar * m_bar);
        }
    }

    // Iteración de potencias para el radio espectral rho(K)
    double v[3] = {1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)};
    double rayleigh = 0.0;
    for (int it = 0; it < 30; ++it) {
        double kv[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                kv[a] += k_mat[a][b] * v[b];
            }
        }
        rayleigh = v[0] * kv[0] + v[1] * kv[1] + v[2] * kv[2];
        const double norm = std::sqrt(kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2]);
        if (norm <= 1e-15) return 0.0;
        v[0] = kv[0] / norm;
        v[1] = kv[1] / norm;
        v[2] = kv[2] / norm;
    }
    return std::max(0.0, rayleigh);
}

double sir_analytical_peak_fraction(double r0) {
    if (r0 <= 1.0) return 0.0;
    return std::max(0.0, 1.0 - (1.0 + std::log(r0)) / r0);
}

double sir_analytical_final_attack_rate(double r0) {
    if (r0 <= 1.0) return 0.0;
    double s = 0.5;
    for (int it = 0; it < 60; ++it) {
        const double f  = s - std::exp(-r0 * (1.0 - s));
        const double df = 1.0 - r0 * std::exp(-r0 * (1.0 - s));
        if (std::fabs(df) > 1e-12) {
            s = std::clamp(s - f / df, 1e-12, 1.0 - 1e-12);
        } else {
            s = std::exp(-r0 * (1.0 - s));
        }
    }
    return std::clamp(1.0 - s, 0.0, 1.0);
}

CpuTelemetryEngine::CpuTelemetryEngine(std::size_t window_size)
    : window_size_(std::max<std::size_t>(window_size, 8)) {}

void CpuTelemetryEngine::reset() {
    update_count_ = 0;
    stats_window_.clear();
    incidence_window_.clear();
    history_.clear();
}

TelemetryMetrics CpuTelemetryEngine::update(const SimStats& stats, const SimParams& params) {
    ++update_count_;

    TelemetryMetrics m;
    m.stats = stats;

    // Tasa de nuevos contagios reales (S -> E) por día usando el contador exacto de GPU
    float elapsed_step = std::max(params.dt, 1e-4f);
    if (!stats_window_.empty()) {
        const float diff = stats.sim_day - stats_window_.back().sim_day;
        if (diff > 1e-4f) elapsed_step = diff;
    } else if (stats.sim_day > 1e-4f) {
        elapsed_step = stats.sim_day;
    }
    const float incidence = static_cast<float>(stats.new_exposures) / elapsed_step;
    m.new_cases_rate = incidence;

    const int total_pop = static_cast<int>(
        static_cast<uint64_t>(stats.count_s) +
        static_cast<uint64_t>(stats.count_e) +
        static_cast<uint64_t>(stats.count_i) +
        static_cast<uint64_t>(stats.count_r) +
        static_cast<uint64_t>(stats.count_d)
    );
    m.r0_ngm = (total_pop > 0) ? static_cast<float>(compute_r0_ngm(total_pop, params)) : 0.0f;

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
            const double val = static_cast<double>(incidence_window_[i]);
            sum    += val;
            sum_sq += val * val;
        }

        const double mean     = sum / static_cast<double>(n);
        const double variance = std::max(0.0, (sum_sq / static_cast<double>(n)) - (mean * mean));
        const double stddev   = std::sqrt(variance);

        m.z_score = (stddev > 1e-5)
            ? static_cast<float>((static_cast<double>(incidence) - mean) / stddev)
            : 0.0f;
    }

    // Estimación puntual de R_t mediante tasa de crecimiento exponencial r y tiempo generacional T_g
    const double t_g = static_cast<double>(params.incubation_days) +
                       0.5 * static_cast<double>(params.infectious_days);
    if (n >= 4) {
        const int lag = std::min(n - 1, 10);
        const SimStats& past = stats_window_[n - 1 - lag];

        const double active_now  = static_cast<double>(stats.count_e + stats.count_i);
        const double active_past = static_cast<double>(past.count_e + past.count_i);
        double elapsed = static_cast<double>(stats.sim_day - past.sim_day);
        if (elapsed <= 1e-4) {
            elapsed = static_cast<double>(lag) * static_cast<double>(std::max(params.dt, 1e-4f));
        }

        if (active_past > 0.0 && active_now > 0.0 && elapsed > 0.0) {
            const double r = (std::log(active_now) - std::log(active_past)) / elapsed;
            m.rt = static_cast<float>(std::max(0.0, 1.0 + r * t_g));
        } else if (active_now > 0.0 && active_past == 0.0) {
            m.rt = 1.5f;
        } else {
            m.rt = 0.0f;
        }
    }

    // Bootstrap OpenMP de tasas de crecimiento recientes para intervalos P5-P95 y P(R_t > 1)
    std::vector<double> r_samples;
    if (n >= 2) {
        const int max_intervals = std::min(n - 1, 16);
        const int start_idx     = n - 1 - max_intervals;
        r_samples.reserve(static_cast<std::size_t>(max_intervals));
        for (int k = start_idx; k < n - 1; ++k) {
            const SimStats& s0 = stats_window_[k];
            const SimStats& s1 = stats_window_[k + 1];
            const double a0 = static_cast<double>(s0.count_e + s0.count_i);
            const double a1 = static_cast<double>(s1.count_e + s1.count_i);
            double dt_step  = static_cast<double>(s1.sim_day - s0.sim_day);
            if (dt_step <= 1e-4) dt_step = static_cast<double>(std::max(params.dt, 1e-4f));
            if (a0 > 0.0 && a1 > 0.0 && dt_step > 1e-4) {
                r_samples.push_back((std::log(a1) - std::log(a0)) / dt_step);
            }
        }
    }

    if (!r_samples.empty()) {
        constexpr int B = 256;
        std::vector<float> rt_boot(B, 0.0f);
        const int m_cnt = static_cast<int>(r_samples.size());
        const uint64_t base_seed = update_count_ * 0x9E3779B97F4A7C15ULL;

        // Secuencial en <0.02 ms para evitar spin-wait de OpenMP (vcomp140) en actualizaciones de UI
        for (int b = 0; b < B; ++b) {
            uint64_t rng = base_seed ^ (static_cast<uint64_t>(b + 1) * 0xBF58476D1CE4E5B9ULL);
            double r_sum = 0.0;
            for (int j = 0; j < m_cnt; ++j) {
                const uint64_t rnd = splitmix64_next(rng);
                const int idx = static_cast<int>(rnd % static_cast<uint64_t>(m_cnt));
                r_sum += r_samples[idx];
            }
            const double r_mean = r_sum / static_cast<double>(m_cnt);
            rt_boot[b] = static_cast<float>(std::max(0.0, 1.0 + r_mean * t_g));
        }

        std::sort(rt_boot.begin(), rt_boot.end());
        const int idx_p05 = static_cast<int>(0.05 * static_cast<double>(B - 1));
        const int idx_p95 = static_cast<int>(0.95 * static_cast<double>(B - 1));
        m.rt_p05 = rt_boot[idx_p05];
        m.rt_p95 = rt_boot[idx_p95];

        int gt1 = 0;
        for (int b = 0; b < B; ++b) {
            if (rt_boot[b] > 1.0f) ++gt1;
        }
        m.prob_rt_gt_1 = static_cast<float>(gt1) / static_cast<float>(B);
    } else {
        m.rt_p05       = m.rt;
        m.rt_p95       = m.rt;
        m.prob_rt_gt_1 = (m.rt > 1.0f) ? 1.0f : 0.0f;
    }

    // Case Fatality Rate (CFR) acumulada
    const uint64_t resolved = static_cast<uint64_t>(stats.count_r) + static_cast<uint64_t>(stats.count_d);
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

CpuEnsembleEngine::CpuEnsembleEngine(int num_replicas, std::size_t window_size)
    : window_size_(std::max<std::size_t>(window_size, 8)) {
    reset(num_replicas);
}

void CpuEnsembleEngine::reset(int num_replicas) {
    const int k = std::clamp(num_replicas, 1, MAX_REPLICAS);
    engines_.clear();
    engines_.reserve(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        engines_.emplace_back(window_size_);
    }
    history_.clear();
}

EnsembleMetrics CpuEnsembleEngine::update(
    const std::vector<SimStats>& rep_stats,
    const SimParams& params,
    int agents_per_replica
) {
    EnsembleMetrics em;
    const int k = static_cast<int>(std::min(engines_.size(), rep_stats.size()));
    if (k <= 0) return em;

    em.num_replicas = k;
    em.sim_day      = rep_stats[0].sim_day;

    std::vector<float> i_fracs(static_cast<std::size_t>(k), 0.0f);
    std::vector<float> rt_vals(static_cast<std::size_t>(k), 0.0f);
    int extinct_cnt = 0;
    int rt_gt_1_cnt = 0;
    const float pop = static_cast<float>(std::max(1, agents_per_replica));

    #pragma omp parallel for schedule(static) reduction(+:extinct_cnt, rt_gt_1_cnt) if(k >= 4)
    for (int r = 0; r < k; ++r) {
        const TelemetryMetrics tm = engines_[static_cast<std::size_t>(r)].update(rep_stats[static_cast<std::size_t>(r)], params);
        i_fracs[static_cast<std::size_t>(r)] = static_cast<float>(rep_stats[static_cast<std::size_t>(r)].count_i) / pop;
        rt_vals[static_cast<std::size_t>(r)] = tm.rt;
        if (rep_stats[static_cast<std::size_t>(r)].count_e == 0u &&
            rep_stats[static_cast<std::size_t>(r)].count_i == 0u) {
            ++extinct_cnt;
        }
        if (tm.rt > 1.0f) {
            ++rt_gt_1_cnt;
        }
    }

    std::sort(i_fracs.begin(), i_fracs.end());
    std::sort(rt_vals.begin(), rt_vals.end());

    auto pct = [&](const std::vector<float>& v, double q) -> float {
        const std::size_t idx = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1));
        return v[idx];
    };

    em.i_p05          = pct(i_fracs, 0.05);
    em.i_p50          = pct(i_fracs, 0.50);
    em.i_p95          = pct(i_fracs, 0.95);
    em.rt_p05         = pct(rt_vals, 0.05);
    em.rt_p50         = pct(rt_vals, 0.50);
    em.rt_p95         = pct(rt_vals, 0.95);
    em.prob_rt_gt_1   = static_cast<float>(rt_gt_1_cnt) / static_cast<float>(k);
    em.extinction_pct = 100.0f * static_cast<float>(extinct_cnt) / static_cast<float>(k);

    history_.push_back(em);
    if (history_.size() > 2048) {
        history_.erase(history_.begin());
    }
    return em;
}

} // namespace seird
