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

    const double sigma = (params.incubation_days > 1e-4f) ? (1.0 / static_cast<double>(params.incubation_days)) : 1e4;
    const double inv_inf = (params.infectious_days > 1e-4f) ? (1.0 / static_cast<double>(params.infectious_days)) : 1e4;
    const double max_rate = std::max(sigma, inv_inf);
    const int substeps = std::clamp(static_cast<int>(std::ceil(max_rate * dt / 0.5)), 1, 512);
    const double h_dt  = dt / static_cast<double>(substeps);

    OdeState curr = st;
    for (int sub = 0; sub < substeps; ++sub) {
        const OdeDeriv k1 = eval_seirds_deriv(curr, beta_eff, params);

        auto advance = [&](const OdeDeriv& k, double h) {
            return OdeState{
                std::max(0.0, curr.s + h * k.s),
                std::max(0.0, curr.e + h * k.e),
                std::max(0.0, curr.i + h * k.i),
                std::max(0.0, curr.r + h * k.r),
                std::max(0.0, curr.d + h * k.d),
                curr.day + h
            };
        };

        const OdeDeriv k2 = eval_seirds_deriv(advance(k1, 0.5 * h_dt), beta_eff, params);
        const OdeDeriv k3 = eval_seirds_deriv(advance(k2, 0.5 * h_dt), beta_eff, params);
        const OdeDeriv k4 = eval_seirds_deriv(advance(k3, h_dt),       beta_eff, params);

        curr.s   = std::max(0.0, curr.s + (h_dt / 6.0) * (k1.s + 2.0 * k2.s + 2.0 * k3.s + k4.s));
        curr.e   = std::max(0.0, curr.e + (h_dt / 6.0) * (k1.e + 2.0 * k2.e + 2.0 * k3.e + k4.e));
        curr.i   = std::max(0.0, curr.i + (h_dt / 6.0) * (k1.i + 2.0 * k2.i + 2.0 * k3.i + k4.i));
        curr.r   = std::max(0.0, curr.r + (h_dt / 6.0) * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r));
        curr.d   = std::max(0.0, curr.d + (h_dt / 6.0) * (k1.d + 2.0 * k2.d + 2.0 * k3.d + k4.d));
        curr.day += h_dt;
    }
    return curr;
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

const char* rt_category_label(RtCategory cat) {
    switch (cat) {
        case RT_DECLINING:        return "Declining";
        case RT_LIKELY_DECLINING: return "Likely Declining";
        case RT_UNCERTAIN:        return "Uncertain";
        case RT_LIKELY_GROWING:   return "Likely Growing";
        case RT_GROWING:          return "Growing";
        default:                  return "Uncertain";
    }
}

CpuTelemetryEngine::CpuTelemetryEngine(std::size_t window_size)
    : window_size_(std::max<std::size_t>(window_size, 8)) {}

void CpuTelemetryEngine::reset() {
    update_count_   = 0;
    cum_infections_ = 0;
    stats_window_.clear();
    incidence_window_.clear();
    history_.clear();
}

TelemetryMetrics CpuTelemetryEngine::update(const SimStats& stats, const SimParams& params) {
    ++update_count_;
    if (update_count_ == 1) {
        cum_infections_ = static_cast<uint64_t>(stats.count_e) + static_cast<uint64_t>(stats.count_i) +
                          static_cast<uint64_t>(stats.count_d) + static_cast<uint64_t>(stats.new_exposures);
    } else {
        cum_infections_ += static_cast<uint64_t>(stats.new_exposures);
    }

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

    // Categoría operativa CDC CFA según P(Rt > 1)
    if (m.prob_rt_gt_1 < 0.10f)      m.rt_category = RT_DECLINING;
    else if (m.prob_rt_gt_1 < 0.25f) m.rt_category = RT_LIKELY_DECLINING;
    else if (m.prob_rt_gt_1 <= 0.75f) m.rt_category = RT_UNCERTAIN;
    else if (m.prob_rt_gt_1 <= 0.90f) m.rt_category = RT_LIKELY_GROWING;
    else                              m.rt_category = RT_GROWING;

    // Case Fatality Rate (CFR) acumulada e Infection Fatality Rate (IFR)
    const uint64_t resolved = static_cast<uint64_t>(stats.count_r) + static_cast<uint64_t>(stats.count_d);
    if (resolved > 0) {
        m.cfr = static_cast<float>(static_cast<double>(stats.count_d) / static_cast<double>(resolved));
    }
    if (cum_infections_ > 0) {
        m.ifr = static_cast<float>(static_cast<double>(stats.count_d) / static_cast<double>(cum_infections_));
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
    em.i_p10          = pct(i_fracs, 0.10);
    em.i_p50          = pct(i_fracs, 0.50);
    em.i_p90          = pct(i_fracs, 0.90);
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

ForecastScoreMetrics compute_ensemble_forecast_scores(
    const std::vector<EnsembleMetrics>& ens_traj,
    const std::vector<float>& obs_i_frac
) {
    ForecastScoreMetrics out;
    const std::size_t n = std::min(ens_traj.size(), obs_i_frac.size());
    if (n == 0) return out;

    auto interval_score = [](double l, double u, double y, double alpha) -> double {
        double sc = (u - l);
        if (y < l) sc += (2.0 / alpha) * (l - y);
        else if (y > u) sc += (2.0 / alpha) * (y - u);
        return sc;
    };

    double sum_ae   = 0.0;
    double sum_se   = 0.0;
    double sum_is80 = 0.0;
    double sum_wis  = 0.0;
    std::size_t cov80 = 0;
    std::size_t cov90 = 0;

    for (std::size_t t = 0; t < n; ++t) {
        const double y   = static_cast<double>(obs_i_frac[t]);
        const double med = static_cast<double>(ens_traj[t].i_p50);
        const double l10 = static_cast<double>(ens_traj[t].i_p10);
        const double u90 = static_cast<double>(ens_traj[t].i_p90);
        const double l05 = static_cast<double>(ens_traj[t].i_p05);
        const double u95 = static_cast<double>(ens_traj[t].i_p95);

        const double err = std::fabs(y - med);
        sum_ae += err;
        sum_se += err * err;

        const double is80 = interval_score(l10, u90, y, 0.20);
        const double is90 = interval_score(l05, u95, y, 0.10);
        sum_is80 += is80;
        // WIS estándar con K=2 intervalos simétricos (IC80, IC90)
        sum_wis += (0.5 * err + 0.10 * is80 + 0.05 * is90) / 2.5;

        if (y >= l10 && y <= u90) ++cov80;
        if (y >= l05 && y <= u95) ++cov90;
    }

    const double dn = static_cast<double>(n);
    out.mae         = static_cast<float>(sum_ae / dn);
    out.rmse        = static_cast<float>(std::sqrt(sum_se / dn));
    out.is80        = static_cast<float>(sum_is80 / dn);
    out.wis         = static_cast<float>(sum_wis / dn);
    out.coverage_80 = static_cast<float>(static_cast<double>(cov80) / dn);
    out.coverage_90 = static_cast<float>(static_cast<double>(cov90) / dn);
    return out;
}

std::vector<SensitivityRow> run_global_sensitivity_sweep(
    int num_agents,
    int initial_infected,
    const SimParams& base_params,
    const std::vector<float>& betas,
    const std::vector<float>& quar_factors,
    const std::vector<float>& super_fracs,
    double horizon_days
) {
    std::vector<SensitivityRow> grid;
    for (float b : betas) {
        for (float q : quar_factors) {
            for (float sf : super_fracs) {
                SensitivityRow row;
                row.beta               = b;
                row.quarantine_factor  = q;
                row.superspreader_frac = sf;
                grid.push_back(row);
            }
        }
    }
    const int total = static_cast<int>(grid.size());
    if (total == 0 || num_agents <= 0) return grid;

    #pragma omp parallel for schedule(dynamic) if(total >= 4)
    for (int idx = 0; idx < total; ++idx) {
        SimParams p = base_params;
        p.beta               = grid[static_cast<std::size_t>(idx)].beta;
        p.quarantine_factor  = grid[static_cast<std::size_t>(idx)].quarantine_factor;
        p.superspreader_frac = grid[static_cast<std::size_t>(idx)].superspreader_frac;

        const double beta_kin = compute_abm_kinetic_beta(num_agents, p);
        const double r0       = compute_r0_ngm(num_agents, p);
        const auto traj       = ode_seirds_simulate(
            static_cast<double>(num_agents),
            static_cast<double>(initial_infected),
            beta_kin,
            p,
            horizon_days,
            std::max(1e-3, static_cast<double>(p.dt))
        );

        double peak_i = 0.0;
        for (const auto& st : traj) {
            peak_i = std::max(peak_i, st.i);
        }
        const double n_pop = static_cast<double>(num_agents);
        const double attack = (!traj.empty()) ? (1.0 - (traj.back().s / n_pop)) : 0.0;

        grid[static_cast<std::size_t>(idx)].r0_ngm            = static_cast<float>(r0);
        grid[static_cast<std::size_t>(idx)].peak_i_frac       = static_cast<float>(peak_i / n_pop);
        grid[static_cast<std::size_t>(idx)].final_attack_rate = static_cast<float>(attack);
    }
    return grid;
}

AbcSmcResult calibrate_abc_smc(
    int num_agents,
    int initial_infected,
    const SimParams& base_params,
    const std::vector<float>& obs_days,
    const std::vector<float>& obs_i_frac,
    const AbcSmcConfig& cfg
) {
    AbcSmcResult res;
    const int n_obs = static_cast<int>(std::min(obs_days.size(), obs_i_frac.size()));
    const int Np    = std::max(8, cfg.num_particles);
    const int Ng    = std::max(1, cfg.num_generations);
    if (n_obs == 0 || num_agents <= 0) return res;

    const double max_day = static_cast<double>(obs_days[static_cast<std::size_t>(n_obs - 1)]);
    const double dt      = std::max(1e-3, static_cast<double>(base_params.dt));
    const double n_pop   = static_cast<double>(num_agents);

    auto u01 = [](uint64_t& s) -> double {
        return static_cast<double>(splitmix64_next(s) >> 11) * (1.0 / 9007199254740992.0);
    };
    auto normal01 = [&](uint64_t& s) -> double {
        double u1 = std::max(1e-12, u01(s));
        double u2 = u01(s);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI_D * u2);
    };

    auto eval_particle = [&](float beta, float incub, float infec, uint64_t& rng, std::vector<float>* out_traj) -> float {
        SimParams p = base_params;
        p.beta            = beta;
        p.incubation_days = incub;
        p.infectious_days = infec;

        const double beta_kin = compute_abm_kinetic_beta(num_agents, p);
        const auto sim = ode_seirds_simulate(n_pop, static_cast<double>(initial_infected), beta_kin, p, max_day, dt);

        double sum_se = 0.0;
        if (out_traj) out_traj->resize(static_cast<std::size_t>(n_obs));
        for (int t = 0; t < n_obs; ++t) {
            const std::size_t idx = std::min(
                sim.size() - 1,
                static_cast<std::size_t>(std::llround(static_cast<double>(obs_days[static_cast<std::size_t>(t)]) / dt))
            );
            double pred = sim[idx].i / n_pop;
            if (cfg.obs_noise_cv > 0.0f) {
                pred = std::max(0.0, pred * (1.0 + static_cast<double>(cfg.obs_noise_cv) * normal01(rng)));
            }
            if (out_traj) (*out_traj)[static_cast<std::size_t>(t)] = static_cast<float>(pred);
            const double diff = pred - static_cast<double>(obs_i_frac[static_cast<std::size_t>(t)]);
            sum_se += diff * diff;
        }
        return static_cast<float>(std::sqrt(sum_se / static_cast<double>(n_obs)));
    };

    std::vector<AbcParticle> curr(static_cast<std::size_t>(Np));

    // Generación 0: muestreo directo del prior uniforme
    #pragma omp parallel for schedule(dynamic) if(Np >= 8)
    for (int i = 0; i < Np; ++i) {
        uint64_t rng = cfg.seed ^ (static_cast<uint64_t>(i + 1) * 0x9E3779B97F4A7C15ULL);
        AbcParticle pt;
        pt.beta            = cfg.beta_min  + static_cast<float>(u01(rng)) * (cfg.beta_max  - cfg.beta_min);
        pt.incubation_days = cfg.incub_min + static_cast<float>(u01(rng)) * (cfg.incub_max - cfg.incub_min);
        pt.infectious_days = cfg.infec_min + static_cast<float>(u01(rng)) * (cfg.infec_max - cfg.infec_min);
        pt.weight          = 1.0f / static_cast<float>(Np);
        pt.distance        = eval_particle(pt.beta, pt.incubation_days, pt.infectious_days, rng, nullptr);
        curr[static_cast<std::size_t>(i)] = pt;
    }

    std::vector<float> dists(static_cast<std::size_t>(Np));
    for (int i = 0; i < Np; ++i) dists[static_cast<std::size_t>(i)] = curr[static_cast<std::size_t>(i)].distance;
    std::sort(dists.begin(), dists.end());
    float epsilon = dists[static_cast<std::size_t>(Np / 2)];

    // Generaciones secuenciales g = 1..Ng-1
    for (int g = 1; g < Ng; ++g) {
        std::vector<AbcParticle> prev = curr;
        // Calcular dispersión empírica de la generación previa para el kernel de perturbación
        double mean_b = 0.0, mean_e = 0.0, mean_i = 0.0;
        for (const auto& pt : prev) {
            mean_b += pt.weight * pt.beta;
            mean_e += pt.weight * pt.incubation_days;
            mean_i += pt.weight * pt.infectious_days;
        }
        double var_b = 1e-6, var_e = 1e-4, var_i = 1e-4;
        for (const auto& pt : prev) {
            var_b += pt.weight * (pt.beta - mean_b) * (pt.beta - mean_b);
            var_e += pt.weight * (pt.incubation_days - mean_e) * (pt.incubation_days - mean_e);
            var_i += pt.weight * (pt.infectious_days - mean_i) * (pt.infectious_days - mean_i);
        }
        const double sd_b = std::max(0.005, std::sqrt(2.0 * var_b));
        const double sd_e = std::max(0.10,  std::sqrt(2.0 * var_e));
        const double sd_i = std::max(0.15,  std::sqrt(2.0 * var_i));

        std::vector<double> cdf(static_cast<std::size_t>(Np), 0.0);
        double acc_w = 0.0;
        for (int j = 0; j < Np; ++j) {
            acc_w += static_cast<double>(prev[static_cast<std::size_t>(j)].weight);
            cdf[static_cast<std::size_t>(j)] = acc_w;
        }
        if (acc_w > 0.0) {
            for (double& c : cdf) c /= acc_w;
        }

        #pragma omp parallel for schedule(dynamic) if(Np >= 8)
        for (int i = 0; i < Np; ++i) {
            uint64_t rng = cfg.seed ^ (static_cast<uint64_t>(g * 10000 + i + 1) * 0xBF58476D1CE4E5B9ULL);
            AbcParticle best_pt;
            best_pt.distance = 1e9f;

            constexpr int MAX_TRIES = 40;
            for (int tr = 0; tr < MAX_TRIES; ++tr) {
                double u = u01(rng);
                int pidx = 0;
                while (pidx < Np - 1 && cdf[static_cast<std::size_t>(pidx)] < u) ++pidx;
                const AbcParticle& parent = prev[static_cast<std::size_t>(pidx)];

                float nb = static_cast<float>(parent.beta + sd_b * normal01(rng));
                float ne = static_cast<float>(parent.incubation_days + sd_e * normal01(rng));
                float ni = static_cast<float>(parent.infectious_days + sd_i * normal01(rng));
                if (nb < cfg.beta_min  || nb > cfg.beta_max ||
                    ne < cfg.incub_min || ne > cfg.incub_max ||
                    ni < cfg.infec_min || ni > cfg.infec_max) {
                    continue;
                }
                float d = eval_particle(nb, ne, ni, rng, nullptr);
                if (d < best_pt.distance) {
                    best_pt.beta            = nb;
                    best_pt.incubation_days = ne;
                    best_pt.infectious_days = ni;
                    best_pt.distance        = d;
                }
                if (d <= epsilon) break;
            }
            if (best_pt.distance >= 1e8f) {
                best_pt = prev[static_cast<std::size_t>(i)];
            }
            curr[static_cast<std::size_t>(i)] = best_pt;
        }

        // Actualizar pesos SMC
        double sum_w = 0.0;
        for (int i = 0; i < Np; ++i) {
            const auto& pt = curr[static_cast<std::size_t>(i)];
            double denom = 0.0;
            for (int j = 0; j < Np; ++j) {
                const auto& pr = prev[static_cast<std::size_t>(j)];
                double db = (pt.beta - pr.beta) / sd_b;
                double de = (pt.incubation_days - pr.incubation_days) / sd_e;
                double di = (pt.infectious_days - pr.infectious_days) / sd_i;
                double k  = std::exp(-0.5 * (db * db + de * de + di * di));
                denom += static_cast<double>(pr.weight) * k;
            }
            double w = (denom > 1e-18) ? (1.0 / denom) : 1.0;
            curr[static_cast<std::size_t>(i)].weight = static_cast<float>(w);
            sum_w += w;
        }
        if (sum_w > 0.0) {
            for (auto& pt : curr) pt.weight = static_cast<float>(static_cast<double>(pt.weight) / sum_w);
        }

        for (int i = 0; i < Np; ++i) dists[static_cast<std::size_t>(i)] = curr[static_cast<std::size_t>(i)].distance;
        std::sort(dists.begin(), dists.end());
        epsilon = std::min(epsilon * 0.85f, dists[static_cast<std::size_t>(Np / 2)]);
    }

    res.posterior     = curr;
    res.final_epsilon = epsilon;

    std::vector<float> bs(static_cast<std::size_t>(Np)), es(static_cast<std::size_t>(Np)), is(static_cast<std::size_t>(Np));
    for (int i = 0; i < Np; ++i) {
        bs[static_cast<std::size_t>(i)] = curr[static_cast<std::size_t>(i)].beta;
        es[static_cast<std::size_t>(i)] = curr[static_cast<std::size_t>(i)].incubation_days;
        is[static_cast<std::size_t>(i)] = curr[static_cast<std::size_t>(i)].infectious_days;
    }
    std::sort(bs.begin(), bs.end());
    std::sort(es.begin(), es.end());
    std::sort(is.begin(), is.end());
    res.beta_p05  = bs[static_cast<std::size_t>(0.05 * static_cast<double>(Np - 1))];
    res.beta_p50  = bs[static_cast<std::size_t>(0.50 * static_cast<double>(Np - 1))];
    res.beta_p95  = bs[static_cast<std::size_t>(0.95 * static_cast<double>(Np - 1))];
    res.incub_p50 = es[static_cast<std::size_t>(0.50 * static_cast<double>(Np - 1))];
    res.infec_p50 = is[static_cast<std::size_t>(0.50 * static_cast<double>(Np - 1))];

    // Evaluar métricas probabilísticas del ensamble posterior
    std::vector<std::vector<float>> trajs(static_cast<std::size_t>(Np));
    for (int i = 0; i < Np; ++i) {
        uint64_t rng = cfg.seed ^ static_cast<uint64_t>(i + 999);
        eval_particle(curr[static_cast<std::size_t>(i)].beta,
                      curr[static_cast<std::size_t>(i)].incubation_days,
                      curr[static_cast<std::size_t>(i)].infectious_days,
                      rng, &trajs[static_cast<std::size_t>(i)]);
    }
    std::vector<EnsembleMetrics> ens_post(static_cast<std::size_t>(n_obs));
    for (int t = 0; t < n_obs; ++t) {
        std::vector<float> vals(static_cast<std::size_t>(Np));
        for (int i = 0; i < Np; ++i) vals[static_cast<std::size_t>(i)] = trajs[static_cast<std::size_t>(i)][static_cast<std::size_t>(t)];
        std::sort(vals.begin(), vals.end());
        auto q = [&](double p) { return vals[static_cast<std::size_t>(p * static_cast<double>(Np - 1))]; };
        ens_post[static_cast<std::size_t>(t)].sim_day = obs_days[static_cast<std::size_t>(t)];
        ens_post[static_cast<std::size_t>(t)].i_p05   = q(0.05);
        ens_post[static_cast<std::size_t>(t)].i_p10   = q(0.10);
        ens_post[static_cast<std::size_t>(t)].i_p50   = q(0.50);
        ens_post[static_cast<std::size_t>(t)].i_p90   = q(0.90);
        ens_post[static_cast<std::size_t>(t)].i_p95   = q(0.95);
    }
    res.posterior_scores = compute_ensemble_forecast_scores(ens_post, obs_i_frac);
    return res;
}

} // namespace seird
