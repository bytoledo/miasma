#include "cpu_engine.hpp"
#include "cuda_engine.cuh"
#include "types.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define ASSERT_TRUE(cond, msg)                                                     \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
            return 1;                                                              \
        }                                                                          \
    } while (0)

int test_hardware_limits_and_domain() {
    const seird::HardwareLimits hw = seird::cuda_query_hardware_limits();
    std::printf("[INFO] VRAM total=%zu MB, free=%zu MB, bytes/agent=%zu, max_vram=%d, safe=%d\n",
                hw.vram_total_mb, hw.vram_free_mb, hw.bytes_per_agent,
                hw.max_agents_vram, hw.safe_agents_limit);

    ASSERT_TRUE(hw.vram_total_mb > 0, "VRAM total debe ser > 0");
    ASSERT_TRUE(hw.bytes_per_agent > 0, "bytes_per_agent debe ser > 0");
    ASSERT_TRUE(hw.max_agents_vram >= hw.safe_agents_limit, "max_agents_vram >= safe_agents_limit");
    ASSERT_TRUE(hw.safe_agents_limit > 1500000, "En GPU de 8 GB el limite seguro debe superar 1.5M");

    seird::SimParams p;
    p.infection_radius = 4.5f;
    seird::cuda_adjust_domain_for_agents(p, 4000000);

    ASSERT_TRUE(p.cell_size >= p.infection_radius, "cell_size debe ser >= infection_radius");
    ASSERT_TRUE(p.world_width >= 1999.0f && p.world_height >= 1999.0f,
                "Para 4M agentes el dominio debe escalar ~2x");
    ASSERT_TRUE(std::fabs(p.cam_cx - 0.5f * p.world_width) < 1e-3f, "cam_cx centrado");
    ASSERT_TRUE(std::fabs(p.cam_cy - 0.5f * p.world_height) < 1e-3f, "cam_cy centrado");
    ASSERT_TRUE(p.grid_w * p.cell_size >= p.world_width, "grid_w cubre world_width");
    ASSERT_TRUE(p.grid_h * p.cell_size >= p.world_height, "grid_h cubre world_height");

    std::printf("[PASS] test_hardware_limits_and_domain\n");
    return 0;
}

int test_population_conservation_and_telemetry() {
    constexpr int N = 50000;
    constexpr int INIT_INF = 500;

    seird::SimParams p;
    seird::cuda_adjust_domain_for_agents(p, N);

    seird::AgentSystemDevice dev;
    ASSERT_TRUE(seird::cuda_init_simulation(dev, N, INIT_INF, p, 42ULL),
                "cuda_init_simulation debe tener exito");

    seird::CpuTelemetryEngine telemetry(32);
    seird::SimStats stats;

    for (int step = 0; step < 40; ++step) {
        seird::cuda_step_simulation(dev, p, nullptr, stats, true);
        const uint64_t total = static_cast<uint64_t>(stats.count_s)
                             + static_cast<uint64_t>(stats.count_e)
                             + static_cast<uint64_t>(stats.count_i)
                             + static_cast<uint64_t>(stats.count_r)
                             + static_cast<uint64_t>(stats.count_d);
        ASSERT_TRUE(total == static_cast<uint64_t>(N), "Conservacion S+E+I+R+D == N");
        ASSERT_TRUE(std::isfinite(stats.max_cell_z), "max_cell_z finito");
        ASSERT_TRUE(std::isfinite(stats.sim_day), "sim_day finito");

        const seird::TelemetryMetrics tm = telemetry.update(stats, p);
        ASSERT_TRUE(std::isfinite(tm.rt), "Rt finito");
        ASSERT_TRUE(std::isfinite(tm.rt_p05), "rt_p05 finito");
        ASSERT_TRUE(std::isfinite(tm.rt_p95), "rt_p95 finito");
        ASSERT_TRUE(std::isfinite(tm.prob_rt_gt_1), "prob_rt_gt_1 finito");
        ASSERT_TRUE(std::isfinite(tm.r0_ngm), "r0_ngm finito");
        ASSERT_TRUE(tm.rt_p05 <= tm.rt_p95 + 1e-5f, "rt_p05 <= rt_p95");
        ASSERT_TRUE(tm.prob_rt_gt_1 >= 0.0f && tm.prob_rt_gt_1 <= 1.0f, "prob_rt_gt_1 en [0,1]");
        ASSERT_TRUE(std::isfinite(tm.z_score), "z_score finito");
        ASSERT_TRUE(std::isfinite(tm.new_cases_rate), "new_cases_rate finito");
        ASSERT_TRUE(std::isfinite(tm.cfr), "cfr finito");
    }

    seird::cuda_free_simulation(dev);
    std::printf("[PASS] test_population_conservation_and_telemetry\n");
    return 0;
}

int test_deterministic_reproducibility() {
    constexpr int N = 20000;
    constexpr int INIT_INF = 200;
    constexpr uint64_t SEED = 12345ULL;

    seird::SimParams p;
    seird::cuda_adjust_domain_for_agents(p, N);

    auto run_once = [&](seird::SimStats& final_stats) -> bool {
        seird::AgentSystemDevice dev;
        if (!seird::cuda_init_simulation(dev, N, INIT_INF, p, SEED)) return false;
        seird::SimStats s;
        for (int step = 0; step < 20; ++step) {
            seird::cuda_step_simulation(dev, p, nullptr, s, true);
        }
        final_stats = s;
        seird::cuda_free_simulation(dev);
        return true;
    };

    seird::SimStats s1, s2;
    ASSERT_TRUE(run_once(s1), "Primera corrida determinista");
    ASSERT_TRUE(run_once(s2), "Segunda corrida determinista");

    ASSERT_TRUE(s1.count_s == s2.count_s, "count_s identico");
    ASSERT_TRUE(s1.count_e == s2.count_e, "count_e identico");
    ASSERT_TRUE(s1.count_i == s2.count_i, "count_i identico");
    ASSERT_TRUE(s1.count_r == s2.count_r, "count_r identico");
    ASSERT_TRUE(s1.count_d == s2.count_d, "count_d identico");
    ASSERT_TRUE(s1.new_exposures == s2.new_exposures, "new_exposures identico");
    ASSERT_TRUE(s1.hotspot_cells == s2.hotspot_cells, "hotspot_cells identico");

    std::printf("[PASS] test_deterministic_reproducibility\n");
    return 0;
}

int test_fase2_interventions_conservation() {
    constexpr int N = 30000;
    constexpr int INIT_INF = 300;

    seird::SimParams p;
    seird::cuda_adjust_domain_for_agents(p, N);
    p.focal_quarantine      = true;
    p.urban_clustering      = 0.4f;
    p.vaccination_rate      = 0.05f;
    p.immunity_days         = 2.0f;
    p.hospital_capacity_pct = 0.01f;

    seird::AgentSystemDevice dev;
    ASSERT_TRUE(seird::cuda_init_simulation(dev, N, INIT_INF, p, 999ULL),
                "cuda_init_simulation Fase 2");

    seird::SimStats stats;
    for (int step = 0; step < 60; ++step) {
        seird::cuda_step_simulation(dev, p, nullptr, stats, true);
        const uint64_t total = static_cast<uint64_t>(stats.count_s)
                             + static_cast<uint64_t>(stats.count_e)
                             + static_cast<uint64_t>(stats.count_i)
                             + static_cast<uint64_t>(stats.count_r)
                             + static_cast<uint64_t>(stats.count_d);
        ASSERT_TRUE(total == static_cast<uint64_t>(N), "Conservacion Fase 2 S+E+I+R+D == N");
    }
    ASSERT_TRUE(stats.count_r > 0, "Vacunacion/recuperacion debe generar agentes R");

    seird::cuda_free_simulation(dev);
    std::printf("[PASS] test_fase2_interventions_conservation\n");
    return 0;
}

int test_exact_incidence_under_vaccination() {
    constexpr int N = 20000;
    constexpr int INIT_INF = 100;

    seird::SimParams p;
    seird::cuda_adjust_domain_for_agents(p, N);
    p.beta             = 0.0f;
    p.vaccination_rate = 0.25f;

    seird::AgentSystemDevice dev;
    ASSERT_TRUE(seird::cuda_init_simulation(dev, N, INIT_INF, p, 777ULL),
                "cuda_init_simulation vacunacion sin contagio");

    seird::CpuTelemetryEngine telemetry(32);
    seird::SimStats stats;
    for (int step = 0; step < 25; ++step) {
        seird::cuda_step_simulation(dev, p, nullptr, stats, true);
        const seird::TelemetryMetrics tm = telemetry.update(stats, p);
        ASSERT_TRUE(stats.new_exposures == 0, "Con beta=0 no debe haber nuevas exposiciones S->E");
        ASSERT_TRUE(tm.new_cases_rate == 0.0f, "Con beta=0 new_cases_rate debe ser 0 aunque haya vacunacion");
    }
    ASSERT_TRUE(stats.count_r > 500, "La vacunacion debe haber generado recuperados R > 500");

    seird::cuda_free_simulation(dev);
    std::printf("[PASS] test_exact_incidence_under_vaccination\n");
    return 0;
}

int test_ode_rk4_and_ngm_validation() {
    const double attack_r0_2 = seird::sir_analytical_final_attack_rate(2.0);
    const double peak_r0_2   = seird::sir_analytical_peak_fraction(2.0);
    ASSERT_TRUE(std::fabs(attack_r0_2 - 0.796812) < 1e-3, "Tasa de ataque analitica R0=2");
    ASSERT_TRUE(std::fabs(peak_r0_2   - 0.153426) < 1e-3, "Fraccion pico analitica R0=2");

    seird::SimParams p_sir;
    p_sir.incubation_days  = 0.5f;
    p_sir.infectious_days  = 5.0f;
    p_sir.fatality_rate    = 0.0f;
    p_sir.vaccination_rate = 0.0f;
    p_sir.immunity_days    = 0.0f;

    constexpr double POP = 100000.0;
    const auto traj = seird::ode_seirds_simulate(POP, 100.0, 0.4, p_sir, 120.0, 1.0 / 48.0);
    ASSERT_TRUE(!traj.empty(), "Trayectoria ODE no vacia");

    const seird::OdeState& last = traj.back();
    const double total_ode = last.s + last.e + last.i + last.r + last.d;
    ASSERT_TRUE(std::fabs(total_ode - POP) < 1e-3, "Conservacion ODE RK4");

    const double attack_ode = 1.0 - (last.s / POP);
    ASSERT_TRUE(std::fabs(attack_ode - attack_r0_2) < 0.01, "Tasa de ataque ODE coincide con SIR analitico");

    seird::SimParams p_ngm;
    p_ngm.base_speed         = 4000.0f;
    p_ngm.quarantine_factor  = 1.0f;
    p_ngm.superspreader_frac = 0.0f;
    const double beta_homog   = seird::compute_abm_homogeneous_beta(50000, p_ngm);
    const double beta_kin_hi  = seird::compute_abm_kinetic_beta(50000, p_ngm);
    ASSERT_TRUE(std::fabs(beta_kin_hi - beta_homog) < 1e-4, "Beta cinetico converge a homogeneo en alta difusion");

    const double r0_homog   = beta_homog * static_cast<double>(p_ngm.infectious_days);
    const double r0_ngm     = seird::compute_r0_ngm(50000, p_ngm);
    const double ratio      = (r0_homog > 0.0) ? (r0_ngm / r0_homog) : 0.0;
    ASSERT_TRUE(ratio >= 1.015 && ratio <= 1.030, "Factor estructural NGM (1 + CV^2) en [1.015, 1.030]");

    seird::SimParams p_norm;
    p_norm.superspreader_frac = 0.0f;
    const double r0_normal    = seird::compute_r0_ngm(50000, p_norm);
    const double r0_norm_hom  = seird::compute_abm_homogeneous_beta(50000, p_norm) * static_cast<double>(p_norm.infectious_days);
    ASSERT_TRUE(r0_normal < r0_norm_hom, "R0 NGM cinetico a velocidad normal < limite homogeneo");

    p_norm.quarantine_factor = 0.10f;
    const double r0_quar = seird::compute_r0_ngm(50000, p_norm);
    ASSERT_TRUE(r0_quar < r0_normal, "Cuarentena estricta reduce R0 NGM");

    std::printf("[PASS] test_ode_rk4_and_ngm_validation (attack_ode=%.4f vs sir=%.4f, ngm/homog=%.4f, r0_norm=%.2f, r0_quar=%.2f)\n",
                attack_ode, attack_r0_2, ratio, r0_normal, r0_quar);
    return 0;
}

int test_high_diffusion_abm_convergence() {
    constexpr int N = 20000;
    constexpr int INIT_INF = 200;

    seird::SimParams p;
    p.world_width           = 1000.0f;
    p.world_height          = 1000.0f;
    p.infection_radius      = 2.5f;
    p.cell_size             = 4.0f;
    p.grid_w                = 250;
    p.grid_h                = 250;
    p.beta                  = 0.635f;
    p.incubation_days       = 0.5f;
    p.infectious_days       = 5.0f;
    p.fatality_rate         = 0.0f;
    p.urban_clustering      = 0.0f;
    p.focal_quarantine      = false;
    p.vaccination_rate      = 0.0f;
    p.immunity_days         = 0.0f;
    p.superspreader_frac    = 0.0f;
    p.base_speed            = 4000.0f;
    p.dt                    = 1.0f / 48.0f;

    seird::AgentSystemDevice dev;
    ASSERT_TRUE(seird::cuda_init_simulation(dev, N, INIT_INF, p, 31415ULL),
                "cuda_init_simulation alta difusion");

    seird::SimStats stats;
    constexpr int TOTAL_STEPS = 2160; // 45 dias simulados
    for (int step = 0; step < TOTAL_STEPS; ++step) {
        const bool sample = ((step % 24) == 23) || (step == TOTAL_STEPS - 1);
        seird::cuda_step_simulation(dev, p, nullptr, stats, sample);
    }

    const double a_abm    = 1.0 - (static_cast<double>(stats.count_s) / static_cast<double>(N));
    const double beta_eff = seird::compute_abm_homogeneous_beta(N, p);
    const double r0_homog = beta_eff * static_cast<double>(p.infectious_days);
    const auto ode_traj   = seird::ode_seirds_simulate(N, INIT_INF, beta_eff, p, 45.0, p.dt);
    const double a_ode    = 1.0 - (ode_traj.back().s / static_cast<double>(N));
    const double a_theory = seird::sir_analytical_final_attack_rate(r0_homog);

    std::printf("[INFO] Convergencia alta difusion (R0=%.3f): ABM=%.4f, ODE=%.4f, SIR=%.4f\n",
                r0_homog, a_abm, a_ode, a_theory);
    ASSERT_TRUE(std::fabs(a_abm - a_ode) < 0.08, "Convergencia ABM vs ODE en limite de alta difusion");

    seird::cuda_free_simulation(dev);
    std::printf("[PASS] test_high_diffusion_abm_convergence\n");
    return 0;
}

int test_lagrangian_and_superspreaders() {
    constexpr int N = 30000;
    constexpr int INIT_INF = 150;

    auto run_case = [&](float super_frac, float super_mult, uint64_t& total_infected_out) -> bool {
        seird::SimParams p;
        seird::cuda_adjust_domain_for_agents(p, N);
        p.urban_clustering   = 0.35f;
        p.superspreader_frac = super_frac;
        p.superspreader_mult = super_mult;

        seird::AgentSystemDevice dev;
        if (!seird::cuda_init_simulation(dev, N, INIT_INF, p, 54321ULL)) return false;

        seird::SimStats stats;
        for (int step = 0; step < 60; ++step) {
            seird::cuda_step_simulation(dev, p, nullptr, stats, true);
            const uint64_t total = static_cast<uint64_t>(stats.count_s)
                                 + static_cast<uint64_t>(stats.count_e)
                                 + static_cast<uint64_t>(stats.count_i)
                                 + static_cast<uint64_t>(stats.count_r)
                                 + static_cast<uint64_t>(stats.count_d);
            if (total != static_cast<uint64_t>(N)) {
                seird::cuda_free_simulation(dev);
                return false;
            }
        }
        total_infected_out = static_cast<uint64_t>(N) - static_cast<uint64_t>(stats.count_s);
        seird::cuda_free_simulation(dev);
        return true;
    };

    uint64_t infected_base = 0;
    uint64_t infected_super = 0;
    ASSERT_TRUE(run_case(0.0f, 1.0f, infected_base), "Corrida Lagrangiana base conserva poblacion");
    ASSERT_TRUE(run_case(0.15f, 5.0f, infected_super), "Corrida Lagrangiana con superdispersores conserva poblacion");
    ASSERT_TRUE(infected_super >= infected_base, "Superdispersores incrementan contagios acumulados");

    std::printf("[PASS] test_lagrangian_and_superspreaders (base=%llu, super=%llu)\n",
                static_cast<unsigned long long>(infected_base),
                static_cast<unsigned long long>(infected_super));
    return 0;
}

int test_monte_carlo_ensemble() {
    constexpr int K = 8;
    constexpr int N_REP = 10000;
    constexpr int INIT_INF = 100;

    seird::SimParams p;
    seird::cuda_adjust_domain_for_agents(p, N_REP);
    p.num_replicas = K;

    seird::AgentSystemDevice dev;
    ASSERT_TRUE(seird::cuda_init_simulation(dev, N_REP, INIT_INF, p, 2026ULL),
                "cuda_init_simulation Monte Carlo K=8");

    seird::CpuEnsembleEngine ensemble(K, 32);
    seird::SimStats stats;
    std::vector<seird::SimStats> rep_stats;
    seird::EnsembleMetrics em;

    for (int step = 0; step < 30; ++step) {
        seird::cuda_step_simulation(dev, p, nullptr, stats, true, &rep_stats);
        ASSERT_TRUE(rep_stats.size() == static_cast<std::size_t>(K), "rep_stats debe tener K elementos");
        for (int r = 0; r < K; ++r) {
            const seird::SimStats& rs = rep_stats[static_cast<std::size_t>(r)];
            const uint64_t total = static_cast<uint64_t>(rs.count_s)
                                 + static_cast<uint64_t>(rs.count_e)
                                 + static_cast<uint64_t>(rs.count_i)
                                 + static_cast<uint64_t>(rs.count_r)
                                 + static_cast<uint64_t>(rs.count_d);
            ASSERT_TRUE(total == static_cast<uint64_t>(N_REP), "Conservacion S+E+I+R+D == N_REP en cada replica");
        }
        em = ensemble.update(rep_stats, p, N_REP);
    }

    bool any_diff = false;
    for (int r = 1; r < K; ++r) {
        if (rep_stats[static_cast<std::size_t>(r)].count_s != rep_stats[0].count_s ||
            rep_stats[static_cast<std::size_t>(r)].count_e != rep_stats[0].count_e) {
            any_diff = true;
            break;
        }
    }
    ASSERT_TRUE(any_diff, "Las replicas deben tener trayectorias estocasticas independientes");
    ASSERT_TRUE(em.i_p05 <= em.i_p50 + 1e-6f && em.i_p50 <= em.i_p95 + 1e-6f, "Orden percentiles i_p05 <= i_p50 <= i_p95");
    ASSERT_TRUE(em.rt_p05 <= em.rt_p50 + 1e-6f && em.rt_p50 <= em.rt_p95 + 1e-6f, "Orden percentiles rt_p05 <= rt_p50 <= rt_p95");
    ASSERT_TRUE(em.prob_rt_gt_1 >= 0.0f && em.prob_rt_gt_1 <= 1.0f, "prob_rt_gt_1 en [0,1]");

    seird::cuda_free_simulation(dev);
    std::printf("[PASS] test_monte_carlo_ensemble (K=%d, I_P50=%.3f%%, Rt_P50=%.2f, Ext=%.1f%%)\n",
                K, em.i_p50 * 100.0f, em.rt_p50, em.extinction_pct);
    return 0;
}

int main() {
    if (test_hardware_limits_and_domain() != 0) return 1;
    if (test_population_conservation_and_telemetry() != 0) return 1;
    if (test_deterministic_reproducibility() != 0) return 1;
    if (test_fase2_interventions_conservation() != 0) return 1;
    if (test_exact_incidence_under_vaccination() != 0) return 1;
    if (test_ode_rk4_and_ngm_validation() != 0) return 1;
    if (test_high_diffusion_abm_convergence() != 0) return 1;
    if (test_lagrangian_and_superspreaders() != 0) return 1;
    if (test_monte_carlo_ensemble() != 0) return 1;
    std::printf("=== Todos los tests de invariantes pasaron correctamente ===\n");
    return 0;
}
