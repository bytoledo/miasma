#include "cpu_engine.hpp"
#include "cuda_engine.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// Caso 1: Validación analítica Kermack-McKendrick / Luque et al. (2020)
// SARS-CoV-2 (R0 = 2.5) y Sarampión (R0 = 10.0) + inmunidad de grupo p_c = 1 - 1/R0
void benchmark_analytical_thresholds() {
    std::printf("\n=== 1. Umbrales Analiticos (Kermack-McKendrick / Luque 2020) ===\n");

    const double r0_cases[] = { 1.5, 2.5, 10.0 };
    for (double r0 : r0_cases) {
        const double pc_theory     = 1.0 - 1.0 / r0;
        const double attack_theory = seird::sir_analytical_final_attack_rate(r0);
        const double peak_theory   = seird::sir_analytical_peak_fraction(r0);

        // Integrar ODE SIR equivalente (incubation_days = 0.01 -> límite SIR, alpha = 0)
        seird::SimParams p;
        p.incubation_days = 0.01f;
        p.infectious_days = 10.0f;
        p.fatality_rate   = 0.0f;
        p.vaccination_rate = 0.0f;
        p.immunity_days   = 0.0f;
        const double beta_ode = r0 / static_cast<double>(p.infectious_days);

        const auto traj = seird::ode_seirds_simulate(100000.0, 10.0, beta_ode, p, 250.0, 1.0 / 48.0);
        double peak_ode = 0.0;
        for (const auto& st : traj) peak_ode = std::max(peak_ode, st.i / 100000.0);
        const double attack_ode = 1.0 - (traj.back().s / 100000.0);

        // Verificar inmunidad de grupo: con S(0) = (1 - pc) * N, el brote no debe crecer
        const double s0_immune = (1.0 - pc_theory) * 100000.0;
        seird::OdeState st_imm{ s0_immune - 10.0, 0.0, 10.0, 100000.0 - s0_immune, 0.0, 0.0 };
        double max_i_imm = 10.0;
        for (int k = 0; k < 48 * 60; ++k) {
            st_imm = seird::ode_seirds_step_rk4(st_imm, beta_ode, p, 1.0 / 48.0);
            max_i_imm = std::max(max_i_imm, st_imm.i);
        }

        std::printf("  R0=%.1f | pc=%.1f%% | Ataque: Teoria=%.4f vs ODE=%.4f (err=%.2e) | Pico: Teoria=%.4f vs ODE=%.4f | MaxI@pc=%.1f\n",
                    r0, pc_theory * 100.0, attack_theory, attack_ode,
                    std::fabs(attack_theory - attack_ode), peak_theory, peak_ode, max_i_imm);
    }
}

// Caso 2: Efecto del periodo latente E (IDM ModelingHub 2026)
// E ralentiza el crecimiento inicial y retrasa el pico, pero preserva el tamaño final
void benchmark_latent_period_invariance() {
    std::printf("\n=== 2. Efecto del Periodo Latente E (IDM ModelingHub) ===\n");
    constexpr double N = 100000.0;
    constexpr double I0 = 50.0;
    constexpr double R0 = 2.5;

    auto run_ode = [&](float incub_days, double& peak_day, double& peak_frac, double& attack) {
        seird::SimParams p;
        p.incubation_days = incub_days;
        p.infectious_days = 10.0f;
        p.fatality_rate   = 0.0f;
        const double beta = R0 / static_cast<double>(p.infectious_days);
        const auto traj = seird::ode_seirds_simulate(N, I0, beta, p, 240.0, 1.0 / 48.0);
        peak_frac = 0.0;
        peak_day  = 0.0;
        for (const auto& st : traj) {
            if (st.i / N > peak_frac) {
                peak_frac = st.i / N;
                peak_day  = st.day;
            }
        }
        attack = 1.0 - (traj.back().s / N);
    };

    double d_sir, p_sir, a_sir;
    double d_seir, p_seir, a_seir;
    run_ode(0.01f, d_sir, p_sir, a_sir);
    run_ode(5.0f,  d_seir, p_seir, a_seir);

    std::printf("  SIR  (T_incub=0d): Pico=%.2f%% en dia %.1f | Ataque final=%.4f\n", p_sir * 100.0, d_sir, a_sir);
    std::printf("  SEIR (T_incub=5d): Pico=%.2f%% en dia %.1f | Ataque final=%.4f (diff=%.2e)\n",
                p_seir * 100.0, d_seir, a_seir, std::fabs(a_sir - a_seir));
}

// Caso 3: Gripe de 1918 en GPU ABM (Markel et al. 2007: San Luis vs Pittsburgh y rebote)
void benchmark_1918_npi_gpu_abm() {
    std::printf("\n=== 3. Gripe 1918 en GPU ABM (Markel et al. 2007) ===\n");
    constexpr int N = 20000;
    constexpr int INIT_INF = 60;
    constexpr int DAYS = 90;
    constexpr int STEPS = DAYS * 48;

    auto run_abm = [&](int mode, double& peak_i_pct, double& deaths_pct, double& second_peak_pct) {
        seird::SimParams p;
        seird::cuda_adjust_domain_for_agents(p, N);
        seird::cuda_build_urban_preset(p, seird::PRESET_POLYCENTRIC);
        p.beta            = 0.065f;
        p.incubation_days = 2.0f;
        p.infectious_days = 6.0f;
        p.fatality_rate   = 0.025f;

        seird::AgentSystemDevice dev;
        seird::cuda_init_simulation(dev, N, INIT_INF, p, 1918ULL);

        seird::SimStats stats;
        peak_i_pct      = 0.0;
        second_peak_pct = 0.0;
        for (int s = 0; s < STEPS; ++s) {
            const float day = static_cast<float>(s) * p.dt;
            if (mode == 1) {
                // NPI temprana y sostenida (San Luis)
                if (day >= 6.0f) {
                    p.household_quarantine = true;
                    p.close_schools_work   = true;
                    p.quarantine_factor    = 0.35f;
                }
            } else if (mode == 2) {
                // NPI temprana pero levantada prematuramente al dia 25 (Segundo pico)
                if (day >= 6.0f && day < 25.0f) {
                    p.household_quarantine = true;
                    p.close_schools_work   = true;
                    p.quarantine_factor    = 0.35f;
                } else if (day >= 25.0f) {
                    p.household_quarantine = false;
                    p.close_schools_work   = false;
                    p.quarantine_factor    = 1.0f;
                }
            }
            const bool sample = ((s % 24) == 23) || (s == STEPS - 1);
            seird::cuda_step_simulation(dev, p, nullptr, stats, sample);
            if (sample) {
                double i_pct = 100.0 * static_cast<double>(stats.count_i) / static_cast<double>(N);
                peak_i_pct = std::max(peak_i_pct, i_pct);
                if (day >= 28.0f) {
                    second_peak_pct = std::max(second_peak_pct, i_pct);
                }
            }
        }
        deaths_pct = 100.0 * static_cast<double>(stats.count_d) / static_cast<double>(N);
        seird::cuda_free_simulation(dev);
    };

    double pk0, d0, sp0;
    double pk1, d1, sp1;
    double pk2, d2, sp2;
    run_abm(0, pk0, d0, sp0);
    run_abm(1, pk1, d1, sp1);
    run_abm(2, pk2, d2, sp2);

    const double peak_red  = 100.0 * (pk0 - pk1) / std::max(1e-6, pk0);
    const double death_red = 100.0 * (d0 - d1) / std::max(1e-6, d0);
    std::printf("  Sin NPI (Pittsburgh-like): Pico I=%.2f%% | Mortalidad=%.2f%%\n", pk0, d0);
    std::printf("  NPI sostenida (San Luis):  Pico I=%.2f%% (-%.1f%%) | Mortalidad=%.2f%% (-%.1f%%)\n",
                pk1, peak_red, d1, death_red);
    std::printf("  NPI levantada dia 25:      Pico I=%.2f%% | Segundo pico (>=d28)=%.2f%% | Mortalidad=%.2f%%\n",
                pk2, sp2, d2);
}

// Caso 4: Convergencia ABM GPU (alta difusión) vs ODE vs Teoría SIR + Ensamble Monte Carlo
void benchmark_abm_vs_ode_monte_carlo() {
    std::printf("\n=== 4. Convergencia Estocastica ABM GPU (K=16) vs ODE RK4 ===\n");
    constexpr int K = 16;
    constexpr int N_REP = 15000;
    constexpr int INIT_INF = 150;

    seird::SimParams p;
    p.world_width        = 1000.0f;
    p.world_height       = 1000.0f;
    p.infection_radius   = 2.5f;
    p.cell_size          = 4.0f;
    p.grid_w             = 250;
    p.grid_h             = 250;
    p.beta               = 0.635f;
    p.incubation_days    = 0.5f;
    p.infectious_days    = 5.0f;
    p.fatality_rate      = 0.0f;
    p.urban_clustering   = 0.0f;
    p.superspreader_frac = 0.0f;
    p.base_speed         = 4000.0f;
    p.dt                 = 1.0f / 48.0f;
    p.num_replicas       = K;

    const double beta_eff = seird::compute_abm_kinetic_beta(N_REP, p);
    const auto ode_traj   = seird::ode_seirds_simulate(N_REP, INIT_INF, beta_eff, p, 35.0, p.dt);

    seird::AgentSystemDevice dev;
    seird::cuda_init_simulation(dev, N_REP, INIT_INF, p, 2026ULL);
    seird::CpuEnsembleEngine ensemble(K, 32);
    seird::SimStats stats;
    std::vector<seird::SimStats> rep_stats;
    std::vector<seird::EnsembleMetrics> ens_hist;
    std::vector<float> ode_i_obs;

    constexpr int STEPS = 35 * 48;
    for (int s = 0; s < STEPS; ++s) {
        const bool sample = ((s % 24) == 23);
        seird::cuda_step_simulation(dev, p, nullptr, stats, sample, sample ? &rep_stats : nullptr);
        if (sample) {
            ens_hist.push_back(ensemble.update(rep_stats, p, N_REP));
            const std::size_t oidx = std::min(ode_traj.size() - 1, static_cast<std::size_t>(s + 1));
            ode_i_obs.push_back(static_cast<float>(ode_traj[oidx].i / static_cast<double>(N_REP)));
        }
    }

    const auto scores = seird::compute_ensemble_forecast_scores(ens_hist, ode_i_obs);
    std::printf("  R0_homog=%.3f | MAE=%.4f | RMSE=%.4f | IS80=%.4f | WIS=%.4f | Cov80=%.1f%% | Cov90=%.1f%%\n",
                beta_eff * p.infectious_days, scores.mae, scores.rmse, scores.is80, scores.wis,
                scores.coverage_80 * 100.0f, scores.coverage_90 * 100.0f);
    seird::cuda_free_simulation(dev);
}

// Caso 5: Ondas amortiguadas y equilibrio endémico con pérdida de inmunidad (SEIRDS, IDM / Hethcote)
void benchmark_endemic_waves_seirds() {
    std::printf("\n=== 5. Ondas Endemicas con Perdida de Inmunidad (SEIRDS) ===\n");
    constexpr double N = 100000.0;
    seird::SimParams p;
    p.incubation_days = 3.0f;
    p.infectious_days = 7.0f;
    p.fatality_rate   = 0.0f;
    p.immunity_days   = 35.0f; // Perdida de inmunidad R -> S
    const double beta = 2.8 / static_cast<double>(p.infectious_days);

    const auto traj = seird::ode_seirds_simulate(N, 100.0, beta, p, 220.0, 1.0 / 48.0);
    int peaks = 0;
    double prev_i = 0.0, curr_i = 0.0;
    double last_peak_pct = 0.0;
    for (std::size_t k = 1; k + 1 < traj.size(); ++k) {
        prev_i = traj[k - 1].i;
        curr_i = traj[k].i;
        double next_i = traj[k + 1].i;
        if (curr_i > prev_i && curr_i > next_i && (curr_i / N) > 0.01) {
            ++peaks;
            last_peak_pct = 100.0 * curr_i / N;
        }
    }
    const double i_endemic_pct = 100.0 * traj.back().i / N;
    std::printf("  T_inm=35d, R0=2.8 | Picos detectados=%d | Ultimo pico=%.2f%% | I_final(d220)=%.2f%%\n",
                peaks, last_peak_pct, i_endemic_pct);
}

// Caso 6: Extinción estocástica temprana con R0 > 1 y efecto de superdispersión (Brauer 2017 / Estrada 2020)
void benchmark_stochastic_extinction_and_superspreading() {
    std::printf("\n=== 6. Extincion Estocastica Temprana con R0 > 1 (K=32, I0=3) ===\n");
    constexpr int K = 32;
    constexpr int N_REP = 10000;
    constexpr int INIT_INF = 3;

    auto run_ext = [&](float super_frac, float super_mult, float base_beta, double& ext_pct, double& med_i_pct) {
        seird::SimParams p;
        seird::cuda_adjust_domain_for_agents(p, N_REP);
        p.beta               = base_beta;
        p.incubation_days    = 2.0f;
        p.infectious_days    = 6.0f;
        p.fatality_rate      = 0.0f;
        p.superspreader_frac = super_frac;
        p.superspreader_mult = super_mult;
        p.num_replicas       = K;

        seird::AgentSystemDevice dev;
        seird::cuda_init_simulation(dev, N_REP, INIT_INF, p, 7777ULL);
        seird::CpuEnsembleEngine ens(K, 32);
        seird::SimStats stats;
        std::vector<seird::SimStats> rep_stats;
        seird::EnsembleMetrics em;

        for (int s = 0; s < 30 * 48; ++s) {
            const bool sample = ((s % 24) == 23);
            seird::cuda_step_simulation(dev, p, nullptr, stats, sample, sample ? &rep_stats : nullptr);
            if (sample) em = ens.update(rep_stats, p, N_REP);
        }
        ext_pct   = em.extinction_pct;
        med_i_pct = em.i_p50 * 100.0;
        seird::cuda_free_simulation(dev);
    };

    double ext_homog = 0.0, med_homog = 0.0;
    double ext_super = 0.0, med_super = 0.0;
    run_ext(0.00f, 1.0f, 0.35f, ext_homog, med_homog);
    run_ext(0.10f, 6.0f, 0.23f, ext_super, med_super);
    std::printf("  Homogeneo (super=0%%):   Extincion=%.1f%% | Mediana I(d30)=%.2f%%\n", ext_homog, med_homog);
    std::printf("  Superdispersor (10%%x6): Extincion=%.1f%% | Mediana I(d30)=%.2f%%\n", ext_super, med_super);
}

// Caso 7: Comparación Monocéntrica vs Policéntrica en GPU ABM (mismos parámetros)
void benchmark_monocentric_vs_polycentric() {
    std::printf("\n=== 7. Ciudad Monocentrica vs Policentrica (GPU ABM, N=20000) ===\n");
    constexpr int N = 20000;
    constexpr int INIT_INF = 60;

    auto run_city = [&](int preset, const char* label) {
        seird::SimParams p;
        seird::cuda_adjust_domain_for_agents(p, N);
        seird::cuda_build_urban_preset(p, preset);
        p.beta            = 0.08f;
        p.incubation_days = 3.0f;
        p.infectious_days = 7.0f;

        seird::AgentSystemDevice dev;
        seird::cuda_init_simulation(dev, N, INIT_INF, p, 4242ULL);
        seird::SimStats stats;
        double peak_i = 0.0;
        float peak_day = 0.0f;
        for (int s = 0; s < 45 * 48; ++s) {
            const bool sample = ((s % 24) == 23);
            seird::cuda_step_simulation(dev, p, nullptr, stats, sample);
            if (sample) {
                double ip = 100.0 * static_cast<double>(stats.count_i) / static_cast<double>(N);
                if (ip > peak_i) {
                    peak_i   = ip;
                    peak_day = stats.sim_day;
                }
            }
        }
        std::printf("  %-13s | Pico I=%.2f%% (dia %.1f) | H=%u W=%u Tr=%u Hp=%u L=%u\n",
                    label, peak_i, peak_day, stats.src_home, stats.src_work,
                    stats.src_transit, stats.src_hospital, stats.src_leisure);
        seird::cuda_free_simulation(dev);
    };

    run_city(seird::PRESET_MONOCENTRIC, "Monocentrica");
    run_city(seird::PRESET_POLYCENTRIC, "Policentrica");
}

// Caso 8: Identificabilidad práctica (1 parámetro vs 3 parámetros simultáneos en ABC-SMC, Raimúndez 2021)
void benchmark_identifiability_abc_smc() {
    std::printf("\n=== 8. Identificabilidad Practica ABC-SMC (Raimundez et al. 2021) ===\n");
    constexpr int N = 50000;
    constexpr int INIT_INF = 250;
    seird::SimParams true_p;
    true_p.beta            = 0.28f;
    true_p.incubation_days = 5.0f;
    true_p.infectious_days = 10.0f;

    const double beta_kin = seird::compute_abm_kinetic_beta(N, true_p);
    const auto traj = seird::ode_seirds_simulate(N, INIT_INF, beta_kin, true_p, 30.0, true_p.dt);

    std::vector<float> obs_days, obs_i;
    for (int d = 2; d <= 30; d += 2) {
        std::size_t idx = std::min(traj.size() - 1, static_cast<std::size_t>(std::llround(static_cast<double>(d) / true_p.dt)));
        obs_days.push_back(static_cast<float>(d));
        obs_i.push_back(static_cast<float>(traj[idx].i / static_cast<double>(N)));
    }

    seird::AbcSmcConfig cfg1;
    cfg1.num_particles   = 48;
    cfg1.num_generations = 4;
    cfg1.incub_min = 5.0f; cfg1.incub_max = 5.0f;
    cfg1.infec_min = 10.0f; cfg1.infec_max = 10.0f;
    cfg1.obs_noise_cv = 0.02f;
    const auto r1 = seird::calibrate_abc_smc(N, INIT_INF, true_p, obs_days, obs_i, cfg1);

    seird::AbcSmcConfig cfg3;
    cfg3.num_particles   = 48;
    cfg3.num_generations = 4;
    cfg3.obs_noise_cv    = 0.02f;
    const auto r3 = seird::calibrate_abc_smc(N, INIT_INF, true_p, obs_days, obs_i, cfg3);

    std::printf("  1-param (solo beta):    beta=%.3f [%.3f-%.3f] (ancho=%.3f) | WIS=%.4f\n",
                r1.beta_p50, r1.beta_p05, r1.beta_p95, r1.beta_p95 - r1.beta_p05, r1.posterior_scores.wis);
    std::printf("  3-param (beta, E, I):   beta=%.3f [%.3f-%.3f] (ancho=%.3f) | WIS=%.4f\n",
                r3.beta_p50, r3.beta_p05, r3.beta_p95, r3.beta_p95 - r3.beta_p05, r3.posterior_scores.wis);
}

} // namespace

int main() {
    benchmark_analytical_thresholds();
    benchmark_latent_period_invariance();
    benchmark_1918_npi_gpu_abm();
    benchmark_abm_vs_ode_monte_carlo();
    benchmark_endemic_waves_seirds();
    benchmark_stochastic_extinction_and_superspreading();
    benchmark_monocentric_vs_polycentric();
    benchmark_identifiability_abc_smc();
    std::printf("\n=== Validacion cientifica completada ===\n");
    return 0;
}
