# API Reference — Miasma HPC Engine

Complete reference of types, enumerations, structures, CUDA engine functions, and CPU/OpenMP analytical APIs in `namespace seird`.

---

## 1. Constants & Enumerations (`include/types.hpp`)

### 1.1 Compile-Time Constants
| Constant | Value | Description |
| :--- | :--- | :--- |
| `FRAME_WIDTH` | `1920` | Render framebuffer width in pixels (1080p). |
| `FRAME_HEIGHT` | `1080` | Render framebuffer height in pixels (1080p). |
| `MAX_REPLICAS` | `64` | Maximum concurrent Monte Carlo replicas in VRAM. |
| `MAX_URBAN_ZONES` | `32` | Maximum configurable `UrbanZone` districts per simulation. |
| `RISK_GROUP_MASK` | `0x03u` | Bitmask for `RiskGroup` in `AgentSystemDevice::risk_group`. |
| `FLAG_SUPERSPREADER` | `0x80u` | Bit 7 flag marking superspreader agents (zero extra bytes). |

### 1.2 Enumerations
* **`AgentState : uint8_t`**: `STATE_S` (0, Susceptible), `STATE_E` (1, Exposed), `STATE_I` (2, Infected), `STATE_R` (3, Recovered), `STATE_D` (4, Deceased).
* **`RiskGroup : uint8_t`**: `RISK_LOW` (0, ~50% pop, 0.25× fatality), `RISK_MEDIUM` (1, ~35% pop, 1.0× fatality), `RISK_HIGH` (2, ~15% pop, 3.5× fatality).
* **`ZoneType : uint8_t`**: `ZONE_HOME` (0), `ZONE_WORK` (1), `ZONE_SCHOOL` (2), `ZONE_HOSPITAL` (3), `ZONE_LEISURE` (4), `ZONE_TRANSIT` (5).
* **`InfectionSource : uint8_t`**: `SRC_NONE` (0), `SRC_HOME` (1), `SRC_WORK` (2), `SRC_TRANSIT` (3), `SRC_HOSPITAL` (4), `SRC_LEISURE` (5).
* **`UrbanPreset : int`**: `PRESET_ORGANIC_POLES` (0, 8 organic hubs), `PRESET_MONOCENTRIC` (1, central work/hospital core + residential periphery), `PRESET_POLYCENTRIC` (2, multiple connected sub-centers).
* **`RenderMode : int`**: `RENDER_AGENTS` (0, priority-ordered agents), `RENDER_HEATMAP` (1, bilinear density + $Z > 2.0$ hotspots), `RENDER_HYBRID` (2, heatmap background + `E`/`I` agents).
* **`RtCategory : uint8_t`** (`include/cpu_engine.hpp`): CDC CFA categories based on $\mathbb{P}(R_t > 1)$:
  * `RT_DECLINING` (`< 10%`), `RT_LIKELY_DECLINING` (`10%..25%`), `RT_UNCERTAIN` (`25%..75%`), `RT_LIKELY_GROWING` (`75%..90%`), `RT_GROWING` (`> 90%`).

---

## 2. Core Data Structures

### 2.1 `UrbanZone` (`include/types.hpp`)
Describes a rectangular urban district:
* `float cx, cy`: District center coordinates in world units.
* `float radius, half_w, half_h`: Reference radius and half-dimensions (default `40.0f`).
* `ZoneType type`: Functional category (`ZONE_HOME` .. `ZONE_TRANSIT`).
* `float capacity`: District capacity (e.g., hospital beds).

### 2.2 `SimParams` (`include/types.hpp`, `alignas(32)`)
Simulation configuration passed to both GPU kernels and CPU analytics:

| Group | Fields & Defaults |
| :--- | :--- |
| **Domain & Grid** | `world_width = 1000.0f`, `world_height = 1000.0f`, `grid_w = 256`, `grid_h = 256`, `cell_size = 4.0f` |
| **SEIRD** | `infection_radius = 3.5f`, `beta = 0.25f`, `incubation_days = 5.0f`, `infectious_days = 10.0f`, `fatality_rate = 0.02f` |
| **Mobility & Time** | `dt = 1.0f / 48.0f` (0.5 h/step), `base_speed = 15.0f`, `quarantine_factor = 1.0f` |
| **Epidemiological Realism** | `urban_clustering = 0.25f`, `focal_quarantine = false`, `hospital_capacity_pct = 0.05f`, `vaccination_rate = 0.0f`, `immunity_days = 0.0f`, `superspreader_frac = 0.10f`, `superspreader_mult = 4.0f`, `num_replicas = 1` |
| **Camera & Render** | `render_mode = RENDER_HYBRID`, `enable_trails = true`, `show_rings = true`, `cam_cx = 500.0f`, `cam_cy = 500.0f`, `cam_zoom = 1.0f` |
| **Urban & Ensemble** | `enable_urban_layout = false`, `urban_preset = PRESET_MONOCENTRIC`, `num_urban_zones = 0`, `urban_zones[MAX_URBAN_ZONES]`, `household_quarantine = false`, `close_schools_work = false`, `selected_home_id = -1`, `physics_substeps = 1`, `ensemble_grid_view = false`, `focused_replica = 0` |

### 2.3 `SimStats` (`include/types.hpp`)
Per-step telemetry snapshot returned from GPU:
* `count_s`, `count_e`, `count_i`, `count_r`, `count_d` (`uint32_t`): Compartment counts.
* `new_exposures` (`uint32_t`): Exact `S -> E` transitions since the previous sampling step.
* `hotspot_cells` (`uint32_t`), `max_cell_z` (`float`): Count of cells with $Z > 2.0$ and peak spatial Z-score.
* `sim_day` (`float`): Current simulated day.
* `src_home`, `src_work`, `src_transit`, `src_hospital`, `src_leisure` (`uint32_t`): Infection source attribution counts.
* `hospitalized_count` (`uint32_t`): Currently hospitalized infected agents.

### 2.4 `AgentSystemDevice` (`include/types.hpp`)
Holds SoA device pointers (`num_agents`, `agents_per_replica`, `num_replicas`, `num_cells`), kinematics (`x`, `y`, `vx`, `vy`), state (`state`, `risk_group`, `home_zone`, `rep_id`, `state_timer`), urban/family (`home_id`, `work_zone`, `infection_source`, `infection_flash`), spatial hash (`cell_id`, `agent_idx`, `cell_start`, `cell_end`), opaque `void* rng_states`, and `uint8_t* framebuffer_rgba`.

### 2.5 Helper Function (`include/types.hpp`)
* `void compute_mosaic_grid(int k, int& cols, int& rows)`: Computes `(cols, rows)` layout for $K$ Monte Carlo replicas (from `1x1` up to `8xN`).

---

## 3. CUDA Engine API (`include/cuda_engine.cuh`)

### 3.1 Simulation Lifecycle & Stepping
* `bool cuda_init_simulation(AgentSystemDevice& dev, int agents_per_replica, int initial_infected, const SimParams& params, uint64_t seed = 42ULL)`
  Allocates SoA ping-pong buffers, CUB temporary storage, and 1080p framebuffers in VRAM; initializes agents, family households, urban assignments, and patient-zero cases.
* `void cuda_step_simulation(AgentSystemDevice& dev, const SimParams& params, uint8_t* host_framebuffer_rgba, SimStats& out_stats, bool compute_stats = true, std::vector<SimStats>* out_replica_stats = nullptr)`
  Advances one physics/SEIRD step on `g_sim_stream`. Optionally reduces per-replica `SimStats` and copies the 1080p framebuffer to `host_framebuffer_rgba` (if non-null).
* `void cuda_render_1080p_device(AgentSystemDevice& dev, const SimParams& params)`
  Renders the 1080p RGBA framebuffer in VRAM without copying to host memory.
* `void cuda_sync_simulation()`
  Synchronizes `g_sim_stream`.
* `bool cuda_copy_framebuffer_to_device(const AgentSystemDevice& dev, uint8_t* dst_device_rgba)`
  Device-to-device copy of `framebuffer_rgba` for double-buffered display.
* `void cuda_free_simulation(AgentSystemDevice& dev)`
  Frees all VRAM and pinned host allocations.

### 3.2 Hardware Query, Domain Scaling & Urban Presets
* `HardwareLimits cuda_query_hardware_limits(size_t reserve_vram_mb = 1024)`
  Returns `vram_free_mb`, `vram_total_mb`, `bytes_per_agent`, `max_agents_vram`, and `safe_agents_limit` (capped at 20M for Windows WDDM safety).
* `void cuda_adjust_domain_for_agents(SimParams& params, int num_agents)`
  Updates `cell_size`, `world_width`, `world_height`, `grid_w`, `grid_h`, and camera center to maintain safe cell density.
* `void cuda_build_urban_preset(SimParams& params, int preset)`
  Populates `params.urban_zones` and `params.num_urban_zones` for `PRESET_ORGANIC_POLES`, `PRESET_MONOCENTRIC`, or `PRESET_POLYCENTRIC`.

### 3.3 Family Inspection, Timing & Benchmarking
* `int cuda_pick_nearest_home_id(const AgentSystemDevice& dev, float world_x, float world_y)`
  Finds the `home_id` of the nearest agent in replica 0 using `atomicMin`.
* `bool cuda_get_family_track(const AgentSystemDevice& dev, const SimParams& params, int home_id, FamilyTrackInfo& out_info)`
  Populates `FamilyTrackInfo` (home coordinates and up to 8 `FamilyMemberInfo` entries with position, work destination, state, and risk group).
* `void cuda_timer_begin()` / `float cuda_timer_end_ms()`
  CUDA event timing on `g_sim_stream`.
* `void cuda_get_vram_info(float& used_mb, float& total_mb)`
  Queries current VRAM usage in MB.
* `bool cuda_run_benchmark(int num_agents, int warmup_steps, int measure_steps, const SimParams& params, BenchmarkResult& out_res)`
  Headless GPU benchmark returning `avg_step_ms`, `avg_render_ms`, `agents_per_sec`, and `vram_used_mb`.

---

## 4. CPU / OpenMP Analytical & Calibration API (`include/cpu_engine.hpp`)

### 4.1 Deterministic ODE, NGM & Analytical Helpers
* `OdeState ode_seirds_step_rk4(const OdeState& st, double beta_eff, const SimParams& params, double dt)`
  Single 4th-order Runge-Kutta step of the homogeneous SEIRDS ODE system.
* `std::vector<OdeState> ode_seirds_simulate(double total_pop, double initial_infected, double beta_eff, const SimParams& params, double total_days, double dt = 1.0 / 48.0)`
  Integrates the reference SEIRDS ODE trajectory over `total_days`.
* `double compute_abm_homogeneous_beta(int num_agents, const SimParams& params)`
  Estimates homogeneous equivalent $\beta_{\text{eff}}$ from the ABM infection disk.
* `double compute_abm_kinetic_beta(int num_agents, const SimParams& params)`
  Estimates 2D kinetic $\beta_{\text{eff}}$ accounting for finite speed, quarantine, and superspreading.
* `double compute_r0_ngm(int num_agents, const SimParams& params)`
  Computes structural $R_0$ as the spectral radius of the $3 \times 3$ `RiskGroup` Next-Generation Matrix.
* `double sir_analytical_peak_fraction(double r0)` / `double sir_analytical_final_attack_rate(double r0)`
  Exact Kermack–McKendrick SIR peak $I_{\max}/N$ and final attack rate $1 - S(\infty)/N$.

### 4.2 Telemetry & Ensemble Engines
* **`CpuTelemetryEngine`**:
  * `explicit CpuTelemetryEngine(std::size_t window_size = 64)`
  * `TelemetryMetrics update(const SimStats& stats, const SimParams& params)` — Computes `rt`, `rt_p05`, `rt_p95`, `prob_rt_gt_1`, `rt_category` (`rt_category_label`), `r0_ngm`, `z_score`, `new_cases_rate`, `cfr`, `ifr`, and `outbreak_alert`.
  * `void reset()` / `const std::vector<TelemetryMetrics>& history() const`
* **`CpuEnsembleEngine`**:
  * `explicit CpuEnsembleEngine(int num_replicas = 1, std::size_t window_size = 64)`
  * `void reset(int num_replicas)`
  * `EnsembleMetrics update(const std::vector<SimStats>& rep_stats, const SimParams& params, int agents_per_replica)` — Parallel OpenMP update returning `i_p05`, `i_p10`, `i_p50`, `i_p90`, `i_p95`, `rt_p05`, `rt_p50`, `rt_p95`, `prob_rt_gt_1`, and `extinction_pct`.
  * `const std::vector<EnsembleMetrics>& history() const`

### 4.3 Forecast Scoring, Sensitivity & ABC-SMC Calibration
* `ForecastScoreMetrics compute_ensemble_forecast_scores(const std::vector<EnsembleMetrics>& ens_traj, const std::vector<float>& obs_i_frac)`
  Computes `mae`, `rmse`, `is80`, `wis`, `coverage_80`, and `coverage_90`.
* `std::vector<SensitivityRow> run_global_sensitivity_sweep(int num_agents, int initial_infected, const SimParams& base_params, const std::vector<float>& betas, const std::vector<float>& quar_factors, const std::vector<float>& super_fracs, double horizon_days = 60.0)`
  Parallel OpenMP grid sweep returning `r0_ngm`, `peak_i_frac`, and `final_attack_rate` per parameter combination.
* `AbcSmcResult calibrate_abc_smc(int num_agents, int initial_infected, const SimParams& base_params, const std::vector<float>& obs_days, const std::vector<float>& obs_i_frac, const AbcSmcConfig& cfg = AbcSmcConfig{})`
  Runs `num_generations` of Sequential Monte Carlo ABC over `(beta, incubation_days, infectious_days)` with `num_particles`, returning `beta_p05`, `beta_p50`, `beta_p95`, `incub_p50`, `infec_p50`, `final_epsilon`, weighted `posterior` particles, and `posterior_scores`.
