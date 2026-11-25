# miasma-parallel (Miasma HPC Engine) — Context and Architecture Plan

This document summarizes the technical decisions, mathematical model, and implementation plan defined for the project.

## 1. Hardware and Technology Stack
* **Target GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, Blackwell `sm_120` architecture).
* **Core Stack:** C++17, NVIDIA CUDA 13.x (`cuRAND`, `CUB DeviceRadixSort`), OpenMP + AVX2 (CPU), CMake $\ge 3.20$.
* **Visualization:** Native Win32 + OpenGL interface with a GPU-rendered 1080p RGBA framebuffer and real-time telemetry charts.

## 2. Epidemiological and Computational Model
* **Model:** **Stochastic SEIRD Agent-Based Model (ABM)** in continuous 2D space.
  * **S** (Susceptible) $\xrightarrow{\beta}$ **E** (Exposed / incubating, non-infectious) $\xrightarrow{\sigma}$ **I** (Infected / infectious).
  * At the end of the infectious period ($I$): with probability $\alpha$ (fatality rate) an agent transitions to **D** (Deceased, static and non-infectious), and with probability $1 - \alpha$ transitions to **R** (Recovered / immune).
* **GPU Acceleration:** **Spatial Hashing (*Uniform Grid*)** with `cub::DeviceRadixSort::SortPairs` to evaluate contagion in $O(N)$ instead of $O(N^2)$.
* **Capacity:** Dynamic limits computed via `cudaMemGetInfo` (`cuda_query_hardware_limits`), automatically scaling the spatial domain and grid (`cuda_adjust_domain_for_agents`) to support millions of agents without saturating cells or triggering TDR.

## 3. Memory Design (SoA - *Structure of Arrays*)
Defined in [`include/types.hpp`](../include/types.hpp):
* Kinematics: `x`, `y`, `vx`, `vy` (`float*`).
* SEIRD State: `state` (`uint8_t*`: `0=S, 1=E, 2=I, 3=R, 4=D`), `state_timer` (`float*`).
* Spatial Hashing: `cell_id`, `agent_idx` (`uint32_t*`), `cell_start`, `cell_end` (`int*`).
* RNG: `rng_states` (`curandState*`, opaque in `types.hpp`).
* Visual Output: `framebuffer_rgba` (`uint8_t*`, `1920 x 1080 x 4`).

## 4. Per-Step Kernel Pipeline ($\Delta t$)
1. `kernel_move_and_hash`: Updates position $(x, y)$, applies boundary bounces/quarantine (deceased `D` agents do not move), and computes `cell_id`.
2. `cub::DeviceRadixSort::SortPairs`: Sorts `agent_idx` by `cell_id`.
3. `kernel_find_cell_bounds`: Computes `cell_start` and `cell_end` for each cell.
4. `kernel_seir_step`: Inspects agents in the agent's own cell and 8 neighbors; evaluates contagion and updates `E -> I` and `I -> R / D` transitions.
5. `kernel_render_1080p`: Projects agents onto `framebuffer_rgba` in VRAM.

## 5. Current Status and Roadmap
* **Completed:**
  * [`include/types.hpp`](../include/types.hpp)
  * [`include/cuda_engine.cuh`](../include/cuda_engine.cuh)
  * `.gitignore` and `git init`
  * [x] **Step 2:** Implement [`src/cuda_engine.cu`](../src/cuda_engine.cu).
  * [x] **Step 3:** Create [`CMakeLists.txt`](../CMakeLists.txt) and [`src/cpu_engine.cpp`](../src/cpu_engine.cpp) (OpenMP telemetry, $R_t$, outbreak $Z$-score).
  * [x] **Step 4:** Implement [`src/main.cpp`](../src/main.cpp) (native Win32 + OpenGL UI and reports).
  * [x] **Phase 0:** Dynamic VRAM limits (`cuda_query_hardware_limits`), domain scaling (`cuda_adjust_domain_for_agents`), and invariant test suite ([`tests/test_invariants.cpp`](../tests/test_invariants.cpp)).
  * [x] **Phase 1:** Stateless RNG (`rand_u01`), bit-bounded Radix Sort, physical memory reordering (`kernel_reorder_agents`), empty-infected cell skipping, dedicated CUDA stream, and pinned memory.
  * [x] **Phase 2:** Risk groups (`RiskGroup`), structured urban mobility, Z-score focal quarantine (`[F]`), hospital saturation, vaccination, and waning immunity.
  * [x] **Block 1:** Exact `new_exposures` (`S -> E`) counting on GPU, SEIRDS ODE solver (RK4), $R_0$ via Next-Generation Matrix (NGM), OpenMP $R_t$ bootstrap (`[P5-P95]`, `P(Rt>1)`), and analytical convergence tests.
  * [x] **Block 2:** Lagrangian mobility with diurnal cycle (`home_zone`, 8 organic hubs), superspreading (`FLAG_SUPERSPREADER`), 2D kinetic $R_0(\text{NGM})$, OpenMP spin-wait elimination, CUDA worker thread decoupled from Win32/OpenGL UI, `I_ODE` reference curve in chart, and reproducible CSV+JSON export to `reports/`.
  * [x] **Block 3:** Parallel Monte Carlo ensembles on GPU (*single-launch multi-replica*, `K=1..32`), OpenMP reduction (`CpuEnsembleEngine`) with P5/P50/P95 percentiles, $\mathbb{P}(R_t>1)$, extinction rate, and shaded P5–P95 band in the chart.
  * [x] **Block 4 (Urban Layouts & Family Flow):**
    * **Typed Urban Zones (`UrbanZone`):** Homes (residential), Workplaces/Schools, Hospitals (capacity-limited), Commerce/Leisure, and Transit nodes, with city presets (`PRESET_MONOCENTRIC`, `PRESET_POLYCENTRIC`).
    * **Family Structure & Group Mixing:** Agent grouping by `home_id` (4-member families with mixed `RiskGroup`) and divergent daytime destinations (`work_zone`).
    * **Day/Night & Weekly Cycle:** Hourly phases with `dt = 1/48` days (morning commute, work/school day, return commute, intrahousehold night) and weekday/weekend differentiation.
    * **Contagion Attribution (`infection_source`):** Records the setting where each infection occurs (`Home`, `Work`, `Transit`, `Hospital`, `Leisure`) and percentage breakdown in telemetry.
    * **Targeted Urban Interventions:** School/workplace closure / remote work (`[W]`) and household isolation (`[H]`) when a family member becomes symptomatic/infectious.
    * **Clear GPU Visualization (`kernel_render_urban_zones_1080p` / `kernel_render_agents_prio`):** Differentiated zone outlines, day/night lighting, brief flashes on new infections (`S -> E`), and family inspection mode (`[P]`).
  * [x] **Block 5 (Scientific Consolidation — Theoretical Framework):**
    * Stochastic calibration via **ABC-SMC** (`calibrate_abc_smc`) with parallel OpenMP sequential sampling.
    * Probabilistic forecast scoring (`compute_ensemble_forecast_scores`: **WIS**, **IS80**, **RMSE**, **MAE**, coverages) and global sensitivity analysis (`run_global_sensitivity_sweep`).
    * Operational telemetry alignment: exact `new_exposures` in `CpuTelemetryEngine`, $\mathbb{P}(R_t>1)$ decision categories (`RtCategory`, CDC CFA standard), and explicit distinction between **CFR** and **IFR**.
