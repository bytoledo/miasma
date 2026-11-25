# User & Developer Guide — Miasma HPC Engine

This guide covers building, running, interacting with the native Win32 + OpenGL interface, exporting reproducible reports, and executing the automated invariant and scientific benchmark suites.

---

## 1. Building from Source (Windows)

### Prerequisites
* **OS:** Windows 10/11 (x64) for `miasma_parallel.exe` (Win32 + OpenGL).
* **Compiler:** Microsoft Visual C++ (Visual Studio 2022 / Build Tools) with C++17, `/arch:AVX2`, and `/openmp`.
* **NVIDIA CUDA Toolkit:** 13.x (`nvcc`, `cudart`, `curand`, `cub`).
* **CMake:** 3.20 or higher.

### Option A: CMake (Builds All Targets)
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
To target a specific GPU compute capability other than the default Blackwell `sm_120` (e.g., `89` for Ada Lovelace or `86` for Ampere):
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```
CMake produces:
* `build/miasma_core.lib` — Static CUDA + OpenMP core library.
* `build/miasma_parallel.exe` — Interactive Win32 + OpenGL simulator.
* `build/miasma_tests.exe` — Automated invariant test suite (`ctest` target `miasma_invariants`).
* `build/miasma_scientific_benchmarks.exe` — Scientific validation benchmarks.

### Option B: Quick Batch Script (`build.bat`)
```bat
build.bat
```
Compiles `src/main.cpp`, `src/cpu_engine.cpp`, and `src/cuda_engine.cu` (`sm_120`) directly into `build\miasma_parallel.exe`.

---

## 2. Running the Interactive Simulator

Launch the application from the repository root (so `reports/` is resolved relative to the working directory):
```powershell
.\build\miasma_parallel.exe
```

### 2.1 Sidebar Controls (3 Tabs)
The right-hand sidebar displays live telemetry KPIs (`S`, `E`, `I`, `R`, `D`, $R_t$ with `[P5-P95]` and CDC CFA category, $R_0\text{ (NGM)}$, spatial `Z-score`, `CFR`, `IFR`, and infection source breakdown) above three control tabs:

1. **Control & Ciudad (Tab 0):**
   * **Pause / Resume** and **Reset** buttons.
   * Scenario presets: **Quarantine**, **Vaccine**, **Normal**.
   * **Urban Preset** cycle (`PRESET_ORGANIC_POLES` → `PRESET_MONOCENTRIC` → `PRESET_POLYCENTRIC`).
   * **Single Step** (`+1` step while paused).
   * **Household Quarantine** (isolates entire family when a member is infected) & **Close Schools / Work**.
   * **Follow Family** (toggles family trajectory inspection).
   * **Physics Substeps** slider (`1..48` substeps per render frame).
2. **Parámetros (Tab 1):**
   * Sliders for **Quarantine / Mobility factor**, **Transmission rate ($\beta$)**, **Fatality rate ($\alpha$)**, **Infection radius**, **Chart window (days)**, and **Initial infected ($I_0$)**.
3. **Vista & Export (Tab 2):**
   * **Render Mode** (`Agents` / `Heatmap` / `Hybrid`), **Trails** toggle, **Contagion Rings** toggle.
   * **Export CSV** (writes both `.csv` and `.json` to `reports/`).
   * **RNG Seed** input field (applied on Reset).
   * **Ensemble Mosaic** toggle (switches between `K`-replica grid view and single focused replica).

### 2.2 Keyboard Shortcuts & Mouse Navigation

| Key | Action |
| :--- | :--- |
| `Space` | Pause / Resume simulation |
| `N` or `Right` | Pause and advance a single physics step |
| `R` | Reset simulation using the current RNG seed |
| `Q` | Toggle strict quarantine (`quarantine_factor` `0.10` ↔ `1.00`) |
| `V` | Toggle vaccination campaign (`beta = 0.06, vaccination_rate = 0.02` ↔ `beta = 0.25, vaccination_rate = 0.0`) |
| `F` | Toggle focal cell quarantine ($Z > 2.0$ hotspots) |
| `U` | Cycle urban preset (`ORGANIC_POLES` → `MONOCENTRIC` → `POLYCENTRIC`) and reset |
| `H` | Toggle household quarantine |
| `W` | Toggle school/workplace closure |
| `P` | Toggle family inspection (`selected_home_id = 0` ↔ `-1`) |
| `M` | Cycle render mode (`Agents` → `Heatmap` → `Hybrid`) |
| `T` | Toggle motion trails |
| `A` | Toggle contagion radius rings |
| `G` | Toggle Monte Carlo ensemble mosaic grid view (when `num_replicas > 1`) |
| `C` | Reset camera zoom (`1.0x`) and center |
| `E` | Export reproducible CSV + JSON report to `reports/` |
| `Up` / `+` | Increase physics substeps per frame (max `48`) |
| `Down` / `-` | Decrease physics substeps per frame (min `1`) |
| `Esc` | Exit application |

* **Mouse Wheel:** Zoom in/out on the 2D simulation viewport.
* **Mouse Drag / Click:** Pan camera when zoomed in, or pick the nearest family `home_id` for inspection.

---

## 3. Reproducible Reports (`reports/`)

Pressing `E` or clicking **Exportar CSV [E]** creates a timestamped pair in `reports/`:
* `reports/miasma_reporte_YYYYMMDD_HHMMSS.csv`
* `reports/miasma_reporte_YYYYMMDD_HHMMSS.json`

### CSV Columns
`dia`, `susceptibles`, `expuestos`, `infectados`, `recuperados`, `fallecidos`, `rt`, `rt_p05`, `rt_p95`, `prob_rt_gt_1`, `r0_ngm`, `nuevos_casos_dia`, `cfr_pct`, `focos_brote`, `pico_z`, `i_p05_pct`, `i_p50_pct`, `i_p95_pct`, `extincion_pct`.

### JSON Metadata
Records the GPU device name and compute capability (`gpu_name`, `gpu_sm`), `seed`, `num_agents`, `initial_infected`, linked `csv_file`, full `params` snapshot, and summary peak/CFR metrics.

---

## 4. Automated Test Suite & Scientific Benchmarks

### 4.1 Invariant Test Suite (`miasma_tests.exe`)
Run directly or via `ctest --test-dir build`:
```powershell
.\build\miasma_tests.exe
```
Verifies **12 physical, numerical, and epidemiological invariants**:
1. `test_hardware_limits_and_domain` — Dynamic VRAM query and domain scaling.
2. `test_population_conservation_and_telemetry` — Exact $S + E + I + R + D = N$ conservation.
3. `test_deterministic_reproducibility` — Bit-identical runs from the same seed.
4. `test_fase2_interventions_conservation` — Conservation under quarantine, hospital saturation, and waning immunity.
5. `test_exact_incidence_under_vaccination` — Exact `new_exposures` (`S -> E`) tracking during vaccination.
6. `test_ode_rk4_and_ngm_validation` — RK4 ODE conservation and NGM $R_0$ monotonicity.
7. `test_high_diffusion_abm_convergence` — GPU ABM convergence to the analytical ODE in high-diffusion mixing.
8. `test_lagrangian_and_superspreaders` — Lagrangian hub clustering and superspreader dispersion.
9. `test_monte_carlo_ensemble` — Multi-replica GPU execution (`K = 8`) and `CpuEnsembleEngine` percentile ordering.
10. `test_urban_layout_and_family_flow` — Typed `UrbanZone` presets, family households, and `infection_source` attribution.
11. `test_probabilistic_scores_and_sensitivity` — `WIS`, `IS80`, coverage metrics, and OpenMP global sensitivity sweep.
12. `test_abc_smc_calibration` — Posterior parameter recovery of `(beta, incubation_days, infectious_days)` via `calibrate_abc_smc`.

### 4.2 Scientific Benchmarks (`miasma_scientific_benchmarks.exe`)
```powershell
.\build\miasma_scientific_benchmarks.exe
```
Executes **8 scientific validation experiments**:
1. `benchmark_analytical_thresholds` — Kermack–McKendrick final attack rate & peak error checks.
2. `benchmark_latent_period_invariance` — Final epidemic size invariance to latent period duration.
3. `benchmark_1918_npi_gpu_abm` — 1918-style NPI peak infection and mortality reduction on GPU.
4. `benchmark_abm_vs_ode_monte_carlo` — GPU Monte Carlo ensemble vs. deterministic ODE comparison.
5. `benchmark_endemic_waves_seirds` — Damped endemic oscillations under waning immunity (`immunity_days > 0`).
6. `benchmark_stochastic_extinction_and_superspreading` — Early extinction rates ($K = 32, I_0 = 3$) with and without superspreaders.
7. `benchmark_monocentric_vs_polycentric` — Epidemic peak timing and magnitude across city topologies.
8. `benchmark_identifiability_abc_smc` — 1-parameter vs. 3-parameter ABC-SMC credible interval narrowing and `WIS` improvement.
