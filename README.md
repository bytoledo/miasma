<div align="center">

# Miasma HPC Engine

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=c%2B%2B)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-FF5722?style=flat-square)](https://www.openmp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-064F8C?style=flat-square&logo=cmake)](https://cmake.org/)
[![OpenGL](https://img.shields.io/badge/OpenGL-Win32-5586A4?style=flat-square&logo=opengl)](https://www.opengl.org/)

**Hybrid GPU-accelerated stochastic Agent-Based Model (ABM SEIRD) epidemiological simulation engine in SoA memory (NVIDIA CUDA) with parallel CPU analytics (OpenMP/AVX2), Monte Carlo ensembles, and native 1080p visualization (Win32 + OpenGL).**

[Features](#features) • [Architecture](#hybrid-processing-architecture) • [Structure](#repository-structure) • [Requirements](#system-requirements--dependencies) • [Quickstart](#quickstart) • [Tests](#validation--test-suite) • [Documentation](#documentation--theoretical-framework)

</div>

---

## Features

* **Continuous 2D Stochastic ABM SEIRD Model:** Compartmental dynamics for Susceptible (`S`), Exposed (`E`), Infected (`I`), Recovered (`R`), and Deceased (`D`) states on a *Structure of Arrays* (SoA) layout with risk groups (`RiskGroup`), superspreading (`FLAG_SUPERSPREADER`), Lagrangian mobility with a diurnal cycle (8 organic hubs), vaccination, waning immunity, hospital saturation, and *Z-score* targeted quarantine.
* **Massively Parallel GPU Acceleration (NVIDIA CUDA):** $O(N)$ contagion evaluation via uniform-grid *Spatial Hashing* with `cub::DeviceRadixSort::SortPairs`, stateless RNG (`rand_u01`), physical agent reordering in VRAM, empty-infected cell skipping, and dynamic domain scaling based on available VRAM (`cudaMemGetInfo`).
* **Multi-Replica Monte Carlo Ensembles:** Single-launch parallel execution on GPU (`K = 1..32` replicas) with OpenMP CPU reduction (`CpuEnsembleEngine`) to compute percentiles (`P5`, `P50`, `P95`), epidemic probability $\mathbb{P}(R_t > 1)$, and extinction rate.
* **Telemetry & Analytical Validation (CPU + OpenMP + AVX2):** Deterministic SEIRDS ODE solver (RK4), $R_0$ computation via 2D kinetic Next-Generation Matrix (NGM), spatial outbreak *Z-score* detection, and real-time *bootstrap* $R_t$ estimation.
* **Native Visualization & Reproducible Export:** Win32 + OpenGL interface decoupled from the CUDA worker thread with a 1080p ($1920 \times 1080$) RGBA framebuffer, real-time interactive controls, time-series charting (P5–P95 confidence band and `I_ODE` reference curve), and reproducible CSV/JSON export to `reports/`.

---

## Hybrid Processing Architecture

```mermaid
graph TD
    subgraph GPU ["GPU: NVIDIA CUDA (SoA + Spatial Hashing)"]
        MoveHash[kernel_move_and_hash: Lagrangian Mobility & Cell ID] --> RadixSort[cub::DeviceRadixSort::SortPairs]
        RadixSort --> Reorder[kernel_reorder_agents + Cell Bounds]
        Reorder --> SEIRStep[kernel_seir_step: O(N) Contagion + Stateless RNG]
        SEIRStep --> Render1080p[kernel_render_1080p: RGBA Framebuffer]
        SEIRStep --> Counters[Atomic Reduction S/E/I/R/D & new_exposures]
    end

    subgraph CPU ["CPU: OpenMP & AVX2 (Telemetry & Validation)"]
        Counters -->|Pinned Memory / Stream| CpuEngine[CpuEngine & CpuEnsembleEngine]
        CpuEngine --> RtBoot[Bootstrap Rt & Spatial Z-Score]
        CpuEngine --> ODERK4[SEIRDS RK4 ODE Solver + NGM R0]
        CpuEngine --> Percentiles[Monte Carlo Percentiles P5 / P50 / P95]
    end

    subgraph UI ["Native Frontend: Win32 + OpenGL"]
        Render1080p -->|CUDA-GL Interop| Viewport[Real-Time 1080p Viewport]
        RtBoot --> Chart[Time-Series Chart + P5-P95 Band + I_ODE]
        ODERK4 --> Chart
        Percentiles --> Chart
        Chart --> Export[Reproducible CSV + JSON Export]
    end
```

---

## Repository Structure

* `include/`: C++17 / CUDA headers (`types.hpp`, `cuda_engine.cuh`, `cpu_engine.hpp`).
* `src/`: CUDA/OpenMP engine implementations and native Win32/OpenGL UI (`cuda_engine.cu`, `cpu_engine.cpp`, `main.cpp`).
* `tests/`: Automated invariant and analytical convergence test suite (`test_invariants.cpp`).
* `docs/`: Architecture plan, theoretical framework, and bibliography (`PROJECT_PLAN.md`, `MARCO_TEORICO.md`, `referencias.bib`).
* `web/`: Auxiliary web resources (`index.html`, `app.js`, `style.css`).
* `CMakeLists.txt` & `build.bat`: CMake ($\ge 3.20$) build configuration and quick Windows build script (MSVC + `nvcc`).

---

## System Requirements & Dependencies

### 1. Minimum Recommended Hardware
* **CPU:** Multi-core x86_64 processor with **AVX2** instruction support.
* **GPU:** CUDA-capable **NVIDIA** graphics card (configured by default for Blackwell `sm_120`, adjustable via `CMAKE_CUDA_ARCHITECTURES`).
* **RAM / VRAM:** 8 GB system memory and a dedicated GPU with OpenGL + CUDA support.

### 2. Build Toolchain (C++ & CUDA)
* **Operating System:** Windows (for the native Win32 + OpenGL executable).
* **C++ Compiler:** **Microsoft Visual C++ (MSVC)** (Visual Studio 2022 / Build Tools) with **C++17**, `/arch:AVX2`, and `/openmp` support.
* **NVIDIA CUDA Toolkit:** Version **13.x** (`nvcc`, `cudart`, `curand`, and `CUB`).
* **CMake:** Version **3.20 or higher**.
* **System Libraries (Windows):** `opengl32`, `user32`, `gdi32`, `comctl32`.

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/bytoledo/miasma-parallel.git
cd miasma-parallel
```

### 2. Build the project

#### Option A: CMake (builds `seird_core`, `seird_parallel.exe`, and `seird_tests.exe`)
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

If your GPU uses a CUDA architecture other than `sm_120` (for example, Ada `89` or Ampere `86`):
```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

#### Option B: Direct Windows build script (`build.bat`)
```bat
build.bat
```

### 3. Run the simulation
```powershell
.\build\seird_parallel.exe
```

From the interactive sidebar, you can adjust the mobility/quarantine factor, transmission rate ($\beta$), fatality rate, contagion radius, substeps per frame, initial cases, toggle interventions (targeted quarantine / vaccination), and export reproducible time-series reports to `reports/`.

---

## Validation & Test Suite

The repository includes an automated suite of physical, numerical, and epidemiological invariant tests in [`tests/test_invariants.cpp`](tests/test_invariants.cpp):

```powershell
.\build\seird_tests.exe
```

Verified tests:
* Dynamic VRAM hardware limits and domain scaling.
* Exact population conservation ($S + E + I + R + D = N$) and telemetry.
* Deterministic reproducibility with stateless RNG.
* Conservation under Phase 2 interventions and exact incidence under vaccination.
* RK4 ODE solver and Next-Generation Matrix (NGM) $R_0$ validation.
* ABM convergence toward the analytical model in the high-diffusion regime.
* Lagrangian mobility, superspreaders, and Monte Carlo ensembles (`K = 8`).

---

## Documentation & Theoretical Framework

* **Architecture Plan & Technical Log:** [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
* **Theoretical Framework & Mathematical Formulation:** [`docs/MARCO_TEORICO.md`](docs/MARCO_TEORICO.md)
* **Bibliography (BibTeX):** [`docs/referencias.bib`](docs/referencias.bib)
