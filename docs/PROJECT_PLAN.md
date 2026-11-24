# miasma-parallel (Miasma HPC Engine) — Contexto y Plan de Arquitectura

Este documento resume las decisiones técnicas, el modelo matemático y el plan de implementación definidos para el proyecto.

## 1. Hardware y Stack Tecnológico
* **GPU Objetivo:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, arquitectura Blackwell `sm_120`).
* **Stack Principal:** C++17, NVIDIA CUDA 13.x (`cuRAND`, `CUB DeviceRadixSort`), OpenMP + AVX2 (CPU), CMake $\ge 3.20$.
* **Visualización:** Interfaz nativa Win32 + OpenGL con framebuffer RGBA 1080p renderizado en GPU y gráficas de telemetría en tiempo real.

## 2. Modelo Epidemiológico y Computacional Elegido
* **Modelo:** **ABM (Agent-Based Model) con dinámica SEIRD Estocástica** en espacio 2D continuo.
  * **S** (Susceptible) $\xrightarrow{\beta}$ **E** (Expuesto / incubando, no contagia) $\xrightarrow{\sigma}$ **I** (Infectado / contagioso).
  * Al terminar el periodo infeccioso ($I$): con probabilidad $\alpha$ (tasa de letalidad) pasa a **D** (Fallecido, estático y no contagia), y con $1 - \alpha$ pasa a **R** (Recuperado / inmune).
* **Aceleración en GPU:** **Spatial Hashing (*Uniform Grid*)** con `cub::DeviceRadixSort::SortPairs` para evaluar contagios en $O(N)$ en lugar de $O(N^2)$.
* **Capacidad:** Límites dinámicos calculados vía `cudaMemGetInfo` (`cuda_query_hardware_limits`), escalando el dominio espacial y la malla automáticamente (`cuda_adjust_domain_for_agents`) para soportar millones de agentes sin saturar celdas ni disparar TDR.

## 3. Diseño de Memoria (SoA - *Structure of Arrays*)
Definido en `include/types.hpp`:
* Cinemática: `x`, `y`, `vx`, `vy` (`float*`).
* Estado SEIRD: `state` (`uint8_t*`: `0=S, 1=E, 2=I, 3=R, 4=D`), `state_timer` (`float*`).
* Spatial Hashing: `cell_id`, `agent_idx` (`uint32_t*`), `cell_start`, `cell_end` (`int*`).
* RNG: `rng_states` (`curandState*`, opaco en `types.hpp`).
* Salida visual: `framebuffer_rgba` (`uint8_t*`, `1920 x 1080 x 4`).

## 4. Flujo de Kernels por Paso ($\Delta t$)
1. `kernel_move_and_hash`: Actualiza posición $(x, y)$, aplica rebotes/cuarentena (fallecidos `D` no se mueven) y calcula `cell_id`.
2. `thrust::sort_by_key`: Ordena `agent_idx` por `cell_id`.
3. `kernel_find_cell_bounds`: Calcula `cell_start` y `cell_end` de cada celda.
4. `kernel_seir_step`: Revisa los agentes de la celda propia y las 8 vecinas; evalúa contagio con `cuRAND` y actualiza transiciones `E -> I` e `I -> R / D`.
5. `kernel_render_1080p`: Proyecta los agentes al `framebuffer_rgba` en VRAM.

## 5. Estado Actual y Próximos Pasos
* **Completado:**
  * `include/types.hpp`
  * `include/cuda_engine.cuh`
  * `.gitignore` y `git init`
  * [x] **Paso 2:** Implementar `src/cuda_engine.cu`.
  * [x] **Paso 3:** Crear `CMakeLists.txt` y `src/cpu_engine.cpp` (telemetría OpenMP, $R_t$, $Z$-score de brotes).
  * [x] **Paso 4:** Implementar `src/main.cpp` (UI nativa Win32 + OpenGL y reportes).
  * [x] **Fase 0:** Límites dinámicos por VRAM (`cuda_query_hardware_limits`), escalado de dominio (`cuda_adjust_domain_for_agents`) y suite de pruebas de invariantes (`tests/test_invariants.cpp`).
  * [x] **Fase 1:** RNG *stateless* (`rand_u01`), Radix Sort acotado a bits activos, reordenamiento físico (`kernel_reorder_agents`), salto de celdas sin infectados, *stream* dedicado y *pinned memory*.
  * [x] **Fase 2:** Grupos de riesgo (`RiskGroup`), movilidad urbana estructurada, cuarentena focalizada por Z-score (`[F]`), saturación hospitalaria, vacunación e inmunidad decreciente.
  * [x] **Bloque 1:** Conteo exacto `new_exposures` (`S -> E`) en GPU, Solver ODE SEIRDS (RK4), $R_0$ por Matriz de Siguiente Generación (NGM), bootstrap OpenMP de $R_t$ (`[P5-P95]`, `P(Rt>1)`) y pruebas de convergencia analítica.
  * [x] **Bloque 2:** Movilidad Lagrangiana con ciclo diurno (`home_zone`, 8 polos orgánicos), superdispersión (`FLAG_SUPERSPREADER`), $R_0(\text{NGM})$ cinético 2D, eliminación de *spin-wait* OpenMP, hilo *Worker* CUDA desacoplado de UI Win32/OpenGL, curva `I_ODE` en gráfica y exportación reproducible CSV+JSON en `reports/`.
  * [x] **Bloque 3:** Ensambles Monte Carlo en paralelo en GPU (*single-launch multi-replica*, `K=1..32`), reducción OpenMP (`CpuEnsembleEngine`) con percentiles P5/P50/P95, $\mathbb{P}(R_t>1)$, tasa de extinción y banda sombreada P5–P95 en la gráfica.
  * [x] **Bloque 4 (Layouts Urbanos y Flujo Familiar):**
    * **Zonas Urbanas Tipadas (`UrbanZone`):** Hogares (residencial), Trabajos/Escuelas, Hospitales (con capacidad limitada), Comercio/Ocio y nodos de tránsito, con *presets* de ciudad (`PRESET_MONOCENTRIC`, `PRESET_POLYCENTRIC`).
    * **Estructura Familiar y Mezcla de Grupos:** Agrupación de agentes por `home_id` (familias de 4 miembros con mezcla de `RiskGroup`) y destinos diurnos divergentes (`work_zone`).
    * **Ciclo Día/Noche y Semanal:** Fases horarias con `dt = 1/48` días (traslado ida, día laboral/escolar, retorno, noche intrafamiliar) y diferenciación entre días laborales y fin de semana.
    * **Atribución de Contagio (`infection_source`):** Registro del entorno donde ocurre cada infección (`Hogar`, `Trabajo`, `Tránsito`, `Hospital`, `Ocio`) y desglose porcentual en telemetría.
    * **Intervenciones Urbanas Focalizadas:** Cierre de escuelas / *home office* (`[W]`) y aislamiento domiciliario por hogar (`[H]`) cuando un miembro presenta síntomas.
    * **Visualización Clara en GPU (`kernel_render_urban_zones_1080p` / `kernel_render_agents_prio`):** Contornos de zonas diferenciados, iluminación día/noche, destellos breves en nuevos contagios (`S -> E`) y modo de inspección para seguir a una familia (`[P]`).
* **Pendiente:**
  * [x] **Bloque 5 (Consolidación Científica — Marco Teórico):**
    * Calibración estocástica mediante **ABC-SMC** (`calibrate_abc_smc`) con muestreo secuencial paralelo en OpenMP.
    * Métricas probabilísticas de evaluación (`compute_ensemble_forecast_scores`: **WIS**, **IS80**, **RMSE**, **MAE**, coberturas) y análisis de sensibilidad global (`run_global_sensitivity_sweep`).
    * Alineación de telemetría operativa: uso de `new_exposures` exacto en `CpuTelemetryEngine`, categorías de decisión $\mathbb{P}(R_t>1)$ (`RtCategory`, estándar CDC CFA) y distinción explícita entre **CFR** e **IFR**.
