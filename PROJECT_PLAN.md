# seird-Parallel (SEIRD HPC Engine) — Contexto y Plan de Arquitectura

Este documento resume las decisiones técnicas, el modelo matemático y el plan de implementación definidos para el proyecto.

## 1. Hardware y Stack Tecnológico
* **GPU Objetivo:** NVIDIA GTX 1650 Ti Laptop (4 GB VRAM, arquitectura Turing CC 7.5, 1024 CUDA cores).
* **Stack Principal:** C++17, NVIDIA CUDA 12.x (`cuRAND`, `Thrust`), OpenMP + AVX2 (CPU), CMake $\ge 3.20$.
* **Visualización / Red (estilo `geoThermal-Parallel`):** Servidor HTTP embebido (`httplib.h`) con streaming binario RGBA 1080p (`application/octet-stream`) hacia Canvas HTML5 / Leaflet.js.

## 2. Modelo Epidemiológico y Computacional Elegido
* **Modelo:** **ABM (Agent-Based Model) con dinámica SEIRD Estocástica** en espacio 2D continuo.
  * **S** (Susceptible) $\xrightarrow{\beta}$ **E** (Expuesto / incubando, no contagia) $\xrightarrow{\sigma}$ **I** (Infectado / contagioso).
  * Al terminar el periodo infeccioso ($I$): con probabilidad $\alpha$ (tasa de letalidad) pasa a **D** (Fallecido, estático y no contagia), y con $1 - \alpha$ pasa a **R** (Recuperado / inmune).
* **Aceleración en GPU:** **Spatial Hashing (*Uniform Grid*)** con `thrust::sort_by_key` para evaluar contagios en $O(N)$ en lugar de $O(N^2)$.
* **Capacidad:** ~500,000 a 1,000,000 de agentes en ~100–120 MB de VRAM (sobra margen dentro de los 4 GB).

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
* **Pendiente:**
  * [x] **Paso 4:** Implementar `src/main.cpp` (`httplib`) y frontend en `web/` (Leaflet.js + controles de cuarentena/vacunación).
