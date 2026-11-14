#pragma once

#include "types.hpp"

namespace seird {

// Inicializa memoria en VRAM, estados RNG y población inicial (con pacientes cero)
bool cuda_init_simulation(
    AgentSystemDevice& dev,
    int num_agents,
    int initial_infected,
    const SimParams& params,
    uint64_t seed = 42ULL
);

// Ejecuta un paso completo: movimiento -> spatial hash -> sort -> SEIRD -> (opcional) stats/render
void cuda_step_simulation(
    AgentSystemDevice& dev,
    const SimParams& params,
    uint8_t* host_framebuffer_rgba,
    SimStats& out_stats,
    bool compute_stats = true
);

// Renderiza el framebuffer 1080p únicamente en VRAM (sin copiar a CPU)
void cuda_render_1080p_device(
    AgentSystemDevice& dev,
    const SimParams& params
);

// Consulta uso de VRAM en MB para telemetría
void cuda_get_vram_info(float& used_mb, float& total_mb);

struct BenchmarkResult {
    int    num_agents      = 0;
    int    steps           = 0;
    float  avg_step_ms     = 0.0f;
    float  avg_render_ms   = 0.0f;
    double agents_per_sec  = 0.0;
    float  vram_used_mb    = 0.0f;
};

// Ejecuta un benchmark con cudaEvent_t (sin ventana gráfica)
bool cuda_run_benchmark(
    int num_agents,
    int warmup_steps,
    int measure_steps,
    const SimParams& params,
    BenchmarkResult& out_res
);

// Libera toda la memoria asignada en VRAM
void cuda_free_simulation(AgentSystemDevice& dev);

} // namespace seird
