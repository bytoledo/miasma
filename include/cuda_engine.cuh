#pragma once

#include "types.hpp"

#include <vector>

namespace seird {

// Inicializa memoria en VRAM, estados RNG y población inicial (con pacientes cero)
bool cuda_init_simulation(
    AgentSystemDevice& dev,
    int agents_per_replica,
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
    bool compute_stats = true,
    std::vector<SimStats>* out_replica_stats = nullptr
);

// Renderiza el framebuffer 1080p únicamente en VRAM (sin copiar a CPU)
void cuda_render_1080p_device(
    AgentSystemDevice& dev,
    const SimParams& params
);

// Sincroniza el stream de simulación CUDA
void cuda_sync_simulation();

// Copia framebuffer_rgba (VRAM) a un buffer de despliegue en VRAM (D2D) para doble buffering
bool cuda_copy_framebuffer_to_device(const AgentSystemDevice& dev, uint8_t* dst_device_rgba);

// Temporizador GPU exacto sobre el stream de simulación (sin exponer cudaEvent_t en headers)
void  cuda_timer_begin();
float cuda_timer_end_ms();

// Consulta uso de VRAM en MB para telemetría
void cuda_get_vram_info(float& used_mb, float& total_mb);

struct HardwareLimits {
    size_t vram_free_mb      = 0;
    size_t vram_total_mb     = 0;
    size_t bytes_per_agent   = 0;
    int    max_agents_vram   = 0;
    int    safe_agents_limit = 0; // Tope recomendado considerando densidad y WDDM
};

// Consulta la VRAM disponible y calcula el máximo de agentes soportado dinámicamente
HardwareLimits cuda_query_hardware_limits(size_t reserve_vram_mb = 1024);

// Escala dominio espacial y malla proporcionalmente para mantener densidad segura por celda
void cuda_adjust_domain_for_agents(SimParams& params, int num_agents);

// Construye un layout urbano predefinido (PRESET_ORGANIC_POLES, PRESET_MONOCENTRIC, PRESET_POLYCENTRIC)
void cuda_build_urban_preset(SimParams& params, int preset);

struct FamilyMemberInfo {
    float   x = 0.0f, y = 0.0f;
    float   work_x = 0.0f, work_y = 0.0f;
    uint8_t state = STATE_S;
    uint8_t risk_group = RISK_LOW;
};

struct FamilyTrackInfo {
    int   home_id = -1;
    float home_x  = 0.0f;
    float home_y  = 0.0f;
    int   count   = 0;
    FamilyMemberInfo members[8]{};
};

// Selecciona el home_id del agente más cercano a (world_x, world_y) en la réplica 0
int cuda_pick_nearest_home_id(const AgentSystemDevice& dev, float world_x, float world_y);

// Obtiene posiciones actuales y destinos de los miembros de un home_id en la réplica 0
bool cuda_get_family_track(const AgentSystemDevice& dev, const SimParams& params, int home_id, FamilyTrackInfo& out_info);

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
