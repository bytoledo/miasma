#pragma once

#include <cstdint>

namespace seird {

constexpr int FRAME_WIDTH  = 1920;
constexpr int FRAME_HEIGHT = 1080;

enum AgentState : uint8_t {
    STATE_S = 0, // Susceptible
    STATE_E = 1, // Expuesto (incubando, no contagia)
    STATE_I = 2, // Infectado (contagioso)
    STATE_R = 3, // Recuperado (inmune)
    STATE_D = 4  // Fallecido (estático, no contagia)
};

enum RenderMode : int {
    RENDER_AGENTS  = 0, // Solo agentes con prioridad visual (I > E > R > S > D)
    RENDER_HEATMAP = 1, // Heatmap térmico bilineal de densidad + hotspots Z > 2.0
    RENDER_HYBRID  = 2  // Heatmap de fondo + agentes E/I superpuestos
};

struct alignas(32) SimParams {
    // Dominio espacial (en unidades del mundo, ej. 1000.0 x 1000.0)
    float world_width  = 1000.0f;
    float world_height = 1000.0f;

    // Malla para Spatial Hashing (cell_size >= infection_radius)
    int   grid_w    = 256;
    int   grid_h    = 256;
    float cell_size = 4.0f;

    // Parámetros epidemiológicos SEIRD
    float infection_radius = 3.5f;   // Distancia máxima de contagio
    float beta             = 0.25f;  // Probabilidad de contagio por contacto/paso
    float incubation_days  = 5.0f;   // Duración promedio en estado E (1/sigma)
    float infectious_days  = 10.0f;  // Duración promedio en estado I (1/gamma)
    float fatality_rate    = 0.02f;  // Tasa de letalidad alpha (I -> D)

    // Dinámica temporal y movilidad
    float dt                = 1.0f / 48.0f; // Paso de tiempo (0.5 horas en días)
    float base_speed        = 15.0f;        // Velocidad base (unidades / día)
    float quarantine_factor = 1.0f;         // 1.0 = normal, 0.1 = cuarentena estricta

    // Opciones de visualización en GPU y cámara (Zoom/Pan)
    int   render_mode   = RENDER_HYBRID;
    bool  enable_trails = true;
    bool  show_rings    = true;
    float cam_cx        = 500.0f;
    float cam_cy        = 500.0f;
    float cam_zoom      = 1.0f;
};

struct SimStats {
    uint32_t count_s       = 0;
    uint32_t count_e       = 0;
    uint32_t count_i       = 0;
    uint32_t count_r       = 0;
    uint32_t count_d       = 0;
    uint32_t hotspot_cells = 0;    // Celdas con brote anómalo (Z > 2.0 en GPU)
    float    max_cell_z    = 0.0f; // Z-score espacial máximo calculado en GPU
    float    sim_day       = 0.0f;
};

// Punteros a memoria de GPU en formato Structure of Arrays (SoA)
struct AgentSystemDevice {
    int num_agents = 0;
    int num_cells  = 0;

    // Cinemática
    float* x  = nullptr;
    float* y  = nullptr;
    float* vx = nullptr;
    float* vy = nullptr;

    // Estado SEIRD
    uint8_t* state       = nullptr;
    float*   state_timer = nullptr;

    // Spatial Hashing (Uniform Grid)
    uint32_t* cell_id    = nullptr;
    uint32_t* agent_idx  = nullptr;
    int*      cell_start = nullptr;
    int*      cell_end   = nullptr;

    // Estado RNG opaco para mantener types.hpp libre de headers CUDA en main.cpp
    void* rng_states = nullptr;

    // Framebuffer RGBA 1080p en VRAM (1920 * 1080 * 4 bytes)
    uint8_t* framebuffer_rgba = nullptr;
};

} // namespace seird
