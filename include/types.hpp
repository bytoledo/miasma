#pragma once

#include <cstdint>

namespace seird {

constexpr int FRAME_WIDTH     = 1920;
constexpr int FRAME_HEIGHT    = 1080;
constexpr int MAX_REPLICAS    = 64;
constexpr int MAX_URBAN_ZONES = 32;

enum AgentState : uint8_t {
    STATE_S = 0, // Susceptible
    STATE_E = 1, // Expuesto (incubando, no contagia)
    STATE_I = 2, // Infectado (contagioso)
    STATE_R = 3, // Recuperado (inmune)
    STATE_D = 4  // Fallecido (estático, no contagia)
};

enum RiskGroup : uint8_t {
    RISK_LOW    = 0, // Jóvenes (~50%): mayor movilidad, menor letalidad (0.25x)
    RISK_MEDIUM = 1, // Adultos (~35%): movilidad y letalidad base (1.0x)
    RISK_HIGH   = 2  // Vulnerable/Mayor (~15%): menor movilidad, mayor letalidad (3.5x)
};

enum ZoneType : uint8_t {
    ZONE_HOME     = 0, // Residencial / Hogares
    ZONE_WORK     = 1, // Zona laboral
    ZONE_SCHOOL   = 2, // Escuelas / Universidades
    ZONE_HOSPITAL = 3, // Hospital con capacidad limitada
    ZONE_LEISURE  = 4, // Comercio / Ocio (fines de semana)
    ZONE_TRANSIT  = 5  // Nodo de transporte público
};

enum InfectionSource : uint8_t {
    SRC_NONE     = 0,
    SRC_HOME     = 1,
    SRC_WORK     = 2,
    SRC_TRANSIT  = 3,
    SRC_HOSPITAL = 4,
    SRC_LEISURE  = 5
};

enum UrbanPreset : int {
    PRESET_ORGANIC_POLES = 0, // Polos orgánicos clásicos (Bloque 2)
    PRESET_MONOCENTRIC   = 1, // Centro laboral/hospitalario y periferia residencial
    PRESET_POLYCENTRIC   = 2  // Múltiples subcentros urbanos conectados
};

struct UrbanZone {
    float    cx       = 0.0f;
    float    cy       = 0.0f;
    float    radius   = 40.0f; // Radio equivalente / referencia
    float    half_w   = 40.0f; // Semi-ancho del distrito rectangular
    float    half_h   = 40.0f; // Semi-alto del distrito rectangular
    ZoneType type     = ZONE_HOME;
    float    capacity = 0.0f;  // Capacidad (ej. camas en hospital)
};

constexpr uint8_t RISK_GROUP_MASK    = 0x03u; // Bits 0-1: RISK_LOW / RISK_MEDIUM / RISK_HIGH
constexpr uint8_t FLAG_SUPERSPREADER = 0x80u; // Bit 7: agente superdispersor (0 bytes extra)

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

    // Realismo epidemiológico (Fase 2 / Bloque 2)
    float urban_clustering      = 0.25f; // Atracción hacia centros de actividad (0 = gas uniforme)
    bool  focal_quarantine      = false; // Cuarentena automática en celdas con Z-score > 2.0
    float hospital_capacity_pct = 0.05f; // Umbral de infectados (5% pop) antes de saturación
    float vaccination_rate      = 0.0f;  // Tasa diaria de vacunación (S -> R)
    float immunity_days         = 0.0f;  // Duración inmunidad R -> S (0 = permanente)
    float superspreader_frac    = 0.10f; // Fracción de agentes superdispersores (10%)
    float superspreader_mult    = 4.0f;  // Multiplicador de riesgo de contagio por contacto
    int   num_replicas          = 1;     // Réplicas Monte Carlo en paralelo (1..MAX_REPLICAS)

    // Opciones de visualización en GPU y cámara (Zoom/Pan)
    int   render_mode   = RENDER_HYBRID;
    bool  enable_trails = true;
    bool  show_rings    = true;
    float cam_cx        = 500.0f;
    float cam_cy        = 500.0f;
    float cam_zoom      = 1.0f;

    // Layouts urbanos, flujo familiar y sub-stepping (Bloque 4)
    bool      enable_urban_layout  = false;
    int       urban_preset         = PRESET_MONOCENTRIC;
    int       num_urban_zones      = 0;
    UrbanZone urban_zones[MAX_URBAN_ZONES]{};
    bool      household_quarantine = false; // Aislamiento domiciliario por hogar si hay un I
    bool      close_schools_work   = false; // Cierre de escuelas / home office
    int       selected_home_id     = -1;    // Hogar resaltado para inspección (-1 = ninguno)
    int       physics_substeps     = 1;     // Pasos físicos por frame de render
    bool      ensemble_grid_view   = false; // Vista mosaico de todas las K réplicas Monte Carlo
    int       focused_replica      = 0;     // Réplica ampliada cuando ensemble_grid_view == false
};

struct SimStats {
    uint32_t count_s       = 0;
    uint32_t count_e       = 0;
    uint32_t count_i       = 0;
    uint32_t count_r       = 0;
    uint32_t count_d       = 0;
    uint32_t new_exposures = 0;    // Contagios reales acumulados (S -> E) desde el último muestreo
    uint32_t hotspot_cells = 0;    // Celdas con brote anómalo (Z > 2.0 en GPU)
    float    max_cell_z    = 0.0f; // Z-score espacial máximo calculado en GPU
    float    sim_day       = 0.0f;

    // Atribución de contagios y ocupación hospitalaria (Bloque 4)
    uint32_t src_home           = 0;
    uint32_t src_work           = 0;
    uint32_t src_transit        = 0;
    uint32_t src_hospital       = 0;
    uint32_t src_leisure        = 0;
    uint32_t hospitalized_count = 0;
};

// Punteros a memoria de GPU en formato Structure of Arrays (SoA)
struct AgentSystemDevice {
    int num_agents         = 0; // Total de agentes en GPU (agents_per_replica * num_replicas)
    int agents_per_replica = 0; // Agentes por réplica
    int num_replicas       = 1; // Número de réplicas Monte Carlo
    int num_cells          = 0; // Celdas por réplica (grid_w * grid_h)

    // Cinemática
    float* x  = nullptr;
    float* y  = nullptr;
    float* vx = nullptr;
    float* vy = nullptr;

    // Estado SEIRD y estratificación
    uint8_t* state       = nullptr;
    uint8_t* risk_group  = nullptr;
    uint8_t* home_zone   = nullptr; // Bits 0-3: hogar (0..7), Bits 4-7: actividad/trabajo (0..7)
    uint8_t* rep_id      = nullptr; // Índice de réplica Monte Carlo (0..num_replicas-1)
    float*   state_timer = nullptr;

    // Estructura familiar y atribución de contagios (Bloque 4)
    uint32_t* home_id          = nullptr; // ID de familia compartido (2-5 agentes)
    uint8_t*  work_zone        = nullptr; // Índice de zona diurna asignada (0..MAX_URBAN_ZONES-1)
    uint8_t*  infection_source = nullptr; // InfectionSource donde ocurrió S -> E
    float*    infection_flash  = nullptr; // Temporizador de destello visual al contagiarse

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
