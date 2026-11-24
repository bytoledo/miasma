#include "cuda_engine.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

namespace seird {

namespace {

// ============================================================================
// 1. Estado global de GPU, buffers internos y utilidades de memoria
// ============================================================================

constexpr int BLOCK_SIZE = 256;          // Hilos por bloque CUDA
constexpr float PI_F = 3.14159265358979323846f; // Pi en precisión simple

// Buffers internos en VRAM para conteo de estados, radix sort y telemetría (evita allocs por frame)
uint32_t* g_d_counts        = nullptr; // [K*12]: conteos SEIRD, contagios S->E y orígenes
uint32_t* g_d_cell_infected = nullptr; // [num_cells]: infectados por celda
float*    g_d_spatial_stats = nullptr; // [0]: sum, [1]: sum_sq
uint32_t* g_d_hotspot_out   = nullptr; // [0]: hotspot_cells, [1]: max_z_bits (uint32)
uint32_t* g_d_fb_prio       = nullptr; // [1920*1080]: buffer de prioridad visual (atomicMax)
uint32_t* g_d_cell_id_alt   = nullptr; // Doble buffer Radix Sort: cell_id
uint32_t* g_d_agent_idx_alt = nullptr; // Doble buffer Radix Sort: agent_idx
float*    g_d_x_alt         = nullptr; // Buffer alterno de reordenamiento: x
float*    g_d_y_alt         = nullptr; // Buffer alterno de reordenamiento: y
float*    g_d_vx_alt        = nullptr; // Buffer alterno de reordenamiento: vx
float*    g_d_vy_alt        = nullptr; // Buffer alterno de reordenamiento: vy
uint8_t*  g_d_state_alt     = nullptr; // Buffer alterno de reordenamiento: state
uint8_t*  g_d_risk_alt      = nullptr; // Buffer alterno de reordenamiento: risk_group
uint8_t*  g_d_home_zone_alt = nullptr; // Buffer alterno de reordenamiento: home_zone
uint8_t*  g_d_rep_id_alt    = nullptr; // Buffer alterno de reordenamiento: rep_id
uint32_t* g_d_home_id_alt   = nullptr; // Buffer alterno de reordenamiento: home_id
uint8_t*  g_d_work_zone_alt = nullptr; // Buffer alterno de reordenamiento: work_zone
uint8_t*  g_d_inf_src_alt   = nullptr; // Buffer alterno de reordenamiento: infection_source
float*    g_d_inf_flash_alt = nullptr; // Buffer alterno de reordenamiento: infection_flash
uint8_t*  g_d_home_quarantine = nullptr; // Bandera por hogar (1 si tiene algún infectado)
int       g_max_homes         = 0;       // Capacidad asignada para g_d_home_quarantine
uint8_t*  g_d_cell_quarantine = nullptr; // Bandera por celda (1 si Z-score > 2.0)
uint64_t* g_d_pick_min        = nullptr; // Resultado atómico (dist_bits | home_id) al clicar mapa
FamilyTrackInfo* g_d_family_out = nullptr; // Buffer de salida para inspección familiar
float*    g_d_timer_alt       = nullptr; // Buffer alterno de reordenamiento: state_timer
void*     g_d_sort_temp       = nullptr; // Memoria temporal de CUB DeviceRadixSort
size_t    g_sort_temp_bytes   = 0;       // Tamaño en bytes de g_d_sort_temp

uint32_t*    g_h_counts_pinned     = nullptr; // Memoria pinned en host para conteos
uint32_t*    g_h_hotspot_pinned    = nullptr; // Memoria pinned en host para hotspots
cudaStream_t g_sim_stream          = nullptr; // Stream principal no bloqueante de simulación
cudaEvent_t  g_ev_timer_start      = nullptr; // Evento inicio para cronómetro GPU
cudaEvent_t  g_ev_timer_stop       = nullptr; // Evento fin para cronómetro GPU
uint64_t     g_sim_seed            = 42ULL;   // Semilla RNG activa
uint64_t     g_sim_step            = 0ULL;    // Contador de pasos ejecutados
float        g_sim_day             = 0.0f;    // Día simulado acumulado
float        g_last_infected_ratio = 0.0f;    // Fracción reciente de infectados (para saturación hospitalaria)

constexpr int NUM_URBAN_HUBS = 8;
constexpr int FAMILY_SIZE    = 4;
constexpr int COUNTS_STRIDE  = 12; // 0..4: S,E,I,R,D | 5: new_exp | 6..10: src_home..src_leisure | 11: hospitalized
__device__ __constant__ float kHubX[NUM_URBAN_HUBS] = { 0.50f, 0.22f, 0.78f, 0.25f, 0.76f, 0.50f, 0.16f, 0.85f };
__device__ __constant__ float kHubY[NUM_URBAN_HUBS] = { 0.50f, 0.26f, 0.24f, 0.74f, 0.76f, 0.18f, 0.52f, 0.50f };
__device__ __constant__ float kHubR[NUM_URBAN_HUBS] = { 0.24f, 0.20f, 0.20f, 0.20f, 0.20f, 0.17f, 0.16f, 0.16f };

// Reporta errores CUDA en stderr devolviendo false si falló la llamada
inline bool check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] %s:%d -> %s\n",
                     file, line, cudaGetErrorString(err));
        return false;
    }
    return true;
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)

// Reserva count elementos de tipo T en VRAM
template <typename T>
inline bool cuda_alloc_buf(T*& ptr, size_t count) {
    return CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
}

// Libera y pone en nullptr una lista variádica de punteros en VRAM
template <typename T, typename... Rest>
inline void cuda_free_ptrs(T*& first, Rest&... rest) {
    if (first != nullptr) {
        cudaFree(first);
        first = nullptr;
    }
    if constexpr (sizeof...(rest) > 0) {
        cuda_free_ptrs(rest...);
    }
}

// ============================================================================
// 2. Funciones __device__ auxiliares (RNG, geometría y zonas urbanas)
// ============================================================================

// Empaqueta componentes RGBA (8 bits c/u) en un entero uint32_t little-endian
__device__ __forceinline__ uint32_t pack_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (static_cast<uint32_t>(a) << 24) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(g) << 8)  |
            static_cast<uint32_t>(r);
}

// RNG stateless en [0, 1) basado en SplitMix64 (determinista por paso, agente y secuencia)
__device__ __forceinline__ float rand_u01(uint64_t seed, uint64_t step, uint32_t id, uint32_t seq) {
    uint64_t z = seed ^ (step * 0x9E3779B97F4A7C15ULL) ^ (static_cast<uint64_t>(id) << 32) ^ static_cast<uint64_t>(seq);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    uint32_t bits = static_cast<uint32_t>(z ^ (z >> 32));
    return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

// Calcula la posición fija de un hogar dentro de su distrito residencial
__device__ __forceinline__ void get_family_home_pos(
    const SimParams& params,
    uint64_t r_seed,
    uint32_t fam_id,
    uint8_t home_z_idx,
    float& out_hx,
    float& out_hy
) {
    if (params.num_urban_zones <= 0) {
        out_hx = 0.5f * params.world_width;
        out_hy = 0.5f * params.world_height;
        return;
    }
    int zidx = min(static_cast<int>(home_z_idx), params.num_urban_zones - 1);
    const UrbanZone& hz = params.urban_zones[zidx];
    float ux = (rand_u01(r_seed, 0ULL, fam_id, 100u) - 0.5f) * 2.0f * (hz.half_w * 0.90f);
    float uy = (rand_u01(r_seed, 0ULL, fam_id, 101u) - 0.5f) * 2.0f * (hz.half_h * 0.90f);
    out_hx = fminf(fmaxf(hz.cx + ux, 2.0f), params.world_width - 2.0f);
    out_hy = fminf(fmaxf(hz.cy + uy, 2.0f), params.world_height - 2.0f);
}

// Calcula una posición objetivo dentro de la zona urbana asignada al agente
__device__ __forceinline__ void get_agent_zone_pos(
    const SimParams& params,
    uint64_t r_seed,
    uint32_t agent_key,
    int zone_idx,
    float& out_tx,
    float& out_ty
) {
    if (params.num_urban_zones <= 0 || zone_idx < 0) {
        out_tx = 0.5f * params.world_width;
        out_ty = 0.5f * params.world_height;
        return;
    }
    int zidx = min(zone_idx, params.num_urban_zones - 1);
    const UrbanZone& uz = params.urban_zones[zidx];
    float ux = (rand_u01(r_seed, 0ULL, agent_key, 102u) - 0.5f) * 2.0f * (uz.half_w * 0.78f);
    float uy = (rand_u01(r_seed, 0ULL, agent_key, 103u) - 0.5f) * 2.0f * (uz.half_h * 0.78f);
    out_tx = fminf(fmaxf(uz.cx + ux, 2.0f), params.world_width - 2.0f);
    out_ty = fminf(fmaxf(uz.cy + uy, 2.0f), params.world_height - 2.0f);
}

// Devuelve el índice de la k-ésima zona de tipo ztype (módulo total disponible)
__device__ __forceinline__ int find_zone_by_type(
    const SimParams& params,
    ZoneType ztype,
    int kth
) {
    int count = 0;
    for (int z = 0; z < params.num_urban_zones; ++z) {
        if (params.urban_zones[z].type == ztype) {
            ++count;
        }
    }
    if (count == 0) return -1;
    int target = kth % count;
    int seen   = 0;
    for (int z = 0; z < params.num_urban_zones; ++z) {
        if (params.urban_zones[z].type == ztype) {
            if (seen == target) return z;
            ++seen;
        }
    }
    return -1;
}

// Devuelve el índice de la zona de tipo ztype más cercana a (px, py)
__device__ __forceinline__ int find_nearest_zone_by_type(
    const SimParams& params,
    ZoneType ztype,
    float px,
    float py
) {
    int best = -1;
    float best_d2 = 1e18f;
    for (int z = 0; z < params.num_urban_zones; ++z) {
        if (params.urban_zones[z].type == ztype) {
            float dx = params.urban_zones[z].cx - px;
            float dy = params.urban_zones[z].cy - py;
            float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best    = z;
            }
        }
    }
    return best;
}

// ============================================================================
// 3. Kernels de inicialización y Spatial Hashing
// ============================================================================

// Inicializa posiciones, familias, grupos de riesgo y pacientes cero por réplica
__global__ void kernel_init_agents(
    int total_agents,
    int agents_per_replica,
    int num_replicas,
    int initial_infected,
    SimParams params,
    uint64_t seed,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ vx,
    float* __restrict__ vy,
    uint8_t* __restrict__ state,
    uint8_t* __restrict__ risk_group,
    uint8_t* __restrict__ home_zone,
    uint8_t* __restrict__ rep_id,
    float* __restrict__ state_timer,
    uint32_t* __restrict__ home_id,
    uint8_t* __restrict__ work_zone,
    uint8_t* __restrict__ infection_source,
    float* __restrict__ infection_flash
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_agents) return;

    const int rep     = (num_replicas > 1) ? (i / agents_per_replica) : 0;
    const int local_i = (num_replicas > 1) ? (i % agents_per_replica) : i;
    const uint64_t r_seed = seed ^ (static_cast<uint64_t>(rep) * 0x9E3779B97F4A7C15ULL);
    const uint32_t uid    = static_cast<uint32_t>(local_i);

    rep_id[i] = static_cast<uint8_t>(rep);

    float px = 0.0f;
    float py = 0.0f;
    uint8_t rg = RISK_MEDIUM;

    if (params.enable_urban_layout && params.num_urban_zones > 0) {
        const uint32_t local_fam = static_cast<uint32_t>(local_i / FAMILY_SIZE);
        const int member_idx     = local_i % FAMILY_SIZE;
        const int fams_per_rep   = (agents_per_replica + FAMILY_SIZE - 1) / FAMILY_SIZE;
        const uint32_t g_fam     = static_cast<uint32_t>(rep * fams_per_rep) + local_fam;
        home_id[i]               = g_fam;

        // Mezcla de grupos de riesgo dentro de cada familia
        if (member_idx == 0) {
            rg = RISK_LOW;
        } else if (member_idx == 1) {
            rg = (rand_u01(r_seed, 0ULL, uid, 4u) < 0.5f) ? RISK_LOW : RISK_MEDIUM;
        } else if (member_idx == 2) {
            rg = RISK_MEDIUM;
        } else {
            rg = (rand_u01(r_seed, 0ULL, uid, 4u) < 0.60f) ? RISK_HIGH : RISK_MEDIUM;
        }

        int fam_k = static_cast<int>(local_fam + static_cast<uint32_t>(rand_u01(r_seed, 0ULL, local_fam, 99u) * 16.0f));
        int hz_idx = find_zone_by_type(params, ZONE_HOME, fam_k);
        if (hz_idx < 0) hz_idx = 0;

        float hx = 0.0f, hy = 0.0f;
        get_family_home_pos(params, r_seed, g_fam, static_cast<uint8_t>(hz_idx), hx, hy);
        px = fminf(fmaxf(hx + (rand_u01(r_seed, 0ULL, uid, 0u) - 0.5f) * 3.0f, 1.0f), params.world_width - 1.0f);
        py = fminf(fmaxf(hy + (rand_u01(r_seed, 0ULL, uid, 1u) - 0.5f) * 3.0f, 1.0f), params.world_height - 1.0f);
        home_zone[i] = static_cast<uint8_t>(hz_idx);

        // Asignar destino diurno divergente según rol familiar
        int wz_idx = hz_idx;
        if (rg == RISK_LOW) {
            int s_idx = find_zone_by_type(params, ZONE_SCHOOL, static_cast<int>(local_fam) + member_idx * 5);
            if (s_idx < 0) s_idx = find_zone_by_type(params, ZONE_WORK, static_cast<int>(local_fam) + member_idx * 5);
            if (s_idx >= 0) wz_idx = s_idx;
        } else if (rg == RISK_MEDIUM) {
            int w_idx = find_zone_by_type(params, ZONE_WORK, static_cast<int>(local_fam) + member_idx * 7);
            if (w_idx >= 0) wz_idx = w_idx;
        }
        work_zone[i] = static_cast<uint8_t>(wz_idx);
    } else {
        px = rand_u01(r_seed, 0ULL, uid, 0u) * params.world_width;
        py = rand_u01(r_seed, 0ULL, uid, 1u) * params.world_height;
        px = fminf(fmaxf(px, 0.0f), params.world_width - 1e-4f);
        py = fminf(fmaxf(py, 0.0f), params.world_height - 1e-4f);

        float r_risk = rand_u01(r_seed, 0ULL, uid, 4u);
        rg = (r_risk < 0.50f) ? RISK_LOW : ((r_risk < 0.85f) ? RISK_MEDIUM : RISK_HIGH);

        // Asignar hogar al polo más cercano y zona diurna de actividad/trabajo
        float nx = px / fmaxf(params.world_width, 1.0f);
        float ny = py / fmaxf(params.world_height, 1.0f);
        int h_id = 0;
        float best_d2 = 1e9f;
        for (int h = 0; h < NUM_URBAN_HUBS; ++h) {
            float dx = nx - kHubX[h];
            float dy = ny - kHubY[h];
            float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                h_id = h;
            }
        }
        int w_id = h_id;
        if (rand_u01(r_seed, 0ULL, uid, 6u) >= 0.65f) {
            int offset = 1 + min(NUM_URBAN_HUBS - 2, static_cast<int>(rand_u01(r_seed, 0ULL, uid, 7u) * static_cast<float>(NUM_URBAN_HUBS - 1)));
            w_id = (h_id + offset) % NUM_URBAN_HUBS;
        }
        home_zone[i] = static_cast<uint8_t>((w_id << 4) | (h_id & 0x0F));
        home_id[i]   = static_cast<uint32_t>(rep * agents_per_replica) + uid;
        work_zone[i] = static_cast<uint8_t>(w_id);
    }

    if (params.superspreader_frac > 0.0f && rand_u01(r_seed, 0ULL, uid, 5u) < params.superspreader_frac) {
        rg |= FLAG_SUPERSPREADER;
    }
    risk_group[i] = rg;

    float angle = rand_u01(r_seed, 0ULL, uid, 2u) * 2.0f * PI_F;
    x[i]  = px;
    y[i]  = py;
    vx[i] = cosf(angle) * params.base_speed;
    vy[i] = sinf(angle) * params.base_speed;

    if (local_i < initial_infected) {
        state[i]            = STATE_I;
        state_timer[i]      = params.infectious_days * (0.5f + 0.5f * rand_u01(r_seed, 0ULL, uid, 3u));
        infection_source[i] = SRC_HOME;
        infection_flash[i]  = 1.0f;
    } else {
        state[i]            = STATE_S;
        state_timer[i]      = 0.0f;
        infection_source[i] = SRC_NONE;
        infection_flash[i]  = 0.0f;
    }
}

// Reordena los arreglos SoA según agent_idx tras el Radix Sort por celda
__global__ void kernel_reorder_agents(
    int num_agents,
    const uint32_t* __restrict__ agent_idx,
    const float* __restrict__ x_in,
    const float* __restrict__ y_in,
    const float* __restrict__ vx_in,
    const float* __restrict__ vy_in,
    const uint8_t* __restrict__ state_in,
    const uint8_t* __restrict__ risk_in,
    const uint8_t* __restrict__ home_in,
    const uint8_t* __restrict__ rep_in,
    const float* __restrict__ timer_in,
    const uint32_t* __restrict__ home_id_in,
    const uint8_t* __restrict__ work_zone_in,
    const uint8_t* __restrict__ inf_src_in,
    const float* __restrict__ inf_flash_in,
    float* __restrict__ x_out,
    float* __restrict__ y_out,
    float* __restrict__ vx_out,
    float* __restrict__ vy_out,
    uint8_t* __restrict__ state_out,
    uint8_t* __restrict__ risk_out,
    uint8_t* __restrict__ home_out,
    uint8_t* __restrict__ rep_out,
    float* __restrict__ timer_out,
    uint32_t* __restrict__ home_id_out,
    uint8_t* __restrict__ work_zone_out,
    uint8_t* __restrict__ inf_src_out,
    float* __restrict__ inf_flash_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    int src = static_cast<int>(agent_idx[i]);
    x_out[i]         = x_in[src];
    y_out[i]         = y_in[src];
    vx_out[i]        = vx_in[src];
    vy_out[i]        = vy_in[src];
    state_out[i]     = state_in[src];
    risk_out[i]      = risk_in[src];
    home_out[i]      = home_in[src];
    rep_out[i]       = rep_in[src];
    timer_out[i]     = timer_in[src];
    home_id_out[i]   = home_id_in[src];
    work_zone_out[i] = work_zone_in[src];
    inf_src_out[i]   = inf_src_in[src];
    inf_flash_out[i] = inf_flash_in[src];
}

// ============================================================================
// 4. Kernels de dinámica (cuarentena, movimiento y transición SEIRD)
// ============================================================================

// Marca en cuarentena domiciliaria a los hogares con al menos un infectado
__global__ void kernel_mark_home_quarantine(
    int num_agents,
    int max_homes,
    const uint8_t* __restrict__ state,
    const uint32_t* __restrict__ home_id,
    uint8_t* __restrict__ home_quarantine
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    if (state[i] == STATE_I) {
        uint32_t hid = home_id[i];
        if (hid < static_cast<uint32_t>(max_homes)) {
            home_quarantine[hid] = 1u;
        }
    }
}

// Integra cinemática (rutina urbana / polos / cuarentena) y calcula cell_id para Spatial Hashing
__global__ void kernel_move_and_hash(
    int num_agents,
    int cells_per_replica,
    SimParams params,
    uint64_t seed,
    uint64_t step,
    float sim_day,
    float infected_ratio,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ vx,
    float* __restrict__ vy,
    const uint8_t* __restrict__ state,
    const uint8_t* __restrict__ risk_group,
    const uint8_t* __restrict__ home_zone,
    const uint8_t* __restrict__ rep_id,
    const uint32_t* __restrict__ home_id,
    const uint8_t* __restrict__ work_zone,
    const uint8_t* __restrict__ home_quarantine,
    const uint8_t* __restrict__ cell_quarantine,
    uint32_t* __restrict__ cell_id,
    uint32_t* __restrict__ agent_idx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    const uint32_t rep    = (rep_id != nullptr) ? static_cast<uint32_t>(rep_id[i]) : 0u;
    const uint64_t r_seed = seed ^ (static_cast<uint64_t>(rep) * 0x9E3779B97F4A7C15ULL);

    float px = x[i];
    float py = y[i];

    if (state[i] != STATE_D) {
        float pvx = vx[i];
        float pvy = vy[i];

        uint8_t rg = risk_group[i] & RISK_GROUP_MASK;
        float risk_mob = (rg == RISK_LOW) ? 1.15f : ((rg == RISK_HIGH) ? 0.70f : 1.0f);

        float local_quar = params.quarantine_factor;
        if (params.focal_quarantine && cell_quarantine != nullptr) {
            int cx0 = max(0, min(static_cast<int>(px / params.cell_size), params.grid_w - 1));
            int cy0 = max(0, min(static_cast<int>(py / params.cell_size), params.grid_h - 1));
            int g_cid0 = static_cast<int>(rep) * cells_per_replica + cy0 * params.grid_w + cx0;
            if (cell_quarantine[g_cid0] != 0u) {
                local_quar *= 0.15f;
            }
        }

        if (params.enable_urban_layout && params.num_urban_zones > 0 && home_id != nullptr && work_zone != nullptr) {
            const uint32_t hid = home_id[i];
            const uint8_t hz_idx = (home_zone != nullptr) ? home_zone[i] : 0u;
            float hx = 0.0f, hy = 0.0f;
            get_family_home_pos(params, r_seed, hid, hz_idx, hx, hy);

            float day_phase = sim_day - floorf(sim_day);
            float hour      = day_phase * 24.0f;
            bool is_weekend = ((static_cast<int>(floorf(sim_day)) % 7) >= 5);

            float target_x = hx;
            float target_y = hy;
            float target_r = 8.0f;
            float speed_mult = 1.0f;
            const uint32_t day_slot  = static_cast<uint32_t>(floorf(sim_day * 12.0f)); // Rota cada ~2h dentro del distrito
            const uint32_t agent_key = (hid * 7u) ^ static_cast<uint32_t>(work_zone[i] * 131u) ^ (rg * 17u);

            bool fam_quar = (params.household_quarantine && home_quarantine != nullptr && home_quarantine[hid] != 0u);
            bool need_hosp = (state[i] == STATE_I) && (rg == RISK_HIGH || (hid % 5u == 0u)) &&
                             (infected_ratio <= params.hospital_capacity_pct * 1.5f);

            if (need_hosp) {
                int hosp_idx = find_nearest_zone_by_type(params, ZONE_HOSPITAL, px, py);
                if (hosp_idx >= 0) {
                    get_agent_zone_pos(params, r_seed, agent_key, hosp_idx, target_x, target_y);
                    target_r = 22.0f;
                    speed_mult = 1.6f;
                }
            } else if (fam_quar) {
                local_quar *= 0.30f;
                target_r = 6.0f;
            } else if (hour >= 6.0f && hour < 8.5f) {
                // Mañana: traslado hogar -> (tránsito) -> trabajo/escuela
                if (!is_weekend && !params.close_schools_work && rg != RISK_HIGH) {
                    speed_mult = 1.6f + 0.8f * rand_u01(r_seed, 0ULL, agent_key, 104u);
                    int wz = min(static_cast<int>(work_zone[i]), params.num_urban_zones - 1);
                    if (hour < 7.25f) {
                        int tr = find_nearest_zone_by_type(params, ZONE_TRANSIT, px, py);
                        if (tr >= 0) {
                            get_agent_zone_pos(params, r_seed, agent_key + 3u, tr, target_x, target_y);
                            target_r = 24.0f;
                        } else {
                            get_agent_zone_pos(params, r_seed, agent_key, wz, target_x, target_y);
                            target_r = 28.0f;
                        }
                    } else {
                        get_agent_zone_pos(params, r_seed, agent_key, wz, target_x, target_y);
                        target_r = 28.0f;
                    }
                }
            } else if (hour >= 8.5f && hour < 17.5f) {
                // Día: circulación interna en trabajo/escuela o comercio/ocio en fin de semana
                if (is_weekend) {
                    int lz = find_zone_by_type(params, ZONE_LEISURE, static_cast<int>(hid));
                    if (lz >= 0 && (hid % 3u != 0u)) {
                        get_agent_zone_pos(params, r_seed, agent_key ^ (day_slot * 31u), lz, target_x, target_y);
                        target_r = 30.0f;
                        speed_mult = 1.6f;
                    }
                } else if (!params.close_schools_work && rg != RISK_HIGH) {
                    int wz = min(static_cast<int>(work_zone[i]), params.num_urban_zones - 1);
                    get_agent_zone_pos(params, r_seed, agent_key ^ (day_slot * 31u), wz, target_x, target_y);
                    target_r = 30.0f;
                    speed_mult = 1.6f;
                }
            } else if (hour >= 17.5f && hour < 20.0f) {
                // Tarde: retorno trabajo -> (tránsito) -> hogar
                if (!is_weekend && !params.close_schools_work && rg != RISK_HIGH) {
                    speed_mult = 1.6f + 0.8f * rand_u01(r_seed, 0ULL, agent_key, 105u);
                    if (hour < 18.75f) {
                        int tr = find_nearest_zone_by_type(params, ZONE_TRANSIT, px, py);
                        if (tr >= 0) {
                            get_agent_zone_pos(params, r_seed, agent_key + 5u, tr, target_x, target_y);
                            target_r = 24.0f;
                        }
                    }
                }
            }

            // Deriva angular suave
            float angle_j = (rand_u01(r_seed, step, static_cast<uint32_t>(i), 7u) - 0.5f) * 1.4f * params.dt;
            float cj = cosf(angle_j), sj = sinf(angle_j);
            float nvx = pvx * cj - pvy * sj;
            float nvy = pvx * sj + pvy * cj;
            pvx = nvx;
            pvy = nvy;

            float dx_t = target_x - px;
            float dy_t = target_y - py;
            float dist = sqrtf(dx_t * dx_t + dy_t * dy_t);
            float desired_spd = params.base_speed * speed_mult;
            if (dist > target_r && dist > 1e-3f) {
                float dir_x = dx_t / dist;
                float dir_y = dy_t / dist;
                float jitter_spd = 0.75f + 0.50f * rand_u01(r_seed, 0ULL, agent_key, 106u);
                float commute_spd = fminf(params.world_width * 6.0f, fmaxf(desired_spd, dist * 14.0f * jitter_spd));
                desired_spd = commute_spd;
                pvx = pvx * 0.25f + dir_x * commute_spd * 0.75f;
                pvy = pvy * 0.25f + dir_y * commute_spd * 0.75f;
            }

            float spd = sqrtf(pvx * pvx + pvy * pvy);
            if (spd > 1e-3f) {
                float norm = desired_spd / spd;
                pvx *= norm;
                pvy *= norm;
            }
        } else if (params.urban_clustering > 0.0f && home_zone != nullptr) {
            // Movilidad Lagrangiana orgánica clásica (Bloque 2)
            float day_phase = sim_day - floorf(sim_day);
            bool is_daytime = (day_phase >= 0.30f && day_phase < 0.75f);
            uint8_t hz = home_zone[i];
            int hub = is_daytime ? ((hz >> 4) & 0x0F) : (hz & 0x0F);
            hub = min(hub, NUM_URBAN_HUBS - 1);

            float hub_x = kHubX[hub] * params.world_width;
            float hub_y = kHubY[hub] * params.world_height;
            float hub_r = kHubR[hub] * fminf(params.world_width, params.world_height);

            float angle_j = (rand_u01(r_seed, step, static_cast<uint32_t>(i), 7u) - 0.5f) * 1.4f * params.dt;
            float cj = cosf(angle_j), sj = sinf(angle_j);
            float nvx = pvx * cj - pvy * sj;
            float nvy = pvx * sj + pvy * cj;
            pvx = nvx;
            pvy = nvy;

            float dx_h = hub_x - px;
            float dy_h = hub_y - py;
            float dist = sqrtf(dx_h * dx_h + dy_h * dy_h);
            if (dist > hub_r && dist > 1e-3f) {
                float excess = fminf(1.5f, (dist - hub_r) / fmaxf(hub_r, 1.0f));
                float pull   = params.urban_clustering * excess * params.dt * 4.0f;
                pvx += (dx_h / dist) * params.base_speed * pull;
                pvy += (dy_h / dist) * params.base_speed * pull;
            }

            float spd = sqrtf(pvx * pvx + pvy * pvy);
            if (spd > 1e-3f) {
                float norm = params.base_speed / spd;
                pvx *= norm;
                pvy *= norm;
            }
        }

        float scale = local_quar * risk_mob * params.dt;
        px += pvx * scale;
        py += pvy * scale;

        if (px < 0.0f) {
            px  = -px;
            pvx = fabsf(pvx);
        } else if (px >= params.world_width) {
            px  = 2.0f * params.world_width - px - 1e-4f;
            pvx = -fabsf(pvx);
        }

        if (py < 0.0f) {
            py  = -py;
            pvy = fabsf(pvy);
        } else if (py >= params.world_height) {
            py  = 2.0f * params.world_height - py - 1e-4f;
            pvy = -fabsf(pvy);
        }

        px = fminf(fmaxf(px, 0.0f), params.world_width - 1e-4f);
        py = fminf(fmaxf(py, 0.0f), params.world_height - 1e-4f);

        x[i]  = px;
        y[i]  = py;
        vx[i] = pvx;
        vy[i] = pvy;
    }

    int cx = static_cast<int>(px / params.cell_size);
    int cy = static_cast<int>(py / params.cell_size);
    cx = max(0, min(cx, params.grid_w - 1));
    cy = max(0, min(cy, params.grid_h - 1));

    cell_id[i]   = rep * static_cast<uint32_t>(cells_per_replica) + static_cast<uint32_t>(cy * params.grid_w + cx);
    agent_idx[i] = static_cast<uint32_t>(i);
}

// Determina el rango [cell_start, cell_end) de agentes en cada celda ordenada
__global__ void kernel_find_cell_bounds(
    int num_agents,
    const uint32_t* __restrict__ cell_id,
    int* __restrict__ cell_start,
    int* __restrict__ cell_end
) {
    __shared__ uint32_t smem[BLOCK_SIZE + 1];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_agents) {
        uint32_t cid = cell_id[idx];
        smem[threadIdx.x + 1] = cid;
        if (threadIdx.x == 0) {
            smem[0] = (idx == 0) ? 0xFFFFFFFFu : cell_id[idx - 1];
        }
    }
    __syncthreads();

    if (idx >= num_agents) return;

    uint32_t cid = smem[threadIdx.x + 1];
    if (idx == 0 || cid != smem[threadIdx.x]) {
        cell_start[cid] = idx;
        if (idx > 0) {
            cell_end[smem[threadIdx.x]] = idx;
        }
    }
    if (idx == num_agents - 1) {
        cell_end[cid] = num_agents;
    }
}

// Clasifica el origen espacial de un contagio según la zona urbana o cercanía al hogar
__device__ __forceinline__ uint8_t classify_infection_source(
    const SimParams& params,
    float px,
    float py
) {
    if (!params.enable_urban_layout || params.num_urban_zones <= 0) {
        return SRC_HOME;
    }
    uint8_t best_src = SRC_TRANSIT;
    for (int z = 0; z < params.num_urban_zones; ++z) {
        const UrbanZone& uz = params.urban_zones[z];
        if (fabsf(px - uz.cx) <= uz.half_w && fabsf(py - uz.cy) <= uz.half_h) {
            if (uz.type == ZONE_HOSPITAL) return SRC_HOSPITAL;
            if (uz.type == ZONE_TRANSIT)  return SRC_TRANSIT;
            if (uz.type == ZONE_WORK || uz.type == ZONE_SCHOOL) best_src = SRC_WORK;
            else if (uz.type == ZONE_LEISURE && best_src == SRC_TRANSIT) best_src = SRC_LEISURE;
            else if (uz.type == ZONE_HOME && best_src == SRC_TRANSIT) best_src = SRC_HOME;
        }
    }
    return best_src;
}

// Evalúa contagios S->E en vecindario 3x3 de celdas y transiciones E->I->R/D->S
__global__ void kernel_seir_step(
    int num_agents,
    int cells_per_replica,
    int num_replicas,
    SimParams params,
    uint64_t seed,
    uint64_t step,
    float infected_ratio,
    const float* __restrict__ x,
    const float* __restrict__ y,
    uint8_t* __restrict__ state,
    const uint8_t* __restrict__ risk_group,
    const uint8_t* __restrict__ rep_id,
    float* __restrict__ state_timer,
    uint8_t* __restrict__ infection_source,
    float* __restrict__ infection_flash,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const uint32_t* __restrict__ cell_infected,
    uint32_t* __restrict__ d_counts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    if (infection_flash != nullptr && infection_flash[i] > 0.0f) {
        infection_flash[i] = fmaxf(0.0f, infection_flash[i] - params.dt * 4.0f);
    }

    const int rep         = (num_replicas > 1 && rep_id != nullptr) ? static_cast<int>(rep_id[i]) : 0;
    const uint64_t r_seed = seed ^ (static_cast<uint64_t>(rep) * 0x9E3779B97F4A7C15ULL);
    const uint32_t uid    = static_cast<uint32_t>(i);
    uint8_t my_state      = state[i];

    if (my_state == STATE_D) {
        return;
    }

    if (my_state == STATE_R) {
        if (params.immunity_days > 0.0f) {
            float t = state_timer[i] - params.dt;
            if (t <= 0.0f) {
                state[i]       = STATE_S;
                state_timer[i] = 0.0f;
            } else {
                state_timer[i] = t;
            }
        }
        return;
    }

    if (my_state == STATE_E) {
        float t = state_timer[i] - params.dt;
        if (t <= 0.0f) {
            state[i]       = STATE_I;
            state_timer[i] = params.infectious_days * (0.75f + 0.5f * rand_u01(r_seed, step, uid, 0u));
        } else {
            state_timer[i] = t;
        }
        return;
    }

    if (my_state == STATE_I) {
        float t = state_timer[i] - params.dt;
        if (t <= 0.0f) {
            uint8_t rg = risk_group[i] & RISK_GROUP_MASK;
            float risk_cfr_mult = (rg == RISK_LOW) ? 0.25f : ((rg == RISK_HIGH) ? 3.50f : 1.0f);
            float hosp_mult = (infected_ratio > params.hospital_capacity_pct)
                ? fminf(2.5f, 1.0f + 10.0f * (infected_ratio - params.hospital_capacity_pct))
                : 1.0f;
            float eff_cfr = fminf(0.95f, params.fatality_rate * risk_cfr_mult * hosp_mult);

            float r = rand_u01(r_seed, step, uid, 0u);
            if (r < eff_cfr) {
                state[i]       = STATE_D;
                state_timer[i] = 0.0f;
            } else {
                state[i]       = STATE_R;
                state_timer[i] = (params.immunity_days > 0.0f)
                    ? params.immunity_days * (0.8f + 0.4f * rand_u01(r_seed, step, uid, 1u))
                    : 0.0f;
            }
        } else {
            state_timer[i] = t;
        }
        return;
    }

    // Vacunación diaria progresiva (S -> R)
    if (params.vaccination_rate > 0.0f) {
        float p_vacc = fminf(0.5f, params.vaccination_rate * params.dt);
        if (rand_u01(r_seed, step, uid, 8u) < p_vacc) {
            state[i]       = STATE_R;
            state_timer[i] = (params.immunity_days > 0.0f)
                ? params.immunity_days * (0.8f + 0.4f * rand_u01(r_seed, step, uid, 9u))
                : 0.0f;
            return;
        }
    }

    // my_state == STATE_S: evaluar contagio en la celda propia y las 8 vecinas de su réplica
    float px = x[i];
    float py = y[i];
    int cx = max(0, min(static_cast<int>(px / params.cell_size), params.grid_w - 1));
    int cy = max(0, min(static_cast<int>(py / params.cell_size), params.grid_h - 1));

    float r2_max = params.infection_radius * params.infection_radius;
    float p_step = 1.0f - powf(1.0f - fminf(params.beta, 0.999f), params.dt);
    uint32_t rng_seq = 0u;
    bool infected = false;
    const int rep_cell_base = rep * cells_per_replica;

    for (int dy = -1; dy <= 1 && !infected; ++dy) {
        int ny = cy + dy;
        if (ny < 0 || ny >= params.grid_h) continue;

        for (int dx = -1; dx <= 1 && !infected; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= params.grid_w) continue;

            int ncid = rep_cell_base + ny * params.grid_w + nx;
            if (cell_infected[ncid] == 0u) continue;
            int start = cell_start[ncid];
            if (start < 0) continue;
            int end = cell_end[ncid];

            for (int k = start; k < end; ++k) {
                if (state[k] == STATE_I) {
                    float ddx = x[k] - px;
                    float ddy = y[k] - py;
                    if (ddx * ddx + ddy * ddy <= r2_max) {
                        bool is_super = (risk_group[k] & FLAG_SUPERSPREADER) != 0u;
                        float p_contact = (is_super && params.superspreader_mult > 1.0f)
                            ? fminf(0.99f, 1.0f - powf(1.0f - p_step, params.superspreader_mult))
                            : p_step;
                        if (rand_u01(r_seed, step, uid, rng_seq++) < p_contact) {
                            infected = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    if (infected) {
        state[i]       = STATE_E;
        state_timer[i] = params.incubation_days * (0.75f + 0.5f * rand_u01(r_seed, step, uid, rng_seq));
        if (infection_source != nullptr) {
            infection_source[i] = classify_infection_source(params, px, py);
        }
        if (infection_flash != nullptr) {
            infection_flash[i] = 1.0f;
        }
        if (d_counts != nullptr) {
            if (num_replicas == 1) {
                unsigned mask = __activemask();
                int lane      = threadIdx.x & 31;
                int leader    = __ffs(mask) - 1;
                if (lane == leader) {
                    atomicAdd(&d_counts[5], static_cast<uint32_t>(__popc(mask)));
                }
            } else {
                atomicAdd(&d_counts[rep * COUNTS_STRIDE + 5], 1u);
            }
        }
    }
}

// ============================================================================
// 5. Kernels de renderizado 1080p (heatmap, zonas urbanas, agentes, compositing)
// ============================================================================

// Limpia el framebuffer o atenúa el cuadro previo para generar estelas
__global__ void kernel_prepare_framebuffer(uint32_t* __restrict__ fb, int num_pixels, bool enable_trails) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    if (!enable_trails) {
        fb[idx] = pack_rgba(11, 15, 25, 255);
        return;
    }

    // Desvanecimiento exponencial hacia el fondo #0B0F19 (11, 15, 25) para generar estelas
    uint32_t px = fb[idx];
    uint32_t r  = px & 0xFFu;
    uint32_t g  = (px >> 8) & 0xFFu;
    uint32_t b  = (px >> 16) & 0xFFu;

    uint32_t nr = 11u + (((r > 11u ? r - 11u : 0u) * 204u) >> 8);
    uint32_t ng = 15u + (((g > 15u ? g - 15u : 0u) * 204u) >> 8);
    uint32_t nb = 25u + (((b > 25u ? b - 25u : 0u) * 204u) >> 8);

    fb[idx] = pack_rgba(static_cast<uint8_t>(nr), static_cast<uint8_t>(ng), static_cast<uint8_t>(nb), 255);
}

// Mapea densidad normalizada t in [0, 1] a una paleta térmica RGB
__device__ __forceinline__ void thermal_palette(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    // Paleta térmica estilo Inferno/Magma: #0B0F19 -> Púrpura -> Naranja -> Amarillo brillante
    if (t < 0.33f) {
        float u = t / 0.33f;
        r = static_cast<uint8_t>(11.0f  + u * (120.0f - 11.0f));
        g = static_cast<uint8_t>(15.0f  + u * (28.0f  - 15.0f));
        b = static_cast<uint8_t>(25.0f  + u * (129.0f - 25.0f));
    } else if (t < 0.66f) {
        float u = (t - 0.33f) / 0.33f;
        r = static_cast<uint8_t>(120.0f + u * (239.0f - 120.0f));
        g = static_cast<uint8_t>(28.0f  + u * (85.0f  - 28.0f));
        b = static_cast<uint8_t>(129.0f + u * (30.0f  - 129.0f));
    } else {
        float u = (t - 0.66f) / 0.34f;
        r = static_cast<uint8_t>(239.0f + u * (253.0f - 239.0f));
        g = static_cast<uint8_t>(85.0f  + u * (231.0f - 85.0f));
        b = static_cast<uint8_t>(30.0f  + u * (120.0f - 30.0f));
    }
}

// Dibuja el mapa de calor bilineal de densidad de infectados y hotspots (Z > 2)
__global__ void kernel_render_heatmap_1080p(
    SimParams params,
    const uint32_t* __restrict__ cell_infected,
    const float* __restrict__ spatial_stats,
    uint32_t* __restrict__ fb,
    bool hybrid_mode
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= FRAME_WIDTH || py >= FRAME_HEIGHT) return;

    float zoom    = fmaxf(1.0f, params.cam_zoom);
    float view_w  = params.world_width  / zoom;
    float view_h  = params.world_height / zoom;
    float view_x0 = params.cam_cx - 0.5f * view_w;
    float view_y0 = params.cam_cy - 0.5f * view_h;

    float world_x = view_x0 + ((static_cast<float>(px) + 0.5f) / static_cast<float>(FRAME_WIDTH))  * view_w;
    float world_y = view_y0 + ((static_cast<float>(py) + 0.5f) / static_cast<float>(FRAME_HEIGHT)) * view_h;
    if (world_x < 0.0f || world_x >= params.world_width || world_y < 0.0f || world_y >= params.world_height) {
        fb[py * FRAME_WIDTH + px] = pack_rgba(11, 15, 25, 255);
        return;
    }
    float gx = fmaxf(0.0f, (world_x / params.cell_size) - 0.5f);
    float gy = fmaxf(0.0f, (world_y / params.cell_size) - 0.5f);

    int x0 = max(0, min(static_cast<int>(gx), params.grid_w - 1));
    int y0 = max(0, min(static_cast<int>(gy), params.grid_h - 1));
    int x1 = min(x0 + 1, params.grid_w - 1);
    int y1 = min(y0 + 1, params.grid_h - 1);

    float tx = gx - static_cast<float>(x0);
    float ty = gy - static_cast<float>(y0);

    float c00 = static_cast<float>(cell_infected[y0 * params.grid_w + x0]);
    float c10 = static_cast<float>(cell_infected[y0 * params.grid_w + x1]);
    float c01 = static_cast<float>(cell_infected[y1 * params.grid_w + x0]);
    float c11 = static_cast<float>(cell_infected[y1 * params.grid_w + x1]);

    float val = (1.0f - ty) * ((1.0f - tx) * c00 + tx * c10) +
                ty          * ((1.0f - tx) * c01 + tx * c11);

    float num_cells = static_cast<float>(params.grid_w * params.grid_h);
    float mean      = spatial_stats[0] / fmaxf(1.0f, num_cells);
    float variance  = fmaxf(0.0f, (spatial_stats[1] / fmaxf(1.0f, num_cells)) - (mean * mean));
    float stddev    = sqrtf(variance);

    // Escala logarítmica suave para resaltar tanto brotes incipientes como focos densos
    float norm = log1pf(val) / log1pf(fmaxf(6.0f, mean + 4.0f * stddev));
    if (hybrid_mode) {
        norm *= 0.85f;
    }

    uint8_t hr, hg, hb;
    thermal_palette(norm, hr, hg, hb);

    // Resaltado de hotspot (Z > 2.0) en el centro de celda
    if (stddev > 1e-4f) {
        float z = (val - mean) / stddev;
        if (z > 2.0f) {
            float boost = fminf((z - 2.0f) * 0.18f, 0.45f);
            hr = static_cast<uint8_t>(fminf(255.0f, static_cast<float>(hr) + boost * 255.0f));
            hg = static_cast<uint8_t>(fminf(255.0f, static_cast<float>(hg) + boost * 80.0f));
        }
    }

    int idx = py * FRAME_WIDTH + px;
    if (params.enable_trails) {
        // Mezcla suave con el framebuffer previo para mantener estelas bajo el heatmap
        uint32_t old = fb[idx];
        uint8_t or_ = static_cast<uint8_t>(old & 0xFFu);
        uint8_t og_ = static_cast<uint8_t>((old >> 8) & 0xFFu);
        uint8_t ob_ = static_cast<uint8_t>((old >> 16) & 0xFFu);
        hr = static_cast<uint8_t>(max(static_cast<int>(hr), (static_cast<int>(or_) * 2 + static_cast<int>(hr)) / 3));
        hg = static_cast<uint8_t>(max(static_cast<int>(hg), (static_cast<int>(og_) * 2 + static_cast<int>(hg)) / 3));
        hb = static_cast<uint8_t>(max(static_cast<int>(hb), (static_cast<int>(ob_) * 2 + static_cast<int>(hb)) / 3));
    }
    fb[idx] = pack_rgba(hr, hg, hb, 255);
}

// Dibuja distritos urbanos, iluminación día/noche y cuadrícula del modo mosaico
__global__ void kernel_render_urban_zones_1080p(
    SimParams params,
    int num_replicas,
    float sim_day,
    uint32_t* __restrict__ fb
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= FRAME_WIDTH || py >= FRAME_HEIGHT) return;

    const bool is_mosaic = (num_replicas > 1 && params.ensemble_grid_view);
    int idx = py * FRAME_WIDTH + px;

    float world_x = 0.0f;
    float world_y = 0.0f;
    float px_world = 1.0f;
    bool in_bounds = true;
    bool is_tile_border = false;

    if (is_mosaic) {
        int cols = 1, rows = 1;
        compute_mosaic_grid(num_replicas, cols, rows);
        int tile_w = FRAME_WIDTH / cols;
        int tile_h = FRAME_HEIGHT / rows;
        int col = min(cols - 1, px / max(1, tile_w));
        int row = min(rows - 1, py / max(1, tile_h));
        int r_id = row * cols + col;
        int lx = px - col * tile_w;
        int ly = py - row * tile_h;
        is_tile_border = (lx <= 1 || ly <= 1 || lx >= tile_w - 1 || ly >= tile_h - 1);
        if (r_id >= num_replicas) {
            fb[idx] = pack_rgba(11, 15, 25, 255);
            return;
        }
        world_x = (static_cast<float>(lx) + 0.5f) / static_cast<float>(max(1, tile_w)) * params.world_width;
        world_y = (static_cast<float>(ly) + 0.5f) / static_cast<float>(max(1, tile_h)) * params.world_height;
        px_world = params.world_width / static_cast<float>(max(1, tile_w));
    } else {
        float zoom    = fmaxf(1.0f, params.cam_zoom);
        float view_w  = params.world_width  / zoom;
        float view_h  = params.world_height / zoom;
        float view_x0 = params.cam_cx - 0.5f * view_w;
        float view_y0 = params.cam_cy - 0.5f * view_h;

        world_x = view_x0 + ((static_cast<float>(px) + 0.5f) / static_cast<float>(FRAME_WIDTH))  * view_w;
        world_y = view_y0 + ((static_cast<float>(py) + 0.5f) / static_cast<float>(FRAME_HEIGHT)) * view_h;
        px_world = view_w / static_cast<float>(FRAME_WIDTH);
        in_bounds = (world_x >= 0.0f && world_x < params.world_width &&
                     world_y >= 0.0f && world_y < params.world_height);
    }

    uint32_t old = fb[idx];
    float r = static_cast<float>(old & 0xFFu);
    float g = static_cast<float>((old >> 8) & 0xFFu);
    float b = static_cast<float>((old >> 16) & 0xFFu);

    // Iluminación Día / Noche suave
    float hour = (sim_day - floorf(sim_day)) * 24.0f;
    float day_light = 0.5f - 0.5f * cosf((hour - 1.0f) * (2.0f * PI_F / 24.0f));
    day_light = fminf(fmaxf(day_light, 0.0f), 1.0f);
    if (params.render_mode == RENDER_AGENTS || is_mosaic || params.focused_replica > 0) {
        r = r * 0.85f + 12.0f * day_light + 4.0f;
        g = g * 0.85f + 16.0f * day_light + 6.0f;
        b = b * 0.85f + 10.0f * day_light + 18.0f * (1.0f - day_light);
    }

    if (params.enable_urban_layout && params.num_urban_zones > 0 && in_bounds) {
        float border_w = fmaxf(1.2f, 2.0f * px_world);

        for (int z = 0; z < params.num_urban_zones; ++z) {
            const UrbanZone& uz = params.urban_zones[z];
            float adx = fabsf(world_x - uz.cx);
            float ady = fabsf(world_y - uz.cy);
            if (adx <= uz.half_w && ady <= uz.half_h) {
                float zr = 80.0f, zg = 200.0f, zb = 140.0f;
                switch (uz.type) {
                    case ZONE_HOME:     zr = 74.0f;  zg = 222.0f; zb = 128.0f; break;
                    case ZONE_WORK:     zr = 56.0f;  zg = 189.0f; zb = 248.0f; break;
                    case ZONE_SCHOOL:   zr = 167.0f; zg = 139.0f; zb = 250.0f; break;
                    case ZONE_HOSPITAL: zr = 248.0f; zg = 113.0f; zb = 113.0f; break;
                    case ZONE_LEISURE:  zr = 251.0f; zg = 146.0f; zb = 60.0f;  break;
                    case ZONE_TRANSIT:  zr = 250.0f; zg = 204.0f; zb = 21.0f;  break;
                }
                if (adx >= uz.half_w - border_w || ady >= uz.half_h - border_w) {
                    r = r * 0.55f + zr * 0.45f;
                    g = g * 0.55f + zg * 0.45f;
                    b = b * 0.55f + zb * 0.45f;
                } else {
                    r = r * 0.90f + zr * 0.10f;
                    g = g * 0.90f + zg * 0.10f;
                    b = b * 0.90f + zb * 0.10f;
                }
                // Cruz médica en el centro del hospital
                if (uz.type == ZONE_HOSPITAL) {
                    float cr = fminf(uz.half_w, uz.half_h) * 0.30f;
                    float cw = fmaxf(1.5f, 2.2f * px_world);
                    if ((adx < cw && ady < cr) || (ady < cw && adx < cr)) {
                        r = 255.0f; g = 255.0f; b = 255.0f;
                    }
                }
                break;
            }
        }
    }

    if (is_tile_border) {
        r = 51.0f; g = 65.0f; b = 85.0f;
    }

    fb[idx] = pack_rgba(
        static_cast<uint8_t>(fminf(255.0f, r)),
        static_cast<uint8_t>(fminf(255.0f, g)),
        static_cast<uint8_t>(fminf(255.0f, b)),
        255
    );
}

// Proyecta cada agente en 1080p resolviendo solapamiento por prioridad visual (atomicMax)
__global__ void kernel_render_agents_prio(
    int num_agents,
    int num_replicas,
    SimParams params,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const uint8_t* __restrict__ state,
    const uint8_t* __restrict__ rep_id,
    const uint32_t* __restrict__ home_id,
    const float* __restrict__ infection_flash,
    uint32_t* __restrict__ fb_prio,
    bool only_active_ei
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    const bool is_mosaic = (num_replicas > 1 && params.ensemble_grid_view);
    const int my_rep     = (num_replicas > 1 && rep_id != nullptr) ? static_cast<int>(rep_id[i]) : 0;
    const int target_rep = max(0, min(params.focused_replica, num_replicas - 1));
    if (!is_mosaic && num_replicas > 1 && my_rep != target_rep) return;

    const bool is_selected_fam = (!is_mosaic && params.selected_home_id >= 0 && home_id != nullptr &&
                                  home_id[i] == static_cast<uint32_t>(params.selected_home_id));
    const bool has_flash       = (infection_flash != nullptr && infection_flash[i] > 0.05f);

    uint8_t st = state[i];
    if (!is_mosaic && only_active_ei && st != STATE_E && st != STATE_I && !is_selected_fam && !has_flash) {
        return;
    }

    int px = 0;
    int py = 0;
    float view_w = params.world_width;
    float zoom   = fmaxf(1.0f, params.cam_zoom);

    if (is_mosaic) {
        int cols = 1, rows = 1;
        compute_mosaic_grid(num_replicas, cols, rows);
        int tile_w = FRAME_WIDTH / cols;
        int tile_h = FRAME_HEIGHT / rows;
        int col    = my_rep % cols;
        int row    = my_rep / cols;
        float ux   = fminf(fmaxf(x[i] / fmaxf(params.world_width, 1.0f), 0.0f), 0.9999f);
        float uy   = fminf(fmaxf(y[i] / fmaxf(params.world_height, 1.0f), 0.0f), 0.9999f);
        px = col * tile_w + static_cast<int>(ux * static_cast<float>(tile_w));
        py = row * tile_h + static_cast<int>(uy * static_cast<float>(tile_h));
        zoom = 1.0f;
    } else {
        view_w        = params.world_width / zoom;
        float view_h  = params.world_height / zoom;
        float view_x0 = params.cam_cx - 0.5f * view_w;
        float view_y0 = params.cam_cy - 0.5f * view_h;

        float ux = (x[i] - view_x0) / view_w;
        float uy = (y[i] - view_y0) / view_h;
        if (ux < 0.0f || ux >= 1.0f || uy < 0.0f || uy >= 1.0f) return;

        px = static_cast<int>(ux * static_cast<float>(FRAME_WIDTH));
        py = static_cast<int>(uy * static_cast<float>(FRAME_HEIGHT));
    }
    px = max(0, min(px, FRAME_WIDTH - 1));
    py = max(0, min(py, FRAME_HEIGHT - 1));

    uint32_t prio = 0;
    uint32_t rgb  = 0;
    switch (st) {
        case STATE_D: prio = 1u; rgb = 0x008B7464u; break; // Gris pizarra (100, 116, 139)
        case STATE_S: prio = 2u; rgb = 0x00F8BD38u; break; // Azul cian    (56, 189, 248)
        case STATE_R: prio = 3u; rgb = 0x005EC522u; break; // Verde        (34, 197, 94)
        case STATE_E: prio = 4u; rgb = 0x0015CCFAu; break; // Ámbar        (250, 204, 21)
        case STATE_I: prio = 5u; rgb = 0x004444EFu; break; // Rojo intenso (239, 68, 68)
        default:      prio = 1u; rgb = 0x00FFFFFFu; break;
    }

    int dot_r = (zoom >= 10.0f) ? 4 : ((zoom >= 5.0f) ? 3 : ((zoom >= 2.5f) ? 2 : 0));
    uint32_t packed_center = (prio << 24) | rgb;
    if (dot_r == 0) {
        atomicMax(&fb_prio[py * FRAME_WIDTH + px], packed_center);
    } else {
        int r2 = dot_r * dot_r;
        for (int dy = -dot_r; dy <= dot_r; ++dy) {
            int ny = py + dy;
            if (ny < 0 || ny >= FRAME_HEIGHT) continue;
            for (int dx = -dot_r; dx <= dot_r; ++dx) {
                if (dx * dx + dy * dy > r2) continue;
                int nx = px + dx;
                if (nx < 0 || nx >= FRAME_WIDTH) continue;
                atomicMax(&fb_prio[ny * FRAME_WIDTH + nx], packed_center);
            }
        }
    }

    // Destello de nuevo contagio (S -> E) y resaltado de familia seleccionada
    if (has_flash) {
        uint32_t packed_flash = (6u << 24) | 0x0040E0FFu; // Dorado brillante
        int fr = dot_r + 2;
        for (int dy = -fr; dy <= fr; ++dy) {
            int ny = py + dy;
            if (ny < 0 || ny >= FRAME_HEIGHT) continue;
            for (int dx = -fr; dx <= fr; ++dx) {
                int d2 = dx * dx + dy * dy;
                if (d2 < fr * fr - 2 || d2 > fr * fr + 2) continue;
                int nx = px + dx;
                if (nx < 0 || nx >= FRAME_WIDTH) continue;
                atomicMax(&fb_prio[ny * FRAME_WIDTH + nx], packed_flash);
            }
        }
    }

    if (is_selected_fam) {
        uint32_t packed_fam = (7u << 24) | 0x00FFFFFFu; // Blanco brillante
        int sr = dot_r + 3;
        for (int dy = -sr; dy <= sr; ++dy) {
            int ny = py + dy;
            if (ny < 0 || ny >= FRAME_HEIGHT) continue;
            for (int dx = -sr; dx <= sr; ++dx) {
                int d2 = dx * dx + dy * dy;
                if (d2 > sr * sr) continue;
                int nx = px + dx;
                if (nx < 0 || nx >= FRAME_WIDTH) continue;
                atomicMax(&fb_prio[ny * FRAME_WIDTH + nx], packed_fam);
            }
        }
    }

    // Halo o anillo real de radio de contagio para infectados
    if (st == STATE_I) {
        if (params.show_rings && zoom >= 3.5f) {
            float ring_px = (params.infection_radius / view_w) * static_cast<float>(FRAME_WIDTH);
            if (ring_px >= static_cast<float>(dot_r + 2)) {
                int steps = min(140, max(40, static_cast<int>(ring_px * 4.5f)));
                uint32_t packed_ring = (1u << 24) | 0x00695ADCu; // Anillo coral de 2px detrás de agentes
                for (int s = 0; s < steps; ++s) {
                    float ang = (2.0f * PI_F * static_cast<float>(s)) / static_cast<float>(steps);
                    float ca  = cosf(ang);
                    float sa  = sinf(ang);
                    for (int t = 0; t <= 1; ++t) {
                        float rr = ring_px + static_cast<float>(t);
                        int nx = px + static_cast<int>(roundf(ca * rr));
                        int ny = py + static_cast<int>(roundf(sa * rr));
                        if (nx >= 0 && nx < FRAME_WIDTH && ny >= 0 && ny < FRAME_HEIGHT) {
                            atomicMax(&fb_prio[ny * FRAME_WIDTH + nx], packed_ring);
                        }
                    }
                }
            }
        } else {
            uint32_t packed_halo = (4u << 24) | 0x003333DCu;
            for (int dy = -1; dy <= 1; ++dy) {
                int ny = py + dy;
                if (ny < 0 || ny >= FRAME_HEIGHT) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (abs(dx) <= dot_r && abs(dy) <= dot_r) continue;
                    int nx = px + dx;
                    if (nx < 0 || nx >= FRAME_WIDTH) continue;
                    atomicMax(&fb_prio[ny * FRAME_WIDTH + nx], packed_halo);
                }
            }
        }
    }
}

// Compone el buffer de prioridad sobre el framebuffer final 1080p
__global__ void kernel_composite_prio(uint32_t* __restrict__ fb, const uint32_t* __restrict__ fb_prio, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    uint32_t val = fb_prio[idx];
    if (val != 0u) {
        fb[idx] = 0xFF000000u | (val & 0x00FFFFFFu);
    }
}

// ============================================================================
// 6. Kernels de telemetría e inspección familiar
// ============================================================================

// Acumula conteos SEIRD, orígenes de contagio y hospitalizados por réplica
__global__ void kernel_count_states(
    int num_agents,
    int num_replicas,
    SimParams params,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const uint8_t* __restrict__ state,
    const uint8_t* __restrict__ rep_id,
    const uint8_t* __restrict__ infection_source,
    uint32_t* __restrict__ global_counts
) {
    __shared__ uint32_t s_counts[MAX_REPLICAS * 11]; // 0..4: S..D | 5..9: SRC_HOME..SRC_LEISURE | 10: hospitalized
    const int total_bins = num_replicas * 11;
    for (int idx = threadIdx.x; idx < total_bins; idx += BLOCK_SIZE) {
        s_counts[idx] = 0u;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_agents) {
        uint8_t st = state[i];
        int rep    = (num_replicas > 1 && rep_id != nullptr) ? static_cast<int>(rep_id[i]) : 0;
        if (st < 5 && rep < num_replicas) {
            atomicAdd(&s_counts[rep * 11 + static_cast<int>(st)], 1u);
        }
        if (rep < num_replicas && infection_source != nullptr) {
            uint8_t src = infection_source[i];
            if (src >= SRC_HOME && src <= SRC_LEISURE) {
                atomicAdd(&s_counts[rep * 11 + 5 + static_cast<int>(src - SRC_HOME)], 1u);
            }
        }
        if (rep < num_replicas && st == STATE_I && params.enable_urban_layout && params.num_urban_zones > 0) {
            float px = x[i];
            float py = y[i];
            for (int z = 0; z < params.num_urban_zones; ++z) {
                if (params.urban_zones[z].type == ZONE_HOSPITAL) {
                    if (fabsf(px - params.urban_zones[z].cx) <= params.urban_zones[z].half_w &&
                        fabsf(py - params.urban_zones[z].cy) <= params.urban_zones[z].half_h) {
                        atomicAdd(&s_counts[rep * 11 + 10], 1u);
                        break;
                    }
                }
            }
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < total_bins; idx += BLOCK_SIZE) {
        uint32_t val = s_counts[idx];
        if (val > 0u) {
            int rep = idx / 11;
            int bin = idx % 11;
            int dst = (bin < 5) ? bin : (bin + 1); // Salta el slot 5 (new_exposures)
            atomicAdd(&global_counts[rep * COUNTS_STRIDE + dst], val);
        }
    }
}

// Cuenta infectados por celda y acumula suma y suma de cuadrados espaciales
__global__ void kernel_cell_infected_stats(
    int num_cells,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const uint8_t* __restrict__ state,
    uint32_t* __restrict__ cell_infected,
    float* __restrict__ spatial_stats
) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sum_sq[BLOCK_SIZE];

    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    float inf_count = 0.0f;

    if (cid < num_cells) {
        int start = cell_start[cid];
        uint32_t count = 0;
        if (start >= 0) {
            int end = cell_end[cid];
            for (int k = start; k < end; ++k) {
                uint8_t st = state[k];
                if (st == STATE_I || st == STATE_E) {
                    ++count;
                }
            }
        }
        cell_infected[cid] = count;
        inf_count = static_cast<float>(count);
    }

    s_sum[threadIdx.x]    = inf_count;
    s_sum_sq[threadIdx.x] = inf_count * inf_count;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]    += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (s_sum[0] > 0.0f) {
            atomicAdd(&spatial_stats[0], s_sum[0]);
            atomicAdd(&spatial_stats[1], s_sum_sq[0]);
        }
    }
}

// Calcula Z-score por celda, cuenta hotspots y activa cuarentena focal (Z > 2.0)
__global__ void kernel_cell_zscore(
    int num_cells,
    const uint32_t* __restrict__ cell_infected,
    const float* __restrict__ spatial_stats,
    uint32_t* __restrict__ hotspot_out,
    uint8_t* __restrict__ cell_quarantine,
    bool focal_quarantine
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_cells) return;

    float n = static_cast<float>(num_cells);
    float mean = spatial_stats[0] / n;
    float variance = fmaxf(0.0f, (spatial_stats[1] / n) - (mean * mean));
    float stddev = sqrtf(variance);

    if (stddev <= 1e-4f) {
        if (focal_quarantine && cell_quarantine != nullptr) {
            cell_quarantine[cid] = 0u;
        }
        return;
    }

    float z = (static_cast<float>(cell_infected[cid]) - mean) / stddev;
    if (focal_quarantine && cell_quarantine != nullptr) {
        cell_quarantine[cid] = (z > 2.0f) ? 1u : 0u;
    }
    if (hotspot_out != nullptr) {
        if (z > 2.0f) {
            atomicAdd(&hotspot_out[0], 1u);
        }
        if (z > 0.0f) {
            // Para floats positivos, la representación IEEE-754 mantiene el orden como uint32
            atomicMax(&hotspot_out[1], __float_as_uint(z));
        }
    }
}

} // namespace

// ============================================================================
// 7. API pública: inicialización, paso de simulación, render y consultas
// ============================================================================

bool cuda_init_simulation(
    AgentSystemDevice& dev,
    int agents_per_replica,
    int initial_infected,
    const SimParams& params,
    uint64_t seed
) {
    cuda_free_simulation(dev);

    // Evitar busy-spin de CPU en WDDM al esperar sincronización de kernels/copias
    cudaError_t flag_err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (flag_err != cudaSuccess && flag_err != cudaErrorSetOnActiveProcess) {
        CUDA_CHECK(flag_err);
    }

    const int k_rep        = std::clamp(params.num_replicas, 1, MAX_REPLICAS);
    const int total_agents = agents_per_replica * k_rep;
    const int cells_rep    = params.grid_w * params.grid_h;
    const int total_cells  = cells_rep * k_rep;

    dev.num_agents         = total_agents;
    dev.agents_per_replica = agents_per_replica;
    dev.num_replicas       = k_rep;
    dev.num_cells          = cells_rep;
    g_sim_seed             = seed;
    g_sim_step             = 0ULL;
    g_sim_day              = 0.0f;
    g_last_infected_ratio  = (agents_per_replica > 0)
        ? (static_cast<float>(initial_infected) / static_cast<float>(agents_per_replica))
        : 0.0f;
    g_max_homes = total_agents + 1;

    const size_t n_ag  = static_cast<size_t>(total_agents);
    const size_t n_cel = static_cast<size_t>(total_cells);
    const size_t n_px  = static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT;

    bool ok = true;
    ok &= CUDA_CHECK(cudaStreamCreateWithFlags(&g_sim_stream, cudaStreamNonBlocking));
    ok &= CUDA_CHECK(cudaEventCreate(&g_ev_timer_start));
    ok &= CUDA_CHECK(cudaEventCreate(&g_ev_timer_stop));

    ok &= cuda_alloc_buf(dev.x, n_ag) && cuda_alloc_buf(dev.y, n_ag) &&
          cuda_alloc_buf(dev.vx, n_ag) && cuda_alloc_buf(dev.vy, n_ag) &&
          cuda_alloc_buf(dev.state, n_ag) && cuda_alloc_buf(dev.risk_group, n_ag) &&
          cuda_alloc_buf(dev.home_zone, n_ag) && cuda_alloc_buf(dev.rep_id, n_ag) &&
          cuda_alloc_buf(dev.state_timer, n_ag) && cuda_alloc_buf(dev.home_id, n_ag) &&
          cuda_alloc_buf(dev.work_zone, n_ag) && cuda_alloc_buf(dev.infection_source, n_ag) &&
          cuda_alloc_buf(dev.infection_flash, n_ag) &&
          cuda_alloc_buf(dev.cell_id, n_ag) && cuda_alloc_buf(dev.agent_idx, n_ag) &&
          cuda_alloc_buf(dev.cell_start, n_cel) && cuda_alloc_buf(dev.cell_end, n_cel) &&
          cuda_alloc_buf(dev.framebuffer_rgba, n_px * 4) &&
          cuda_alloc_buf(g_d_fb_prio, n_px) &&
          cuda_alloc_buf(g_d_counts, static_cast<size_t>(k_rep) * COUNTS_STRIDE) &&
          cuda_alloc_buf(g_d_cell_infected, n_cel) &&
          cuda_alloc_buf(g_d_spatial_stats, 2) && cuda_alloc_buf(g_d_hotspot_out, 2) &&
          cuda_alloc_buf(g_d_cell_id_alt, n_ag) && cuda_alloc_buf(g_d_agent_idx_alt, n_ag) &&
          cuda_alloc_buf(g_d_x_alt, n_ag) && cuda_alloc_buf(g_d_y_alt, n_ag) &&
          cuda_alloc_buf(g_d_vx_alt, n_ag) && cuda_alloc_buf(g_d_vy_alt, n_ag) &&
          cuda_alloc_buf(g_d_state_alt, n_ag) && cuda_alloc_buf(g_d_risk_alt, n_ag) &&
          cuda_alloc_buf(g_d_home_zone_alt, n_ag) && cuda_alloc_buf(g_d_rep_id_alt, n_ag) &&
          cuda_alloc_buf(g_d_home_id_alt, n_ag) && cuda_alloc_buf(g_d_work_zone_alt, n_ag) &&
          cuda_alloc_buf(g_d_inf_src_alt, n_ag) && cuda_alloc_buf(g_d_inf_flash_alt, n_ag) &&
          cuda_alloc_buf(g_d_timer_alt, n_ag) &&
          cuda_alloc_buf(g_d_home_quarantine, static_cast<size_t>(g_max_homes)) &&
          cuda_alloc_buf(g_d_cell_quarantine, n_cel);

    ok &= CUDA_CHECK(cudaMallocHost(&g_h_counts_pinned, static_cast<size_t>(k_rep) * COUNTS_STRIDE * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMallocHost(&g_h_hotspot_pinned, 2 * sizeof(uint32_t)));

    if (ok) {
        cub::DoubleBuffer<uint32_t> d_keys(dev.cell_id, g_d_cell_id_alt);
        cub::DoubleBuffer<uint32_t> d_vals(dev.agent_idx, g_d_agent_idx_alt);
        g_sort_temp_bytes = 0;
        ok &= CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g_sort_temp_bytes, d_keys, d_vals, total_agents));
        if (ok && g_sort_temp_bytes > 0) {
            ok &= CUDA_CHECK(cudaMalloc(&g_d_sort_temp, g_sort_temp_bytes));
        }
    }

    if (!ok) {
        cuda_free_simulation(dev);
        return false;
    }

    CUDA_CHECK(cudaMemsetAsync(g_d_cell_quarantine, 0, static_cast<size_t>(total_cells) * sizeof(uint8_t), g_sim_stream));
    CUDA_CHECK(cudaMemsetAsync(g_d_home_quarantine, 0, static_cast<size_t>(g_max_homes) * sizeof(uint8_t), g_sim_stream));
    CUDA_CHECK(cudaMemsetAsync(g_d_counts, 0, static_cast<size_t>(k_rep) * COUNTS_STRIDE * sizeof(uint32_t), g_sim_stream));

    int grid = (total_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_init_agents<<<grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        total_agents,
        agents_per_replica,
        k_rep,
        initial_infected,
        params,
        seed,
        dev.x,
        dev.y,
        dev.vx,
        dev.vy,
        dev.state,
        dev.risk_group,
        dev.home_zone,
        dev.rep_id,
        dev.state_timer,
        dev.home_id,
        dev.work_zone,
        dev.infection_source,
        dev.infection_flash
    );

    if (!CUDA_CHECK(cudaGetLastError()) || !CUDA_CHECK(cudaStreamSynchronize(g_sim_stream))) {
        cuda_free_simulation(dev);
        return false;
    }

    const int num_pixels = FRAME_WIDTH * FRAME_HEIGHT;
    const int fb_grid    = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_prepare_framebuffer<<<fb_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        reinterpret_cast<uint32_t*>(dev.framebuffer_rgba), num_pixels, false
    );
    CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));

    return true;
}

namespace {

// Orquesta los kernels de renderizado 1080p según modo visual y mosaico
void launch_render_1080p(
    AgentSystemDevice& dev,
    const SimParams& params,
    bool cell_stats_ready
) {
    const int agent_grid = (dev.num_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_pixels = FRAME_WIDTH * FRAME_HEIGHT;
    const int fb_grid    = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t* fb_u32     = reinterpret_cast<uint32_t*>(dev.framebuffer_rgba);

    const bool is_mosaic    = (dev.num_replicas > 1 && params.ensemble_grid_view);
    const bool use_heatmap  = !is_mosaic && (params.focused_replica == 0) &&
                              (params.render_mode == RENDER_HEATMAP || params.render_mode == RENDER_HYBRID);

    // 1. Limpiar o atenuar framebuffer previo (estelas)
    kernel_prepare_framebuffer<<<fb_grid, BLOCK_SIZE, 0, g_sim_stream>>>(fb_u32, num_pixels, params.enable_trails && !is_mosaic);

    // 2. Heatmap térmico bilineal si el modo es HEATMAP o HYBRID
    if (use_heatmap) {
        if (!cell_stats_ready) {
            const int cell_grid = (dev.num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
            CUDA_CHECK(cudaMemsetAsync(g_d_spatial_stats, 0, 2 * sizeof(float), g_sim_stream));
            kernel_cell_infected_stats<<<cell_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
                dev.num_cells,
                dev.cell_start,
                dev.cell_end,
                dev.state,
                g_d_cell_infected,
                g_d_spatial_stats
            );
        }

        dim3 block2d(16, 16);
        dim3 grid2d((FRAME_WIDTH + block2d.x - 1) / block2d.x,
                    (FRAME_HEIGHT + block2d.y - 1) / block2d.y);
        kernel_render_heatmap_1080p<<<grid2d, block2d, 0, g_sim_stream>>>(
            params,
            g_d_cell_infected,
            g_d_spatial_stats,
            fb_u32,
            params.render_mode == RENDER_HYBRID
        );
    }

    // 2b. Zonas urbanas, mosaico e iluminación Día/Noche
    if (is_mosaic || params.enable_urban_layout || params.render_mode == RENDER_AGENTS || params.focused_replica > 0) {
        dim3 block2d(16, 16);
        dim3 grid2d((FRAME_WIDTH + block2d.x - 1) / block2d.x,
                    (FRAME_HEIGHT + block2d.y - 1) / block2d.y);
        kernel_render_urban_zones_1080p<<<grid2d, block2d, 0, g_sim_stream>>>(params, dev.num_replicas, g_sim_day, fb_u32);
    }

    // 3. Agentes con prioridad visual (atomicMax) si el modo es AGENTS o HYBRID (o mosaico)
    if (is_mosaic || params.render_mode == RENDER_AGENTS || params.render_mode == RENDER_HYBRID) {
        CUDA_CHECK(cudaMemsetAsync(g_d_fb_prio, 0, static_cast<size_t>(num_pixels) * sizeof(uint32_t), g_sim_stream));
        kernel_render_agents_prio<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_agents,
            dev.num_replicas,
            params,
            dev.x,
            dev.y,
            dev.state,
            dev.rep_id,
            dev.home_id,
            dev.infection_flash,
            g_d_fb_prio,
            !is_mosaic && (params.render_mode == RENDER_HYBRID) && (params.focused_replica == 0)
        );
        kernel_composite_prio<<<fb_grid, BLOCK_SIZE, 0, g_sim_stream>>>(fb_u32, g_d_fb_prio, num_pixels);
    }
}

} // namespace

void cuda_step_simulation(
    AgentSystemDevice& dev,
    const SimParams& params,
    uint8_t* host_framebuffer_rgba,
    SimStats& out_stats,
    bool compute_stats,
    std::vector<SimStats>* out_replica_stats
) {
    if (dev.num_agents <= 0) return;

    ++g_sim_step;
    const int k_rep       = std::max(1, dev.num_replicas);
    const int total_cells = dev.num_cells * k_rep;
    const int agent_grid  = (dev.num_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int cell_grid   = (dev.num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int all_cell_grid = (total_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Movimiento Lagrangiano con ciclo diurno/urbano, cuarentena focalizada y cálculo de cell_id global
    if (params.enable_urban_layout && params.household_quarantine && g_d_home_quarantine != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(g_d_home_quarantine, 0, static_cast<size_t>(g_max_homes) * sizeof(uint8_t), g_sim_stream));
        kernel_mark_home_quarantine<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_agents,
            g_max_homes,
            dev.state,
            dev.home_id,
            g_d_home_quarantine
        );
    }

    kernel_move_and_hash<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        dev.num_agents,
        dev.num_cells,
        params,
        g_sim_seed,
        g_sim_step,
        g_sim_day,
        g_last_infected_ratio,
        dev.x,
        dev.y,
        dev.vx,
        dev.vy,
        dev.state,
        dev.risk_group,
        dev.home_zone,
        dev.rep_id,
        dev.home_id,
        dev.work_zone,
        g_d_home_quarantine,
        g_d_cell_quarantine,
        dev.cell_id,
        dev.agent_idx
    );

    // 2. Ordenamiento por celda global con CUB acotado a los bits activos
    int sort_end_bit = 1;
    while (sort_end_bit < 32 && (1ULL << sort_end_bit) < static_cast<uint64_t>(total_cells)) {
        ++sort_end_bit;
    }

    cub::DoubleBuffer<uint32_t> d_keys(dev.cell_id, g_d_cell_id_alt);
    cub::DoubleBuffer<uint32_t> d_vals(dev.agent_idx, g_d_agent_idx_alt);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        g_d_sort_temp, g_sort_temp_bytes, d_keys, d_vals,
        dev.num_agents, 0, sort_end_bit, g_sim_stream
    ));
    if (d_keys.Current() != dev.cell_id) {
        std::swap(dev.cell_id, g_d_cell_id_alt);
    }
    if (d_vals.Current() != dev.agent_idx) {
        std::swap(dev.agent_idx, g_d_agent_idx_alt);
    }

    // 2b. Reordenamiento físico en memoria para localidad L1/L2 coalescente
    kernel_reorder_agents<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        dev.num_agents,
        dev.agent_idx,
        dev.x, dev.y, dev.vx, dev.vy, dev.state, dev.risk_group, dev.home_zone, dev.rep_id, dev.state_timer,
        dev.home_id, dev.work_zone, dev.infection_source, dev.infection_flash,
        g_d_x_alt, g_d_y_alt, g_d_vx_alt, g_d_vy_alt, g_d_state_alt, g_d_risk_alt, g_d_home_zone_alt, g_d_rep_id_alt, g_d_timer_alt,
        g_d_home_id_alt, g_d_work_zone_alt, g_d_inf_src_alt, g_d_inf_flash_alt
    );
    std::swap(dev.x, g_d_x_alt);
    std::swap(dev.y, g_d_y_alt);
    std::swap(dev.vx, g_d_vx_alt);
    std::swap(dev.vy, g_d_vy_alt);
    std::swap(dev.state, g_d_state_alt);
    std::swap(dev.risk_group, g_d_risk_alt);
    std::swap(dev.home_zone, g_d_home_zone_alt);
    std::swap(dev.rep_id, g_d_rep_id_alt);
    std::swap(dev.state_timer, g_d_timer_alt);
    std::swap(dev.home_id, g_d_home_id_alt);
    std::swap(dev.work_zone, g_d_work_zone_alt);
    std::swap(dev.infection_source, g_d_inf_src_alt);
    std::swap(dev.infection_flash, g_d_inf_flash_alt);

    // 3. Limpiar e identificar límites de cada celda y contar infectados por celda (todas las réplicas)
    CUDA_CHECK(cudaMemsetAsync(dev.cell_start, 0xFF, static_cast<size_t>(total_cells) * sizeof(int), g_sim_stream));
    CUDA_CHECK(cudaMemsetAsync(dev.cell_end,   0xFF, static_cast<size_t>(total_cells) * sizeof(int), g_sim_stream));
    CUDA_CHECK(cudaMemsetAsync(g_d_spatial_stats, 0, 2 * sizeof(float), g_sim_stream));

    kernel_find_cell_bounds<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        dev.num_agents,
        dev.cell_id,
        dev.cell_start,
        dev.cell_end
    );
    kernel_cell_infected_stats<<<all_cell_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        total_cells,
        dev.cell_start,
        dev.cell_end,
        dev.state,
        g_d_cell_infected,
        g_d_spatial_stats
    );

    // 4. Transiciones SEIRD y contagio en vecindad 3x3 (aislado por réplica)
    kernel_seir_step<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        dev.num_agents,
        dev.num_cells,
        k_rep,
        params,
        g_sim_seed,
        g_sim_step,
        g_last_infected_ratio,
        dev.x,
        dev.y,
        dev.state,
        dev.risk_group,
        dev.rep_id,
        dev.state_timer,
        dev.infection_source,
        dev.infection_flash,
        dev.cell_start,
        dev.cell_end,
        g_d_cell_infected,
        g_d_counts
    );

    // 5. Reducción de estadísticas S/E/I/R/D y Z-score espacial (réplica 0 para mapa, todas para ensamble)
    if (compute_stats) {
        CUDA_CHECK(cudaMemset2DAsync(g_d_counts, COUNTS_STRIDE * sizeof(uint32_t), 0, 5 * sizeof(uint32_t), k_rep, g_sim_stream));
        CUDA_CHECK(cudaMemset2DAsync(g_d_counts + 6, COUNTS_STRIDE * sizeof(uint32_t), 0, 6 * sizeof(uint32_t), k_rep, g_sim_stream));
        CUDA_CHECK(cudaMemsetAsync(g_d_spatial_stats, 0, 2 * sizeof(float), g_sim_stream));
        CUDA_CHECK(cudaMemsetAsync(g_d_hotspot_out, 0, 2 * sizeof(uint32_t), g_sim_stream));

        kernel_count_states<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_agents, k_rep, params, dev.x, dev.y, dev.state, dev.rep_id, dev.infection_source, g_d_counts
        );
        kernel_cell_infected_stats<<<cell_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_cells,
            dev.cell_start,
            dev.cell_end,
            dev.state,
            g_d_cell_infected,
            g_d_spatial_stats
        );
        kernel_cell_zscore<<<cell_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_cells,
            g_d_cell_infected,
            g_d_spatial_stats,
            g_d_hotspot_out,
            g_d_cell_quarantine,
            params.focal_quarantine
        );

        CUDA_CHECK(cudaMemcpyAsync(
            g_h_counts_pinned, g_d_counts, static_cast<size_t>(k_rep) * COUNTS_STRIDE * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, g_sim_stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(g_h_hotspot_pinned, g_d_hotspot_out, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, g_sim_stream));
        CUDA_CHECK(cudaMemset2DAsync(g_d_counts + 5, COUNTS_STRIDE * sizeof(uint32_t), 0, sizeof(uint32_t), k_rep, g_sim_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));

        const float next_day = out_stats.sim_day + params.dt;
        uint32_t hotspot_cells = g_h_hotspot_pinned[0];
        float max_z = 0.0f;
        std::memcpy(&max_z, &g_h_hotspot_pinned[1], sizeof(float));

        if (out_replica_stats != nullptr) {
            out_replica_stats->resize(static_cast<std::size_t>(k_rep));
            for (int r = 0; r < k_rep; ++r) {
                SimStats& rs = (*out_replica_stats)[static_cast<std::size_t>(r)];
                const uint32_t* row = g_h_counts_pinned + r * COUNTS_STRIDE;
                rs.count_s            = row[STATE_S];
                rs.count_e            = row[STATE_E];
                rs.count_i            = row[STATE_I];
                rs.count_r            = row[STATE_R];
                rs.count_d            = row[STATE_D];
                rs.new_exposures      = row[5];
                rs.src_home           = row[6];
                rs.src_work           = row[7];
                rs.src_transit        = row[8];
                rs.src_hospital       = row[9];
                rs.src_leisure        = row[10];
                rs.hospitalized_count = row[11];
                rs.hotspot_cells      = (r == 0) ? hotspot_cells : 0u;
                rs.max_cell_z         = (r == 0) ? max_z : 0.0f;
                rs.sim_day            = next_day;
            }
        }

        out_stats.count_s            = g_h_counts_pinned[STATE_S];
        out_stats.count_e            = g_h_counts_pinned[STATE_E];
        out_stats.count_i            = g_h_counts_pinned[STATE_I];
        out_stats.count_r            = g_h_counts_pinned[STATE_R];
        out_stats.count_d            = g_h_counts_pinned[STATE_D];
        out_stats.new_exposures      = g_h_counts_pinned[5];
        out_stats.src_home           = g_h_counts_pinned[6];
        out_stats.src_work           = g_h_counts_pinned[7];
        out_stats.src_transit        = g_h_counts_pinned[8];
        out_stats.src_hospital       = g_h_counts_pinned[9];
        out_stats.src_leisure        = g_h_counts_pinned[10];
        out_stats.hospitalized_count = g_h_counts_pinned[11];
        out_stats.hotspot_cells      = hotspot_cells;
        out_stats.max_cell_z         = max_z;

        uint64_t total_i = 0;
        for (int r = 0; r < k_rep; ++r) total_i += g_h_counts_pinned[r * COUNTS_STRIDE + STATE_I];
        g_last_infected_ratio = static_cast<float>(total_i) / static_cast<float>(dev.num_agents);
    } else if (params.focal_quarantine) {
        kernel_cell_zscore<<<cell_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_cells,
            g_d_cell_infected,
            g_d_spatial_stats,
            nullptr,
            g_d_cell_quarantine,
            true
        );
    }
    g_sim_day += params.dt;
    out_stats.sim_day += params.dt;

    // 6. Renderizado 1080p en VRAM y copia opcional a CPU
    if (host_framebuffer_rgba != nullptr) {
        launch_render_1080p(dev, params, compute_stats);
        const size_t fb_bytes = static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * 4 * sizeof(uint8_t);
        CUDA_CHECK(cudaMemcpyAsync(host_framebuffer_rgba, dev.framebuffer_rgba, fb_bytes, cudaMemcpyDeviceToHost, g_sim_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));
    }

    CUDA_CHECK(cudaGetLastError());
}

void cuda_render_1080p_device(
    AgentSystemDevice& dev,
    const SimParams& params
) {
    if (dev.num_agents <= 0) return;
    launch_render_1080p(dev, params, false);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_sync_simulation() {
    if (g_sim_stream != nullptr) {
        CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));
    }
}

bool cuda_copy_framebuffer_to_device(const AgentSystemDevice& dev, uint8_t* dst_device_rgba) {
    if (dev.framebuffer_rgba == nullptr || dst_device_rgba == nullptr) return false;
    const size_t fb_bytes = static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * 4 * sizeof(uint8_t);
    bool ok = CUDA_CHECK(cudaMemcpyAsync(
        dst_device_rgba, dev.framebuffer_rgba, fb_bytes, cudaMemcpyDeviceToDevice, g_sim_stream
    ));
    ok &= CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));
    return ok;
}

void cuda_timer_begin() {
    if (g_ev_timer_start != nullptr && g_sim_stream != nullptr) {
        CUDA_CHECK(cudaEventRecord(g_ev_timer_start, g_sim_stream));
    }
}

float cuda_timer_end_ms() {
    if (g_ev_timer_start == nullptr || g_ev_timer_stop == nullptr || g_sim_stream == nullptr) {
        return 0.0f;
    }
    CUDA_CHECK(cudaEventRecord(g_ev_timer_stop, g_sim_stream));
    CUDA_CHECK(cudaEventSynchronize(g_ev_timer_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_ev_timer_start, g_ev_timer_stop));
    return ms;
}

void cuda_get_vram_info(float& used_mb, float& total_mb) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b > 0) {
        total_mb = static_cast<float>(total_b) / (1024.0f * 1024.0f);
        used_mb  = static_cast<float>(total_b - free_b) / (1024.0f * 1024.0f);
    } else {
        used_mb  = 0.0f;
        total_mb = 0.0f;
    }
}

HardwareLimits cuda_query_hardware_limits(size_t reserve_vram_mb) {
    HardwareLimits hw;
    // Ping-pong SoA: 2*(6*float) (48B) + 2*(6*uint8) (12B) + 2*(3*uint32) (24B) + home_quar (1B) + margen CUB (4B)
    hw.bytes_per_agent = 2 * (6 * sizeof(float) + 6 * sizeof(uint8_t) + 3 * sizeof(uint32_t))
                       + sizeof(uint8_t) + 4;

    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b == 0) {
        hw.max_agents_vram   = 1500000;
        hw.safe_agents_limit = 1500000;
        return hw;
    }

    hw.vram_free_mb  = free_b / (1024ULL * 1024ULL);
    hw.vram_total_mb = total_b / (1024ULL * 1024ULL);

    const size_t fixed_fb_bytes = static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * (4 + sizeof(uint32_t));
    const size_t fixed_misc     = 32ULL * 1024ULL * 1024ULL; // malla + contexto CUDA
    const size_t reserve_bytes  = reserve_vram_mb * 1024ULL * 1024ULL;
    const size_t overhead       = fixed_fb_bytes + fixed_misc + reserve_bytes;

    const size_t usable_bytes = (free_b > overhead) ? (free_b - overhead) : (free_b / 2);
    const size_t raw_max      = usable_bytes / hw.bytes_per_agent;

    hw.max_agents_vram = static_cast<int>(std::min<size_t>(raw_max, 2147483647ULL));
    // Tope interactivo seguro en Windows WDDM para evitar TDR (>2s por kernel)
    constexpr int WDDM_SAFE_CAP = 20000000;
    hw.safe_agents_limit = std::max(100000, std::min(hw.max_agents_vram, WDDM_SAFE_CAP));
    return hw;
}

void cuda_adjust_domain_for_agents(SimParams& params, int num_agents) {
    params.cell_size = std::max(4.0f, params.infection_radius + 0.5f);

    // Mantener densidad base <= ~15 agentes/celda respecto al diseño original (1M en 1000x1000)
    if (num_agents > 1000000) {
        const float scale = std::sqrt(static_cast<float>(num_agents) / 1000000.0f);
        const float side  = std::ceil((1000.0f * scale) / params.cell_size) * params.cell_size;
        params.world_width  = side;
        params.world_height = side;
    } else {
        params.world_width  = 1000.0f;
        params.world_height = 1000.0f;
    }

    params.grid_w = std::max(1, static_cast<int>(std::ceil(params.world_width / params.cell_size)));
    params.grid_h = std::max(1, static_cast<int>(std::ceil(params.world_height / params.cell_size)));
    params.cam_cx = 0.5f * params.world_width;
    params.cam_cy = 0.5f * params.world_height;
}

// Encuentra mediante atomicMin el home_id del agente más cercano a (wx, wy)
__global__ void kernel_pick_nearest_home(
    int n,
    float wx,
    float wy,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const uint32_t* __restrict__ home_id,
    unsigned long long* __restrict__ out_best
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dx = x[i] - wx;
    float dy = y[i] - wy;
    uint32_t d2_q = static_cast<uint32_t>(fminf(4e9f, (dx * dx + dy * dy) * 100.0f));
    unsigned long long key = (static_cast<unsigned long long>(d2_q) << 32) | static_cast<unsigned long long>(home_id[i]);
    atomicMin(out_best, key);
}

// Recolecta en VRAM posiciones y destinos de los miembros del hogar target_home_id
__global__ void kernel_collect_family_track(
    int n,
    uint32_t target_home_id,
    SimParams params,
    uint64_t seed,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const uint8_t* __restrict__ state,
    const uint8_t* __restrict__ risk_group,
    const uint8_t* __restrict__ home_zone,
    const uint32_t* __restrict__ home_id,
    const uint8_t* __restrict__ work_zone,
    FamilyTrackInfo* __restrict__ out_info
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (home_id[i] != target_home_id) return;

    int slot = atomicAdd(&out_info->count, 1);
    if (slot == 0) {
        if (params.enable_urban_layout && params.num_urban_zones > 0 && home_zone != nullptr) {
            get_family_home_pos(params, seed, target_home_id, home_zone[i], out_info->home_x, out_info->home_y);
        } else {
            out_info->home_x = x[i];
            out_info->home_y = y[i];
        }
    }
    if (slot < 8) {
        FamilyMemberInfo& m = out_info->members[slot];
        m.x = x[i];
        m.y = y[i];
        m.state = state[i];
        m.risk_group = risk_group[i] & RISK_GROUP_MASK;
        if (params.enable_urban_layout && params.num_urban_zones > 0 && work_zone != nullptr) {
            int wz = min(static_cast<int>(work_zone[i]), params.num_urban_zones - 1);
            m.work_x = params.urban_zones[wz].cx;
            m.work_y = params.urban_zones[wz].cy;
        } else {
            m.work_x = x[i];
            m.work_y = y[i];
        }
    }
}

int cuda_pick_nearest_home_id(const AgentSystemDevice& dev, float world_x, float world_y) {
    if (dev.agents_per_replica <= 0 || dev.home_id == nullptr) return -1;
    if (g_d_pick_min == nullptr) {
        if (!CUDA_CHECK(cudaMalloc(&g_d_pick_min, sizeof(uint64_t)))) return -1;
    }
    uint64_t init_val = 0xFFFFFFFFFFFFFFFFULL;
    CUDA_CHECK(cudaMemcpyAsync(g_d_pick_min, &init_val, sizeof(uint64_t), cudaMemcpyHostToDevice, g_sim_stream));
    int n = dev.agents_per_replica;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_pick_nearest_home<<<grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        n, world_x, world_y, dev.x, dev.y, dev.home_id,
        reinterpret_cast<unsigned long long*>(g_d_pick_min)
    );
    uint64_t best = 0xFFFFFFFFFFFFFFFFULL;
    CUDA_CHECK(cudaMemcpyAsync(&best, g_d_pick_min, sizeof(uint64_t), cudaMemcpyDeviceToHost, g_sim_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));
    if (best == 0xFFFFFFFFFFFFFFFFULL) return -1;
    return static_cast<int>(best & 0xFFFFFFFFULL);
}

bool cuda_get_family_track(const AgentSystemDevice& dev, const SimParams& params, int home_id, FamilyTrackInfo& out_info) {
    out_info = FamilyTrackInfo{};
    out_info.home_id = home_id;
    if (home_id < 0 || dev.agents_per_replica <= 0 || dev.home_id == nullptr) return false;
    if (g_d_family_out == nullptr) {
        if (!CUDA_CHECK(cudaMalloc(&g_d_family_out, sizeof(FamilyTrackInfo)))) return false;
    }
    CUDA_CHECK(cudaMemcpyAsync(g_d_family_out, &out_info, sizeof(FamilyTrackInfo), cudaMemcpyHostToDevice, g_sim_stream));
    int n = dev.agents_per_replica;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_collect_family_track<<<grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        n, static_cast<uint32_t>(home_id), params, g_sim_seed,
        dev.x, dev.y, dev.state, dev.risk_group, dev.home_zone, dev.home_id, dev.work_zone,
        g_d_family_out
    );
    CUDA_CHECK(cudaMemcpyAsync(&out_info, g_d_family_out, sizeof(FamilyTrackInfo), cudaMemcpyDeviceToHost, g_sim_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));
    out_info.count = std::min(out_info.count, 8);
    return out_info.count > 0;
}

void cuda_build_urban_preset(SimParams& params, int preset) {
    params.urban_preset = preset;
    if (preset == PRESET_ORGANIC_POLES) {
        params.enable_urban_layout = false;
        params.num_urban_zones     = 0;
        return;
    }

    params.enable_urban_layout = true;
    const float W = params.world_width;
    const float H = params.world_height;
    constexpr int COLS = 5;
    constexpr int ROWS = 5;
    const float cell_w = W / static_cast<float>(COLS);
    const float cell_h = H / static_cast<float>(ROWS);
    const float hw = cell_w * 0.465f; // Margen de ~7% como avenida/calle entre manzanas
    const float hh = cell_h * 0.465f;
    const float rr = std::min(hw, hh);

    // Retícula 5x5 = 25 distritos teselando todo el plano urbano
    const ZoneType mono_grid[ROWS][COLS] = {
        { ZONE_HOME, ZONE_HOME,     ZONE_HOME,    ZONE_HOME,     ZONE_HOME },
        { ZONE_HOME, ZONE_HOSPITAL, ZONE_SCHOOL,  ZONE_TRANSIT,  ZONE_HOME },
        { ZONE_HOME, ZONE_LEISURE,  ZONE_WORK,    ZONE_WORK,     ZONE_HOME },
        { ZONE_HOME, ZONE_TRANSIT,  ZONE_SCHOOL,  ZONE_HOSPITAL, ZONE_HOME },
        { ZONE_HOME, ZONE_HOME,     ZONE_LEISURE, ZONE_HOME,     ZONE_HOME }
    };
    const ZoneType poly_grid[ROWS][COLS] = {
        { ZONE_HOME,   ZONE_HOME,     ZONE_TRANSIT,  ZONE_HOME,    ZONE_HOME    },
        { ZONE_HOME,   ZONE_WORK,     ZONE_HOSPITAL, ZONE_WORK,    ZONE_HOME    },
        { ZONE_SCHOOL, ZONE_TRANSIT,  ZONE_LEISURE,  ZONE_HOME,    ZONE_SCHOOL  },
        { ZONE_HOME,   ZONE_HOSPITAL, ZONE_WORK,     ZONE_TRANSIT, ZONE_HOME    },
        { ZONE_LEISURE,ZONE_HOME,     ZONE_SCHOOL,   ZONE_HOME,    ZONE_LEISURE }
    };

    int n = 0;
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (n >= MAX_URBAN_ZONES) break;
            ZoneType t = (preset == PRESET_MONOCENTRIC) ? mono_grid[r][c] : poly_grid[r][c];
            float cx = (static_cast<float>(c) + 0.5f) * cell_w;
            float cy = (static_cast<float>(r) + 0.5f) * cell_h;
            float cap = (t == ZONE_HOSPITAL) ? 0.025f : 0.0f;
            params.urban_zones[n++] = UrbanZone{ cx, cy, rr, hw, hh, t, cap };
        }
    }
    params.num_urban_zones = n;
}

bool cuda_run_benchmark(
    int num_agents,
    int warmup_steps,
    int measure_steps,
    const SimParams& params,
    BenchmarkResult& out_res
) {
    AgentSystemDevice dev;
    int init_inf = std::max(10, num_agents / 1000);
    if (!cuda_init_simulation(dev, num_agents, init_inf, params, 42ULL)) {
        return false;
    }

    SimStats stats;
    for (int i = 0; i < warmup_steps; ++i) {
        cuda_step_simulation(dev, params, nullptr, stats, false);
        cuda_render_1080p_device(dev, params);
    }
    CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));

    cudaEvent_t ev_start, ev_mid, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_mid));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double total_sim_ms    = 0.0;
    double total_render_ms = 0.0;

    for (int i = 0; i < measure_steps; ++i) {
        CUDA_CHECK(cudaEventRecord(ev_start, g_sim_stream));
        cuda_step_simulation(dev, params, nullptr, stats, false);
        CUDA_CHECK(cudaEventRecord(ev_mid, g_sim_stream));
        cuda_render_1080p_device(dev, params);
        CUDA_CHECK(cudaEventRecord(ev_stop, g_sim_stream));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float sim_ms = 0.0f, rnd_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&sim_ms, ev_start, ev_mid));
        CUDA_CHECK(cudaEventElapsedTime(&rnd_ms, ev_mid, ev_stop));
        total_sim_ms    += static_cast<double>(sim_ms);
        total_render_ms += static_cast<double>(rnd_ms);
    }

    float vram_used = 0.0f, vram_total = 0.0f;
    cuda_get_vram_info(vram_used, vram_total);

    out_res.num_agents     = num_agents;
    out_res.steps          = measure_steps;
    out_res.avg_step_ms    = static_cast<float>(total_sim_ms / static_cast<double>(measure_steps));
    out_res.avg_render_ms  = static_cast<float>(total_render_ms / static_cast<double>(measure_steps));
    out_res.agents_per_sec = (out_res.avg_step_ms > 0.0f)
        ? (static_cast<double>(num_agents) / (static_cast<double>(out_res.avg_step_ms) / 1000.0))
        : 0.0;
    out_res.vram_used_mb   = vram_used;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_mid);
    cudaEventDestroy(ev_stop);
    cuda_free_simulation(dev);
    return true;
}

void cuda_free_simulation(AgentSystemDevice& dev) {
    cuda_free_ptrs(
        dev.x, dev.y, dev.vx, dev.vy,
        dev.state, dev.risk_group, dev.home_zone, dev.rep_id, dev.state_timer,
        dev.home_id, dev.work_zone, dev.infection_source, dev.infection_flash,
        dev.cell_id, dev.agent_idx, dev.cell_start, dev.cell_end,
        dev.rng_states, dev.framebuffer_rgba,
        g_d_fb_prio, g_d_counts, g_d_cell_infected, g_d_spatial_stats, g_d_hotspot_out,
        g_d_cell_id_alt, g_d_agent_idx_alt, g_d_x_alt, g_d_y_alt, g_d_vx_alt, g_d_vy_alt,
        g_d_state_alt, g_d_risk_alt, g_d_home_zone_alt, g_d_rep_id_alt,
        g_d_home_id_alt, g_d_work_zone_alt, g_d_inf_src_alt, g_d_inf_flash_alt,
        g_d_home_quarantine, g_d_cell_quarantine, g_d_pick_min, g_d_family_out,
        g_d_timer_alt, g_d_sort_temp
    );
    g_max_homes       = 0;
    g_sort_temp_bytes = 0;

    if (g_h_counts_pinned)  { cudaFreeHost(g_h_counts_pinned);  g_h_counts_pinned  = nullptr; }
    if (g_h_hotspot_pinned) { cudaFreeHost(g_h_hotspot_pinned); g_h_hotspot_pinned = nullptr; }
    if (g_ev_timer_start)   { cudaEventDestroy(g_ev_timer_start); g_ev_timer_start = nullptr; }
    if (g_ev_timer_stop)    { cudaEventDestroy(g_ev_timer_stop);  g_ev_timer_stop  = nullptr; }
    if (g_sim_stream)       { cudaStreamDestroy(g_sim_stream);    g_sim_stream     = nullptr; }

    dev.num_agents         = 0;
    dev.agents_per_replica = 0;
    dev.num_replicas       = 1;
    dev.num_cells          = 0;
}

} // namespace seird
