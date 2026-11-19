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

constexpr int BLOCK_SIZE = 256;
constexpr float PI_F = 3.14159265358979323846f;

// Buffers internos en VRAM para conteo de estados, radix sort y telemetría (evita allocs por frame)
uint32_t* g_d_counts        = nullptr; // [0..4]: S,E,I,R,D; [5]: acumulador S->E (new_exposures)
uint32_t* g_d_cell_infected = nullptr; // [num_cells]: infectados por celda
float*    g_d_spatial_stats = nullptr; // [0]: sum, [1]: sum_sq
uint32_t* g_d_hotspot_out   = nullptr; // [0]: hotspot_cells, [1]: max_z_bits (uint32)
uint32_t* g_d_fb_prio       = nullptr; // [1920*1080]: buffer de prioridad visual (atomicMax)
uint32_t* g_d_cell_id_alt   = nullptr;
uint32_t* g_d_agent_idx_alt = nullptr;
float*    g_d_x_alt         = nullptr;
float*    g_d_y_alt         = nullptr;
float*    g_d_vx_alt        = nullptr;
float*    g_d_vy_alt        = nullptr;
uint8_t*  g_d_state_alt       = nullptr;
uint8_t*  g_d_risk_alt        = nullptr;
uint8_t*  g_d_home_zone_alt   = nullptr;
uint8_t*  g_d_rep_id_alt      = nullptr;
uint8_t*  g_d_cell_quarantine = nullptr;
float*    g_d_timer_alt       = nullptr;
void*     g_d_sort_temp       = nullptr;
size_t    g_sort_temp_bytes   = 0;

uint32_t*    g_h_counts_pinned     = nullptr;
uint32_t*    g_h_hotspot_pinned    = nullptr;
cudaStream_t g_sim_stream          = nullptr;
cudaEvent_t  g_ev_timer_start      = nullptr;
cudaEvent_t  g_ev_timer_stop       = nullptr;
uint64_t     g_sim_seed            = 42ULL;
uint64_t     g_sim_step            = 0ULL;
float        g_sim_day             = 0.0f;
float        g_last_infected_ratio = 0.0f;

constexpr int NUM_URBAN_HUBS = 8;
__device__ __constant__ float kHubX[NUM_URBAN_HUBS] = { 0.50f, 0.22f, 0.78f, 0.25f, 0.76f, 0.50f, 0.16f, 0.85f };
__device__ __constant__ float kHubY[NUM_URBAN_HUBS] = { 0.50f, 0.26f, 0.24f, 0.74f, 0.76f, 0.18f, 0.52f, 0.50f };
__device__ __constant__ float kHubR[NUM_URBAN_HUBS] = { 0.24f, 0.20f, 0.20f, 0.20f, 0.20f, 0.17f, 0.16f, 0.16f };

inline bool check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] %s:%d -> %s\n",
                     file, line, cudaGetErrorString(err));
        return false;
    }
    return true;
}

#define CUDA_CHECK(call) check_cuda((call), __FILE__, __LINE__)

__device__ __forceinline__ uint32_t pack_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (static_cast<uint32_t>(a) << 24) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(g) << 8)  |
            static_cast<uint32_t>(r);
}

__device__ __forceinline__ float rand_u01(uint64_t seed, uint64_t step, uint32_t id, uint32_t seq) {
    uint64_t z = seed ^ (step * 0x9E3779B97F4A7C15ULL) ^ (static_cast<uint64_t>(id) << 32) ^ static_cast<uint64_t>(seq);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    uint32_t bits = static_cast<uint32_t>(z ^ (z >> 32));
    return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

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
    float* __restrict__ state_timer
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_agents) return;

    const int rep     = (num_replicas > 1) ? (i / agents_per_replica) : 0;
    const int local_i = (num_replicas > 1) ? (i % agents_per_replica) : i;
    const uint64_t r_seed = seed ^ (static_cast<uint64_t>(rep) * 0x9E3779B97F4A7C15ULL);
    const uint32_t uid    = static_cast<uint32_t>(local_i);

    rep_id[i] = static_cast<uint8_t>(rep);

    float px = rand_u01(r_seed, 0ULL, uid, 0u) * params.world_width;
    float py = rand_u01(r_seed, 0ULL, uid, 1u) * params.world_height;
    px = fminf(fmaxf(px, 0.0f), params.world_width - 1e-4f);
    py = fminf(fmaxf(py, 0.0f), params.world_height - 1e-4f);

    float angle = rand_u01(r_seed, 0ULL, uid, 2u) * 2.0f * PI_F;
    x[i]  = px;
    y[i]  = py;
    vx[i] = cosf(angle) * params.base_speed;
    vy[i] = sinf(angle) * params.base_speed;

    float r_risk = rand_u01(r_seed, 0ULL, uid, 4u);
    uint8_t rg = (r_risk < 0.50f) ? RISK_LOW : ((r_risk < 0.85f) ? RISK_MEDIUM : RISK_HIGH);
    if (params.superspreader_frac > 0.0f && rand_u01(r_seed, 0ULL, uid, 5u) < params.superspreader_frac) {
        rg |= FLAG_SUPERSPREADER;
    }
    risk_group[i] = rg;

    // Asignar hogar al polo más cercano y zona diurna de actividad/trabajo
    float nx = px / fmaxf(params.world_width, 1.0f);
    float ny = py / fmaxf(params.world_height, 1.0f);
    int home_id = 0;
    float best_d2 = 1e9f;
    for (int h = 0; h < NUM_URBAN_HUBS; ++h) {
        float dx = nx - kHubX[h];
        float dy = ny - kHubY[h];
        float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            home_id = h;
        }
    }
    int work_id = home_id;
    if (rand_u01(r_seed, 0ULL, uid, 6u) >= 0.65f) {
        int offset = 1 + min(NUM_URBAN_HUBS - 2, static_cast<int>(rand_u01(r_seed, 0ULL, uid, 7u) * static_cast<float>(NUM_URBAN_HUBS - 1)));
        work_id = (home_id + offset) % NUM_URBAN_HUBS;
    }
    home_zone[i] = static_cast<uint8_t>((work_id << 4) | (home_id & 0x0F));

    if (local_i < initial_infected) {
        state[i]       = STATE_I;
        state_timer[i] = params.infectious_days * (0.5f + 0.5f * rand_u01(r_seed, 0ULL, uid, 3u));
    } else {
        state[i]       = STATE_S;
        state_timer[i] = 0.0f;
    }
}

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
    float* __restrict__ x_out,
    float* __restrict__ y_out,
    float* __restrict__ vx_out,
    float* __restrict__ vy_out,
    uint8_t* __restrict__ state_out,
    uint8_t* __restrict__ risk_out,
    uint8_t* __restrict__ home_out,
    uint8_t* __restrict__ rep_out,
    float* __restrict__ timer_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    int src = static_cast<int>(agent_idx[i]);
    x_out[i]     = x_in[src];
    y_out[i]     = y_in[src];
    vx_out[i]    = vx_in[src];
    vy_out[i]    = vy_in[src];
    state_out[i] = state_in[src];
    risk_out[i]  = risk_in[src];
    home_out[i]  = home_in[src];
    rep_out[i]   = rep_in[src];
    timer_out[i] = timer_in[src];
}

__global__ void kernel_move_and_hash(
    int num_agents,
    int cells_per_replica,
    SimParams params,
    uint64_t seed,
    uint64_t step,
    float sim_day,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ vx,
    float* __restrict__ vy,
    const uint8_t* __restrict__ state,
    const uint8_t* __restrict__ risk_group,
    const uint8_t* __restrict__ home_zone,
    const uint8_t* __restrict__ rep_id,
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

        // Movilidad Lagrangiana orgánica con ciclo diurno (hogar <-> centro de actividad) y radio libre
        if (params.urban_clustering > 0.0f && home_zone != nullptr) {
            float day_phase = sim_day - floorf(sim_day);
            bool is_daytime = (day_phase >= 0.30f && day_phase < 0.75f);
            uint8_t hz = home_zone[i];
            int hub = is_daytime ? ((hz >> 4) & 0x0F) : (hz & 0x0F);
            hub = min(hub, NUM_URBAN_HUBS - 1);

            float hub_x = kHubX[hub] * params.world_width;
            float hub_y = kHubY[hub] * params.world_height;
            float hub_r = kHubR[hub] * fminf(params.world_width, params.world_height);

            // Deriva angular aleatoria para exploración orgánica dentro del barrio
            float angle_j = (rand_u01(r_seed, step, static_cast<uint32_t>(i), 7u) - 0.5f) * 1.4f * params.dt;
            float cj = cosf(angle_j), sj = sinf(angle_j);
            float nvx = pvx * cj - pvy * sj;
            float nvy = pvx * sj + pvy * cj;
            pvx = nvx;
            pvy = nvy;

            // Fuerza restauradora elástica SOLO fuera del radio libre hub_r (evita colapso al centro)
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
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const uint32_t* __restrict__ cell_infected,
    uint32_t* __restrict__ d_counts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

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
        if (d_counts != nullptr) {
            if (num_replicas == 1) {
                unsigned mask = __activemask();
                int lane      = threadIdx.x & 31;
                int leader    = __ffs(mask) - 1;
                if (lane == leader) {
                    atomicAdd(&d_counts[5], static_cast<uint32_t>(__popc(mask)));
                }
            } else {
                atomicAdd(&d_counts[rep * 6 + 5], 1u);
            }
        }
    }
}

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

__global__ void kernel_render_agents_prio(
    int num_agents,
    int num_replicas,
    SimParams params,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const uint8_t* __restrict__ state,
    const uint8_t* __restrict__ rep_id,
    uint32_t* __restrict__ fb_prio,
    bool only_active_ei
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;
    if (num_replicas > 1 && rep_id != nullptr && rep_id[i] != 0u) return;

    uint8_t st = state[i];
    if (only_active_ei && st != STATE_E && st != STATE_I) {
        return;
    }

    float zoom    = fmaxf(1.0f, params.cam_zoom);
    float view_w  = params.world_width  / zoom;
    float view_h  = params.world_height / zoom;
    float view_x0 = params.cam_cx - 0.5f * view_w;
    float view_y0 = params.cam_cy - 0.5f * view_h;

    float ux = (x[i] - view_x0) / view_w;
    float uy = (y[i] - view_y0) / view_h;
    if (ux < 0.0f || ux >= 1.0f || uy < 0.0f || uy >= 1.0f) return;

    int px = static_cast<int>(ux * static_cast<float>(FRAME_WIDTH));
    int py = static_cast<int>(uy * static_cast<float>(FRAME_HEIGHT));
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

__global__ void kernel_composite_prio(uint32_t* __restrict__ fb, const uint32_t* __restrict__ fb_prio, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    uint32_t val = fb_prio[idx];
    if (val != 0u) {
        fb[idx] = 0xFF000000u | (val & 0x00FFFFFFu);
    }
}

__global__ void kernel_count_states(
    int num_agents,
    int num_replicas,
    const uint8_t* __restrict__ state,
    const uint8_t* __restrict__ rep_id,
    uint32_t* __restrict__ global_counts
) {
    __shared__ uint32_t s_counts[MAX_REPLICAS * 5];
    const int total_bins = num_replicas * 5;
    for (int idx = threadIdx.x; idx < total_bins; idx += BLOCK_SIZE) {
        s_counts[idx] = 0u;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_agents) {
        uint8_t st = state[i];
        int rep    = (num_replicas > 1 && rep_id != nullptr) ? static_cast<int>(rep_id[i]) : 0;
        if (st < 5 && rep < num_replicas) {
            atomicAdd(&s_counts[rep * 5 + static_cast<int>(st)], 1u);
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < total_bins; idx += BLOCK_SIZE) {
        uint32_t val = s_counts[idx];
        if (val > 0u) {
            int rep = idx / 5;
            int st  = idx % 5;
            atomicAdd(&global_counts[rep * 6 + st], val);
        }
    }
}

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

    const size_t float_bytes = static_cast<size_t>(total_agents) * sizeof(float);
    const size_t u8_bytes    = static_cast<size_t>(total_agents) * sizeof(uint8_t);
    const size_t u32_bytes   = static_cast<size_t>(total_agents) * sizeof(uint32_t);
    const size_t cell_bytes  = static_cast<size_t>(total_cells) * sizeof(int);
    const size_t fb_bytes    = static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * 4 * sizeof(uint8_t);

    bool ok = true;
    ok &= CUDA_CHECK(cudaStreamCreateWithFlags(&g_sim_stream, cudaStreamNonBlocking));
    ok &= CUDA_CHECK(cudaEventCreate(&g_ev_timer_start));
    ok &= CUDA_CHECK(cudaEventCreate(&g_ev_timer_stop));
    ok &= CUDA_CHECK(cudaMalloc(&dev.x, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.y, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.vx, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.vy, float_bytes));

    ok &= CUDA_CHECK(cudaMalloc(&dev.state, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.risk_group, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.home_zone, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.rep_id, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.state_timer, float_bytes));

    ok &= CUDA_CHECK(cudaMalloc(&dev.cell_id, u32_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.agent_idx, u32_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.cell_start, cell_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.cell_end, cell_bytes));

    ok &= CUDA_CHECK(cudaMalloc(&dev.framebuffer_rgba, fb_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_fb_prio, static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_counts, static_cast<size_t>(k_rep) * 6 * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_cell_infected, static_cast<size_t>(total_cells) * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_spatial_stats, 2 * sizeof(float)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_hotspot_out, 2 * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_cell_id_alt, u32_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_agent_idx_alt, u32_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_x_alt, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_y_alt, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_vx_alt, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_vy_alt, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_state_alt, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_risk_alt, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_home_zone_alt, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_rep_id_alt, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_cell_quarantine, static_cast<size_t>(total_cells) * sizeof(uint8_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_timer_alt, float_bytes));
    ok &= CUDA_CHECK(cudaMallocHost(&g_h_counts_pinned, static_cast<size_t>(k_rep) * 6 * sizeof(uint32_t)));
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
    CUDA_CHECK(cudaMemsetAsync(g_d_counts, 0, static_cast<size_t>(k_rep) * 6 * sizeof(uint32_t), g_sim_stream));

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
        dev.state_timer
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

void launch_render_1080p(
    AgentSystemDevice& dev,
    const SimParams& params,
    bool cell_stats_ready
) {
    const int agent_grid = (dev.num_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_pixels = FRAME_WIDTH * FRAME_HEIGHT;
    const int fb_grid    = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t* fb_u32     = reinterpret_cast<uint32_t*>(dev.framebuffer_rgba);

    // 1. Limpiar o atenuar framebuffer previo (estelas)
    kernel_prepare_framebuffer<<<fb_grid, BLOCK_SIZE, 0, g_sim_stream>>>(fb_u32, num_pixels, params.enable_trails);

    // 2. Heatmap térmico bilineal si el modo es HEATMAP o HYBRID
    if (params.render_mode == RENDER_HEATMAP || params.render_mode == RENDER_HYBRID) {
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

    // 3. Agentes con prioridad visual (atomicMax) si el modo es AGENTS o HYBRID (solo réplica 0)
    if (params.render_mode == RENDER_AGENTS || params.render_mode == RENDER_HYBRID) {
        CUDA_CHECK(cudaMemsetAsync(g_d_fb_prio, 0, static_cast<size_t>(num_pixels) * sizeof(uint32_t), g_sim_stream));
        kernel_render_agents_prio<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_agents,
            dev.num_replicas,
            params,
            dev.x,
            dev.y,
            dev.state,
            dev.rep_id,
            g_d_fb_prio,
            params.render_mode == RENDER_HYBRID
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

    // 1. Movimiento Lagrangiano con ciclo diurno, cuarentena focalizada y cálculo de cell_id global
    kernel_move_and_hash<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
        dev.num_agents,
        dev.num_cells,
        params,
        g_sim_seed,
        g_sim_step,
        g_sim_day,
        dev.x,
        dev.y,
        dev.vx,
        dev.vy,
        dev.state,
        dev.risk_group,
        dev.home_zone,
        dev.rep_id,
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
        g_d_x_alt, g_d_y_alt, g_d_vx_alt, g_d_vy_alt, g_d_state_alt, g_d_risk_alt, g_d_home_zone_alt, g_d_rep_id_alt, g_d_timer_alt
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
        dev.cell_start,
        dev.cell_end,
        g_d_cell_infected,
        g_d_counts
    );

    // 5. Reducción de estadísticas S/E/I/R/D y Z-score espacial (réplica 0 para mapa, todas para ensamble)
    if (compute_stats) {
        CUDA_CHECK(cudaMemset2DAsync(g_d_counts, 6 * sizeof(uint32_t), 0, 5 * sizeof(uint32_t), k_rep, g_sim_stream));
        CUDA_CHECK(cudaMemsetAsync(g_d_spatial_stats, 0, 2 * sizeof(float), g_sim_stream));
        CUDA_CHECK(cudaMemsetAsync(g_d_hotspot_out, 0, 2 * sizeof(uint32_t), g_sim_stream));

        kernel_count_states<<<agent_grid, BLOCK_SIZE, 0, g_sim_stream>>>(
            dev.num_agents, k_rep, dev.state, dev.rep_id, g_d_counts
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
            g_h_counts_pinned, g_d_counts, static_cast<size_t>(k_rep) * 6 * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, g_sim_stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(g_h_hotspot_pinned, g_d_hotspot_out, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, g_sim_stream));
        CUDA_CHECK(cudaMemset2DAsync(g_d_counts + 5, 6 * sizeof(uint32_t), 0, sizeof(uint32_t), k_rep, g_sim_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_sim_stream));

        const float next_day = out_stats.sim_day + params.dt;
        uint32_t hotspot_cells = g_h_hotspot_pinned[0];
        float max_z = 0.0f;
        std::memcpy(&max_z, &g_h_hotspot_pinned[1], sizeof(float));

        if (out_replica_stats != nullptr) {
            out_replica_stats->resize(static_cast<std::size_t>(k_rep));
            for (int r = 0; r < k_rep; ++r) {
                SimStats& rs = (*out_replica_stats)[static_cast<std::size_t>(r)];
                const uint32_t* row = g_h_counts_pinned + r * 6;
                rs.count_s       = row[STATE_S];
                rs.count_e       = row[STATE_E];
                rs.count_i       = row[STATE_I];
                rs.count_r       = row[STATE_R];
                rs.count_d       = row[STATE_D];
                rs.new_exposures = row[5];
                rs.hotspot_cells = (r == 0) ? hotspot_cells : 0u;
                rs.max_cell_z    = (r == 0) ? max_z : 0.0f;
                rs.sim_day       = next_day;
            }
        }

        out_stats.count_s       = g_h_counts_pinned[STATE_S];
        out_stats.count_e       = g_h_counts_pinned[STATE_E];
        out_stats.count_i       = g_h_counts_pinned[STATE_I];
        out_stats.count_r       = g_h_counts_pinned[STATE_R];
        out_stats.count_d       = g_h_counts_pinned[STATE_D];
        out_stats.new_exposures = g_h_counts_pinned[5];
        out_stats.hotspot_cells = hotspot_cells;
        out_stats.max_cell_z    = max_z;

        uint64_t total_i = 0;
        for (int r = 0; r < k_rep; ++r) total_i += g_h_counts_pinned[r * 6 + STATE_I];
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
    // Ping-pong SoA: 2*(x,y,vx,vy,state_timer) (40B) + 2*(state,risk_group,home_zone,rep_id) (8B) + 2*(cell_id,agent_idx) (16B) + margen CUB (4B)
    hw.bytes_per_agent = 2 * (5 * sizeof(float) + 4 * sizeof(uint8_t) + 2 * sizeof(uint32_t))
                       + 4; // margen para buffer temporal de cub::DeviceRadixSort

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
    auto safe_free = [](auto*& ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    };

    safe_free(dev.x);
    safe_free(dev.y);
    safe_free(dev.vx);
    safe_free(dev.vy);
    safe_free(dev.state);
    safe_free(dev.risk_group);
    safe_free(dev.home_zone);
    safe_free(dev.rep_id);
    safe_free(dev.state_timer);
    safe_free(dev.cell_id);
    safe_free(dev.agent_idx);
    safe_free(dev.cell_start);
    safe_free(dev.cell_end);
    safe_free(dev.rng_states);
    safe_free(dev.framebuffer_rgba);
    safe_free(g_d_fb_prio);
    safe_free(g_d_counts);
    safe_free(g_d_cell_infected);
    safe_free(g_d_spatial_stats);
    safe_free(g_d_hotspot_out);
    safe_free(g_d_cell_id_alt);
    safe_free(g_d_agent_idx_alt);
    safe_free(g_d_x_alt);
    safe_free(g_d_y_alt);
    safe_free(g_d_vx_alt);
    safe_free(g_d_vy_alt);
    safe_free(g_d_state_alt);
    safe_free(g_d_risk_alt);
    safe_free(g_d_home_zone_alt);
    safe_free(g_d_rep_id_alt);
    safe_free(g_d_cell_quarantine);
    safe_free(g_d_timer_alt);
    safe_free(g_d_sort_temp);
    g_sort_temp_bytes = 0;

    if (g_h_counts_pinned != nullptr) {
        cudaFreeHost(g_h_counts_pinned);
        g_h_counts_pinned = nullptr;
    }
    if (g_h_hotspot_pinned != nullptr) {
        cudaFreeHost(g_h_hotspot_pinned);
        g_h_hotspot_pinned = nullptr;
    }
    if (g_ev_timer_start != nullptr) {
        cudaEventDestroy(g_ev_timer_start);
        g_ev_timer_start = nullptr;
    }
    if (g_ev_timer_stop != nullptr) {
        cudaEventDestroy(g_ev_timer_stop);
        g_ev_timer_stop = nullptr;
    }
    if (g_sim_stream != nullptr) {
        cudaStreamDestroy(g_sim_stream);
        g_sim_stream = nullptr;
    }

    dev.num_agents         = 0;
    dev.agents_per_replica = 0;
    dev.num_replicas       = 1;
    dev.num_cells          = 0;
}

} // namespace seird
