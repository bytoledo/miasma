#include "cuda_engine.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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
uint32_t* g_d_counts        = nullptr; // [0..4]: S,E,I,R,D
uint32_t* g_d_cell_infected = nullptr; // [num_cells]: infectados por celda
float*    g_d_spatial_stats = nullptr; // [0]: sum, [1]: sum_sq
uint32_t* g_d_hotspot_out   = nullptr; // [0]: hotspot_cells, [1]: max_z_bits (uint32)
uint32_t* g_d_fb_prio       = nullptr; // [1920*1080]: buffer de prioridad visual (atomicMax)
uint32_t* g_d_cell_id_alt   = nullptr;
uint32_t* g_d_agent_idx_alt = nullptr;
void*     g_d_sort_temp     = nullptr;
size_t    g_sort_temp_bytes = 0;

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

__global__ void kernel_init_agents(
    int num_agents,
    int initial_infected,
    SimParams params,
    uint64_t seed,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ vx,
    float* __restrict__ vy,
    uint8_t* __restrict__ state,
    float* __restrict__ state_timer,
    curandState* __restrict__ rng_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    curandState local_rng;
    curand_init(seed, static_cast<unsigned long long>(i), 0ULL, &local_rng);

    float px = curand_uniform(&local_rng) * params.world_width;
    float py = curand_uniform(&local_rng) * params.world_height;
    px = fminf(fmaxf(px, 0.0f), params.world_width - 1e-4f);
    py = fminf(fmaxf(py, 0.0f), params.world_height - 1e-4f);

    float angle = curand_uniform(&local_rng) * 2.0f * PI_F;
    x[i]  = px;
    y[i]  = py;
    vx[i] = cosf(angle) * params.base_speed;
    vy[i] = sinf(angle) * params.base_speed;

    if (i < initial_infected) {
        state[i]       = STATE_I;
        state_timer[i] = params.infectious_days * (0.5f + 0.5f * curand_uniform(&local_rng));
    } else {
        state[i]       = STATE_S;
        state_timer[i] = 0.0f;
    }

    rng_states[i] = local_rng;
}

__global__ void kernel_move_and_hash(
    int num_agents,
    SimParams params,
    float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ vx,
    float* __restrict__ vy,
    const uint8_t* __restrict__ state,
    uint32_t* __restrict__ cell_id,
    uint32_t* __restrict__ agent_idx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    float px = x[i];
    float py = y[i];

    if (state[i] != STATE_D) {
        float pvx = vx[i];
        float pvy = vy[i];
        float scale = params.quarantine_factor * params.dt;

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

    cell_id[i]   = static_cast<uint32_t>(cy * params.grid_w + cx);
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
    SimParams params,
    const float* __restrict__ x,
    const float* __restrict__ y,
    uint8_t* __restrict__ state,
    float* __restrict__ state_timer,
    const uint32_t* __restrict__ agent_idx,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    curandState* __restrict__ rng_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

    // Procesar en orden espacial mejora la localidad de caché L1/L2
    int me = static_cast<int>(agent_idx[i]);
    uint8_t my_state = state[me];

    if (my_state == STATE_R || my_state == STATE_D) {
        return;
    }

    if (my_state == STATE_E) {
        float t = state_timer[me] - params.dt;
        if (t <= 0.0f) {
            curandState local_rng = rng_states[me];
            state[me]       = STATE_I;
            state_timer[me] = params.infectious_days * (0.75f + 0.5f * curand_uniform(&local_rng));
            rng_states[me]  = local_rng;
        } else {
            state_timer[me] = t;
        }
        return;
    }

    if (my_state == STATE_I) {
        float t = state_timer[me] - params.dt;
        if (t <= 0.0f) {
            curandState local_rng = rng_states[me];
            float r = curand_uniform(&local_rng);
            state[me]       = (r < params.fatality_rate) ? STATE_D : STATE_R;
            state_timer[me] = 0.0f;
            rng_states[me]  = local_rng;
        } else {
            state_timer[me] = t;
        }
        return;
    }

    // my_state == STATE_S: evaluar contagio en la celda propia y las 8 vecinas
    float px = x[me];
    float py = y[me];
    int cx = max(0, min(static_cast<int>(px / params.cell_size), params.grid_w - 1));
    int cy = max(0, min(static_cast<int>(py / params.cell_size), params.grid_h - 1));

    float r2_max = params.infection_radius * params.infection_radius;
    float p_step = 1.0f - powf(1.0f - fminf(params.beta, 0.999f), params.dt * 20.0f);
    curandState local_rng = rng_states[me];
    bool infected = false;

    for (int dy = -1; dy <= 1 && !infected; ++dy) {
        int ny = cy + dy;
        if (ny < 0 || ny >= params.grid_h) continue;

        for (int dx = -1; dx <= 1 && !infected; ++dx) {
            int nx = cx + dx;
            if (nx < 0 || nx >= params.grid_w) continue;

            int ncid = ny * params.grid_w + nx;
            int start = cell_start[ncid];
            if (start < 0) continue;
            int end = cell_end[ncid];

            for (int k = start; k < end; ++k) {
                int other = static_cast<int>(agent_idx[k]);
                if (state[other] == STATE_I) {
                    float ddx = x[other] - px;
                    float ddy = y[other] - py;
                    if (ddx * ddx + ddy * ddy <= r2_max) {
                        if (curand_uniform(&local_rng) < p_step) {
                            infected = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    if (infected) {
        state[me]       = STATE_E;
        state_timer[me] = params.incubation_days * (0.75f + 0.5f * curand_uniform(&local_rng));
    }
    rng_states[me] = local_rng;
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
    SimParams params,
    const float* __restrict__ x,
    const float* __restrict__ y,
    const uint8_t* __restrict__ state,
    uint32_t* __restrict__ fb_prio,
    bool only_active_ei
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents) return;

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
    const uint8_t* __restrict__ state,
    uint32_t* __restrict__ global_counts
) {
    __shared__ uint32_t s_counts[5];
    if (threadIdx.x < 5) {
        s_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_agents) {
        uint8_t st = state[i];
        if (st < 5) {
            atomicAdd(&s_counts[st], 1u);
        }
    }
    __syncthreads();

    if (threadIdx.x < 5 && s_counts[threadIdx.x] > 0) {
        atomicAdd(&global_counts[threadIdx.x], s_counts[threadIdx.x]);
    }
}

__global__ void kernel_cell_infected_stats(
    int num_cells,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const uint32_t* __restrict__ agent_idx,
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
                uint8_t st = state[agent_idx[k]];
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
    uint32_t* __restrict__ hotspot_out
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_cells) return;

    float n = static_cast<float>(num_cells);
    float mean = spatial_stats[0] / n;
    float variance = fmaxf(0.0f, (spatial_stats[1] / n) - (mean * mean));
    float stddev = sqrtf(variance);

    if (stddev <= 1e-4f) return;

    float z = (static_cast<float>(cell_infected[cid]) - mean) / stddev;
    if (z > 2.0f) {
        atomicAdd(&hotspot_out[0], 1u);
    }
    if (z > 0.0f) {
        // Para floats positivos, la representación IEEE-754 mantiene el orden como uint32
        atomicMax(&hotspot_out[1], __float_as_uint(z));
    }
}

} // namespace

bool cuda_init_simulation(
    AgentSystemDevice& dev,
    int num_agents,
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

    dev.num_agents = num_agents;
    dev.num_cells  = params.grid_w * params.grid_h;

    const size_t float_bytes = static_cast<size_t>(num_agents) * sizeof(float);
    const size_t u8_bytes    = static_cast<size_t>(num_agents) * sizeof(uint8_t);
    const size_t u32_bytes   = static_cast<size_t>(num_agents) * sizeof(uint32_t);
    const size_t cell_bytes  = static_cast<size_t>(dev.num_cells) * sizeof(int);
    const size_t rng_bytes   = static_cast<size_t>(num_agents) * sizeof(curandState);
    const size_t fb_bytes    = static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * 4 * sizeof(uint8_t);

    bool ok = true;
    ok &= CUDA_CHECK(cudaMalloc(&dev.x, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.y, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.vx, float_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.vy, float_bytes));

    ok &= CUDA_CHECK(cudaMalloc(&dev.state, u8_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.state_timer, float_bytes));

    ok &= CUDA_CHECK(cudaMalloc(&dev.cell_id, u32_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.agent_idx, u32_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.cell_start, cell_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.cell_end, cell_bytes));

    ok &= CUDA_CHECK(cudaMalloc(&dev.rng_states, rng_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&dev.framebuffer_rgba, fb_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_fb_prio, static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_counts, 5 * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_cell_infected, static_cast<size_t>(dev.num_cells) * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_spatial_stats, 2 * sizeof(float)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_hotspot_out, 2 * sizeof(uint32_t)));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_cell_id_alt, u32_bytes));
    ok &= CUDA_CHECK(cudaMalloc(&g_d_agent_idx_alt, u32_bytes));

    if (ok) {
        cub::DoubleBuffer<uint32_t> d_keys(dev.cell_id, g_d_cell_id_alt);
        cub::DoubleBuffer<uint32_t> d_vals(dev.agent_idx, g_d_agent_idx_alt);
        g_sort_temp_bytes = 0;
        ok &= CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, g_sort_temp_bytes, d_keys, d_vals, num_agents));
        if (ok && g_sort_temp_bytes > 0) {
            ok &= CUDA_CHECK(cudaMalloc(&g_d_sort_temp, g_sort_temp_bytes));
        }
    }

    if (!ok) {
        cuda_free_simulation(dev);
        return false;
    }

    int grid = (num_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_init_agents<<<grid, BLOCK_SIZE>>>(
        num_agents,
        initial_infected,
        params,
        seed,
        dev.x,
        dev.y,
        dev.vx,
        dev.vy,
        dev.state,
        dev.state_timer,
        reinterpret_cast<curandState*>(dev.rng_states)
    );

    if (!CUDA_CHECK(cudaGetLastError()) || !CUDA_CHECK(cudaDeviceSynchronize())) {
        cuda_free_simulation(dev);
        return false;
    }

    const int num_pixels = FRAME_WIDTH * FRAME_HEIGHT;
    const int fb_grid    = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_prepare_framebuffer<<<fb_grid, BLOCK_SIZE>>>(
        reinterpret_cast<uint32_t*>(dev.framebuffer_rgba), num_pixels, false
    );
    CUDA_CHECK(cudaDeviceSynchronize());

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
    kernel_prepare_framebuffer<<<fb_grid, BLOCK_SIZE>>>(fb_u32, num_pixels, params.enable_trails);

    // 2. Heatmap térmico bilineal si el modo es HEATMAP o HYBRID
    if (params.render_mode == RENDER_HEATMAP || params.render_mode == RENDER_HYBRID) {
        if (!cell_stats_ready) {
            const int cell_grid = (dev.num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
            CUDA_CHECK(cudaMemset(g_d_spatial_stats, 0, 2 * sizeof(float)));
            kernel_cell_infected_stats<<<cell_grid, BLOCK_SIZE>>>(
                dev.num_cells,
                dev.cell_start,
                dev.cell_end,
                dev.agent_idx,
                dev.state,
                g_d_cell_infected,
                g_d_spatial_stats
            );
        }

        dim3 block2d(16, 16);
        dim3 grid2d((FRAME_WIDTH + block2d.x - 1) / block2d.x,
                    (FRAME_HEIGHT + block2d.y - 1) / block2d.y);
        kernel_render_heatmap_1080p<<<grid2d, block2d>>>(
            params,
            g_d_cell_infected,
            g_d_spatial_stats,
            fb_u32,
            params.render_mode == RENDER_HYBRID
        );
    }

    // 3. Agentes con prioridad visual (atomicMax) si el modo es AGENTS o HYBRID
    if (params.render_mode == RENDER_AGENTS || params.render_mode == RENDER_HYBRID) {
        CUDA_CHECK(cudaMemset(g_d_fb_prio, 0, static_cast<size_t>(num_pixels) * sizeof(uint32_t)));
        kernel_render_agents_prio<<<agent_grid, BLOCK_SIZE>>>(
            dev.num_agents,
            params,
            dev.x,
            dev.y,
            dev.state,
            g_d_fb_prio,
            params.render_mode == RENDER_HYBRID
        );
        kernel_composite_prio<<<fb_grid, BLOCK_SIZE>>>(fb_u32, g_d_fb_prio, num_pixels);
    }
}

} // namespace

void cuda_step_simulation(
    AgentSystemDevice& dev,
    const SimParams& params,
    uint8_t* host_framebuffer_rgba,
    SimStats& out_stats,
    bool compute_stats
) {
    if (dev.num_agents <= 0) return;

    const int agent_grid = (dev.num_agents + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Movimiento, rebotes y cálculo de cell_id
    kernel_move_and_hash<<<agent_grid, BLOCK_SIZE>>>(
        dev.num_agents,
        params,
        dev.x,
        dev.y,
        dev.vx,
        dev.vy,
        dev.state,
        dev.cell_id,
        dev.agent_idx
    );

    // 2. Ordenamiento por celda con CUB sin asignaciones dinámicas por frame
    cub::DoubleBuffer<uint32_t> d_keys(dev.cell_id, g_d_cell_id_alt);
    cub::DoubleBuffer<uint32_t> d_vals(dev.agent_idx, g_d_agent_idx_alt);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        g_d_sort_temp, g_sort_temp_bytes, d_keys, d_vals, dev.num_agents
    ));
    if (d_keys.Current() != dev.cell_id) {
        std::swap(dev.cell_id, g_d_cell_id_alt);
    }
    if (d_vals.Current() != dev.agent_idx) {
        std::swap(dev.agent_idx, g_d_agent_idx_alt);
    }

    // 3. Limpiar e identificar límites de cada celda
    CUDA_CHECK(cudaMemset(dev.cell_start, 0xFF, static_cast<size_t>(dev.num_cells) * sizeof(int)));
    CUDA_CHECK(cudaMemset(dev.cell_end,   0xFF, static_cast<size_t>(dev.num_cells) * sizeof(int)));

    kernel_find_cell_bounds<<<agent_grid, BLOCK_SIZE>>>(
        dev.num_agents,
        dev.cell_id,
        dev.cell_start,
        dev.cell_end
    );

    // 4. Transiciones SEIRD y contagio en vecindad 3x3
    kernel_seir_step<<<agent_grid, BLOCK_SIZE>>>(
        dev.num_agents,
        params,
        dev.x,
        dev.y,
        dev.state,
        dev.state_timer,
        dev.agent_idx,
        dev.cell_start,
        dev.cell_end,
        reinterpret_cast<curandState*>(dev.rng_states)
    );

    // 5. Reducción de estadísticas S/E/I/R/D y Z-score espacial (solo cuando se solicita)
    if (compute_stats) {
        const int cell_grid = (dev.num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CUDA_CHECK(cudaMemset(g_d_counts, 0, 5 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(g_d_spatial_stats, 0, 2 * sizeof(float)));
        CUDA_CHECK(cudaMemset(g_d_hotspot_out, 0, 2 * sizeof(uint32_t)));

        kernel_count_states<<<agent_grid, BLOCK_SIZE>>>(dev.num_agents, dev.state, g_d_counts);
        kernel_cell_infected_stats<<<cell_grid, BLOCK_SIZE>>>(
            dev.num_cells,
            dev.cell_start,
            dev.cell_end,
            dev.agent_idx,
            dev.state,
            g_d_cell_infected,
            g_d_spatial_stats
        );
        kernel_cell_zscore<<<cell_grid, BLOCK_SIZE>>>(
            dev.num_cells,
            g_d_cell_infected,
            g_d_spatial_stats,
            g_d_hotspot_out
        );

        uint32_t h_counts[5] = {0, 0, 0, 0, 0};
        uint32_t h_hotspot[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(h_counts, g_d_counts, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_hotspot, g_d_hotspot_out, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        out_stats.count_s       = h_counts[STATE_S];
        out_stats.count_e       = h_counts[STATE_E];
        out_stats.count_i       = h_counts[STATE_I];
        out_stats.count_r       = h_counts[STATE_R];
        out_stats.count_d       = h_counts[STATE_D];
        out_stats.hotspot_cells = h_hotspot[0];
        std::memcpy(&out_stats.max_cell_z, &h_hotspot[1], sizeof(float));
    }
    out_stats.sim_day += params.dt;

    // 6. Renderizado 1080p en VRAM y copia opcional a CPU (solo modo servidor web)
    if (host_framebuffer_rgba != nullptr) {
        launch_render_1080p(dev, params, compute_stats);
        const size_t fb_bytes = static_cast<size_t>(FRAME_WIDTH) * FRAME_HEIGHT * 4 * sizeof(uint8_t);
        CUDA_CHECK(cudaMemcpy(host_framebuffer_rgba, dev.framebuffer_rgba, fb_bytes, cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_start, ev_mid, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_mid));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double total_sim_ms    = 0.0;
    double total_render_ms = 0.0;

    for (int i = 0; i < measure_steps; ++i) {
        CUDA_CHECK(cudaEventRecord(ev_start, 0));
        cuda_step_simulation(dev, params, nullptr, stats, false);
        CUDA_CHECK(cudaEventRecord(ev_mid, 0));
        cuda_render_1080p_device(dev, params);
        CUDA_CHECK(cudaEventRecord(ev_stop, 0));
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
    safe_free(g_d_sort_temp);
    g_sort_temp_bytes = 0;

    dev.num_agents = 0;
    dev.num_cells  = 0;
}

} // namespace seird
