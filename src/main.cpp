#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commctrl.h>
#endif

#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cpu_engine.hpp"
#include "cuda_engine.cuh"
#include "types.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
// Solicitar a NVIDIA Optimus usar la GPU dedicada para el contexto OpenGL/CUDA
extern "C" {
__declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

namespace {

constexpr int SIDEBAR_W = 330;

enum ControlId : int {
    ID_BTN_PAUSE       = 101,
    ID_BTN_RESET       = 102,
    ID_BTN_QUARANTINE  = 103,
    ID_BTN_VACCINE     = 104,
    ID_BTN_NORMAL      = 105,
    ID_BTN_RENDER_MODE = 106,
    ID_BTN_TRAILS      = 107,
    ID_BTN_EXPORT      = 108,
    ID_BTN_RINGS       = 109,
    ID_TRK_QUARANTINE  = 201,
    ID_TRK_BETA        = 202,
    ID_TRK_FATALITY    = 203,
    ID_TRK_RADIUS      = 204,
    ID_TRK_SUBSTEPS    = 205,
    ID_TRK_CHART_DAYS  = 206,
    ID_TRK_INIT_INF    = 207
};

struct HistoryPoint {
    float day;
    float s, e, i, r, d;
    float ode_i;
    float i_p05, i_p50, i_p95;
};

std::atomic<bool> g_running{true};
std::atomic<bool> g_paused{false}; // Se define desde el Launcher al iniciar
std::atomic<bool> g_reset_requested{false};
std::atomic<bool> g_force_render{false};
std::atomic<bool> g_export_requested{false};

std::mutex g_state_mtx;
int    g_substeps         = 4;    // 4 * 0.05 = 0.20 días/frame
int    g_chart_days       = 30;   // Ventana de días en la gráfica inferior
int    g_initial_infected = 150;  // Casos iniciales al iniciar/reiniciar
float  g_chart_scroll_day = 0.0f; // Día inicial visible al deslizar la gráfica
bool   g_chart_autofollow = true; // Sigue automáticamente el último día
bool   g_dragging_chart   = false;
bool   g_dragging_map     = false;
int    g_drag_last_x      = 0;
int    g_drag_last_y      = 0;
int    g_win_w            = 1440;
int    g_win_h            = 840;
GLuint g_gl_font_base     = 0;

seird::SimParams g_params;

HWND g_gl_hwnd         = nullptr;
HWND g_lbl_stats       = nullptr;
HWND g_btn_pause       = nullptr;
HWND g_btn_render_mode = nullptr;
HWND g_btn_trails      = nullptr;
HWND g_btn_export      = nullptr;
HWND g_btn_rings       = nullptr;
HWND g_lbl_quarantine  = nullptr;
HWND g_lbl_beta        = nullptr;
HWND g_lbl_fatality    = nullptr;
HWND g_lbl_radius      = nullptr;
HWND g_lbl_substeps    = nullptr;
HWND g_lbl_chart_days  = nullptr;
HWND g_lbl_init_inf    = nullptr;
HWND g_trk_quarantine  = nullptr;
HWND g_trk_beta        = nullptr;
HWND g_trk_fatality    = nullptr;
HWND g_trk_radius      = nullptr;
HWND g_trk_substeps    = nullptr;
HWND g_trk_chart_days  = nullptr;
HWND g_trk_init_inf    = nullptr;

void update_slider_labels() {
    char buf[80];
    if (g_lbl_quarantine) {
        std::snprintf(buf, sizeof(buf), "Movilidad / Cuarentena: %.2f", g_params.quarantine_factor);
        SetWindowTextA(g_lbl_quarantine, buf);
    }
    if (g_lbl_beta) {
        std::snprintf(buf, sizeof(buf), "Tasa de Contagio (beta): %.2f", g_params.beta);
        SetWindowTextA(g_lbl_beta, buf);
    }
    if (g_lbl_fatality) {
        std::snprintf(buf, sizeof(buf), "Letalidad (alpha): %.3f", g_params.fatality_rate);
        SetWindowTextA(g_lbl_fatality, buf);
    }
    if (g_lbl_radius) {
        std::snprintf(buf, sizeof(buf), "Radio de Contagio: %.2f", g_params.infection_radius);
        SetWindowTextA(g_lbl_radius, buf);
    }
    if (g_lbl_substeps) {
        float hours_per_frame = static_cast<float>(g_substeps) * g_params.dt * 24.0f;
        std::snprintf(buf, sizeof(buf), "Velocidad: %.1f h/frame (%d pasos)", hours_per_frame, g_substeps);
        SetWindowTextA(g_lbl_substeps, buf);
    }
    if (g_lbl_chart_days) {
        std::snprintf(buf, sizeof(buf), "Ventana de Grafica: %d dias", g_chart_days);
        SetWindowTextA(g_lbl_chart_days, buf);
    }
    if (g_lbl_init_inf) {
        std::snprintf(buf, sizeof(buf), "Casos iniciales (Reiniciar): %d", g_initial_infected);
        SetWindowTextA(g_lbl_init_inf, buf);
    }
}

void update_visual_buttons() {
    if (g_btn_render_mode) {
        const char* label = "Vista: Agentes [M]";
        if (g_params.render_mode == seird::RENDER_HEATMAP) label = "Vista: Heatmap [M]";
        else if (g_params.render_mode == seird::RENDER_HYBRID) label = "Vista: Hibrida [M]";
        SetWindowTextA(g_btn_render_mode, label);
    }
    if (g_btn_trails) {
        SetWindowTextA(g_btn_trails, g_params.enable_trails ? "Estelas: ON [T]" : "Estelas: OFF [T]");
    }
    if (g_btn_rings) {
        SetWindowTextA(g_btn_rings, g_params.show_rings ? "Anillos: ON [A]" : "Anillos: OFF [A]");
    }
}

void sync_sliders_from_params() {
    if (g_trk_quarantine) SendMessageA(g_trk_quarantine, TBM_SETPOS, TRUE, static_cast<LPARAM>(g_params.quarantine_factor * 100.0f));
    if (g_trk_beta)       SendMessageA(g_trk_beta,       TBM_SETPOS, TRUE, static_cast<LPARAM>(g_params.beta * 100.0f));
    if (g_trk_fatality)   SendMessageA(g_trk_fatality,   TBM_SETPOS, TRUE, static_cast<LPARAM>(g_params.fatality_rate * 1000.0f));
    if (g_trk_radius)     SendMessageA(g_trk_radius,     TBM_SETPOS, TRUE, static_cast<LPARAM>(g_params.infection_radius * 10.0f));
    if (g_trk_substeps)   SendMessageA(g_trk_substeps,   TBM_SETPOS, TRUE, static_cast<LPARAM>(g_substeps));
    if (g_trk_chart_days) SendMessageA(g_trk_chart_days, TBM_SETPOS, TRUE, static_cast<LPARAM>(g_chart_days));
    if (g_trk_init_inf)   SendMessageA(g_trk_init_inf,   TBM_SETPOS, TRUE, static_cast<LPARAM>(g_initial_infected));
    update_slider_labels();
}

void layout_controls(int w, int h) {
    int gl_w = std::max(100, w - SIDEBAR_W);
    if (g_gl_hwnd) {
        MoveWindow(g_gl_hwnd, 0, 0, gl_w, h, TRUE);
        glViewport(0, 0, gl_w, h);
    }
    // Reubicar controles del panel derecho
    int x = gl_w + 12;
    int cw = SIDEBAR_W - 24;
    if (g_lbl_stats)      MoveWindow(g_lbl_stats,      x, 10,  cw, 232, TRUE);
    if (g_btn_pause)      MoveWindow(g_btn_pause,      x, 248, (cw - 8) / 2, 28, TRUE);
    HWND btn_reset = GetDlgItem(GetParent(g_btn_pause), ID_BTN_RESET);
    if (btn_reset)        MoveWindow(btn_reset,        x + (cw + 8) / 2, 248, (cw - 8) / 2, 28, TRUE);

    HWND bq = GetDlgItem(GetParent(g_btn_pause), ID_BTN_QUARANTINE);
    HWND bv = GetDlgItem(GetParent(g_btn_pause), ID_BTN_VACCINE);
    HWND bn = GetDlgItem(GetParent(g_btn_pause), ID_BTN_NORMAL);
    if (bq) MoveWindow(bq, x, 282, cw, 25, TRUE);
    if (bv) MoveWindow(bv, x, 310, cw, 25, TRUE);
    if (bn) MoveWindow(bn, x, 338, cw, 25, TRUE);
    if (g_btn_render_mode) MoveWindow(g_btn_render_mode, x, 368, (cw - 8) / 2, 25, TRUE);
    if (g_btn_trails)      MoveWindow(g_btn_trails,      x + (cw + 8) / 2, 368, (cw - 8) / 2, 25, TRUE);
    if (g_btn_export)      MoveWindow(g_btn_export,      x, 398, (cw - 8) / 2, 25, TRUE);
    if (g_btn_rings)       MoveWindow(g_btn_rings,       x + (cw + 8) / 2, 398, (cw - 8) / 2, 25, TRUE);

    int y = 432;
    auto place_slider = [&](HWND lbl, HWND trk) {
        if (lbl) MoveWindow(lbl, x, y, cw, 18, TRUE);
        if (trk) MoveWindow(trk, x, y + 18, cw, 26, TRUE);
        y += 48;
    };
    place_slider(g_lbl_quarantine, g_trk_quarantine);
    place_slider(g_lbl_beta,       g_trk_beta);
    place_slider(g_lbl_fatality,   g_trk_fatality);
    place_slider(g_lbl_radius,     g_trk_radius);
    place_slider(g_lbl_substeps,   g_trk_substeps);
    place_slider(g_lbl_chart_days, g_trk_chart_days);
    place_slider(g_lbl_init_inf,   g_trk_init_inf);
}

void create_sidebar_controls(HWND hwnd, HINSTANCE hinst) {
    HFONT font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
    auto make_ctrl = [&](const char* cls, const char* txt, DWORD style, int id) -> HWND {
        HWND c = CreateWindowExA(0, cls, txt, WS_CHILD | WS_VISIBLE | style,
                                 0, 0, 100, 24, hwnd,
                                 reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                 hinst, nullptr);
        SendMessageA(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return c;
    };

    g_lbl_stats = make_ctrl("STATIC", "Inicializando...", SS_LEFT, 0);
    g_btn_pause = make_ctrl("BUTTON", "Iniciar [SPACE]", BS_PUSHBUTTON, ID_BTN_PAUSE);
    make_ctrl("BUTTON", "Reiniciar [R]", BS_PUSHBUTTON, ID_BTN_RESET);

    make_ctrl("BUTTON", "Cuarentena Estricta [Q]", BS_PUSHBUTTON, ID_BTN_QUARANTINE);
    make_ctrl("BUTTON", "Vacunacion (beta=0.06) [V]", BS_PUSHBUTTON, ID_BTN_VACCINE);
    make_ctrl("BUTTON", "Normalidad (sin restric.)", BS_PUSHBUTTON, ID_BTN_NORMAL);
    g_btn_render_mode = make_ctrl("BUTTON", "Vista: Hibrida [M]", BS_PUSHBUTTON, ID_BTN_RENDER_MODE);
    g_btn_trails      = make_ctrl("BUTTON", "Estelas: ON [T]", BS_PUSHBUTTON, ID_BTN_TRAILS);
    g_btn_export      = make_ctrl("BUTTON", "Exportar CSV [E]", BS_PUSHBUTTON, ID_BTN_EXPORT);
    g_btn_rings       = make_ctrl("BUTTON", "Anillos: ON [A]", BS_PUSHBUTTON, ID_BTN_RINGS);

    g_lbl_quarantine = make_ctrl("STATIC", "Movilidad / Cuarentena", SS_LEFT, 0);
    g_trk_quarantine = make_ctrl(TRACKBAR_CLASSA, "", TBS_HORZ | TBS_NOTICKS, ID_TRK_QUARANTINE);
    SendMessageA(g_trk_quarantine, TBM_SETRANGE, TRUE, MAKELPARAM(5, 150));

    g_lbl_beta = make_ctrl("STATIC", "Tasa de Contagio (beta)", SS_LEFT, 0);
    g_trk_beta = make_ctrl(TRACKBAR_CLASSA, "", TBS_HORZ | TBS_NOTICKS, ID_TRK_BETA);
    SendMessageA(g_trk_beta, TBM_SETRANGE, TRUE, MAKELPARAM(1, 100));

    g_lbl_fatality = make_ctrl("STATIC", "Letalidad (alpha)", SS_LEFT, 0);
    g_trk_fatality = make_ctrl(TRACKBAR_CLASSA, "", TBS_HORZ | TBS_NOTICKS, ID_TRK_FATALITY);
    SendMessageA(g_trk_fatality, TBM_SETRANGE, TRUE, MAKELPARAM(0, 300));

    g_lbl_radius = make_ctrl("STATIC", "Radio de Contagio", SS_LEFT, 0);
    g_trk_radius = make_ctrl(TRACKBAR_CLASSA, "", TBS_HORZ | TBS_NOTICKS, ID_TRK_RADIUS);
    SendMessageA(g_trk_radius, TBM_SETRANGE, TRUE, MAKELPARAM(10, 70));

    g_lbl_substeps = make_ctrl("STATIC", "Velocidad", SS_LEFT, 0);
    g_trk_substeps = make_ctrl(TRACKBAR_CLASSA, "", TBS_HORZ | TBS_NOTICKS, ID_TRK_SUBSTEPS);
    SendMessageA(g_trk_substeps, TBM_SETRANGE, TRUE, MAKELPARAM(1, 48));

    g_lbl_chart_days = make_ctrl("STATIC", "Ventana de Grafica", SS_LEFT, 0);
    g_trk_chart_days = make_ctrl(TRACKBAR_CLASSA, "", TBS_HORZ | TBS_NOTICKS, ID_TRK_CHART_DAYS);
    SendMessageA(g_trk_chart_days, TBM_SETRANGE, TRUE, MAKELPARAM(2, 120));

    g_lbl_init_inf = make_ctrl("STATIC", "Casos iniciales (Reiniciar)", SS_LEFT, 0);
    g_trk_init_inf = make_ctrl(TRACKBAR_CLASSA, "", TBS_HORZ | TBS_NOTICKS, ID_TRK_INIT_INF);
    SendMessageA(g_trk_init_inf, TBM_SETRANGE, TRUE, MAKELPARAM(1, 5000));

    sync_sliders_from_params();
}

void clamp_camera_locked(seird::SimParams& params) {
    params.cam_zoom = std::clamp(params.cam_zoom, 1.0f, 20.0f);
    float vw = params.world_width  / params.cam_zoom;
    float vh = params.world_height / params.cam_zoom;
    float half_w = 0.5f * vw;
    float half_h = 0.5f * vh;
    params.cam_cx = std::clamp(params.cam_cx, half_w, params.world_width  - half_w);
    params.cam_cy = std::clamp(params.cam_cy, half_h, params.world_height - half_h);
}

LRESULT CALLBACK GLCanvasProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_ERASEBKGND:
            return 1;
        case WM_LBUTTONDOWN: {
            SetFocus(GetParent(hwnd));
            int mx = static_cast<short>(LOWORD(lParam));
            int my = static_cast<short>(HIWORD(lParam));
            RECT rc{};
            GetClientRect(hwnd, &rc);
            int chart_top_px = static_cast<int>(static_cast<float>(rc.bottom) * 0.78f);
            float zoom = 1.0f;
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                zoom = g_params.cam_zoom;
            }
            if (my >= chart_top_px) {
                g_dragging_chart = true;
                g_drag_last_x    = mx;
                SetCapture(hwnd);
            } else if (zoom > 1.01f) {
                g_dragging_map = true;
                g_drag_last_x  = mx;
                g_drag_last_y  = my;
                SetCapture(hwnd);
            }
            return 0;
        }
        case WM_RBUTTONDOWN: {
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                g_params.cam_zoom = 1.0f;
                g_params.cam_cx   = 0.5f * g_params.world_width;
                g_params.cam_cy   = 0.5f * g_params.world_height;
            }
            g_force_render.store(true);
            return 0;
        }
        case WM_MOUSEMOVE: {
            if (g_dragging_chart) {
                int mx = static_cast<short>(LOWORD(lParam));
                int dx = mx - g_drag_last_x;
                g_drag_last_x = mx;
                RECT rc{};
                GetClientRect(hwnd, &rc);
                int gl_w = std::max(1, static_cast<int>(rc.right));
                float days_per_px = static_cast<float>(g_chart_days) / static_cast<float>(gl_w);
                g_chart_scroll_day -= static_cast<float>(dx) * days_per_px;
                g_chart_autofollow = false;
            } else if (g_dragging_map) {
                int mx = static_cast<short>(LOWORD(lParam));
                int my = static_cast<short>(HIWORD(lParam));
                int dx = mx - g_drag_last_x;
                int dy = my - g_drag_last_y;
                g_drag_last_x = mx;
                g_drag_last_y = my;
                RECT rc{};
                GetClientRect(hwnd, &rc);
                float gl_w = static_cast<float>(std::max(1L, rc.right));
                float sim_h = static_cast<float>(std::max(1L, rc.bottom)) * 0.78f;
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    float vw = g_params.world_width  / g_params.cam_zoom;
                    float vh = g_params.world_height / g_params.cam_zoom;
                    g_params.cam_cx -= (static_cast<float>(dx) / gl_w)  * vw;
                    g_params.cam_cy -= (static_cast<float>(dy) / sim_h) * vh;
                    clamp_camera_locked(g_params);
                }
                g_force_render.store(true);
            }
            return 0;
        }
        case WM_LBUTTONUP: {
            if (g_dragging_chart || g_dragging_map) {
                g_dragging_chart = false;
                g_dragging_map   = false;
                ReleaseCapture();
            }
            return 0;
        }
        case WM_MOUSEWHEEL: {
            POINT pt{ static_cast<short>(LOWORD(lParam)), static_cast<short>(HIWORD(lParam)) };
            ScreenToClient(hwnd, &pt);
            RECT rc{};
            GetClientRect(hwnd, &rc);
            int chart_top_px = static_cast<int>(static_cast<float>(rc.bottom) * 0.78f);
            int delta = GET_WHEEL_DELTA_WPARAM(wParam);

            if (pt.y >= chart_top_px) {
                float step = std::max(0.5f, static_cast<float>(g_chart_days) * 0.12f);
                if (delta > 0) {
                    g_chart_scroll_day -= step;
                    g_chart_autofollow = false;
                } else if (delta < 0) {
                    g_chart_scroll_day += step;
                }
            } else {
                float gl_w  = static_cast<float>(std::max(1L, rc.right));
                float sim_h = static_cast<float>(std::max(1, chart_top_px));
                float u = std::clamp(static_cast<float>(pt.x) / gl_w,  0.0f, 1.0f);
                float v = std::clamp(static_cast<float>(pt.y) / sim_h, 0.0f, 1.0f);

                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    float old_vw = g_params.world_width  / g_params.cam_zoom;
                    float old_vh = g_params.world_height / g_params.cam_zoom;
                    float wx = (g_params.cam_cx - 0.5f * old_vw) + u * old_vw;
                    float wy = (g_params.cam_cy - 0.5f * old_vh) + v * old_vh;

                    float factor = (delta > 0) ? 1.20f : (1.0f / 1.20f);
                    g_params.cam_zoom = std::clamp(g_params.cam_zoom * factor, 1.0f, 20.0f);

                    float new_vw = g_params.world_width  / g_params.cam_zoom;
                    float new_vh = g_params.world_height / g_params.cam_zoom;
                    g_params.cam_cx = wx - (u - 0.5f) * new_vw;
                    g_params.cam_cy = wy - (v - 0.5f) * new_vh;
                    clamp_camera_locked(g_params);
                }
                g_force_render.store(true);
            }
            return 0;
        }
        default:
            return DefWindowProcA(hwnd, msg, wParam, lParam);
    }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CLOSE:
        case WM_DESTROY:
            g_running.store(false);
            PostQuitMessage(0);
            return 0;
        case WM_SIZE:
            g_win_w = std::max(400, static_cast<int>(LOWORD(lParam)));
            g_win_h = std::max(400, static_cast<int>(HIWORD(lParam)));
            layout_controls(g_win_w, g_win_h);
            return 0;
        case WM_COMMAND: {
            int id = LOWORD(wParam);
            if (id == ID_BTN_PAUSE) {
                bool p = !g_paused.load();
                g_paused.store(p);
                SetWindowTextA(g_btn_pause, p ? "Reanudar [SPACE]" : "Pausar [SPACE]");
            } else if (id == ID_BTN_RESET) {
                g_reset_requested.store(true);
            } else if (id == ID_BTN_QUARANTINE) {
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    g_params.quarantine_factor = 0.10f;
                }
                sync_sliders_from_params();
            } else if (id == ID_BTN_VACCINE) {
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    g_params.beta = 0.06f;
                    g_params.vaccination_rate = 0.02f;
                }
                sync_sliders_from_params();
            } else if (id == ID_BTN_NORMAL) {
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    g_params.quarantine_factor = 1.00f;
                    g_params.beta              = 0.25f;
                    g_params.vaccination_rate  = 0.0f;
                    g_params.focal_quarantine  = false;
                }
                sync_sliders_from_params();
            } else if (id == ID_BTN_RENDER_MODE) {
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    g_params.render_mode = (g_params.render_mode + 1) % 3;
                }
                update_visual_buttons();
                g_force_render.store(true);
            } else if (id == ID_BTN_TRAILS) {
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    g_params.enable_trails = !g_params.enable_trails;
                }
                update_visual_buttons();
                g_force_render.store(true);
            } else if (id == ID_BTN_EXPORT) {
                g_export_requested.store(true);
            } else if (id == ID_BTN_RINGS) {
                {
                    std::lock_guard<std::mutex> lk(g_state_mtx);
                    g_params.show_rings = !g_params.show_rings;
                }
                update_visual_buttons();
                g_force_render.store(true);
            }
            SetFocus(hwnd);
            return 0;
        }
        case WM_HSCROLL: {
            HWND src = reinterpret_cast<HWND>(lParam);
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                if (src == g_trk_quarantine) {
                    g_params.quarantine_factor = static_cast<float>(SendMessageA(src, TBM_GETPOS, 0, 0)) / 100.0f;
                } else if (src == g_trk_beta) {
                    g_params.beta = static_cast<float>(SendMessageA(src, TBM_GETPOS, 0, 0)) / 100.0f;
                } else if (src == g_trk_fatality) {
                    g_params.fatality_rate = static_cast<float>(SendMessageA(src, TBM_GETPOS, 0, 0)) / 1000.0f;
                } else if (src == g_trk_radius) {
                    g_params.infection_radius = static_cast<float>(SendMessageA(src, TBM_GETPOS, 0, 0)) / 10.0f;
                } else if (src == g_trk_substeps) {
                    g_substeps = static_cast<int>(SendMessageA(src, TBM_GETPOS, 0, 0));
                } else if (src == g_trk_chart_days) {
                    g_chart_days = static_cast<int>(SendMessageA(src, TBM_GETPOS, 0, 0));
                } else if (src == g_trk_init_inf) {
                    g_initial_infected = static_cast<int>(SendMessageA(src, TBM_GETPOS, 0, 0));
                }
            }
            update_slider_labels();
            return 0;
        }
        case WM_LBUTTONDOWN: {
            int mx = static_cast<short>(LOWORD(lParam));
            int my = static_cast<short>(HIWORD(lParam));
            int gl_w = std::max(100, g_win_w - SIDEBAR_W);
            int chart_top_px = static_cast<int>(g_win_h * 0.78f);
            if (mx >= 0 && mx < gl_w && my >= chart_top_px) {
                g_dragging_chart = true;
                g_drag_last_x    = mx;
                SetCapture(hwnd);
            }
            return 0;
        }
        case WM_MOUSEMOVE: {
            if (g_dragging_chart) {
                int mx = static_cast<short>(LOWORD(lParam));
                int dx = mx - g_drag_last_x;
                g_drag_last_x = mx;
                int gl_w = std::max(100, g_win_w - SIDEBAR_W);
                float days_per_px = static_cast<float>(g_chart_days) / static_cast<float>(std::max(1, gl_w));
                g_chart_scroll_day -= static_cast<float>(dx) * days_per_px;
                g_chart_autofollow = false;
            }
            return 0;
        }
        case WM_LBUTTONUP: {
            if (g_dragging_chart) {
                g_dragging_chart = false;
                ReleaseCapture();
            }
            return 0;
        }
        case WM_MOUSEWHEEL: {
            int delta = GET_WHEEL_DELTA_WPARAM(wParam);
            float step = std::max(0.5f, static_cast<float>(g_chart_days) * 0.12f);
            if (delta > 0) {
                g_chart_scroll_day -= step;
                g_chart_autofollow = false;
            } else if (delta < 0) {
                g_chart_scroll_day += step;
            }
            return 0;
        }
        case WM_KEYDOWN:
            switch (wParam) {
                case VK_ESCAPE:
                    g_running.store(false);
                    PostQuitMessage(0);
                    break;
                case VK_SPACE: {
                    bool p = !g_paused.load();
                    g_paused.store(p);
                    if (g_btn_pause) SetWindowTextA(g_btn_pause, p ? "Reanudar [SPACE]" : "Pausar [SPACE]");
                    break;
                }
                case 'R':
                    g_reset_requested.store(true);
                    break;
                case 'Q':
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_params.quarantine_factor = (g_params.quarantine_factor < 0.5f) ? 1.00f : 0.10f;
                    }
                    sync_sliders_from_params();
                    break;
                case 'V':
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        if (g_params.beta < 0.12f) {
                            g_params.beta = 0.25f;
                            g_params.vaccination_rate = 0.0f;
                        } else {
                            g_params.beta = 0.06f;
                            g_params.vaccination_rate = 0.02f;
                        }
                    }
                    sync_sliders_from_params();
                    break;
                case 'M':
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_params.render_mode = (g_params.render_mode + 1) % 3;
                    }
                    update_visual_buttons();
                    g_force_render.store(true);
                    break;
                case 'T':
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_params.enable_trails = !g_params.enable_trails;
                    }
                    update_visual_buttons();
                    g_force_render.store(true);
                    break;
                case 'C':
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_params.cam_zoom = 1.0f;
                        g_params.cam_cx   = 0.5f * g_params.world_width;
                        g_params.cam_cy   = 0.5f * g_params.world_height;
                    }
                    g_force_render.store(true);
                    break;
                case 'E':
                    g_export_requested.store(true);
                    break;
                case 'A':
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_params.show_rings = !g_params.show_rings;
                    }
                    update_visual_buttons();
                    g_force_render.store(true);
                    break;
                case 'F':
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_params.focal_quarantine = !g_params.focal_quarantine;
                    }
                    break;
                case VK_UP:
                case VK_OEM_PLUS:
                case VK_ADD:
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_substeps = std::min(48, g_substeps + 1);
                    }
                    sync_sliders_from_params();
                    break;
                case VK_DOWN:
                case VK_OEM_MINUS:
                case VK_SUBTRACT:
                    {
                        std::lock_guard<std::mutex> lk(g_state_mtx);
                        g_substeps = std::max(1, g_substeps - 1);
                    }
                    sync_sliders_from_params();
                    break;
            }
            return 0;
        default:
            return DefWindowProcA(hwnd, msg, wParam, lParam);
    }
}

void draw_gl_text(float x, float y, const char* text) {
    if (g_gl_font_base == 0 || !text) return;
    glRasterPos2f(x, y);
    glPushAttrib(GL_LIST_BIT);
    glListBase(g_gl_font_base - 32);
    glCallLists(static_cast<GLsizei>(std::strlen(text)), GL_UNSIGNED_BYTE, text);
    glPopAttrib();
}

void draw_epidemic_curve_overlay(
    const std::deque<HistoryPoint>& hist,
    float current_day,
    int chart_days,
    int num_replicas
) {
    // Panel acoplado en el 22% inferior (ancho completo, sin tapar la simulación)
    const float x0 = -1.00f, x1 = 1.00f;
    const float y0 = -1.00f, y1 = -0.56f;
    const float px0 = x0 + 0.075f, px1 = x1 - 0.025f;
    const float py0 = y0 + 0.075f, py1 = y1 - 0.060f;

    // Fondo sólido oscuro
    glColor3f(0.035f, 0.050f, 0.085f);
    glBegin(GL_QUADS);
        glVertex2f(x0, y0);
        glVertex2f(x1, y0);
        glVertex2f(x1, y1);
        glVertex2f(x0, y1);
    glEnd();

    // Línea separadora superior
    glColor3f(0.25f, 0.35f, 0.50f);
    glLineWidth(1.5f);
    glBegin(GL_LINES);
        glVertex2f(x0, y1);
        glVertex2f(x1, y1);
    glEnd();

    // Borde del área de gráfico
    glColor3f(0.22f, 0.30f, 0.42f);
    glLineWidth(1.0f);
    glBegin(GL_LINE_LOOP);
        glVertex2f(px0, py0);
        glVertex2f(px1, py0);
        glVertex2f(px1, py1);
        glVertex2f(px0, py1);
    glEnd();

    float span = static_cast<float>(std::max(2, chart_days));
    float max_scroll = std::max(0.0f, current_day - span);
    if (g_chart_autofollow) {
        g_chart_scroll_day = max_scroll;
    } else {
        g_chart_scroll_day = std::clamp(g_chart_scroll_day, 0.0f, max_scroll);
        if (max_scroll > 0.0f && g_chart_scroll_day >= max_scroll - 0.05f) {
            g_chart_autofollow = true;
        }
    }

    float t_min = g_chart_scroll_day;
    float t_max = t_min + span;

    int tick_step = 1;
    if (chart_days > 60) tick_step = 10;
    else if (chart_days > 24) tick_step = 5;
    else if (chart_days > 12) tick_step = 2;

    // Líneas verticales por segmento de días y línea de 50%
    int first_tick = static_cast<int>(std::ceil(t_min / static_cast<float>(tick_step))) * tick_step;
    glColor3f(0.14f, 0.20f, 0.28f);
    glBegin(GL_LINES);
    glVertex2f(px0, 0.5f * (py0 + py1));
    glVertex2f(px1, 0.5f * (py0 + py1));
    for (int d = first_tick; static_cast<float>(d) <= t_max + 1e-4f; d += tick_step) {
        float u = (static_cast<float>(d) - t_min) / span;
        float vx = px0 + u * (px1 - px0);
        glVertex2f(vx, py0);
        glVertex2f(vx, py1);
    }
    glEnd();

    // Etiquetas de ejes y leyenda
    glColor3f(0.75f, 0.82f, 0.92f);
    draw_gl_text(x0 + 0.012f, py0, "0%");
    draw_gl_text(x0 + 0.010f, 0.5f * (py0 + py1) - 0.01f, "50%");
    draw_gl_text(x0 + 0.006f, py1 - 0.02f, "100%");

    for (int d = first_tick; static_cast<float>(d) <= t_max + 1e-4f; d += tick_step) {
        float u = (static_cast<float>(d) - t_min) / span;
        if (u > 0.97f) continue;
        float vx = px0 + u * (px1 - px0);
        char lbl[16];
        std::snprintf(lbl, sizeof(lbl), "%dd", d);
        draw_gl_text(vx - 0.012f, y0 + 0.022f, lbl);
    }

    char header[180];
    std::snprintf(
        header, sizeof(header),
        "Curva SEIRD [%.1fd - %.1fd]  (%s | Punteada: I_ODE%s)",
        t_min, t_max,
        g_chart_autofollow ? "Auto-seguir | Arrastra o rueda" : "Deslizado manual | Rueda abajo",
        (num_replicas > 1) ? " | Banda P5-P95" : ""
    );
    draw_gl_text(px0, y1 - 0.042f, header);

    if (hist.size() >= 2) {
        if (num_replicas > 1) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glColor4f(0.94f, 0.27f, 0.27f, 0.24f);
            glBegin(GL_QUAD_STRIP);
            for (const auto& pt : hist) {
                if (pt.day < t_min - 0.5f || pt.day > t_max + 0.5f) continue;
                float u  = std::clamp((pt.day - t_min) / span, 0.0f, 1.0f);
                float vx = px0 + u * (px1 - px0);
                float y_lo = py0 + std::clamp(pt.i_p05, 0.0f, 1.0f) * (py1 - py0);
                float y_hi = py0 + std::clamp(pt.i_p95, 0.0f, 1.0f) * (py1 - py0);
                glVertex2f(vx, y_lo);
                glVertex2f(vx, y_hi);
            }
            glEnd();
            glDisable(GL_BLEND);
        }

        auto draw_series = [&](auto getter, float r, float g, float b, float lw = 2.0f) {
            glColor3f(r, g, b);
            glLineWidth(lw);
            glBegin(GL_LINE_STRIP);
            for (const auto& pt : hist) {
                if (pt.day < t_min - 0.5f || pt.day > t_max + 0.5f) continue;
                float u = std::clamp((pt.day - t_min) / span, 0.0f, 1.0f);
                float vx = px0 + u * (px1 - px0);
                float vy = py0 + std::clamp(getter(pt), 0.0f, 1.0f) * (py1 - py0);
                glVertex2f(vx, vy);
            }
            glEnd();
        };

        draw_series([](const HistoryPoint& p) { return p.s; }, 0.22f, 0.74f, 0.97f);
        draw_series([](const HistoryPoint& p) { return p.e; }, 0.98f, 0.80f, 0.08f);
        draw_series([](const HistoryPoint& p) { return p.i; }, 0.94f, 0.27f, 0.27f);
        if (num_replicas > 1) {
            draw_series([](const HistoryPoint& p) { return p.i_p50; }, 1.00f, 0.55f, 0.35f, 2.0f);
        }
        draw_series([](const HistoryPoint& p) { return p.r; }, 0.13f, 0.77f, 0.37f);
        draw_series([](const HistoryPoint& p) { return p.d; }, 0.39f, 0.45f, 0.55f);

        glEnable(GL_LINE_STIPPLE);
        glLineStipple(2, 0xAAAA);
        draw_series([](const HistoryPoint& p) { return p.ode_i; }, 0.95f, 0.72f, 0.90f, 1.5f);
        glDisable(GL_LINE_STIPPLE);
    }
}

void export_simulation_report(
    HWND owner,
    const std::vector<seird::TelemetryMetrics>& hist,
    const std::vector<seird::EnsembleMetrics>& ens_hist,
    const seird::EnsembleMetrics& current_ens,
    const seird::SimStats& current_stats,
    const seird::SimParams& params,
    uint64_t seed,
    int num_agents,
    int initial_infected,
    double last_gpu_ms
) {
    std::error_code ec;
    std::filesystem::create_directories("reports", ec);

    std::time_t now_t = std::time(nullptr);
    std::tm tm_now{};
    localtime_s(&tm_now, &now_t);

    char csv_name[128];
    char json_name[128];
    std::strftime(csv_name,  sizeof(csv_name),  "reports/miasma_reporte_%Y%m%d_%H%M%S.csv",  &tm_now);
    std::strftime(json_name, sizeof(json_name), "reports/miasma_reporte_%Y%m%d_%H%M%S.json", &tm_now);

    std::ofstream out(csv_name);
    if (!out.is_open()) {
        MessageBoxA(owner, "No se pudo crear el archivo CSV en reports/.", "Error al exportar", MB_OK | MB_ICONERROR);
        return;
    }

    out << "dia,susceptibles,expuestos,infectados,recuperados,fallecidos,rt,rt_p05,rt_p95,prob_rt_gt_1,r0_ngm,nuevos_casos_dia,cfr_pct,focos_brote,pico_z,i_p05_pct,i_p50_pct,i_p95_pct,extincion_pct\n";

    uint32_t peak_i = current_stats.count_i;
    float    peak_i_day = current_stats.sim_day;
    uint32_t peak_e = current_stats.count_e;
    float    peak_rt = 0.0f;
    float    peak_rt_day = 0.0f;

    for (std::size_t idx = 0; idx < hist.size(); ++idx) {
        const auto& m = hist[idx];
        const float i05 = (idx < ens_hist.size()) ? (ens_hist[idx].i_p05 * 100.0f) : 0.0f;
        const float i50 = (idx < ens_hist.size()) ? (ens_hist[idx].i_p50 * 100.0f) : 0.0f;
        const float i95 = (idx < ens_hist.size()) ? (ens_hist[idx].i_p95 * 100.0f) : 0.0f;
        const float ext = (idx < ens_hist.size()) ? ens_hist[idx].extinction_pct : 0.0f;

        out << m.stats.sim_day << ","
            << m.stats.count_s << ","
            << m.stats.count_e << ","
            << m.stats.count_i << ","
            << m.stats.count_r << ","
            << m.stats.count_d << ","
            << m.rt << ","
            << m.rt_p05 << ","
            << m.rt_p95 << ","
            << m.prob_rt_gt_1 << ","
            << m.r0_ngm << ","
            << m.new_cases_rate << ","
            << (m.cfr * 100.0f) << ","
            << m.stats.hotspot_cells << ","
            << m.stats.max_cell_z << ","
            << i05 << ","
            << i50 << ","
            << i95 << ","
            << ext << "\n";

        if (m.stats.count_i > peak_i) {
            peak_i = m.stats.count_i;
            peak_i_day = m.stats.sim_day;
        }
        if (m.stats.count_e > peak_e) {
            peak_e = m.stats.count_e;
        }
        if (m.rt > peak_rt) {
            peak_rt = m.rt;
            peak_rt_day = m.stats.sim_day;
        }
    }
    out.close();

    uint32_t total_affected = static_cast<uint32_t>(std::max<int64_t>(0, static_cast<int64_t>(num_agents) - static_cast<int64_t>(current_stats.count_s)));
    double   pct_affected   = (num_agents > 0) ? (100.0 * static_cast<double>(total_affected) / static_cast<double>(num_agents)) : 0.0;
    double   pct_peak_i     = (num_agents > 0) ? (100.0 * static_cast<double>(peak_i) / static_cast<double>(num_agents)) : 0.0;
    uint64_t resolved       = static_cast<uint64_t>(current_stats.count_r) + static_cast<uint64_t>(current_stats.count_d);
    double   cfr_pct        = (resolved > 0) ? (100.0 * static_cast<double>(current_stats.count_d) / static_cast<double>(resolved)) : 0.0;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::snprintf(prop.name, sizeof(prop.name), "CUDA GPU");
    }

    std::ofstream jout(json_name);
    if (jout.is_open()) {
        jout << "{\n"
             << "  \"gpu_name\": \"" << prop.name << "\",\n"
             << "  \"gpu_sm\": \"" << prop.major << "." << prop.minor << "\",\n"
             << "  \"seed\": " << seed << ",\n"
             << "  \"num_agents\": " << num_agents << ",\n"
             << "  \"initial_infected\": " << initial_infected << ",\n"
             << "  \"csv_file\": \"" << csv_name << "\",\n"
             << "  \"params\": {\n"
             << "    \"world_width\": " << params.world_width << ",\n"
             << "    \"world_height\": " << params.world_height << ",\n"
             << "    \"grid_w\": " << params.grid_w << ",\n"
             << "    \"grid_h\": " << params.grid_h << ",\n"
             << "    \"cell_size\": " << params.cell_size << ",\n"
             << "    \"infection_radius\": " << params.infection_radius << ",\n"
             << "    \"beta\": " << params.beta << ",\n"
             << "    \"incubation_days\": " << params.incubation_days << ",\n"
             << "    \"infectious_days\": " << params.infectious_days << ",\n"
             << "    \"fatality_rate\": " << params.fatality_rate << ",\n"
             << "    \"dt\": " << params.dt << ",\n"
             << "    \"base_speed\": " << params.base_speed << ",\n"
             << "    \"quarantine_factor\": " << params.quarantine_factor << ",\n"
             << "    \"urban_clustering\": " << params.urban_clustering << ",\n"
             << "    \"focal_quarantine\": " << (params.focal_quarantine ? "true" : "false") << ",\n"
             << "    \"hospital_capacity_pct\": " << params.hospital_capacity_pct << ",\n"
             << "    \"vaccination_rate\": " << params.vaccination_rate << ",\n"
             << "    \"immunity_days\": " << params.immunity_days << ",\n"
             << "    \"superspreader_frac\": " << params.superspreader_frac << ",\n"
             << "    \"superspreader_mult\": " << params.superspreader_mult << ",\n"
             << "    \"num_replicas\": " << params.num_replicas << "\n"
             << "  },\n"
             << "  \"ensemble\": {\n"
             << "    \"num_replicas\": " << params.num_replicas << ",\n"
             << "    \"i_p05_pct\": " << (current_ens.i_p05 * 100.0f) << ",\n"
             << "    \"i_p50_pct\": " << (current_ens.i_p50 * 100.0f) << ",\n"
             << "    \"i_p95_pct\": " << (current_ens.i_p95 * 100.0f) << ",\n"
             << "    \"prob_rt_gt_1\": " << current_ens.prob_rt_gt_1 << ",\n"
             << "    \"extinction_pct\": " << current_ens.extinction_pct << "\n"
             << "  },\n"
             << "  \"summary\": {\n"
             << "    \"final_day\": " << current_stats.sim_day << ",\n"
             << "    \"samples\": " << hist.size() << ",\n"
             << "    \"count_s\": " << current_stats.count_s << ",\n"
             << "    \"count_e\": " << current_stats.count_e << ",\n"
             << "    \"count_i\": " << current_stats.count_i << ",\n"
             << "    \"count_r\": " << current_stats.count_r << ",\n"
             << "    \"count_d\": " << current_stats.count_d << ",\n"
             << "    \"attack_rate_pct\": " << pct_affected << ",\n"
             << "    \"peak_i\": " << peak_i << ",\n"
             << "    \"peak_i_day\": " << peak_i_day << ",\n"
             << "    \"peak_e\": " << peak_e << ",\n"
             << "    \"peak_rt\": " << peak_rt << ",\n"
             << "    \"peak_rt_day\": " << peak_rt_day << ",\n"
             << "    \"cfr_pct\": " << cfr_pct << ",\n"
             << "    \"last_gpu_ms\": " << last_gpu_ms << "\n"
             << "  }\n"
             << "}\n";
        jout.close();
    }

    char summary[896];
    std::snprintf(
        summary, sizeof(summary),
        "RESUMEN NUMERICO DE LA SIMULACION\n"
        "==========================================\n"
        "Poblacion total: %d personas (seed=%llu)\n"
        "Dia alcanzado: %.2f dias (%zu muestras)\n\n"
        "IMPACTO EPIDEMIOLOGICO:\n"
        "- Contagiados acumulados: %u (%.2f%%)\n"
        "- Pico de infectados activos: %u (%.2f%%) en dia %.1f\n"
        "- Pico de expuestos: %u\n"
        "- Rt maximo observado: %.2f (dia %.1f)\n"
        "- Recuperados finales: %u\n"
        "- Fallecidos totales: %u (CFR: %.2f%%)\n\n"
        "RENDIMIENTO GPU:\n"
        "- Ultimo paso GPU: %.2f ms/frame\n\n"
        "Archivos guardados:\n- %s\n- %s",
        num_agents, static_cast<unsigned long long>(seed),
        current_stats.sim_day, hist.size(),
        total_affected, pct_affected,
        peak_i, pct_peak_i, peak_i_day,
        peak_e,
        peak_rt, peak_rt_day,
        current_stats.count_r,
        current_stats.count_d, cfr_pct,
        last_gpu_ms,
        csv_name, json_name
    );
    MessageBoxA(owner, summary, "Reporte Exportado (CSV + JSON)", MB_OK | MB_ICONINFORMATION);
}

struct DiseasePreset {
    const char* name;
    const char* info;
    float beta;
    float radius;
    float incubation_days;
    float infectious_days;
    float fatality_rate;
};

constexpr DiseasePreset kDiseasePresets[] = {
    {
        "Personalizado",
        "Configura manualmente todos los parametros de transmision y gravedad.",
        0.25f, 3.5f, 5.0f, 10.0f, 0.020f
    },
    {
        "COVID-19 Original (R0 ~2.8)",
        "SARS-CoV-2 ancestral: incubacion ~4.5d, infeccioso ~9d, letalidad ~1.5%.",
        0.22f, 3.5f, 4.5f, 9.0f, 0.015f
    },
    {
        "COVID-19 Omicron (R0 ~9.0)",
        "Variante de muy alta contagiosidad: incubacion corta ~3d, infeccioso ~7d, CFR ~0.3%.",
        0.55f, 4.5f, 3.0f, 7.0f, 0.003f
    },
    {
        "Influenza Estacional (R0 ~1.5)",
        "Gripe comun: propagacion moderada, incubacion ~2d, infeccioso ~5d, CFR ~0.1%.",
        0.12f, 3.0f, 2.0f, 5.0f, 0.001f
    },
    {
        "Gripe Espanola 1918 (R0 ~2.5)",
        "Pandemia H1N1 historica: evolucion rapida (incubacion ~2d, infeccioso ~4d), CFR ~2.5%.",
        0.25f, 3.5f, 2.0f, 4.0f, 0.025f
    },
    {
        "Sarampion / Measles (R0 ~15.0)",
        "Contagiosidad extrema por via aerea: incubacion ~12d, infeccioso ~8d, CFR ~0.2%.",
        0.80f, 5.5f, 12.0f, 8.0f, 0.002f
    },
    {
        "Ebola (R0 ~2.0 | CFR 50%)",
        "Contacto directo cercano (radio bajo): incubacion ~9d, infeccioso ~10d, letalidad ~50%.",
        0.18f, 2.5f, 9.0f, 10.0f, 0.500f
    },
    {
        "Viruela / Smallpox (R0 ~5.0 | CFR 30%)",
        "Enfermedad historica erradicada: incubacion ~12d, infeccioso ~21d, letalidad ~30%.",
        0.38f, 4.0f, 12.0f, 21.0f, 0.300f
    }
};

struct LauncherState {
    bool  done       = false;
    bool  launch     = false;
    int   agents     = 500000;
    int   max_agents = 1500000;
    int   infected   = 150;
    int   substeps   = 4;
    int   chart_days = 30;
    bool  start_paused = false;
    seird::SimParams params;

    HWND cmb_preset   = nullptr;
    HWND lbl_info     = nullptr;
    HWND edt_agents   = nullptr;
    HWND edt_infected = nullptr;
    HWND edt_speed    = nullptr;
    HWND edt_quar     = nullptr;
    HWND edt_beta     = nullptr;
    HWND edt_radius   = nullptr;
    HWND edt_incub    = nullptr;
    HWND edt_inf_days = nullptr;
    HWND edt_cfr      = nullptr;
    HWND edt_substeps = nullptr;
    HWND edt_chart    = nullptr;
    HWND cmb_view     = nullptr;
    HWND cmb_replicas = nullptr;
    HWND chk_trails   = nullptr;
    HWND chk_paused   = nullptr;
};

LauncherState* g_launcher_state = nullptr;

float read_edit_float(HWND h, float fallback, float min_v, float max_v) {
    char buf[64]{};
    GetWindowTextA(h, buf, sizeof(buf));
    try {
        float v = std::stof(buf);
        return std::clamp(v, min_v, max_v);
    } catch (...) {
        return fallback;
    }
}

int read_edit_int(HWND h, int fallback, int min_v, int max_v) {
    char buf[64]{};
    GetWindowTextA(h, buf, sizeof(buf));
    try {
        int v = std::stoi(buf);
        return std::clamp(v, min_v, max_v);
    } catch (...) {
        return fallback;
    }
}

void apply_preset_to_launcher(LauncherState* st, int idx) {
    if (!st || idx < 0 || idx >= static_cast<int>(std::size(kDiseasePresets))) return;
    const DiseasePreset& p = kDiseasePresets[idx];
    SetWindowTextA(st->lbl_info, p.info);
    if (idx == 0) return; // Personalizado no sobreescribe valores actuales

    auto set_f = [](HWND h, float v, const char* fmt) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), fmt, v);
        SetWindowTextA(h, buf);
    };
    set_f(st->edt_beta,     p.beta,            "%.2f");
    set_f(st->edt_radius,   p.radius,          "%.2f");
    set_f(st->edt_incub,    p.incubation_days, "%.1f");
    set_f(st->edt_inf_days, p.infectious_days, "%.1f");
    set_f(st->edt_cfr,      p.fatality_rate,   "%.3f");
}

LRESULT CALLBACK LauncherProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    LauncherState* st = g_launcher_state;
    switch (msg) {
        case WM_CLOSE:
            if (st) { st->done = true; st->launch = false; }
            DestroyWindow(hwnd);
            return 0;
        case WM_COMMAND: {
            if (!st) break;
            int id   = LOWORD(wParam);
            int code = HIWORD(wParam);
            if (id == 1001 && code == CBN_SELCHANGE) {
                int sel = static_cast<int>(SendMessageA(st->cmb_preset, CB_GETCURSEL, 0, 0));
                apply_preset_to_launcher(st, sel);
            } else if (id == 1020) { // Lanzar simulacion
                st->agents     = read_edit_int(st->edt_agents, 500000, 1000, st->max_agents);
                st->infected   = read_edit_int(st->edt_infected, 150, 1, st->agents);
                st->params.base_speed        = read_edit_float(st->edt_speed, 15.0f, 1.0f, 100.0f);
                st->params.quarantine_factor = read_edit_float(st->edt_quar, 1.0f, 0.05f, 1.50f);
                st->params.beta              = read_edit_float(st->edt_beta, 0.25f, 0.01f, 1.00f);
                st->params.infection_radius  = read_edit_float(st->edt_radius, 3.5f, 1.0f, 8.0f);
                st->params.incubation_days   = read_edit_float(st->edt_incub, 5.0f, 0.5f, 30.0f);
                st->params.infectious_days   = read_edit_float(st->edt_inf_days, 10.0f, 0.5f, 45.0f);
                st->params.fatality_rate     = read_edit_float(st->edt_cfr, 0.02f, 0.0f, 1.0f);
                st->substeps   = read_edit_int(st->edt_substeps, 4, 1, 48);
                st->chart_days = read_edit_int(st->edt_chart, 30, 2, 120);
                st->params.render_mode   = static_cast<int>(SendMessageA(st->cmb_view, CB_GETCURSEL, 0, 0));
                const int rep_sel        = static_cast<int>(SendMessageA(st->cmb_replicas, CB_GETCURSEL, 0, 0));
                constexpr int kRepChoices[] = { 1, 8, 16, 32 };
                const int k_rep = (rep_sel >= 0 && rep_sel < 4) ? kRepChoices[rep_sel] : 1;
                st->params.num_replicas  = k_rep;
                st->agents               = std::clamp(st->agents, 1000, std::max(1000, st->max_agents / k_rep));
                st->infected             = std::clamp(st->infected, 1, st->agents);
                st->params.enable_trails = (SendMessageA(st->chk_trails, BM_GETCHECK, 0, 0) == BST_CHECKED);
                st->start_paused         = (SendMessageA(st->chk_paused, BM_GETCHECK, 0, 0) == BST_CHECKED);

                // Garantizar cell_size >= infection_radius y escalar dominio según población
                seird::cuda_adjust_domain_for_agents(st->params, st->agents);

                st->launch = true;
                st->done   = true;
                DestroyWindow(hwnd);
            } else if (id == 1021) { // Benchmark GPU
                int n_bench = read_edit_int(st->edt_agents, 500000, 10000, st->max_agents);
                seird::SimParams bench_params = st->params;
                seird::cuda_adjust_domain_for_agents(bench_params, n_bench);
                seird::BenchmarkResult res;
                if (seird::cuda_run_benchmark(n_bench, 15, 60, bench_params, res)) {
                    cudaDeviceProp prop{};
                    cudaGetDeviceProperties(&prop, 0);
                    char msg_buf[320];
                    std::snprintf(
                        msg_buf, sizeof(msg_buf),
                        "Benchmark %s (%d agentes):\n\n"
                        "- Simulacion SEIRD: %.3f ms/paso\n"
                        "- Render 1080p: %.3f ms/frame\n"
                        "- Rendimiento: %.2f M agentes/seg\n"
                        "- VRAM usada: %.0f MB",
                        prop.name, res.num_agents, res.avg_step_ms, res.avg_render_ms,
                        res.agents_per_sec / 1e6, res.vram_used_mb
                    );
                    MessageBoxA(hwnd, msg_buf, "Benchmark GPU CUDA", MB_OK | MB_ICONINFORMATION);
                }
            } else if (id == 1022) { // Salir
                st->launch = false;
                st->done   = true;
                DestroyWindow(hwnd);
            }
            return 0;
        }
        default:
            return DefWindowProcA(hwnd, msg, wParam, lParam);
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

bool show_launcher_dialog(
    HINSTANCE hinst,
    int& num_agents,
    int& initial_infected,
    seird::SimParams& params,
    int& substeps,
    int& chart_days,
    bool& start_paused
) {
    WNDCLASSA lwc{};
    lwc.lpfnWndProc   = LauncherProc;
    lwc.hInstance     = hinst;
    lwc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    lwc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    lwc.lpszClassName = "SEIRDLauncherClass";
    RegisterClassA(&lwc);

    const seird::HardwareLimits hw = seird::cuda_query_hardware_limits();

    LauncherState st;
    st.agents       = std::min(num_agents, std::max(1000, hw.safe_agents_limit));
    st.max_agents   = std::max(1000, hw.safe_agents_limit);
    st.infected     = initial_infected;
    st.params       = params;
    st.substeps     = substeps;
    st.chart_days   = chart_days;
    st.start_paused = start_paused;
    g_launcher_state = &st;

    const int W = 580, H = 585;
    int sx = (GetSystemMetrics(SM_CXSCREEN) - W) / 2;
    int sy = (GetSystemMetrics(SM_CYSCREEN) - H) / 2;

    HWND lwnd = CreateWindowExA(
        WS_EX_DLGMODALFRAME, lwc.lpszClassName,
        "Miasma-Parallel | Configuracion de Simulacion (Launcher)",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
        sx, sy, W, H, nullptr, nullptr, hinst, nullptr
    );
    if (!lwnd) return false;

    HFONT font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
    auto mk = [&](const char* cls, const char* txt, DWORD style, int x, int y, int w, int h, int id = 0) -> HWND {
        HWND c = CreateWindowExA(0, cls, txt, WS_CHILD | WS_VISIBLE | style,
                                 x, y, w, h, lwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                 hinst, nullptr);
        SendMessageA(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return c;
    };
    auto mk_edit_i = [&](const char* label, int val, int x, int y) -> HWND {
        mk("STATIC", label, SS_LEFT, x, y + 3, 166, 18);
        char b[32];
        std::snprintf(b, sizeof(b), "%d", val);
        return mk("EDIT", b, WS_BORDER | ES_AUTOHSCROLL, x + 172, y, 78, 22);
    };
    auto mk_edit_f = [&](const char* label, float val, const char* fmt, int x, int y) -> HWND {
        mk("STATIC", label, SS_LEFT, x, y + 3, 166, 18);
        char b[32];
        std::snprintf(b, sizeof(b), fmt, val);
        return mk("EDIT", b, WS_BORDER | ES_AUTOHSCROLL, x + 172, y, 78, 22);
    };

    const int COL_L = 18;
    const int COL_R = 292;
    const int FULL_W = 524;

    mk("STATIC", "1. Enfermedad / Escenario Epidemiologico:", SS_LEFT, COL_L, 14, 320, 18);
    st.cmb_preset = mk("COMBOBOX", "", CBS_DROPDOWNLIST | WS_VSCROLL, COL_L, 34, FULL_W, 220, 1001);
    for (const auto& p : kDiseasePresets) {
        SendMessageA(st.cmb_preset, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(p.name));
    }
    SendMessageA(st.cmb_preset, CB_SETCURSEL, 1, 0); // COVID-19 Original por defecto
    st.lbl_info = mk("STATIC", kDiseasePresets[1].info, SS_LEFT, COL_L, 62, FULL_W, 32);

    mk("STATIC", "2. Poblacion y Movilidad:", SS_LEFT, COL_L, 104, 240, 18);
    char agents_lbl[48];
    std::snprintf(agents_lbl, sizeof(agents_lbl), "Agentes (1k - %.1fM):",
                  static_cast<double>(st.max_agents) / 1e6);
    st.edt_agents   = mk_edit_i(agents_lbl,                st.agents,                  COL_L, 126);
    st.edt_infected = mk_edit_i("Infectados iniciales:",   st.infected,                COL_R, 126);
    st.edt_speed    = mk_edit_f("Velocidad base (u/dia):", st.params.base_speed,       "%.1f", COL_L, 154);
    st.edt_quar     = mk_edit_f("Movilidad (0.05-1.50):",  st.params.quarantine_factor,"%.2f", COL_R, 154);

    mk("STATIC", "3. Parametros SEIRD:", SS_LEFT, COL_L, 194, 240, 18);
    st.edt_beta     = mk_edit_f("Contagio beta (0-1):",    st.params.beta,             "%.2f", COL_L, 216);
    st.edt_radius   = mk_edit_f("Radio contagio (1-8):",   st.params.infection_radius, "%.2f", COL_R, 216);
    st.edt_incub    = mk_edit_f("Incubacion (dias):",      st.params.incubation_days,  "%.1f", COL_L, 244);
    st.edt_inf_days = mk_edit_f("Periodo infec. (dias):",  st.params.infectious_days,  "%.1f", COL_R, 244);
    st.edt_cfr      = mk_edit_f("Letalidad CFR (0-1):",    st.params.fatality_rate,    "%.3f", COL_L, 272);

    mk("STATIC", "4. Visualizacion y Tiempo:", SS_LEFT, COL_L, 312, 240, 18);
    mk("STATIC", "Modo visual inicial:", SS_LEFT, COL_L, 338, 166, 18);
    st.cmb_view = mk("COMBOBOX", "", CBS_DROPDOWNLIST, COL_L + 172, 334, 92, 120);
    SendMessageA(st.cmb_view, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Agentes"));
    SendMessageA(st.cmb_view, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Heatmap"));
    SendMessageA(st.cmb_view, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("Hibrida"));
    SendMessageA(st.cmb_view, CB_SETCURSEL, 2, 0);

    st.edt_substeps = mk_edit_i("Velocidad (1h - 24h/frame):", st.substeps,   COL_R, 334);
    st.edt_chart    = mk_edit_i("Ventana grafica (dias):",    st.chart_days, COL_L, 366);

    mk("STATIC", "Replicas Monte Carlo (K):", SS_LEFT, COL_R, 369, 166, 18);
    st.cmb_replicas = mk("COMBOBOX", "", CBS_DROPDOWNLIST, COL_R + 172, 366, 78, 120);
    SendMessageA(st.cmb_replicas, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("1 (Normal)"));
    SendMessageA(st.cmb_replicas, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("8"));
    SendMessageA(st.cmb_replicas, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("16"));
    SendMessageA(st.cmb_replicas, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>("32"));
    SendMessageA(st.cmb_replicas, CB_SETCURSEL, 0, 0);

    st.chk_trails = mk("BUTTON", "Activar estelas de movimiento", BS_AUTOCHECKBOX, COL_L, 402, 240, 22);
    SendMessageA(st.chk_trails, BM_SETCHECK, BST_CHECKED, 0);

    st.chk_paused = mk("BUTTON", "Iniciar pausado", BS_AUTOCHECKBOX, COL_R, 402, 200, 22);

    apply_preset_to_launcher(&st, 1);

    mk("BUTTON", "Lanzar Simulacion (CUDA)", BS_DEFPUSHBUTTON, COL_L,       450, 240, 38, 1020);
    mk("BUTTON", "Benchmark GPU",            BS_PUSHBUTTON,    COL_L + 252, 450, 154, 38, 1021);
    mk("BUTTON", "Salir",                    BS_PUSHBUTTON,    COL_L + 418, 450, 106, 38, 1022);

    mk("STATIC", "Tip: Los presets usan datos epidemiologicos reales (R0, incubacion, periodo infeccioso y CFR).",
       SS_LEFT, COL_L, 502, FULL_W, 30);

    MSG msg;
    while (!st.done && GetMessageA(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageA(lwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }

    g_launcher_state = nullptr;
    if (!st.launch) return false;

    num_agents       = st.agents;
    initial_infected = st.infected;
    params           = st.params;
    substeps         = st.substeps;
    chart_days       = st.chart_days;
    start_paused     = st.start_paused;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    _putenv_s("OMP_WAIT_POLICY", "PASSIVE");

    const seird::HardwareLimits hw = seird::cuda_query_hardware_limits();
    int  num_agents = 500000;
    bool bench_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bench") {
            bench_mode = true;
        } else if (arg == "--agents" && i + 1 < argc) {
            num_agents = std::clamp(std::atoi(argv[++i]), 1000, std::max(1000, hw.safe_agents_limit));
        } else if (arg == "--infected" && i + 1 < argc) {
            g_initial_infected = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--substeps" && i + 1 < argc) {
            g_substeps = std::clamp(std::atoi(argv[++i]), 1, 24);
        }
    }

    if (bench_mode) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::printf("\n=== Miasma-Parallel Benchmark (GPU: %s, SM %d.%d) ===\n",
                    prop.name, prop.major, prop.minor);
        std::printf("%-12s | %-15s | %-17s | %-18s | %-12s\n",
                    "Agentes", "Sim (ms/paso)", "Render 1080p (ms)", "M agentes/seg", "VRAM (MB)");
        std::printf("------------------------------------------------------------------------------------\n");

        const int sizes[] = { 100000, 250000, 500000, 1000000 };
        for (int n : sizes) {
            seird::BenchmarkResult res;
            if (seird::cuda_run_benchmark(n, 20, 100, g_params, res)) {
                std::printf("%-12d | %-15.3f | %-17.3f | %-18.2f | %-12.0f\n",
                            res.num_agents,
                            res.avg_step_ms,
                            res.avg_render_ms,
                            res.agents_per_sec / 1e6,
                            res.vram_used_mb);
            } else {
                std::printf("%-12d | ERROR en CUDA\n", n);
            }
        }
        std::printf("====================================================================================\n\n");
        return 0;
    }

    INITCOMMONCONTROLSEX icex = { sizeof(icex), ICC_BAR_CLASSES };
    InitCommonControlsEx(&icex);

    HINSTANCE hinst = GetModuleHandleA(nullptr);

    bool start_paused_flag = g_paused.load();
    if (!show_launcher_dialog(hinst, num_agents, g_initial_infected, g_params, g_substeps, g_chart_days, start_paused_flag)) {
        return 0;
    }
    g_paused.store(start_paused_flag);

    WNDCLASSA wc = {};
    wc.style         = CS_OWNDC;
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hinst;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    wc.lpszClassName = "SEIRDNativeMainClass";
    if (!RegisterClassA(&wc)) return 1;

    WNDCLASSA gl_wc = {};
    gl_wc.style         = CS_OWNDC;
    gl_wc.lpfnWndProc   = GLCanvasProc;
    gl_wc.hInstance     = hinst;
    gl_wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    gl_wc.lpszClassName = "SEIRDGLCanvas";
    if (!RegisterClassA(&gl_wc)) return 1;

    HWND hwnd = CreateWindowExA(
        0, wc.lpszClassName,
        "Miasma-Parallel | Simulacion Epidemiologica CUDA",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, g_win_w, g_win_h,
        nullptr, nullptr, hinst, nullptr
    );
    if (!hwnd) return 1;

    g_gl_hwnd = CreateWindowExA(
        0, "SEIRDGLCanvas", "",
        WS_CHILD | WS_VISIBLE,
        0, 0, g_win_w - SIDEBAR_W, g_win_h,
        hwnd, nullptr, hinst, nullptr
    );

    create_sidebar_controls(hwnd, hinst);
    if (g_btn_pause) SetWindowTextA(g_btn_pause, g_paused.load() ? "Iniciar [SPACE]" : "Pausar [SPACE]");
    update_visual_buttons();
    layout_controls(g_win_w, g_win_h);

    HDC hdc = GetDC(g_gl_hwnd);
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize      = sizeof(pfd);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;

    int pf = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, pf, &pfd);
    HGLRC hglrc = wglCreateContext(hdc);
    wglMakeCurrent(hdc, hglrc);

    // Fuente bitmap OpenGL para etiquetas de días en la gráfica
    HFONT gl_font = CreateFontA(
        -13, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY, FF_DONTCARE | DEFAULT_PITCH, "Consolas"
    );
    HGDIOBJ old_font = SelectObject(hdc, gl_font);
    g_gl_font_base = glGenLists(96);
    wglUseFontBitmapsA(hdc, 32, 96, g_gl_font_base);
    SelectObject(hdc, old_font);
    DeleteObject(gl_font);

    // Activar VSync si el driver lo soporta para evitar que el loop de render sature la CPU
    using PFNWGLSWAPINTERVALEXTPROC = BOOL(WINAPI*)(int);
    auto wglSwapIntervalEXT = reinterpret_cast<PFNWGLSWAPINTERVALEXTPROC>(
        wglGetProcAddress("wglSwapIntervalEXT")
    );
    if (wglSwapIntervalEXT) {
        wglSwapIntervalEXT(1);
    }

    GLuint gl_tex = 0;
    glGenTextures(1, &gl_tex);
    glBindTexture(GL_TEXTURE_2D, gl_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8,
        seird::FRAME_WIDTH, seird::FRAME_HEIGHT, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, nullptr
    );
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsResource_t cuda_tex_res = nullptr;
    std::vector<uint8_t> host_fb_fallback;
    cudaError_t reg_err = cudaGraphicsGLRegisterImage(
        &cuda_tex_res, gl_tex, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard
    );
    if (reg_err != cudaSuccess) {
        const char* gl_renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        std::fprintf(
            stderr,
            "[AVISO] cudaGraphicsGLRegisterImage fallo (%s) en GL_RENDERER='%s'. "
            "Usando copia CPU->OpenGL como respaldo.\n",
            cudaGetErrorString(reg_err),
            gl_renderer ? gl_renderer : "desconocido"
        );
        cuda_tex_res = nullptr;
        host_fb_fallback.resize(
            static_cast<size_t>(seird::FRAME_WIDTH) * seird::FRAME_HEIGHT * 4
        );
    }

    const size_t fb_bytes = static_cast<size_t>(seird::FRAME_WIDTH) * seird::FRAME_HEIGHT * 4;
    uint8_t* d_fb_display = nullptr;
    cudaStream_t gl_copy_stream = nullptr;
    if (cudaMalloc(&d_fb_display, fb_bytes) != cudaSuccess ||
        cudaStreamCreateWithFlags(&gl_copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "Error al reservar framebuffer de despliegue en CUDA.\n");
        return 1;
    }
    cudaMemset(d_fb_display, 0, fb_bytes);

    std::mutex g_fb_mtx;
    std::mutex g_publish_mtx;
    std::atomic<bool> g_frame_ready{false};
    std::atomic<bool> worker_init_done{false};
    std::atomic<bool> worker_init_ok{false};

    seird::SimStats                 pub_stats{};
    seird::TelemetryMetrics         pub_metrics{};
    seird::EnsembleMetrics          pub_ensemble{};
    std::deque<HistoryPoint>        pub_curve_hist;
    std::vector<seird::TelemetryMetrics> pub_history;
    std::vector<seird::EnsembleMetrics>  pub_ens_history;
    double   pub_gpu_ms     = 0.0;
    uint64_t pub_seed       = 42ULL;
    int      pub_sim_frames = 0;

    std::thread sim_worker([&]() {
        cudaSetDevice(0);

        seird::SimParams local_params;
        int local_substeps = 4;
        int local_init_inf = 150;
        {
            std::lock_guard<std::mutex> lk(g_state_mtx);
            local_params   = g_params;
            local_substeps = g_substeps;
            local_init_inf = g_initial_infected;
        }

        seird::AgentSystemDevice dev;
        seird::SimStats stats;
        std::vector<seird::SimStats> rep_stats;
        seird::CpuTelemetryEngine telemetry(64);
        seird::CpuEnsembleEngine ensemble(local_params.num_replicas, 64);
        seird::TelemetryMetrics metrics;
        seird::EnsembleMetrics ens_metrics;
        std::deque<HistoryPoint> curve_hist;
        seird::OdeState ode_ref{};
        uint64_t active_seed = 42ULL;

        if (!seird::cuda_init_simulation(dev, num_agents, local_init_inf, local_params, active_seed)) {
            worker_init_ok.store(false);
            worker_init_done.store(true);
            return;
        }

        auto reset_host_state = [&](int init_inf, const seird::SimParams& p) {
            int clamped_init = std::min(init_inf, num_agents);
            stats = seird::SimStats{};
            stats.count_s = static_cast<uint32_t>(std::max(0, num_agents - clamped_init));
            stats.count_i = static_cast<uint32_t>(clamped_init);

            ode_ref = seird::OdeState{
                static_cast<double>(stats.count_s), 0.0,
                static_cast<double>(stats.count_i), 0.0, 0.0, 0.0
            };

            metrics = seird::TelemetryMetrics{};
            metrics.stats  = stats;
            metrics.r0_ngm = static_cast<float>(seird::compute_r0_ngm(num_agents, p));

            ensemble.reset(p.num_replicas);
            ens_metrics = seird::EnsembleMetrics{};
            ens_metrics.num_replicas = p.num_replicas;

            float tot = static_cast<float>(std::max(1, num_agents));
            float i0  = static_cast<float>(stats.count_i) / tot;
            ens_metrics.i_p05 = i0;
            ens_metrics.i_p50 = i0;
            ens_metrics.i_p95 = i0;

            curve_hist.clear();
            curve_hist.push_back({
                0.0f,
                static_cast<float>(stats.count_s) / tot,
                0.0f,
                i0,
                0.0f,
                0.0f,
                static_cast<float>(ode_ref.i) / tot,
                i0, i0, i0
            });

            seird::cuda_render_1080p_device(dev, p);
            {
                std::lock_guard<std::mutex> fb_lk(g_fb_mtx);
                seird::cuda_copy_framebuffer_to_device(dev, d_fb_display);
            }
            {
                std::lock_guard<std::mutex> pub_lk(g_publish_mtx);
                pub_stats       = stats;
                pub_metrics     = metrics;
                pub_ensemble    = ens_metrics;
                pub_curve_hist  = curve_hist;
                pub_history     = telemetry.history();
                pub_ens_history = ensemble.history();
                pub_seed        = active_seed;
            }
            g_frame_ready.store(true);
        };

        reset_host_state(local_init_inf, local_params);
        worker_init_ok.store(true);
        worker_init_done.store(true);

        auto last_stats_time = std::chrono::steady_clock::now() - std::chrono::milliseconds(250);

        while (g_running.load()) {
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                local_params   = g_params;
                local_substeps = g_substeps;
                local_init_inf = g_initial_infected;
            }

            if (g_reset_requested.exchange(false)) {
                telemetry.reset();
                active_seed = static_cast<uint64_t>(
                    std::chrono::steady_clock::now().time_since_epoch().count()
                );
                int clamped_init = std::min(local_init_inf, num_agents);
                seird::cuda_init_simulation(dev, num_agents, clamped_init, local_params, active_seed);
                reset_host_state(clamped_init, local_params);
            }

            if (!g_paused.load()) {
                auto step_t0 = std::chrono::steady_clock::now();
                bool want_stats = (std::chrono::duration<double>(step_t0 - last_stats_time).count() >= 0.20);

                seird::cuda_timer_begin();
                const double beta_kin = seird::compute_abm_kinetic_beta(num_agents, local_params);
                for (int s = 0; s < local_substeps; ++s) {
                    bool is_last = (s == local_substeps - 1);
                    const bool sample_now = want_stats && is_last;
                    seird::cuda_step_simulation(
                        dev, local_params, nullptr, stats, sample_now,
                        (sample_now && local_params.num_replicas > 1) ? &rep_stats : nullptr
                    );
                    ode_ref = seird::ode_seirds_step_rk4(ode_ref, beta_kin, local_params, static_cast<double>(local_params.dt));
                }

                if (want_stats) {
                    last_stats_time = step_t0;
                    metrics = telemetry.update(stats, local_params);
                    float tot = static_cast<float>(std::max(1, num_agents));
                    float i_frac = static_cast<float>(stats.count_i) / tot;
                    float i05 = i_frac, i50 = i_frac, i95 = i_frac;
                    if (local_params.num_replicas > 1 && !rep_stats.empty()) {
                        ens_metrics = ensemble.update(rep_stats, local_params, num_agents);
                        metrics.rt_p05       = ens_metrics.rt_p05;
                        metrics.rt_p95       = ens_metrics.rt_p95;
                        metrics.prob_rt_gt_1 = ens_metrics.prob_rt_gt_1;
                        i05 = ens_metrics.i_p05;
                        i50 = ens_metrics.i_p50;
                        i95 = ens_metrics.i_p95;
                    }
                    curve_hist.push_back({
                        stats.sim_day,
                        static_cast<float>(stats.count_s) / tot,
                        static_cast<float>(stats.count_e) / tot,
                        i_frac,
                        static_cast<float>(stats.count_r) / tot,
                        static_cast<float>(stats.count_d) / tot,
                        static_cast<float>(ode_ref.i) / tot,
                        i05, i50, i95
                    });
                    if (curve_hist.size() > 2048) curve_hist.pop_front();
                }

                seird::cuda_render_1080p_device(dev, local_params);
                double gpu_ms = static_cast<double>(seird::cuda_timer_end_ms());

                {
                    std::lock_guard<std::mutex> fb_lk(g_fb_mtx);
                    seird::cuda_copy_framebuffer_to_device(dev, d_fb_display);
                }
                {
                    std::lock_guard<std::mutex> pub_lk(g_publish_mtx);
                    pub_stats  = stats;
                    pub_gpu_ms = gpu_ms;
                    ++pub_sim_frames;
                    if (want_stats) {
                        pub_metrics     = metrics;
                        pub_ensemble    = ens_metrics;
                        pub_curve_hist  = curve_hist;
                        pub_history     = telemetry.history();
                        pub_ens_history = ensemble.history();
                    }
                }
                g_frame_ready.store(true);

                // Control de cadencia para preservar cámara lenta en 1 paso/frame sin frenar cargas pesadas
                const double target_sim_ms = (local_substeps == 1) ? 33.3 : 16.6;
                const double step_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - step_t0
                ).count();
                if (step_ms < target_sim_ms) {
                    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(target_sim_ms - step_ms));
                }
            } else {
                if (g_force_render.exchange(false)) {
                    seird::cuda_render_1080p_device(dev, local_params);
                    {
                        std::lock_guard<std::mutex> fb_lk(g_fb_mtx);
                        seird::cuda_copy_framebuffer_to_device(dev, d_fb_display);
                    }
                    g_frame_ready.store(true);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(8));
                }
            }
        }

        seird::cuda_free_simulation(dev);
    });

    while (!worker_init_done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!worker_init_ok.load()) {
        if (sim_worker.joinable()) sim_worker.join();
        std::fprintf(stderr, "Error al inicializar CUDA.\n");
        return 1;
    }

    seird::SimStats                 ui_stats{};
    seird::TelemetryMetrics         ui_metrics{};
    seird::EnsembleMetrics          ui_ensemble{};
    std::deque<HistoryPoint>        ui_curve_hist;
    double ui_gpu_ms = 0.0;

    float vram_used_mb = 0.0f, vram_total_mb = 0.0f;
    seird::cuda_get_vram_info(vram_used_mb, vram_total_mb);

    int ui_frame_counter = 0;
    auto last_ui_time = std::chrono::steady_clock::now() - std::chrono::milliseconds(300);

    while (g_running.load()) {
        MSG msg;
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        if (!g_running.load()) break;

        auto frame_t0 = std::chrono::steady_clock::now();

        if (g_export_requested.exchange(false)) {
            std::vector<seird::TelemetryMetrics> hist_copy;
            std::vector<seird::EnsembleMetrics>  ens_hist_copy;
            seird::EnsembleMetrics ens_copy{};
            seird::SimStats stats_copy{};
            seird::SimParams params_copy{};
            uint64_t seed_copy = 42ULL;
            double gpu_ms_copy = 0.0;
            int init_inf_copy  = 150;
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                params_copy   = g_params;
                init_inf_copy = g_initial_infected;
            }
            {
                std::lock_guard<std::mutex> pub_lk(g_publish_mtx);
                hist_copy     = pub_history;
                ens_hist_copy = pub_ens_history;
                ens_copy      = pub_ensemble;
                stats_copy    = pub_stats;
                seed_copy     = pub_seed;
                gpu_ms_copy   = pub_gpu_ms;
            }
            export_simulation_report(
                hwnd, hist_copy, ens_hist_copy, ens_copy, stats_copy, params_copy,
                seed_copy, num_agents, init_inf_copy, gpu_ms_copy
            );
        }

        if (g_frame_ready.exchange(false)) {
            std::lock_guard<std::mutex> fb_lk(g_fb_mtx);
            if (cuda_tex_res != nullptr) {
                cudaArray_t cu_array = nullptr;
                if (cudaGraphicsMapResources(1, &cuda_tex_res, gl_copy_stream) == cudaSuccess) {
                    if (cudaGraphicsSubResourceGetMappedArray(&cu_array, cuda_tex_res, 0, 0) == cudaSuccess) {
                        cudaMemcpy2DToArrayAsync(
                            cu_array, 0, 0,
                            d_fb_display,
                            static_cast<size_t>(seird::FRAME_WIDTH) * 4,
                            static_cast<size_t>(seird::FRAME_WIDTH) * 4,
                            seird::FRAME_HEIGHT,
                            cudaMemcpyDeviceToDevice,
                            gl_copy_stream
                        );
                    }
                    cudaGraphicsUnmapResources(1, &cuda_tex_res, gl_copy_stream);
                    cudaStreamSynchronize(gl_copy_stream);
                }
            } else {
                if (cudaMemcpyAsync(host_fb_fallback.data(), d_fb_display, fb_bytes, cudaMemcpyDeviceToHost, gl_copy_stream) == cudaSuccess) {
                    cudaStreamSynchronize(gl_copy_stream);
                    glBindTexture(GL_TEXTURE_2D, gl_tex);
                    glTexSubImage2D(
                        GL_TEXTURE_2D, 0, 0, 0,
                        seird::FRAME_WIDTH, seird::FRAME_HEIGHT,
                        GL_RGBA, GL_UNSIGNED_BYTE, host_fb_fallback.data()
                    );
                    glBindTexture(GL_TEXTURE_2D, 0);
                }
            }
        }

        {
            std::lock_guard<std::mutex> pub_lk(g_publish_mtx);
            ui_stats      = pub_stats;
            ui_metrics    = pub_metrics;
            ui_ensemble   = pub_ensemble;
            ui_curve_hist = pub_curve_hist;
            ui_gpu_ms     = pub_gpu_ms;
        }

        int rep_count_ui = 1;
        {
            std::lock_guard<std::mutex> lk(g_state_mtx);
            rep_count_ui = g_params.num_replicas;
        }

        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBindTexture(GL_TEXTURE_2D, gl_tex);
        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -0.56f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -0.56f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);

        draw_epidemic_curve_overlay(ui_curve_hist, ui_stats.sim_day, g_chart_days, rep_count_ui);
        SwapBuffers(hdc);

        // Mantener UI fluida (~60 FPS) sin bloquear por substeps de CUDA
        auto frame_t1 = std::chrono::steady_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(frame_t1 - frame_t0).count();
        if (frame_ms < 16.0) {
            Sleep(static_cast<DWORD>(16.0 - frame_ms));
        }

        ++ui_frame_counter;
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - last_ui_time).count();
        if (elapsed_s >= 0.25) {
            int sim_frames_dt = 0;
            {
                std::lock_guard<std::mutex> pub_lk(g_publish_mtx);
                sim_frames_dt  = pub_sim_frames;
                pub_sim_frames = 0;
            }
            const bool paused_now = g_paused.load();
            double fps = (!paused_now && sim_frames_dt > 0)
                ? (static_cast<double>(sim_frames_dt) / elapsed_s)
                : (paused_now ? 0.0 : (static_cast<double>(ui_frame_counter) / elapsed_s));

            seird::cuda_get_vram_info(vram_used_mb, vram_total_mb);
            float vram_pct = (vram_total_mb > 0.0f) ? (100.0f * vram_used_mb / vram_total_mb) : 0.0f;

            seird::SimParams params_snap;
            {
                std::lock_guard<std::mutex> lk(g_state_mtx);
                params_snap = g_params;
            }

            int whole_days = static_cast<int>(ui_stats.sim_day);
            int hours = static_cast<int>((ui_stats.sim_day - static_cast<float>(whole_days)) * 24.0f + 0.5f) % 24;
            const float inf_pct = (num_agents > 0)
                ? (static_cast<float>(ui_stats.count_i) / static_cast<float>(num_agents))
                : 0.0f;
            const bool hosp_sat = (inf_pct > params_snap.hospital_capacity_pct);
            const char* status_str = paused_now
                ? "PAUSADO"
                : (hosp_sat
                    ? "HOSP. SATURADO"
                    : (params_snap.focal_quarantine
                        ? "CUARENTENA FOCAL [F]"
                        : (ui_metrics.outbreak_alert ? "BROTE ACTIVO" : "NORMAL")));

            char ens_line[80];
            if (params_snap.num_replicas > 1) {
                std::snprintf(ens_line, sizeof(ens_line), "Ensamble K=%d | Ext: %.0f%% | %s",
                              params_snap.num_replicas, ui_ensemble.extinction_pct, status_str);
            } else {
                std::snprintf(ens_line, sizeof(ens_line), "Focos: %u (Pico %.1fx) | %s",
                              ui_stats.hotspot_cells, ui_stats.max_cell_z, status_str);
            }

            char stats_txt[768];
            std::snprintf(
                stats_txt, sizeof(stats_txt),
                "SIMULACION SEIRD | %d%s\n"
                "Dia: %.2f (%dd %02dh) | Rt: %.2f [%.2f-%.2f]\n"
                "P(Rt>1): %.0f%% | R0(NGM): %.2f | Inc: %.0f/d\n"
                "%s\n"
                "------------------------------\n"
                "Susceptibles (S): %u\n"
                "Expuestos    (E): %u\n"
                "Infectados   (I): %u\n"
                "Recuperados  (R): %u\n"
                "Fallecidos   (D): %u\n"
                "CFR: %.2f%%\n"
                "--- TELEMETRIA GPU -----------\n"
                "FPS: %.0f  |  GPU: %.2f ms  |  Zoom: %.1fx\n"
                "VRAM: %.0f / %.0f MB (%.1f%%)",
                num_agents,
                (params_snap.num_replicas > 1) ? " x K" : " personas",
                ui_stats.sim_day, whole_days, hours, ui_metrics.rt, ui_metrics.rt_p05, ui_metrics.rt_p95,
                ui_metrics.prob_rt_gt_1 * 100.0f, ui_metrics.r0_ngm, ui_metrics.new_cases_rate,
                ens_line,
                ui_stats.count_s, ui_stats.count_e, ui_stats.count_i, ui_stats.count_r, ui_stats.count_d,
                ui_metrics.cfr * 100.0f,
                fps, ui_gpu_ms, params_snap.cam_zoom,
                vram_used_mb, vram_total_mb, vram_pct
            );
            SetWindowTextA(g_lbl_stats, stats_txt);
            ui_frame_counter = 0;
            last_ui_time = now;
        }
    }

    if (sim_worker.joinable()) {
        sim_worker.join();
    }
    if (gl_copy_stream != nullptr) {
        cudaStreamDestroy(gl_copy_stream);
    }
    if (d_fb_display != nullptr) {
        cudaFree(d_fb_display);
    }
    if (g_gl_font_base != 0) {
        glDeleteLists(g_gl_font_base, 96);
    }
    if (cuda_tex_res != nullptr) {
        cudaGraphicsUnregisterResource(cuda_tex_res);
    }
    glDeleteTextures(1, &gl_tex);
    wglMakeCurrent(nullptr, nullptr);
    wglDeleteContext(hglrc);
    ReleaseDC(g_gl_hwnd, hdc);
    DestroyWindow(hwnd);
    return 0;
}
