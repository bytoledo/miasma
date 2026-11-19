# seird-Parallel

Motor de simulación epidemiológica estocástica basada en agentes (**ABM SEIRD**) acelerado en GPU con **NVIDIA CUDA**, análisis paralelo en CPU con **OpenMP + AVX2** y visualización nativa en tiempo real mediante **Win32 + OpenGL**.

## Características principales

- **Modelo ABM SEIRD estocástico en 2D continuo:** Estados Susceptible (`S`), Expuesto (`E`), Infectado (`I`), Recuperado (`R`) y Fallecido (`D`), con grupos de riesgo, superdispersión, movilidad Lagrangiana con ciclo diurno, vacunación, inmunidad decreciente, saturación hospitalaria y cuarentena focalizada por *Z-score*.
- **Aceleración en GPU (CUDA):** *Spatial Hashing* sobre malla uniforme con `cub::DeviceRadixSort::SortPairs` en $O(N)$, RNG *stateless*, reordenamiento físico de agentes en memoria SoA, salto de celdas sin infectados, ajuste dinámico de límites según VRAM disponible y renderizado directo a framebuffer RGBA 1080p con interoperabilidad CUDA–OpenGL.
- **Ensambles Monte Carlo en GPU:** Ejecución multi-réplica en un único lanzamiento (`K = 1..32`) con reducción en OpenMP para percentiles (`P5`, `P50`, `P95`), probabilidad epidémica $\mathbb{P}(R_t > 1)$ y tasa de extinción.
- **Telemetría y validación analítica (CPU + OpenMP):** Solver ODE SEIRDS (RK4), cálculo de $R_0$ mediante Matriz de Siguiente Generación (NGM) cinética 2D y estimación bootstrap de $R_t$.
- **Interfaz nativa Win32 + OpenGL:** Controles interactivos en tiempo real, gráfica de series temporales (con banda P5–P95 y curva ODE de referencia) y exportación reproducible a CSV y JSON en `reports/`.

## Estructura del repositorio

```text
seird-Parallel/
├── include/          # Cabeceras C++/CUDA (types.hpp, cuda_engine.cuh, cpu_engine.hpp)
├── src/              # Motores CUDA/OpenMP e interfaz Win32 + OpenGL
├── tests/            # Suite de pruebas de invariantes y convergencia (test_invariants.cpp)
├── docs/             # Documentación de arquitectura, marco teórico y bibliografía
│   ├── PROJECT_PLAN.md
│   ├── MARCO_TEORICO.md
│   └── referencias.bib
├── web/              # Recursos web auxiliares
├── CMakeLists.txt    # Configuración de compilación CMake (C++17 / CUDA)
└── build.bat         # Script de compilación rápida en Windows (MSVC + nvcc)
```

## Requisitos

- **Sistema operativo:** Windows (para el ejecutable con UI nativa Win32 + OpenGL).
- **Compilador C++:** MSVC (Visual Studio 2022 / Build Tools) con soporte C++17, AVX2 y OpenMP.
- **GPU y SDK:** GPU NVIDIA y NVIDIA CUDA Toolkit (por defecto configurado para `sm_120`; ajustable vía `CMAKE_CUDA_ARCHITECTURES`).
- **Build:** CMake $\ge 3.20$ (o el script `build.bat`).

## Compilación y ejecución

### Opción 1: CMake (compila el motor, la app y los tests)

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Si tu GPU utiliza otra arquitectura CUDA (por ejemplo `sm_86` o `sm_89`):

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

Ejecutar la aplicación y la suite de pruebas:

```powershell
.\build\seird_parallel.exe
.\build\seird_tests.exe
```

### Opción 2: `build.bat` (compilación directa de `seird_parallel.exe`)

```bat
build.bat
```

## Documentación adicional

- [Plan de arquitectura y estado del proyecto](docs/PROJECT_PLAN.md)
- [Marco teórico y formulación matemática](docs/MARCO_TEORICO.md)
- [Referencias bibliográficas (BibTeX)](docs/referencias.bib)
