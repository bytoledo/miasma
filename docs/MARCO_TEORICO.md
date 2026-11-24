# Marco Teórico y Fundamentación Científica — `seird-Parallel`

Este documento establece el sustento matemático, epidemiológico y metodológico del motor de simulación **seird-Parallel** ([PROJECT_PLAN.md](PROJECT_PLAN.md)), articulando la teoría compartimental clásica, sus extensiones modernas (estocásticas, espaciales, metapoblacionales y basadas en agentes) y los estándares operativos de salud pública con las decisiones concretas implementadas en el repositorio. Todas las citas bibliográficas corresponden a las entradas definidas en [referencias.bib](referencias.bib).

---

## 1. Introducción y evolución histórica de la epidemiología matemática

Aunque el estudio cuantitativo de la mortalidad por enfermedades infecciosas se remonta al siglo XVIII, la epidemiología matemática moderna se consolida a comienzos del siglo XX a partir de las investigaciones pioneras de Ronald Ross y Hilda Hudson (1915–1917) y, de forma decisiva, con la publicación en 1927 del trabajo fundacional de **William Ogilvy Kermack y Anderson Gray McKendrick** `[kermack1927contribution]`. Su formulación original —planteada inicialmente en términos de edad de infección antes de reducirse al caso markoviano de tasas constantes— demostró matemáticamente un resultado que ni la medicina ni la epidemiología empírica habían deducido previamente: la existencia de un **umbral epidémico** que determina si un patógeno desencadena un brote exponencial o se extingue sin agotar a la población susceptible `[kermack1927contribution]`, `[luque2020como]`.

A lo largo de las décadas posteriores, este núcleo teórico se expandió en tratados de referencia como los de Anderson y May `[anderson1991infectious]` y Keeling y Rohani `[keeling2008modeling]`, así como en la sistematización rigurosa de Hethcote `[hethcote2000mathematics]`. Como sintetiza Brauer `[brauer2017mathematical]`, la disciplina evolucionó desde **modelos estratégicos** —deliberadamente simples, orientados a iluminar mecanismos cualitativos universales como el umbral de invasión, el pico de contagios o la inmunidad de grupo— hacia **modelos tácticos** de alta resolución destinados a evaluar intervenciones sanitarias específicas en tiempo real.

La relevancia práctica de ambos niveles queda documentada en la evidencia histórica de la pandemia de gripe de 1918–1919 y en la respuesta a la COVID-19:

* **Lecciones de la gripe de 1918 (43 ciudades de EE. UU.):** El análisis retrospectivo de Markel et al. `[markel2007nonpharmaceutical]` (discutido también en `[luque2020como]`) mostró que medidas no farmacológicas (NPI) aplicadas tempranamente y sostenidas en el tiempo —como en San Luis frente a Pittsburgh— redujeron los picos semanales de mortalidad hasta en un **50 %** y la mortalidad total acumulada alrededor de un **20 %**, mientras que el levantamiento prematuro de restricciones al remitir la primera ola provocó **segundos picos infecciosos**.
* **Dinámica inicial de la COVID-19:** Durante marzo de 2020, las gráficas semilogarítmicas de casos confirmados exhibieron el crecimiento exponencial predicho por la teoría linealizada, aunque con pendientes muy distintas según la dinámica social y las medidas adoptadas (por ejemplo, $0{,}32$ en España frente a $0{,}085$ en Japón entre el 1 y el 16 de marzo de 2020; tras la implantación del estado de alarma en España el 14 de marzo, el número básico de reproducción estimado descendió de $\sim 2{,}5$ a $\sim 1{,}9$) `[luque2020como]`. Asimismo, modelos individuales de gran escala desarrollados desde mediados de la década de 2000 `[ferguson2006strategies]` sirvieron de base directa para las evaluaciones de demanda hospitalaria y unidades de cuidados intensivos (UCI) durante la emergencia de 2020 `[ferguson2020report9]`.

---

## 2. Fundamentos matemáticos de los modelos compartimentales

### 2.1 El modelo SIR clásico

En su versión determinista estándar para una población cerrada de tamaño $N$ y mezcla homogénea, el modelo de Kermack y McKendrick divide a los individuos en tres compartimentos según su estado epidemiológico (`S -> I -> R`) `[kermack1927contribution]`, `[luque2020como]`, `[idm2026modelinghub]`:

* **$S(t)$ (Susceptibles):** individuos no inmunes en riesgo de contraer la infección.
* **$I(t)$ (Infectados):** individuos infectados capaces de transmitir el patógeno.
* **$R(t)$ (Recuperados):** individuos que han superado la infección, adquieren inmunidad y dejan de transmitir.

La conservación de la población impone $N = S(t) + I(t) + R(t)$. Suponiendo que cada individuo infectado realiza $\mu$ contactos diarios efectivos en promedio, de los cuales una fracción $S(t)/N$ ocurre con susceptibles y una proporción $\tau \le 1$ resulta en contagio, se define la constante de transmisión $\beta = \tau \mu / N$ (en formulación no normalizada) y la tasa de recuperación $\gamma = 1/D$, siendo $D$ el tiempo medio de infecciosidad en días. En el límite continuo, el sistema de ecuaciones diferenciales ordinarias (EDO) es `[luque2020como]`:

$$
\frac{dS}{dt} = -\beta S(t) I(t), \qquad
\frac{dI}{dt} = \beta S(t) I(t) - \gamma I(t), \qquad
\frac{dR}{dt} = \gamma I(t)
$$

*(Nota de notación: en la formulación normalizada estándar usada por el IDM `[idm2026modelinghub]` y otros autores, $\beta_{\text{norm}} = \tau\mu$ absorbe el factor $1/N$ de modo que la incidencia se escribe $\beta_{\text{norm}} S I / N$ y $R_0 = \beta_{\text{norm}}/\gamma$.)*

A partir de este sistema emergen analíticamente cinco propiedades fundamentales `[luque2020como]`, `[hethcote2000mathematics]`, `[idm2026modelinghub]`:

1. **Fase inicial exponencial y umbral epidémico:** Al inicio del brote ($S(t) \approx S(0)$), la ecuación de infectados se linealiza a $I'(t) = (\beta S(0) - \gamma) I(t)$, cuya solución es:
   $$
   I(t) = I(0)\exp\big((\beta S(0) - \gamma)t\big)
   $$
   La epidemia solo crece si el **número efectivo de reproducción inicial** cumple $R_e = \beta S(0)/\gamma > 1$; en caso contrario se extingue. Cuando toda la población es inicialmente vulnerable ($S(0) = N$), $R_e$ coincide con el **número básico de reproducción**:
   $$
   R_0 = \frac{\beta N}{\gamma} = \tau \mu D
   $$
   En escala semilogarítmica, tanto los casos diarios como los acumulados $A(t) = (I(0)/\alpha)e^{\alpha t}$ forman una recta de pendiente $\alpha = \gamma(R_0 - 1)$ `[luque2020como]`. En la literatura formal, Hethcote `[hethcote2000mathematics]` distingue con precisión entre el número básico de reproducción ($R_0$), el número de contacto ($\sigma$) y el número de reemplazo ($R$).

2. **Identidad fundamental del SIR:** Dividiendo las ecuaciones de $S(t)$ e $I(t)$ se obtiene $dS/dI = \beta S / (\gamma - \beta S)$, que al integrarse produce la invariante exacta:
   $$
   I(t) + S(t) - \frac{\gamma}{\beta}\ln S(t) = I(0) + S(0) - \frac{\gamma}{\beta}\ln S(0)
   $$

3. **Pico máximo de infectados («aplanar la curva»):** El máximo $I_{\text{máx}}$ se alcanza cuando $I'(t_{\text{máx}}) = 0 \implies S(t_{\text{máx}}) = \gamma/\beta = N/R_0$. Sustituyendo en la identidad fundamental (con $S(0) = N$ e $I(0) \approx 0$) resulta:
   $$
   I_{\text{máx}} = N\left(1 - \frac{1 + \ln R_0}{R_0}\right)
   $$
   Esta relación cuantifica directamente cómo las intervenciones que disminuyen $\mu$ o $\tau$ reducen $R_0$ y abaten el pico de demanda hospitalaria $I_{\text{máx}}$ `[luque2020como]`, `[ferguson2020report9]`.

4. **Tasa de ataque final:** Evaluando la identidad en $t \to \infty$ (donde $I(\infty) = 0$) se obtiene la ecuación implícita para la fracción de susceptibles remanentes:
   $$
   \ln\left(\frac{S(\infty)}{N}\right) = -R_0\left(1 - \frac{S(\infty)}{N}\right)
   $$
   Esto demuestra que, incluso sin intervenciones, una epidemia remite dejando una fracción $S(\infty)/N > 0$ sin infectar, por lo que la tasa de ataque $1 - S(\infty)/N$ es estrictamente menor que el $100\,\%$ `[luque2020como]`.

5. **Inmunidad de grupo y umbral crítico de vacunación ($p_c$):** Si una fracción $p$ de la población es inmunizada al inicio ($R(0) = pN$, $S(0) = (1-p)N$), el umbral $R_e = \beta(1-p)N/\gamma = 1$ define la cobertura crítica necesaria para impedir el crecimiento epidémico `[luque2020como]`, `[idm2026modelinghub]`:
   $$
   p_c = 1 - \frac{1}{R_0}
   $$
   Por ejemplo, para enfermedades de alto $R_0$ como el sarampión o la rubeola ($p_c$ teórico $\sim 90\,\%$), al incorporar una eficacia vacunal imperfecta del $95\,\%$ las coberturas requeridas ascienden al $99\,\%$ y $91\,\%$ respectivamente (la viruela es hasta la fecha la única enfermedad infecciosa humana erradicada globalmente por vacunación); para el SARS-CoV-2 con $R_0 \approx 2{,}5$, $p_c$ se sitúa en el rango del $60\,\%$ al $75\,\%$ `[luque2020como]`.

---

### 2.2 Extensión a SEIR, SEIRD y SEIRDS con vacunación e inmunidad decreciente

Para enfermedades con un periodo de latencia o incubación no despreciable y mortalidad específica significativa, el esquema SIR se amplía incorporando el compartimento **Expuesto ($E$)** —individuos infectados que aún no transmiten— y el compartimento **Fallecido ($D$)** `[raimundez2021covid]`, `[sudhi2022scalable]`, `[idm2026modelinghub]`, `[rama2026systematic]`.

Incluyendo además una tasa diaria de vacunación $\nu_v$ ($S \to R$) y una tasa de pérdida de inmunidad $\xi = 1/T_{\text{inm}}$ ($R \to S$) en población cerrada sin demografía de fondo ($N = S + E + I + R + D$), el sistema **SEIRDS** queda formulado como:

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta \frac{S I}{N} - \nu_v S + \xi R \\
\frac{dE}{dt} &= \beta \frac{S I}{N} - \sigma E \\
\frac{dI}{dt} &= \sigma E - (\gamma + \mu_d) I \\
\frac{dR}{dt} &= \gamma I + \nu_v S - \xi R \\
\frac{dD}{dt} &= \mu_d I
\end{aligned}
$$

donde:
* $\beta$: tasa de transmisión efectiva (en unidades de $\text{día}^{-1}$, bajo incidencia estándar $\beta S I / N$).
* $\sigma = 1 / T_{\text{incub}}$: tasa de progresión del estado latente al estado infeccioso.
* $\gamma$: tasa de recuperación hacia $R$.
* $\mu_d$: tasa de mortalidad específica por la enfermedad hacia $D$ (equivalente, en modelos parametrizados por duración media infecciosa $D_{\text{inf}}$ y probabilidad de letalidad $\alpha_{\text{fat}}$, a una tasa total de salida de $I$ igual a $1/D_{\text{inf}}$ repartida en $(1 - \alpha_{\text{fat}})/D_{\text{inf}}$ hacia $R$ y $\alpha_{\text{fat}}/D_{\text{inf}}$ hacia $D$).

Como documenta el *ModelingHub* del Institute for Disease Modeling (IDM) `[idm2026modelinghub]`, la inclusión del compartimento latente $E$ **ralentiza la tasa de crecimiento exponencial inicial** en comparación con el SIR equivalente; sin embargo, en población cerrada sin mortalidad ni pérdida de inmunidad ($\mu_d = \nu_v = \xi = 0$), deja inalterados tanto $R_0 = \beta/\gamma$ como el tamaño acumulado final de infectados. Cuando se activa la pérdida de inmunidad ($\xi > 0$, modelo SEIRDS/SEIRS) o el reemplazo demográfico, los susceptibles se reabastecen continuamente y el sistema admite equilibrios endémicos acompañados de ondas o rebrotes epidémicos amortiguados `[idm2026modelinghub]`, `[hethcote2000mathematics]`.

---

### 2.3 Generalización de $R_0$ en sistemas complejos: Matriz de Siguiente Generación y dinámicas no lineales

En modelos con múltiples compartimentos infectivos (por ejemplo, $E$, $I$ asintomáticos, $I$ sintomáticos, múltiples grupos de riesgo o parches espaciales), la fórmula escalar $\beta/\gamma$ deja de ser suficiente. El método estándar para derivar $R_0$ de manera rigurosa es el **Operador / Matriz de Siguiente Generación (NGM)** formulado por van den Driessche y Watmough `[vandendriessche2002reproduction]` y Diekmann, Heesterbeek y Roberts `[diekmann2010construction]`:

1. Se agrupan los compartimentos infectivos en un vector $x \in \mathbb{R}^m$ y se descompone su dinámica como $\dot{x}_i = \mathcal{F}_i(x) - \mathcal{V}_i(x)$, donde $\mathcal{F}_i$ representa la aparición de *nuevas infecciones* y $\mathcal{V}_i$ agrupa las transiciones internas entre estados infectivos y salidas (recuperación o muerte).
2. Evaluando los jacobianos en el **equilibrio libre de enfermedad (DFE)** $x_0$:
   $$
   F = \left[\frac{\partial \mathcal{F}_i}{\partial x_j}(x_0)\right], \qquad V = \left[\frac{\partial \mathcal{V}_i}{\partial x_j}(x_0)\right]
   $$
3. El número básico de reproducción queda definido por el radio espectral de la matriz de siguiente generación $K = FV^{-1}$:
   $$
   R_0 = \rho(FV^{-1})
   $$
   bajo las condiciones de estabilidad del DFE establecidas en `[vandendriessche2002reproduction]`.

Además del análisis lineal alrededor del DFE, la literatura moderna advierte sobre comportamientos que escapan a los modelos simples `[brauer2017mathematical]`:
* **Bifurcaciones hacia atrás (*backward bifurcations*):** En modelos multigrupo o con asimetrías estructurales, pueden coexistir equilibrios endémicos estables incluso cuando $R_0 < 1$, lo que implica que reducir $R_0$ por debajo de $1$ no siempre garantiza por sí solo la eliminación inmediata de la enfermedad `[vandendriessche2002reproduction]`, `[brauer2017mathematical]`.
* **Extinción estocástica temprana y superdispersión:** Mediante procesos de ramificación de Galton–Watson y funciones generatrices de redes de contacto $G_0(z) = \sum_k p_k z^k$, Brauer `[brauer2017mathematical]` demuestra que incluso cuando $R_0 > 1$, existe una probabilidad positiva $z_\infty \in (0, 1)$ de que el brote se extinga de forma aleatoria antes de convertirse en una epidemia mayor, y que la heterogeneidad individual (eventos de superdispersión) desacelera la tasa de crecimiento inicial respecto a la predicción exponencial de mezcla homogénea `[brauer2017mathematical]`, `[estrada2020covid]`.

---

## 3. Taxonomía comparativa de paradigmas de modelado epidemiológico

La elección del paradigma de modelado responde a un compromiso entre tratabilidad analítica, realismo estructural, requerimientos de datos y costo computacional. La siguiente tabla resume los seis paradigmas principales documentados en las fuentes revisadas:

| Paradigma | Formalismo matemático | Fortalezas principales | Limitaciones clave | Referencias representativas |
| :--- | :--- | :--- | :--- | :--- |
| **Compartimental determinista (SIR / SEIR / SEIRD)** | Sistemas de EDO acopladas | Analíticamente tratable ($R_0$, umbral, $I_{\text{máx}}$, tasa de ataque); costo computacional mínimo; ideal como referencia base | Asume mezcla homogénea e infinita divisibilidad; inadecuado para brotes pequeños, geografía explícita o heterogeneidad individual | `[kermack1927contribution]`, `[hethcote2000mathematics]`, `[brauer2017mathematical]`, `[idm2026modelinghub]` |
| **Estratificado y metapoblacional** | EDO acopladas con matrices de contacto $C_{aa'}$ y movilidad entre parches | Incorpora grupos de edad, $\text{IFR}_a$ diferenciada y flujos geográficos entre regiones | Alta dimensionalidad; resultados y $R_0$ muy sensibles al modelo de movilidad: flujo **Euleriano** (sin memoria de origen) vs. viaje **Lagrangiano** (*Simple Trip*, matriz $\Psi_{ij}$) | `[vandendriessche2002reproduction]`, `[citron2021comparing]`, `[arenas2020derivation]`, `[rama2026systematic]` |
| **Espacial continuo (Reacción–Difusión)** | EDP parabólicas no lineales + FEM / diferencias finitas | Captura frentes de propagación continua, difusión dependiente de densidad, efecto Allee y fronteras geográficas (condiciones de Neumann); altamente escalable en MPI/GPU | Requiere mallas geográficas finas, condiciones de contorno y *solvers* paralelos complejos (p. ej., Euler implícito + descomposición RAS + AMG) | `[sudhi2022scalable]` |
| **Estocástico / procesos de ramificación** | Cadenas de Markov, *chain-binomial*, Gillespie, Galton–Watson, binomial negativa | Modela extinción aleatoria temprana, varianza finita, incertidumbre intrínseca y superdispersión | Requiere ensambles de miles de réplicas Monte Carlo; mayor costo de cómputo si no se paraleliza | `[brauer2017mathematical]`, `[estrada2020covid]`, `[rama2026systematic]` |
| **Basado en agentes (ABM / Microsimulación)** | Autómatas / partículas discretas con interacción espacial o en red | Máxima expresividad para heterogeneidad individual, grupos de riesgo, movilidad urbana y NPIs focalizadas; representó el **47,7 %** de los estudios espaciales COVID-19 (frente al **29,4 %** de EDO/EDP) | Complejidad de calibración e identificabilidad; alto consumo de memoria y complejidad $O(N^2)$ ingenua si no se emplean estructuras espaciales $O(N)$ | `[ferguson2006strategies]`, `[ferguson2020report9]`, `[oshinubi2026systematic]` |
| **Estadístico / Híbrido (GAM, Bayesiano, PINN)** | GAM jerárquico, MCMC, ABC-SMC, Redes Neuronales Informadas por Física (PINN) | Excelente para *nowcasting* de rezagos de reporte y estimación operativa de $R_t$ con intervalos creíbles | Menor extrapolación contrafactual fuera de régimen; las PINN mostraron intervalos predictivos sobreconfiados frente a ABC-SMC mecanicista en series ruidosas | `[cdc2026cfa]`, `[rama2026systematic]` |

---

## 4. Fundamentación científica de la arquitectura `seird-Parallel`

El motor `seird-Parallel` adopta un paradigma híbrido de alta escala: un **Modelo Basado en Agentes (ABM) con dinámica SEIRD estocástica en espacio 2D continuo** ejecutado masivamente en paralelo sobre GPU NVIDIA CUDA, acoplado a un módulo de telemetría estadística en CPU con OpenMP ([PROJECT_PLAN.md](PROJECT_PLAN.md), [include/types.hpp](include/types.hpp), [src/cpu_engine.cpp](src/cpu_engine.cpp)). En términos de la taxonomía **GPM (Geografía–Población–Movimiento)** introducida por Oshinubi et al. `[oshinubi2026systematic]`, el proyecto combina una arena geográfica 2D continua con malla de aceleración, una población discreta estratificada por riesgo (`RiskGroup`) y un movimiento estocástico con atracción hacia centros de actividad urbana (`urban_clustering`).

La siguiente matriz vincula cada componente real del código con su fundamento científico:

| Componente en `seird-Parallel` | Ubicación en el repositorio | Sustento teórico y evidencia científica |
| :--- | :--- | :--- |
| **Estados `AgentState` (`S, E, I, R, D`) y parámetros base** (`incubation_days = 5.0f`, `infectious_days = 10.0f`, `fatality_rate = 0.02f`, `dt = 1/48` días) | [include/types.hpp](include/types.hpp) | Refleja la transición $S \to E \to I \to \{R, D\}$ estándar para patógenos con periodo latente (`[idm2026modelinghub]`, `[raimundez2021covid]`). Los valores por defecto coinciden con órdenes de magnitud documentados para SARS-CoV-2 y gripe de 1918: incubación de 5–6 días (`[raimundez2021covid]`) y tasa de letalidad en torno al $2\,\%$ (`[luque2020como]`). |
| **Estratificación `RiskGroup`** (`RISK_LOW` ~50 %, letalidad $0{,}25\times$; `RISK_MEDIUM` ~35 %, $1{,}0\times$; `RISK_HIGH` ~15 %, letalidad $3{,}5\times$) | [include/types.hpp](include/types.hpp) | Responde a una de las carencias centrales del SIR homogéneo señaladas en `[luque2020como]`, `[hethcote2000mathematics]` y `[rama2026systematic]`: la movilidad y la razón de letalidad por infección ($\text{IFR}_a$) varían drásticamente a lo largo de la pirámide poblacional (alto riesgo en mayores de 70 años frente a población joven). |
| **Espacio continuo 2D, `urban_clustering` y *Spatial Hashing* $O(N)$ con `cub::DeviceRadixSort::SortPairs`** | [PROJECT_PLAN.md](PROJECT_PLAN.md), [include/types.hpp](include/types.hpp), [src/cuda_engine.cu](src/cuda_engine.cu) | Supera el supuesto irrealista de mezcla homogénea (`[luque2020como]`, `[arenas2020derivation]`, `[citron2021comparing]`). La malla uniforme con `cell_size >= infection_radius` reduce la búsqueda de vecinos de $O(N^2)$ a $O(N)$ en memoria SoA (*Structure of Arrays*), con reordenamiento físico coalescente y salto de celdas sin infectados (`[sudhi2022scalable]`, `[oshinubi2026systematic]`). |
| **Layouts urbanos teselados (`UrbanZone`), familias (`home_id`, `work_zone`) y atribución `infection_source`** | [include/types.hpp](include/types.hpp), [src/cuda_engine.cu](src/cuda_engine.cu) | Operativiza la movilidad **Lagrangiana** con retorno al hogar (`[citron2021comparing]`, `[arenas2020derivation]`): distritos rectangulares 5×5 (`PRESET_MONOCENTRIC`, `PRESET_POLYCENTRIC`), ciclo día/noche y semanal (`dt = 1/48` d), mezcla intrafamiliar de `RiskGroup` y registro del entorno de contagio (`Hogar, Trabajo, Tránsito, Hospital, Ocio`). |
| **Intervenciones (`quarantine_factor`, `focal_quarantine`, `household_quarantine`, `close_schools_work`, `vaccination_rate`, `immunity_days`) y `hospital_capacity_pct`** | [include/types.hpp](include/types.hpp), [src/cuda_engine.cu](src/cuda_engine.cu) | Modela NPIs globales, focalizadas por $Z$-score ($Z > 2{,}0$) y domiciliarias por hogar para abatir el contacto efectivo $\mu$ y el pico $I_{\text{máx}}$ respecto a la capacidad asistencial (`[luque2020como]`, `[ferguson2020report9]`, `[markel2007nonpharmaceutical]`). La vacunación ($S \to R$) operativiza el umbral $p_c = 1 - 1/R_0$ y `immunity_days` explora ondas endémicas (`[idm2026modelinghub]`). |
| **Telemetría, ensambles Monte Carlo (`CpuEnsembleEngine`), NGM, ODE RK4, sensibilidad y ABC-SMC** | [include/cpu_engine.hpp](include/cpu_engine.hpp), [src/cpu_engine.cpp](src/cpu_engine.cpp) | Conecta la microsimulación estocástica en GPU con métricas agregadas de vigilancia (`[luque2020como]`, `[cdc2026cfa]`), $R_0$ por NGM (`[vandendriessche2002reproduction]`, `[diekmann2010construction]`), evaluación probabilística (`WIS`, `IS80`, `RMSE`, `MAE`) y calibración bayesiana aproximada ABC-SMC (`[rama2026systematic]`, `[raimundez2021covid]`). |

### 4.1 Análisis matemático de la telemetría implementada en `CpuTelemetryEngine`

En [src/cpu_engine.cpp](src/cpu_engine.cpp), `CpuTelemetryEngine::update` calcula cuatro indicadores a partir de las series temporales de `SimStats`:

1. **Tasa de incidencia diaria (`new_cases_rate`):** Se calcula directamente a partir del contador atómico exacto de transiciones $S \to E$ en GPU (`stats.new_exposures / elapsed_step`). Al medir el flujo real de nuevas exposiciones en lugar de $\Delta S / \Delta t$, la incidencia permanece exacta incluso cuando se activan simultáneamente vacunación (`vaccination_rate > 0`) o pérdida de inmunidad (`immunity_days > 0`).
2. **Número reproductivo efectivo ($R_t$) y categorías CDC CFA:** A partir de la ventana móvil de casos activos $A_{\text{act}}(t) = E(t) + I(t)$ con un desfase `lag` (hasta 10 muestras), el motor estima la tasa de crecimiento exponencial instantánea:
   $$
   r = \frac{\ln\big(E(t) + I(t)\big) - \ln\big(E(t-\Delta t_{\text{lag}}) + I(t-\Delta t_{\text{lag}})\big)}{\Delta t_{\text{lag}}}
   $$
   y obtiene $R_t = \max\big(0,\; 1 + r \cdot T_g\big)$ con $T_g \approx T_{\text{incub}} + \frac{1}{2} T_{\text{infec}}$ (aproximación de primer orden de Wallinga–Lipsitch / Lotka–Euler). Adicionalmente, mediante *bootstrap* sobre las tasas recientes (y de forma empírica sobre las $K$ réplicas en `CpuEnsembleEngine`), calcula los intervalos `[rt_p05, rt_p95]`, $\mathbb{P}(R_t > 1)$ y asigna la categoría operativa `RtCategory` estándar del CDC CFA `[cdc2026cfa]` (*Declining* $<10\,\%$, *Likely Declining* $<25\,\%$, *Uncertain*, *Likely Growing* $>75\,\%$, *Growing* $>90\,\%$).
3. **Tasa de letalidad acumulada (`cfr` vs. `ifr`):** El motor distingue explícitamente entre la **CFR** sobre casos resueltos $\text{CFR} = D / (R + D)$ y la **IFR** real sobre el total acumulado de infecciones verdaderas $\text{IFR} = D / \text{cum\_infections\_}$ (acumulado a partir de `new_exposures`, evitando sesgos cuando existe vacunación hacia $R$ o reinfecciones $R \to S$), en línea con las advertencias de `[luque2020como]`, `[raimundez2021covid]` y `[rama2026systematic]`.
4. **Alerta de brote (`outbreak_alert`):** Se activa cuando `((z_score > 2.0f) || (stats.max_cell_z > 2.0f)) && (rt > 1.0f)`, combinando un criterio estadístico de anomalía temporal o espacial ($> 2\sigma$ sobre la media local) con la condición epidemiológica de expansión sostenida ($R_t > 1$).

---

## 5. Rigurosidad metodológica: calibración, identificabilidad, incertidumbre y verificación

Para que un simulador epidemiológico paralelo sea científicamente defendible y no un ejercicio puramente ilustrativo, la literatura revisada establece cuatro exigencias metodológicas clave:

### 5.1 Identificabilidad práctica y sesgos en datos agregados
Raimúndez et al. `[raimundez2021covid]` evaluaron distintas variantes del modelo SEIRD frente a series públicas tempranas de Wuhan y demostraron que los conteos agregados de casos infectados, recuperados y fallecidos suelen ser insuficientes para identificar de forma única los parámetros internos del modelo. En particular:
* Los métodos bayesianos MCMC con *priors* acotados poco informativos produjeron intervalos de credibilidad al $99\,\%$ artificialmente estrechos pero biológicamente implausibles (por ejemplo, periodos de incubación de $[23{,}4,\; 257{,}6]$ días), mientras que el análisis frecuentista mediante **verosimilitud de perfil (*profile likelihood*)** reveló correctamente la amplia no-identificabilidad real.
* Fijar arbitrariamente condiciones iniciales no observadas (como $E_0$) introduce sesgos severos en las tasas de progresión estimadas `[raimundez2021covid]`.

### 5.2 Límites de predictibilidad y efecto de la heterogeneidad
Estrada `[estrada2020covid]` documenta que, durante la fase de crecimiento exponencial, pequeñas variaciones paramétricas generan una divergencia rápida de los intervalos predictivos posteriores (oscilando entre extinción y crecimiento sostenido), por lo que la predicción cuantitativa de puntos de inflexión y picos solo es confiable en horizontes cortos. Asimismo, Brauer `[brauer2017mathematical]` enfatiza que la variabilidad en la tasa de contactos individuales (superdispersión) altera sustancialmente la probabilidad de extinción temprana y la eficacia del rastreo frente al modelo de mezcla homogénea.

### 5.3 Calibración estocástica, sustitutos híbridos y marcos de decisión
Rama et al. `[rama2026systematic]` compararon un modelo SEIRD estocástico por edades calibrado mediante **Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC)** frente a Redes Neuronales Informadas por Física (PINN) calibradas vía MCMC. Sobre datos sintéticos con ruido binomial negativo, el enfoque mecanicista estocástico superó sistemáticamente a las PINN en precisión puntual y cobertura probabilística (MAE, RMSE, WIS, IS80), mientras que las PINN tendieron a producir intervalos excesivamente estrechos (sobreconfiados). A su vez, tanto el marco interagencial OMS/OCDE/Banco Mundial `[who2024integrated]` como el *ModelingHub* del IDM `[idm2026modelinghub]` y el *Modeling Handbook* del CDC CFA `[cdc2026cfa]` coinciden en que la calibración debe ser un flujo iterativo con cuantificación explícita de incertidumbre, ensambles multimodelo y transparencia sobre supuestos estructurales cuando cambian las intervenciones.

### 5.4 Verificación, invariantes y validación científica empírica en `seird-Parallel`
La revisión sistemática de Oshinubi et al. `[oshinubi2026systematic]` sobre 109 estudios espaciales reveló una severa brecha de reproducibilidad: apenas el **30{,}6\,\%** puso su código fuente en acceso abierto. En `seird-Parallel`, la verificación cubre tanto propiedades estructurales como validación cuantitativa frente a la literatura:

1. **Suite de invariantes ([tests/test_invariants.cpp](tests/test_invariants.cpp), 12 pruebas):**
   * **Conservación de masa y telemetría:** Valida paso a paso $S + E + I + R + D = N$ en el modelo base, con intervenciones, en ensambles Monte Carlo (`K = 8`) y en layouts urbanos (`PRESET_MONOCENTRIC`, `household_quarantine`), junto con la finitud numérica de todas las métricas.
   * **Consistencia de dominio, memoria y reproducibilidad:** Verifica límites dinámicos de VRAM, la invariante geométrica `cell_size >= infection_radius` y reproducibilidad bit a bit ante la misma semilla.
   * **Validación analítica, sensibilidad y ABC-SMC:** Verifica la convergencia ABM vs. ODE RK4 en el límite de alta difusión, $R_0(\text{NGM})$ sobre los 3 estratos `RiskGroup` (`[vandendriessche2002reproduction]`, `[diekmann2010construction]`), métricas probabilísticas (`WIS`, `IS80`, `RMSE`, `MAE`, coberturas), barridos de sensibilidad global (`run_global_sensitivity_sweep`) y calibración estocástica (`calibrate_abc_smc`, `[rama2026systematic]`).

2. **Suite de benchmarks científicos ([tests/test_scientific_benchmarks.cpp](tests/test_scientific_benchmarks.cpp), 8 experimentos):**
   * **Umbrales analíticos (`[kermack1927contribution]`, `[luque2020como]`):** Para $R_0 \in \{1{,}5,\; 2{,}5,\; 10{,}0\}$, el solver RK4 con *sub-stepping* adaptativo reproduce la tasa de ataque final y $I_{\text{máx}}/N$ (error de `1.44e-5` en $R_0=2{,}5$ y `4.54e-9` en $R_0=10{,}0$) y confirma la contención exacta en el umbral de inmunidad de grupo $p_c = 1 - 1/R_0$.
   * **Invariancia del periodo latente (`[idm2026modelinghub]`):** Comparando SIR vs. SEIR ($T_{\text{incub}} = 5\text{ d}$, $R_0 = 2{,}5$), el compartimento $E$ retrasa el pico del día `53.5` al `98.7` y lo reduce de `23.35 %` a `15.38 %`, manteniendo inalterado el ataque final (`0.8927`, diferencia `4.03e-5`).
   * **NPIs de la gripe de 1918 (`[markel2007nonpharmaceutical]`):** En GPU ABM (`N = 20 000`), la aplicación temprana y sostenida de NPIs reduce el pico de `27.52 %` a `2.04 %` y la mortalidad de `4.12 %` a `0.40 %`, mientras que el levantamiento prematuro al día 25 provoca un **segundo pico infeccioso** (`24.68 %`).
   * **Ondas endémicas (`[hethcote2000mathematics]`):** Con pérdida de inmunidad ($T_{\text{inm}} = 35\text{ d}$, $R_0 = 2{,}8$), el sistema SEIRDS exhibe ondas amortiguadas y converge a un equilibrio endémico (`9.97 %`).
   * **Extinción estocástica temprana y superdispersión (`[brauer2017mathematical]`, `[estrada2020covid]`):** Con $I_0 = 3$ y $R_0 > 1$ (`K = 32`), el escenario homogéneo presenta `53.1 %` de extinción temprana frente a `78.1 %` con superdispersores (`10 % × 6`).
   * **Estructura urbana (`[citron2021comparing]`):** Bajo parámetros idénticos, la ciudad policéntrica reduce y retrasa el pico respecto a la monocéntrica (`39.43 %` en día `33.0` vs. `46.91 %` en día `29.0`).
   * **Identificabilidad práctica (`[raimundez2021covid]`):** `calibrate_abc_smc` recupera $\beta$ con precisión en 1 parámetro (`0.259` vs. `0.280`), mientras que en 3 parámetros simultáneos muestra compensación paramétrica (`0.353 [0.213–0.474]`) con menor `WIS` (`0.0050`).

---

## 6. Referencias bibliográficas

1. `[kermack1927contribution]` Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics. *Proceedings of the Royal Society of London. Series A*, 115(772), 700–721. DOI: [10.1098/rspa.1927.0118](https://doi.org/10.1098/rspa.1927.0118).
2. `[anderson1991infectious]` Anderson, R. M., & May, R. M. (1991). *Infectious Diseases of Humans: Dynamics and Control*. Oxford University Press. ISBN: `978-0-19-854040-3`.
3. `[hethcote2000mathematics]` Hethcote, H. W. (2000). The Mathematics of Infectious Diseases. *SIAM Review*, 42(4), 599–653. DOI: [10.1137/S0036144500371907](https://doi.org/10.1137/S0036144500371907).
4. `[vandendriessche2002reproduction]` van den Driessche, P., & Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180(1–2), 29–48. DOI: [10.1016/s0025-5564(02)00108-6](https://doi.org/10.1016/s0025-5564(02)00108-6).
5. `[ferguson2006strategies]` Ferguson, N. M., et al. (2006). Strategies for mitigating an influenza pandemic. *Nature*, 442, 448–452.
6. `[markel2007nonpharmaceutical]` Markel, H., et al. (2007). Nonpharmaceutical Interventions Implemented by US Cities During the 1918–1919 Influenza Pandemic. *JAMA*, 298(6), 644–654.
7. `[keeling2008modeling]` Keeling, M. J., & Rohani, P. (2008). *Modeling Infectious Diseases in Humans and Animals*. Princeton University Press. ISBN: `978-0-691-11617-3`. DOI: [10.1515/9781400841035](https://doi.org/10.1515/9781400841035).
8. `[diekmann2010construction]` Diekmann, O., Heesterbeek, J. A. P., & Roberts, M. G. (2010). The construction of next-generation matrices for compartmental epidemic models. *Journal of the Royal Society Interface*, 7(47), 873–885. DOI: [10.1098/rsif.2009.0386](https://doi.org/10.1098/rsif.2009.0386).
9. `[brauer2017mathematical]` Brauer, F. (2017). Mathematical epidemiology: Past, present, and future. *Infectious Disease Modelling*, 2(2), 113–127. DOI: [10.1016/j.idm.2017.02.001](https://doi.org/10.1016/j.idm.2017.02.001) · [PMC6001967](https://pmc.ncbi.nlm.nih.gov/articles/PMC6001967/).
10. `[ferguson2020report9]` Imperial College COVID-19 Response Team (Ferguson, N. M., et al.) (2020). *Report 9: Impact of non-pharmaceutical interventions (NPIs) to reduce COVID-19 mortality and healthcare demand*. Imperial College London. DOI: [10.25561/77482](https://doi.org/10.25561/77482).
11. `[arenas2020derivation]` Arenas, A., et al. (2020). Derivation of the effective reproduction number R for COVID-19 in relation to mobility restrictions and confinement. *medRxiv* 2020.04.06.20054320. DOI: [10.1101/2020.04.06.20054320](https://doi.org/10.1101/2020.04.06.20054320).
12. `[luque2020como]` Luque, B., Ballesteros, F., & Miramontes, O. (2020). Cómo modelizar una pandemia. *Investigación y Ciencia*, mayo de 2020, 52–60.
13. `[estrada2020covid]` Estrada, E. (2020). COVID-19 and SARS-CoV-2. Modeling the present, looking at the future. *Physics Reports*, 869, 1–51. DOI: [10.1016/j.physrep.2020.07.005](https://doi.org/10.1016/j.physrep.2020.07.005) · [PMC7386394](https://pmc.ncbi.nlm.nih.gov/articles/PMC7386394/).
14. `[raimundez2021covid]` Raimúndez, E., Dudkin, E., Vanhoefer, J., Alamoudi, E., Merkt, S., Fuhrmann, L., Bai, F., & Hasenauer, J. (2021). COVID-19 outbreak in Wuhan demonstrates the limitations of publicly available case numbers for epidemiological modeling. *Epidemics*, 34, 100439. DOI: [10.1016/j.epidem.2021.100439](https://doi.org/10.1016/j.epidem.2021.100439) · [PMC7845523](https://pmc.ncbi.nlm.nih.gov/articles/PMC7845523/).
15. `[citron2021comparing]` Citron, D. T., Guerra, C. A., Dolgert, A. J., Wu, S. L., Henry, J. M., Sánchez C., H. M., & Smith, D. L. (2021). Comparing metapopulation dynamics of infectious diseases under different models of human movement. *Proceedings of the National Academy of Sciences (PNAS)*, 118(18), e2007488118. DOI: [10.1073/pnas.2007488118](https://doi.org/10.1073/pnas.2007488118) · [PMC8106338](https://pmc.ncbi.nlm.nih.gov/articles/PMC8106338/).
16. `[sudhi2022scalable]` Sudhi P. V., Dolean, V., Jolivet, P., Robinson, B., Edwards, J. D., Kendzerska, T., & Sarkar, A. (2022). Scalable Computational Algorithms for Geo-spatial Covid-19 Spread in High Performance Computing. *arXiv:2208.01839*. DOI: [10.48550/arXiv.2208.01839](https://doi.org/10.48550/arXiv.2208.01839) · [arXiv](https://arxiv.org/abs/2208.01839).
17. `[who2024integrated]` World Health Organization, OECD, & World Bank (2024). *Strengthening Pandemic Preparedness and Response Through Integrated Modelling*. World Health Organization.
18. `[idm2026modelinghub]` Institute for Disease Modeling (2026). *ModelingHub Documentation: Compartmental Models and Calibration*. [docs.idmod.org/modeling-hub](https://docs.idmod.org/modeling-hub/).
19. `[cdc2026cfa]` U.S. Centers for Disease Control and Prevention, Center for Forecasting and Outbreak Analytics (2026). *Modeling Handbook and Time-Varying Reproduction Number ($R_t$) Technical Methodology*. [cdc.gov/cfa-modeling-and-forecasting](https://www.cdc.gov/cfa-modeling-and-forecasting/modeling-handbook/index.html).
20. `[oshinubi2026systematic]` Oshinubi, K., Chen, Y., Doerry, E., Gel, E. S., Hepp, C., Lant, T., Mehrotra, S., Sabo, S., & Mihaljevic, J. (2026). A systematic review of spatial epidemiological modeling approaches applied during the COVID-19 pandemic. *BMC Public Health*, 26, 1691. DOI: [10.1186/s12889-026-27321-4](https://doi.org/10.1186/s12889-026-27321-4) · [PMC13200481](https://pmc.ncbi.nlm.nih.gov/articles/PMC13200481/).
21. `[rama2026systematic]` Rama, M., Li, Y., Lepri, B., Perra, N., & Tizzoni, M. (2026). A systematic comparison of physics-informed neural networks and mechanistic models for epidemic modeling and forecasting. *arXiv:2611.18389*. [arXiv](https://arxiv.org/abs/2611.18389).
