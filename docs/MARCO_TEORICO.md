# Theoretical Framework and Scientific Foundation — `seird-Parallel`

This document establishes the mathematical, epidemiological, and methodological foundation of the **seird-Parallel** simulation engine ([PROJECT_PLAN.md](PROJECT_PLAN.md)), connecting classical compartmental theory, its modern extensions (stochastic, spatial, metapopulation, and agent-based), and operational public health standards with the concrete design decisions implemented in the repository. All bibliographic citations correspond to entries defined in [referencias.bib](referencias.bib).

---

## 1. Introduction and Historical Evolution of Mathematical Epidemiology

Although the quantitative study of infectious disease mortality dates back to the 18th century, modern mathematical epidemiology was consolidated in the early 20th century through the pioneering work of Ronald Ross and Hilda Hudson (1915–1917) and, decisively, with the 1927 foundational paper by **William Ogilvy Kermack and Anderson Gray McKendrick** `[kermack1927contribution]`. Their original formulation—initially framed in terms of infection age before reducing to the Markovian constant-rate case—proved mathematically a result that neither clinical medicine nor empirical epidemiology had previously deduced: the existence of an **epidemic threshold** determining whether a pathogen triggers an exponential outbreak or dies out without exhausting the susceptible population `[kermack1927contribution]`, `[luque2020como]`.

Over subsequent decades, this theoretical core expanded in reference treatises such as Anderson and May `[anderson1991infectious]` and Keeling and Rohani `[keeling2008modeling]`, as well as Hethcote's rigorous systematization `[hethcote2000mathematics]`. As synthesized by Brauer `[brauer2017mathematical]`, the discipline evolved from **strategic models**—deliberately simple, aimed at illuminating universal qualitative mechanisms such as the invasion threshold, epidemic peak, and herd immunity—toward high-resolution **tactical models** designed to evaluate specific public health interventions in real time.

The practical relevance of both levels is documented in the historical evidence from the 1918–1919 influenza pandemic and the COVID-19 response:

* **Lessons from the 1918 Influenza Pandemic (43 U.S. cities):** Retrospective analysis by Markel et al. `[markel2007nonpharmaceutical]` (also discussed in `[luque2020como]`) showed that non-pharmaceutical interventions (NPIs) implemented early and sustained over time—as in St. Louis compared to Pittsburgh—reduced weekly peak mortality by up to **50%** and total cumulative mortality by roughly **20%**, whereas premature lifting of restrictions as the first wave subsided triggered **second infectious peaks**.
* **Early COVID-19 Dynamics:** During March 2020, semi-logarithmic plots of confirmed cases exhibited the exponential growth predicted by linearized theory, albeit with markedly different slopes depending on social dynamics and adopted measures (e.g., $0.32$ in Spain vs. $0.085$ in Japan between March 1 and March 16, 2020; following Spain's state of alarm on March 14, the estimated reproduction number dropped from $\sim 2.5$ to $\sim 1.9$) `[luque2020como]`. Likewise, large-scale individual-based models developed since the mid-2000s `[ferguson2006strategies]` formed the direct basis for hospital and intensive care unit (ICU) demand assessments during the 2020 emergency `[ferguson2020report9]`.

---

## 2. Mathematical Foundations of Compartmental Models

### 2.1 The Classical SIR Model

In its standard deterministic version for a closed, homogeneously mixed population of size $N$, the Kermack–McKendrick model divides individuals into three compartments by epidemiological state (`S -> I -> R`) `[kermack1927contribution]`, `[luque2020como]`, `[idm2026modelinghub]`:

* **$S(t)$ (Susceptible):** non-immune individuals at risk of contracting the infection.
* **$I(t)$ (Infected):** infected individuals capable of transmitting the pathogen.
* **$R(t)$ (Recovered):** individuals who have cleared the infection, acquired immunity, and ceased transmitting.

Population conservation enforces $N = S(t) + I(t) + R(t)$. Assuming each infected individual makes an average of $\mu$ effective daily contacts, of which a fraction $S(t)/N$ occurs with susceptibles and a proportion $\tau \le 1$ results in transmission, the transmission constant is defined as $\beta = \tau \mu / N$ (in unnormalized form) and the recovery rate as $\gamma = 1/D$, where $D$ is the mean infectious period in days. In the continuous limit, the system of ordinary differential equations (ODEs) is `[luque2020como]`:

$$
\frac{dS}{dt} = -\beta S(t) I(t), \qquad
\frac{dI}{dt} = \beta S(t) I(t) - \gamma I(t), \qquad
\frac{dR}{dt} = \gamma I(t)
$$

*(Notation note: in the standard normalized formulation used by IDM `[idm2026modelinghub]` and other authors, $\beta_{\text{norm}} = \tau\mu$ absorbs the $1/N$ factor so that incidence is written $\beta_{\text{norm}} S I / N$ and $R_0 = \beta_{\text{norm}}/\gamma$.)*

Five fundamental properties emerge analytically from this system `[luque2020como]`, `[hethcote2000mathematics]`, `[idm2026modelinghub]`:

1. **Initial Exponential Phase and Epidemic Threshold:** At the onset of an outbreak ($S(t) \approx S(0)$), the infected equation linearizes to $I'(t) = (\beta S(0) - \gamma) I(t)$, yielding:
   $$
   I(t) = I(0)\exp\big((\beta S(0) - \gamma)t\big)
   $$
   The epidemic grows only if the **initial effective reproduction number** satisfies $R_e = \beta S(0)/\gamma > 1$; otherwise it dies out. When the entire population is initially susceptible ($S(0) = N$), $R_e$ coincides with the **basic reproduction number**:
   $$
   R_0 = \frac{\beta N}{\gamma} = \tau \mu D
   $$
   On a semi-logarithmic scale, both daily and cumulative cases $A(t) = (I(0)/\alpha)e^{\alpha t}$ form a straight line of slope $\alpha = \gamma(R_0 - 1)$ `[luque2020como]`. In formal literature, Hethcote `[hethcote2000mathematics]` distinguishes carefully between the basic reproduction number ($R_0$), the contact number ($\sigma$), and the replacement number ($R$).

2. **Fundamental SIR Identity:** Dividing the $S(t)$ and $I(t)$ equations gives $dS/dI = \beta S / (\gamma - \beta S)$, which integrates to the exact invariant:
   $$
   I(t) + S(t) - \frac{\gamma}{\beta}\ln S(t) = I(0) + S(0) - \frac{\gamma}{\beta}\ln S(0)
   $$

3. **Maximum Infected Peak ("Flattening the Curve"):** The peak $I_{\max}$ occurs when $I'(t_{\max}) = 0 \implies S(t_{\max}) = \gamma/\beta = N/R_0$. Substituting into the fundamental invariant (with $S(0) = N$ and $I(0) \approx 0$) yields:
   $$
   I_{\max} = N\left(1 - \frac{1 + \ln R_0}{R_0}\right)
   $$
   This relation directly quantifies how interventions that lower $\mu$ or $\tau$ reduce $R_0$ and suppress peak hospital demand $I_{\max}$ `[luque2020como]`, `[ferguson2020report9]`.

4. **Final Attack Rate:** Evaluating the invariant as $t \to \infty$ (where $I(\infty) = 0$) yields the implicit equation for the remaining susceptible fraction:
   $$
   \ln\left(\frac{S(\infty)}{N}\right) = -R_0\left(1 - \frac{S(\infty)}{N}\right)
   $$
   This proves that even without interventions, an epidemic subsides leaving a fraction $S(\infty)/N > 0$ uninfected, so the final attack rate $1 - S(\infty)/N$ is strictly less than $100\%$ `[luque2020como]`.

5. **Herd Immunity and Critical Vaccination Threshold ($p_c$):** If a fraction $p$ of the population is immunized initially ($R(0) = pN$, $S(0) = (1-p)N$), the threshold $R_e = \beta(1-p)N/\gamma = 1$ defines the critical coverage required to prevent epidemic growth `[luque2020como]`, `[idm2026modelinghub]`:
   $$
   p_c = 1 - \frac{1}{R_0}
   $$
   For example, for high-$R_0$ diseases such as measles or rubella (theoretical $p_c \sim 90\%$), accounting for imperfect $95\%$ vaccine efficacy raises required coverage to $99\%$ and $91\%$, respectively (smallpox remains the only human infectious disease eradicated globally by vaccination); for SARS-CoV-2 with $R_0 \approx 2.5$, $p_c$ lies in the $60\%$–$75\%$ range `[luque2020como]`.

---

### 2.2 Extension to SEIR, SEIRD, and SEIRDS with Vaccination and Waning Immunity

For diseases with a non-negligible latent/incubation period and significant disease-specific mortality, the SIR scheme is extended by adding the **Exposed ($E$)** compartment—infected individuals who do not yet transmit—and the **Deceased ($D$)** compartment `[raimundez2021covid]`, `[sudhi2022scalable]`, `[idm2026modelinghub]`, `[rama2026systematic]`.

Also including a daily vaccination rate $\nu_v$ ($S \to R$) and an immunity waning rate $\xi = 1/T_{\text{imm}}$ ($R \to S$) in a closed population without background demography ($N = S + E + I + R + D$), the **SEIRDS** system is formulated as:

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta \frac{S I}{N} - \nu_v S + \xi R \\
\frac{dE}{dt} &= \beta \frac{S I}{N} - \sigma E \\
\frac{dI}{dt} &= \sigma E - (\gamma + \mu_d) I \\
\frac{dR}{dt} &= \gamma I + \nu_v S - \xi R \\
\frac{dD}{dt} &= \mu_d I
\end{aligned}
$$

where:
* $\beta$: effective transmission rate (in $\text{day}^{-1}$, under standard $\beta S I / N$ incidence).
* $\sigma = 1 / T_{\text{incub}}$: progression rate from the latent state to the infectious state.
* $\gamma$: recovery rate into $R$.
* $\mu_d$: disease-specific mortality rate into $D$ (equivalent, in models parameterized by mean infectious duration $D_{\text{inf}}$ and case fatality probability $\alpha_{\text{fat}}$, to a total outflow rate from $I$ of $1/D_{\text{inf}}$ split into $(1 - \alpha_{\text{fat}})/D_{\text{inf}}$ toward $R$ and $\alpha_{\text{fat}}/D_{\text{inf}}$ toward $D$).

As documented by the Institute for Disease Modeling (IDM) *ModelingHub* `[idm2026modelinghub]`, adding the latent compartment $E$ **slows the initial exponential growth rate** compared to the equivalent SIR model; however, in a closed population without mortality or waning immunity ($\mu_d = \nu_v = \xi = 0$), it leaves both $R_0 = \beta/\gamma$ and the final cumulative attack size unchanged. When waning immunity ($\xi > 0$, SEIRDS/SEIRS model) or demographic replacement is active, susceptibles are continuously replenished and the system admits endemic equilibria accompanied by damped epidemic waves or resurgences `[idm2026modelinghub]`, `[hethcote2000mathematics]`.

---

### 2.3 Generalizing $R_0$ in Complex Systems: Next-Generation Matrix and Nonlinear Dynamics

In models with multiple infectious compartments (e.g., $E$, asymptomatic $I$, symptomatic $I$, multiple risk groups, or spatial patches), the scalar formula $\beta/\gamma$ is no longer sufficient. The standard method to rigorously derive $R_0$ is the **Next-Generation Operator / Matrix (NGM)** formulated by van den Driessche and Watmough `[vandendriessche2002reproduction]` and Diekmann, Heesterbeek, and Roberts `[diekmann2010construction]`:

1. Group infectious compartments into a vector $x \in \mathbb{R}^m$ and decompose their dynamics as $\dot{x}_i = \mathcal{F}_i(x) - \mathcal{V}_i(x)$, where $\mathcal{F}_i$ represents the appearance of *new infections* and $\mathcal{V}_i$ aggregates internal transitions between infectious states and outflows (recovery or death).
2. Evaluate the Jacobians at the **disease-free equilibrium (DFE)** $x_0$:
   $$
   F = \left[\frac{\partial \mathcal{F}_i}{\partial x_j}(x_0)\right], \qquad V = \left[\frac{\partial \mathcal{V}_i}{\partial x_j}(x_0)\right]
   $$
3. The basic reproduction number is defined by the spectral radius of the next-generation matrix $K = FV^{-1}$:
   $$
   R_0 = \rho(FV^{-1})
   $$
   under the DFE stability conditions established in `[vandendriessche2002reproduction]`.

Beyond linear analysis around the DFE, modern literature highlights behaviors that simple models miss `[brauer2017mathematical]`:
* **Backward bifurcations:** In multi-group or structurally asymmetric models, stable endemic equilibria can coexist even when $R_0 < 1$, implying that reducing $R_0$ below $1$ does not always guarantee immediate disease elimination `[vandendriessche2002reproduction]`, `[brauer2017mathematical]`.
* **Early stochastic extinction and superspreading:** Using Galton–Watson branching processes and contact-network generating functions $G_0(z) = \sum_k p_k z^k$, Brauer `[brauer2017mathematical]` shows that even when $R_0 > 1$, there is a positive probability $z_\infty \in (0, 1)$ that an outbreak dies out randomly before becoming a major epidemic, and that individual heterogeneity (superspreading events) slows the initial growth rate relative to the homogeneous-mixing exponential prediction `[brauer2017mathematical]`, `[estrada2020covid]`.

---

## 3. Comparative Taxonomy of Epidemiological Modeling Paradigms

Choosing a modeling paradigm involves a trade-off between analytical tractability, structural realism, data requirements, and computational cost. The following table summarizes the six main paradigms documented in the reviewed sources:

| Paradigm | Mathematical Formalism | Key Strengths | Key Limitations | Representative References |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic Compartmental (SIR / SEIR / SEIRD)** | Coupled ODE systems | Analytically tractable ($R_0$, threshold, $I_{\max}$, attack rate); minimal compute cost; ideal baseline reference | Assumes homogeneous mixing and infinite divisibility; unsuitable for small outbreaks, explicit geography, or individual heterogeneity | `[kermack1927contribution]`, `[hethcote2000mathematics]`, `[brauer2017mathematical]`, `[idm2026modelinghub]` |
| **Stratified & Metapopulation** | Coupled ODEs with contact matrices $C_{aa'}$ and inter-patch mobility | Incorporates age groups, age-specific $\text{IFR}_a$, and geographic flows between regions | High dimensionality; outputs and $R_0$ are highly sensitive to mobility model: **Eulerian** flux (origin-memoryless) vs. **Lagrangian** travel (*Simple Trip*, matrix $\Psi_{ij}$) | `[vandendriessche2002reproduction]`, `[citron2021comparing]`, `[arenas2020derivation]`, `[rama2026systematic]` |
| **Continuous Spatial (Reaction–Diffusion)** | Nonlinear parabolic PDEs + FEM / finite differences | Captures continuous propagation fronts, density-dependent diffusion, Allee effect, and geographic boundaries (Neumann conditions); highly scalable on MPI/GPU | Requires fine geographic meshes, boundary conditions, and complex parallel solvers (e.g., implicit Euler + RAS decomposition + AMG) | `[sudhi2022scalable]` |
| **Stochastic / Branching Processes** | Markov chains, *chain-binomial*, Gillespie, Galton–Watson, negative binomial | Models early random extinction, finite-size variance, intrinsic uncertainty, and superspreading | Requires ensembles of thousands of Monte Carlo replicas; high compute cost if not parallelized | `[brauer2017mathematical]`, `[estrada2020covid]`, `[rama2026systematic]` |
| **Agent-Based (ABM / Microsimulation)** | Discrete automata / particles with spatial or network interaction | Maximum expressiveness for individual heterogeneity, risk groups, urban mobility, and targeted NPIs; represented **47.7%** of spatial COVID-19 studies (vs. **29.4%** for ODE/PDE) | Calibration and identifiability complexity; high memory footprint and naive $O(N^2)$ scaling without $O(N)$ spatial structures | `[ferguson2006strategies]`, `[ferguson2020report9]`, `[oshinubi2026systematic]` |
| **Statistical / Hybrid (GAM, Bayesian, PINN)** | Hierarchical GAM, MCMC, ABC-SMC, Physics-Informed Neural Networks (PINNs) | Strong for reporting-lag *nowcasting* and operational $R_t$ estimation with credible intervals | Weak out-of-regime counterfactual extrapolation; PINNs showed overconfident predictive intervals compared to mechanistic ABC-SMC on noisy series | `[cdc2026cfa]`, `[rama2026systematic]` |

---

## 4. Scientific Foundation of the `seird-Parallel` Architecture

The `seird-Parallel` engine adopts a large-scale hybrid paradigm: a **Stochastic SEIRD Agent-Based Model (ABM) in continuous 2D space** executed massively in parallel on NVIDIA CUDA GPUs, coupled to a CPU statistical telemetry module with OpenMP ([PROJECT_PLAN.md](PROJECT_PLAN.md), [include/types.hpp](../include/types.hpp), [src/cpu_engine.cpp](../src/cpu_engine.cpp)). In terms of the **GPM (Geography–Population–Movement)** taxonomy introduced by Oshinubi et al. `[oshinubi2026systematic]`, the project combines a continuous 2D geographic arena with an acceleration grid, a discrete population stratified by risk (`RiskGroup`), and stochastic movement attracted toward urban activity hubs (`urban_clustering`).

The matrix below links each code component to its scientific foundation:

| Component in `seird-Parallel` | Repository Location | Theoretical & Scientific Basis |
| :--- | :--- | :--- |
| **`AgentState` (`S, E, I, R, D`) and base parameters** (`incubation_days = 5.0f`, `infectious_days = 10.0f`, `fatality_rate = 0.02f`, `dt = 1/48` days) | [include/types.hpp](../include/types.hpp) | Reflects the standard $S \to E \to I \to \{R, D\}$ transition for pathogens with a latent period (`[idm2026modelinghub]`, `[raimundez2021covid]`). Default values match documented magnitudes for SARS-CoV-2 and 1918 influenza: 5–6 day incubation (`[raimundez2021covid]`) and ~2% case fatality (`[luque2020como]`). |
| **`RiskGroup` stratification** (`RISK_LOW` ~50%, $0.25\times$ fatality; `RISK_MEDIUM` ~35%, $1.0\times$; `RISK_HIGH` ~15%, $3.5\times$ fatality) | [include/types.hpp](../include/types.hpp) | Addresses a core limitation of homogeneous SIR noted in `[luque2020como]`, `[hethcote2000mathematics]`, and `[rama2026systematic]`: mobility and infection fatality ratio ($\text{IFR}_a$) vary sharply across age/risk strata. |
| **Continuous 2D space, `urban_clustering`, and $O(N)$ *Spatial Hashing* with `cub::DeviceRadixSort::SortPairs`** | [PROJECT_PLAN.md](PROJECT_PLAN.md), [include/types.hpp](../include/types.hpp), [src/cuda_engine.cu](../src/cuda_engine.cu) | Overcomes the unrealistic homogeneous-mixing assumption (`[luque2020como]`, `[arenas2020derivation]`, `[citron2021comparing]`). A uniform grid with `cell_size >= infection_radius` reduces neighbor lookup from $O(N^2)$ to $O(N)$ in SoA (*Structure of Arrays*) memory, with coalescing physical reordering and empty-infected cell skipping (`[sudhi2022scalable]`, `[oshinubi2026systematic]`). |
| **Tessellated urban layouts (`UrbanZone`), households (`home_id`, `work_zone`), and `infection_source` attribution** | [include/types.hpp](../include/types.hpp), [src/cuda_engine.cu](../src/cuda_engine.cu) | Operationalizes **Lagrangian** home-return mobility (`[citron2021comparing]`, `[arenas2020derivation]`): rectangular 5×5 districts (`PRESET_MONOCENTRIC`, `PRESET_POLYCENTRIC`), day/night and weekly cycles (`dt = 1/48` d), intrahousehold `RiskGroup` mixing, and contagion-setting tracking (`Home, Work, Transit, Hospital, Leisure`). |
| **Interventions (`quarantine_factor`, `focal_quarantine`, `household_quarantine`, `close_schools_work`, `vaccination_rate`, `immunity_days`) and `hospital_capacity_pct`** | [include/types.hpp](../include/types.hpp), [src/cuda_engine.cu](../src/cuda_engine.cu) | Models global, $Z$-score-targeted ($Z > 2.0$), and household-level NPIs to suppress effective contact $\mu$ and peak $I_{\max}$ relative to healthcare capacity (`[luque2020como]`, `[ferguson2020report9]`, `[markel2007nonpharmaceutical]`). Vaccination ($S \to R$) operationalizes $p_c = 1 - 1/R_0$ and `immunity_days` explores endemic waves (`[idm2026modelinghub]`). |
| **Telemetry, Monte Carlo ensembles (`CpuEnsembleEngine`), NGM, RK4 ODE, sensitivity, and ABC-SMC** | [include/cpu_engine.hpp](../include/cpu_engine.hpp), [src/cpu_engine.cpp](../src/cpu_engine.cpp) | Connects GPU microsimulation with aggregate surveillance metrics (`[luque2020como]`, `[cdc2026cfa]`), NGM $R_0$ (`[vandendriessche2002reproduction]`, `[diekmann2010construction]`), probabilistic scoring (`WIS`, `IS80`, `RMSE`, `MAE`), and Approximate Bayesian Computation ABC-SMC (`[rama2026systematic]`, `[raimundez2021covid]`). |

### 4.1 Mathematical Analysis of Telemetry in `CpuTelemetryEngine`

In [src/cpu_engine.cpp](../src/cpu_engine.cpp), `CpuTelemetryEngine::update` computes four indicators from `SimStats` time series:

1. **Daily Incidence Rate (`new_cases_rate`):** Computed directly from the exact GPU atomic counter of $S \to E$ transitions (`stats.new_exposures / elapsed_step`). Measuring true new-exposure flux instead of $\Delta S / \Delta t$ keeps incidence exact even when vaccination (`vaccination_rate > 0`) or waning immunity (`immunity_days > 0`) is simultaneously active.
2. **Effective Reproduction Number ($R_t$) and CDC CFA Categories:** From the sliding window of active cases $A_{\text{act}}(t) = E(t) + I(t)$ with a lag of up to 10 samples, the engine estimates the instantaneous exponential growth rate:
   $$
   r = \frac{\ln\big(E(t) + I(t)\big) - \ln\big(E(t-\Delta t_{\text{lag}}) + I(t-\Delta t_{\text{lag}})\big)}{\Delta t_{\text{lag}}}
   $$
   and obtains $R_t = \max\big(0,\; 1 + r \cdot T_g\big)$ with $T_g \approx T_{\text{incub}} + \frac{1}{2} T_{\text{infec}}$ (first-order Wallinga–Lipsitch / Lotka–Euler approximation). Additionally, via bootstrap resampling over recent rates (and empirically across $K$ replicas in `CpuEnsembleEngine`), it computes `[rt_p05, rt_p95]`, $\mathbb{P}(R_t > 1)$, and assigns the standard CDC CFA `RtCategory` `[cdc2026cfa]` (*Declining* $<10\%$, *Likely Declining* $<25\%$, *Uncertain*, *Likely Growing* $>75\%$, *Growing* $>90\%$).
3. **Cumulative Fatality Ratios (`cfr` vs. `ifr`):** The engine explicitly separates **CFR** over resolved cases $\text{CFR} = D / (R + D)$ from true **IFR** over cumulative infections $\text{IFR} = D / \text{cum\_infections\_}$ (accumulated from `new_exposures`, avoiding distortion from vaccination into $R$ or reinfections $R \to S$), aligned with `[luque2020como]`, `[raimundez2021covid]`, and `[rama2026systematic]`.
4. **Outbreak Alert (`outbreak_alert`):** Triggered when `((z_score > 2.0f) || (stats.max_cell_z > 2.0f)) && (rt > 1.0f)`, combining a statistical temporal or spatial anomaly criterion ($> 2\sigma$ above local mean) with sustained epidemic growth ($R_t > 1$).

---

## 5. Methodological Rigor: Calibration, Identifiability, Uncertainty, and Verification

For a parallel epidemiological simulator to be scientifically defensible rather than purely illustrative, the reviewed literature establishes four core methodological requirements:

### 5.1 Practical Identifiability and Aggregate Data Bias
Raimúndez et al. `[raimundez2021covid]` evaluated SEIRD model variants against early public Wuhan time series and demonstrated that aggregate counts of infected, recovered, and deceased individuals are often insufficient to uniquely identify internal model parameters. In particular:
* Bayesian MCMC with weakly informative bounded priors produced artificially narrow 99% credible intervals at biologically implausible values (e.g., incubation periods of $[23.4,\; 257.6]$ days), whereas frequentist **profile likelihood** analysis correctly exposed the underlying non-identifiability.
* Arbitrarily fixing unobserved initial conditions (such as $E_0$) introduces severe bias into estimated progression rates `[raimundez2021covid]`.

### 5.2 Predictability Limits and Heterogeneity Effects
Estrada `[estrada2020covid]` documents that during the exponential growth phase, small parameter variations cause rapid divergence in posterior predictive intervals (spanning from extinction to sustained growth), making quantitative peak and inflection-point forecasts reliable only over short horizons. Likewise, Brauer `[brauer2017mathematical]` emphasizes that individual contact-rate heterogeneity (superspreading) substantially alters early extinction probability and contact-tracing effectiveness compared to homogeneous mixing.

### 5.3 Stochastic Calibration, Hybrid Surrogates, and Decision Frameworks
Rama et al. `[rama2026systematic]` compared an age-structured stochastic SEIRD model calibrated via **Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC)** against Physics-Informed Neural Networks (PINNs) calibrated via MCMC. On synthetic data with negative-binomial observation noise, the mechanistic stochastic approach consistently outperformed PINNs in point accuracy and probabilistic coverage (MAE, RMSE, WIS, IS80), whereas PINNs tended to produce overconfident (excessively narrow) intervals. Meanwhile, the WHO/OECD/World Bank interagency framework `[who2024integrated]`, the IDM *ModelingHub* `[idm2026modelinghub]`, and the CDC CFA *Modeling Handbook* `[cdc2026cfa]` all emphasize iterative calibration with explicit uncertainty quantification, multi-model ensembles, and transparency regarding structural assumptions when interventions change.

### 5.4 Verification, Invariants, and Empirical Scientific Validation in `seird-Parallel`
The systematic review of 109 spatial studies by Oshinubi et al. `[oshinubi2026systematic]` revealed a major reproducibility gap: only **30.6%** made their source code openly available. In `seird-Parallel`, verification covers both structural invariants and quantitative validation against published literature:

1. **Invariant Suite ([tests/test_invariants.cpp](../tests/test_invariants.cpp), 12 tests):**
   * **Mass Conservation and Telemetry:** Validates step-by-step $S + E + I + R + D = N$ in the baseline model, under interventions, in Monte Carlo ensembles (`K = 8`), and in urban layouts (`PRESET_MONOCENTRIC`, `household_quarantine`), along with numerical finiteness of all metrics.
   * **Domain, Memory, and Reproducibility Consistency:** Verifies dynamic VRAM limits, the geometric invariant `cell_size >= infection_radius`, and bit-identical reproducibility for a given seed.
   * **Analytical Validation, Sensitivity, and ABC-SMC:** Verifies ABM vs. RK4 ODE convergence in the high-diffusion limit, $R_0(\text{NGM})$ across the 3 `RiskGroup` strata (`[vandendriessche2002reproduction]`, `[diekmann2010construction]`), probabilistic scores (`WIS`, `IS80`, `RMSE`, `MAE`, coverages), global sensitivity sweeps (`run_global_sensitivity_sweep`), and stochastic calibration (`calibrate_abc_smc`, `[rama2026systematic]`).

2. **Scientific Benchmark Suite ([tests/test_scientific_benchmarks.cpp](../tests/test_scientific_benchmarks.cpp), 8 experiments):**
   * **Analytical Thresholds (`[kermack1927contribution]`, `[luque2020como]`):** For $R_0 \in \{1.5,\; 2.5,\; 10.0\}$, the adaptive sub-stepping RK4 solver reproduces the final attack rate and $I_{\max}/N$ (error `1.44e-5` at $R_0=2.5$ and `4.54e-9` at $R_0=10.0$) and confirms exact containment at the herd immunity threshold $p_c = 1 - 1/R_0$.
   * **Latent Period Invariance (`[idm2026modelinghub]`):** Comparing SIR vs. SEIR ($T_{\text{incub}} = 5\text{ d}$, $R_0 = 2.5$), compartment $E$ delays the peak from day `53.5` to `98.7` and lowers it from `23.35%` to `15.38%`, while leaving the final attack rate unchanged (`0.8927`, difference `4.03e-5`).
   * **1918 Influenza NPIs (`[markel2007nonpharmaceutical]`):** On GPU ABM (`N = 20,000`), early sustained NPIs reduce peak infection from `27.52%` to `2.04%` and mortality from `4.12%` to `0.40%`, whereas lifting restrictions prematurely at day 25 triggers a **second infectious peak** (`24.68%`).
   * **Endemic Waves (`[hethcote2000mathematics]`):** With waning immunity ($T_{\text{imm}} = 35\text{ d}$, $R_0 = 2.8$), the SEIRDS system exhibits damped oscillations converging to an endemic equilibrium (`9.97%`).
   * **Early Stochastic Extinction & Superspreading (`[brauer2017mathematical]`, `[estrada2020covid]`):** With $I_0 = 3$ and $R_0 > 1$ (`K = 32`), homogeneous mixing yields `53.1%` early extinction versus `78.1%` with superspreaders (`10% × 6`).
   * **Urban Topology (`[citron2021comparing]`):** Under identical parameters, the polycentric city lowers and delays the peak compared to the monocentric city (`39.43%` at day `33.0` vs. `46.91%` at day `29.0`).
   * **Practical Identifiability (`[raimundez2021covid]`):** `calibrate_abc_smc` accurately recovers $\beta$ in 1-parameter estimation (`0.259` vs. `0.280`), whereas joint 3-parameter estimation exhibits parameter compensation (`0.353 [0.213–0.474]`) with lower `WIS` (`0.0050`).

---

## 6. Bibliographic References

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
12. `[luque2020como]` Luque, B., Ballesteros, F., & Miramontes, O. (2020). Cómo modelizar una pandemia. *Investigación y Ciencia*, May 2020, 52–60.
13. `[estrada2020covid]` Estrada, E. (2020). COVID-19 and SARS-CoV-2. Modeling the present, looking at the future. *Physics Reports*, 869, 1–51. DOI: [10.1016/j.physrep.2020.07.005](https://doi.org/10.1016/j.physrep.2020.07.005) · [PMC7386394](https://pmc.ncbi.nlm.nih.gov/articles/PMC7386394/).
14. `[raimundez2021covid]` Raimúndez, E., Dudkin, E., Vanhoefer, J., Alamoudi, E., Merkt, S., Fuhrmann, L., Bai, F., & Hasenauer, J. (2021). COVID-19 outbreak in Wuhan demonstrates the limitations of publicly available case numbers for epidemiological modeling. *Epidemics*, 34, 100439. DOI: [10.1016/j.epidem.2021.100439](https://doi.org/10.1016/j.epidem.2021.100439) · [PMC7845523](https://pmc.ncbi.nlm.nih.gov/articles/PMC7845523/).
15. `[citron2021comparing]` Citron, D. T., Guerra, C. A., Dolgert, A. J., Wu, S. L., Henry, J. M., Sánchez C., H. M., & Smith, D. L. (2021). Comparing metapopulation dynamics of infectious diseases under different models of human movement. *Proceedings of the National Academy of Sciences (PNAS)*, 118(18), e2007488118. DOI: [10.1073/pnas.2007488118](https://doi.org/10.1073/pnas.2007488118) · [PMC8106338](https://pmc.ncbi.nlm.nih.gov/articles/PMC8106338/).
16. `[sudhi2022scalable]` Sudhi P. V., Dolean, V., Jolivet, P., Robinson, B., Edwards, J. D., Kendzerska, T., & Sarkar, A. (2022). Scalable Computational Algorithms for Geo-spatial Covid-19 Spread in High Performance Computing. *arXiv:2208.01839*. DOI: [10.48550/arXiv.2208.01839](https://doi.org/10.48550/arXiv.2208.01839) · [arXiv](https://arxiv.org/abs/2208.01839).
17. `[who2024integrated]` World Health Organization, OECD, & World Bank (2024). *Strengthening Pandemic Preparedness and Response Through Integrated Modelling*. World Health Organization.
18. `[idm2026modelinghub]` Institute for Disease Modeling (2026). *ModelingHub Documentation: Compartmental Models and Calibration*. [docs.idmod.org/modeling-hub](https://docs.idmod.org/modeling-hub/).
19. `[cdc2026cfa]` U.S. Centers for Disease Control and Prevention, Center for Forecasting and Outbreak Analytics (2026). *Modeling Handbook and Time-Varying Reproduction Number ($R_t$) Technical Methodology*. [cdc.gov/cfa-modeling-and-forecasting](https://www.cdc.gov/cfa-modeling-and-forecasting/modeling-handbook/index.html).
20. `[oshinubi2026systematic]` Oshinubi, K., Chen, Y., Doerry, E., Gel, E. S., Hepp, C., Lant, T., Mehrotra, S., Sabo, S., & Mihaljevic, J. (2026). A systematic review of spatial epidemiological modeling approaches applied during the COVID-19 pandemic. *BMC Public Health*, 26, 1691. DOI: [10.1186/s12889-026-27321-4](https://doi.org/10.1186/s12889-026-27321-4) · [PMC13200481](https://pmc.ncbi.nlm.nih.gov/articles/PMC13200481/).
21. `[rama2026systematic]` Rama, M., Li, Y., Lepri, B., Perra, N., & Tizzoni, M. (2026). A systematic comparison of physics-informed neural networks and mechanistic models for epidemic modeling and forecasting. *arXiv:2611.18389*. [arXiv](https://arxiv.org/abs/2611.18389).
