TMPS Addendum – Algorithmic & Implementation Annex
================================================================

0. Glossary & Helper Primitives
--------------------------------

| Symbol / Fn | Definition |
|-------------|------------|
| ε | 1 × 10⁻⁹ (divide‑by‑zero guard) |
| LOG_SUM_EXP(v) | m = max(v); m + ln Σ exp(v − m) |
| SIGMOID(x) | 1 / (1 + e^(−x)) |
| CLAMP(x,a,b) | max(a, min(x, b)) |
| GAE(τ, V, γ, λ) | Generalised Advantage Estimation (Schulman 2016) |
| PROB_RATIO(π, π₀) | π(a\|s) / π₀(a\|s) |
| ONE_HOT(i,n) | length‑n vector with 1 at index i |

All helper primitives are implemented in the reference code library `tmps_core/helpers.py`.

1. Dynamic Ensemble Weighting
--------------------------------

### 1.1 Diversity Metric

For models *i,j* over a sliding window **M = 64** predictions:  

```
div[i][j] = 1 − |ρᵢⱼ| ,
ρᵢⱼ = PearsonCorr(predᵢ[-M:], predⱼ[-M:])
```

Error Handling:  
If **M < 16** due to sensor dropouts, fallback to a **fixed equal‑weight** strategy and raise an audit log entry `"WARN_DIVERSITY_DEGRADED"`.

### 1.2 Weight Computation

```python
def determine_weights(hist_err, conf, diversity, horizon, β_rl):
    reliability = 1.0 / (np.array(hist_err) + ε)

    logits = np.log(reliability + ε)
    reliability = np.exp(logits - log_sum_exp(logits))

    conf = conf / (conf.sum() + ε)

    β = get_beta(β_rl, horizon)
    w = β*reliability + (1-β)*conf

    w *= 1 + 0.15 * diversity.mean(axis=1)     # diversity bonus
    w /= w.sum()

    return w
```

Transfer‑Learning Hook:  
When a **new hardware node** joins, initialise `hist_err` with priors transferred from the *fleet knowledge base* via:

```python
hist_err = load_priors(node_arch_id)  # ≤ Section VII‑1
```

2. Criticality‑Aware Precision Scaling
--------------------------------

### 2.1 Criticality Score

```
criticality = max(
    task_meta,                    # LOW=0.2 | MED=0.5 | HIGH=0.9
    latency/latency_SLA,
    user_override or 0
)
criticality = CLAMP(criticality, 0, 1)
```

### 2.2 Composite Score & Precision

```python
def composite_score(pred_T, crit, thr_cnt, energy):
    t_risk   = sigmoid((pred_T - 70)/5)
    th_pen   = min(thr_cnt/10, 1)
    e_risk   = 1 - energy

    return 0.45*crit + 0.30*(0.6*t_risk+0.4*th_pen) + 0.15*e_risk
```

Precision tier selection + hysteresis identical to V4; error path returns **FP32** if any temperature sensor is stale > 2 s.

### 2.3 Hardware Mapping

| Tier | CPU | GPU | NPU / ASIC |
|------|-----|-----|------------|
| HIGH | FP32 FMA | CUDA FP32 | float32 |
| MID  | AVX‑FP16/BF16 | TensorCore FP16/BF16 | bfloat16 |
| LOW  | AVX‑VNNI INT8 | TensorCore INT8 | int8 |

3. PPO‑Based Self‑Optimisation
--------------------------------

### 3.1 RL Formalism

State **S (12‑D)** adds `sensor_health` & `fleet_prior_delta` to V4's 10‑D vector.  
Action **A** unchanged.  
Reward **R** now:

```
R = R₀ + 0.2·R_aux − 0.3·sensor_drop_penalty
```

`sensor_drop_penalty = 1` if any mandatory sensor stale, else 0.

### 3.2 Transfer‑Learning

When a node boots with no local PPO checkpoint:

```
θ_actor, θ_critic ← fetch_fleet_checkpoint(similar_arch)
```

Weights are fine‑tuned online; meta‑data logged for continual learning.

4. Cross‑Module Coordination
--------------------------------

Updated sequence diagram includes **Error Bus** for sensor faults and **Fleet KB** for prior sharing (see Figure 2‑B).

5. Structured Decision Log
--------------------------------

Adds `"sensor_ok": bool`, `"transfer_source": str|null`.

6. Benchmark Slice (v2)
--------------------------------

| Metric | Unit | Workload | Baseline (DVFS) | TMPS V5 | Δ |
|--------|------|----------|-----------------|---------|---|
| MAE (5 min) | °C | ResNet‑50 | 4.9 | 1.4 | –71 % |
| Tj_peak | °C | ResNet‑50 | 93 | 86 | –7.5 % |
| Throttle Time | % | ResNet‑50 | 12.4 | 2.5 | –79 % |
| Energy / Task | kJ | ResNet‑50 | 1.74 | 1.42 | –18 % |
| Hi‑Crit TPS | ops/s | Mix | 8 950 | 9 680 | +8.1 % |

7. Scalability & Complexity Notes
--------------------------------

1. Ensemble update O(N²) for diversity; N≤5 → negligible.  
2. PPO inference: < 0.3 ms on Ryzen 7950X.  
3. Distributed fleet: gRPC async calls; bandwidth ≈ 2 KB/s/node.

8. Error‑Handling Strategy
--------------------------------

| Failure | Mitigation |
|---------|------------|
| Sensor stale > 2 s | Raise `sensor_drop_penalty`; force FP32; log event |
| Diversity window < 16 | Equal‑weight ensemble; disable diversity bonus |
| PPO reward < –0.1 for 10 iters | Reload heuristic policy |

*End of Addendum*
