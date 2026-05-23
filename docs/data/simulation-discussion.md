# Simulation Discussion — Noise & Anomaly Design

> This document captures the full discussion about improving the welding robot data simulator's realism, covering noise model refinements and additional anomaly types for the test set.

---

## 1. Noise Model Improvements

### 1.1 Problem: Current White Noise Is Unrealistic

The current `physics/noise.py` uses **independent white noise** for S0 (wire_feed_speed) and S1 (welding_current). Each time step is sampled independently with no memory of previous values:

```python
# S1 — add_current_noise: purely i.i.d. Gaussian
noise = self._rng.normal(0, s1_arc_noise_std * signal_amplitude, n)
return signal + noise
```

This produces the **"Bad" pattern** where values jump erratically:

```
250 → 252 → 248 → 255 → 256 → 247
```

In real industrial sensors, noise has **temporal autocorrelation** (momentum).

### 1.2 Proposed Fix: Momentum Noise + Local Window Variance

Two key changes:

**A. Momentum (AR-1 style noise):** Each step depends strongly on the previous step plus a small random innovation, producing the **"Good" pattern**:

```
250 → 251 → 251 → 252 → 252 → 252 → 255 → 255 → 254 → 251 → 249
```

**B. Local window variance:** Instead of computing `signal_amplitude` globally (`np.mean(np.abs(signal))`), compute it from a **rolling window** (e.g., ±10–20 steps). This captures phase-dependent variance naturally:
- **Idle phase:** noise variance ≈ 0 (just electrical floor)
- **Rampup:** variance grows as the process becomes active
- **Active:** variance is steady
- **Winddown:** variance shrinks

| Current signal | Current model | Problem | Proposed fix |
|---|---|---|---|
| **S0** (wire_feed_speed) | Sinusoid + i.i.d. Gaussian | No momentum in Gaussian part | AR(1) noise + local-window std |
| **S1** (welding_current) | Pure i.i.d. Gaussian | No momentum at all | AR(1) process with phase-dependent variance |
| **S2** (temperature) | Random walk + glitches | Already has momentum, but uses global amplitude | Local-window variance for random walk step size |

### 1.3 U-Pattern for Stable Sensors

Sensors controlled by digital setpoints exhibit **multi-state noise**:

```
0.5 → 0.6 → 0.6 → 0.5 → 0.5 → 0.501 → 0.5
                         ↓
0.2 → 0.2 → 0.2 → 0.2 → ...
```

This is characteristic of:
- Digitally-controlled setpoints (DAC outputs that stay constant, then jump)
- Quantized sensors with finite resolution
- Hysteresis-controlled systems (thermostat-like deadband behavior)

Can be modeled as a **regime-switching process** or **discrete-state Markov chain**.

---

## 2. Additional Smoothing & Realism Ideas

### 2.1 S0 — wire_feed_speed (mechanical)

| # | Idea | Why realistic |
|---|---|---|
| 1 | **Micro-stutter at ramp transitions** — Add 3–5 steps of tiny hesitation (speed dips ~5%) before motor overcomes static friction. | Real motors have **stiction** — static friction > kinetic friction, causing micro-stall at startup. |
| 2 | **Gear meshing ripple** — Superimpose low-frequency, low-amplitude sinusoid (period ~10–20 steps) on the active phase only. | Gear teeth engagement creates periodic torque ripple. |
| 3 | **Encoder quantization dither** — Round feed speed to discrete grid (±0.5 mm/min) with small hysteresis (value sticks at a level for 2–4 steps). | Real encoders have **finite resolution** — no infinite-precision floats. |

### 2.2 S1 — welding_current (electrical arc)

| # | Idea | Why realistic |
|---|---|---|
| 4 | **Arc re-ignition micro-oscillations** — Add decaying 5–8 Hz oscillations persisting ~20–30 steps into active phase, after the initial arc strike. | Arc plasma column **rings** at natural resonance after ignition. |
| 5 | **Dropout micro-events (droplet transfer)** — Inject 1–2 step mini-dips (20–30% drop) at random intervals during active phase, with smooth exponential recovery over 5–8 steps. | Real welding has **droplet transfer events** — current dips briefly when molten droplet detaches. |
| 6 | **Cross-weld variability envelope** — Each welding cycle should have slightly different mean current (±2–3%) even for same program, drawn from a random distribution. | Real welding varies per cycle due to **part fit-up, gas flow, material thickness**. |

### 2.3 S2 — temperature (thermal)

| # | Idea | Why realistic |
|---|---|---|
| 7 | **Multi-time-constant thermal dynamics** — Add a second, slower exponential (τ₂ ≈ 5× τ₁) for deep thermal soak of the robot arm structure. | Real robots have **thermal mass at multiple scales** — weld tip heats fast, arm structure heats slowly across cycles. |
| 8 | **Inter-cycle thermal accumulation** — Temperature at start of cycle N+1 depends on end temperature of cycle N (baseline drifts up 5–10°C over a busy day). | Real robots **heat up over a shift** — baseline creeps upward. |
| 9 | **Sensor thermal lag** — Add 2–4 step delay between true temperature and reported value (thermocouple sheath response time). | Thermocouples have **response time** — they don't instantly read true temperature. |

### 2.4 Cross-sensor (all 3 signals)

| # | Idea | Why realistic |
|---|---|---|
| 10 | **Correlated burst noise** — 1 in 2000 steps, all 3 sensors jump simultaneously for 1–2 steps (common-mode EMI event). | Factory floor has **electromagnetic interference** affecting all sensors at once. |
| 11 | **Drift in noise variance over a day** — S1 noise +20% in first hour (cold start) and +10% in last hour (electrode wear). | Electrode condition changes over time. |
| 12 | **Random missing-tick interpolation** — 1 in 5000 steps, repeat previous value with tiny interpolation jitter (simulating dropped PLC sample). | Real PLC data acquisition occasionally **drops a sample** due to bus timing. |

---

## 3. Anomaly Type Analysis

### 3.1 How MaskFormer Detects Anomalies

MaskFormer works by **masking one sensor** and **reconstructing it from the others** using learned causal relationships:

- **S1 (current)** → predictable from **S0 (feed speed)** via LTI overshoot dynamics
- **S2 (temperature)** → predictable from **S1 (current)** via exponential charge curve
- **S0 (feed)** → the actuator; less dependent on others, but phase pattern is self-predictable

**Anomalies that break these causal links** are the most detectable.

### 3.2 Existing 4 Anomalies — Detection Difficulty

| # | Name | Sensor | Pattern | MaskFormer Detection | Difficulty |
|---|---|---|---|---|---|
| 1 | **Spike** | S1 spike, S0 normal | Impulse | **Easy** — Mask S1 → reconstruct from S0 → huge reconstruction error at spike | Easy |
| 2 | **Thermal Drift** | S2 linear rise | Trend shift | **Medium** — Mask S2 → reconstruct from S1 → error grows over time because S1 is normal but S2 keeps rising | Medium |
| 3 | **Flatline** | S0=0 for 20 steps | Step dropout | **Medium-Hard** — Mask S0: hard to reconstruct S0 from S1 alone (actuator is cause, not effect). But S0's flat pattern is itself abnormal vs. normal phase envelope | Medium-Hard |
| 4 | **Phase Shift** | S1 delayed 15 steps | Temporal shift | **Hard** — Mask S1 → reconstruct from S0: S0 rises but S1 doesn't → error during delay window. For a short 15-step delay, this is subtle | Hard |

### 3.3 Six New Anomaly Proposals

#### Anomaly 5: Wire Stutter / Feed Motor Jerk (S0)

| Aspect | Detail |
|---|---|
| **Physical cause** | Feed motor bearing wear — motor jerks instead of smooth rotation |
| **Pattern** | S0 shows repeated micro-dips (5–10% drop) every 15–25 steps during active phase. Each dip recovers with a small overshoot. S1 follows S0's dips with LTI dynamics (slightly smoothed). S2 barely notices. |
| **Why realistic** | Worn motor bearings cause periodic torque loss — very common in high-cycling robots |
| **MaskFormer detection** | **Medium** — Mask S0: S0's clean phase envelope is predictable, but the stutter pattern repeats at an interval that doesn't match any normal profile. The model sees S1 dipping **after** S0 dips → temporal lag is correct, but periodic repetition is wrong. |
| **Signal shape** | `normal_value → drop 8% over 2 steps → hold 3 steps → recover with 3% overshoot → stabilize → repeat` |

#### Anomaly 6: Arc Blow / Magnetic Deflection (S1)

| Aspect | Detail |
|---|---|
| **Physical cause** | Magnetic field from nearby equipment deflects the arc |
| **Pattern** | S1 noise amplitude increases 3–5× above normal during a window (30–60 steps in active phase). S0 is normal. S2 shows slightly lower temperature (less efficient heating). |
| **Why realistic** | Common when welding near strong magnets or with poor ground clamp placement |
| **MaskFormer detection** | **Easy** — Mask S1 → reconstruct from S0. S0 is normal so predicted S1 is normal, but actual S1 has 5× noise → huge reconstruction error. The **noise variance itself** becomes a detectible feature. |
| **Signal shape** | `normal S1 → S1 noise std jumps to 5× for 40 steps → returns to normal` |

#### Anomaly 7: Gas Nozzle Blockage / Contaminated Shielding Gas (S1 + S2)

| Aspect | Detail |
|---|---|
| **Physical cause** | Weld spatter blocks gas nozzle → poor shielding |
| **Pattern** | Dual sensor signature: (1) S1 develops intermittent micro-spikes (1-step spikes, 2× normal) every 5–10 steps, AND (2) S2 runs 3–5°C hotter than expected for the given current (exothermic oxidation from contaminated gas). |
| **Why realistic** | Most common weld defect in production — spatter buildup happens every 50–100 welds |
| **MaskFormer detection** | **Easy** — Two independent signals both deviate: Mask S1 → S0 can't predict micro-spikes. Mask S2 → S1 can't explain extra heat. Dual evidence makes detection robust. |
| **Signal shape** | `S1: normal + random 1-step 2× spikes; S2: +3-5°C offset above equilibrium` |

#### Anomaly 8: Thermal Sensor Stuck / Sticky Thermocouple (S2)

| Aspect | Detail |
|---|---|
| **Physical cause** | Thermocouple degrades — reading sticks at a constant value then jumps |
| **Pattern** | S2 goes flat (repeats same value for 15–40 steps) then **jumps** 2–5°C instantly to a new value, goes flat again. Repeats 2–3 times. Unlike normal smooth exponential, it's a staircase. |
| **Why realistic** | Thermocouples in high-vibration environments develop intermittent contact |
| **MaskFormer detection** | **Medium** — Mask S2 → reconstruct from S1. Predicted S2 is a smooth exponential, but actual S2 is a staircase. Error is high at the **jump points** and moderate during flat regions. S2's flat segments have **zero variance** which is unrealistic for a noisy sensor. |
| **Signal shape** | `smooth temp → flat at 85°C for 20 steps → jump to 87°C → flat 25 steps → jump to 89°C` |

#### Anomaly 9: Cable Wear / Intermittent Short (S1 Dropout)

| Aspect | Detail |
|---|---|
| **Physical cause** | Welding cable insulation wears through → intermittent ground short |
| **Pattern** | S1 randomly drops to near-zero for 1–3 steps, 8–15 times during the active phase. Each dropout has a sharp drop and sharp recovery (no exponential decay — it's an electrical short, not mechanical). S0 is completely normal. |
| **Why realistic** | Cable wear is the #1 maintenance issue in production welding cells |
| **MaskFormer detection** | **Easy-Hard mix** — Single dropouts (1 step): hard to distinguish from normal noise. Multiple dropouts (clusters of 3+): easy because repeated sharp drops pattern is not in normal data. To hit 95% F1, make 60% "obvious clusters" and 40% "subtle singles." |
| **Signal shape** | `S1: 150A → 5A (1 step) → 150A → 148A → 3A (2 steps) → 152A ... (8–15 events)` |

#### Anomaly 10: Progressive Overshoot Degradation (S1 Shape Change)

| Aspect | Detail |
|---|---|
| **Physical cause** | Welding power supply capacitor aging — arc ignition overshoot changes |
| **Pattern** | The normal overshoot at arc ignition (currently ~12% with ζ=0.55) either: (a) **disappears** (ζ→1.0, critically damped — no overshoot), or (b) **becomes excessive** (ζ→0.3, 30% overshoot with prolonged ringing for 20+ steps). S0 is normal. S2 follows the changed S1. |
| **Why realistic** | Power supply filter capacitors degrade over thousands of cycles — damping ratio shifts |
| **MaskFormer detection** | **Hard** — This is a **shape anomaly**, not a magnitude anomaly. Mask S1 → S0 predicts the normal overshoot shape, but actual S1 has wrong damping. Error is concentrated in the first 20 steps after rampup. Tests whether MaskFormer learned the **temporal dynamics** (overshoot shape) or just the **steady-state mapping**. |
| **Signal shape** | `Normal: 12% overshoot, settles in 10 steps. Anomaly: 0% overshoot (smooth rise) OR 30% overshoot, rings for 25 steps` |

### 3.4 Complete Anomaly Matrix (All 10)

| # | Name | Sensor(s) | Pattern | Difficulty | Purpose |
|---|---|---|---|---|---|
| 1 | Spike | S1 | Impulse 4× | Easy | Baseline — high recall |
| 2 | Thermal Drift | S2 | Linear rise | Medium | Tests trend detection |
| 3 | Flatline | S0 | Step=0 | Medium-Hard | Tests actuator-side detection |
| 4 | Phase Shift | S1→S0 | Delay | Hard | Tests temporal attention |
| 5 | Wire Stutter | S0 | Periodic dips | Medium | Tests periodic pattern detection |
| 6 | Arc Blow | S1 | 5× noise burst | Easy | High SNR — easy recall boost |
| 7 | Gas Blockage | S1+S2 | Dual signature | Easy | Dual evidence — robust |
| 8 | Sticky TC | S2 | Staircase | Medium | Tests zero-variance detection |
| 9 | Cable Wear | S1 | Dropout cluster | Easy-Hard | Mix of subtle + obvious |
| 10 | Overshoot Degradation | S1 | Shape change | Hard | Tests dynamic shape learning |

---

## 4. Recommended Test Set Distribution for 95% F1

To reach 95% F1, the test set should mix easy (high recall) and hard (push model boundaries) anomalies:

| Category | Anomalies | % of test set | Purpose |
|---|---|---|---|
| **Easy** (obvious pattern) | 1, 6, 7 | 15% each = 45% | Guarantee floor recall |
| **Medium** (requires context) | 2, 5, 8 | 10% each = 30% | Core test cases |
| **Hard** (subtle / shape) | 3, 4, 9, 10 | 5–10% each = 25% | Push model to learn true causal dynamics |

**Rationale:**
- Even a weak model gets ~70–80% recall from easy anomalies alone
- A good model reaches ~90% by adding medium ones
- Only a model that truly learned the **causal dynamics** (LTI overshoot, exponential thermal) can detect the hard ones → this is where 90% → 95% comes from
