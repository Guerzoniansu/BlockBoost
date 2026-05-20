# Experimental Design for Rigorous Evaluation of BlockBoost

---

## Phase 0 — Preliminaries and Baselines

Before running a single boosting experiment, establish ground truth.

**0.1 Fix the data generating process completely.** For each experimental condition, fix the random seed, document T, n_features, ar_coeffs, noise_std, sigma_y, alpha_markov, and η. Every result must be exactly reproducible.

**0.2 Characterize the DGP analytically.** Compute the spectral radius ρ of your AR(5) companion matrix, the theoretical β-mixing rate, the Bayes risk under your label noise construction, and the theoretical optimal block size a_T* from the formula derived earlier. These are your theoretical anchors — every empirical result gets interpreted against them.

**0.3 Establish weak learner baseline.** For each (DGP, η) combination, train a single depth-1 stump and record clean test accuracy, noisy test accuracy, and noisy train accuracy. You need clean accuracy strictly between 0.55 and 0.80. If outside this range, adjust DGP parameters before proceeding. **Do not run any boosting experiment until this condition is satisfied.**

---

## Phase 1 — Controlled Single-Factor Experiments

Run each experiment independently, varying one factor at a time with all others fixed at reference values. Reference configuration: T=2000, block_size=a_T*, η=0.1, M=1500 rounds, 20 independent seeds.

### Experiment 1.1 — Noise rate η

Vary η ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.

**Primary question:** At what η does AdaBoost collapse? Does BlockBoost maintain its plateau across all η? Does the empirical plateau value match the theoretical η + (1-2η)R* from Lemma 1.10?

**Measurements per (algorithm, η, seed):**
- Final train accuracy (noisy)
- Final test accuracy (noisy)
- Final test accuracy (clean)
- Round at which collapse occurs for AdaBoost (if any)
- Empirical plateau value averaged over rounds 1000–1500

**Expected result:** A phase transition in AdaBoost at some η* ∈ (0.1, 0.2). BlockBoost plateau should track η + (1-2η)R* monotonically.

---

### Experiment 1.2 — Block size a_T

Vary a_T ∈ {2, 5, 10, 15, 20, 30, 50} with η=0.1 fixed.

**Primary question:** Does the empirical optimal block size match the theoretical a_T* derived from the mixing rate? Is there a U-shaped loss curve in block size?

**Measurements:**
- Final clean test accuracy
- Final noisy test accuracy
- Stability metric: standard deviation of test accuracy over rounds 1000–1500
- Empirical ε* (minimum weighted block error observed during training) — compare against theoretical ε*(1-ε*)^(a_T-1) = 2^(-a_T)

**Expected result:** Performance peaks near a_T* ≈ 15–20, degrades for very small (insufficient regularization) and very large (too few blocks, poor concentration) block sizes.

---

### Experiment 1.3 — Sample size T

Vary T ∈ {500, 1000, 2000, 5000, 10000} with η=0.1 and a_T=a_T*(T) updated accordingly.

**Primary question:** Does the generalization bound from Theorem 1.13 tighten as T grows? Does BlockBoost's advantage over AdaBoost persist at all sample sizes?

**Measurements:**
- Generalization gap: |train accuracy - test accuracy| at convergence
- Collapse rate for AdaBoost: fraction of seeds where collapse occurs
- Compare empirical gap against theoretical bound 2R_{K_T}(H,μ) + sqrt(a_T log(2/δ)/2T)

---

### Experiment 1.4 — Number of boosting rounds M

Vary M ∈ {100, 200, 500, 1000, 1500, 2000, 3000} with η=0.1.

**Primary question:** Does BlockBoost's plateau hold indefinitely as M → ∞? Does AdaBoost's collapse time scale with M or is it a fixed-round phenomenon?

**This directly tests Corollary 1.12** — the lim sup statement requires M → ∞ behavior.

---

## Phase 2 — Robustness to DGP Misspecification

These experiments test whether BlockBoost's advantage is specific to your DGP or general.

### Experiment 2.1 — Varying AR order

Test DGPs: AR(1), AR(2), AR(5), AR(10).

For each, recompute ρ, a_T*, and run Experiment 1.1. The question is whether the phase transition in AdaBoost sharpens as AR order increases (more complex temporal dependence).

### Experiment 2.2 — Non-stationary DGP

Introduce a structural break at T/2: change AR coefficients mid-sequence. This tests the 2SDM motivation from your introduction. BlockBoost should degrade gracefully; AdaBoost should be more sensitive.

### Experiment 2.3 — Markov noise intensity

Vary alpha_markov ∈ {0.0, 0.3, 0.5, 0.7, 0.9} with η fixed. Higher alpha_markov means temporally clustered label flips — the setting where your theory predicts BlockBoost's block averaging is most protective. This is Option D from your earlier design and is theoretically the most compelling stress test.

---

## Phase 3 — Direct Theory Verification

These experiments are designed to confirm or falsify specific theoretical claims.

### Experiment 3.1 — Verify Proposition 1.11 (bounded step size)

Record α_m at every boosting round for both algorithms. Compute the theoretical upper bound (1/2)log((1-ε*)/ε*) from Proposition 1.11.

**Test:** Is max_m(α_m) for BlockBoost always below the theoretical bound? Does AdaBoost's α_m diverge before collapse?

### Experiment 3.2 — Verify Lemma 1.10 (plateau value)

For each (η, a_T) combination, compute the theoretical plateau η + (1-2η)R* where R* is estimated from a clean run (η=0). Compare against empirical plateau of BlockBoost averaged over seeds 1000–1500.

**Test:** Does |empirical plateau - theoretical plateau| decrease as T increases? This would confirm the √(log(2/δ)/2T) convergence rate.

### Experiment 3.3 — Verify Theorem 1.13 (generalization bound)

For each T, compute the empirical generalization gap and the theoretical bound. Plot both on the same axis as a function of T.

**Test:** Does the empirical gap stay below the theoretical bound? Does the bound tighten at the predicted rate?

---

## Phase 4 — Comparative Benchmarking

Only run this phase after Phases 1–3 are complete and interpreted.

Compare BlockBoost against: AdaBoost, LogitBoost, GentleAdaBoost, and a naive block-average ensemble (no boosting, just average over blocks). The last baseline is critical — it tells you how much of BlockBoost's gain comes from blocking alone versus the interaction of blocking with boosting.

**Primary metric:** Area under the accuracy-vs-rounds curve (AUC of the learning curve), normalized by the clean Bayes accuracy. This single number summarizes both convergence speed and plateau quality.

---

## Statistical Reporting Requirements

For every experiment, across 20 seeds report:

- Mean ± standard deviation of all metrics
- 95% confidence intervals using t-distribution (not normal, since n=20 seeds)
- For collapse detection: report fraction of seeds that collapse, not just mean behavior
- For block size experiment: report Friedman test across block sizes before claiming optimal a_T

**Never report a single seed result as representative.**

---

## Execution Order

```
Phase 0 → must pass before anything else
Phase 1.1 (noise rate) → most important single experiment
Phase 1.2 (block size) → second priority, validates theory
Phase 3.1 (step size bound) → run in parallel with Phase 1
Phase 1.3, 1.4 → after 1.1 and 1.2 confirmed
Phase 2 → robustness, run after Phase 1 complete
Phase 3.2, 3.3 → requires Phase 1 data, run last
Phase 4 → final, only after all theory verified
```

This ordering ensures that if your DGP is still broken, you discover it in Phase 0 before wasting compute on 50 experiments built on a flawed foundation.



ARMA(p,q) should come first because it introduces moving average components that create richer short-memory dependence structures not captured by AR alone. This directly tests whether BlockBoost handles non-Markovian dependence.

GARCH processes should come second and are actually the most important addition given your Chapter 2 is about VaR. GARCH has time-varying volatility, which means the noise structure is itself temporally dependent — exactly the regime where AdaBoost's weight concentration is most dangerous and BlockBoost's averaging should shine most.

Regime-switching processes (Markov-switching) should come third, as they introduce non-stationarity in a controlled way, directly motivating your 2SDM framework.

ARIMA should be low priority or excluded — integrated processes are non-stationary by construction, which violates your theoretical assumptions. Testing BlockBoost on ARIMA without the 2SDM extension would be testing outside your theoretical guarantees, which is scientifically misleading unless you frame it explicitly as an out-of-distribution stress test.