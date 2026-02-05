# Scratchpad

## Session: 2026-02-05 - Final Verification Complete

### Task Status
✅ COMPLETE - Markov property fix fully verified and validated

### Full Simulation Results (200 sims, N=5,000)

**Markov Property Tests:**
- ALL checks passed with flying colors
- Mean partial correlation: -0.003 (target: 0, before: 0.076)
- KS test for p-value uniformity: p=0.928 (Z1), p=0.618 (Z2)
- Individual tests show uniform p-values across all simulations

**Causal Estimators:**
| Estimator | Bias | Status |
|-----------|------|--------|
| Naive OLS | 0.267 | Biased (expected) |
| IPW | 0.0004 | ✅ Unbiased |
| G-comp | 0.002 | ✅ Unbiased |
| AIPW | 0.002 | ✅ Unbiased |

**Chain Structure Verification:**
- Rank uniformity tests confirmed proper chain structure
- All marginal and conditional distributions correct

### Power Validation Tests

Confirmed tests can detect real dependencies when they exist:

| Conditioning Set | Partial Corr | p-value | Status |
|-----------------|--------------|---------|--------|
| Full (Z2_t, X_t, Y_{t-1}) | 0.005 | 0.443 | ✅ Pass (correct) |
| Omit Z1_t | 0.078 | <0.0001 | ✅ Detected |
| Omit Z2_t | 0.093 | <0.0001 | ✅ Detected |
| Omit Y_{t-1} | 0.235 | <0.0001 | ✅ Detected |
| Marginal (none) | 0.461 | <0.0001 | ✅ Detected |

### ChatGPT Comparison

Confirmed our implementation aligns with canonical algorithm:

**ChatGPT's Gaussian BN Algorithm:**
```
1. Z_k | Z_{pa(k)} ~ N(μ_k(pa), σ_k²)
2. U_k = Φ((Z_k - μ_k)/σ_k)  ← Independent of parents
3. X_k = F^{-1}_{k|pa}(U_k | X_{pa})
```

**Our Shortcut (equivalent):**
```
1. U_k ~ Unif(0,1) directly  ← Independent by definition
2. X_k = F^{-1}_{k|pa}(U_k | X_{pa})
```

Both work because the Gaussian intermediate in ChatGPT's step 2 produces U_k ~ Unif(0,1) that is independent of parents.

### Files Modified
- `nonparanormal/causal_validation_longitudinal.R` — Replaced function with BN approach

### Files to Commit
- `.claude/scratchpad.md` — This file
- `.claude/learnings.md` — Updated with session summary
- `.claude/plan.md` — Marked complete
- `.claude/handoff.md` — Final handoff
- `nonparanormal/causal_validation_longitudinal.R` — Fixed implementation

### Files to Clean Up (temporary)
- `nonparanormal/test_should_fail.R` — Power validation test
- `nonparanormal/validation_output.log` — Output log

---

## Session: 2026-02-04 (continued) - Fix Verified & Documented

### Task Completed
Replaced `generate_longitudinal_data()` function in `causal_validation_longitudinal.R` with the corrected BN parameterization approach.

### Verification Results

**Markov Property Test (N=10,000):**
| Test | Before Fix | After Fix |
|------|------------|-----------|
| pcor(Y_t, Z1_{t-1} \| cond) | ~0.076 | **-0.0043** |
| pcor(Y_t, Z2_{t-1} \| cond) | ~0.05 | **0.0157** |
| p-values | < 0.001 | > 0.10 |

**Causal Estimation (20 sims, N=3,000):**
| Estimator | Mean | Bias | Status |
|-----------|------|------|--------|
| Naive OLS | 1.07 | 0.57 | Biased (expected) |
| IPW | 0.499 | -0.001 | ✅ Unbiased |

### Key Insight: Equivalence with Gaussian Copula BN Sampling

ChatGPT provided a detailed explanation of sampling from Gaussian copula BNs with conditional marginals. The key algorithm is:

**ChatGPT's Approach:**
```
1. Draw Z_k | Z_{pa(k)} ~ N(μ_k(Z_pa), σ_k²)
2. Compute conditional rank: U_k = Φ((Z_k - μ_k)/σ_k)  ← Independent of parents
3. Transform: X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})
```

**Our Shortcut (mathematically equivalent):**
```
1. Draw U_k ~ Unif(0,1) directly  ← Independent by definition
2. Transform: X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})
```

**Why these are equivalent:**
- In ChatGPT's step 2, U_k = Φ((Z_k - μ_k)/σ_k) where (Z_k - μ_k) ~ N(0, σ_k²) independent of parents
- So U_k ~ Unif(0,1) and U_k ⊥ Z_{pa(k)}
- This is exactly what runif(n) gives directly!

**When to use the full Gaussian intermediate:**
- When fitting copula correlations from data
- When you need a unified latent Gaussian representation
- For theoretical clarity

**When the shortcut suffices:**
- When conditional marginals are specified directly (our case)
- For simulation where you control the DGP

### Files Modified

- `nonparanormal/causal_validation_longitudinal.R` — Replaced function with v2 implementation
- `.claude/plan.md` — Marked steps complete
- `.claude/learnings.md` — Added canonical algorithm documentation

---

## Session: 2026-02-04 - Debugging Markov Property Violations

### Current Task
Debugging independence/Markov property violations in longitudinal causal validation experiments

### Problem Identified
The `causal_validation_longitudinal.R` script was showing Markov violations (partial correlations ~0.076 for Z1_{t-1}) despite using DAG-aware rank unconditioning.

### Root Cause Analysis

**The Fundamental Design Error:**
1. Fitting a copula to full variable set (Z_{t-1}, Z_t, Y_{t-1})
2. Extracting conditional ranks U_{Z_j | pa(Z_j)} from this copula
3. Using conditional marginals F(Z_j | pa(Z_j)) for transformation

**Why This Fails:**
- When you extract conditional ranks U_{Z_j | pa(Z_j)}, you make them INDEPENDENT of parent Q values
- Using conditional CDFs on independent ranks creates NO dependence at all
- This broke the intended dependence structure

### The Correct Approach: BN Parameterization

**For Z-Z dependencies:**
- Use INDEPENDENT conditional ranks (iid Uniform)
- Transform via CONDITIONAL CDFs: Z_j = F^{-1}_{Z_j|pa(Z_j)}(U_j; pa(Z_j))
- Dependence comes from shape parameters depending on parents, NOT from correlated ranks

**For Y-Z dependencies:**
- Use Gaussian copula to encode Y's dependence on conditioning set
- Requires marginal ranks for the conditioning variables

**Key Insight:**
With conditional marginals (BN parameterization), Z-Z dependence is encoded in the CONDITIONAL CDFs (shape parameters depend on parent values), NOT in copula ranks. The copula is only needed for Y-Z dependence.

### Files Created/Modified

**Created:**
- `nonparanormal/generate_longitudinal_data_v2.R` - Corrected implementation (~250 lines)
  - Separate function for each DGP component
  - Independent ranks for Z variables (BN approach)
  - Gaussian copula only for Y-Z dependence
- `nonparanormal/debug_independence.R` - Step-by-step debugging script

**Modified:**
- `nonparanormal/causal_validation_longitudinal.R` - Updated marginal parameter specification
  - Changed from FIXED to CONDITIONAL for Z2 parameters
  - Function still needs to be replaced with v2 implementation

### Current State
- Correct algorithm written in `generate_longitudinal_data_v2.R`
- Main script still has old broken function
- Tests run but Markov violations persist (partial corr ~0.076)
- **Next step:** Replace function in main script with v2 version and re-test

### Key Design Decision

**Two Approaches Contrasted:**

| Aspect | Option A (Vine/Copula for Z) | Option B (BN for Z) |
|--------|------------------------------|---------------------|
| Z-Z ranks | Correlated (from copula) | Independent (iid) |
| Z-Z CDFs | Marginal | Conditional |
| Dependence encoding | Copula correlation | CDF parameters |
| Complexity | Higher | Lower |
| Natural for | General dependencies | Bayesian networks |

**Decision:** Use Option B (BN parameterization) for validation experiments because:
1. More natural for BN structures
2. Simpler implementation
3. Clearer separation: BN for Z, copula for Y-Z

### Open Questions
None - approach is clear, just needs implementation and testing

---

## Session: 2026-02-03 17:00 - Added Causal Validation to Paper

### Current Task
Added new subsection 6.5 "Causal Effect Validation" to nonparanormal.tex with experimental results

### Key Accomplishments

**Modified `Hybrid-Frugal-Paper/sections/nonparanormal.tex`**
- Added subsection 6.5 "Causal Effect Validation" (~120 lines of LaTeX)
- Two TikZ DAG diagrams (static and dynamic models)
- Two results tables with statistical summaries
- Figure reference for boxplots

**Content Added:**

1. **6.5.1 Static Model Validation**
   - DAG: Z → X → Y, Z → Y (single confounder)
   - Table: Naive OLS biased (0.409), IPW/AIPW unbiased
   - p-values from t-tests for bias=0

2. **6.5.2 Dynamic Model Validation**
   - Two time-point longitudinal model
   - Tests Markov property: Y₂ ⊥ Z₁ | Z₂, X₂, Y₁
   - All causal estimators unbiased

3. **Figure 11** (new): Boxplots showing estimator distributions

**Files Modified:**
- `Hybrid-Frugal-Paper/sections/nonparanormal.tex` — Added subsection 6.5

**Files Copied:**
- `nonparanormal/results/causal_validation_static_boxplot.png` → `Hybrid-Frugal-Paper/images/plots/`
- `nonparanormal/results/causal_validation_dynamic_boxplot.png` → `Hybrid-Frugal-Paper/images/plots/`

### Statistical Results Summary

| Model | Estimator | Mean | Bias | RMSE | p-value |
|-------|-----------|------|------|------|---------|
| Static | Naive OLS | 0.909 | 0.409 | 0.410 | --- |
| Static | IPW | 0.502 | 0.002 | 0.028 | 0.295 |
| Static | G-comp | 0.505 | 0.005 | 0.027 | 0.004 |
| Static | AIPW | 0.501 | 0.001 | 0.028 | 0.786 |
| Dynamic | Naive OLS | 0.767 | 0.267 | 0.269 | --- |
| Dynamic | IPW | 0.501 | 0.001 | 0.033 | 0.547 |
| Dynamic | G-comp | 0.501 | 0.001 | 0.031 | 0.663 |
| Dynamic | AIPW | 0.500 | 0.000 | 0.032 | 0.827 |

### Key Narrative Points
- X does NOT enter the copula structure — affects Y only through causal margin
- IPW and AIPW are unbiased, confirming p(Y|do(X)) preservation
- G-comp shows small bias in static model (outcome model misspecification)
- Combined with independence tests, validates both structure and causal quantities

---

## Session: 2026-02-03 16:00 - Causal Effect Validation Experiments

### Current Task
Implemented experiments to validate that nonparanormal approximation preserves causal margins p(Y|do(X))

### Key Accomplishments

**Created `R/generate_frugal_outcome.R`** (~400 lines)
- `generate_frugal_outcome()` - Core function for frugal Y with known causal margin
- `generate_frugal_outcome_approach_A()` - DAG-aware version
- Causal estimators: `compute_ipw_ate()`, `compute_gcomp_ate()`, `compute_aipw_ate()`, `compute_naive_ate()`
- Propensity score estimation helper

**Created `causal_validation_static.R`** (~350 lines)
- Single time point: Z -> X -> Y, Z -> Y
- Known ATE = 0.5 (TRUE_BETA1)
- 200 Monte Carlo simulations, N=5000 each
- Tests IPW, G-comp, AIPW vs known truth
- Verification checklist + visualizations

**Created `causal_validation_dynamic.R`** (~450 lines)
- Two time points with treatment at T=2
- Dynamic margin: E[Y2|do(X2), Y1] = γ₀ + γ₁X2 + γ₂Y1
- Z2 depends on (Z1, Y1) - temporal confounding
- Markov property test: Y2 ⊥ Z1 | Z2, X2, Y1
- Same estimator comparison as static

### Critical Design Insight

**X does NOT enter the copula structure**
- The copula encodes φ*(Y, Z | do(X)) - dependence between Y and confounders
- X affects Y ONLY through the causal margin specification
- This ensures p(Y|do(X)) = ∫ p(Y|X,Z) p(Z) dZ correctly

### Files Created
- `nonparanormal/R/generate_frugal_outcome.R`
- `nonparanormal/causal_validation_static.R`
- `nonparanormal/causal_validation_dynamic.R`

### Expected Results (When Run)
| Estimator | Bias for True ATE |
|-----------|------------------|
| Naive OLS | Biased (confounding) |
| IPW | ~0 (unbiased) |
| G-computation | ~0 (unbiased) |
| AIPW | ~0 (unbiased) |

### Next Steps
- [ ] Run experiments and verify results
- [ ] Add to paper Section 6 experiments
- [ ] Consider sensitivity analysis varying rho_Y_Z

---

## Session: 2026-02-03 14:30 - Two Approaches to Nonparanormal Approximation

### Current Task
Clarifying fundamental design choice for nonparanormal approximation: regenerate vs preserve original data

### Key Discussion

**User Clarification:** There are TWO distinct approaches with different trade-offs

#### Approach A: Regenerate from Gaussian Copula
- Sample FRESH ranks from Gaussian copula with fitted correlation matrix
- Conditional ranks are uniform by construction (from Gaussian copula h-functions)
- Preserves: Marginals (exact), Markov/CI structure (exact)
- Approximates: Z-Z tail dependence, conditional shapes

#### Approach B: Preserve Original Data
- Keep original Z values from non-Gaussian DGP
- Use their (non-uniform) conditional ranks with Gaussian formulas
- Preserves: Marginals (exact), Z-Z tail dependence, conditional shapes
- Approximates: Markov/CI structure (~0.02 violations)

### Critical Insight

**Previous confusion:** The `longitudinal_fixed.R` implementation was doing Approach B
- Taking conditional ranks from Gamma BN (non-uniform)
- Trying to "uncondition" with Gaussian formulas
- Result: Mismatch causing ~0.02 partial correlation violations

**Correct understanding of Approach A:**
- When you generate fresh from Gaussian copula, conditional ranks ARE uniform
- This is by construction - the h-function maps to [0,1]
- No "mismatch" because everything is Gaussian

### Design Decision Needed for Paper

Both approaches preserve the causal margin p(Y|do(X)). The choice is:
- **Approach A:** Accept approximated Z-Z dependence to get exact Markov/CI
- **Approach B:** Accept small Markov/CI violations to preserve Z-Z dependence

**Recommendation:** Paper Section 6 should explicitly state which approach is used and justify the trade-off.

### Files Modified
- `.claude/learnings.md` — Added comprehensive comparison of both approaches

### Next Steps
- [ ] Update paper text to clarify which approach is being used
- [ ] Possibly add appendix comparing both approaches empirically
- [ ] Document this design choice in decisions.md

---

## Session: 2026-02-03 - DAG-Aware Rank Unconditioning Implementation

### Current Task
Implemented DAG-aware rank unconditioning to fix Markov property preservation in longitudinal models.

### Problem Solved
The `uncondition_conditional_ranks` function assumed a fully-connected D-vine structure where each variable j is conditioned on ALL previous variables (1, 2, ..., j-1). This is incorrect for longitudinal models where variables should only be conditioned on their actual DAG parents.

**Example (T=3, 2 covariates):**
| Column | Variable | D-vine (wrong) | DAG (correct) |
|--------|----------|----------------|---------------|
| 3 | Z1_2 | Z1_1, Z2_1 | Z1_1 only |
| 5 | Z1_3 | all prev | Z1_2 only |

### Files Created
- `nonparanormal/R/dag.R` - DAG utility functions (~300 lines):
  - `dag_from_edges()` - Convert edge list to parent structure
  - `dag_from_adjacency()` - Convert adjacency matrix to parent structure
  - `make_chain_dag()` - Simple X1->X2->X3 chain
  - `make_longitudinal_dag()` - Temporal model structure (markov, ar1, full)
  - `check_dag_order()` - Validate topological order
  - `get_topological_order()` - Compute valid ordering (Kahn's algorithm)
  - `print_dag()` - Pretty-print DAG structure

- `nonparanormal/tests/testthat/test-dag.R` - 47 tests for DAG functions

### Files Modified
- `nonparanormal/R/rank_transform.R`:
  - Added `check_order` parameter for validation
  - Added caching for repeated `solve(R_sub)` calls
  - Enhanced documentation with `@seealso` links

- `nonparanormal/nonparanormal.R`:
  - Added `make_longitudinal_dag()` and `make_chain_dag()` helper functions
  - Updated `uncondition_conditional_ranks()` with same enhancements

- `nonparanormal/NAMESPACE`:
  - Exported 7 new DAG functions

- `nonparanormal/tests/testthat/test-rank_transform.R`:
  - Added 7 DAG-aware unconditioning tests

- `nonparanormal/longitudinal_fixed.R`:
  - Replaced manual parent specification with `make_longitudinal_dag()` helper

### Test Results
- **DAG tests:** 47/47 passed
- **Rank transform tests:** 18/18 passed
- **Longitudinal experiment:**
  - Partial correlations all ~0 (correct Markov property)
  - GCM tests: 3/4 passed (borderline failure at 0.04 vs 0.05 threshold)

### Key Design Decisions
1. **Column ordering:** Z1_1, Z2_1, Z1_2, Z2_2, ... (matches longitudinal_fixed.R)
2. **Markov structure:** Z_d^t depends on Z_d^{t-1} (AR term) + Z_1^t,...,Z_{d-1}^t (cross-sectional)
3. **Caching:** R_sub inversions cached by parent set to avoid redundant solve() calls
4. **Validation:** Topological order validated by default, can disable with check_order=FALSE

### API Summary
```r
# Create longitudinal DAG
parents <- make_longitudinal_dag(n_time = 3, n_cov = 2, structure = "markov")

# Uncondition with DAG structure
marginal_ranks <- uncondition_conditional_ranks(cond_ranks, R, parents)
```

---
