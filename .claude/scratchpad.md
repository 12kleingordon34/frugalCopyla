# Scratchpad

## Session: 2026-02-05 (Current) - Paper Section 6 Updates + All Validation Complete

### Current Task
Complete Paper Section 6 revisions with all three validation experiments (static, dynamic, longitudinal) showing consistent unbiased ATE estimates.

### Key Accomplishments

**All 4 phases of plan completed and committed (ddadae4), pushed to origin/inversion:**

**Phase 1: Paper Text Edits**
- Added Remark 1 after Algorithm 1 explaining BN vs copula distinction (rem:bn-vs-copula)
- Clarified Section 6.5 - replaced vague "Gaussian copula approximation" with explicit BN/copula description
- Added "Sampling Mechanism" paragraph explaining independent ranks + conditional CDFs
- Added Dynamic Model Validation subsubsection with DAG, Markov test results, ATE table
- Fixed appendix: GCM → partial correlation throughout app:ci-pvalues section

**Phase 2: Dynamic Script Upgrade**
- Added library(ppcor) import to causal_validation_dynamic.R
- Rewrote test_markov_property() to use ppcor::pcor.test (matching longitudinal pattern)
- Integrated Markov test into run_single_dynamic_simulation()
- Added KS uniformity test, p-value histogram, updated verification checklist

**Phase 3: Re-ran All Three Experiments**
- Static: IPW unbiased (p=0.295), AIPW unbiased (p=0.786)
- Dynamic: All pass, KS p=0.485, mean pcor=-0.0001
- Longitudinal: KS improved to 0.928/0.618 (from previous 0.044/0.805)

**Phase 4: Verification**
- Code reviewer identified minor issues; fixed KS p-value in dynamic section and Z^{t-1} notation in appendix
- Pre-existing issues noted but out of scope

### Validation Results Summary

| Model | Markov Test | IPW Bias | AIPW Bias | Status |
|-------|-------------|----------|-----------|--------|
| Static | N/A | 0.002 (p=0.295) | 0.001 (p=0.786) | ✅ Unbiased |
| Dynamic | KS p=0.485 | 0.001 (p=0.547) | 0.000 (p=0.827) | ✅ Unbiased |
| Longitudinal | KS p=0.928/0.618 | 0.0004 | 0.002 | ✅ Unbiased |

### Files Modified
- `Hybrid-Frugal-Paper/sections/nonparanormal.tex` - All text updates (Remark, Section 6.5, dynamic subsection)
- `Hybrid-Frugal-Paper/sections/appendix.tex` - GCM→partial correlation fix
- `nonparanormal/causal_validation_dynamic.R` - ppcor Markov test upgrade
- `Hybrid-Frugal-Paper/images/plots/gcm_pvalue_histogram_Z1.png` and `Z2.png` - updated plots

### Commit
- **ddadae4** - All changes committed and pushed to origin/inversion

### Known Issues (Out of Scope, Pre-existing)
- \label on equation* (line 102)
- Z_1 subscript typo (line 79)
- Duplicate expression in Algorithm 1 (line 153)
- M_1/M_2 vs M_A/M_B naming inconsistency
- Double period at line 269
- "p-values p-values" duplicate at line 288
- Image filenames still use gcm_ prefix (functional but inconsistent)

---

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
