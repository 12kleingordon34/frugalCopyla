# Current Plan

**Goal:** Debug and Fix Markov Property Violations in Longitudinal Causal Validation
**Started:** 2026-02-04
**Status:** ✅ COMPLETE - Fully Verified with 200 Simulations

## Overview

Debugging Markov property violations (~0.076 partial correlations) in the longitudinal causal validation experiment. Identified fundamental design error in mixing copula and BN parameterization approaches.

## Completed Steps

- [x] Identified problem: Markov violations in `causal_validation_longitudinal.R`
- [x] Analyzed root cause: Incorrect mixing of copula extraction and conditional marginals
- [x] Determined correct approach: BN parameterization (Option B)
  - Independent ranks for Z variables
  - Conditional CDFs for transformation
  - Gaussian copula only for Y-Z dependence
- [x] Created corrected implementation: `generate_longitudinal_data_v2.R`
- [x] Created debugging script: `debug_independence.R`

## Remaining Steps

### Immediate (Current Session) - COMPLETE
- [x] Replace function in `causal_validation_longitudinal.R` with v2 implementation ✅
- [x] Re-run validation tests ✅
- [x] Verify Markov property is preserved (partial corr < 0.01) ✅ pcor = -0.003
- [x] Verify causal estimators remain unbiased ✅ IPW bias = 0.0004
- [x] Run full simulation study (200 sims, N=5,000) ✅
- [x] Validate power of tests with omitted conditioning variables ✅
- [x] Compare with ChatGPT canonical algorithm ✅
- [x] Document findings in learnings.md ✅

### Follow-up (Next Session)
- [ ] Update paper text to clarify BN parameterization approach
- [ ] Ensure consistency across all validation experiments
- [ ] Consider adding methodological note explaining the two approaches

## Key Insights

### The Design Error
**What was wrong:**
1. Fit copula to full variable set (Z_{t-1}, Z_t, Y_{t-1})
2. Extract conditional ranks U_{Z_j | pa(Z_j)}
3. Use conditional CDFs F(Z_j | pa(Z_j))

**Why it failed:**
- Extracting conditional ranks makes them INDEPENDENT of parent values
- Using conditional CDFs on independent ranks creates NO dependence
- Violated the intended BN structure

### The Correct Approach
**Option B: BN Parameterization**
- For Z-Z: Independent ranks + conditional CDFs
- For Y-Z: Gaussian copula with marginal ranks
- Dependence in Z-Z comes from CDF parameters, not rank correlation

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `nonparanormal/generate_longitudinal_data_v2.R` | CREATE | Corrected implementation (~250 lines) |
| `nonparanormal/debug_independence.R` | CREATE | Step-by-step debugging script |
| `nonparanormal/causal_validation_longitudinal.R` | EDIT | Updated marginal params (FIXED→CONDITIONAL) |

## Blockers

None - approach is clear, just needs implementation and testing.

---

# Previous Plan: Section 6.5 Restructure (Complete)

**Goal:** Restructure Section 6.5 - Longitudinal Causal Validation
**Started:** 2026-02-04
**Status:** ✅ Complete

See archived scratchpad for details.
