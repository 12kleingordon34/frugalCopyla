# Current Plan

**Goal:** Complete Paper Section 6 Updates with All Three Validation Experiments
**Started:** 2026-02-05
**Status:** ✅ COMPLETE - All Committed (ddadae4) and Pushed to origin/inversion

## Overview

Complete comprehensive updates to Paper Section 6 (Nonparanormal) with consistent validation results across all three experiments (static, dynamic, longitudinal) showing unbiased ATE estimates.

## Completed Steps

### Phase 1: Paper Text Edits ✅
- [x] Add Remark after Algorithm 1 explaining BN vs copula distinction (commit: ddadae4)
- [x] Clarify Section 6.5 validation subsection with explicit BN/copula terminology (commit: ddadae4)
- [x] Add "Sampling Mechanism" paragraph explaining independent ranks + conditional CDFs (commit: ddadae4)
- [x] Add Dynamic Model Validation subsubsection with DAG, Markov test results, ATE table (commit: ddadae4)
- [x] Fix appendix GCM → partial correlation labels in Section app:ci-pvalues (commit: ddadae4)

### Phase 2: Dynamic Script Upgrade ✅
- [x] Add library(ppcor) import to causal_validation_dynamic.R (commit: ddadae4)
- [x] Rewrite test_markov_property() to use ppcor::pcor.test (commit: ddadae4)
- [x] Integrate Markov test into run_single_dynamic_simulation() (commit: ddadae4)
- [x] Add KS uniformity test for p-values (commit: ddadae4)
- [x] Add p-value histogram output (commit: ddadae4)
- [x] Update verification checklist in script comments (commit: ddadae4)

### Phase 3: Re-run All Three Experiments ✅
- [x] Static model: Confirmed IPW unbiased (p=0.295), AIPW unbiased (p=0.786)
- [x] Dynamic model: All pass, KS p=0.485, mean pcor=-0.0001
- [x] Longitudinal model: KS improved to 0.928/0.618 (from previous 0.044/0.805)

### Phase 4: Verification & Final Fixes ✅
- [x] Code review identified issues (commit: ddadae4)
- [x] Fixed KS p-value display in dynamic section (commit: ddadae4)
- [x] Fixed Z^{t-1} notation in appendix (commit: ddadae4)
- [x] Documented pre-existing issues as out of scope
- [x] Committed all changes (commit: ddadae4)
- [x] Pushed to origin/inversion

## Summary of Results

### All Three Validation Experiments Complete

| Model | Markov Test | IPW Bias | AIPW Bias | Status |
|-------|-------------|----------|-----------|--------|
| Static | N/A | 0.002 (p=0.295) | 0.001 (p=0.786) | ✅ Unbiased |
| Dynamic | KS p=0.485 | 0.001 (p=0.547) | 0.000 (p=0.827) | ✅ Unbiased |
| Longitudinal | KS p=0.928/0.618 | 0.0004 | 0.002 | ✅ Unbiased |

**Key achievement:** All three experiments show consistent unbiased ATE estimates, confirming that nonparanormal approximation preserves causal margins p(Y|do(X)).

## Files Modified

| File | Description | Commit |
|------|-------------|--------|
| `Hybrid-Frugal-Paper/sections/nonparanormal.tex` | All text updates | ddadae4 |
| `Hybrid-Frugal-Paper/sections/appendix.tex` | GCM→partial correlation fix | ddadae4 |
| `nonparanormal/causal_validation_dynamic.R` | ppcor Markov test upgrade | ddadae4 |
| `Hybrid-Frugal-Paper/images/plots/gcm_pvalue_histogram_Z*.png` | Updated plots | ddadae4 |

## Known Issues (Out of Scope)

Pre-existing issues noted by code reviewer but not fixed (would require broader paper review):
- \label on equation* (line 102)
- Z_1 subscript typo (line 79)
- Duplicate expression in Algorithm 1 (line 153)
- M_1/M_2 vs M_A/M_B naming inconsistency
- Double period at line 269
- "p-values p-values" duplicate at line 288
- Image filenames still use gcm_ prefix (functional but inconsistent)

## Next Steps (Future Work)

- [ ] Address pre-existing paper issues in separate pass
- [ ] Consider adding methodological note about BN vs vine approaches
- [ ] Review entire paper for notation consistency
- [ ] Final submission preparation

## Blockers

None - all work complete and committed.

---

# Previous Plan: Markov Property Fix (Complete)

**Goal:** Debug and Fix Markov Property Violations in Longitudinal Causal Validation
**Started:** 2026-02-04
**Status:** ✅ COMPLETE - Fully Verified with 200 Simulations (2026-02-05)

All steps completed. See archived scratchpad for details.

---
