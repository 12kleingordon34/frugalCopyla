# Handoff Summary

**Last Updated:** 2026-02-05
**Context Usage:** ~30%
**Branch:** inversion

---

## What Was Accomplished This Session

### Major Achievement: Markov Property Violations RESOLVED

Successfully fixed and fully validated the Markov property preservation in longitudinal causal validation experiments. This was a critical bug causing statistical test failures (~0.076 partial correlations instead of ~0).

### Complete Verification Results

**Full Simulation Study (200 simulations, N=5,000 each):**

1. **Markov Property Tests** - ✅ ALL PASSED
   - Mean partial correlation: -0.003 (target: 0, before fix: 0.076)
   - KS test for p-value uniformity: p=0.928 (Z1), p=0.618 (Z2)
   - P-values uniformly distributed as expected

2. **Causal Estimators** - ✅ ALL UNBIASED
   - IPW bias: 0.0004
   - G-computation bias: 0.002
   - AIPW bias: 0.002
   - Naive OLS biased: 0.267 (expected due to confounding)

3. **Chain Structure Verification** - ✅ CONFIRMED
   - Rank uniformity tests passed
   - Marginal and conditional distributions correct

4. **Power Validation** - ✅ TESTS WORK CORRECTLY
   - Full conditioning: pcor=0.005, p=0.443 (correctly passes)
   - Omit Z1_t: pcor=0.078, p<0.0001 (correctly detects dependency)
   - Omit Z2_t: pcor=0.093, p<0.0001 (correctly detects dependency)
   - Omit Y_{t-1}: pcor=0.235, p<0.0001 (correctly detects dependency)
   - Marginal: cor=0.461, p<0.0001 (correctly detects dependency)

### The Fix: BN Parameterization

**Root Cause (now fully understood):**
The old implementation was:
1. Fitting a copula to ALL variables (Z_{t-1}, Z_t, Y_{t-1})
2. Extracting conditional ranks U_{Z_j | pa(Z_j)} from this copula
3. Transforming through conditional CDFs F(Z_j | pa(Z_j))

This fails because extracting conditional ranks makes them INDEPENDENT of parent values, then using conditional CDFs on independent ranks creates NO dependence.

**The Correct Solution:**
```r
# For Z variables - BN parameterization
U_j <- runif(n)  # INDEPENDENT ranks
shape_j <- base_shape + beta * parent_value  # Parameters depend on parent VALUES
Z_j <- qgamma(U_j, shape = shape_j, scale = scale)

# For Y variable - Gaussian copula
# Use marginal ranks of Z variables (after they're fully generated)
```

**Key insight:** With conditional CDFs, dependence comes from PARAMETERS depending on parent values, NOT from correlated ranks.

### Documentation Updates

**Updated tracking files:**
- `.claude/learnings.md` — Added complete session summary with results
- `.claude/scratchpad.md` — Added final verification results
- `.claude/plan.md` — Marked all steps complete
- `.claude/decisions.md` — Added formal decision entry (previous session)

**ChatGPT Comparison Added:**
Confirmed our "shortcut" (independent uniforms + conditional CDFs) is mathematically equivalent to the canonical Gaussian copula BN algorithm (Gaussian intermediate + conditional CDFs). Both produce U_j ⊥ pa(j).

---

## Current State

### Code Status

**Working Perfectly:**
- `nonparanormal/causal_validation_longitudinal.R` — Fixed with BN parameterization
  - Function `generate_longitudinal_data()` replaced with corrected version
  - All validation tests pass with 200 simulations

**Temporary Files (can be deleted):**
- `nonparanormal/test_should_fail.R` — Power validation test script
- `nonparanormal/validation_output.log` — Simulation output log

### Repository Status

**Modified Files (uncommitted):**
- `nonparanormal/causal_validation_longitudinal.R` — Fixed implementation
- `.claude/scratchpad.md` — Session notes
- `.claude/learnings.md` — Updated with verification results
- `.claude/plan.md` — Updated status
- `.claude/handoff.md` — This file
- `.claude/codebase-map.md` — (previously modified)
- `.claude/decisions.md` — (previously modified)

**Branch:** inversion

**Ready to commit:** Yes - all validation complete

---

## Next Steps

### Priority 1: Paper Updates

1. **Update Section 6.5 in `Hybrid-Frugal-Paper/sections/nonparanormal.tex`:**
   - Add clarification about BN parameterization for Z variables
   - Explain that copula is used ONLY for Y-Z dependence
   - Emphasize distinction: independent ranks + conditional CDFs (not conditional ranks + conditional CDFs)

2. **Consider adding methodological note:**
   - Explain the two approaches: BN parameterization vs vine copula for covariates
   - State when each is appropriate
   - Clarify that both preserve causal margins, differ in how Z-Z dependence is modeled

### Priority 2: Code Consistency

3. **Check other validation experiments:**
   - Review `nonparanormal/causal_validation_static.R` (if exists)
   - Ensure consistent use of BN parameterization approach
   - Verify no other experiments have the same mixing error

4. **Clean up temporary files:**
   ```bash
   rm nonparanormal/test_should_fail.R
   rm nonparanormal/validation_output.log
   ```

5. **Consider checking dynamic models:**
   - May need same fix if they exist
   - Run full simulation study for those as well

### Priority 3: Commit Changes

6. **Commit all changes:**
   ```bash
   git add nonparanormal/causal_validation_longitudinal.R
   git add .claude/
   git commit -m "fix: Resolve Markov property violations in longitudinal validation

   - Replaced copula extraction approach with BN parameterization
   - Z variables now use independent ranks + conditional CDFs
   - Y uses Gaussian copula for dependence on Z
   - Verified with 200 simulations: pcor=-0.003, all estimators unbiased
   - Added power validation tests confirming tests work correctly"
   ```

---

## Important Context

### The Critical Design Pattern

**When using Bayesian Network factorization with conditional marginals:**

```r
# ✅ CORRECT: Independent ranks + conditional CDFs
U_j <- runif(n)  # Independent
param_j <- f(parent_values)  # Parameters depend on parents
X_j <- qf(U_j, param_j)

# ❌ WRONG: Conditional ranks + conditional CDFs
U_j_cond <- extract_from_copula(...)  # Already independent!
X_j <- qf(U_j_cond, param_j)  # Creates no dependence
```

**Why this matters:**
- Conditional ranks from a copula are ALREADY independent of their conditioning set
- Using them with conditional CDFs creates no dependence
- Result: Violates the intended DAG structure

### Two Valid Approaches Contrasted

| Aspect | Approach A (BN) | Approach B (Vine) |
|--------|----------------|-------------------|
| Ranks | Independent (iid) | Correlated (from copula) |
| CDFs | Conditional | Marginal |
| Dependence | In CDF parameters | In copula correlation |
| Best for | Bayesian networks | General dependencies |
| Used where | Validation expts | General frugal models |

**Both are valid, but don't mix them!**

### Key Files Reference

**Main implementation:**
- `/Users/danielmanela/.../nonparanormal/causal_validation_longitudinal.R` — Fixed and verified

**Documentation:**
- `/Users/danielmanela/.../.claude/learnings.md` — Complete session summary with results
- `/Users/danielmanela/.../.claude/decisions.md` — Formal decision entry

**Paper:**
- `/Users/danielmanela/.../Hybrid-Frugal-Paper/sections/nonparanormal.tex` — Section 6.5 (needs update)

---

## How to Resume

### Quick Start (If Continuing Immediately)

1. **Update paper Section 6.5:**
   ```bash
   cd "Hybrid-Frugal-Paper"
   # Edit sections/nonparanormal.tex
   # Add clarification about BN parameterization approach
   ```

2. **Check for consistency in other scripts:**
   ```bash
   cd nonparanormal
   # Look for other causal_validation_*.R files
   # Verify they use the same correct approach
   ```

3. **Commit everything:**
   ```bash
   git add -A
   git commit -m "fix: Resolve Markov violations + update docs"
   git push origin inversion
   ```

### If Starting Fresh Session

1. **Read this handoff completely** — Contains all context needed

2. **Verify the fix is still working:**
   ```r
   source("nonparanormal/causal_validation_longitudinal.R")
   # Check results/causal_validation_longitudinal_results.csv
   # Verify partial correlations ~0, causal estimators unbiased
   ```

3. **Proceed to paper updates** — The main task remaining

### If Issues Arise (Unlikely)

**If you see Markov violations again:**
- Check that the function uses `runif(n)` for Z variables (not copula extraction)
- Verify conditional CDFs have parameters that depend on parent VALUES
- Compare with `generate_longitudinal_data_v2.R` (the corrected version)

**If causal estimators are biased:**
- Check that Y generation uses Gaussian copula with marginal ranks
- Verify X does NOT enter the copula structure
- Check causal margin specification is correct

---

## Expected Timeline

**Immediate next steps:**
- Paper Section 6.5 update: 30-45 minutes
- Consistency check of other scripts: 15-20 minutes
- Commit all changes: 5 minutes
- **Total: ~1 hour**

---

## Open Questions

None - the fix is complete, verified, and well-understood.

---

## Summary

**Status:** ✅ COMPLETE AND VERIFIED

**What Changed:**
- Fixed fundamental design error mixing copula extraction with conditional marginals
- Replaced with correct BN parameterization (independent ranks + conditional CDFs)

**Verification:**
- 200 simulations with N=5,000 each
- All Markov tests pass (pcor=-0.003)
- All causal estimators unbiased
- Power validation confirms tests work

**What's Needed:**
- Update paper text to clarify the approach
- Check consistency in other validation experiments
- Commit changes

**Confidence:**
- Very high - fully verified with extensive simulation
- Theory confirmed via ChatGPT comparison
- Power tests show our tests work correctly

**Key Takeaway:**
When using conditional CDFs F(Z|pa), you MUST use independent ranks. Dependence comes from the CDF parameters depending on parent values, NOT from correlated ranks. This is the canonical Gaussian copula BN algorithm.
