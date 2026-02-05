# Project Learnings

> Distilled insights that persist across sessions. Updated when archiving old scratchpad sessions.

## Codebase Insights
- frugalCopyla is a Python package for copula-based causal inference
- Uses JAX/NumPyro for probabilistic computation
- Has companion R scripts in nonparanormal/ for experiments

## Recurring Issues
- [To be populated when issues recur]

## Effective Approaches
- [To be populated as effective patterns emerge]

## Key Decisions Summary
- [To be populated from decisions.md]

---

## Update: 2026-02-03

### Critical Insight: Two Approaches to Nonparanormal Approximation

There are TWO distinct approaches to nonparanormal approximation with fundamentally different trade-offs. This distinction is critical for the paper and implementation.

#### Approach A: Regenerate from Gaussian Copula
**Workflow:**
1. Observe Z from true DGP (e.g., Gamma BN)
2. Fit Gaussian copula to Z → get correlation matrix R
3. Sample FRESH U from Gaussian copula with correlation R
4. These U's are uniform by construction (from Gaussian copula)
5. Apply Z_i = qgamma(U_i^marginal, shape_i, scale_i)

**Properties:**
- Marginal distributions: ✅ Exact (via quantile transform)
- Dependency structure (CI/Markov): ✅ Exact (from Gaussian copula)
- Dependency metrics (tails, higher moments): ❌ Approximated (Gaussian)
- Conditional distributions Z_j|Z_k: Gaussian copula conditionals
- Conditional ranks: ✅ Uniform (as required by copula theory)

#### Approach B: Preserve Original Data
**Workflow:**
1. Observe Z from true DGP
2. Fit Gaussian copula to Z → get R
3. Keep original Z values (don't regenerate)
4. Use Gaussian copula formulas with original (non-uniform) conditional ranks
5. Only generate Y using these ranks

**Properties:**
- Marginal distributions: ✅ Exact (original data)
- Dependency structure (CI/Markov): ⚠️ Small violations (~0.02 partial correlations)
- Dependency metrics (tails, higher moments): ✅ Preserved from true DGP
- Conditional distributions Z_j|Z_k: Original DGP conditionals preserved
- Conditional ranks: ❌ Non-uniform (from true copula, not Gaussian)

#### Trade-off Summary

| Property | Approach A (Regenerate) | Approach B (Preserve) |
|----------|------------------------|----------------------|
| Marginals | Exact | Exact |
| Markov/CI structure | ✅ Exact | ⚠️ ~0.02 violations |
| Tail dependence | ❌ Gaussian approx | ✅ Preserved |
| Conditional Z|Z shapes | Gaussian copula | Original DGP |
| Conditional ranks | Uniform | Non-uniform |

**Key insight:** Both preserve the causal margin p(Y|do(X)). The choice depends on what you're willing to approximate:
- **Approach A:** Approximate Z-Z dependence details (tails, conditional shapes)
- **Approach B:** Approximate Y-Z conditional independence structure (small Markov violations)

#### Why This Matters

**Conditional ranks from Gaussian copula ARE uniform when sampling fresh:**
- When you generate U_{j|pa(j)} from a Gaussian copula, they are uniform by construction
- This is because the copula's h-function maps the conditional distribution to [0,1]
- Using non-Gaussian conditional ranks with Gaussian h-functions creates inconsistencies

**The previous implementation mistake (longitudinal_fixed.R):**
- Was using Approach B: original ranks + Gaussian unconditioning
- This created a mismatch: non-Gaussian conditional ranks being "unconditioned" with Gaussian formulas
- Result: ~0.02 partial correlations (statistically significant violations)

**The correct implementation (Approach A):**
- Regenerate ranks fresh from the Gaussian copula
- Gives uniform conditional ranks consistent with Gaussian assumptions
- Result: partial correlations ~0.005 (statistically zero)

### Experimental Evidence

With N=30,000 samples:

| Test | Approach B (Original) | Approach A (Regenerate) |
|------|---------------------|---------------------------|
| ρ(Y_2, Z1^1 \| Z^2) | -0.026 (z=-4.56)* | -0.006 (z=-1.07) |
| ρ(Y_2, Z2^1 \| Z^2) | -0.016 (z=-2.70)* | -0.006 (z=-1.01) |
| ρ(Y_3, Z1^2 \| Z^3) | -0.022 (z=-3.75)* | -0.004 (z=-0.68) |
| ρ(Y_3, Z2^2 \| Z^3) | -0.009 (z=-1.57) | -0.002 (z=-0.40) |

*Statistically significant (|z| > 1.96)

Approach A reduces violations by ~4x and eliminates statistical significance.

### Implications for Paper Section 6

The paper should explicitly state which approach is being used and why:

1. **For Section 6.3 experiments:** Likely using Approach A to demonstrate Markov preservation
2. **Statement about "linear Gaussian SEM for DGP":** This is Approach A - regenerating from Gaussian
3. **Key clarification needed:** The nonparanormal approximation approximates the Z-Z copula, NOT the Y-Z copula

### Related Implementation Files
- `nonparanormal/R/dag.R` - DAG utility functions
- `nonparanormal/R/rank_transform.R` - `uncondition_conditional_ranks()` with DAG-aware parents
- `nonparanormal/longitudinal_fixed.R` - Experiment comparing approaches
- `nonparanormal/R/simulate.R` - Core vine simulation

---

### Critical Insight: Nonparanormal Approximation Trade-off (Old - Superseded)

The nonparanormal approximation provides TWO distinct use cases with different trade-offs:

#### Option 1: Approximate Covariate Interdependence
- **What's preserved exactly:** Marginal distributions of covariates
- **What's approximated:** Interdependence structure between covariates (via Gaussian copula)
- **Use case:** When you have the exact marginal CDFs and want to model dependencies

#### Option 2: Preserve Covariate Structure, Approximate Outcome Edges
- **What's preserved exactly:** Both the marginal distributions AND the dependency structure between covariates
- **What's approximated:** The dependence between outcomes (Y) and covariates (Z)
- **Use case:** When the covariate model is known/observed and you're adding an outcome

### Key Finding: Regenerating Conditional Ranks (Old - Superseded)

When using nonparanormal approximation with a DAG structure:

**Wrong approach (causes ~0.02 partial correlations):**
1. Take conditional ranks U_{j|pa(j)} from original non-Gaussian model (e.g., Gamma BN)
2. Try to "uncondition" them using Gaussian copula formulas
3. Problem: Mismatch between true copula and Gaussian assumption

**Correct approach (partial correlations ~0.005, statistically zero):**
1. Observe Z values from any marginal distribution
2. Compute empirical marginal ranks: U_j^marginal = rank(Z_j)/(n+1)
3. Transform to Gaussian scale: X_j = Φ^{-1}(U_j^marginal)
4. Compute correlation matrix R from Gaussianized data
5. **Extract conditional ranks from Gaussian copula using DAG structure:**
   - For root nodes: U_{j} = Φ(X_j)
   - For non-roots: U_{j|pa(j)} = Φ((X_j - μ_{j|pa}) / σ_{j|pa})
6. Use these Gaussian-derived conditional ranks for downstream generation

This ensures the conditional ranks are consistent with both the Gaussian copula AND the DAG structure.

### Implication for Paper

The statement "by imposing a linear Gaussian SEM for the DGP of time-varying confounders, the Markov property can be imposed" is correct, but the implementation requires:
1. Using the Gaussian copula correlation matrix computed from marginal ranks
2. Extracting conditional ranks FROM the Gaussian copula structure (not from the original non-Gaussian model)
3. The DAG structure must be respected when extracting these conditional ranks

**Last Updated:** 2026-02-04

---

## Update: 2026-02-04

### From archived sessions (Feb 2-3):

**Codebase Structure:**
- Python package (`frugalCopyla/`) is ~40% complete - MCMC-based, missing inverse h-functions
- R package (`nonparanormal/`) is production-ready - modular structure, 60% test coverage
- Paper (`Hybrid-Frugal-Paper/`) is submission-ready, focuses on R implementation for experiments

**R Package Refactoring Success:**
- Transformed 1100-line monolith into 8 focused modules with test suite
- YAML-based experiment framework enables reproducible validation
- DAG-aware rank unconditioning critical for longitudinal models

**Key Gotchas:**
- VineCopula R-vine matrices must have sequential diagonal
- `topoOrder` default includes outcome Y (usually wrong for frugal models)
- GCM tests show OpenMP warnings on macOS (platform-specific, not a bug)
- JAX/JAXlib requires manual installation due to version conflicts

---

## Update: 2026-02-04 - Markov Property Debugging Session

### Critical Design Error Identified

**The Root Cause:**
The longitudinal causal validation was incorrectly mixing copula approaches:
1. Fitting a copula to the FULL variable set (Z_{t-1}, Z_t, Y_{t-1})
2. Extracting conditional ranks U_{Z_j | pa(Z_j)} from this joint copula
3. Then trying to use CONDITIONAL CDFs F(Z_j | pa(Z_j)) for transformation

**Why This Fails:**
- Extracting conditional ranks U_{Z_j | pa(Z_j)} makes them INDEPENDENT of parent Q values by construction
- Using conditional CDFs on independent ranks creates NO dependence at all
- Result: Violated intended dependence structure, broke Markov property

### The Correct Approach: BN Parameterization (Option B)

**For Z-Z dependencies (BN factorization):**
1. Use INDEPENDENT conditional ranks (iid Uniform) for each Z_j
2. Transform through CONDITIONAL CDFs: Z_j = F^{-1}_{Z_j|pa(Z_j)}(U_j; pa(Z_j))
3. Dependence comes from the SHAPE parameters depending on parent values, NOT from correlated ranks

**For Y-Z dependencies (copula):**
1. Use Gaussian copula to encode Y's dependence on its conditioning set ONLY
2. This requires marginal ranks for the conditioning variables

**Key Insight:**
With conditional marginals (BN parameterization), the Z-Z dependence is encoded in the CONDITIONAL CDFs themselves (shape parameters depend on parents), NOT in the copula structure. The copula is only needed for Y-Z dependence.

### Implementation Files Created

- `nonparanormal/generate_longitudinal_data_v2.R` - Corrected implementation using BN parameterization
- `nonparanormal/debug_independence.R` - Step-by-step debugging script showing the error

### Contrasting Approaches

**Option A (Vine/Copula for Z-Z):**
- Use correlated ranks + marginal CDFs
- Dependence encoded in copula correlation structure
- Works but requires more complex vine specification

**Option B (BN for Z-Z):**
- Use independent ranks + conditional CDFs
- Dependence encoded in conditional CDF parameters
- Simpler, more direct for Bayesian network structures
- THIS IS THE CORRECT APPROACH for the validation experiments

### Implications for Paper

The validation experiments should:
1. Use BN factorization for Z variables (independent ranks + conditional CDFs)
2. Use Gaussian copula ONLY for linking Y to Z (after Z is fully generated)
3. Make this distinction explicit in the paper text

**Last Updated:** 2026-02-04

---

## CRITICAL: Sampling from Gaussian Copula BN with Conditional Marginals

### The Canonical Algorithm (Markov-Preserving)

**Goal:** Sample from a DAG Z_1 → Z_2 → Z_3 where each variable has a conditional marginal F_{k|pa(k)} but the overall copula structure is Gaussian.

**Full Algorithm (conceptually clean):**
```
For each node k in topological order:
  1. Draw Z_k | Z_{pa(k)} ~ N(μ_k(Z_pa), σ_k²)    # From Gaussian BN
  2. U_k = Φ((Z_k - μ_k) / σ_k)                   # Conditional rank (INDEPENDENT of parents)
  3. X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})      # Transform through conditional marginal
```

**Shortcut (mathematically equivalent, simpler):**
```
For each node k in topological order:
  1. U_k ~ Unif(0,1)                              # Independent by definition
  2. X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})      # Transform through conditional marginal
```

### Why These Are Equivalent

In the full algorithm, step 2 computes:
- U_k = Φ((Z_k - μ_k) / σ_k)
- Where (Z_k - μ_k) ~ N(0, σ_k²) is independent of Z_{pa(k)} by construction
- So U_k ~ Unif(0,1) and U_k ⊥ Z_{pa(k)}

This is exactly what `U_k ~ Unif(0,1)` gives directly!

### Why the Markov Property Holds

The data are generated from a recursive SEM:
```
X_1 = g_1(U_1)
X_2 = g_2(X_1, U_2)
X_3 = g_3(X_2, U_3)
```
with independent innovations (U_1, U_2, U_3). This directly yields X_1 ⊥ X_3 | X_2.

### When to Use Each Version

**Use the full Gaussian intermediate when:**
- Fitting copula correlations from empirical data
- You need a unified latent Gaussian representation
- For theoretical clarity or proofs

**Use the shortcut when:**
- Conditional marginals are specified directly (simulation)
- Simplicity is preferred
- You control the DGP

### Common Mistake (What Broke Our Code)

**WRONG approach that causes Markov violations:**
```
1. Fit a joint copula to ALL variables
2. Extract conditional ranks U_{k|pa} from this copula  ← These ARE independent of parents!
3. Transform through conditional marginals F^{-1}_{k|pa}
```

**Why this fails:**
- Step 2 extracts ranks that are ALREADY independent of parents
- Step 3 then transforms through conditional CDFs
- But conditional CDFs need dependence to come from SOMEWHERE
- Result: No dependence created → violates intended DAG structure

**The fix:** Either use the canonical algorithm (independent ranks + conditional marginals) OR use correlated ranks with MARGINAL marginals. Never mix conditional ranks with conditional marginals from different sources.

### R Implementation Pattern

```r
# CORRECT: Independent ranks + conditional marginals
generate_bn_sample <- function(n) {
  # Root node
  U_1 <- runif(n)
  X_1 <- qgamma(U_1, shape = SHAPE_1, scale = SCALE_1)

  # Child node: parameters depend on parent VALUE
  U_2 <- runif(n)  # INDEPENDENT
  shape_2 <- BASE_SHAPE_2 + EFFECT * X_1  # Parameter depends on X_1
  X_2 <- qgamma(U_2, shape = shape_2, scale = SCALE_2)

  return(list(X_1 = X_1, X_2 = X_2))
}
```

### Reference

This algorithm is the "conditional ranks through conditional marginals" approach for Gaussian copula BNs. The key insight is that dependence comes from the conditional CDF parameters (which depend on parent values), NOT from correlated ranks

---

## Update: 2026-02-05

### Session Completed: Markov Property Fix Verified

**Major achievement:** Successfully fixed and validated the Markov property violations in longitudinal causal validation experiments.

**Key Results (200 simulations, N=5,000):**
- Markov property preserved: Mean partial correlation = -0.003 (vs 0.076 before fix)
- KS test for p-value uniformity: p=0.928 (Z1), p=0.618 (Z2) - perfect uniformity
- All causal estimators unbiased: IPW bias=0.0004, G-comp bias=0.002, AIPW bias=0.002
- Chain structure verified through rank uniformity tests

**Power validation confirmed:**
- Tests correctly detect dependencies when conditioning variables omitted
- Full conditioning: pcor ~0.005, p>0.4 (pass)
- Omit Z1_t: pcor=0.078, p<0.0001 (correctly detected)
- Omit Z2_t: pcor=0.093, p<0.0001 (correctly detected)
- Omit Y_{t-1}: pcor=0.235, p<0.0001 (correctly detected)
- Marginal: cor=0.461, p<0.0001 (correctly detected)

**The Solution:**
Replaced the old `generate_longitudinal_data()` function with BN parameterization approach:
- Z variables use independent Uniform ranks (NOT extracted from joint copula)
- Transform through conditional CDFs with parent-dependent parameters
- Y uses Gaussian copula for dependence on Z (after Z is fully generated)

**Key Implementation Pattern:**
```r
# CORRECT: Independent ranks + conditional marginals
U_Z2 <- runif(n)  # Independent!
shape_Z2 <- BASE_SHAPE + BETA * Z1  # Parameter depends on parent VALUE
Z2 <- qgamma(U_Z2, shape = shape_Z2, scale = SCALE)
```

**Verification Process:**
1. Quick test with N=10,000 showed immediate improvement (pcor ~0.005 vs 0.076)
2. Full simulation with 200 reps confirmed statistical validity
3. Power tests verified the tests can detect real dependencies
4. Comparison with ChatGPT confirmed our shortcut is mathematically equivalent to canonical Gaussian BN algorithm

**Files Successfully Modified:**
- `nonparanormal/causal_validation_longitudinal.R` - Replaced function with corrected implementation

**Files for Cleanup:**
- `nonparanormal/test_should_fail.R` - Temporary power validation script
- `nonparanormal/validation_output.log` - Temporary output log

**Next Steps Remaining:**
- Update paper text to clarify BN parameterization approach (Section 6.5)
- Check static and dynamic validation experiments for consistency
- Consider methodological note explaining the two approaches (BN vs vine)
- Commit all changes

**Key Lesson Learned:**
When using conditional CDFs F(Z_j | pa(j)), you MUST use independent ranks. The dependence comes from the conditional CDF parameters (which depend on parent values), NOT from correlated ranks. Mixing conditional ranks from a copula with conditional CDFs creates no dependence and violates the intended structure.

**Last Updated:** 2026-02-05
