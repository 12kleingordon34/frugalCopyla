# Decisions Log

> Document architecture, model, statistical, and design decisions with rationale.

---

## 2026-02-02 - Initialize Context Management
**Type:** Architecture
**Status:** Accepted

### Context
Setting up a new project with comprehensive tracking for frugalCopyla - a research project with both paper and code components.

### Decision
Use the standard context management system with scratchpad, plan, handoff, errors, decisions, codebase-map, and references files.

### Rationale
Enables session continuity, traceable decisions, and verifiable sources. Critical for managing dual Python/R codebase aligned with academic paper.

---

## 2026-02-03 - Two Approaches to Nonparanormal Approximation
**Type:** Statistical Method / Model Choice
**Status:** Analysis Complete, Decision Pending

### Context
When implementing nonparanormal approximation for frugal models, there are two fundamentally different approaches with different preservation/approximation trade-offs. This affects Section 6 of the paper and all simulation experiments.

### Decision Options

**Approach A: Regenerate from Gaussian Copula**
- Sample fresh U from fitted Gaussian copula
- Transform to target margins via inverse CDFs
- Conditional ranks are uniform by construction

**Approach B: Preserve Original Data**
- Keep original Z values from non-Gaussian DGP
- Use their (non-uniform) conditional ranks
- Only regenerate Y using Gaussian vine

### Alternatives Considered

| Aspect | Approach A | Approach B |
|--------|-----------|-----------|
| Marginal distributions | ✅ Exact | ✅ Exact |
| Markov/CI structure | ✅ Exact | ⚠️ ~0.02 violations |
| Z-Z tail dependence | ❌ Gaussian approx | ✅ Preserved |
| Z-Z conditional shapes | ❌ Gaussian copula | ✅ Original DGP |
| Conditional ranks uniformity | ✅ Uniform | ❌ Non-uniform |
| Best use case | Causal inference | Sensitivity analysis |

### Experimental Evidence

With N=30,000 samples testing Markov property Y_t ⊥ Z^{t-1} | Z^t:

| Partial Correlation Test | Approach B | Approach A |
|--------------------------|-----------|-----------|
| ρ(Y_2, Z1^1 \| Z^2) | -0.026 (p<0.001)* | -0.006 (p=0.28) |
| ρ(Y_2, Z2^1 \| Z^2) | -0.016 (p=0.007)* | -0.006 (p=0.31) |
| ρ(Y_3, Z1^2 \| Z^3) | -0.022 (p<0.001)* | -0.004 (p=0.50) |
| ρ(Y_3, Z2^2 \| Z^3) | -0.009 (p=0.12) | -0.002 (p=0.69) |

*Statistically significant violations

Approach A reduces violations by ~4x and eliminates statistical significance.

### Rationale (Pending User Decision)

**Arguments for Approach A (Recommended):**
- Exact Markov/CI preservation is critical for causal inference validity
- Small tail dependence approximation error is acceptable trade-off
- Consistent with "imposing Gaussian copula" framing in paper
- Aligns with typical nonparanormal usage in literature

**Arguments for Approach B:**
- Preserves observed Z-Z dependence patterns exactly
- Useful for sensitivity analysis to copula misspecification
- Small Markov violations (~0.02) may be acceptable for some applications
- More faithful to original data generating process

**Hybrid Option:**
- Use Approach A for main experiments (Section 6)
- Add Approach B as appendix/sensitivity analysis

### Consequences

**If Approach A chosen:**
- Section 6 text should state: "regenerate from fitted Gaussian copula"
- All experiments should use fresh sampling
- Tail dependence preservation is NOT claimed

**If Approach B chosen:**
- Section 6 text should acknowledge small Markov violations
- Need to report violation magnitudes
- Can claim tail dependence preservation

**If Hybrid:**
- Main text uses Approach A
- Appendix compares both approaches
- Provides completeness and robustness

### References
- See `.claude/learnings.md` for detailed comparison
- `nonparanormal/longitudinal_fixed.R` - Experimental comparison

### Related Files
- `nonparanormal/R/simulate.R` - Implementation of vine generation
- `nonparanormal/R/rank_transform.R` - Rank transformation utilities
- `nonparanormal/longitudinal_fixed.R` - Experiment comparing approaches

### Status
**Awaiting user decision on which approach to use in paper.**

---

## 2026-02-04 - Gaussian Copula BN Sampling: Independent Ranks + Conditional Marginals
**Type:** Statistical Method / Algorithm Design
**Status:** Accepted and Fully Verified (2026-02-05)

### Context
When sampling from a Bayesian Network where the copula structure is Gaussian but marginals are arbitrary (conditional on parents), there was confusion about how to correctly generate samples that preserve the Markov property.

The previous implementation incorrectly:
1. Fitted a joint copula to all variables
2. Extracted conditional ranks from this joint copula
3. Transformed through conditional marginals

This caused Markov violations (~0.076 partial correlations) because extracting conditional ranks makes them independent of parents, so applying conditional CDFs created NO dependence.

### Decision
Use the **"independent ranks + conditional marginals"** algorithm:

```
For each node k in topological order:
  1. U_k ~ Unif(0,1)                              # Independent innovation
  2. X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})      # Conditional marginal transform
```

This is equivalent to the full Gaussian copula BN approach:
```
  1. Z_k | Z_{pa} ~ N(μ_k(Z_pa), σ_k²)            # Sample from Gaussian BN
  2. U_k = Φ((Z_k - μ_k) / σ_k)                   # Extract conditional rank
  3. X_k = F^{-1}_{k|pa(k)}(U_k | X_{pa(k)})      # Transform
```

The shortcut works because step 2 produces U_k ~ Unif(0,1) independent of parents.

### Alternatives Considered
| Approach | Pros | Cons |
|----------|------|------|
| Vine copula for all | Handles complex dependencies | Overkill for BN structure |
| Joint copula + extract | Conceptually unified | WRONG: breaks Markov property |
| Independent ranks + conditional | Simple, correct | Less intuitive at first |

### Rationale
- Dependence comes from the conditional CDF PARAMETERS (which depend on parent values), NOT from correlated ranks
- This is the standard way to sample from recursive SEMs: X_k = g_k(X_{pa}, U_k) with independent U_k
- Directly yields Markov property: X_1 ⊥ X_3 | X_2

### Consequences
- `causal_validation_longitudinal.R` now uses this approach
- Markov violations eliminated (pcor from ~0.076 to ~-0.005)
- Causal estimators remain unbiased (IPW bias = -0.001)

### Verification

**Initial Test (N=10,000):**
- Partial corr (Y_t, Z1_{t-1} | Z_t, X_t, Y_{t-1}): **-0.0043** (p=0.67)
- Partial corr (Y_t, Z2_{t-1} | Z_t, X_t, Y_{t-1}): **0.0157** (p=0.12)

**Full Simulation Study (2026-02-05, 200 sims, N=5,000 each):**
- Mean partial correlation: **-0.003** (target: 0)
- KS test for p-value uniformity: p=0.928 (Z1), p=0.618 (Z2) — perfect uniformity
- All causal estimators unbiased: IPW bias=0.0004, G-comp bias=0.002, AIPW bias=0.002
- Power validation confirmed: tests correctly detect dependencies when conditioning variables omitted

### References
- ChatGPT analysis of Gaussian copula BN sampling (2026-02-04)
- See `.claude/learnings.md` for canonical algorithm documentation

### Related Files
- `nonparanormal/causal_validation_longitudinal.R` — Fixed implementation
- `nonparanormal/generate_longitudinal_data_v2.R` — Reference implementation
- `.claude/learnings.md` — Detailed algorithm documentation

---

## 2026-02-02 - Create CLAUDE.md for Paper-Code Context
**Type:** Documentation
**Status:** Accepted

### Context
Project has a comprehensive academic paper (Hybrid-Frugal-Paper) that needs to be tightly linked to the codebase. Standard codebase-map.md insufficient for capturing mathematical definitions, theorems, and paper structure.

### Decision
Create comprehensive CLAUDE.md file (500+ lines) containing:
- Complete paper summary with key contributions
- Full terminology glossary (frugal, feasible, natural, h-functions, vines)
- Mathematical notation reference
- Code-paper mapping showing which files implement which sections
- Section-by-section breakdown with key theorems

### Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| Just use codebase-map.md | Simpler | Too limited for academic context |
| Separate paper-summary.md | Clean separation | Harder to find code-paper links |
| Inline in code comments | Close to implementation | Fragmented, hard to get overview |

### Rationale
Need single source of truth that connects abstract mathematical concepts in paper with concrete implementations in code. Future sessions need to quickly understand what "feasible" or "h-function" means without re-reading the entire paper.

### Consequences
- CLAUDE.md is now required reading for any code changes
- Must keep CLAUDE.md updated if paper changes
- Larger context overhead but massive time savings on understanding

### Related Files
- `/Users/danielmanela/Library/CloudStorage/GoogleDrive-danielmanela@gmail.com/My Drive/work/Oxford/frugalCopyla/CLAUDE.md` — The context file
- `/Users/danielmanela/Library/CloudStorage/GoogleDrive-danielmanela@gmail.com/My Drive/work/Oxford/frugalCopyla/Hybrid-Frugal-Paper/main.tex` — Paper source

---
