# frugalCopyla Project Context

> This file provides comprehensive context for the frugalCopyla project, linking the Python/R codebase with the academic paper "Hybridized Conditional Copula Networks Parameterization for Frugal Causal Simulation".

---

## Project Overview

**frugalCopyla** is a Python package implementing copula-based methods for causal inference, specifically focused on:
1. Simulating from marginal structural models (MSMs) with exact causal effects
2. Combining Bayesian Networks with Pair Copula Constructions (PCCs)
3. Enabling "frugal parameterization" where causal margins are explicitly specified

**Target Journal:** JRSS-B or similar statistical methodology journal

---

## Paper Summary

### Title
"Hybridized Conditional Copula Networks Parameterization for Frugal Causal Simulation"

### Authors
- Daniel de Vassimon Manela (Oxford)
- Xi Lin (Oxford)
- Chase Mathis (Duke)
- Robin J. Evans (Oxford)

### Core Contribution
The paper explores how to combine Bayesian Networks (BNs) with Pair Copula Constructions (PCCs) to create "frugal" causal models where:
- The marginal causal effect p(Y|do(X)) is explicitly parameterized
- Covariate dependencies can be modeled using familiar BN conditional factors
- Dependencies between covariates and outcomes use copulas

### Key Challenge Addressed
When modeling covariates as a BN with conditional factors like p(Z₂|Z₁), the intervened copula φ*(Y,Z) requires **marginal** CDFs F(Z₂), not conditional ones F(Z₂|Z₁). Naively using conditional CDFs creates incorrect dependency structures or biased causal margins.

---

## Key Terminology & Definitions

### Frugal Parameterization
A parameterization of causal models with three components:
1. **Causal margin**: p(Y|do(X)) - explicitly specified
2. **The past**: p(Z,X) = p(X|Z)·p(Z) - covariate and treatment distributions
3. **Intervened dependence measure**: φ*(Y,Z|do(X)) - copula encoding Y-Z dependence

### Feasible vs Natural Frugal Models
- **Natural**: Copula parameterized by minimal variables with direct causal edges to Y
- **Feasible**: All conditional CDFs required can be analytically extracted from the BN without numerical integration

### Frugal Hierarchy of Causal Needs (Priority Order)
1. **Preservation of marginal causal effect** p(Y|do(X)) - CRITICAL
2. **Correct copula parameterization** φ*(Z,Y) - for sensitivity analysis
3. **Conditional dependency structure** - Markov to specified graph
4. **Pre-treatment covariate margins** - ideally identical to target

### h-function
The partial derivative of a copula CDF with respect to one argument:
```
h_{UV|W}(u, v | w) = ∂C_{UV|W}(u, v) / ∂v
```
Used for:
- Computing conditional CDFs from copulas
- Inversion sampling from vines
- "Flipping" conditioning sets (Lemma 3.6)

### Pair Copula Construction (PCC)
Decomposition of multivariate copula into bivariate copulas:
```
p(y|x₁,...,xₐ) = p(y) × ∏ᵢ c_{YXᵢ|X₁:(ᵢ₋₁)}(F_{Y|X₁:(ᵢ₋₁)}, F_{Xᵢ|X₁:(ᵢ₋₁)})
```

### Vine Copula
A specific PCC structure where all conditional CDFs can be computed via h-functions without integration (regular vines satisfy the "proximity condition").

### Nonparanormal Approximation
A distribution where after transforming margins to Gaussian, the joint is multivariate normal. Used to approximate infeasible models with feasible Gaussian copula alternatives.

---

## Paper Structure & Key Results

### Section 2: Background
- Sklar's Theorem (separating margins from dependence)
- PCCs and their evaluation (some require integration)
- h-functions and inversion sampling
- Frugal parameterization basics

### Section 3: Parameterizing Frugal Models
**Key Results:**
- **Lemma 3.1**: Natural + feasible if all covariates prior to max_Y have direct edges to Y
- **Lemma 3.2** (Dynamic): Extension to longitudinal models
- **Definition 3.2**: "History" notation Z̄ₐᵗ for longitudinal settings

### Section 4: Non-Feasible Models & Incorrect CDF Specification
**Key Results:**
- **Theorem 4.1**: Mismatched conditioning sets imply unintended independencies
- **Theorem 4.2**: CDF approximation errors bias the *correctly specified* variable's margin
- **Theorem 4.3**: Upper bound on margin error via Onicescu's Informational Energy

### Section 5: Conditions for Feasible Frugal Models
**Four Conditions for Feasibility:**
1. **Condition 1**: Copula factorization on V follows topological order
2. **Condition 2**: Clique structure on W + downstream copula edges from V
3. **Condition 3**: Local feasibility for non-parent ancestors (contiguous blocks, cliques)
4. **Condition 4**: Global feasibility (nested child sets for dependent ancestors)

**Key Results:**
- **Theorem 5.1**: Feasibility when Q = ∅ (all ancestors are parents)
- **Theorem 5.2**: Feasibility with single non-parent ancestor Q
- **Theorem 5.3**: Feasibility with multiple non-parent ancestors

### Section 6: Nonparanormal Approximation
- Algorithm 1: Sampling from frugal models with nonparanormal approximation
- Experimental validation with Kendall-τ and GCM tests
- Models M_A (Clayton vine) and M_B (Gamma BN + Gaussian vine)

---

## Code-Paper Mapping

### Python Package (`frugalCopyla/`)

| File | Paper Section | Purpose |
|------|---------------|---------|
| `model.py` | §3 | Core frugal model definitions |
| `copula_lpdfs.py` | §2.1-2.3 | Log-PDF implementations for copulas |
| `copula_hfunctions.py` | §2.4, Lemmas 3.5-3.6 | h-functions for inversion sampling |
| `diagnostics.py` | §6 | Validation and diagnostic tools |

### R Scripts (`nonparanormal/`)

| File | Paper Section | Purpose |
|------|---------------|---------|
| `nonparanormal.R` | §6 | Nonparanormal approximation implementation |
| `simulation_expt_v1.R` | §6.3 | Model M_A experiments |
| `simulation_expt_v2.R` | §6.3 | Model M_B experiments |

---

## Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| Y | Outcome variable |
| X | Treatment variable |
| **Z** | Pre-treatment covariates |
| **W** | Parents of Y modeled via BN edges |
| **V** | Parents of Y modeled via copulas |
| **Q** | Non-parent ancestors of Y |
| **T** | All parents of Y (T = V ∪ W) |
| p(Y\|do(X)) | Causal/interventional distribution |
| φ*(Y,Z\|do(X)) | Intervened copula dependence measure |
| F_{A\|B} | Conditional CDF of A given B |
| u_{A\|B} | Conditional rank (quantile) |
| c_{AB\|C} | Bivariate copula density |
| C_{AB\|C} | Bivariate copula CDF |
| h_{A\|BC} | h-function (partial derivative of copula) |
| ρ_{AB\|C} | Partial correlation |
| Z̄ₐᵗ | History up to covariate d at time t |

---

## Copula Families Used

| Family | h-function | Inverse h-function | Tail Dependence | Notes |
|--------|------------|-------------------|-----------------|-------|
| Gaussian | Closed-form | Closed-form | None | Default for nonparanormal |
| Student-t | Closed-form | Closed-form | Symmetric | Heavier tails |
| Frank | Closed-form | Closed-form | None | Symmetric |
| Clayton | Closed-form | Closed-form | Lower | Asymmetric |
| Gumbel | Closed-form | **Numerical** | Upper | Asymmetric |

---

## Key Proofs & Appendices

| Appendix | Content |
|----------|---------|
| A | Vine sampling by inversion algorithm |
| B | Feasible/infeasible example walkthrough |
| C.1 | Proof: h-function flipping trick (Lemma 3.5) |
| C.2 | Proof: Large h-function trick (Lemma 3.6) |
| C.3 | Proof: Erroneous CDF choice (Theorem 4.2) |
| C.4 | Proof: Bivariate copula error bound (Theorem 4.3) |
| D | Nonparanormal reparameterization details |

---

## Common Tasks & How To

### Adding a New Copula Family
1. Implement lpdf in `copula_lpdfs.py`
2. Implement h-function and inverse in `copula_hfunctions.py`
3. Add tests in `tests/test_copula_functions.py`

### Running Nonparanormal Experiments
```r
source("nonparanormal/nonparanormal.R")
source("nonparanormal/simulation_expt_v2.R")
```

### Checking Feasibility of a Model
Use the graphical conditions from Section 5:
1. Check Condition 1 (copula factorization)
2. Check Condition 2 (W clique + V→W edges)
3. Check Condition 3 (Q local constraints)
4. Check Condition 4 (Q global constraints)

---

## Related Papers & References

### Core References
- Evans (2023) - Original frugal parameterization paper
- Lin et al. (2025) - Exact simulation from longitudinal MSMs (arXiv:2502.07991)
- Bedford & Cooke (2002) - Vine copula foundations
- Liu et al. (2009) - Nonparanormal distribution

### Background
- Joe (2011, 2014) - Dependence modeling with copulas
- Czado & Nagler (2022) - Vine copula review
- Aas et al. (2009) - Pair copula constructions

---

## Current Work Status

**Branch:** `inversion`

**Active Areas:**
- Nonparanormal simulation experiments (R)
- Validation of independence preservation
- Paper writing and revision

**Known Issues:**
- JAX/JAXlib versions need manual configuration
- Some copula families lack closed-form inverse h-functions

---

## Writing Style Notes

- Use "frugal" consistently (not "parsimonious")
- Distinguish "natural" (minimal PCC) from "feasible" (no integration needed)
- Always clarify marginal vs conditional CDFs
- Cite Evans (2023) for original frugal parameterization
- Cite Lin et al. (2025) for h-function flipping result
