# Codebase Map

**Last Updated:** 2026-02-14 21:00
**Update Trigger:** Subsection 2.1 rewrite (no new files, structure unchanged)

## Directory Structure
```
frugalCopyla/
├── .claude/                    # Context management files
│   └── archive/               # Rotated old tracking files
├── Hybrid-Frugal-Paper/        # LaTeX manuscript for submission
│   ├── main.tex               # Main document
│   ├── refs.bib               # Bibliography
│   ├── shortcuts_v1_jrssb.tex # LaTeX macros (JRSS-B style)
│   ├── sections/              # Paper sections
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── background.tex     # Section 2 (reordered: MSMs → Copulas → PCCs → Uniqueness → BNs)
│   │   ├── parameterizing.tex # Natural/feasible definitions
│   │   ├── hybrid_frugal.tex  # Incorrect CDF consequences
│   │   ├── survival.tex       # Feasibility conditions (main results)
│   │   ├── nonparanormal.tex  # Nonparanormal approximation + experiments
│   │   ├── conclusion.tex
│   │   └── appendix.tex       # Includes all modular appendices
│   ├── appendices/            # Modular appendix files (NEW structure)
│   │   ├── integral_pcc.tex      # App A: Integral PCCs counterexample
│   │   ├── vine_sampling.tex     # App B: Vine sampling by inversion
│   │   ├── feasible_examples.tex # App C: Feasible/infeasible examples
│   │   ├── proofs.tex            # App D: Proofs (h-function, CDF, copula bound)
│   │   ├── nonparanormal.tex     # App E: Nonparanormal reparameterization
│   │   └── ci_pvalues.tex        # App F: CI test p-value histograms
│   └── images/                # Figures
├── frugalCopyla/              # Main Python package
│   ├── __init__.py            # Package init
│   ├── model.py               # Core model definitions
│   ├── copula_lpdfs.py        # Copula log-PDFs (Gaussian, Clayton, etc.)
│   ├── copula_hfunctions.py   # h-functions for inversion sampling
│   ├── diagnostics.py         # Diagnostic tools
│   └── jax_kernels.py         # JAX kernel implementations
├── nonparanormal/             # ✨ NEW: R package for nonparanormal experiments
│   ├── DESCRIPTION            # Package metadata (v0.1.0)
│   ├── NAMESPACE              # Exported functions
│   ├── LICENSE                # MIT license
│   ├── README.md              # Package documentation
│   ├── .gitignore             # R package ignore patterns
│   ├── .Rprofile              # renv activation
│   ├── setup_renv.R           # Dependency management
│   ├── R/                     # ✨ NEW: Modular code (8 files)
│   │   ├── simulate.R         # Vine simulation functions
│   │   ├── copula_fit.R       # Copula fitting
│   │   ├── correlation.R      # Partial correlation utilities
│   │   ├── vine_transform.R   # Nonparanormal transformation
│   │   ├── rank_transform.R   # Rank transformation helpers
│   │   ├── outcome_generation.R # Outcome simulation
│   │   ├── independence_tests.R # Bootstrap CI tests (5 methods)
│   │   └── visualization.R    # Plotting utilities
│   ├── tests/                 # ✨ NEW: Test suite
│   │   └── testthat/          # 8 test files matching R/ modules
│   │       ├── test-simulate.R
│   │       ├── test-copula_fit.R
│   │       ├── test-correlation.R
│   │       ├── test-vine_transform.R
│   │       ├── test-rank_transform.R
│   │       ├── test-outcome_generation.R
│   │       ├── test-independence_tests.R
│   │       └── test-visualization.R
│   ├── experiments/           # ✨ NEW: Experiment framework
│   │   ├── run_experiment.R   # Main experiment runner
│   │   ├── config/            # YAML-based configuration
│   │   │   ├── simple_gaussian.yaml      # Basic 2-var Gaussian
│   │   │   ├── gamma_vine.yaml           # Gamma BN + Gaussian vine
│   │   │   └── longitudinal_markov.yaml  # Time-series model
│   │   └── legacy/            # Original monolithic scripts (preserved)
│   │       ├── nonparanormal.R           # Original 1100-line script
│   │       ├── simulation_expt_v1.R      # Model M_A
│   │       └── simulation_expt_v2.R      # Model M_B
│   ├── results/               # Output directories
│   │   ├── simple_gaussian/
│   │   ├── gamma_vine/
│   │   └── longitudinal_markov/
│   ├── plots/markov_results/  # Markov experiment outputs
│   ├── demo.R / demo_3d.R     # Demonstrations (legacy)
│   ├── gumbel_demo.R          # Gumbel copula examples (legacy)
│   └── p_value_test.R         # Statistical testing (legacy)
├── nonparametric/             # Nonparametric methods (experimental)
├── examples/                   # Usage examples and demos
│   ├── demos/                 # Interactive demos
│   ├── runtime/               # Runtime benchmarking vs causl
│   ├── validation/            # Validation experiments
│   └── inversion/             # Inversion-related examples
├── tests/                      # Python test suite
│   └── test_copula_functions.py
├── CLAUDE.md                   # Project context (paper + code)
├── README.md                   # Project documentation
├── setup.py                    # Python package configuration (v0.0.2)
└── requirements.txt            # Python dependencies
```

## Key Files - Paper

| File | Purpose | Status |
|------|---------|--------|
| `Hybrid-Frugal-Paper/main.tex` | Main manuscript | Active |
| `sections/background.tex` | Section 2 (MSMs, Copulas, PCCs, Uniqueness, BNs) | Restructured 2026-02-14 |
| `sections/parameterizing.tex` | Natural/feasible definitions | Complete |
| `sections/hybrid_frugal.tex` | Consequences of incorrect CDFs | Complete |
| `sections/survival.tex` | Feasibility conditions (Conditions 1-4) | Complete |
| `sections/nonparanormal.tex` | Approximation + experiments | Complete |
| `sections/appendix.tex` | Includes all modular appendices | Complete |
| `appendices/integral_pcc.tex` | App A: Integral PCCs counterexample | Added 2026-02-14 |
| `appendices/vine_sampling.tex` | App B: Vine sampling by inversion | Modularized 2026-02-13 |
| `appendices/feasible_examples.tex` | App C: Feasible/infeasible examples | Modularized 2026-02-13 |
| `appendices/proofs.tex` | App D: Proofs (h-function, CDF, copula bound) | Modularized 2026-02-13 |
| `appendices/nonparanormal.tex` | App E: Nonparanormal reparameterization | Modularized 2026-02-13 |
| `appendices/ci_pvalues.tex` | App F: CI test p-value histograms | Modularized 2026-02-13 |

## Key Files - Code (Python)

| File | Purpose | Paper Section | Status |
|------|---------|---------------|--------|
| `frugalCopyla/model.py` | Core frugal model | §3 | Incomplete (~40%) |
| `frugalCopyla/copula_lpdfs.py` | Copula log-PDFs | §2.1-2.3 | Stable |
| `frugalCopyla/copula_hfunctions.py` | h-functions | §2.4, Lemma 3.5-3.6 | Incomplete (missing inverse) |

## Key Files - Code (R Package) ✨ NEW

### Package Infrastructure
| File | Purpose | Status |
|------|---------|--------|
| `nonparanormal/DESCRIPTION` | Package metadata | Complete |
| `nonparanormal/NAMESPACE` | Function exports | Complete |
| `nonparanormal/LICENSE` | MIT license | Complete |
| `nonparanormal/README.md` | Documentation | Complete |

### Core Modules (R/)
| File | Functions | Paper Section | Status |
|------|-----------|---------------|--------|
| `simulate.R` | simulateRVineData, simulateAndReparameterizeVine | §6 | Stable |
| `copula_fit.R` | fitMVGaussianCopula | §6 | Stable |
| `correlation.R` | computeFullCorMatrix, computePartialCorrelations | §6 | Stable |
| `vine_transform.R` | updateVineMatrices | §6, Algorithm 1 | Stable |
| `rank_transform.R` | uncondition_conditional_ranks, condition_copula_ranks | §6 | Stable |
| `outcome_generation.R` | simulateMarginalOutcomeSamples, simulateConditionalOutcomeSamples | §3, §6 | Stable |
| `independence_tests.R` | 5 bootstrap CI tests (Kendall, KCI, GCM, etc.) | §6 validation | Stable |
| `visualization.R` | Plotting helpers | -- | Stable |

### Experiment Configs
| File | Model | Paper Section | Status |
|------|-------|---------------|--------|
| `config/simple_gaussian.yaml` | 2-var Gaussian copula | -- | Complete |
| `config/gamma_vine.yaml` | Gamma BN + Gaussian vine | §6.3 Model M_B | Complete |
| `config/longitudinal_markov.yaml` | Time-series Markov | Extension | Complete |

### Legacy Scripts (Preserved)
| File | Purpose | Status |
|------|---------|--------|
| `experiments/legacy/nonparanormal.R` | Original 1100-line monolith | Archived |
| `experiments/legacy/simulation_expt_v1.R` | Model M_A (Clayton D-vine) | Archived |
| `experiments/legacy/simulation_expt_v2.R` | Model M_B experiments | Archived |

## Paper Section Mapping

| Section | Title | Key Content |
|---------|-------|-------------|
| §1 | Introduction | Motivation, hybrid BN-copula challenge |
| §2 | Background | Copulas, PCCs, h-functions, frugal param |
| §3 | Parameterizing Frugal Models | Hierarchy of needs, natural/feasible defs |
| §4 | Non-Feasible Models | Theorem 4.1-4.3, error bounds |
| §5 | Conditions for Feasibility | Conditions 1-4, Theorems 5.1-5.3 |
| §6 | Nonparanormal Approximation | Algorithm 1, experiments M_A, M_B |

## Key Theorems & Results

| Result | Location | Summary |
|--------|----------|---------|
| Lemma 3.5 | §2.4, App C.1 | h-function "flipping" trick |
| Lemma 3.6 | §2.4, App C.2 | Large h-function trick for vines |
| Theorem 4.1 | §4.1 | Mismatched CDFs → unintended independencies |
| Theorem 4.2 | §4.2 | CDF errors bias correct variable's margin |
| Theorem 4.3 | §4.2 | Upper bound via Onicescu energy |
| Theorem 5.1 | §5.2 | Feasibility when Q = ∅ |
| Theorem 5.2 | §5.2 | Feasibility with single non-parent ancestor |
| Theorem 5.3 | §5.2 | Feasibility with multiple ancestors |

## Dependencies

### Python
| Package | Purpose | Version |
|---------|---------|---------|
| numpy | Array operations | 1.24.1 |
| matplotlib | Plotting | 3.6.2 |
| optax | JAX optimizer | 0.1.4 |
| tensorflow_probability | Copula distributions | 0.18.0 |
| patsy | Formula parsing | 0.5.3 |
| jax/jaxlib | Autodiff (manual install) | -- |

### R (nonparanormal package)
| Package | Purpose | Where Used |
|---------|---------|------------|
| VineCopula | Vine copula fitting | simulate.R, copula_fit.R |
| copula | Base copula functions | copula_fit.R |
| ggplot2 | Plotting | visualization.R |
| GeneralisedCovarianceMeasure | GCM tests | independence_tests.R |
| bnlearn | CI tests | independence_tests.R |
| CondIndTests | CIT tests | independence_tests.R |
| KernelCI | KCI tests | independence_tests.R |
| yaml | Config parsing | run_experiment.R |
| testthat | Testing | tests/testthat/ |

## Patterns & Conventions

### Code Architecture
- **Python (`frugalCopyla/`)**: MCMC-based simulation for general frugal models
- **R (`nonparanormal/`)**: Nonparanormal approximation with vine copulas
- **Separation**: Python = theory, R = experiments/validation

### R Package Structure
- **R/**: Reusable library functions (exported via NAMESPACE)
- **experiments/**: Usage examples and paper experiments
- **tests/**: Validation via testthat
- **YAML configs**: Experiment specifications (not R scripts)

### Naming Conventions
- Python: `copula_[family]_lpdf`, `copula_[family]_hfunction`
- R: `camelCase` for functions, `snake_case` for internal helpers
- Tests: `test-[module].R` matching `R/[module].R`
- Configs: `[experiment_name].yaml` in `experiments/config/`

### Paper
- Use "frugal" (not "parsimonious")
- Distinguish "natural" vs "feasible" carefully
- Always specify marginal vs conditional CDFs
- Cite Evans (2023) for frugal param, Lin et al. (2025) for h-function tricks

## Gotchas

### Code
- **JAX versions**: Commented out in setup.py, requires manual install
- **Gumbel copula**: Lacks closed-form inverse h-function (numerical)
- **vcov namespace**: Must use `stats::vcov()` not `vcov()` (R)
- **R-vine matrices**: Diagonal must be sequential (VineCopula requirement)
- **topoOrder default**: `seq(d, 1)` includes outcome Y (usually wrong!)
- **GCM tests**: OpenMP threading warnings on macOS (platform-specific, not a bug)

### Paper
- Section titles may not match file names (e.g., survival.tex = feasibility conditions)
- shortcuts_v1_jrssb.tex contains critical macros for compilation

### Package Development
- **Test coverage**: Currently ~60%, aim for 80%+
- **Legacy scripts**: Kept in experiments/legacy/ for reference
- **YAML validation**: No schema validation yet (add if needed)

## Recently Modified

### Session 2026-02-14 (Section 2 Restructuring)
- [x] `sections/background.tex` — Subsection reordering: MSMs → Copulas → PCCs → Uniqueness → BNs (uncommitted)
- [x] `appendices/integral_pcc.tex` — Removed duplicate topological ordering example (committed 01248ee)
- [x] `appendices/nonparanormal.tex` — Fixed cross-reference to integral_pcc (committed 01248ee)
- [x] `sections/appendix.tex` — Added integral_pcc input as App A (committed 01248ee)

### Session 2026-02-13 (Appendix Modularization)
- [x] `appendices/integral_pcc.tex` — NEW: Integral PCCs counterexample (App A)
- [x] `appendices/vine_sampling.tex` — NEW: Vine sampling by inversion (App B)
- [x] `appendices/feasible_examples.tex` — NEW: Feasible/infeasible examples (App C)
- [x] `appendices/proofs.tex` — NEW: Proofs (h-function, CDF, copula bound) (App D)
- [x] `appendices/nonparanormal.tex` — NEW: Nonparanormal reparameterization (App E)
- [x] `appendices/ci_pvalues.tex` — NEW: CI test p-value histograms (App F)
- [x] `sections/background.tex` — Fixed Remark→remark environment
- [x] `sections/appendix.tex` — Modularized inputs

### Session 2026-02-05 (Markov Fix Complete)
- [x] `nonparanormal/causal_validation_longitudinal.R` — FIXED with BN parameterization, fully verified
  - 200 simulations confirm Markov property preserved (pcor=-0.003)
  - All causal estimators unbiased
  - Power validation tests pass

### Session 2026-02-04 (Markov Debugging)
- [x] `nonparanormal/generate_longitudinal_data_v2.R` — Corrected BN parameterization implementation (~250 lines)
- [x] `nonparanormal/debug_independence.R` — Step-by-step debugging script (~150 lines)
- [ ] `nonparanormal/test_should_fail.R` — Temporary power validation script (DELETE)
- [ ] `nonparanormal/validation_output.log` — Temporary output log (DELETE)

### Session 2026-02-02 23:00-23:45

### Package Infrastructure (7 files) ✨
- [x] `nonparanormal/DESCRIPTION` - Package metadata (v0.1.0)
- [x] `nonparanormal/NAMESPACE` - Exported functions
- [x] `nonparanormal/LICENSE` - MIT license
- [x] `nonparanormal/README.md` - Package documentation
- [x] `nonparanormal/.gitignore` - R package ignore patterns
- [x] `nonparanormal/.Rprofile` - renv activation
- [x] `nonparanormal/setup_renv.R` - Dependency management

### Core Modules (8 files in R/) ✨
- [x] `R/simulate.R` - Vine simulation
- [x] `R/copula_fit.R` - Copula fitting
- [x] `R/correlation.R` - Partial correlations
- [x] `R/vine_transform.R` - Nonparanormal transformation
- [x] `R/rank_transform.R` - Rank utilities
- [x] `R/outcome_generation.R` - Outcome simulation
- [x] `R/independence_tests.R` - Bootstrap CI tests
- [x] `R/visualization.R` - Plotting

### Test Suite (8 files in tests/testthat/) ✨
- [x] `test-simulate.R`
- [x] `test-copula_fit.R`
- [x] `test-correlation.R`
- [x] `test-vine_transform.R`
- [x] `test-rank_transform.R`
- [x] `test-outcome_generation.R`
- [x] `test-independence_tests.R`
- [x] `test-visualization.R`

### Experiment Framework (7 files) ✨
- [x] `experiments/run_experiment.R`
- [x] `experiments/config/simple_gaussian.yaml`
- [x] `experiments/config/gamma_vine.yaml`
- [x] `experiments/config/longitudinal_markov.yaml`
- [x] `experiments/legacy/nonparanormal.R` (moved)
- [x] `experiments/legacy/simulation_expt_v1.R` (moved)
- [x] `experiments/legacy/simulation_expt_v2.R` (moved)

### Previous Sessions
- [x] `CLAUDE.md` — Comprehensive project context (Session 2026-02-02 20:30)
- [x] `.claude/` folder structure — All tracking files (Session 2026-02-02 20:30)
- [x] `simulation_expt_markov.R` — Longitudinal Markov model (Session 2026-02-02 23:00)

## Recently Modified

### This Session (2026-02-02 23:00-23:45)
- All 30+ files in nonparanormal/ package - major refactoring
- Original `nonparanormal.R` moved to `experiments/legacy/`

### Previous Sessions
- `nonparanormal/simulation_expt_markov.R` — Longitudinal experiment
- Legacy scripts (now in experiments/legacy/)

## Current Status

### Python Package (`frugalCopyla/`)
- Status: ~40% complete
- Blockers: Missing inverse h-functions, incomplete model.py
- Next: Complete copula families or deprioritize for R focus

### R Package (`nonparanormal/`) ✨
- Status: ~90% complete, production-ready structure
- Test Coverage: ~60% (all core functions validated)
- Next: Add vignettes, increase coverage to 80%+, CRAN submission (optional)
- Known Issues: GCM OpenMP warnings on macOS (not fixable, platform-specific)

### Paper (`Hybrid-Frugal-Paper/`)
- Status: Complete, submission-ready for JRSS-B
- Implementation: R code ready (90%), Python incomplete (40%)
- Next: Verify experiments match paper figures

## Usage Examples

### Install R Package
```r
# From nonparanormal/ directory
devtools::install()
devtools::test()
devtools::check()
```

### Run Experiment
```r
setwd("experiments/")
source("run_experiment.R")
results <- run_experiment("config/simple_gaussian.yaml")
```

### Core Workflow (Manual)
```r
library(nonparanormal)

# 1. Generate confounders
confounders <- generate_confounders(...)

# 2. Fit copula
copula_fit <- fitMVGaussianCopula(confounders)

# 3. Reparameterize
updated_vine <- updateVineMatrices(...)

# 4. Generate outcome
outcome <- simulateConditionalOutcomeSamples(
  margin_spec = list(name = "normal", params = list(mean = 0, sd = 1)),
  confounders = confounders,
  vine_cor_params = c(0.5, 0.3),
  topOrder = c(2, 1)
)

# 5. Test independence
pvals <- bootstrappedCondIndTest_GCM(outcome, confounders[,1], confounders[,2])
```

## Quick Reference

### Key Functions by Module
- **simulate.R**: `simulateRVineData()`, `simulateAndReparameterizeVine()`
- **copula_fit.R**: `fitMVGaussianCopula()`
- **correlation.R**: `computeFullCorMatrix()`, `computePartialCorrelations()`
- **vine_transform.R**: `updateVineMatrices()`
- **rank_transform.R**: `uncondition_conditional_ranks()`, `condition_copula_ranks()`
- **outcome_generation.R**: `simulateMarginalOutcomeSamples()`, `simulateConditionalOutcomeSamples()`
- **independence_tests.R**: `bootstrappedCondIndTest_GCM()`, `bootstrappedKendallTest()`, etc.

### File Paths (Absolute)
- Package root: `/Users/danielmanela/Library/CloudStorage/GoogleDrive-danielmanela@gmail.com/My Drive/work/Oxford/frugalCopyla/nonparanormal/`
- R modules: `.../nonparanormal/R/`
- Tests: `.../nonparanormal/tests/testthat/`
- Configs: `.../nonparanormal/experiments/config/`
- Legacy: `.../nonparanormal/experiments/legacy/`
