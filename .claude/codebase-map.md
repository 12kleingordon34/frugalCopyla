# Codebase Map

**Last Updated:** 2026-02-02 20:30
**Update Trigger:** Paper analysis completed

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
│   │   ├── background.tex     # Copulas, PCCs, h-functions, frugal param
│   │   ├── parameterizing.tex # Natural/feasible definitions
│   │   ├── hybrid_frugal.tex  # Incorrect CDF consequences
│   │   ├── survival.tex       # Feasibility conditions (main results)
│   │   ├── nonparanormal.tex  # Nonparanormal approximation + experiments
│   │   ├── conclusion.tex
│   │   └── appendix.tex       # Proofs, examples
│   ├── images/                # Figures
│   └── appendices/            # Additional appendices
├── frugalCopyla/              # Main Python package
│   ├── __init__.py            # Package init
│   ├── model.py               # Core model definitions
│   ├── copula_lpdfs.py        # Copula log-PDFs (Gaussian, Clayton, etc.)
│   ├── copula_hfunctions.py   # h-functions for inversion sampling
│   ├── diagnostics.py         # Diagnostic tools
│   └── jax_kernels.py         # JAX kernel implementations
├── nonparanormal/             # R scripts for nonparanormal experiments
│   ├── nonparanormal.R        # Core nonparanormal implementation
│   ├── simulation_expt_v1.R   # Model M_A (Clayton D-vine)
│   ├── simulation_expt_v2.R   # Model M_B (Gamma BN + Gaussian vine)
│   ├── simulation_expt_markov.R  # Longitudinal Markov model experiment
│   ├── plots/markov_results/  # Output for Markov experiments
│   ├── demo.R / demo_3d.R     # Demonstrations
│   ├── gumbel_demo.R          # Gumbel copula examples
│   └── p_value_test.R         # Statistical testing
├── nonparametric/             # Nonparametric methods (experimental)
├── examples/                   # Usage examples and demos
│   ├── demos/                 # Interactive demos
│   ├── runtime/               # Runtime benchmarking vs causl
│   ├── validation/            # Validation experiments
│   └── inversion/             # Inversion-related examples
├── tests/                      # Test suite
│   └── test_copula_functions.py
├── CLAUDE.md                   # Project context (paper + code)
├── README.md                   # Project documentation
├── setup.py                    # Package configuration (v0.0.2)
└── requirements.txt            # Python dependencies
```

## Key Files - Paper

| File | Purpose | Status |
|------|---------|--------|
| `Hybrid-Frugal-Paper/main.tex` | Main manuscript | Active |
| `sections/background.tex` | Copulas, PCCs, Sklar, h-functions | Complete |
| `sections/parameterizing.tex` | Natural/feasible definitions | Complete |
| `sections/hybrid_frugal.tex` | Consequences of incorrect CDFs | Complete |
| `sections/survival.tex` | Feasibility conditions (Conditions 1-4) | Complete |
| `sections/nonparanormal.tex` | Approximation + experiments | Complete |
| `sections/appendix.tex` | Proofs for all theorems | Complete |

## Key Files - Code

| File | Purpose | Paper Section | Status |
|------|---------|---------------|--------|
| `frugalCopyla/model.py` | Core frugal model | §3 | Stable |
| `frugalCopyla/copula_lpdfs.py` | Copula log-PDFs | §2.1-2.3 | Stable |
| `frugalCopyla/copula_hfunctions.py` | h-functions | §2.4, Lemma 3.5-3.6 | Stable |
| `nonparanormal/nonparanormal.R` | Nonparanormal approx | §6 | Active |
| `nonparanormal/simulation_expt_v2.R` | Model M_B experiments | §6.3 | Active |
| `nonparanormal/simulation_expt_markov.R` | Longitudinal Markov model | §6 extension | Active - needs treatment |

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

### R
| Package | Purpose |
|---------|---------|
| VineCopula | Vine copula fitting |
| copula | Base copula functions |
| ggplot2 | Plotting |

## Patterns & Conventions

### Code
- JAX-based computation for automatic differentiation
- Copula functions: `copula_[family]_lpdf`, `copula_[family]_hfunction`
- R scripts in `nonparanormal/` for statistical experiments

### Paper
- Use "frugal" (not "parsimonious")
- Distinguish "natural" vs "feasible" carefully
- Always specify marginal vs conditional CDFs
- Cite Evans (2023) for frugal param, Lin et al. (2025) for h-function tricks

## Gotchas

### Code
- JAX/JAXlib versions commented out in setup.py - manual install needed
- Gumbel copula lacks closed-form inverse h-function (numerical)

### Paper
- Section titles may not match file names (e.g., survival.tex = feasibility conditions)
- shortcuts_v1_jrssb.tex contains critical macros for compilation

## Recently Added

### Session 2026-02-02 23:00
- [x] `nonparanormal/simulation_expt_markov.R` — Longitudinal Markov model experiment (~500 lines)
- [x] `nonparanormal/plots/markov_results/` — Output directory for Markov experiment

### Session 2026-02-02 20:30
- [x] `CLAUDE.md` — Comprehensive project context linking paper & code
- [x] `.claude/` folder structure — All tracking files initialized

## Recently Modified
- `nonparanormal/simulation_expt_markov.R` — **NEW**: Longitudinal experiment, needs treatment extension
- `nonparanormal/nonparanormal.R` — Active R experiment code (modified per git status)
- `nonparanormal/simulation_expt_v1.R` — Simulation v1 (modified per git status)
- `nonparanormal/simulation_expt_v2.R` — Simulation v2 (modified per git status)
