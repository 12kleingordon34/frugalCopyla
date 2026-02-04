# nonparanormal

R package implementing nonparanormal approximation for frugal causal simulation.

## Overview

This package provides tools for:
- Simulating from marginal structural models (MSMs) with exact causal effects
- Combining Bayesian Networks with Pair Copula Constructions (PCCs)
- Enabling "frugal parameterization" where causal margins are explicitly specified

## Installation

### Development Installation

```r
# Clone the repository
# Then from within the nonparanormal directory:

# Option 1: Using devtools
devtools::install()

# Option 2: Using renv (recommended for reproducibility)
source("setup_renv.R")
```

### Dependencies

Core packages:
- `copula` - Copula modeling
- `VineCopula` - Vine copula structures
- `ppcor` - Partial correlations

Optional packages (for testing):
- `GeneralisedCovarianceMeasure` - GCM independence tests
- `CondIndTests` - Conditional independence tests
- `bnlearn` - Bayesian network tests

## Quick Start

```r
library(nonparanormal)

# Generate covariates from a Bayesian Network
set.seed(42)
n <- 1000

U1 <- runif(n)
Z1 <- qgamma(U1, shape = 2, scale = 2)

U2_1 <- runif(n)
Z2 <- qgamma(U2_1, shape = 2 + 1.5 * Z1, scale = 1)

# Prepare data
covariate_data <- cbind(Z1, Z2)
cond_covariate_ranks <- cbind(U1, U2_1)

# Define copula structure
topoOrder <- c(2, 1)  # Z2 first, then Z1
vine_cor_params <- c(0.5, 0.3)  # Correlations with outcome

# Generate outcome with nonparanormal approximation
result <- simulateConditionalOutcomeSamples(
  covariate_data = covariate_data,
  cond_covariate_ranks = cond_covariate_ranks,
  vine_cor_params = vine_cor_params,
  topoOrder = topoOrder
)

# Outcome samples (uniform on [0, 1])
outcome_ranks <- result$outcomeRankSamples

# Transform to Gaussian
Y <- qnorm(outcome_ranks)
```

## Running Experiments

Experiments are configured via YAML files:

```bash
# Run from the experiments directory
cd experiments
Rscript run_experiment.R config/longitudinal_markov.yaml
```

Available experiment configurations:
- `static_v2.yaml` - 3-variable Gamma Bayesian Network
- `longitudinal_markov.yaml` - Longitudinal Markov model (T=3)
- `treatment_causal.yaml` - Treatment with causal margin preservation

## Testing

```r
# Run all tests
devtools::test()

# Run specific test file
testthat::test_file("tests/testthat/test-correlation.R")
```

## Package Structure

```
nonparanormal/
├── R/                          # Package source code
│   ├── simulate.R              # Vine simulation functions
│   ├── copula_fit.R            # Gaussian copula fitting
│   ├── correlation.R           # Correlation computations
│   ├── vine_transform.R        # Vine reparameterization
│   ├── rank_transform.R        # Conditional to marginal ranks
│   ├── outcome_generation.R    # Outcome sampling
│   ├── independence_tests.R    # Bootstrapped CI tests
│   ├── visualization.R         # Plotting functions
│   └── experiment_framework.R  # YAML config and runner
├── tests/testthat/             # Test suite
├── experiments/                # Experiment scripts
│   ├── config/                 # YAML configurations
│   ├── run_experiment.R        # Main runner
│   └── legacy/                 # Original scripts
├── results/                    # Experiment outputs
│   ├── raw/
│   ├── processed/
│   └── figures/
├── DESCRIPTION                 # Package metadata
├── NAMESPACE                   # Exports
└── renv.lock                   # Dependency lockfile
```

## Key Functions

### Simulation
- `simulateRVineData()` - Simulate from R-vine copula
- `simulateAndReparameterizeVine()` - Simulate and reparameterize for nonparanormal

### Outcome Generation
- `simulateConditionalOutcomeSamples()` - Generate outcomes with conditional ranks
- `simulateMarginalOutcomeSamples()` - Generate outcomes with marginal ranks

### Correlation
- `computeFullCorMatrix()` - Full correlation from partial
- `computePartialCorrelations()` - Partial from full correlation

### Independence Tests
- `bootstrappedCondIndTest_GCM()` - GCM-based conditional independence
- `bootstrappedKendallTest()` - Kendall's tau for marginal dependence

### Visualization
- `plot_pvalue_histogram()` - P-value distribution with KS test
- `theme_nonparanormal()` - Consistent plotting theme

## References

- Evans (2023) - Original frugal parameterization
- Lin et al. (2025) - Exact simulation from longitudinal MSMs
- Bedford & Cooke (2002) - Vine copula foundations

## Authors

- Daniel de Vassimon Manela (Oxford)
- Xi Lin (Oxford)
- Chase Mathis (Duke)
- Robin J. Evans (Oxford)

## License

MIT
