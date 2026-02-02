# Current Plan

**Goal:** Extend Markov experiment to include treatment variable and causal effects
**Started:** 2026-02-02
**Status:** Handoff - Next Task Identified

## Completed Steps
- [x] Read existing R code (nonparanormal.R, simulation_expt_v2.R)
- [x] Create output directory `nonparanormal/plots/markov_results/`
- [x] Implement `generate_longitudinal_confounders()` function
- [x] Implement `generate_outcome_at_time()` function
- [x] Implement `run_markov_experiment()` main function
- [x] Add statistical tests (conditional independence, marginal dependence)
- [x] Add plotting and summary output
- [x] Write complete script to `nonparanormal/simulation_expt_markov.R`

## Next Steps (User Request)
- [ ] Add treatment variable X_t to longitudinal model
- [ ] Specify causal margin: Y_t | do(X_t) ~ Normal(X_t + 1, 1)
- [ ] Update vine specification to link Y_t to both X_t and Z^t
- [ ] Verify Markov property preserved: Y_t ⊥ Z^{t-1} | (Z^t, X_t)
- [ ] Add tests for causal margin preservation

## Notes
- Script reuses existing functions from `nonparanormal.R`:
  - `simulateConditionalOutcomeSamples()`
  - `bootstrappedCondIndTest_GCM()`
  - `bootstrappedKendallTest()`
  - `fitMVGaussianCopula()`
- Model uses Gamma BN (non-linear, non-Gaussian) to demonstrate generality
- Key insight: Markov property emerges from vine specification (only linking Y_t to Z^t)
- Experiment addresses competing paper's claim about Gaussian copula insufficiency

## Files Created
- `nonparanormal/simulation_expt_markov.R` - Main experiment script
- `nonparanormal/plots/markov_results/` - Output directory for plots

## Expected Output
After running the experiment:
1. P-value histograms for conditional independence tests (should be uniform)
2. P-value histograms for marginal dependence tests (should be concentrated near 0)
3. Summary table with KS test results
4. `experiment_results.rds` saved for later analysis
