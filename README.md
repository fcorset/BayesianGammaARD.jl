# BayesianGammaARD

[![Build Status](https://github.com/fcorset/BayesianGammaARD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/fcorset/BayesianGammaARD.jl/actions/workflows/CI.yml?query=branch%3Amain)

## MCMC diagnostics

`algoMCMC` can return Metropolis acceptance rates, effective sample sizes (ESS),
and autocorrelation estimates in addition to the samples:

```julia
result = algoMCMC(gp, df, priors, 10_000; return_diagnostics=true, max_lag=100)
samples = result.samples
result.diagnostics.acceptance_rates  # α, β, ρ; θ is a Gibbs update (not applicable)
result.diagnostics.ess               # ESS for α, β, θ, ρ
result.diagnostics.autocorrelation   # DataFrame indexed by lag, from 0 to max_lag
```

By default, the existing `algoMCMC` return value remains the samples DataFrame.
When diagnostics are requested, their calculations exclude the initial state in
the first row of `samples`; the reported acceptance rates cover the `K` updates.
For diagnostics after discarding burn-in, apply `mcmc_diagnostics` to the retained
rows, for example `mcmc_diagnostics(samples[1001:end, :])`. In standalone calls,
acceptance rates for α, β, and ρ are estimated as the proportion of transitions
that changed the parameter; `θ` is updated by Gibbs sampling, so an acceptance
rate does not apply. If a parameter is constant, its ESS and autocorrelations are
reported as `missing`. The ESS uses the initial-positive-pair truncation of the
autocorrelation sequence and is an estimate, not a convergence guarantee.
