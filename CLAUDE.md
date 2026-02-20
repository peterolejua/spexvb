# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`spexvb` implements **Sparse Parameter-Expanded Variational Bayes** for high-dimensional regression with spike-and-slab priors. Three distinct VB algorithms are provided:

| R function | C++ backend | Model |
|---|---|---|
| `spexvb()` | `fit_linear_alpha_remap` | Linear regression, fixed `tau_b`, alpha expansion + remapping |
| `hspvb()` | `fit_linear_gamma_hierarchy` | Linear regression, hierarchical Gamma prior on `tau_b` |
| `spexvb.logistic()` | `fit_logistic_alpha_remap` | Logistic regression, alpha expansion + remapping |

## Algorithm Structure

All three algorithms follow the same pattern in C++:
1. **Coordinate ascent** over predictors: update `mu_j`, `sigma_j`, `omega_j` (inclusion prob), and `q(pi)` sequentially in the order given by `update_order` (0-indexed).
2. **Alpha step** (`spexvb`/`spexvb.logistic` only): compute the optimal expansion scalar `alpha` in closed form.
3. **Remapping step**: absorb `alpha` into `mu`, `sigma`, `tau_b`, and `mu_alpha` so the next iteration starts at the correct scale.
4. **Convergence check**: `inf`-norm of change in entropy of `omega` plus, for linear PX-VB, a condition on `alpha_diff`.

Initialization (`get.initials`, `get.initials.logistic`) always runs LASSO and Ridge CV via `glmnet` to produce starting `mu_0`, `omega_0`, `tau_e`, `c_pi_0`, `d_pi_0`, and `update_order`.

## Key Files

- [R/spexvb.R](R/spexvb.R) — main exported wrapper; handles standardization, calls C++, unscales results
- [R/hspvb.R](R/hspvb.R) — hierarchical variant wrapper
- [R/spexvb.logistic.R](R/spexvb.logistic.R) — logistic wrapper; includes a fallback loop that increases `tau_alpha` if convergence fails
- [R/get.initials.R](R/get.initials.R) — LASSO/Ridge initialization for linear models
- [R/get.initials.logistic.R](R/get.initials.logistic.R) — logistic initialization
- [R/cv.spexvb.R](R/cv.spexvb.R) — k-fold CV over `tau_alpha` grid using `foreach %dopar%`
- [R/cv.spexvb.fit.R](R/cv.spexvb.fit.R) — calls `cv.spexvb()` then fits final model on full data
- [src/common_helpers.h](src/common_helpers.h) + [src/common_helpers.cpp](src/common_helpers.cpp) — shared C++ utilities: `entropy()`, `sigmoid()`, `gram_diag()`, `r_digamma()`
- [src/fit_linear_alpha_remap.cpp](src/fit_linear_alpha_remap.cpp) — SPEXVB core loop
- [src/fit_linear_gamma_hierarchy.cpp](src/fit_linear_gamma_hierarchy.cpp) — HSPVB core loop
- [src/fit_logistic_alpha_remap.cpp](src/fit_logistic_alpha_remap.cpp) — logistic SPEXVB core loop

## Build & Check Commands

```r
devtools::load_all()          # compile C++ and load
devtools::document()          # rebuild man/ and NAMESPACE from roxygen2
devtools::check()             # full CRAN check
```

There is currently **no test suite** (`tests/` does not exist). When adding tests, create `tests/testthat/` and mirror source file names (e.g., `R/spexvb.R` → `tests/testthat/test-spexvb.R`).

## Optimization Notes (current branch: `optimize`)

- `src/Makevars` currently only sets `PKG_LIBS`. Adding `PKG_CXXFLAGS = -O2` would enable compiler optimization (do not use `-march=native`).
- In `fit_linear_alpha_remap.cpp`, `XtX = X.t() * X` is precomputed but not used in the inner loop — the inner loop recomputes column dot products via `arma::dot(X.col(j), W)`. This is a known optimization opportunity.
- `gram_diag(X)` (squared column norms) is precomputed once before the outer loop — preserve this.
- Follow the Safe Optimization Protocol from global CLAUDE.md before changing any numerical code: benchmark first, verify `all.equal(..., tolerance = 1e-8)`, check iteration count has not increased.

## Parallel CV

`cv.spexvb()` uses `foreach %dopar%` for the outer fold loop. The `parallel` argument controls whether the inner `tau_alpha` grid search also uses `%dopar%` vs `%do%`. Users must register a parallel backend (e.g., `doParallel::registerDoParallel()`) before calling CV functions.
