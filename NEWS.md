# spexvb (development version)

## spexvb 0.1.0.9000

### Performance

* Hoisted iteration-constant computations (`sigma`, log-odds,
  `sqrt(tau_b * tau_e)`) out of coordinate-ascent inner loops in all three
  C++ backends (`fit_linear_alpha_remap`, `fit_linear_gamma_hierarchy`,
  `fit_logistic_alpha_remap`).
* Replaced O(np) post-remapping matrix-vector product (`W = X * approx_mean`)
  with O(n) scalar multiply (`W *= alpha`).
* Vectorized `entropy()`, replaced `std::pow(x, 2)` with `x * x`, reused
  precomputed `X_2` in the logistic backend, and removed dead allocations.
* R side: direct C++ function calls instead of `do.call()` string lookup,
  eliminated redundant `lapply` copy in return list, reduced standardization
  from two `scale()` passes to one via `Var = E[X^2] - E[X]^2`.
* Measured 24--33% speedup on moderate to high-p settings with identical
  coefficients (within tolerance 1e-8) and zero iteration count changes.

## spexvb 0.1.0

* Initial CRAN release.
* Implements `spexvb()` (linear PX-VB), `hspvb()` (hierarchical Gamma prior),
  and `spexvb.logistic()` (logistic PX-VB) for high-dimensional variable
  selection with spike-and-slab priors.
* Cross-validation via `cv.spexvb()` and `cv.spexvb.fit()` with parallel
  support through `foreach`.
