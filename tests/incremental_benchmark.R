## incremental_benchmark.R — quick before/after comparison for a single fix
## Run from package root: Rscript tests/incremental_benchmark.R

library(bench)

pkg_dir <- normalizePath(getwd())
devtools::load_all(pkg_dir, quiet = TRUE)

generate_data <- function(n, p, s, beta_mag, noise_sd, seed) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- rep(0, p)
  beta_true[seq_len(s)] <- beta_mag
  Y <- as.numeric(X %*% beta_true) + rnorm(n, sd = noise_sd)
  set.seed(seed + 1000)
  X_test <- matrix(rnorm(n * p), n, p)
  Y_test <- as.numeric(X_test %*% beta_true) + rnorm(n, sd = noise_sd)
  list(X = X, Y = Y, X_test = X_test, Y_test = Y_test, beta_true = beta_true)
}

settings <- list(
  sparse_easy = list(n = 200, p = 500, s = 5, beta_mag = 3, noise_sd = 1, seed = 101),
  moderate    = list(n = 200, p = 500, s = 25, beta_mag = 1, noise_sd = 1, seed = 202),
  dense_hard  = list(n = 150, p = 600, s = 60, beta_mag = 0.5, noise_sd = 1, seed = 303),
  low_dim     = list(n = 500, p = 200, s = 10, beta_mag = 2, noise_sd = 1, seed = 404),
  high_p      = list(n = 200, p = 1000, s = 20, beta_mag = 1, noise_sd = 1, seed = 505)
)

baseline_file <- file.path(pkg_dir, "tests", "baseline_v0.1.0.rds")
has_baseline <- file.exists(baseline_file)
if (has_baseline) {
  baseline <- readRDS(baseline_file)
  message("Loaded baseline from ", baseline_file)
} else {
  message("No baseline file found — running as fresh capture")
}

incremental_file <- file.path(pkg_dir, "tests", "incremental_baseline.rds")
has_incremental <- file.exists(incremental_file)
if (has_incremental) {
  inc_baseline <- readRDS(incremental_file)
  message("Loaded incremental baseline from ", incremental_file)
}

results <- list()
all_pass <- TRUE
for (nm in names(settings)) {
  s <- settings[[nm]]
  message(sprintf("\n=== %s (n=%d, p=%d, s=%d) ===", nm, s$n, s$p, s$s))

  dat <- generate_data(s$n, s$p, s$s, s$beta_mag, s$noise_sd, s$seed)

  set.seed(s$seed)
  fit <- spexvb(X = dat$X, Y = dat$Y)

  coef <- fit$mu * fit$omega
  y_pred_test <- cbind(1, dat$X_test) %*% fit$beta
  mspe <- mean((dat$Y_test - y_pred_test)^2)

  set.seed(s$seed)
  bm <- bench::mark(
    spexvb(X = dat$X, Y = dat$Y),
    iterations = 5,
    check = FALSE
  )

  results[[nm]] <- list(
    coef = coef, beta = fit$beta, mspe = mspe,
    iterations = fit$iterations, converged = fit$converged,
    alpha_vec = fit$alpha_vec,
    convergence_criterion = fit$convergence_criterion,
    timing = bm
  )

  if (has_incremental) {
    old <- inc_baseline$results[[nm]]
    coef_ok <- isTRUE(all.equal(old$coef, coef, tolerance = 1e-8))
    mspe_delta <- (mspe - old$mspe) / old$mspe
    mspe_ok <- abs(mspe_delta) < 0.01
    iter_ok <- fit$iterations <= old$iterations
    time_old <- as.numeric(old$timing$median) * 1000
    time_new <- as.numeric(bm$median) * 1000

    pass <- coef_ok && mspe_ok && iter_ok
    if (!pass) all_pass <- FALSE

    message(sprintf("  Coef OK: %s | MSPE delta: %+.2f%% | Time: %.0f -> %.0f ms | Iters: %d -> %d | %s",
                    coef_ok, mspe_delta * 100,
                    time_old, time_new,
                    old$iterations, fit$iterations,
                    if (pass) "PASS" else "FAIL"))
  } else {
    message(sprintf("  MSPE=%.4f, iters=%d, converged=%s, median=%s",
                    mspe, fit$iterations, fit$converged, format(bm$median)))
  }
}

if (!all_pass && has_incremental) {
  message("\n*** VALIDATION FAILED — REVERT THIS CHANGE ***")
} else {
  if (has_incremental) message("\n*** ALL CHECKS PASSED ***")
  saveRDS(list(
    timestamp = Sys.time(), settings = settings, results = results
  ), file = incremental_file)
  message(sprintf("Incremental baseline saved to %s", incremental_file))
}
