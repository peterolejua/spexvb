## temp_benchmark.R — baseline benchmarking for spexvb optimization cycle
## Run from package root: Rscript tests/temp_benchmark.R

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
  n_test <- n
  X_test <- matrix(rnorm(n_test * p), n_test, p)
  Y_test <- as.numeric(X_test %*% beta_true) + rnorm(n_test, sd = noise_sd)

  list(X = X, Y = Y, X_test = X_test, Y_test = Y_test, beta_true = beta_true)
}

settings <- list(
  sparse_easy = list(n = 200, p = 500, s = 5, beta_mag = 3, noise_sd = 1, seed = 101),
  moderate    = list(n = 200, p = 500, s = 25, beta_mag = 1, noise_sd = 1, seed = 202),
  dense_hard  = list(n = 150, p = 600, s = 60, beta_mag = 0.5, noise_sd = 1, seed = 303),
  low_dim     = list(n = 500, p = 200, s = 10, beta_mag = 2, noise_sd = 1, seed = 404),
  high_p      = list(n = 200, p = 1000, s = 20, beta_mag = 1, noise_sd = 1, seed = 505)
)

results <- list()
for (nm in names(settings)) {
  s <- settings[[nm]]
  message(sprintf("=== %s (n=%d, p=%d, s=%d) ===", nm, s$n, s$p, s$s))

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
    coef = coef,
    beta = fit$beta,
    mspe = mspe,
    iterations = fit$iterations,
    converged = fit$converged,
    alpha_vec = fit$alpha_vec,
    convergence_criterion = fit$convergence_criterion,
    timing = bm
  )

  message(sprintf("  MSPE=%.4f, iters=%d, converged=%s, median=%s",
                  mspe, fit$iterations, fit$converged, format(bm$median)))
}

version <- read.dcf(file.path(pkg_dir, "DESCRIPTION"), fields = "Version")[[1]]
rds_file <- file.path(pkg_dir, "tests", sprintf("baseline_v%s.rds", version))
saveRDS(list(
  version = version, timestamp = Sys.time(),
  settings = settings, results = results
), file = rds_file)
message(sprintf("Baseline saved to %s", rds_file))
