test_that("cv.spexvb returns correct CVE with precomputed initials", {
  foreach::registerDoSEQ()

  set.seed(42)
  n <- 100
  p <- 200
  X <- matrix(rnorm(n * p), n, p)
  true_beta <- c(rep(3, 5), rep(0, p - 5))
  Y <- X %*% true_beta + rnorm(n)

  result <- cv.spexvb(
    k = 3,
    X = X,
    Y = Y,
    tau_alpha = c(0, 1e3, 1e5),
    standardize = TRUE,
    intercept = TRUE,
    max_iter = 100L,
    tol = 1e-5,
    seed = 12376,
    verbose = FALSE,
    parallel = FALSE
  )

  expect_true(is.list(result))
  expect_named(result, c("ordered_tau_alpha", "epe_test_k", "CVE", "tau_alpha_opt"))
  expect_equal(result$tau_alpha_opt, 0)
  expect_length(result$CVE, 3)
  expect_equal(nrow(result$epe_test_k), 3)
  expect_equal(ncol(result$epe_test_k), 3)
  expect_true(all(is.finite(result$CVE)))
  expect_true(result$CVE[1] < result$CVE[2])
})

test_that("cv.spexvb precomputed initials match per-fold manual initials", {
  foreach::registerDoSEQ()

  set.seed(42)
  n <- 100
  p <- 200
  X <- matrix(rnorm(n * p), n, p)
  true_beta <- c(rep(3, 5), rep(0, p - 5))
  Y <- X %*% true_beta + rnorm(n)

  tau_alpha_grid <- c(0, 1e3, 1e5)
  seed_val <- 12376

  # Manually compute fold results using spexvb with NULL initials
  # (old behavior: get.initials runs for each call)
  set.seed(seed_val)
  folds <- caret::createFolds(Y, k = 3, list = TRUE)

  manual_epe <- matrix(NA, 3, length(tau_alpha_grid))
  for (i in seq_along(folds)) {
    train_idx <- unlist(folds[-i])
    test_idx <- unlist(folds[i])
    X_train <- X[train_idx, , drop = FALSE]
    Y_train <- Y[train_idx]
    X_test <- X[test_idx, , drop = FALSE]
    Y_test <- Y[test_idx]

    for (j in seq_along(tau_alpha_grid)) {
      fit <- spexvb(
        X = X_train, Y = Y_train,
        tau_alpha = tau_alpha_grid[j],
        standardize = TRUE, intercept = TRUE,
        max_iter = 100L, tol = 1e-5, seed = seed_val
      )
      y_pred <- cbind(1, X_test) %*% fit$beta
      manual_epe[i, j] <- mean((Y_test - y_pred)^2)
    }
  }

  # Run optimized cv.spexvb
  set.seed(seed_val)
  cv_result <- cv.spexvb(
    k = 3, X = X, Y = Y,
    tau_alpha = tau_alpha_grid,
    standardize = TRUE, intercept = TRUE,
    max_iter = 100L, tol = 1e-5,
    seed = seed_val, verbose = FALSE, parallel = FALSE
  )

  expect_equal(
    cv_result$epe_test_k, manual_epe,
    tolerance = 1e-12, ignore_attr = TRUE
  )
  expect_equal(
    as.numeric(cv_result$CVE), colMeans(manual_epe),
    tolerance = 1e-12, ignore_attr = TRUE
  )
})
