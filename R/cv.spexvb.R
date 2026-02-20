#' @title Cross-validation for Sparse Parameter Expanded Variational Bayes (spexvb)
#' @description Performs k-fold cross-validation for the spexvb model,
#'   allowing for evaluation of model performance across different tau_alpha values.
#' @param k Integer, the number of folds for cross-validation. Must be greater than 2.
#' @param X A design matrix.
#' @param Y A response vector.
#' @param mu_0 Initial variational mean (posterior expectation of beta_j | s_j = 1). If NULL, initialized automatically.
#' @param omega_0 Initial variational probability (posterior expectation of s_j). If NULL, initialized automatically.
#' @param c_pi_0 Prior parameter for pi (beta distribution shape1). If NULL, initialized automatically.
#' @param d_pi_0 Prior parameter for pi (beta distribution shape2). If NULL, initialized automatically.
#' @param tau_e Initial precision of errors. If NULL, initialized automatically.
#' @param update_order A numeric vector specifying the order of updates for coefficients. If NULL, initialized automatically.
#' @param mu_alpha Mean for the prior on alpha (expansion parameter).
#' @param tau_alpha A numeric vector of tau_alpha values to cross-validate over. Must have at least two values.
#' @param tau_b Initial precision for beta_j (when s_j = 1).
#' @param standardize Logical. Center Y, and center and scale X. Default is TRUE.
#' @param intercept Logical. Whether to include an intercept. Default is TRUE. After the model is fit on the centered and scaled data, the final coefficients are "unscaled" to put them back on the original scale of your data. The intercept is then calculated separately using the means and the final coefficients.
#' @param max_iter Maximum number of outer loop iterations for each spexvb fit.
#' @param tol Convergence tolerance for each spexvb fit.
#' @param seed Seed for reproducibility of data splitting and `glmnet` initials.
#' @param verbose Logical, if TRUE, prints progress messages during cross-validation.
#' @param parallel Logical, if TRUE, search in parallel.
#' @return A list containing cross-validation results:
#'   \item{ordered_tau_alpha}{The sorted vector of tau_alpha values used.}
#'   \item{epe_test_k}{A matrix of prediction errors (MSE) for each fold (rows) and each tau_alpha (columns).}
#'   \item{CVE}{Cross-Validation Error (mean MSE) for each tau_alpha.}
#'   \item{tau_alpha_opt}{The tau_alpha value that minimizes the CVE.}
#' @details This function performs k-fold cross-validation to find the optimal `tau_alpha`
#'   for the `spexvb` model. It iterates through different `tau_alpha` values, trains
#'   the model on training folds, and evaluates performance on the held-out test fold.
#'   To leverage parallel processing, ensure a parallel backend (e.g., from `doParallel` or `doSNOW` packages)
#'   is registered using `registerDoParallel()` or similar before calling this function.
#' @examples
#' \donttest{
#' n <- 50
#' p <- 100
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[,1] * 2 + rnorm(n)
#'
#' # Run cross-validation only (returns errors and optimal tau_alpha)
#' cv_res <- cv.spexvb(k = 3, X = X, Y = Y)
#'
#' # Inspect the optimal tau_alpha
#' print(cv_res$tau_alpha_opt)
#' }
#'
#' @importFrom caret createFolds
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom stats sd
#' @importFrom stats setNames
#' @export
cv.spexvb <- function(
    k = 5, #the number of folds to use
    X, # design matrix
    Y, # response vector
    mu_0 = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = NULL, # \eqn{\pi \sim Beta(a_\pi, b_\pi)}, known/estimated
    d_pi_0 = NULL, # \eqn{\pi \sim Beta(a_\pi, b_\pi)}, known/estimated
    tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = NULL,
    mu_alpha = 1, # alpha is N(mu_alpha, (tau_e*tau_alphalpha)^{-1}), known/estimated
    tau_alpha = c(0,10^(3:7)), # Can be a vector now
    tau_b = 400, # initial. b_j is N(0, (tau_e*tau_b)^{-1}), known/estimated
    standardize = TRUE, # Center Y, and center and scale X
    intercept = TRUE,
    max_iter = 100L, # Ensure it's an integer literal
    tol = 1e-5,
    seed = 12376, # seed for cv.glmnet initials
    verbose = TRUE,
    parallel = TRUE
){

  set.seed(seed)

  ## verification before start -----------------

  ### check whether tau_alpha is a vector ----
  if (!is.null(tau_alpha) && length(tau_alpha) < 2){
    stop("Need more than one value of tau_alpha for cross validation.")
  }

  if( sum(tau_alpha < 0) > 0 ){
    stop("Error Message: Some tau_alpha is less than 0.")
  }


  ### check the number of folds is a whole number of at least two ----
  if (!is.numeric(k) || length(k) != 1 || k != as.integer(k) ){ # Changed check
    stop("The number of folds must be an integer.")
  }

  if(k < 3){
    stop("The number of folds must be greater than two.")
  }

  ### dimensions ----
  n.y <- length(Y)
  n.x <- nrow(X)
  p <- ncol(X)
  if(n.x == n.y){
    n <- n.x
  }else{
    stop("The dimensions for X and Y matrix do not match.")
  }

  if( tol < 0 ){
    stop("Error Message: tol is less than 0.")
  }

  if( max_iter < 0 || !is.integer(max_iter)){ # Changed check
    stop("Error Message: max_iter must be a positive whole number")
  }

  # order the tau_alpha
  ordered_tau_alpha <- sort(tau_alpha, decreasing = FALSE) # Changed to FALSE

  ## Folds creation for k-fold Cross Validation ----
  folds <- caret::createFolds(Y, k = k, list = TRUE)

  # Use %dopar% for parallel execution if a backend is registered
  i <- current_tau_alpha <- NULL # For CRAN
  epe_test_k <- foreach(
    i = 1:k,
    .combine = 'rbind',
    .packages = c('foreach','glmnet', 'caret', 'spexvb') # Explicitly list necessary packages
  ) %dopar% { # Changed from %do% to %dopar%
    train_indices <- unlist(folds[-i])
    test_indices <- unlist(folds[i])

    X_train <- X[train_indices, , drop = FALSE]
    Y_train <- Y[train_indices]
    X_test <- X[test_indices, , drop = FALSE]
    Y_test <- Y[test_indices]

    # Precompute initials once per fold to avoid redundant cv.glmnet calls
    # across the tau_alpha grid. spexvb() standardizes internally before
    # calling get.initials(), so we mirror that standardization here.
    do_std <- standardize || intercept
    if (do_std) {
      fold_means <- colMeans(X_train)
      X_c <- scale(X_train, center = fold_means, scale = FALSE)
      fold_sds <- sqrt(colMeans(X_c^2))
      X_init <- scale(X_c, center = FALSE, scale = fold_sds)
      Y_init <- Y_train - mean(Y_train)
    } else {
      X_init <- X_train
      Y_init <- Y_train
    }

    fold_initials <- get.initials(
      X = X_init,
      Y = Y_init,
      mu_0 = mu_0,
      omega_0 = omega_0,
      c_pi_0 = c_pi_0,
      d_pi_0 = d_pi_0,
      tau_e = tau_e,
      update_order = update_order,
      seed = seed
    )

    `%loop_op%` <- if (parallel) foreach::`%dopar%` else foreach::`%do%`

    if (verbose) {
      message(sprintf("Starting grid search over %d hyperparameters (Parallel: %s)",
                      length(tau_alpha), parallel))
    }
    fold_mses <- foreach(
      current_tau_alpha = ordered_tau_alpha,
      .combine = 'c'
    ) %loop_op% {
      if (verbose) {
        message(paste("Fold:", i, "tau_alpha:", current_tau_alpha))
      }

      fit_spexvb <- spexvb(
        X = X_train,
        Y = Y_train,
        mu_0 = fold_initials$mu_0,
        omega_0 = fold_initials$omega_0,
        c_pi_0 = fold_initials$c_pi_0,
        d_pi_0 = fold_initials$d_pi_0,
        mu_alpha = mu_alpha,
        tau_alpha = current_tau_alpha,
        tau_b = tau_b,
        standardize = standardize,
        intercept = intercept,
        tau_e = fold_initials$tau_e,
        update_order = fold_initials$update_order,
        max_iter = max_iter,
        tol = tol,
        seed = seed
      )

      if (is.null(fit_spexvb) || is.null(fit_spexvb$mu) || is.null(fit_spexvb$omega)) {
        if (verbose) {
          message(paste("Warning: spexvb fit failed or returned incomplete results for Fold:", i, ", tau_alpha:", current_tau_alpha))
        }
        NA_real_
      } else {
        y_pred <- if (intercept) {
          cbind(1, X_test) %*% fit_spexvb$beta
        } else {
          X_test %*% fit_spexvb$beta
        }
        mean((Y_test - y_pred)^2)
      }
    }
    names(fold_mses) <- as.character(ordered_tau_alpha)
    fold_mses
  } # End of outer foreach

  # Calculate CVE (Cross-Validation Error)
  CVE <- colMeans(epe_test_k, na.rm = TRUE)

  #--------------------- output ----------------------
  cv_results <- list(
    "ordered_tau_alpha" = ordered_tau_alpha,
    "epe_test_k" = epe_test_k,
    "CVE" = CVE,
    "tau_alpha_opt" = ordered_tau_alpha[ which.min(CVE)]
  )

  return(cv_results)
}
