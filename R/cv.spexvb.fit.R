#' @title Cross-validation and Final Model Fitting for SPEXVB
#' @description This function performs k-fold cross-validation to determine the optimal
#'   `tau_alpha` parameter for the `spexvb` model, and then fits a final `spexvb` model
#'   to the full dataset using this optimal `tau_alpha`. Initial values for the final
#'   model are also derived from the full dataset.
#' @param k Integer, the number of folds to use for cross-validation. Must be greater than 2.
#' @param X A design matrix.
#' @param Y A response vector.
#' @param mu_0 Initial variational mean (posterior expectation of beta_j | s_j = 1).
#'   If NULL, initialized automatically by `get.initials`.
#' @param omega_0 Initial variational probability (posterior expectation of s_j).
#'   If NULL, initialized automatically by `get.initials`.
#' @param c_pi_0 Prior parameter for pi (beta distribution shape1).
#'   If NULL, initialized automatically by `get.initials`.
#' @param d_pi_0 Prior parameter for pi (beta distribution shape2).
#'   If NULL, initialized automatically by `get.initials`.
#' @param tau_e Initial precision of errors.
#'   If NULL, initialized automatically by `get.initials`.
#' @param update_order A numeric vector specifying the order of updates for coefficients.
#'   If NULL, initialized automatically by `get.initials`.
#' @param mu_alpha Mean for the prior on alpha (expansion parameter).
#' @param tau_alpha A numeric vector of `tau_alpha` values to cross-validate over.
#'   Must have at least two values.
#' @param tau_b Initial precision for beta_j (when s_j = 1).
#' @param standardize Logical. Center Y, and center and scale X. Default is TRUE.
#' @param intercept Logical. Whether to include an intercept. Default is TRUE. After the model is fit on the centered and scaled data, the final coefficients are "unscaled" to put them back on the original scale of your data. The intercept is then calculated separately using the means and the final coefficients.
#' @param max_iter Maximum number of outer loop iterations for both CV fits and the final fit.
#' @param tol Convergence tolerance for both CV fits and the final fit.
#' @param seed Seed for reproducibility of data splitting and `glmnet` initials.
#' @param verbose Logical, if TRUE, prints progress messages during cross-validation.
#' @param parallel Logical, if TRUE, search in parallel.
#' @return The final fitted `spexvb` model, which is a list containing the approximate
#'   posterior parameters and convergence information for the full dataset using the
#'   optimal `tau_alpha` determined by cross-validation.
#' @details This function orchestrates the cross-validation process and the final model fit.
#'   It first gets initial values for the full dataset, then uses `cv.spexvb` to find
#'   the `tau_alpha` that minimizes cross-validation error, and finally calls `spexvb`
#'   on the complete dataset with the chosen `tau_alpha`.
#' @examples
#' \donttest{
#' # Generate simple synthetic data
#' n <- 50
#' p <- 100
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[,1] * 2 + rnorm(n)
#'
#' # Run cross-validation and fit final model
#' # (Setting k=3 to keep it quick for the example)
#' fit <- cv.spexvb.fit(k = 3, X = X, Y = Y)
#' }
#' @seealso \code{\link{cv.spexvb}}, \code{\link{spexvb}}
#' @export
cv.spexvb.fit <- function(
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

  # get initials for the *full* dataset
  initials <- get.initials(
    X = X, # design matrix
    Y = Y, # response vector
    mu_0 = mu_0, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = omega_0, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = c_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi_0 = d_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
    tau_e = tau_e, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = update_order,
    seed = seed # seed for cv
  )

  # Perform cross-validation to find optimal tau_alpha
  cv_results <- cv.spexvb(
    k = k, #the number of folds to use
    X = X, # design matrix
    Y = Y, # response vector
    mu_0 = initials$mu_0, # Use initials from full data for CV, but note that CV itself will re-initialize per fold
    omega_0 = initials$omega_0, # This argument is passed to cv.spexvb, which then passes it to spexvb.
    # Note: cv.spexvb re-initializes within each fold using get.initials
    # so these `mu` and `omega` from `initials` might not be directly used *within* the CV folds,
    # unless you explicitly changed `get.initials` to pass them through.
    # However, it's good to pass them here as part of the consistent parameter set for cv.spexvb.
    c_pi_0 = initials$c_pi_0,
    d_pi_0 = initials$d_pi_0,
    tau_e = initials$tau_e,
    update_order = initials$update_order,
    mu_alpha = mu_alpha,
    tau_alpha = tau_alpha,
    tau_b = tau_b,
    standardize = standardize,
    intercept = intercept,
    max_iter = max_iter,
    tol = tol,
    seed = seed,
    verbose = verbose,
    parallel = parallel
  )

  # Fit the final model with the optimal tau_alpha on the full dataset
  # Using try() to gracefully handle potential errors during the final fit
  fit_spexvb <- try(
    spexvb(
      X = X, # design matrix
      Y = Y, # response vector
      mu_0 = initials$mu_0, # Use the initial values derived from the full dataset
      omega_0 = initials$omega_0,
      c_pi_0 = initials$c_pi_0,
      d_pi_0 = initials$d_pi_0,
      tau_e = initials$tau_e,
      update_order = initials$update_order,
      mu_alpha = mu_alpha,
      tau_alpha = cv_results$tau_alpha_opt, # Use the optimal tau_alpha from CV
      tau_b = tau_b,
      standardize = standardize,
      intercept = intercept,
      max_iter = max_iter,
      tol = tol,
      seed = seed
    ),
    silent = TRUE # Prevents printing error messages directly from try()
  )

  if (inherits(fit_spexvb, "try-error")) {
    warning("Final spexvb model fit failed with optimal tau_alpha. Returning CV results only.")
    return(cv_results)
  }

  if (verbose) {
    message(paste("Best tau_alpha selected by CV:", cv_results$tau_alpha_opt))
  }

  # Return the final fitted model
  return(fit_spexvb)
}
