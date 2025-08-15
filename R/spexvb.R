#' @title Parameter Expanded Variational Bayes for Well-Calibrated High-Dimensional Linear Regression with Spike-and-Slab Priors
#' @description Fits a sparse linear regression model using variational inference with an alpha expansion step.
#' The model uses spike-and-slab priors.
#' @param X A numeric matrix. The design matrix (n observations × p predictors).
#' @param Y A numeric vector. The response vector of length n.
#' @param mu_0 Optional numeric vector. Initial variational means for regression coefficients.
#' @param omega_0 Optional numeric vector. Initial spike probabilities.
#' @param c_pi_0 Optional numeric. Prior Beta(a, b) parameter a for the spike probability.
#' @param d_pi_0 Optional numeric. Prior Beta(a, b) parameter b for the spike probability.
#' @param tau_e Optional numeric. Known or estimated error precision.
#' @param update_order Optional integer vector. The coordinate update order (0-indexed for C++).
#' @param mu_alpha Prior mean for alpha. Default is 1.
#' @param tau_alpha Prior precision for alpha. Default is 1000.
#' @param tau_b Slab prior precision. Default is 400.
#' @param standardize Logical. Center Y, and center and scale X. Default is TRUE.
#' @param intercept Logical. Whether to include an intercept. Default is TRUE. After the model is fit on the centered and scaled data, the final coefficients are "unscaled" to put them back on the original scale of your data. The intercept is then calculated separately using the means and the final coefficients.
#' @param max_iter Maximum number of iterations for the variational update. Default is 1000.
#' @param tol Convergence threshold for entropy and alpha change. Default is 1e-5.
#' @param seed Integer seed for cross-validation in glmnet. Default is 12376.
#'
#' @return A list with posterior summaries including estimated coefficients (`mu`),
#' inclusion probabilities (`omega`), intercept (if applicable), alpha path, convergence status, etc.
#' @details This function acts as a wrapper for various C++ implementations of the SPEVXB algorithm.
#'   It handles initial parameter setup and dynamically dispatches to the chosen C++ version.
#' @examples
#' \dontrun{
#' # Example usage (assuming X and Y are defined)
#' # result <- spexvb(X = my_X, Y = my_Y)
#' }
#' @useDynLib spexvb, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef
#' @export
spexvb <- function(
    X, # design matrix
    Y, # response vector
    mu_0 = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = NULL,
    mu_alpha = 1, # alpha is N(mu_alpha, (tau_e*tau_alpha)^{-1}), known/estimated
    tau_alpha = 1000,
    tau_b = 400, # initial. b_j is N(0, (tau_e*tau_b)^{-1}), known/estimated
    standardize = T, # Center Y, and center and scale X
    intercept = T,
    max_iter = 1000,
    tol = 1e-5,
    seed = 12376 # seed for cv.glmnet initials
) {

  #extract problem dimensions
  n = nrow(X)
  p = ncol(X)

  #rescale data if necessary
  if(intercept & !standardize){
    warning("Setting standardize <- T to calculate intercept")
    standardize <- T
    }
  
  if (standardize){
    X_means <- colMeans(X)
    X_c <- scale(X, center = X_means, scale = F)
    sigma_estimate <- sqrt(colMeans(X_c^2))
    X_cs <- scale(X_c, center = F, scale = sigma_estimate)

    Y_mean <- mean(Y)
    Y_c <- Y - Y_mean

  } else {
    X_cs = X
    Y_c = Y
  }


  # get.initials.spexvb is in R/ directory and is automatically available
  initials <- get.initials(
    X = X_cs, # design matrix
    Y = Y_c, # response vector
    mu_0 = mu_0, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = omega_0, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = c_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi_0 = d_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
    tau_e = tau_e, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = update_order,
    seed = seed # seed for cv
  )

  mu_0 = initials$mu_0
  omega_0 = initials$omega_0
  c_pi_0 = initials$c_pi_0
  d_pi_0 = initials$d_pi_0
  tau_e = initials$tau_e
  update_order = initials$update_order

  #match internal function call and generate list of arguments
  arg = list(
    X_cs,
    Y_c,
    mu_0,
    omega_0,
    c_pi_0,
    d_pi_0,
    mu_alpha,
    tau_alpha,
    tau_b,
    tau_e,
    update_order,
    max_iter,
    tol
  )

  fn <- "fit_linear_alpha_remap"

  approximate_posterior = do.call(fn, arg)

  # Unscale solution
  if (standardize) {
    beta <- approximate_posterior$mu * approximate_posterior$omega/sigma_estimate
  } else {
    beta <- approximate_posterior$mu * approximate_posterior$omega
  }


  #add intercept
  if(intercept){
    beta <- c(
      beta0 = Y_mean - sum(beta*X_means),
      beta
    )
  }

  test <- list(
    converged = as.logical(approximate_posterior$converged),
    tau_alpha = tau_alpha,
    tau_b_0 = tau_b,
    tau_b = approximate_posterior$tau_b,
    tau_e = tau_e,
    mu_0 = mu_0,
    mu = if (standardize) {
      as.numeric(approximate_posterior$mu[1:p])/sigma_estimate
      } else {
      as.numeric(approximate_posterior$mu[1:p])
    }, # unscale mu
    omega_0 = omega_0,
    omega = as.numeric(approximate_posterior$omega[1:p]),
    beta = beta,
    c_pi_0 = c_pi_0,
    c_pi_p = as.numeric(approximate_posterior$c_pi_p),
    d_pi_0 = d_pi_0,
    d_pi_p = as.numeric(approximate_posterior$d_pi_p),
    approximate_posterior = lapply(approximate_posterior, as.numeric),
    alpha_vec = as.numeric(approximate_posterior$alpha_vec),
    alpha = as.numeric(approximate_posterior$alpha_vec[approximate_posterior$iterations]),
    iterations = as.numeric(approximate_posterior$iterations),
    convergence_criterion = as.numeric(approximate_posterior$convergence_criterion),
    update_order = update_order
  )
  return(test)
}
