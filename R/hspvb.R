#' @title Hierarchical Spike-and-Slab Variational Bayes (HSPVB) for High-Dimensional Linear Regression
#' @description Fits a sparse linear regression model using variational inference with a Gamma prior on the slab precision ($\tau_b$).
#' The model uses spike-and-slab priors where the slab scale is adaptively learned.
#' @param X A numeric matrix. The design matrix (n observations × p predictors).
#' @param Y A numeric vector. The response vector of length n.
#' @param mu_0 Optional numeric vector. Initial variational means for regression coefficients.
#' @param omega_0 Optional numeric vector. Initial spike probabilities.
#' @param c_pi_0 Optional numeric. Prior Beta(a, b) parameter a for the spike probability $\pi$.
#' @param d_pi_0 Optional numeric. Prior Beta(a, b) parameter b for the spike probability $\pi$.
#' @param a_prior_tau_b Optional numeric. Gamma prior shape parameter (a) for the slab precision $\tau_b$. Default is 1.
#' @param b_prior_tau_b Optional numeric. Gamma prior rate parameter (b) for the slab precision $\tau_b$. Default is 0.01.
#' @param tau_e Optional numeric. Known or estimated error precision $\tau_\epsilon$ (tau\_e in C++).
#' @param update_order Optional integer vector. The coordinate update order (0-indexed for C++).
#' @param standardize Logical. Center Y, and center and scale X. Default is TRUE.
#' @param intercept Logical. Whether to include an intercept. Default is TRUE.
#' @param max_iter Maximum number of iterations for the variational update. Default is 1000.
#' @param tol Convergence threshold for entropy change. Default is 1e-5.
#' @param seed Integer seed for initialization, passed to `get.initials`. Default is 12376.
#'
#' @return A list with posterior summaries including estimated coefficients (`mu`),
#' inclusion probabilities (`omega`), final expected slab precision (`tau_b`),
#' intercept (if applicable), convergence status, etc.
#' @details This function acts as a wrapper for the C++ implementation of the variational Bayes algorithm
#' with a hierarchical Gamma prior on the slab precision, \code{fit_linear_gamma_hierarchy}.
#'
#' @examples
#' \dontrun{
#' # Example usage (assuming X and Y are defined)
#' # result <- hspvb(X = my_X, Y = my_Y)
#' }
#' @useDynLib spexvb, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef
#' @export
hspvb <- function(
    X, # design matrix
    Y, # response vector
    mu_0 = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = NULL, # $\pi \sim \text{Beta}(a_\pi, b_\pi)$
    d_pi_0 = NULL, # $\pi \sim \text{Beta}(a_\pi, b_\pi)$
    a_prior_tau_b = 0.1, # $\tau_b \sim \text{Gamma}(a_\tau, b_\tau)$ shape
    b_prior_tau_b = 1, # $\tau_b \sim \text{Gamma}(a_\tau, b_\tau)$ rate
    tau_e = NULL, # errors iid $N(0, \tau_\epsilon^{-1})$, known/estimated
    update_order = NULL,
    standardize = TRUE, # Center Y, and center and scale X
    intercept = TRUE,
    max_iter = 1000,
    tol = 1e-5,
    seed = 12376 # seed for cv.glmnet initials
) {

  # extract problem dimensions
  p = ncol(X)

  # Check and handle data standardization
  if(intercept & !standardize){
    warning("Setting standardize <- TRUE to calculate intercept")
    standardize <- TRUE
  }

  if (standardize){
    X_means <- colMeans(X)
    X_c <- scale(X, center = X_means, scale = FALSE)
    sigma_estimate <- sqrt(colMeans(X_c^2))
    X_cs <- scale(X_c, center = FALSE, scale = sigma_estimate)

    Y_mean <- mean(Y)
    Y_c <- Y - Y_mean

  } else {
    X_cs = X
    Y_c = Y
  }

  # Initial parameter setup using a helper function (assumed to exist)
  initials <- get.initials(
    X = X_cs, # design matrix
    Y = Y_c, # response vector
    mu_0 = mu_0,
    omega_0 = omega_0,
    c_pi_0 = c_pi_0,
    d_pi_0 = d_pi_0,
    tau_e = tau_e,
    update_order = update_order,
    seed = seed
  )

  mu_0 = initials$mu_0
  omega_0 = initials$omega_0
  c_pi_0 = initials$c_pi_0
  d_pi_0 = initials$d_pi_0
  tau_e = initials$tau_e
  update_order = initials$update_order

  # Match internal function call and generate list of arguments
  arg = list(
    X_cs,
    Y_c,
    mu_0,
    omega_0,
    c_pi_0,
    d_pi_0,
    a_prior_tau_b,
    b_prior_tau_b,
    tau_e,
    update_order,
    max_iter,
    tol
  )

  fn <- "fit_linear_gamma_hierarchy"

  # Call the C++ function
  # NOTE: The C++ function is assumed to be available via Rcpp::sourceCpp or package loading
  approximate_posterior = do.call(fn, arg)

  # Unscale solution
  if (standardize) {
    # E[beta_j] = E[s_j] * E[b_j|s_j=1] / scale_j
    beta <- approximate_posterior$mu * approximate_posterior$omega / sigma_estimate
  } else {
    beta <- approximate_posterior$mu * approximate_posterior$omega
  }

  # add intercept
  if(intercept){
    beta <- c(
      beta0 = Y_mean - sum(beta * X_means),
      beta
    )
  }

  # Prepare final return list
  result <- list(
    converged = as.logical(approximate_posterior$converged),
    # Tau B parameters
    a_prior_tau_b = a_prior_tau_b,
    b_prior_tau_b = b_prior_tau_b,
    tau_b_post_a = as.numeric(approximate_posterior$a_posterior_tau_b),
    tau_b_post_b = as.numeric(approximate_posterior$b_posterior_tau_b),
    tau_b = as.numeric(approximate_posterior$tau_b), # Final E[tau_b]
    tau_b_vec = as.numeric(approximate_posterior$tau_b_vec), # History of E[tau_b]
    tau_e = tau_e,
    # Coefficient parameters
    mu = if (standardize) {
      as.numeric(approximate_posterior$mu[1:p]) / sigma_estimate
    } else {
      as.numeric(approximate_posterior$mu[1:p])
    }, # unscale mu (E[b_j|s_j=1])
    omega = as.numeric(approximate_posterior$omega[1:p]), # E[s_j]
    beta = beta, # Final estimated coefficients (E[beta])
    # Pi parameters
    c_pi_0 = c_pi_0,
    c_pi_p = as.numeric(approximate_posterior$c_pi_p),
    d_pi_0 = d_pi_0,
    d_pi_p = as.numeric(approximate_posterior$d_pi_p),
    # Diagnostics
    iterations = as.numeric(approximate_posterior$iterations),
    convergence_criterion = as.numeric(approximate_posterior$convergence_criterion),
    update_order = update_order
  )

  return(result)
}
