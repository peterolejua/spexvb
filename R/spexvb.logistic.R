#' @title Parameter Expanded Variational Bayes for Sparse Logistic Regression
#' @description Fits a sparse logistic regression model using variational inference with an alpha expansion step.
#' @param X A numeric matrix. The design matrix (n observations × p predictors).
#' @param Y A numeric vector. The response vector (0/1).
#' @param mu_0 Optional numeric vector. Initial variational means.
#' @param omega_0 Optional numeric vector. Initial spike probabilities.
#' @param c_pi_0 Optional numeric. Prior Beta(a, b) parameter a.
#' @param d_pi_0 Optional numeric. Prior Beta(a, b) parameter b.
#' @param update_order Optional integer vector. Coordinate update order (0-indexed).
#' @param mu_alpha Prior mean for alpha. Default is 1.
#' @param tau_alpha Prior precision for alpha. Default is 10.
#' @param tau_b Slab prior precision. Default is 1.
#' @param max_iter Maximum iterations. Default is 300.
#' @param tol Convergence tolerance. Default is 1e-4.
#' @return A list with posterior summaries including estimated coefficients (`mu`),
#' inclusion probabilities (`omega`), final expected slab precision (`tau_b`),
#' intercept (if applicable), convergence status, etc.
#' @examples
#' \donttest{
#' n <- 50
#' p <- 100
#' X <- matrix(rnorm(n * p), n, p)
#' # Generate binary response
#' Y <- rbinom(n, 1, plogis(X[,1] * 2))
#'
#' fit <- spexvb.logistic(X, Y)
#'
#' # Check convergence
#' print(fit$converged)
#' }
#' @export
spexvb.logistic <- function(
    X,
    Y,
    mu_0 = NULL,
    omega_0 = NULL,
    c_pi_0 = NULL,
    d_pi_0 = NULL,
    update_order = NULL,
    mu_alpha = 1,
    tau_alpha = 10,
    tau_b = 1,
    max_iter = 300,
    tol = 1e-4
) {
  n <- nrow(X)
  p_orig <- ncol(X)


  # Prepend Intercept column

  X_full <- cbind(1, X)

  # Initialize parameters using helper, if not null returns the same

  initials <- get.initials.logistic(
    X,
    Y,
    mu_0 = mu_0,
    omega_0 = omega_0,
    c_pi_0 = c_pi_0,
    d_pi_0 = d_pi_0,
    update_order = update_order,
    seed = 12376
  )

  # Initial call to C++

  cpp_results <- fit_logistic_alpha_remap(
    X = X_full,
    Y = Y,
    mu = initials$mu_0,
    omega = initials$omega_0,
    c_pi = initials$c_pi_0,
    d_pi =initials$d_pi_0,
    update_order = initials$update_order,
    mu_alpha = mu_alpha,
    tau_alpha = tau_alpha,
    tau_b = tau_b,
    max_iter = max_iter,
    tol = tol
  )

  required_loop <- FALSE
  P_tau_alpha <- tau_alpha
  while(
    (
      !cpp_results$converged ||
      abs(cpp_results$alpha - 1) > 0.1 ||
      is.na(cpp_results$alpha) ||
      is.na(sum(cpp_results$mu)) ||
      is.na(sum(cpp_results$omega)) ||
      is.na(sum(cpp_results$c_pi)) ||
      is.na(sum(cpp_results$d_pi))
    ) && P_tau_alpha < 1e+6
    ){
    message("Running again. P_tau_alpha: ", P_tau_alpha)

    required_loop <- TRUE
    P_tau_alpha <- P_tau_alpha*10
    cpp_results <- fit_logistic_alpha_remap(
      X = X_full,
      Y = Y,
      mu = initials$mu_0,
      omega = initials$omega_0,
      c_pi = initials$c_pi_0,
      d_pi =initials$d_pi_0,
      update_order = initials$update_order,
      mu_alpha = mu_alpha,
      tau_alpha = P_tau_alpha,
      tau_b = tau_b,
      max_iter = max_iter,
      tol = tol
    )

  }


beta0 <- cpp_results$mu[1]
beta_0 <- cpp_results$mu[-1]*cpp_results$omega[-1]
beta <- c(beta0, beta_0)

names(beta) <- c("beta0", paste0("V", 1:p_orig))

r_results <- list(
  beta = beta,
  mu = cpp_results$mu,
  omega = cpp_results$omega,
  sigma = cpp_results$sigma,
  alpha = cpp_results$alpha,
  # tau_alpha_final = curr_tau_alpha,
  tau_alpha = P_tau_alpha,
  tau_b = cpp_results$tau_b,
  converged = cpp_results$converged,
  iterations = as.numeric(cpp_results$iterations),
  required_loop = required_loop,
  convg_crit_vec = as.numeric(cpp_results$convg_crit_vec)
)

return(r_results)
}
