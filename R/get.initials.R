#' Generate Initial Values for Variational Inference in Sparse Regression
#'
#' This helper function estimates initial values for variational parameters such as
#' regression coefficients (`mu`), spike probabilities (`omega`), and hyperparameters
#' like `tau_e`, `c_pi`, and `d_pi` using LASSO and Ridge regression fits.
#'
#' @title Get initial values for spexvb
#' @description This function initializes parameters for the spexvb model.
#' @param X A design matrix.
#' @param Y A response vector.
#' @param mu_0 Initial mean.
#' @param omega_0 Initial omega.
#' @param c_pi_0 Initial c_pi.
#' @param d_pi_0 Initial d_pi.
#' @param tau_e Initial tau_e.
#' @param update_order Initial update order.
#' @param seed Seed for reproducibility.
#' @return A list of initialized parameters.
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef
#' @export
get.initials <- function(
    X, # design matrix
    Y, # response vector
    mu_0 = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = NULL, # \eqn{\pi \sim Beta(a_\pi, b_\pi)}, known/estimated
    d_pi_0 = NULL, # \eqn{\pi \sim Beta(a_\pi, b_\pi)}, known/estimated
    tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
    update_order = NULL,
    seed = 12376 # seed for cv
) {

  set.seed(seed)

  ### dimensions ----
  n <- nrow(X)
  p <- ncol(X)

  if(any(c(is.null(tau_e),is.null(c_pi_0),is.null(d_pi_0),is.null(omega_0)))) {

    # Load the glmnet package if not already loaded
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("Package 'glmnet' needed for this function to work. Please install it.",
           call. = FALSE)
    }

    lasso_cv <-  glmnet::cv.glmnet(
      X,
      Y,
      alpha = 1, # lasso
      family = "gaussian",
      standardize = F,
      standardize.response = FALSE,
      intercept = F,
      parallel = T
    )

    nz_ind_lambda.min <- predict(
      lasso_cv, s = "lambda.min", type = "nonzero"
    )$lambda.min

    nz_ind_lambda.1se <-predict(
      lasso_cv, s = "lambda.1se", type = "nonzero"
    )$lambda.1se

    nz_min <- min(
      length(nz_ind_lambda.min),
      length(nz_ind_lambda.1se),
      length(Y) - 2
    )

    yhat <-  predict(lasso_cv, X, s = "lambda.min", type = "response")

    noise_sd <- sqrt(sum((Y - yhat)^2)/(length(Y) - nz_min - 1))

    tau_e <- 1/noise_sd^2

    s_hat <- max(c(nz_min,1))

    if(is.null(c_pi_0)) {
      c_pi_0 = s_hat*exp(0.5)
    }

    if(is.null(d_pi_0)) {
      d_pi_0 = p - c_pi_0
    }

    if(is.null(omega_0)){
      omega_0 = rep(s_hat/p, p)
      # Ensure nz_ind_lambda.min is not empty and contains valid indices
      if(length(nz_ind_lambda.min) > 0) {
        # glmnet returns 1-based indexing for non-zero coefficients
        omega_0[nz_ind_lambda.min] = 1
      }
    }

  }

  if(is.null(mu_0)) {
    # Load the glmnet package if not already loaded
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("Package 'glmnet' needed for this function to work. Please install it.",
           call. = FALSE)
    }

    ridge_cv = glmnet::cv.glmnet(
      X,
      Y,
      family = "gaussian",
      standardize = F,
      standardize.response = FALSE,
      intercept = F,
      alpha = 0, # Ridge regression (alpha=0)
      parallel = T
    )

    mu_0 = as.numeric(coef(ridge_cv, s = "lambda.min"))
    mu_0 = mu_0[2:(p+1)]
  }

  #generate prioritized updating order
  if(is.null(update_order)) {
    # Order by absolute value of mu, decreasingly, for features only
    update_order = order(abs(mu_0[1:p]), decreasing = TRUE)
    update_order = update_order - 1 # Convert to 0-based indexing for C++
  }

  return(
    list(
      mu_0 = mu_0, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
      omega_0 = omega_0, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectaion of sj
      c_pi_0 = c_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
      d_pi_0 = d_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
      tau_e = tau_e, # errors iid N(0, tau_e^{-1}), known/estimated
      update_order = update_order
    )
  )
}
