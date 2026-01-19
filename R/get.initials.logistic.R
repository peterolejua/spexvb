#' Generate Initial Values for Variational Inference in Sparse Logistic Regression
#'
#' This helper function estimates initial values for variational parameters such as
#' regression coefficients (`mu`), spike probabilities (`omega`), and hyperparameters
#' like  `c_pi`, and `d_pi` using LASSO regression.
#'
#' @title Get initial values for spexvb
#' @description This function initializes parameters for the spexvb model.
#' @param X A design matrix.
#' @param Y A response vector.
#' @param mu_0 Initial mean.
#' @param omega_0 Initial omega.
#' @param c_pi_0 Initial c_pi.
#' @param d_pi_0 Initial d_pi.
#' @param update_order Initial update order.
#' @param seed Seed for reproducibility.
#' @return A list of initialized parameters.
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict coef
#' @export
get.initials.logistic <- function(
    X, # design matrix
    Y, # response vector
    mu_0 = NULL, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
    omega_0 = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
    c_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    d_pi_0 = NULL, # π ∼ Beta(aπ, bπ), known/estimated
    update_order = NULL,
    seed = 12376 # seed for cv
) {
  n <- nrow(X)
  p <- ncol(X)

  if(any(c(is.null(c_pi_0),is.null(d_pi_0),is.null(omega_0)))) {

    # Load the glmnet package if not already loaded
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("Package 'glmnet' needed for this function to work. Please install it.",
           call. = FALSE)
    }

  set.seed(seed)

  ### dimensions ----
  ##### Getting initial values for spexvb #####
  lasso_cv <- glmnet::cv.glmnet(
    X,
    Y,
    alpha = 1,
    family = "binomial",
    intercept = T
  )
  nz_ind_lambda.min <- predict(
    lasso_cv,
    s = "lambda.min",
    type = "nonzero"
    )$lambda.min
  nz_min <- min(length(nz_ind_lambda.min), length(Y) -2)
  s_hat <- max(c(nz_min, 1))
  c_pi_0 = s_hat * exp(0.5)
  d_pi_0 = p - c_pi_0
  omega_0 = rep(s_hat/p, p)
  omega_0[nz_ind_lambda.min] = 1 - s_hat/p
  # Add a 1 for the intercept
  omega_0 = c(1,omega_0)

  # Get initial mu
  mu_0 = as.numeric(coef(lasso_cv, s = "lambda.min"))
  # Get update order, add a zero for the intercept (updated first)
  # Don't need to subtract one, since X used in program will add an intercept
  update_order = c(0,order(abs(mu_0[-1]), decreasing = T))
  }
  return(
    list(
      mu_0 = mu_0, # Variational Normal mean estimated beta coefficient from lasso, posterior expectation of bj|sj = 1
      omega_0 = omega_0, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectaion of sj
      c_pi_0 = c_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
      d_pi_0 = d_pi_0, # π ∼ Beta(aπ, bπ), known/estimated
      update_order = update_order
    )
  )
}
