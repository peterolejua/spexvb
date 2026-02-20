## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(spexvb)

## ----installation, eval = FALSE-----------------------------------------------
# install.packages("spexvb")

## ----example------------------------------------------------------------------
library(spexvb)
library(doParallel)
cl <- makeCluster(min(2, parallel::detectCores())) 
registerDoParallel( cl)


# 1. Simulate high-dimensional data (n=100, p=500)
set.seed(17)
n <- 100
p <- 500
X <- matrix(rnorm(n * p), n, p)
true_beta <- c(rep(3, 5), rep(0, p - 5)) # 5 active predictors
Y <- X %*% true_beta + rnorm(n)

# 2. Perform 5-fold CV to find optimal tau_alpha and fit final model
fit <- cv.spexvb.fit(
  k = 5,
  X = X, 
  Y = Y,
  tau_alpha = c(0, 10^(3:6)), # Precision for expansion parameter alpha
  standardize = TRUE,
  intercept = TRUE
)

# 4. Visualize results
plot(true_beta, main = "True Coefficients", ylab = "Value")
plot(fit$beta, main = "Estimated Coefficients", ylab = "Value")
abline(h = 0, col = "red", lty = 2)

stopCluster(cl)

