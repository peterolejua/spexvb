# spexvb: Parameter Expanded Variational Bayes for Sparse High-Dimensional Regression

[![CRAN status](https://www.r-pkg.org/badges/version/spexvb)](https://CRAN.R-project.org/package=spexvb) The `spexvb` package implements a **Sparse Parameter-Expanded Variational Bayes** algorithm for well-calibrated linear and logistic regression in high-dimensional settings. By utilizing parameter expansion and spike-and-slab priors, the methodology improves robustness to prior specifications and enhances predictive calibration compared to standard Variational Bayes.

## Installation

You can install the released version of `spexvb` from [CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("spexvb")
```

Or the development version from GitHub:

``` r
# install.packages("devtools")
devtools::install_github("peterolejua/spexvb")
```

## Quick Start Example

This example demonstrates how to perform cross-validation and fit the optimal model using simulated high-dimensional data.

``` r
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
```

## Authors

-   **Peter Olejua** - *University of South Carolina*

-   **Enakshi Saha** - *University of South Carolina*

-   **Rahul Ghosal** - *University of South Carolina*

-   **Ray Bai** - *George Mason University*

-   **Alexander McLain** - *University of South Carolina*

## Citation

If you use this package in your research, please cite:\
\
Olejua, P., Saha, E., Ghosal, R., Bai, R., & McLain, A. (2026). Parameter Expanded Variational Bayes for Well-Calibrated High-Dimensional Linear Regression with Spike-and-Slab Priors. *Statistics and Computing*, Under Revision.
[Preprint DOI: 10.21203/rs.3.rs-7208847/v1](https://doi.org/10.21203/rs.3.rs-7208847/v1)
