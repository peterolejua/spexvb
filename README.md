# spexvb
This is a package to performParameter Expanded Variational Bayes for Linear Regression
with High-dimensional Variable Selection and Spike-and-slab
Priors

## Installation
To install the package, please follow the code snippet below: 

```
library(devtools)
install_github("peterolejua/spexvb")
```

## Example
Here is an example for conducting analysis using spexvb: 

```
library(spexvb)
# For parallel computing ####
library(doParallel)
no_cores <- detectCores()
# no_cores <- 60
cl <- makeCluster(no_cores)
registerDoParallel(cl)
# Load other libraries
library(tictoc)

# load data from the  CLL study as provided in MOFAdata
# use methylation data, gene expression data and drug responses as predictors
library(MOFAdata)
data(CLL_data)
CLL_data <- CLL_data[1:3]
CLL_data <- lapply(CLL_data,t)
CLL_data <- Reduce(cbind, CLL_data)

#only include patient samples profiles in all three omics
CLL_data <- CLL_data[apply(CLL_data,1, function(p) !any(is.na(p))),]
dim(CLL_data)

# prepare design matrix and response
X <- CLL_data[,!grepl("D_002", colnames(CLL_data))]
Y <- rowMeans(CLL_data[,grepl("D_002", colnames(CLL_data))])

# rescaling
X_means <- colMeans(X)
X_c <- scale(X, center = X_means, scale = F)
sigma_estimate <- sqrt(colMeans(X_c^2)) 
X_cs <- scale(X, center = F, scale = sigma_estimate)

Y_mean <- mean(Y)
Y_c <- Y - Y_mean

seed <- 17
set.seed(seed)

# This might take 3 minutes depending on your computer
tic("Spexvb CV and optimal fit")
fit_spexvb <- cv.spexvb.fit(
  k = 5,
  X = X_cs, # design matrix
  Y = Y_c, # response vector
  # Calculate the initials for each of the 10 folds
  mu = NULL, # Variational Normal mean estimated beta coefficient from lasso,  posterior expectation of bj|sj = 1
  omega = NULL, # Variational probability, expectation that the coefficient from lasso is not zero, the posterior expectation of sj
  c_pi = NULL, # π ∼ Beta(aπ, bπ), known/estimated
  d_pi = NULL, # π ∼ Beta(aπ, bπ), known/estimated
  tau_e = NULL, # errors iid N(0, tau_e^{-1}), known/estimated
  update_order = NULL,
  mu_alpha = 1, # alpha is N(mu_alpha, (tau_e*tau_alphalpha)^{-1}), known/estimated
  tau_alpha = c(0,10^(3:7)),
  tau_b = 400, # initial. b_j is N(0, (tau_e*tau_b)^{-1}), known/estimated
  standardize = F,
  intercept = F,
  max_iter = 500L,
  tol = 1e-5,
  seed = 17 # seed for cv.glmnet initials
)  # Run the fit function
spxlvb_time1 <- toc()

spexvb_beta <- fit_spexvb$mu * fit_spexvb$omega/sigma_estimate
plot(spexvb_beta)

# Stop the parallel backend ####
stopCluster(cl)
```
