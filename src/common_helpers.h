// common_helpers.h
#ifndef COMMON_HELPERS_H
#define COMMON_HELPERS_H

#include <RcppArmadillo.h>
#include <cmath> // For std::log2, std::exp, std::pow, std::sqrt, std::abs

// Forward declarations of helper functions
arma::vec entropy(const arma::vec &x);
double sigmoid(const double &x);
arma::vec gram_diag(const arma::mat &X);
double r_digamma(double x);

#endif // COMMON_HELPERS_H
