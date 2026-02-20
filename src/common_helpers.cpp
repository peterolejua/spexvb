// common_helpers.cpp
#include "common_helpers.h" // Include its own header first
// Instead of Rcpp::Function, use the R standalone math library
#include <Rmath.h>

// Define helper functions here
arma::vec entropy(const arma::vec &x) {
  arma::vec ent(x.n_elem, arma::fill::zeros);
  for (arma::uword j = 0; j < x.n_elem; ++j) {
    // clamp values to avoid -Inf
    if ((x(j) > 1e-10) && (x(j) < 1 - 1e-10)) {
      ent(j) -= x(j) * std::log2(x(j)) + (1 - x(j)) * std::log2(1 - x(j));
    }
  }
  return ent;
}

double sigmoid(const double &x) {
  if (x > 32.0) {
    return 1;
  } else if (x < -32.0) {
    return 0;
  } else {
    return 1 / (1 + std::exp(-x));
  }
}

arma::vec gram_diag(const arma::mat &X) {
  return arma::sum(arma::square(X), 0).t();
}

// // Function to calculate the digamma function using R's digamma
// double r_digamma(double x) {
//   static Rcpp::Function digamma("digamma");
//   return Rcpp::as<double>(digamma(Rcpp::Named("x") = x));
// }

double r_digamma(double x) {
  return Rf_digamma(x); // This uses the internal R C-code directly
}
