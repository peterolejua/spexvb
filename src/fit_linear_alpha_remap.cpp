// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rcpp.h> // Explicitly include Rcpp.h
#include <cmath>  // For std::sqrt, std::log, std::abs

#include "common_helpers.h" // Include your new general helpers header

// [[Rcpp::export(fit_linear_alpha_remap)]]
Rcpp::List fit_linear_alpha_remap(
    const arma::mat &X,
    const arma::vec &Y,
    arma::vec mu,
    arma::vec omega,
    double c_pi,
    double d_pi,
    double mu_alpha,
    const double &tau_alpha,
    double tau_b,
    const double &tau_e,
    const arma::uvec &update_order,
    const size_t &max_iter,
    const double &tol
) {

  // dimensions
  double n = X.n_rows;
  double p = X.n_cols;

  // initialize entropy loss function
  arma::vec old_entr = entropy(omega);

  // pre-process update parameters
  arma::rowvec YX_vec = Y.t() * X;
  arma::mat XtX = X.t() * X; // Precompute X^t X
  arma::vec half_diag = gram_diag(X);
  arma::vec approx_mean = omega % mu;
  arma::vec W = X * approx_mean;
  arma::mat X_2 = arma::square(X);

  Rcpp::NumericVector c_pi_p_out = Rcpp::NumericVector::create(c_pi);
  Rcpp::NumericVector d_pi_p_out = Rcpp::NumericVector::create(d_pi);

  double a_pi = c_pi;
  double b_pi = d_pi;

  arma::vec mu_alpha_vec = arma::vec(max_iter, arma::fill::zeros);
  arma::vec alpha_vec = arma::vec(max_iter, arma::fill::zeros);
  arma::vec W_squared_vec = arma::vec(max_iter, arma::fill::zeros);

  double test_var_y = arma::dot(Y,Y)/n;


  arma::vec sigma = 1 / arma::sqrt(tau_e * (half_diag + tau_b));

  // initial alpha (expansion parameter)
  double alpha = 1;
  double alpha_diff = 0;
  double p1 = 1/2;
  bool converged = false;
  double convergence_criterion = -999.0;

  size_t t = 0;
  for (t = 0; t < max_iter; ++t) {

    // step 1: q_j (θ_j) are updated sequentially
    // This step results in (µj, σ2j) for all j,
    // and any required expectations of π

    // coordinate update loop (inner loop)
    for (arma::uword k = 0; k < mu.n_elem; ++k) {
      Rcpp::checkUserInterrupt();

      arma::uword j = update_order(k);

      W -= approx_mean(j) * X.col(j);

      sigma(j) = 1 / std::sqrt(tau_e * (half_diag(j) + tau_b));

      mu(j) = tau_e*std::pow(sigma(j), 2) * (YX_vec(j) - arma::dot(X.col(j), W));

      omega(j) = sigmoid(
        std::log(c_pi) - std::log(d_pi) + 0.5 +
          std::log(sigma(j) * std::sqrt(tau_b * tau_e)) +
          0.5 * std::pow(mu(j) / (sigma(j)), 2)
      );


      // update q(π)
      double M = arma::sum(omega);
      a_pi = c_pi + M;
      b_pi = d_pi + p - M;

      approx_mean(j) = omega(j) * mu(j);
      W += approx_mean(j) * X.col(j);
    }


    // step 2: find α(t+1)
    // Calibration step
    double t_YW = arma::dot(W, Y);
    arma::vec var_W = X_2 * (arma::square(mu) % omega % (1 - omega) + sigma % sigma % omega);
    double t_W2 = arma::sum(var_W + arma::square(W));
    W_squared_vec(t) = t_W2;

    alpha_diff = alpha;
    alpha = (t_YW + tau_alpha*mu_alpha) / (t_W2 + tau_alpha);
    alpha_diff = alpha - alpha_diff;
    alpha_vec(t) = alpha;

    // step 3: Remapping step
    // Reverse mapping (need to consider omega>1. push that to mu.)

    mu = alpha*mu;
    sigma = std::abs(alpha)*sigma;
    tau_b = tau_b/std::pow(alpha,2);
    mu_alpha = 1 - (alpha - mu_alpha);
    mu_alpha_vec(t) = mu_alpha;

    // redefine W
    approx_mean = (omega % mu);
    W = X * approx_mean;


    // test if things have gone off the rails
    double test_var = arma::dot(W,W)/n;
    if((test_var/10 > test_var_y) && (t > 99)){
      alpha = 1;
    }

    // check for convergence
    arma::vec new_entr = entropy(omega);

    if ((arma::norm(new_entr - old_entr, "inf") <= tol) &&
        (std::abs(alpha_diff*p1) <= 0.1) &&
        (alpha_diff<=0)
    ) {
      converged = true;
      convergence_criterion = arma::norm(new_entr - old_entr, "inf");
      break;
    } else {
      convergence_criterion = arma::norm(new_entr - old_entr, "inf");
      old_entr = new_entr;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("mu") = mu,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("alpha_vec") = alpha_vec.subvec(0, t-1),
    Rcpp::Named("mu_alpha_vec") = mu_alpha_vec.subvec(0, t-1),
    Rcpp::Named("W_squared_vec") = W_squared_vec.subvec(0, t-1),
    Rcpp::Named("tau_alpha") = tau_alpha,
    Rcpp::Named("tau_b") = tau_b,
    Rcpp::Named("c_pi_p") = a_pi,
    Rcpp::Named("d_pi_p") = b_pi,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = t,
    Rcpp::Named("convergence_criterion") = convergence_criterion
  );

}
