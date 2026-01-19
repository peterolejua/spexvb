// [[Rcpp::depends(RcppArmadillo)]]

#include <fstream>
#include <iostream>
#include <RcppArmadillo.h>
#include <Rcpp.h> // Explicitly include Rcpp.h
#include <cmath>  // For std::sqrt, std::log, std::abs

#include "common_helpers.h" // Include your new general helpers header

// [[Rcpp::export(fit_logistic_alpha_remap)]]
Rcpp::List fit_logistic_alpha_remap(
    const arma::mat &X,
    const arma::vec &Y,
    arma::vec mu,
    arma::vec omega,
    double c_pi,
    double d_pi,
    const arma::uvec &update_order,
    double mu_alpha,
    const double &tau_alpha,
    double tau_b,
    const size_t &max_iter,
    const double &tol
) {
  // initialize entropy loss function
  arma::vec old_entr = entropy(omega);


  // pre-process update parameters
  arma::vec approx_mean = omega % mu;
  arma::vec sigma = 0*omega  + 1;
  arma::vec omega_old = omega;
  arma::vec W = X * approx_mean;
  arma::mat X_2 = arma::square(X);
  arma::rowvec YX_vec = (Y - 0.5).t() * X;
  arma::vec eta(Y.n_elem, arma::fill::ones);
  arma::vec eta_hyp = 0.25 * arma::tanh(0.5 * eta) / eta;

  double const_lodds = std::log(c_pi) - std::log(d_pi) + 0.5*std::log(tau_b);

  // initial alpha (expansion parameter)
  double alpha = 1;
  double alpha_diff = 0;
  arma::vec convg_crit_vec = arma::vec(max_iter, arma::fill::zeros);
  arma::vec mu_alpha_vec = arma::vec(max_iter, arma::fill::zeros);
  arma::vec alpha_vec = arma::vec(max_iter, arma::fill::zeros);
  arma::vec W_squared_vec = arma::vec(max_iter, arma::fill::zeros);

  // iteration loop
  bool converged = false;
  size_t t = 0;
  for (t = 0; t < max_iter; ++t) {

    // pre-processing per iteration
    arma::rowvec coef_sq = eta_hyp.t() * X_2;

    // implements equation (25) of Carbonetto et al.
    sigma = 1 / arma::sqrt(2 * coef_sq.t() + tau_b);
    omega_old = omega;

    // coordinate update loop
    for (arma::uword k = 0; k < mu.n_elem; ++k) {
      // step 1: q_j (θ_j) are updated sequentially
      // This step results in (µj, σ2j, omegaj) for all j,

      // check if interrupt signal was sent from R
      Rcpp::checkUserInterrupt();

      // the current update dimension
      arma::uword j = update_order(k);

      // delete the j-th column from running sum
      W -= approx_mean(j) * X.col(j);

      // implements equation (26) of Carbonetto et al.
      mu(j) = std::pow(sigma(j), 2) *
        (YX_vec(j) - 2 * arma::dot(eta_hyp % X.col(j), W));

      // implements equation (27) of Carbonetto et al.
      if (j > 0) {
        omega(j) = sigmoid(const_lodds + std::log(sigma(j)) +
          0.5 * std::pow(mu(j) / sigma(j), 2));
      }

      // add j-th column with updated values
      approx_mean(j) = omega(j) * mu(j);
      W += approx_mean(j) * X.col(j);
    }

    // implements equation (32) of the paper
    eta = arma::sqrt(
      arma::square(X) * (omega % (arma::square(mu) + arma::square(sigma))) +
        arma::square(W) - arma::square(X) * arma::square(approx_mean));
    eta_hyp = 0.25 * arma::tanh(0.5 * eta) / eta;

    // step 2: find α(t+1)
    // Calibration step
    double t_YW = arma::dot(W, Y);
    arma::vec var_W = X_2 * (arma::square(mu) % omega % (1 - omega) + sigma % sigma % omega);
    arma::vec t_W2 = var_W + arma::square(W);
    W_squared_vec(t) = arma::sum(t_W2);

    alpha = (t_YW + tau_alpha*mu_alpha) / (2 * arma::dot(eta_hyp, t_W2) + tau_alpha);
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

    // check for convergence
    arma::vec new_entr = entropy(omega);
    double convg_crit = arma::norm(new_entr - old_entr, "inf");
    convg_crit_vec(t) = convg_crit;
    if (convg_crit<= tol) {
      converged = true;
      break;
    } else {
      // if (i > 5 && new_bound > old_bound) {
      //   break;
      // }
      //   old_bound = new_bound;
      old_entr = new_entr;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("mu") = mu,
    Rcpp::Named("sigma") = sigma,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("omega_old") = omega_old,
    Rcpp::Named("alpha") = alpha,
    Rcpp::Named("tau_b") = tau_b,
    Rcpp::Named("alpha_vec") = alpha_vec,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = t,
    Rcpp::Named("convg_crit_vec") = convg_crit_vec
  );
}
