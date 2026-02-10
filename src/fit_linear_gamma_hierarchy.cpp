// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rcpp.h> // Explicitly include Rcpp.h
#include <cmath>  // For std::sqrt, std::log, std::abs

#include "common_helpers.h" // Assuming this exists as in your previous files

// [[Rcpp::export(fit_linear_gamma_hierarchy)]]
Rcpp::List fit_linear_gamma_hierarchy(
    const arma::mat &X,
    const arma::vec &Y,
    arma::vec mu,
    arma::vec omega,
    double c_pi,             // Beta prior for pi (a)
    double d_pi,             // Beta prior for pi (b)
    double a_prior_tau_b,            // Gamma prior shape for tau_b
    double b_prior_tau_b,            // Gamma prior rate for tau_b
    const double &tau_e,     // Error precision (fixed)
    const arma::uvec &update_order,
    const size_t &max_iter,
    const double &tol
) {

  // dimensions
  double p = X.n_cols;

  // initialize entropy loss function for convergence check
  arma::vec old_entr = entropy(omega);

  // pre-process update parameters
  arma::rowvec YX_vec = Y.t() * X;
  arma::vec half_diag = gram_diag(X);
  arma::vec approx_mean = omega % mu;
  arma::vec W = X * approx_mean;

  // Initialize tau_b
  double tau_b = a_prior_tau_b/b_prior_tau_b;
  arma::vec tau_b_vec = arma::vec(max_iter, arma::fill::zeros);

  double a_pi = c_pi;
  double b_pi = d_pi;

  // Variables for Gamma posterior
  double a_posterior_tau_b = a_prior_tau_b + p / 2.0;
  double b_posterior_tau_b = b_prior_tau_b; // Initial placeholder

  arma::vec sigma = 1.0 / arma::sqrt(tau_e * (half_diag + tau_b));

  bool converged = false;
  double convergence_criterion = -999.0;
  size_t t = 0;

  // Main Variational Bayes Loop
  for (t = 0; t < max_iter; ++t) {

    // --- Step 1: Coordinate Ascent for Coefficients (q_j) ---
    // Updates mu, sigma, and omega for each predictor

    for (arma::uword k = 0; k < p; ++k) {
      Rcpp::checkUserInterrupt();
      arma::uword j = update_order(k);

      // Remove contribution of j from residual W
      W -= approx_mean(j) * X.col(j);

      // Update sigma_j
      // Note: standard VB update uses the *expectation* of tau_b.
      // Here tau_b variable holds E[tau_b] = a_post / b_post.
      sigma(j) = 1.0 / std::sqrt(tau_e * (half_diag(j) + tau_b));

      // Update mu_j
      mu(j) = tau_e * std::pow(sigma(j), 2) * (YX_vec(j) - arma::dot(X.col(j), W));

      // Update omega_j
      // The term log(sigma * sqrt(tau_b * tau_e)) comes from the ELBO.
      // It represents the ratio of posterior width to prior width.
      omega(j) = sigmoid(
        std::log(c_pi) - std::log(d_pi) + 0.5 +
          std::log(sigma(j) * std::sqrt(tau_b * tau_e)) +
          0.5 * std::pow(mu(j) / sigma(j), 2)
      );

      // Update q(pi) parameters (optional if we just use expectations)
      double M = arma::sum(omega);
      a_pi = c_pi + M;
      b_pi = d_pi + p - M;

      // Add contribution of j back to residual W
      approx_mean(j) = omega(j) * mu(j);
      W += approx_mean(j) * X.col(j);
    }

    // --- Step 2: Update Hierarchical Prior (q(tau_b)) ---
    // We compute the expectation of b^2 under the current q(b, s).
    // E[b_j^2] = omega_j * (mu_j^2 + sigma_j^2) + (1 - omega_j) * (1/tau_b)
    // Note: (1 - omega_j) * (1/tau_b) accounts for the variance of b_j when s_j=0.
    // Even though the spike is at 0 for beta, the latent b is usually modeled as coming
    // from the prior N(0, 1/tau_b).

    arma::vec E_b_sq_vec = (arma::square(mu) + arma::square(sigma)) % omega  + (b_posterior_tau_b / (tau_e * (a_posterior_tau_b - 1))) * (1.0 - omega) ;

    double sum_E_b_sq = arma::sum(E_b_sq_vec);

    // Update Gamma posterior parameters
    // Prior: Gamma(a_prior_tau_b, b_prior_tau_b)
    // Posterior: Gamma(a_posterior_tau_b, b_posterior_tau_b)
    // b_posterior_tau_b = b_prior_tau_b + 0.5 * sum_E_b_sq;
    // b_post = b_prior_tau_b + 0.5 * sum(E[b^2])
    // b_posterior_tau_b = b_prior_tau_b + 0.5 * tau_b * sum_E_b_sq;
    b_posterior_tau_b = b_prior_tau_b + 0.5 * tau_e * sum_E_b_sq;

    // Update expected precision
    tau_b = a_posterior_tau_b / b_posterior_tau_b;
    tau_b_vec(t) = tau_b;

    // --- Step 3: Convergence Check ---
    arma::vec new_entr = entropy(omega);
    double entr_diff = arma::norm(new_entr - old_entr, "inf");

    if (entr_diff <= tol) {
      converged = true;
      convergence_criterion = entr_diff;
      break;
    } else {
      convergence_criterion = entr_diff;
      old_entr = new_entr;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("mu") = mu,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("sigma") = sigma, // Returning sigma might be useful for diagnostics
    Rcpp::Named("tau_b") = tau_b,
    Rcpp::Named("tau_b_vec") = tau_b_vec.subvec(0, t-1),
    Rcpp::Named("a_posterior_tau_b") = a_posterior_tau_b,
    Rcpp::Named("b_posterior_tau_b") = b_posterior_tau_b,
    Rcpp::Named("c_pi_p") = a_pi,
    Rcpp::Named("d_pi_p") = b_pi,
    Rcpp::Named("converged") = converged,
    Rcpp::Named("iterations") = t,
    Rcpp::Named("convergence_criterion") = convergence_criterion
  );
}
