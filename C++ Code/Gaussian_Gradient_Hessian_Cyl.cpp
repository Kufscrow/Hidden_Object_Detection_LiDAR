#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

arma::vec compute_SS1(const arma::vec& zz, const arma::mat& ss, 
                      const arma::vec& hh, const arma::vec& ww, 
                      const arma::vec& yy) {
  int dd = zz.n_elem;
  arma::vec SS(dd, fill::zeros);
  double ww_sum = sum(ww);
  double ww_yy_sum = dot(ww, yy);
  
  for (int i = 0; i < dd; i++) {
    arma::vec diff = zz(i) - ss.col(i);
    arma::vec dww = -diff / (hh(i) * hh(i)) % ww;
    
    double dww_sum = sum(dww);
    double dww_yy = dot(dww, yy);
    
    SS(i) = (dww_yy * ww_sum - dww_sum * ww_yy_sum) / (ww_sum * ww_sum);
  }
  return SS;
}

arma::mat compute_SS(const arma::vec& zz, const arma::mat& ss, 
                     const arma::vec& h, const arma::vec& ww, 
                     const arma::vec& yy) {
  int n = ss.n_rows;
  int dd = zz.n_elem;
  double ww_sum = sum(ww);
  double ww_yy_sum = dot(ww, yy);
  
  // Precompute D matrix: (ss - zz.t()) / h²
  arma::mat D = ss;
  D.each_row() -= zz.t();
  for (int i = 0; i < dd; i++) {
    D.col(i) /= h(i) * h(i);
  }
  
  // Precompute dww matrix and related terms
  arma::mat dww_mat = -D;
  dww_mat.each_col() %= ww;
  arma::rowvec dww_sums = sum(dww_mat, 0);
  arma::colvec dww_dot_yy = dww_mat.t() * yy;
  
  // Precompute 1/h² for diagonal
  arma::vec inv_h2 = 1 / (h % h);
  
  arma::mat SS(dd, dd, fill::zeros);
  
  for (int i = 0; i < dd; i++) {
    for (int j = 0; j < dd; j++) {
      // Compute second derivative
      arma::vec d2ww_ij = ww % (D.col(i) % D.col(j));
      if (i == j) {
        d2ww_ij -= ww * inv_h2(i);
      }
      
      double term1 = dot(d2ww_ij, yy) / ww_sum;
      double term2 = -sum(d2ww_ij) * ww_yy_sum / (ww_sum * ww_sum);
      double term3 = -(dww_sums(i) * dww_dot_yy(j) + 
                       dww_sums(j) * dww_dot_yy(i)) / (ww_sum * ww_sum);
      double term4 = 2 * dww_sums(i) * dww_sums(j) * ww_yy_sum / 
        (ww_sum * ww_sum * ww_sum);
      
      SS(i, j) = term1 + term2 + term3 + term4;
    }
  }
  return SS;
}

arma::mat compute_ww0(const arma::vec& zz, const arma::mat& ss, 
                      const arma::vec& h) {
  int n = ss.n_rows;
  int dd = ss.n_cols;
  arma::mat ww0(n, dd);
  
  for (int i = 0; i < dd; i++) {
    double log_const = -0.5 * std::log(2 * M_PI) - std::log(h(i));
    arma::vec diff = ss.col(i) - zz(i);
    ww0.col(i) = log_const - 0.5 * (diff % diff) / (h(i) * h(i));
  }
  return ww0;
}

arma::vec dKernel_Gaussian(const arma::vec& zz, const arma::vec& yy, 
                           const arma::mat& ss, const arma::vec& h) {
  arma::mat ww0 = compute_ww0(zz, ss, h);
  arma::vec log_ww = sum(ww0, 1);
  arma::vec ww = exp(log_ww);
  return compute_SS1(zz, ss, h, ww, yy);
}

arma::mat d2Kernel_Gaussian(const arma::vec& zz, const arma::vec& yy, 
                            const arma::mat& ss, const arma::vec& h) {
  arma::mat ww0 = compute_ww0(zz, ss, h);
  arma::vec log_ww = sum(ww0, 1);
  arma::vec ww = exp(log_ww);
  return compute_SS(zz, ss, h, ww, yy);
}

// [[Rcpp::export]]
arma::vec compute_hessi_eig_dif_cyl(int len, const arma::mat& point_data_high, 
                                const arma::mat& h, const arma::vec& height_high) {
  arma::vec Hessi_eig_dif(len);
  
  for (int i = 0; i < len; i++) {
    arma::vec point = point_data_high.row(i).cols(0, 1).t();
    arma::mat Hess = d2Kernel_Gaussian(point, height_high, 
                                       point_data_high.cols(0, 1), 
                                       h.row(i).t());
    
    // Symmetrize Hessian and compute eigenvalues
    Hess = symmatu(Hess);
    arma::vec eig_val;
    bool success = eig_sym(eig_val, Hess);
    if (!success) Rcpp::stop("Eigen decomposition failed");
    
    double min_eig = min(abs(eig_val));
    double sum_sq_eig = sum(square(eig_val));
    Hessi_eig_dif[i] = -log((min_eig * min_eig) / sum_sq_eig);
    
    if ((i+1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return Hessi_eig_dif;
}

// [[Rcpp::export]]
arma::mat compute_Grad_Cyl(int len, const arma::mat& point_data_high, 
                       const arma::mat& h, const arma::vec& height_high) {
  arma::mat Gradi(len, 2);
  
  for (int i = 0; i < len; i++) {
    arma::vec point = point_data_high.row(i).cols(0, 1).t();
    Gradi.row(i) = dKernel_Gaussian(point, height_high, 
              point_data_high.cols(0, 1), 
              h.row(i).t()).t();
    
    if ((i+1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return Gradi;
}

// [[Rcpp::export]]
arma::vec compute_hessi_det_ratio_cyl(int len, const arma::mat& point_data_high, 
                                  const arma::mat& h, const arma::vec& height_high) {
  arma::vec results(len);
  const double delta = 1e-20; // Regularization constant
  const double min_log = -log(delta); // Minimum value for normalization
  
  for (int i = 0; i < len; i++) {
    // Extract current point
    arma::vec point = point_data_high(i, span(0, 1)).t();
    
    // Compute Hessian
    arma::mat Hess = d2Kernel_Gaussian(point, height_high, 
                                       point_data_high(span::all, span(0, 1)), 
                                       h.row(i).t());
    Hess = symmatu(Hess); // Symmetrize
    
    // Compute determinant and diagonal product
    double det_val = det(Hess);
    arma::vec diag_vals = Hess.diag();
    double diag_prod = prod(abs(diag_vals));
    
    // Compute absolute values with regularization
    double abs_det = std::abs(det_val) + delta;
    double abs_diag_prod = diag_prod + delta;
    
    // Compute singularity metric (always ≥ 0)
    double ratio = abs_diag_prod / abs_det; // Inverted ratio
    results(i) = std::log(1 + ratio); // Always ≥ 0, increases with singularity
    
    // Progress reporting
    if ((i+1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return results;
}

