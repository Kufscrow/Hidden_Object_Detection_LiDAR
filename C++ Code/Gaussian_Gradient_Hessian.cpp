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
arma::vec compute_kde(int len, const arma::mat& point_data_high, 
                      const arma::mat& h) {
  arma::vec kde_values(len, fill::zeros);
  arma::mat positions = point_data_high.cols(0, 2); // XYZ coordinates
  
  for (int i = 0; i < len; i++) {
    arma::vec point_i = positions.row(i).t();
    arma::vec h_i = h.row(i).t();
    double max_bandwidth = arma::max(h_i);
    double radius = 3.0 * max_bandwidth;
    
    // Compute differences and distances
    arma::mat diff = positions - repmat(point_i.t(), positions.n_rows, 1);
    arma::vec dists = arma::sqrt(arma::sum(arma::square(diff), 1));
    arma::uvec in_radius = arma::find(dists <= radius);
    
    // Skip if no neighbors (set to 0.0)
    if (in_radius.n_elem < 1) {
      kde_values(i) = 0.0;
      continue;
    }
    
    // Extract local neighborhood
    arma::mat local_points = positions.rows(in_radius);
    
    // Compute kernel weights using existing compute_ww0 function
    arma::mat ww0 = compute_ww0(point_i, local_points, h_i);
    arma::vec log_ww = sum(ww0, 1);
    arma::vec ww = exp(log_ww);
    
    // KDE value is the average of the kernel weights
    kde_values(i) = mean(ww);
    
    if ((i+1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return kde_values;
}

// [[Rcpp::export]]
arma::vec compute_kde_knn(int len, const arma::mat& point_data_high, 
                          const arma::mat& h, int k_neighbors) {
  arma::vec kde_values(len, fill::zeros);
  arma::mat positions = point_data_high.cols(0, 2); // XYZ coordinates
  
  for (int i = 0; i < len; i++) {
    arma::vec point_i = positions.row(i).t();
    arma::vec h_i = h.row(i).t();
    double max_bandwidth = arma::max(h_i);
    double radius = 3.0 * max_bandwidth;
    
    // Compute differences and distances
    arma::mat diff = positions - repmat(point_i.t(), positions.n_rows, 1);
    arma::vec dists = arma::sqrt(arma::sum(arma::square(diff), 1));
    arma::uvec in_radius = arma::find(dists <= radius);
    
    // Skip if no neighbors (set to 0.0)
    if (in_radius.n_elem < 1) {
      kde_values(i) = 0.0;
      continue;
    }
    
    // Get distances for points within radius
    arma::vec radius_dists = dists.elem(in_radius);
    
    // If we have more points than requested neighbors, select k nearest
    if (in_radius.n_elem > k_neighbors) {
      // Get indices of k smallest distances
      arma::uvec sorted_indices = sort_index(radius_dists);
      arma::uvec k_indices = sorted_indices.head(k_neighbors);
      
      // Update in_radius to only include k nearest neighbors
      in_radius = in_radius.elem(k_indices);
    }
    
    // Extract local neighborhood (now with k nearest within radius)
    arma::mat local_points = positions.rows(in_radius);
    
    // Compute kernel weights using existing compute_ww0 function
    arma::mat ww0 = compute_ww0(point_i, local_points, h_i);
    arma::vec log_ww = sum(ww0, 1);
    arma::vec ww = exp(log_ww);
    
    // KDE value is the average of the kernel weights
    kde_values(i) = mean(ww);
    
    if ((i+1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return kde_values;
}

// [[Rcpp::export]]
arma::vec compute_hessi_eig_dif(int len, const arma::mat& point_data_high, 
                                const arma::mat& h, const arma::vec& height_high) {
  arma::vec Hessi_eig_dif(len, fill::zeros);
  arma::mat positions = point_data_high.cols(0, 2); // XYZ coordinates
  
  for (int i = 0; i < len; i++) {
    arma::vec point_i = positions.row(i).t();
    arma::vec h_i = h.row(i).t();
    double max_bandwidth = arma::max(h_i);
    double radius = 3.0 * max_bandwidth;
    
    // Compute differences and distances
    arma::mat diff = positions - repmat(point_i.t(), positions.n_rows, 1);
    arma::vec dists = arma::sqrt(arma::sum(arma::square(diff), 1));
    arma::uvec in_radius = arma::find(dists <= radius);
    
    // Skip if too few neighbors (set to 0.0)
    if (in_radius.n_elem < 5) {
      Hessi_eig_dif(i) = 0.0;
      continue;
    }
    
    // Extract local neighborhood
    arma::mat local_points = positions.rows(in_radius);
    arma::vec local_heights = height_high.elem(in_radius);
    
    // Compute Hessian
    arma::mat Hess = d2Kernel_Gaussian(point_i, local_heights, local_points, h_i);
    Hess = symmatu(Hess);
    
    // Compute eigenvalues
    arma::vec eig_val;
    if (!eig_sym(eig_val, Hess)) {
      Hessi_eig_dif(i) = 0.0;
      continue;
    }
    
    double min_eig = min(abs(eig_val));
    double sum_sq_eig = sum(square(eig_val));
    Hessi_eig_dif(i) = (sum_sq_eig < 1e-20) ? 0.0 : 
      -log((min_eig * min_eig) / sum_sq_eig);
    
    if ((i+1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return Hessi_eig_dif;
}

// [[Rcpp::export]]
arma::mat compute_Grad(int len, const arma::mat& point_data_high, 
                       const arma::mat& h, const arma::vec& height_high) {
  arma::mat Gradi(len, 3, fill::zeros);
  arma::mat positions = point_data_high.cols(0, 2); // XYZ coordinates
  
  for (int i = 0; i < len; i++) {
    arma::vec point_i = positions.row(i).t();
    arma::vec h_i = h.row(i).t();
    double max_bandwidth = arma::max(h_i);
    double radius = 3.0 * max_bandwidth;
    
    // Compute differences and distances
    arma::mat diff = positions - repmat(point_i.t(), positions.n_rows, 1);
    arma::vec dists = arma::sqrt(arma::sum(arma::square(diff), 1));
    arma::uvec in_radius = arma::find(dists <= radius);
    
    // Skip if too few neighbors (set to 0.0)
    if (in_radius.n_elem < 5) {
      Gradi.row(i).zeros();
      continue;
    }
    
    // Extract local neighborhood
    arma::mat local_points = positions.rows(in_radius);
    arma::vec local_heights = height_high.elem(in_radius);
    
    // Compute gradient
    arma::vec grad = dKernel_Gaussian(point_i, local_heights, local_points, h_i);
    Gradi.row(i) = (grad.n_elem == 3) ? grad.t() : arma::rowvec(3, fill::zeros);
    
    if ((i+1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return Gradi;
}

// [[Rcpp::export]]
arma::vec compute_hessi_det_ratio(int len, const arma::mat& point_data_high, 
                                  const arma::mat& h, const arma::vec& height_high) {
  arma::vec results(len);
  const double delta = 1e-20; // Regularization constant
  const double min_log = -log(delta); // Minimum value for normalization
  
  for (int i = 0; i < len; i++) {
    // Extract current point
    arma::vec point = point_data_high(i, span(0, 2)).t();
    
    // Compute Hessian
    arma::mat Hess = d2Kernel_Gaussian(point, height_high, 
                                            point_data_high(span::all, span(0, 2)), 
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

// [[Rcpp::export]]
arma::vec compute_hessi_eig_dif_4_sectored(int len, 
                                         const arma::mat& point_data_high, 
                                         const arma::mat& h, 
                                         const arma::vec& height_high) {
  arma::vec Hessi_eig_dif(len, fill::zeros);
  arma::mat positions = point_data_high.cols(0, 2); // XYZ coordinates
  
  for (int i = 0; i < len; i++) {
    arma::vec point_i = positions.row(i).t();
    arma::vec h_i = h.row(i).t();
    double max_bandwidth = arma::max(h_i);
    double radius = 3.0 * max_bandwidth;
    
    // Compute distances and vectors from point_i to all others
    arma::mat diff = positions - repmat(point_i.t(), positions.n_rows, 1);
    arma::vec dists = arma::sqrt(arma::sum(arma::square(diff), 1));
    arma::uvec in_radius = arma::find(dists <= radius);
    
    // Skip if too few neighbors
    if (in_radius.n_elem < 5) {
      Hessi_eig_dif(i) = 0.0;
      continue;
    }
    
    // Compute angles in XY plane [0, 2π]
    arma::mat rel_xy = diff.rows(in_radius);
    rel_xy = rel_xy.cols(0, 1);
    arma::vec angles = arma::atan2(rel_xy.col(1), rel_xy.col(0));
    angles.elem(arma::find(angles < 0)) += 2 * M_PI;
    
    // Process each sector
    std::vector<double> sector_metrics;
    double sector_width = M_PI / 2.0; // 90 degrees
    
    for (int s = 0; s < 4; s++) {
      double start_angle = s * sector_width;
      double end_angle = (s + 1) * sector_width;
      
      // Find points in this sector
      arma::uvec in_sector = arma::find(angles >= start_angle && 
        angles < end_angle);
      
      if (in_sector.n_elem < 5) continue; // Skip small sectors
      
      arma::uvec global_indices = in_radius.elem(in_sector);
      arma::mat sector_points = positions.rows(global_indices);
      arma::vec sector_heights = height_high.elem(global_indices);
      
      // Compute Hessian for this sector
      arma::mat Hess;
      try {
        Hess = d2Kernel_Gaussian(point_i, sector_heights, sector_points, h_i);
        Hess = arma::symmatu(Hess); // Ensure symmetry
        
        arma::vec eig_val;
        if (!arma::eig_sym(eig_val, Hess)) continue; // Skip on failure
        
        double min_eig = arma::min(arma::abs(eig_val));
        double sum_sq_eig = arma::sum(arma::square(eig_val));
        if (sum_sq_eig < 1e-20) continue;
        
        double metric = -std::log((min_eig * min_eig) / sum_sq_eig);
        sector_metrics.push_back(metric);
      } 
      catch (...) { continue; } // Skip on error
    }
    
    // Determine final metric
    if (!sector_metrics.empty()) {
      Hessi_eig_dif(i) = *std::max_element(sector_metrics.begin(), 
                    sector_metrics.end());
    } 
    else {
      // Fallback: Use full neighborhood if no sectors
      arma::mat full_points = positions.rows(in_radius);
      arma::vec full_heights = height_high.elem(in_radius);
      arma::mat Hess = d2Kernel_Gaussian(point_i, full_heights, full_points, h_i);
      Hess = arma::symmatu(Hess);
      
      arma::vec eig_val;
      if (arma::eig_sym(eig_val, Hess)) {
        double min_eig = arma::min(arma::abs(eig_val));
        double sum_sq_eig = arma::sum(arma::square(eig_val));
        Hessi_eig_dif(i) = -std::log((min_eig * min_eig) / sum_sq_eig);
      } 
      else {
        Hessi_eig_dif(i) = 0.0;
      }
    }
    
    // Progress reporting
    if ((i + 1) % 100 == 0) {
      Rcpp::Rcout << "Point " << i + 1 << " completed" << std::endl;
    }
  }
  return Hessi_eig_dif;
}