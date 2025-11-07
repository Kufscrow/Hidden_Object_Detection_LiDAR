#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List ground_filtering_uniform_quadratic_patches_optimized(
    const arma::mat& temp1, 
    const arma::mat& temp2, 
    const arma::mat& temp3, 
    double hh, 
    int Size = 40, 
    int More = 5, 
    double adjust = 1.5, 
    bool house = false, 
    int threshold = 8) {
  
  int n3 = temp3.n_rows;
  arma::mat measure(n3, 4, fill::zeros);
  arma::vec labels_temp3 = arma::zeros<arma::vec>(n3);
  
  double x_lower = std::floor(arma::min(temp1.col(0)) - 1e-8);
  double x_upper = std::floor(arma::max(temp1.col(0)) + 1e-8) + 1;
  int x_k = std::floor((x_upper - x_lower) / Size);
  arma::vec x_cut = linspace<vec>(x_lower, x_upper, x_k + 1);
  
  double y_lower = std::floor(arma::min(temp1.col(1)) - 1e-8);
  double y_upper = std::floor(arma::max(temp1.col(1)) + 1e-8) + 1;
  int y_k = std::floor((y_upper - y_lower) / Size);
  arma::vec y_cut = linspace<vec>(y_lower, y_upper, y_k + 1);
  
  const double hh_sq = hh * hh; // Precompute squared radius
  
  // Enable OpenMP for parallel processing (if supported)
  // #pragma omp parallel for collapse(2) schedule(dynamic)
  for (uword i = 0; i < x_cut.n_elem - 1; ++i) {
    for (uword j = 0; j < y_cut.n_elem - 1; ++j) {
      
      // Extract relevant points from temp2 and temp3 for current grid
      uvec idx2_all = find(
        temp2.col(0) > (x_cut[i] - More) && 
          temp2.col(0) <= (x_cut[i + 1] + More) &&
          temp2.col(1) > (y_cut[j] - More) && 
          temp2.col(1) <= (y_cut[j + 1] + More)
      );
      
      uvec idx3 = find(
        temp3.col(0) > x_cut[i] && 
          temp3.col(0) <= x_cut[i + 1] &&
          temp3.col(1) > y_cut[j] && 
          temp3.col(1) <= y_cut[j + 1]
      );
      
      if (idx2_all.empty() || idx3.empty()) continue;
      
      arma::mat sub_temp2_all = temp2.rows(idx2_all);
      arma::mat sub_temp3 = temp3.rows(idx3);
      
      // Precompute coordinates for faster access
      const vec x2 = sub_temp2_all.col(0);
      const vec y2 = sub_temp2_all.col(1);
      const int n_sub3 = sub_temp3.n_rows;
      
      arma::mat measure_sub1(n_sub3, 4, fill::zeros);
      
      for (int k = 0; k < n_sub3; ++k) {
        const double x3 = sub_temp3(k, 0);
        const double y3 = sub_temp3(k, 1);
        const double z3 = sub_temp3(k, 2);
        
        // Efficient distance calculation using vectorization
        const vec dx = x2 - x3;
        const vec dy = y2 - y3;
        const vec dist2 = dx % dx + dy % dy;
        
        const uvec kernel_idx = find(dist2 <= hh_sq);
        if (kernel_idx.empty()) continue;
        
        const arma::mat B = sub_temp2_all.rows(kernel_idx);
        const int n_B = B.n_rows;
        
        double local_std = 0.0;
        arma::vec beta(6, fill::zeros);
        
        if (n_B >= 6) {
          const vec x = B.col(0);
          const vec y = B.col(1);
          const vec z = B.col(2);
          
          // Construct design matrix
          mat X(n_B, 6);
          X.col(0).ones();
          X.col(1) = x;
          X.col(2) = y;
          X.col(3) = x % x;
          X.col(4) = x % y;
          X.col(5) = y % y;
          
          // Solve using normal equations for efficiency
          mat XtX = X.t() * X;
          vec Xtz = X.t() * z;
          bool success = solve(beta, XtX, Xtz, solve_opts::fast);
          
          if (success) {
            const vec residuals = z - X * beta;
            const int df = n_B - 6;
            if (df > 0) local_std = std::sqrt(accu(square(residuals)) / df);
          }
        }
        
        // Compute fitted value and deviation
        const double z_fit = beta[0] + beta[1] * x3 + beta[2] * y3 + 
          beta[3] * x3 * x3 + beta[4] * x3 * y3 + beta[5] * y3 * y3;
        const double deviation = z3 - z_fit;
        
        // Update labels and measurements
        if (n_B >= 6 && deviation <= 2 * local_std) {
          labels_temp3(idx3[k]) = 1.0;
        }
        
        measure_sub1(k, 0) = z_fit;
        measure_sub1(k, 1) = local_std;
        measure_sub1(k, 2) = deviation;
        measure_sub1(k, 3) = (deviation > 2 * local_std) ? 1.0 : 0.0;
      }
      
      measure.rows(idx3) = measure_sub1;
    }
  }
  
  return List::create(
    Named("measure") = measure,
    Named("labels_temp3") = labels_temp3
  );
}

// [[Rcpp::export]]
List ground_filtering_uniform_cubic_patches_optimized(
    const arma::mat& temp1, 
    const arma::mat& temp2, 
    const arma::mat& temp3, 
    double hh, 
    int Size = 40, 
    int More = 5, 
    double adjust = 1.5, 
    bool house = false, 
    int threshold = 8) {
  
  int n3 = temp3.n_rows;
  arma::mat measure(n3, 4, fill::zeros);
  arma::vec labels_temp3 = arma::zeros<arma::vec>(n3);
  
  double x_lower = std::floor(arma::min(temp1.col(0)) - 1e-8);
  double x_upper = std::floor(arma::max(temp1.col(0)) + 1e-8) + 1;
  int x_k = std::floor((x_upper - x_lower) / Size);
  arma::vec x_cut = linspace<vec>(x_lower, x_upper, x_k + 1);
  
  double y_lower = std::floor(arma::min(temp1.col(1)) - 1e-8);
  double y_upper = std::floor(arma::max(temp1.col(1)) + 1e-8) + 1;
  int y_k = std::floor((y_upper - y_lower) / Size);
  arma::vec y_cut = linspace<vec>(y_lower, y_upper, y_k + 1);
  
  const double hh_sq = hh * hh; // Precompute squared radius
  
  // Enable OpenMP for parallel processing (if supported)
  // #pragma omp parallel for collapse(2) schedule(dynamic)
  for (uword i = 0; i < x_cut.n_elem - 1; ++i) {
    for (uword j = 0; j < y_cut.n_elem - 1; ++j) {
      
      // Extract relevant points from temp2 and temp3 for current grid
      uvec idx2_all = find(
        temp2.col(0) > (x_cut[i] - More) && 
          temp2.col(0) <= (x_cut[i + 1] + More) &&
          temp2.col(1) > (y_cut[j] - More) && 
          temp2.col(1) <= (y_cut[j + 1] + More)
      );
      
      uvec idx3 = find(
        temp3.col(0) > x_cut[i] && 
          temp3.col(0) <= x_cut[i + 1] &&
          temp3.col(1) > y_cut[j] && 
          temp3.col(1) <= y_cut[j + 1]
      );
      
      if (idx2_all.empty() || idx3.empty()) continue;
      
      arma::mat sub_temp2_all = temp2.rows(idx2_all);
      arma::mat sub_temp3 = temp3.rows(idx3);
      
      // Precompute coordinates for faster access
      const vec x2 = sub_temp2_all.col(0);
      const vec y2 = sub_temp2_all.col(1);
      const int n_sub3 = sub_temp3.n_rows;
      
      arma::mat measure_sub1(n_sub3, 4, fill::zeros);
      
      for (int k = 0; k < n_sub3; ++k) {
        const double x3 = sub_temp3(k, 0);
        const double y3 = sub_temp3(k, 1);
        const double z3 = sub_temp3(k, 2);
        
        // Efficient distance calculation using vectorization
        const vec dx = x2 - x3;
        const vec dy = y2 - y3;
        const vec dist2 = dx % dx + dy % dy;
        
        const uvec kernel_idx = find(dist2 <= hh_sq);
        if (kernel_idx.empty()) continue;
        
        const arma::mat B = sub_temp2_all.rows(kernel_idx);
        const int n_B = B.n_rows;
        
        double local_std = 0.0;
        arma::vec beta(10, fill::zeros);
        
        if (n_B >= 10) {
          const vec x = B.col(0);
          const vec y = B.col(1);
          const vec z = B.col(2);
          
          // Construct design matrix
          mat X(n_B, 10);
          X.col(0).ones();
          X.col(1) = x;
          X.col(2) = y;
          X.col(3) = x % x;
          X.col(4) = x % y;
          X.col(5) = y % y;
          X.col(6) = x % x % x;
          X.col(7) = x % x % y;
          X.col(8) = x % y % y;
          X.col(9) = y % y % y;
          
          // Solve using normal equations for efficiency
          mat XtX = X.t() * X;
          vec Xtz = X.t() * z;
          bool success = solve(beta, XtX, Xtz, solve_opts::fast);
          
          if (success) {
            const vec residuals = z - X * beta;
            const int df = n_B - 10;
            if (df > 0) local_std = std::sqrt(accu(square(residuals)) / df);
          }
        }
        
        // Compute fitted value and deviation
        const double z_fit = beta[0] + beta[1] * x3 + beta[2] * y3 + 
          beta[3] * x3 * x3 + beta[4] * x3 * y3 + beta[5] * y3 * y3 +
          beta[6] * x3 * x3 * x3 + beta[7] * x3 * x3 * y3 + beta[8] * x3 * y3 * y3+
          beta[9] * y3 * y3 * y3;
        const double deviation = z3 - z_fit;
        
        // Update labels and measurements
        if (n_B >= 10 && deviation <= 2 * local_std) {
          labels_temp3(idx3[k]) = 1.0;
        }
        
        measure_sub1(k, 0) = z_fit;
        measure_sub1(k, 1) = local_std;
        measure_sub1(k, 2) = deviation;
        measure_sub1(k, 3) = (deviation > 2 * local_std) ? 1.0 : 0.0;
      }
      
      measure.rows(idx3) = measure_sub1;
    }
  }
  
  return List::create(
    Named("measure") = measure,
    Named("labels_temp3") = labels_temp3
  );
}

// [[Rcpp::export]]
List ground_filtering_uniform_quadratic_filter_patches_optimized(
    const arma::mat& temp1, 
    const arma::mat& temp2, 
    const arma::mat& temp3, 
    double hh, 
    int Size = 40, 
    int More = 5, 
    double adjust = 1.5, 
    bool house = false, 
    int threshold = 8) {
  
  int n3 = temp3.n_rows;
  arma::mat measure(n3, 4, fill::zeros);
  arma::vec labels_temp3 = arma::zeros<arma::vec>(n3);
  
  double x_lower = std::floor(arma::min(temp1.col(0)) - 1e-8);
  double x_upper = std::floor(arma::max(temp1.col(0)) + 1e-8) + 1;
  int x_k = std::floor((x_upper - x_lower) / Size);
  arma::vec x_cut = linspace<vec>(x_lower, x_upper, x_k + 1);
  
  double y_lower = std::floor(arma::min(temp1.col(1)) - 1e-8);
  double y_upper = std::floor(arma::max(temp1.col(1)) + 1e-8) + 1;
  int y_k = std::floor((y_upper - y_lower) / Size);
  arma::vec y_cut = linspace<vec>(y_lower, y_upper, y_k + 1);
  
  const double hh_sq = hh * hh; // Precompute squared radius
  
  // Enable OpenMP for parallel processing (if supported)
  // #pragma omp parallel for collapse(2) schedule(dynamic)
  for (uword i = 0; i < x_cut.n_elem - 1; ++i) {
    for (uword j = 0; j < y_cut.n_elem - 1; ++j) {
      
      // Extract relevant points from temp2 and temp3 for current grid
      uvec idx2_all = find(
        temp2.col(0) > (x_cut[i] - More) && 
          temp2.col(0) <= (x_cut[i + 1] + More) &&
          temp2.col(1) > (y_cut[j] - More) && 
          temp2.col(1) <= (y_cut[j + 1] + More)
      );
      
      uvec idx3 = find(
        temp3.col(0) > x_cut[i] && 
          temp3.col(0) <= x_cut[i + 1] &&
          temp3.col(1) > y_cut[j] && 
          temp3.col(1) <= y_cut[j + 1]
      );
      
      if (idx2_all.empty() || idx3.empty()) continue;
      
      arma::mat sub_temp2_all = temp2.rows(idx2_all);
      arma::mat sub_temp3 = temp3.rows(idx3);
      
      // Precompute coordinates for faster access
      const vec x2 = sub_temp2_all.col(0);
      const vec y2 = sub_temp2_all.col(1);
      const int n_sub3 = sub_temp3.n_rows;
      
      arma::mat measure_sub1(n_sub3, 4, fill::zeros);
      
      for (int k = 0; k < n_sub3; ++k) {
        const double x3 = sub_temp3(k, 0);
        const double y3 = sub_temp3(k, 1);
        const double z3 = sub_temp3(k, 2);
        
        // Efficient distance calculation using vectorization
        const vec dx = x2 - x3;
        const vec dy = y2 - y3;
        const vec dist2 = dx % dx + dy % dy;
        
        const uvec kernel_idx = find(dist2 <= hh_sq);
        if (kernel_idx.empty()) continue;
        
        const arma::mat B = sub_temp2_all.rows(kernel_idx);
        const int n_B = B.n_rows;
        
        double local_std = 0.0;
        arma::vec beta(6, fill::zeros);
        
        if (n_B >= 6) {
          const vec x = B.col(0);
          const vec y = B.col(1);
          const vec z = B.col(2);
          
          // Initial quadratic fit
          mat X(n_B, 6);
          X.col(0).ones();
          X.col(1) = x;
          X.col(2) = y;
          X.col(3) = x % x;
          X.col(4) = x % y;
          X.col(5) = y % y;
          
          mat XtX = X.t() * X;
          vec Xtz = X.t() * z;
          bool success = solve(beta, XtX, Xtz, solve_opts::fast);
          
          if (success) {
            vec residuals = z - X * beta;
            int df = n_B - 6;
            if (df > 0) local_std = std::sqrt((accu(square(residuals)) / df)); // Fixed
            
            // --- Filter and refit ---
            uvec keep = find(residuals <= local_std);
            if (keep.n_elem >= 6) {
              mat B_filtered = B.rows(keep);
              vec x_filt = B_filtered.col(0);
              vec y_filt = B_filtered.col(1);
              vec z_filt = B_filtered.col(2);
              
              mat X_filt(keep.n_elem, 6);
              X_filt.col(0).ones();
              X_filt.col(1) = x_filt;
              X_filt.col(2) = y_filt;
              X_filt.col(3) = x_filt % x_filt;
              X_filt.col(4) = x_filt % y_filt;
              X_filt.col(5) = y_filt % y_filt;
              
              mat XtX_filt = X_filt.t() * X_filt;
              vec Xtz_filt = X_filt.t() * z_filt;
              vec beta_filt;
              bool success_filt = solve(beta_filt, XtX_filt, Xtz_filt, solve_opts::fast);
              
              if (success_filt) {
                beta = beta_filt;
                residuals = z_filt - X_filt * beta_filt;
                df = keep.n_elem - 6;
                if (df > 0) local_std = std::sqrt((accu(square(residuals)) / df)); // Fixed
              }
            }
          }
        }
        
        const double z_fit = beta[0] + beta[1] * x3 + beta[2] * y3 + 
          beta[3] * x3 * x3 + beta[4] * x3 * y3 + beta[5] * y3 * y3;
        const double deviation = z3 - z_fit;
        
        
        // Update labels and measurements
        if (n_B >= 6 && deviation <= 2 * local_std) {
          labels_temp3(idx3[k]) = 1.0;
        }
        
        measure_sub1(k, 0) = z_fit;
        measure_sub1(k, 1) = local_std;
        measure_sub1(k, 2) = deviation;
        measure_sub1(k, 3) = (deviation > 2 * local_std) ? 1.0 : 0.0;
      }
      
      measure.rows(idx3) = measure_sub1;
    }
  }
  
  return List::create(
    Named("measure") = measure,
    Named("labels_temp3") = labels_temp3
  );
}

// [[Rcpp::export]]
List ground_filtering_uniform_cubic_filter_patches_optimized(
    const arma::mat& temp1, 
    const arma::mat& temp2, 
    const arma::mat& temp3, 
    double hh, 
    int Size = 40, 
    int More = 5, 
    double adjust = 1.5, 
    bool house = false, 
    int threshold = 8) {
  
  int n3 = temp3.n_rows;
  arma::mat measure(n3, 4, fill::zeros);
  arma::vec labels_temp3 = arma::zeros<arma::vec>(n3);
  
  double x_lower = std::floor(arma::min(temp1.col(0)) - 1e-8);
  double x_upper = std::floor(arma::max(temp1.col(0)) + 1e-8) + 1;
  int x_k = std::floor((x_upper - x_lower) / Size);
  arma::vec x_cut = linspace<vec>(x_lower, x_upper, x_k + 1);
  
  double y_lower = std::floor(arma::min(temp1.col(1)) - 1e-8);
  double y_upper = std::floor(arma::max(temp1.col(1)) + 1e-8) + 1;
  int y_k = std::floor((y_upper - y_lower) / Size);
  arma::vec y_cut = linspace<vec>(y_lower, y_upper, y_k + 1);
  
  const double hh_sq = hh * hh; // Precompute squared radius
  
  // Enable OpenMP for parallel processing (if supported)
  // #pragma omp parallel for collapse(2) schedule(dynamic)
  for (uword i = 0; i < x_cut.n_elem - 1; ++i) {
    for (uword j = 0; j < y_cut.n_elem - 1; ++j) {
      
      // Extract relevant points from temp2 and temp3 for current grid
      uvec idx2_all = find(
        temp2.col(0) > (x_cut[i] - More) && 
          temp2.col(0) <= (x_cut[i + 1] + More) &&
          temp2.col(1) > (y_cut[j] - More) && 
          temp2.col(1) <= (y_cut[j + 1] + More)
      );
      
      uvec idx3 = find(
        temp3.col(0) > x_cut[i] && 
          temp3.col(0) <= x_cut[i + 1] &&
          temp3.col(1) > y_cut[j] && 
          temp3.col(1) <= y_cut[j + 1]
      );
      
      if (idx2_all.empty() || idx3.empty()) continue;
      
      arma::mat sub_temp2_all = temp2.rows(idx2_all);
      arma::mat sub_temp3 = temp3.rows(idx3);
      
      // Precompute coordinates for faster access
      const vec x2 = sub_temp2_all.col(0);
      const vec y2 = sub_temp2_all.col(1);
      const int n_sub3 = sub_temp3.n_rows;
      
      arma::mat measure_sub1(n_sub3, 4, fill::zeros);
      
      for (int k = 0; k < n_sub3; ++k) {
        const double x3 = sub_temp3(k, 0);
        const double y3 = sub_temp3(k, 1);
        const double z3 = sub_temp3(k, 2);
        
        // Efficient distance calculation using vectorization
        const vec dx = x2 - x3;
        const vec dy = y2 - y3;
        const vec dist2 = dx % dx + dy % dy;
        
        const uvec kernel_idx = find(dist2 <= hh_sq);
        if (kernel_idx.empty()) continue;
        
        const arma::mat B = sub_temp2_all.rows(kernel_idx);
        const int n_B = B.n_rows;
        
        double local_std = 0.0;
        arma::vec beta(10, fill::zeros);
        
        if (n_B >= 10) {
          const vec x = B.col(0);
          const vec y = B.col(1);
          const vec z = B.col(2);
          
          // Initial quadratic fit
          mat X(n_B, 10);
          X.col(0).ones();
          X.col(1) = x;
          X.col(2) = y;
          X.col(3) = x % x;
          X.col(4) = x % y;
          X.col(5) = y % y;
          X.col(6) = x % x % x;
          X.col(7) = x % x % y;
          X.col(8) = x % y % y;
          X.col(9) = y % y % y;
          
          mat XtX = X.t() * X;
          vec Xtz = X.t() * z;
          bool success = solve(beta, XtX, Xtz, solve_opts::fast);
          
          if (success) {
            vec residuals = z - X * beta;
            int df = n_B - 10;
            if (df > 0) local_std = std::sqrt((accu(square(residuals)) / df)); // Fixed
            
            // --- Filter and refit ---
            uvec keep = find(residuals <= local_std);
            if (keep.n_elem >= 10) {
              mat B_filtered = B.rows(keep);
              vec x_filt = B_filtered.col(0);
              vec y_filt = B_filtered.col(1);
              vec z_filt = B_filtered.col(2);
              
              mat X_filt(keep.n_elem, 10);
              X_filt.col(0).ones();
              X_filt.col(1) = x_filt;
              X_filt.col(2) = y_filt;
              X_filt.col(3) = x_filt % x_filt;
              X_filt.col(4) = x_filt % y_filt;
              X_filt.col(5) = y_filt % y_filt;
              X_filt.col(6) = x_filt % x_filt % x_filt;
              X_filt.col(7) = x_filt % x_filt % y_filt;
              X_filt.col(8) = x_filt % y_filt % y_filt;
              X_filt.col(9) = y_filt % y_filt % y_filt;
              
              mat XtX_filt = X_filt.t() * X_filt;
              vec Xtz_filt = X_filt.t() * z_filt;
              vec beta_filt;
              bool success_filt = solve(beta_filt, XtX_filt, Xtz_filt, solve_opts::fast);
              
              if (success_filt) {
                beta = beta_filt;
                residuals = z_filt - X_filt * beta_filt;
                df = keep.n_elem - 10;
                if (df > 0) local_std = std::sqrt((accu(square(residuals)) / df)); // Fixed
              }
            }
          }
        }
        
        const double z_fit = beta[0] + beta[1] * x3 + beta[2] * y3 + 
          beta[3] * x3 * x3 + beta[4] * x3 * y3 + beta[5] * y3 * y3 +
          beta[6] * x3 * x3 * x3 + beta[7] * x3 * x3 * y3 + beta[8] * x3 * y3 * y3+
          beta[9] * y3 * y3 * y3;;
        const double deviation = z3 - z_fit;
        
        
        // Update labels and measurements
        if (n_B >= 10 && deviation <= 2 * local_std) {
          labels_temp3(idx3[k]) = 1.0;
        }
        
        measure_sub1(k, 0) = z_fit;
        measure_sub1(k, 1) = local_std;
        measure_sub1(k, 2) = deviation;
        measure_sub1(k, 3) = (deviation > 2 * local_std) ? 1.0 : 0.0;
      }
      
      measure.rows(idx3) = measure_sub1;
    }
  }
  
  return List::create(
    Named("measure") = measure,
    Named("labels_temp3") = labels_temp3
  );
}