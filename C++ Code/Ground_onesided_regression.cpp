#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Helper function for linear regression
List lm_fit(const vec& y, const mat& X) {
  // Add intercept column
  mat X_with_intercept = join_horiz(ones<vec>(X.n_rows), X);
  
  // Solve linear system: (X'X) * beta = X'y
  vec coeff = solve(X_with_intercept, y);
  
  // Calculate residuals and sigma
  vec residuals = y - X_with_intercept * coeff;
  double sigma = std::sqrt(accu(square(residuals)) / (X.n_rows - X.n_cols - 1));
  
  return List::create(
    Named("coeff") = coeff,
    Named("residuals") = residuals,
    Named("sigma") = sigma
  );
}

// oneside_half_reg function
List oneside_half_reg(const vec& y, const mat& x, const vec& bound = vec({-2.0, 2.0})) {
  int n = x.n_rows;
  int p = x.n_cols;
  
  // Initialize beta estimates (including intercept)
  vec BETA_est = zeros<vec>(p + 1);
  int iteration = 0;
  double DIFF = 1.0;
  double SSE_new = accu(square(y - min(y)));
  
  // Golden ratio constant
  double golden_ratio = (std::sqrt(5.0) - 1.0) / 2.0;
  
  while ((iteration < 10) && (DIFF > 1e-6)) {
    iteration++;
    double SSE_old = SSE_new;
    
    // Loop through each predictor
    for (int i = 0; i < p; i++) {
      // Create indices for columns excluding i
      uvec other_cols = regspace<uvec>(0, p - 1);
      other_cols.shed_row(i);
      
      // Calculate y_hat = y - sum(x[,-i] * BETA_est[-c(1,i+1)])
      vec y_hat = y - x.cols(other_cols) * BETA_est.elem(other_cols + 1);
      vec x_selected = x.col(i);
      
      // Initialize BETA_temp matrix [intercept, slope, SSE]
      mat BETA_temp(4, 3);
      BETA_temp(0, 1) = bound(0);  // lower bound
      BETA_temp(3, 1) = bound(1);  // upper bound
      BETA_temp(1, 1) = bound(1) - (bound(1) - bound(0)) * golden_ratio;
      BETA_temp(2, 1) = bound(0) + (bound(1) - bound(0)) * golden_ratio;
      
      // Calculate intercept and SSE for each candidate slope
      for (int j = 0; j < 4; j++) {
        double slope = BETA_temp(j, 1);
        BETA_temp(j, 0) = min(y_hat - slope * x_selected);  // intercept
        vec residuals = y_hat - BETA_temp(j, 0) - slope * x_selected;
        BETA_temp(j, 2) = accu(square(residuals));  // SSE
      }
      
      // Find minimum SSE
      uword j_min = index_min(BETA_temp.col(2));
      
      // Golden section search iteration
      if (j_min <= 1) {  // minimum in first two points
        BETA_temp.row(2) = BETA_temp.row(1);
        BETA_temp.row(3) = BETA_temp.row(2);
        BETA_temp(1, 1) = BETA_temp(3, 1) - (BETA_temp(3, 1) - BETA_temp(0, 1)) * golden_ratio;
      } else {  // minimum in last two points
        BETA_temp.row(0) = BETA_temp.row(1);
        BETA_temp.row(1) = BETA_temp.row(2);
        BETA_temp(2, 1) = BETA_temp(0, 1) + (BETA_temp(3, 1) - BETA_temp(0, 1)) * golden_ratio;
      }
      
      // Recalculate for new points
      for (int j = 1; j <= 2; j++) {
        double slope = BETA_temp(j, 1);
        BETA_temp(j, 0) = min(y_hat - slope * x_selected);
        vec residuals = y_hat - BETA_temp(j, 0) - slope * x_selected;
        BETA_temp(j, 2) = accu(square(residuals));
      }
      
      int iteration_i = 0;
      while ((iteration_i < 20) && ((BETA_temp(3, 1) - BETA_temp(0, 1)) > 1e-6)) {
        iteration_i++;
        j_min = index_min(BETA_temp.col(2));
        
        if (j_min <= 1) {  // minimum in first two points
          BETA_temp.row(2) = BETA_temp.row(1);
          BETA_temp.row(3) = BETA_temp.row(2);
          BETA_temp(1, 1) = BETA_temp(3, 1) - (BETA_temp(3, 1) - BETA_temp(0, 1)) * golden_ratio;
        } else {  // minimum in last two points
          BETA_temp.row(0) = BETA_temp.row(1);
          BETA_temp.row(1) = BETA_temp.row(2);
          BETA_temp(2, 1) = BETA_temp(0, 1) + (BETA_temp(3, 1) - BETA_temp(0, 1)) * golden_ratio;
        }
        
        // Recalculate for new middle points
        int start_j = (j_min <= 1) ? 1 : 2;
        int end_j = (j_min <= 1) ? 2 : 3;
        
        for (int j = start_j; j < end_j; j++) {
          double slope = BETA_temp(j, 1);
          BETA_temp(j, 0) = min(y_hat - slope * x_selected);
          vec residuals = y_hat - BETA_temp(j, 0) - slope * x_selected;
          BETA_temp(j, 2) = accu(square(residuals));
        }
      }
      
      // Update coefficients
      j_min = index_min(BETA_temp.col(2));
      BETA_est(0) = BETA_temp(j_min, 0);        // intercept
      BETA_est(i + 1) = BETA_temp(j_min, 1);    // slope for i-th predictor
      SSE_new = BETA_temp(j_min, 2);
    }
    
    DIFF = SSE_old - SSE_new;
  }
  
  return List::create(
    Named("coeff") = BETA_est,
    Named("SSE") = SSE_new
  );
}

// oneside_reg_halfbottom_adj function
// [[Rcpp::export]]
List oneside_reg_halfbottom_adj(const vec& y, const mat& x, const vec& fic, int iter) {
  int n = x.n_rows;
  int p = x.n_cols;
  
  // Use R's density function for y
  Rcpp::Function density("density");
  Rcpp::List y_density = density(y);
  vec dens_x = Rcpp::as<vec>(y_density["x"]);
  vec dens_y = Rcpp::as<vec>(y_density["y"]);
  
  uword max_idx = index_max(dens_y);
  double dens = dens_x(max_idx);
  
  // Initial regression
  List g = oneside_half_reg(y, x);
  vec coeff = g["coeff"];
  
  // Calculate residuals
  vec res = y - (coeff(0) + x * coeff.tail(p));
  Rcpp::List res_density = density(res);
  vec res_dens_x = Rcpp::as<vec>(res_density["x"]);
  vec res_dens_y = Rcpp::as<vec>(res_density["y"]);
  
  // Adjust residuals by fic (assuming fic is a scalar or vector of same length)
  res = res - fic(0);  // Using first element if fic is vector
  
  // Initial outlier detection
  uvec index_negative = find(res < 0);
  double sigmasq_hat = 0.0;
  
  if (index_negative.n_elem > 0) {
    sigmasq_hat = mean(square(res.elem(index_negative)));
  }
  
  double threshold = std::sqrt(2.0 * std::log(n)) * std::sqrt(sigmasq_hat);
  uvec index_outlier = find(res > threshold);
  uvec index_remain = find(res <= threshold);
  
  // Iterative process
  for (int i = 0; i < iter; ++i) {
    if (index_remain.n_elem == 0) break;
    
    vec y_remain = y.elem(index_remain);
    mat x_remain = x.rows(index_remain);
    
    // Fit linear regression on remaining points
    List g_remain = lm_fit(y_remain, x_remain);
    vec coeff_remain = g_remain["coeff"];
    
    // Predict for all points
    vec y_pred = coeff_remain(0) + x * coeff_remain.tail(p);
    res = y - y_pred;
    
    // Update outlier indices
    index_negative = find(res < 0);
    if (index_negative.n_elem > 0) {
      sigmasq_hat = mean(square(res.elem(index_negative)));
    }
    
    threshold = std::sqrt(2.0 * std::log(n)) * std::sqrt(sigmasq_hat);
    index_outlier = find(res > threshold);
    index_remain = find(res <= threshold);
  }
  
  // Final fit on remaining points
  vec y_remain = y.elem(index_remain);
  mat x_remain = x.rows(index_remain);
  List g_remain = lm_fit(y_remain, x_remain);
  vec final_coeff = g_remain["coeff"];
  double final_sigma = g_remain["sigma"];
  
  // Convert indices to binary vectors for consistency with R
  vec outlier_vec = zeros<vec>(n);
  outlier_vec.elem(index_outlier).fill(1);
  
  vec remain_vec = zeros<vec>(n);
  remain_vec.elem(index_remain).fill(1);
  
  return List::create(
    Named("outlier") = outlier_vec,
    Named("remain") = remain_vec,
    Named("model") = g_remain,
    Named("sigma") = final_sigma,
    Named("coeff") = final_coeff
  );
}

// height_ramp_base_densmax function
// [[Rcpp::export]]
vec height_ramp_base_densmax(const vec& abs_dist, const vec& outlier, double sigma, 
                             double zfix, double hcoef, double q) {
  
  int n = abs_dist.n_elem;
  
  // Calculate quantile range
  double min_dist = min(abs_dist);
  double max_dist = max(abs_dist);
  double point_data_range_quantile = min_dist + (max_dist - min_dist) * q;
  
  // Find indices below quantile
  uvec indice_below_quantile = find(abs_dist <= point_data_range_quantile);
  
  double tol = 0.1;
  int z_desrandmax;
  
  if (indice_below_quantile.n_elem > 1) {
    // Use R's density function
    Rcpp::Function density("density");
    vec dist_below = abs_dist.elem(indice_below_quantile);
    Rcpp::List z_dens_quantile = density(dist_below);
    vec dens_x = Rcpp::as<vec>(z_dens_quantile["x"]);
    vec dens_y = Rcpp::as<vec>(z_dens_quantile["y"]);
    
    // Find x value at maximum density
    uword max_idx = index_max(dens_y);
    double z_densmax_quantile = dens_x(max_idx);
    
    // Find tolerance indices
    uvec z_tolindex = find(abs(abs_dist - z_densmax_quantile) <= tol);
    
    // Expand tolerance if no points found
    while (z_tolindex.n_elem == 0) {
      tol += 10.0;
      z_tolindex = find(abs(abs_dist - z_densmax_quantile) <= tol);
    }
    
    // Randomly sample one index from tolerance indices
    uvec random_idx = randi<uvec>(1, distr_param(0, z_tolindex.n_elem - 1));
    z_desrandmax = z_tolindex(random_idx(0));
  } else {
    z_desrandmax = indice_below_quantile(0);
  }
  
  double base_height;
  if (accu(outlier) == n) {  // sum(outlier) == length(abs_dist)
    base_height = zfix * hcoef;
  } else {
    base_height = abs_dist(z_desrandmax);
  }
  
  vec height = abs_dist - base_height;
  return height;
}

// Main function: ground_fit_base_densmax_halfbottom_fix
// [[Rcpp::export]]
List ground_fit_base_densmax_halfbottom_fix(const mat& point_data, double a, double b, double h, 
                                            double xmin, double ymin, double q, 
                                            bool pla_fix, const vec& ini_fix, int iter) {
  
  // Extract columns - note: Armadillo uses 0-based indexing
  vec z = point_data.col(2);  // third column
  mat xy = point_data.cols(0, 1);  // first two columns
  
  // Call the regression function
  List result = oneside_reg_halfbottom_adj(z, xy, ini_fix, iter);
  
  // Extract coefficients from result
  vec coeff = result["coeff"];
  
  double z_fix;
  if (pla_fix) {
    // Calculate residuals
    vec z_act = z - (coeff(0) + coeff(1) * xy.col(0) + coeff(2) * xy.col(1));
    
    // Use R's density function via Rcpp
    Rcpp::Function density("density");
    Rcpp::List z_actdens = density(z_act);
    vec dens_x = Rcpp::as<vec>(z_actdens["x"]);
    vec dens_y = Rcpp::as<vec>(z_actdens["y"]);
    
    // Find x value at maximum density
    uword max_idx = index_max(dens_y);
    z_fix = dens_x(max_idx);
  } else {
    z_fix = 0.0;
  }
  
  // Calculate absolute distances
  double denom = sqrt(coeff(1)*coeff(1) + coeff(2)*coeff(2) + 1.0);
  vec abs_dist = (-coeff(1) * xy.col(0) - coeff(2) * xy.col(1) + z - coeff(0) - z_fix) / denom;
  
  double hcoef = 1.0 / denom;
  
  // Calculate grid indices
  vec gridnx = floor((xy.col(0) - xmin) / (2.0 * a));
  vec gridny = floor((xy.col(1) - ymin) / (2.0 * b));
  
  // Extract sigma and outlier from result
  double sigma = result["sigma"];
  vec outlier = result["outlier"];
  
  // Initialize height vector
  vec height = zeros<vec>(abs_dist.n_elem);
  
  // Get maximum grid indices
  int max_gridnx = max(gridnx);
  int max_gridny = max(gridny);
  
  // Loop over grid cells
  for (int l = 0; l <= max_gridnx; ++l) {
    for (int m = 0; m <= max_gridny; ++m) {
      // Find indices in current grid cell
      uvec indices = find((gridnx == l) && (gridny == m));
      
      if (indices.n_elem > 0) {
        // Extract partial data
        vec dist_partial = abs_dist.elem(indices);
        vec outlier_partial = outlier.elem(indices);
        
        // Calculate height for this grid cell
        vec height_partial = height_ramp_base_densmax(dist_partial, outlier_partial, 
                                                      sigma, z_fix, hcoef, q);
        
        // Store results
        height.elem(indices) = height_partial;
      }
    }
  }
  
  // Calculate indicator
  uvec indice_height = (height <= h * hcoef);
  
  return List::create(
    Named("height") = height,
    Named("indice_height") = indice_height + 1  // Convert to 1-based indexing for R
  );
}
