#define ARMA_WARN_LEVEL 1
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

List smooth_local_quadratic_fit_2D(mat X_mat, vec y_vec, int num_windows_x1, int num_windows_x2, double overlap) {
  int n = X_mat.n_rows;
  vec y_fit(n, fill::zeros);
  vec count(n, fill::zeros);  // Track how many times each point is predicted
  
  // Store coefficients for each window
  std::vector<mat> betas(num_windows_x1 * num_windows_x2);
  
  // Define window boundaries for x1 and x2
  rowvec x_min = min(X_mat, 0);
  rowvec x_max = max(X_mat, 0);
  double window_width_x1 = (x_max(0) - x_min(0)) / num_windows_x1;
  double window_width_x2 = (x_max(1) - x_min(1)) / num_windows_x2;
  double overlap_width_x1 = window_width_x1 * overlap;
  double overlap_width_x2 = window_width_x2 * overlap;
  
  // For each window, fit the local quadratic model
  for (int w1 = 0; w1 < num_windows_x1; w1++) {
    for (int w2 = 0; w2 < num_windows_x2; w2++) {
      rowvec left_bound = x_min + rowvec({static_cast<double>(w1), static_cast<double>(w2)}) % rowvec({window_width_x1, window_width_x2}) - rowvec({overlap_width_x1, overlap_width_x2}) / 2;
      rowvec right_bound = left_bound + rowvec({window_width_x1, window_width_x2}) + rowvec({overlap_width_x1, overlap_width_x2});
      
      // Select data points within the current window
      uvec indices = find(X_mat.col(0) >= left_bound(0) && X_mat.col(0) <= right_bound(0) &&
        X_mat.col(1) >= left_bound(1) && X_mat.col(1) <= right_bound(1));
      
      // if (indices.n_elem < 1) continue; // Skip if not enough points
      
      mat X_window = X_mat.rows(indices);
      vec y_window = y_vec.elem(indices);
      
      // Construct the design matrix for quadratic regression
      mat X_design(indices.n_elem, 6);
      X_design.col(0).ones();       // Intercept
      X_design.col(1) = X_window.col(0);  // x1
      X_design.col(2) = X_window.col(1);  // x2
      X_design.col(3) = square(X_window.col(0));  // x1^2
      X_design.col(4) = square(X_window.col(1));  // x2^2
      X_design.col(5) = X_window.col(0) % X_window.col(1);  // x1 * x2
      
      // Solve for beta using least squares
      vec beta = solve(X_design.t() * X_design, X_design.t() * y_window);
      
      // Store coefficients
      betas[w1 * num_windows_x2 + w2] = beta;
      
      // Predict for all points in this window
      vec y_pred = beta(0) + beta(1) * X_window.col(0) + beta(2) * X_window.col(1) +
        beta(3) * square(X_window.col(0)) + beta(4) * square(X_window.col(1)) +
        beta(5) * (X_window.col(0) % X_window.col(1));
      
      // Blend overlapping predictions using Euclidean distance to the center of the window
      rowvec window_center = left_bound + 0.5 * rowvec({window_width_x1, window_width_x2});
      vec weights = 1.0 / (1.0 + sqrt(square(X_window.col(0) - window_center(0)) + 
        square(X_window.col(1) - window_center(1))) / 
        sqrt(overlap_width_x1*overlap_width_x1 + overlap_width_x2*overlap_width_x2));
      
      y_fit.elem(indices) += weights % y_pred;
      count.elem(indices) += weights;
    }
  }
  
  // Normalize final predictions
  y_fit /= count;
  
  // Return both predicted values and list of coefficients
  return List::create(Named("y_fit") = y_fit,
                      Named("betas") = betas);
}

List smooth_local_linear_fit_2D(mat X_mat, vec y_vec, int num_windows_x1, int num_windows_x2, double overlap) {
  int n = X_mat.n_rows;
  vec y_fit(n, fill::zeros);
  vec count(n, fill::zeros);  // Track how many times each point is predicted
  
  // Store coefficients for each window
  std::vector<mat> betas(num_windows_x1 * num_windows_x2);
  
  // Define window boundaries for x1 and x2
  rowvec x_min = min(X_mat, 0);
  rowvec x_max = max(X_mat, 0);
  double window_width_x1 = (x_max(0) - x_min(0)) / num_windows_x1;
  double window_width_x2 = (x_max(1) - x_min(1)) / num_windows_x2;
  double overlap_width_x1 = window_width_x1 * overlap;
  double overlap_width_x2 = window_width_x2 * overlap;
  
  for (int w1 = 0; w1 < num_windows_x1; w1++) {
    for (int w2 = 0; w2 < num_windows_x2; w2++) {
      rowvec left_bound = x_min + rowvec({static_cast<double>(w1), static_cast<double>(w2)}) % rowvec({window_width_x1, window_width_x2}) - rowvec({overlap_width_x1, overlap_width_x2}) / 2;
      rowvec right_bound = left_bound + rowvec({window_width_x1, window_width_x2}) + rowvec({overlap_width_x1, overlap_width_x2});
      
      // Select data points within the current window
      uvec indices = find(X_mat.col(0) >= left_bound(0) && X_mat.col(0) <= right_bound(0) &&
        X_mat.col(1) >= left_bound(1) && X_mat.col(1) <= right_bound(1));
      
      // if (indices.n_elem < 1) continue; // Skip if not enough points
      
      mat X_window = X_mat.rows(indices);
      vec y_window = y_vec.elem(indices);
      
      // Construct the design matrix for linear regression
      mat X_design(indices.n_elem, 3);
      X_design.col(0).ones();       // Intercept
      X_design.col(1) = X_window.col(0);  // x1
      X_design.col(2) = X_window.col(1);  // x2
      
      // Solve for beta using least squares
      vec beta = solve(X_design.t() * X_design, X_design.t() * y_window);
      
      // Store coefficients
      betas[w1 * num_windows_x2 + w2] = beta;
      
      // Predict for all points in this window
      vec y_pred = beta(0) + beta(1) * X_window.col(0) + beta(2) * X_window.col(1);
      
      // Blend overlapping predictions using Euclidean distance to the center of the window
      rowvec window_center = left_bound + 0.5 * rowvec({window_width_x1, window_width_x2});
      vec weights = 1.0 / (1.0 + sqrt(square(X_window.col(0) - window_center(0)) + 
        square(X_window.col(1) - window_center(1))) / 
        sqrt(overlap_width_x1*overlap_width_x1 + overlap_width_x2*overlap_width_x2));
      
      y_fit.elem(indices) += weights % y_pred;
      count.elem(indices) += weights;
    }
  }
  
  // Normalize final predictions
  y_fit /= count;
  
  // Return both predicted values and list of coefficients
  return List::create(Named("y_fit") = y_fit,
                      Named("betas") = betas);
}

List smooth_robust_local_quadratic_fit_2D(mat X_mat, vec y_vec, int num_windows_x1, int num_windows_x2, double overlap, int max_iter = 5, double tol = 1e-6) {
  int n = X_mat.n_rows;
  vec y_fit(n, fill::zeros);
  vec count(n, fill::zeros);  // Track how many times each point is predicted
  
  // Store coefficients for each window
  std::vector<mat> betas(num_windows_x1 * num_windows_x2);
  
  // Define window boundaries for x1 and x2
  rowvec x_min = min(X_mat, 0);
  rowvec x_max = max(X_mat, 0);
  double window_width_x1 = (x_max(0) - x_min(0)) / num_windows_x1;
  double window_width_x2 = (x_max(1) - x_min(1)) / num_windows_x2;
  double overlap_width_x1 = window_width_x1 * overlap;
  double overlap_width_x2 = window_width_x2 * overlap;
  
  // For each window, fit the local quadratic model
  for (int w1 = 0; w1 < num_windows_x1; w1++) {
    for (int w2 = 0; w2 < num_windows_x2; w2++) {
      rowvec left_bound = x_min + rowvec({static_cast<double>(w1), static_cast<double>(w2)}) % rowvec({window_width_x1, window_width_x2}) - rowvec({overlap_width_x1, overlap_width_x2}) / 2;
      rowvec right_bound = left_bound + rowvec({window_width_x1, window_width_x2}) + rowvec({overlap_width_x1, overlap_width_x2});
      
      // Select data points within the current window
      uvec indices = find(X_mat.col(0) >= left_bound(0) && X_mat.col(0) <= right_bound(0) &&
        X_mat.col(1) >= left_bound(1) && X_mat.col(1) <= right_bound(1));
      
      if (indices.n_elem < 3) continue;  // Need at least 3 points for quadratic regression
      
      mat X_window = X_mat.rows(indices);
      vec y_window = y_vec.elem(indices);
      
      // Construct the design matrix for quadratic regression
      mat X_design(indices.n_elem, 6);
      X_design.col(0).ones();       // Intercept
      X_design.col(1) = X_window.col(0);  // x1
      X_design.col(2) = X_window.col(1);  // x2
      X_design.col(3) = square(X_window.col(0));  // x1^2
      X_design.col(4) = square(X_window.col(1));  // x2^2
      X_design.col(5) = X_window.col(0) % X_window.col(1);  // x1 * x2
      
      // Solve for beta using least squares
      vec beta = solve(X_design.t() * X_design, X_design.t() * y_window);
      vec residuals = y_window - X_design * beta;
      double sigma = median(abs(residuals)) / 0.6745;
      
      vec weights_alt(indices.n_elem, fill::ones);
      
      for (int iter = 0; iter < max_iter; iter++) {
        vec prev_beta = beta;
        
        for (unsigned int i = 0; i < indices.n_elem; i++) {
          double r = abs(residuals(i)) / (sigma + 1e-6);
          if (r <= 1.345) {
            weights_alt(i) = 1;
          } else {
            weights_alt(i) = 1.345 / r;
          }
        }
        
        mat W = diagmat(weights_alt);
        beta = solve(X_design.t() * W * X_design, X_design.t() * W * y_window);
        
        residuals = y_window - X_design * beta;
        if (norm(beta - prev_beta, 2) < tol) break;
      }
      
      // Store coefficients
      betas[w1 * num_windows_x2 + w2] = beta;
      
      // Predict for all points in this window
      vec y_pred = beta(0) + beta(1) * X_window.col(0) + beta(2) * X_window.col(1) +
        beta(3) * square(X_window.col(0)) + beta(4) * square(X_window.col(1)) +
        beta(5) * (X_window.col(0) % X_window.col(1));
      
      // Blend overlapping predictions using Euclidean distance to the center of the window
      rowvec window_center = left_bound + 0.5 * rowvec({window_width_x1, window_width_x2});
      vec weights = 1.0 / (1.0 + sqrt(square(X_window.col(0) - window_center(0)) + 
        square(X_window.col(1) - window_center(1))) / 
        sqrt(overlap_width_x1*overlap_width_x1 + overlap_width_x2*overlap_width_x2));
      
      y_fit.elem(indices) += weights % y_pred;
      count.elem(indices) += weights;
    }
  }
  
  // Normalize final predictions
  y_fit /= count;
  
  // Return both predicted values and list of coefficients
  return List::create(Named("y_fit") = y_fit,
                      Named("betas") = betas);
}

vec smooth_robust_local_linear_fit_2D(mat X_mat, vec y_vec, int num_windows_x1, int num_windows_x2, double overlap, int max_iter = 5, double tol = 1e-6) {
  int n = X_mat.n_rows;
  vec y_fit(n, fill::zeros);
  vec count(n, fill::zeros);  // Track how many times each point is predicted
  
  // Define window boundaries for x1 and x2
  rowvec x_min = min(X_mat, 0);
  rowvec x_max = max(X_mat, 0);
  double window_width_x1 = (x_max(0) - x_min(0)) / num_windows_x1;
  double window_width_x2 = (x_max(1) - x_min(1)) / num_windows_x2;
  double overlap_width_x1 = window_width_x1 * overlap;
  double overlap_width_x2 = window_width_x2 * overlap;
  
  for (int w1 = 0; w1 < num_windows_x1; w1++) {
    for (int w2 = 0; w2 < num_windows_x2; w2++) {
      rowvec left_bound = x_min + rowvec({static_cast<double>(w1), static_cast<double>(w2)}) % rowvec({window_width_x1, window_width_x2}) - rowvec({overlap_width_x1, overlap_width_x2}) / 2;
      rowvec right_bound = left_bound + rowvec({window_width_x1, window_width_x2}) + rowvec({overlap_width_x1, overlap_width_x2});
      
      // Select data points within the current window
      uvec indices = find(X_mat.col(0) >= left_bound(0) && X_mat.col(0) <= right_bound(0) &&
        X_mat.col(1) >= left_bound(1) && X_mat.col(1) <= right_bound(1));
      
      if (indices.n_elem < 3) continue;  // Need at least 3 points for linear regression
      
      mat X_window = X_mat.rows(indices);
      vec y_window = y_vec.elem(indices);
      
      // Construct the design matrix for linear regression
      mat X_design(indices.n_elem, 3);
      X_design.col(0).ones();       // Intercept
      X_design.col(1) = X_window.col(0);  // x1
      X_design.col(2) = X_window.col(1);  // x2
      
      // Solve for beta using least squares
      vec beta = solve(X_design.t() * X_design, X_design.t() * y_window);
      vec residuals = y_window - X_design * beta;
      double sigma = median(abs(residuals)) / 0.6745;
      
      vec weights_alt(indices.n_elem, fill::ones);
      
      for (int iter = 0; iter < max_iter; iter++) {
        vec prev_beta = beta;
        
        for (unsigned int i = 0; i < indices.n_elem; i++) {
          double r = abs(residuals(i)) / (sigma + 1e-6);
          if (r <= 1.345) {
            weights_alt(i) = 1;
          } else {
            weights_alt(i) = 1.345 / r;
          }
        }
        
        mat W = diagmat(weights_alt);
        beta = solve(X_design.t() * W * X_design, X_design.t() * W * y_window);
        
        residuals = y_window - X_design * beta;
        if (norm(beta - prev_beta, 2) < tol) break;
      }
      
      
      // Predict for all points in this window
      vec y_pred = beta(0) + beta(1) * X_window.col(0) + beta(2) * X_window.col(1);
      
      // Blend overlapping predictions using Euclidean distance to the center of the window
      rowvec window_center = left_bound + 0.5 * rowvec({window_width_x1, window_width_x2});
      vec weights = 1.0 / (1.0 + sqrt(square(X_window.col(0) - window_center(0)) + 
        square(X_window.col(1) - window_center(1))) / 
        sqrt(overlap_width_x1*overlap_width_x1 + overlap_width_x2*overlap_width_x2));
      
      y_fit.elem(indices) += weights % y_pred;
      count.elem(indices) += weights;
    }
  }
  
  // Normalize final predictions
  y_fit /= count;
  return y_fit;
}

vec predict_local_quadratic_2D(mat X1_mat, List betas, int num_windows_x1, int num_windows_x2, double overlap) {
  int n_new = X1_mat.n_rows;
  vec y_pred(n_new, fill::zeros);
  
  rowvec x_min = min(X1_mat, 0);
  rowvec x_max = max(X1_mat, 0);
  double window_width_x1 = (x_max(0) - x_min(0)) / num_windows_x1;
  double window_width_x2 = (x_max(1) - x_min(1)) / num_windows_x2;
  
  // Iterate over the new dataset X1
  for (int i = 0; i < n_new; i++) {
    rowvec x = X1_mat.row(i);
    
    // Find which window the point belongs to
    int w1 = floor((x(0) - x_min(0)) / window_width_x1);
    int w2 = floor((x(1) - x_min(1)) / window_width_x2);
    
    // Ensure window indices are within bounds
    w1 = std::max(0, std::min(w1, num_windows_x1 - 1));
    w2 = std::max(0, std::min(w2, num_windows_x2 - 1));
    
    // Retrieve the corresponding beta coefficients for this window
    vec beta = as<vec>(betas[w1 * num_windows_x2 + w2]);
    
    // Make the prediction using the quadratic model
    y_pred(i) = beta(0) + beta(1) * x(0) + beta(2) * x(1) +
      beta(3) * (x(0) * x(0)) + beta(4) * (x(1) * x(1)) +
      beta(5) * (x(0) * x(1));
  }
  
  return y_pred;
}

vec predict_local_linear_2D(mat X1_mat, List betas, int num_windows_x1, int num_windows_x2, double overlap) {
  int n_new = X1_mat.n_rows;
  vec y_pred(n_new, fill::zeros);
  
  rowvec x_min = min(X1_mat, 0);
  rowvec x_max = max(X1_mat, 0);
  double window_width_x1 = (x_max(0) - x_min(0)) / num_windows_x1;
  double window_width_x2 = (x_max(1) - x_min(1)) / num_windows_x2;
  
  // Iterate over the new dataset X1
  for (int i = 0; i < n_new; i++) {
    rowvec x = X1_mat.row(i);
    
    // Find which window the point belongs to
    int w1 = floor((x(0) - x_min(0)) / window_width_x1);
    int w2 = floor((x(1) - x_min(1)) / window_width_x2);
    
    // Ensure window indices are within bounds
    w1 = std::max(0, std::min(w1, num_windows_x1 - 1));
    w2 = std::max(0, std::min(w2, num_windows_x2 - 1));
    
    // Retrieve the corresponding beta coefficients for this window
    vec beta = as<vec>(betas[w1 * num_windows_x2 + w2]);
    
    // Make the prediction using the quadratic model
    y_pred(i) = beta(0) + beta(1) * x(0) + beta(2) * x(1);
  }
  
  return y_pred;
}

// [[Rcpp::export]]
List ground_fit_quadratic_local(mat point_data, int iter = 2, int num_windows_x1 = 4, int num_windows_x2 = 4,
                               double overlap = 0.3, bool noise = true, double coeff = 3) {
  int n = point_data.n_rows;
  int m = n;
  vec mean_vals = mean(point_data, 0).t();  // Compute column-wise mean
  
  // Centering the first two columns
  mat point_data_new = point_data;
  point_data_new.col(0) -= mean_vals(0);
  point_data_new.col(1) -= mean_vals(1);
  
  mat point_data_begin = point_data_new;
  vec pred;
  vec res_reg_residuals;
  List pred_results;
  List res_pred_reg;
  
  for (int i = 0; i < iter+1; ++i) {
    m = point_data_begin.n_rows;
    mat X(m, 2, fill::ones);
    X.col(0) = point_data_begin.col(0);
    X.col(1) = point_data_begin.col(1);
    
    // Local quadratic regression
    vec Y = point_data_begin.col(2);
    pred_results = smooth_local_quadratic_fit_2D(X, Y, num_windows_x1, num_windows_x2, overlap);
    vec y_fit = pred_results["y_fit"];
    vec residuals = Y - y_fit;
    
    // Fit residual regression
    mat X_res(m, 2, fill::ones);
    X_res.col(0) = point_data_begin.col(0);
    X_res.col(1) = point_data_begin.col(1);
    
    res_pred_reg = smooth_local_linear_fit_2D(X_res, residuals, num_windows_x1, num_windows_x2, overlap);
    vec res_pred = res_pred_reg["y_fit"];
    res_reg_residuals = residuals - res_pred;
    
    // Filter indices where residual is negative
    uvec res_new_lowhalf_index = find(res_reg_residuals < 0);
    point_data_begin = point_data_begin.rows(res_new_lowhalf_index);
  }
  
  // Final residual computation
  mat X_final(n, 2, fill::ones);
  X_final.col(0) = point_data_new.col(0);
  X_final.col(1) = point_data_new.col(1);
  List betas = pred_results["betas"];
  
  vec final_pred = predict_local_quadratic_2D(X_final, betas, num_windows_x1, num_windows_x2, overlap);
  vec final_res = point_data_new.col(2) - final_pred;
  
  // Compute sigma
  double sigma = std::sqrt(sum(square(res_reg_residuals)) / (m - 6));
  
  // Assign ground labels based on noise threshold
  arma::vec ground_label(n);
  for (int i = 0; i < n; ++i) {
    if (noise) {
      ground_label(i) = (final_res(i) < coeff * sigma && final_res(i) > -3 * sigma) + 1;
    } else {
      ground_label(i) = (final_res(i) < coeff * sigma) + 1;
    }
  }
  
  return List::create(Named("final_residual") = final_res,
                      Named("ground_label") = ground_label);
}

// [[Rcpp::export]]
List ground_fit_linear_local(mat point_data, int iter = 2, int num_windows_x1 = 4, int num_windows_x2 = 4,
                                double overlap = 0.3, bool noise = true, double coeff = 3, double low_coeff = 3) {
  int n = point_data.n_rows;
  int m = n;
  vec mean_vals = mean(point_data, 0).t();  // Compute column-wise mean
  
  // Centering the first two columns
  mat point_data_new = point_data;
  point_data_new.col(0) -= mean_vals(0);
  point_data_new.col(1) -= mean_vals(1);
  
  mat point_data_begin = point_data_new;
  vec pred;
  vec res_reg_residuals;
  List pred_results;
  List res_pred_reg;
  
  for (int i = 0; i < iter+1; ++i) {
    m = point_data_begin.n_rows;
    mat X(m, 2, fill::ones);
    X.col(0) = point_data_begin.col(0);
    X.col(1) = point_data_begin.col(1);
    
    // Local linear regression
    vec Y = point_data_begin.col(2);
    pred_results = smooth_local_linear_fit_2D(X, Y, num_windows_x1, num_windows_x2, overlap);
    vec y_fit = pred_results["y_fit"];
    vec residuals = Y - y_fit;
    
    // Fit residual regression
    mat X_res(m, 2, fill::ones);
    X_res.col(0) = point_data_begin.col(0);
    X_res.col(1) = point_data_begin.col(1);
    
    res_pred_reg = smooth_local_linear_fit_2D(X_res, residuals, num_windows_x1, num_windows_x2, overlap);
    vec res_pred = res_pred_reg["y_fit"];
    res_reg_residuals = residuals - res_pred;
    
    // Filter indices where residual is negative
    uvec res_new_lowhalf_index = find(res_reg_residuals < 0);
    point_data_begin = point_data_begin.rows(res_new_lowhalf_index);
  }
  
  // Final residual computation
  mat X_final(n, 2, fill::ones);
  X_final.col(0) = point_data_new.col(0);
  X_final.col(1) = point_data_new.col(1);
  List betas = pred_results["betas"];
  
  vec final_pred = predict_local_linear_2D(X_final, betas, num_windows_x1, num_windows_x2, overlap);
  vec final_res = point_data_new.col(2) - final_pred;
  
  // Compute sigma
  double sigma = std::sqrt(sum(square(res_reg_residuals)) / (m - 3));
  
  // Assign ground labels based on noise threshold
  arma::vec ground_label(n);
  for (int i = 0; i < n; ++i) {
    if (noise) {
      ground_label(i) = (final_res(i) < coeff * sigma) + 1 + 5 * (final_res(i) <= -low_coeff * sigma);
    } else {
      ground_label(i) = (final_res(i) < coeff * sigma) + 1;
    }
  }
  
  return List::create(Named("final_residual") = final_res,
                      Named("ground_label") = ground_label);
}

// [[Rcpp::export]]
List ground_fit_quadratic_local_alt(mat point_data, int iter = 2, int num_windows_x1 = 4, int num_windows_x2 = 4,
                                   double overlap = 0.3, bool noise = true, double coeff = 3, double low_coeff = 3) {
  int n = point_data.n_rows;
  int m = n;
  vec mean_vals = mean(point_data, 0).t();  // Compute column-wise mean
  
  // Centering the first two columns
  mat point_data_new = point_data;
  point_data_new.col(0) -= mean_vals(0);
  point_data_new.col(1) -= mean_vals(1);
  
  mat point_data_begin = point_data_new;
  vec pred;
  vec res_reg_residuals;
  List pred_results;
  List res_pred_reg;
  
  for (int i = 0; i < iter+1; ++i) {
    m = point_data_begin.n_rows;
    mat X(m, 2, fill::ones);
    X.col(0) = point_data_begin.col(0);
    X.col(1) = point_data_begin.col(1);
    
    // Local quadratic regression
    vec Y = point_data_begin.col(2);
    pred_results = smooth_robust_local_quadratic_fit_2D(X, Y, num_windows_x1, num_windows_x2, overlap);
    vec y_fit = pred_results["y_fit"];
    vec residuals = Y - y_fit;
    
    // Fit residual regression
    mat X_res(m, 2, fill::ones);
    X_res.col(0) = point_data_begin.col(0);
    X_res.col(1) = point_data_begin.col(1);
    
    res_pred_reg = smooth_local_linear_fit_2D(X_res, residuals, num_windows_x1, num_windows_x2, overlap);
    vec res_pred = res_pred_reg["y_fit"];
    res_reg_residuals = residuals - res_pred;
    
    // Filter indices where residual is negative
    uvec res_new_lowhalf_index = find(res_reg_residuals < 0);
    point_data_begin = point_data_begin.rows(res_new_lowhalf_index);
  }
  
  // Final residual computation
  mat X_final(n, 2, fill::ones);
  X_final.col(0) = point_data_new.col(0);
  X_final.col(1) = point_data_new.col(1);
  List betas = pred_results["betas"];
  
  vec final_pred = predict_local_quadratic_2D(X_final, betas, num_windows_x1, num_windows_x2, overlap);
  vec final_res = point_data_new.col(2) - final_pred;
  
  // Compute sigma
  double sigma = std::sqrt(sum(square(res_reg_residuals)) / (m - 6));
  
  // Assign ground labels based on noise threshold
  arma::vec ground_label(n);
  for (int i = 0; i < n; ++i) {
    if (noise) {
      ground_label(i) = (final_res(i) < coeff * sigma) + 1 + 5 * (final_res(i) <= -low_coeff * sigma);
    } else {
      ground_label(i) = (final_res(i) < coeff * sigma) + 1;
    }
  }
  
  return List::create(Named("final_residual") = final_res,
                      Named("ground_label") = ground_label, 
                      Named("sigma") = sigma);
}
