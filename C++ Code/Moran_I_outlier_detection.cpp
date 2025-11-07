#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::vec moran_anisotropic_vec(const arma::mat& points, int k, double sigma_x, double sigma_y, double sigma_z) {
  int n = points.n_rows;
  arma::vec moran_i(n, fill::zeros);
  
  // Step 1: Nearest neighbor search (using R's nn2)
  NumericMatrix points_r = wrap(points);
  Function nn2("nn2");
  List knn_result = nn2(points_r, points_r, Named("k") = k + 1);
  arma::umat neighbors = as<arma::umat>(knn_result["nn.idx"]) - 1;  // 0-based indexing
  arma::mat distances = as<arma::mat>(knn_result["nn.dists"]);
  
  // Remove self from neighbors
  neighbors.shed_col(0);
  distances.shed_col(0);
  
  // Step 2: Fully vectorized anisotropic Gaussian kernel calculation
  arma::cube diff(n, k, 3, fill::zeros);
  
  for (int i = 0; i < n; ++i) {
    // Extract neighbors
    arma::mat neighbor_points = points.rows(neighbors.row(i));
    
    // Broadcast current point across all neighbors
    arma::mat current_point = arma::repmat(points.row(i), k, 1);
    
    // Assign the entire slice rather than a single row
    diff.tube(i, 0, i, k - 1) = neighbor_points - current_point;
  }
  
  // Apply the anisotropic Gaussian kernel formula
  arma::mat weights = exp(
    -pow(diff.slice(0), 2) / (2 * sigma_x * sigma_x)
    -pow(diff.slice(1), 2) / (2 * sigma_y * sigma_y)
    -pow(diff.slice(2), 2) / (2 * sigma_z * sigma_z)
  );
  
  // Step 3: Use mean distance as the local feature
  arma::vec values = mean(distances, 1);
  double mean_val = mean(values);
  double var_val = var(values);
  
  // Step 4: Fully vectorized Local Moran's I calculation
  arma::mat neighbor_vals(n, k);
  for (int i = 0; i < n; ++i) {
    neighbor_vals.row(i) = values.elem(neighbors.row(i)).t();
  }
  
  arma::mat mean_matrix = arma::repmat(arma::mat(1, 1, fill::value(mean_val)), n, k);
  arma::mat mean_diff = neighbor_vals - mean_matrix;  // (Xj - mean)
  
  arma::mat w_mean_diff = weights % mean_diff;  // Element-wise multiplication
  
  // Moran's I calculation
  moran_i = (values - mean_val) % sum(w_mean_diff, 1) / var_val;
  
  return moran_i;
}