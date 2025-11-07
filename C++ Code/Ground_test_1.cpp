#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List local_regression_cpp(const arma::mat& temp1, const arma::mat& temp2,
                          double hh, double adjust = 2.0, int Size = 40) {
  int n1 = temp1.n_rows;
  int n2 = temp2.n_rows;
  int p = temp1.n_cols;
  
  mat measure(n1, 2, fill::zeros);
  mat COEFF(n1, p, fill::zeros);
  ivec count(n1, fill::zeros);
  mat nearest(n1, 10, fill::zeros);
  
  for (int i = 0; i < n1; ++i) {
    cout << i << "\n";
    rowvec point_used = temp1.row(i);
    mat A = repmat(point_used, n2, 1) - temp2;
    
    A.col(2) *= std::sqrt(adjust);  // Only Z coordinate adjusted
    vec distance_used = sqrt(sum(square(A), 1));
    
    uvec sorted_indices = sort_index(distance_used);
    int index_current = sorted_indices(0);
    uvec index4 = sorted_indices.subvec(1, std::min(Size, n2 - 1));
    
    uvec index_hh = find((distance_used <= hh) && (distance_used > 1e-8));
    uvec index_all = unique(join_vert(index4, index_hh));
    index_all = sort(index_all);
    
    uvec index_used(index_all.n_elem + 1);
    index_used(0) = index_current;
    index_used.subvec(1, index_all.n_elem) = index_all;
    
    mat temp_selected = temp2.rows(index_used);
    
    // Prepare X matrix for regression (with intercept)
    vec x1 = temp_selected.col(0);
    vec x2 = temp_selected.col(1);
    vec y = temp_selected.col(2);
    mat X(temp_selected.n_rows, 3, fill::ones);
    X.col(1) = x1;
    X.col(2) = x2;
    
    vec coeff = solve(X, y);  // Linear regression
    vec residuals = y - X * coeff;
    double sigma = std::sqrt(mean(square(residuals)));
    double rsq = 1.0 - accu(square(residuals)) / accu(square(y - mean(y)));
    
    // Save results
    measure(i, 0) = sigma;
    measure(i, 1) = rsq;
    COEFF(i, 0) = coeff(0);  // Intercept
    COEFF(i, 1) = coeff(1);  // x1
    COEFF(i, 2) = coeff(2);  // x2
    count(i) = 1 + index_hh.n_elem;
    
    uvec near_idx = sorted_indices.subvec(1, 10);
    nearest.row(i) = trans(distance_used(near_idx));
  }
  
  return List::create(
    Named("measure") = measure,
    Named("COEFF") = COEFF,
    Named("count") = count,
    Named("nearest") = nearest
  );
}
