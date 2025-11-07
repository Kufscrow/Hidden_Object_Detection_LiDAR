#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
// hyperbolic filtering without sectors
List ground_filtering_hyperbola_patches(const arma::mat& temp1, 
                                        const arma::mat& temp2, 
                                        const arma::vec& hh, 
                                        int Size = 40, 
                                        int More = 5, 
                                        double adjust = 1.5, 
                                        bool house = false, 
                                        int threshold = 8, 
                                        int outlier = 50) {
  
  int n1 = temp1.n_rows;
  // int n2 = temp2.n_rows;
  // int pp = temp1.n_cols;
  
  arma::mat measure(n1, 8, fill::zeros);
  
  double x_lower = std::floor(arma::min(temp1.col(0)) - 1e-8);
  double x_upper = std::floor(arma::max(temp1.col(0)) + 1e-8) + 1;
  int x_k = std::floor((x_upper - x_lower) / Size);
  
  arma::vec x_cut = linspace<vec>(x_lower, x_upper, x_k + 1);
  
  double y_lower = std::floor(arma::min(temp1.col(1)) - 1e-8);
  double y_upper = std::floor(arma::max(temp1.col(1)) + 1e-8) + 1;
  int y_k = std::floor((y_upper - y_lower) / Size);
  
  arma::vec y_cut = linspace<vec>(y_lower, y_upper, y_k + 1);
  
  for (unsigned int i = 0; i < x_cut.n_elem - 1; ++i) {
    for (unsigned int j = 0; j < y_cut.n_elem - 1; ++j) {
      
      uvec idx1 = find(temp1.col(0) > x_cut[i] && temp1.col(0) <= x_cut[i + 1] &&
        temp1.col(1) > y_cut[j] && temp1.col(1) <= y_cut[j + 1]);
      
      uvec idx2_all = find(temp2.col(0) > (x_cut[i] - More) && temp2.col(0) <= (x_cut[i + 1] + More) &&
        temp2.col(1) > (y_cut[j] - More) && temp2.col(1) <= (y_cut[j + 1] + More));
      
      arma::mat sub_temp1 = temp1.rows(idx1);
      arma::mat sub_temp2_all = temp2.rows(idx2_all);
      
      if ((int)sub_temp2_all.n_rows > outlier) {
        uvec order = sort_index(sub_temp2_all.col(2));
        sub_temp2_all = sub_temp2_all.rows(order.subvec(outlier, sub_temp2_all.n_rows - 1));
      }
      
      int n_sub1 = sub_temp1.n_rows;
      int n_sub2 = sub_temp2_all.n_rows;
      
      arma::mat measure_sub1(n_sub1, 8, fill::zeros);
      
      for (int k = 0; k < n_sub1; ++k) {
        arma::rowvec point_used = sub_temp1.row(k);
        arma::mat diffs_xy = square(repmat(point_used.subvec(0, 1), n_sub2, 1) - sub_temp2_all.cols(0, 1));
        arma::vec diffs_z = square(sub_temp2_all.col(2) - point_used(2));
        arma::vec distance_used = sum(diffs_xy, 1) / std::pow(hh(0), 2) - diffs_z / std::pow(hh(1), 2);
        
        uvec valid_idx = find(distance_used <= 1);
        
        if (!valid_idx.empty()) {
          arma::mat B = sub_temp2_all.rows(valid_idx);
          uvec index_lower = find(B.col(2) < (point_used(2) - adjust));
          uvec index_upper = find(B.col(2) >= (point_used(2) - adjust));
          uvec index_lower_not = find(B.col(2) <= point_used(2));
          uvec index_upper_not = find(B.col(2) > point_used(2));
          measure_sub1.row(k) = arma::rowvec({
            (double)index_lower.n_elem,
            (double)index_upper.n_elem,
            (double)index_lower_not.n_elem,
            (double)index_upper_not.n_elem,
            (double)(index_lower.n_elem == 0),
            (double)(index_lower_not.n_elem == 0),
            (double)(index_upper.n_elem == 0),
            (double)(index_upper_not.n_elem == 0),
          });
        }
      }
      measure.rows(idx1) = measure_sub1;
    }
  }
  List result;
  result["measure"] = wrap(measure);
  return result;
}