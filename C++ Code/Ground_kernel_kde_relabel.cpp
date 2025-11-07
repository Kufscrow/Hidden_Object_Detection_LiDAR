#include <RcppArmadillo.h>
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
List relabel_cliffs_labels(const arma::mat& points, const arma::vec& labels,
                                double grid_res = 1.0, double grad_thresh = 45.0,
                                double kde_thresh = 0.8, double mu_z = -5.0) {
  // Extract ground/non-ground indices
  uvec ground_idx = find(labels == 2);
  uvec ng_idx = find(labels == 1);
  
  mat ground_pts = points.rows(ground_idx);
  mat ng_pts = points.rows(ng_idx);
  
  // Create elevation grid from ground points
  double xmin = ground_pts.col(0).min();
  double ymin = ground_pts.col(1).min();
  double xmax = ground_pts.col(0).max();
  double ymax = ground_pts.col(1).max();
  
  uword grid_nx = ceil((xmax - xmin) / grid_res);
  uword grid_ny = ceil((ymax - ymin) / grid_res);
  
  mat elevation_grid(grid_nx, grid_ny, fill::zeros);
  mat count_grid(grid_nx, grid_ny, fill::zeros);
  
  // Bin ground points into grid cells
  for(uword i = 0; i < ground_pts.n_rows; ++i) {
    uword ix = (ground_pts(i, 0) - xmin) / grid_res;
    uword iy = (ground_pts(i, 1) - ymin) / grid_res;
    if(ix < grid_nx && iy < grid_ny) {
      elevation_grid(ix, iy) += ground_pts(i, 2);
      count_grid(ix, iy) += 1;
    }
  }
  elevation_grid /= (count_grid + 1e-9);  // Avoid division by zero
  
  // Sobel edge detection
  mat sobel_x = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  mat sobel_y = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  mat grad_x = conv2(elevation_grid, sobel_x, "same");
  mat grad_y = conv2(elevation_grid, sobel_y, "same");
  mat grad_mag = sqrt(square(grad_x) + square(grad_y));
  
  // Dilate cliff regions
  umat cliff_grid = grad_mag > grad_thresh;
  umat dilated_grid = cliff_grid;
  for(uword i = 1; i < grid_nx - 1; ++i) {
    for(uword j = 1; j < grid_ny - 1; ++j) {
      if(cliff_grid(i, j)) {
        dilated_grid.submat(i-1, j-1, i+1, j+1).fill(1);
      }
    }
  }
  
  // Process non-ground points
  vec new_labels = labels;
  double sigma_xy = 1.0;
  double sigma_z = 2.0;
  vec kde_ng_all = zeros<vec>(ng_pts.n_rows);
  
  for(uword i = 0; i < ng_pts.n_rows; ++i) {
    double x = ng_pts(i, 0);
    double y = ng_pts(i, 1);
    double z = ng_pts(i, 2);
    
    uword ix = (x - xmin) / grid_res;
    uword iy = (y - ymin) / grid_res;
    
    if(ix < grid_nx && iy < grid_ny && dilated_grid(ix, iy)) {
      double cell_z = elevation_grid(ix, iy);
      double dxy = grid_res;
      double dz = z - cell_z;
      
      double kde_score = exp(-pow(dxy, 2)/(2*pow(sigma_xy, 2))) *
        exp(-pow(dz - mu_z, 2)/(2*pow(sigma_z, 2)));
      kde_ng_all(i) = kde_score;
      
      if(kde_score > kde_thresh) {
        new_labels(ng_idx(i)) = 2;  // Update original index
      }
    }
  }
  
  return List::create(Named("label") = new_labels, 
                      Named("grad") = grad_mag, 
                      Named("cliff") = cliff_grid, 
                      Named("kde") = kde_ng_all);
}