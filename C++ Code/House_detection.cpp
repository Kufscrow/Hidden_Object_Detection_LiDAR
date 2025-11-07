#include <RcppArmadillo.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Union-Find (Disjoint Set Union) data structure
class UnionFind {
private:
  std::vector<int> parent;
  std::vector<int> rank;
  
public:
  UnionFind(int n) {
    parent.resize(n);
    rank.resize(n, 0);
    for (int i = 0; i < n; i++) {
      parent[i] = i;
    }
  }
  
  int find(int x) {
    if (parent[x] != x) {
      parent[x] = find(parent[x]); // Path compression
    }
    return parent[x];
  }
  
  void unite(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);
    
    if (rootX != rootY) {
      // Union by rank
      if (rank[rootX] < rank[rootY]) {
        parent[rootX] = rootY;
      } else if (rank[rootX] > rank[rootY]) {
        parent[rootY] = rootX;
      } else {
        parent[rootY] = rootX;
        rank[rootX]++;
      }
    }
  }
};

// [[Rcpp::export]]
arma::vec detect_houses_with_features_threshold_cpp(
    const arma::mat& points, 
    const arma::vec& metrics, 
    double planarity_threshold = 0.7,
    int min_cluster_size = 50,
    double radius = 2.0) {
  
  int n_points = points.n_rows;
  arma::vec house_labels = zeros<vec>(n_points);
  
  // Step 1: Identify house candidate points
  std::vector<int> candidate_indices;
  for (int i = 0; i < n_points; i++) {
    if (metrics(i) > metrics.max() - planarity_threshold) {
      candidate_indices.push_back(i);
    }
  }
  
  if (candidate_indices.empty()) {
    Rcpp::Rcout << "No house candidates found." << std::endl;
    return house_labels;
  }
  
  int n_candidates = candidate_indices.size();
  
  // Step 2: Create a mapping from global index to candidate index
  std::vector<int> global_to_candidate(n_points, -1);
  for (int i = 0; i < n_candidates; i++) {
    global_to_candidate[candidate_indices[i]] = i;
  }
  
  // Step 3: Use Union-Find to find connected components
  UnionFind uf(n_candidates);
  double radius_sq = radius * radius;
  
  // Build a simple spatial index (grid) to reduce comparisons
  double min_x = arma::min(points.col(0));
  double max_x = arma::max(points.col(0));
  double min_y = arma::min(points.col(1));
  double max_y = arma::max(points.col(1));
  
  int grid_size = static_cast<int>((max_x - min_x) / radius) + 1;
  std::vector<std::vector<int>> grid(grid_size * grid_size);
  
  // Assign candidate points to grid cells
  for (int i = 0; i < n_candidates; i++) {
    int global_idx = candidate_indices[i];
    double x = points(global_idx, 0);
    double y = points(global_idx, 1);
    
    int grid_x = static_cast<int>((x - min_x) / radius);
    int grid_y = static_cast<int>((y - min_y) / radius);
    int grid_idx = grid_y * grid_size + grid_x;
    
    if (grid_idx < grid.size()) {
      grid[grid_idx].push_back(i);
    }
  }
  
  // Find connected components using Union-Find
  for (int grid_idx = 0; grid_idx < grid.size(); grid_idx++) {
    const std::vector<int>& cell = grid[grid_idx];
    
    // Check points in the same cell
    for (int i = 0; i < cell.size(); i++) {
      int idx_i = cell[i];
      int global_i = candidate_indices[idx_i];
      const arma::rowvec& point_i = points.row(global_i);
      
      for (int j = i + 1; j < cell.size(); j++) {
        int idx_j = cell[j];
        int global_j = candidate_indices[idx_j];
        const arma::rowvec& point_j = points.row(global_j);
        
        double dist_sq = arma::sum(arma::square(point_i - point_j));
        if (dist_sq < radius_sq) {
          uf.unite(idx_i, idx_j);
        }
      }
      
      // Check points in adjacent cells
      int grid_x = grid_idx % grid_size;
      int grid_y = grid_idx / grid_size;
      
      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          if (dx == 0 && dy == 0) continue;
          
          int neighbor_x = grid_x + dx;
          int neighbor_y = grid_y + dy;
          int neighbor_idx = neighbor_y * grid_size + neighbor_x;
          
          if (neighbor_x >= 0 && neighbor_x < grid_size &&
              neighbor_y >= 0 && neighbor_y < grid_size &&
              neighbor_idx < grid.size()) {
            
            const std::vector<int>& neighbor_cell = grid[neighbor_idx];
            for (int k = 0; k < neighbor_cell.size(); k++) {
              int idx_k = neighbor_cell[k];
              int global_k = candidate_indices[idx_k];
              const arma::rowvec& point_k = points.row(global_k);
              
              double dist_sq = arma::sum(arma::square(point_i - point_k));
              if (dist_sq < radius_sq) {
                uf.unite(idx_i, idx_k);
              }
            }
          }
        }
      }
    }
  }
  
  // Step 4: Count cluster sizes and filter by minimum size
  std::vector<int> root_sizes(n_candidates, 0);
  for (int i = 0; i < n_candidates; i++) {
    int root = uf.find(i);
    root_sizes[root]++;
  }
  
  // Create a mapping from root to cluster ID (only for clusters that meet the size requirement)
  std::unordered_map<int, int> root_to_cluster;
  int cluster_id = 1;
  for (int i = 0; i < n_candidates; i++) {
    int root = uf.find(i);
    if (root_sizes[root] >= min_cluster_size) {
      if (root_to_cluster.find(root) == root_to_cluster.end()) {
        root_to_cluster[root] = cluster_id++;
      }
      house_labels[candidate_indices[i]] = root_to_cluster[root];
    }
  }
  
  // Step 5: Refine house boundaries (add points surrounded by houses)
  // Create a grid for all points for efficient neighborhood queries
  std::vector<std::vector<int>> all_points_grid(grid_size * grid_size);
  for (int i = 0; i < n_points; i++) {
    double x = points(i, 0);
    double y = points(i, 1);
    
    int grid_x = static_cast<int>((x - min_x) / radius);
    int grid_y = static_cast<int>((y - min_y) / radius);
    int grid_idx = grid_y * grid_size + grid_x;
    
    if (grid_idx < all_points_grid.size()) {
      all_points_grid[grid_idx].push_back(i);
    }
  }
  
  // Find points that are mostly surrounded by houses
  for (int grid_idx = 0; grid_idx < all_points_grid.size(); grid_idx++) {
    const std::vector<int>& cell = all_points_grid[grid_idx];
    
    for (int i = 0; i < cell.size(); i++) {
      int point_idx = cell[i];
      if (house_labels[point_idx] > 0) continue; // Skip already labeled houses
      
      int house_neighbors = 0;
      int total_neighbors = 0;
      
      // Check points in the same cell
      const arma::rowvec& point_i = points.row(point_idx);
      for (int j = 0; j < cell.size(); j++) {
        if (i == j) continue;
        
        int neighbor_idx = cell[j];
        const arma::rowvec& point_j = points.row(neighbor_idx);
        
        double dist_sq = arma::sum(arma::square(point_i - point_j));
        if (dist_sq < radius_sq) {
          total_neighbors++;
          if (house_labels[neighbor_idx] > 0) {
            house_neighbors++;
          }
        }
      }
      
      // Check points in adjacent cells
      int grid_x = grid_idx % grid_size;
      int grid_y = grid_idx / grid_size;
      
      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          if (dx == 0 && dy == 0) continue;
          
          int neighbor_x = grid_x + dx;
          int neighbor_y = grid_y + dy;
          int neighbor_grid_idx = neighbor_y * grid_size + neighbor_x;
          
          if (neighbor_x >= 0 && neighbor_x < grid_size &&
              neighbor_y >= 0 && neighbor_y < grid_size &&
              neighbor_grid_idx < all_points_grid.size()) {
            
            const std::vector<int>& neighbor_cell = all_points_grid[neighbor_grid_idx];
            for (int k = 0; k < neighbor_cell.size(); k++) {
              int neighbor_idx = neighbor_cell[k];
              const arma::rowvec& point_k = points.row(neighbor_idx);
              
              double dist_sq = arma::sum(arma::square(point_i - point_k));
              if (dist_sq < radius_sq) {
                total_neighbors++;
                if (house_labels[neighbor_idx] > 0) {
                  house_neighbors++;
                }
              }
            }
          }
        }
      }
      
      // If mostly surrounded by houses, find the most common cluster
      if (total_neighbors > 0 && static_cast<double>(house_neighbors) / total_neighbors > 0.7) {
        std::unordered_map<int, int> cluster_counts;
        
        // Count clusters in neighborhood (same cell)
        for (int j = 0; j < cell.size(); j++) {
          if (i == j) continue;
          
          int neighbor_idx = cell[j];
          const arma::rowvec& point_j = points.row(neighbor_idx);
          
          double dist_sq = arma::sum(arma::square(point_i - point_j));
          if (dist_sq < radius_sq && house_labels[neighbor_idx] > 0) {
            cluster_counts[house_labels[neighbor_idx]]++;
          }
        }
        
        // Count clusters in neighborhood (adjacent cells)
        for (int dx = -1; dx <= 1; dx++) {
          for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            
            int neighbor_x = grid_x + dx;
            int neighbor_y = grid_y + dy;
            int neighbor_grid_idx = neighbor_y * grid_size + neighbor_x;
            
            if (neighbor_x >= 0 && neighbor_x < grid_size &&
                neighbor_y >= 0 && neighbor_y < grid_size &&
                neighbor_grid_idx < all_points_grid.size()) {
              
              const std::vector<int>& neighbor_cell = all_points_grid[neighbor_grid_idx];
              for (int k = 0; k < neighbor_cell.size(); k++) {
                int neighbor_idx = neighbor_cell[k];
                const arma::rowvec& point_k = points.row(neighbor_idx);
                
                double dist_sq = arma::sum(arma::square(point_i - point_k));
                if (dist_sq < radius_sq && house_labels[neighbor_idx] > 0) {
                  cluster_counts[house_labels[neighbor_idx]]++;
                }
              }
            }
          }
        }
        
        // Assign to the most common cluster
        if (!cluster_counts.empty()) {
          int max_cluster = 0;
          int max_count = 0;
          for (const auto& pair : cluster_counts) {
            if (pair.second > max_count) {
              max_count = pair.second;
              max_cluster = pair.first;
            }
          }
          house_labels[point_idx] = max_cluster;
        }
      }
    }
  }
  
  return house_labels;
}