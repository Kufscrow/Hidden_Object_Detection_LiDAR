library(lidR)
library(rgl)
library(Rcpp)
library(RcppArmadillo)
library(cluster)
library(mclust)
library(RANN)
library(FNN)
library(utils)
library(Rvcg)

sourceCpp("C++ code/Ground_onesided_regression.cpp")
sourceCpp("C++ code/Gaussian_Gradient_Hessian.cpp")

# Read the lidar file without outliers
las <- readLAS("LiDAR Datasets/in2018_30600970_12.las", filter="-drop_withheld")
lasw <- cbind(las$X, las$Y, las$Z)
lasw_sample <- as.matrix(lasw)
lasw_intensity <- las$Intensity

# Assign label to colors
label_to_color_1 <- list(
  rgb(1, 0, 0),
  rgb(1, 1, 0),
  rgb(0, 1, 1),
  rgb(1, 0, 1),
  rgb(0, 1, 0),
  rgb(0, 0, 1),
  rgb(0, 0, 0),
  rgb(0, 0, 0.5),
  rgb(0, 0.5, 0),
  rgb(0.5, 0, 0),
  rgb(0.5, 0.5, 0),
  rgb(0.5, 0, 0.5),
  rgb(0, 0.5, 0.5),
  rgb(0.5, 0.5, 0.5),
  rgb(0, 0, 0.2),
  rgb(0, 0.2, 0),
  rgb(0.2, 0, 0),
  rgb(0.1, 0, 0)
)

plot3d(lasw, aspect=c(1, 1, 0.1), col=label_to_color_1[las$Classification], decorate=(box=FALSE))

k1 <- 10
k2 <- 10
a.x <- seq(min(las$X),max(las$X),(max(las$X)-min(las$X))/k1)
a.y <- seq(min(las$Y),max(las$Y),(max(las$Y)-min(las$Y))/k2)
region <- matrix(0,k1*k2,12)
region[,1] <- rep(a.x[-(k1+1)],rep(k2,k1))
region[,2] <- rep(a.y[-(k2+1)],k1)
region[,3] <- rep(a.x[-1],rep(k2,k1))
region[,4] <- rep(a.y[-1],k1)
region[,5] <- region[,1]-40
region[,6] <- region[,2]-40
region[,7] <- region[,3]+40
region[,8] <- region[,4]+40

i <- 71
indice_base <- which((lasw[, 1]>=region[i, 1]+360)&(lasw[, 2]>=region[i, 2])
                     &(lasw[, 1]<=region[i, 3]+360)&(lasw[, 2]<=region[i, 4]))
temp_2 <- lasw[indice_base, ]
intensity_2 <- lasw_intensity[indice_base]
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[las$Classification[indice_base]], decorate=(box=FALSE))

kernel_density_knn <- function(h, point_data_high, height_high, k=20){
  intensity <- compute_kde_knn(length(height_high), point_data_high, h, k)
  return(intensity)
}

kernel_density <- function(h, point_data_high, height_high){
  intensity <- compute_kde(length(height_high), point_data_high, h)
  return(intensity)
}

GMM_Cluster_4 <- function(Hessg, density_high, intensity_high, N){
  orig_data_high <- Hessg
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(label)
}

GMM_Cluster_4_1 <- function(Hessg, density_high, intensity_high, height_high, N){
  orig_data_high <- cbind(Hessg, height_high)
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(label)
}

GMM_Cluster_5 <- function(Hessg, density_high, intensity_high, N){
  orig_data_high <- cbind(Hessg, intensity_high)
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(label)
}

GMM_Cluster_5_1 <- function(Hessg, density_high, intensity_high, height_high, N){
  orig_data_high <- cbind(Hessg, height_high, intensity_high)
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(label)
}

label_fix_GMM_nonground_1 <- function(GMM_label, Hessg){
  if (length(GMM_label)!=length(Hessg)){
    stop("Dimension Mismatch")
  }
  label <- -2*as.numeric(GMM_label!=GMM_label[which.max(Hessg)])+3
  return(label)
}

label_fix_GMM_nonground_2 <- function(GMM_label, Hessg){
  if (length(GMM_label)!=length(Hessg)){
    stop("Dimension Mismatch")
  }
  label <- -2*as.numeric(GMM_label!=GMM_label[which.min(Hessg)])+3
  return(label)
}

label_fix_GMM_nonground_3 <- function(GMM_label, Hessg){
  if (length(GMM_label)!=length(Hessg)){
    stop("Dimension Mismatch")
  }
  label <- 2*as.numeric(GMM_label!=GMM_label[which.max(Hessg)])+1
  return(label)
}

label_fix_GMM_nonground_4 <- function(GMM_label, Hessg){
  if (length(GMM_label)!=length(Hessg)){
    stop("Dimension Mismatch")
  }
  label <- 2*as.numeric(GMM_label!=GMM_label[which.min(Hessg)])+1
  return(label)
}

joint_knn_prob <- function(coords, A, B, k = 20, tau = NULL, chunk_size = 5000) {
  stopifnot(nrow(coords) == length(A), length(A) == length(B))
  
  A_bin <- as.integer(A == 3)
  B_bin <- as.integer(B == 3)
  J_hard <- A_bin & B_bin
  N <- nrow(coords)
  p_joint <- numeric(N)
  idx_chunks <- split(1:N, ceiling(seq_along(1:N) / chunk_size))
  for (chunk in idx_chunks) {
    nn <- get.knnx(data = coords, query = coords[chunk, , drop = FALSE], k = k)
    p_joint[chunk] <- sapply(1:length(chunk), function(i) {
      nbr_idx <- nn$nn.index[i, ]
      mean(J_hard[nbr_idx])
    })
  }
  if (!is.null(tau)) {
    return(2*as.integer(p_joint >= tau)+1)
  }
  return(p_joint)
}

relabel_roof_edges_1 <- function(pts, is_center, metric, k = 100, radius = 2.0, metric_threshold = 0.7) {
  require(RANN)
  nn <- nn2(pts[,1:3], k=k, searchtype = "radius", radius = radius)
  is_edge <- rep(FALSE, length(pts[, 1]))
  progress <- txtProgressBar(min = 0, max = nrow(pts), style = 3)
  for (i in which(is_center)) {
    neighbors <- nn$nn.idx[i, nn$nn.idx[i,] > 0]
    metric_diff <- abs(metric[neighbors] - metric[i])
    edge_candidates <- neighbors[
      (metric_diff > metric_threshold) 
    ]
    is_edge[edge_candidates[!is_center[edge_candidates]]] <- TRUE
    
    setTxtProgressBar(progress, i)
  }
  close(progress)
  is_edge[is_center] <- FALSE
  return(is_edge)
}

relabel_houses_1 <- function(expand, pts, int_label, is_center, metric, k=100, radius=2.0, metric_threshold=0.7){
  if(expand==TRUE){
    relabel_edge <- relabel_roof_edges_1(pts, is_center, metric, k, radius, metric_threshold)
    relabeled_label <- 2*as.numeric(relabel_edge|(int_label==3))+1
  }
  else{
    relabeled_label <- int_label
  }
  return(relabeled_label)
}

ground_result_osr <- readRDS("Results/4/SouthernIndiana_WUI_region14_OSR.rds")
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[ground_result_osr], decorate=(box=FALSE))

nonground_patch_osr <- temp_2[which(ground_result_osr!=2), ]
intensity_nonground_patch_osr <- intensity_2[which(ground_result_osr!=2)]
height_nonground_patch_osr <- nonground_patch_osr[, 3]-min(nonground_patch_osr[, 3])

h2 <- c(15, 15, 30)
h_matrix_osr <- matrix(rep(h2, length(nonground_patch_osr[, 1])), length(nonground_patch_osr[, 1]), byrow = TRUE)

GMM_cluster_whole_osr_sectored_kernel_Hessg <- compute_hessi_eig_dif_4_sectored(length(height_nonground_patch_osr), nonground_patch_osr, h_matrix_osr, height_nonground_patch_osr)
GMM_cluster_whole_osr_sectored_kernel_Hessg[which(is.infinite(GMM_cluster_whole_osr_sectored_kernel_Hessg))] <- 200

GMM_cluster_whole_osr_kernel_Hessg <- compute_hessi_eig_dif(length(height_nonground_patch_osr), nonground_patch_osr, h_matrix_osr, height_nonground_patch_osr)
GMM_cluster_whole_osr_kernel_Hessg[which(is.infinite(GMM_cluster_whole_osr_kernel_Hessg))] <- 200

GMM_cluster_whole_osr_density <- kernel_density_knn(h_matrix_osr, nonground_patch_osr, height_nonground_patch_osr, k=20)

GMM_cluster_whole_osr_4 <- GMM_Cluster_4(GMM_cluster_whole_osr_sectored_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, 2)
GMM_cluster_whole_osr_4_k <- GMM_Cluster_4(GMM_cluster_whole_osr_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, 2)
GMM_cluster_whole_osr_4_1 <- GMM_Cluster_4_1(GMM_cluster_whole_osr_sectored_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, height_nonground_patch_osr, 2)
GMM_cluster_whole_osr_4_1_k <- GMM_Cluster_4_1(GMM_cluster_whole_osr_sectored_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, height_nonground_patch_osr, 2)

GMM_cluster_whole_osr_5 <- GMM_Cluster_5(GMM_cluster_whole_osr_sectored_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, 2)
GMM_cluster_whole_osr_5_k <- GMM_Cluster_5(GMM_cluster_whole_osr_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, 2)
GMM_cluster_whole_osr_5_1 <- GMM_Cluster_5_1(GMM_cluster_whole_osr_sectored_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, height_nonground_patch_osr, 2)
GMM_cluster_whole_osr_5_1_k <- GMM_Cluster_5_1(GMM_cluster_whole_osr_sectored_kernel_Hessg, GMM_cluster_whole_osr_density, intensity_nonground_patch_osr, height_nonground_patch_osr, 2)

result_nonground_osr_4_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4, GMM_cluster_whole_osr_sectored_kernel_Hessg)
result_nonground_osr_4_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4_k, GMM_cluster_whole_osr_kernel_Hessg)
result_nonground_osr_4_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4_1, GMM_cluster_whole_osr_sectored_kernel_Hessg)
result_nonground_osr_4_1_kw <- label_fix_GMM_nonground_3(GMM_cluster_whole_osr_4_1_k, GMM_cluster_whole_osr_kernel_Hessg)

result_nonground_osr_5_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_5, GMM_cluster_whole_osr_sectored_kernel_Hessg)
result_nonground_osr_5_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_5_k, GMM_cluster_whole_osr_kernel_Hessg)
result_nonground_osr_5_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_5_1, GMM_cluster_whole_osr_sectored_kernel_Hessg)
result_nonground_osr_5_1_kw <- label_fix_GMM_nonground_3(GMM_cluster_whole_osr_5_1_k, GMM_cluster_whole_osr_kernel_Hessg)

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_w], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_kw], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_1_kw], decorate=(box=FALSE))

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_w], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_kw], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_1_kw], decorate=(box=FALSE))

GMM_cluster_whole_osr_4_d <- Mclust(GMM_cluster_whole_osr_density, G=3)$classification

result_nonground_osr_4_d <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4_d, GMM_cluster_whole_osr_density)

result_nonground_osr_4_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_4_w, result_nonground_osr_4_d, k=20, tau=0.5)
result_nonground_osr_4_k_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_4_kw, result_nonground_osr_4_d, k=20, tau=0.5)
result_nonground_osr_4_1_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_4_1_w, result_nonground_osr_4_d, k=20, tau=0.5)
result_nonground_osr_4_1_k_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_4_1_kw, result_nonground_osr_4_d, k=20, tau=0.5)

result_nonground_osr_5_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_5_w, result_nonground_osr_4_d, k=20, tau=0.5)
result_nonground_osr_5_k_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_5_kw, result_nonground_osr_4_d, k=20, tau=0.5)
result_nonground_osr_5_1_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_5_1_w, result_nonground_osr_4_d, k=20, tau=0.5)
result_nonground_osr_5_1_k_int <- joint_knn_prob(nonground_patch_osr, result_nonground_osr_5_1_kw, result_nonground_osr_4_d, k=20, tau=0.5)

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_int], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_1_k_int], decorate=(box=FALSE))

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_int], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_1_k_int], decorate=(box=FALSE))

result_nonground_osr_4 <- relabel_houses_1(TRUE, nonground_patch_osr, result_nonground_osr_4_int, 
                                           (result_nonground_osr_4_int==3), GMM_cluster_whole_osr_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_osr_4_k <- relabel_houses_1(TRUE, nonground_patch_osr, result_nonground_osr_4_k_int, 
                                             (result_nonground_osr_4_k_int==3), GMM_cluster_whole_osr_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_osr_4_1 <- relabel_houses_1(FALSE, nonground_patch_osr, result_nonground_osr_4_1_int, 
                                             (result_nonground_osr_4_1_int==3), GMM_cluster_whole_osr_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_osr_4_1_k <- relabel_houses_1(FALSE, nonground_patch_osr, result_nonground_osr_4_1_k_int, 
                                               (result_nonground_osr_4_1_k_int==3), GMM_cluster_whole_osr_sectored_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)

result_nonground_osr_5 <- relabel_houses_1(FALSE, nonground_patch_osr, result_nonground_osr_5_int, 
                                           (result_nonground_osr_5_int==3), GMM_cluster_whole_osr_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_osr_5_k <- relabel_houses_1(FALSE, nonground_patch_osr, result_nonground_osr_5_k_int, 
                                             (result_nonground_osr_5_k_int==3), GMM_cluster_whole_osr_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_osr_5_1 <- relabel_houses_1(FALSE, nonground_patch_osr, result_nonground_osr_5_1_int, 
                                             (result_nonground_osr_5_1_int==3), GMM_cluster_whole_osr_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_osr_5_1_k <- relabel_houses_1(FALSE, nonground_patch_osr, result_nonground_osr_5_1_k_int, 
                                               (result_nonground_osr_5_1_k_int==3), GMM_cluster_whole_osr_sectored_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_k], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_4_1_k], decorate=(box=FALSE))

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_k], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_osr_5_1_k], decorate=(box=FALSE))

result_osr_4 <- rep(2, length(temp_2[, 1]))
result_osr_4_k <- rep(2, length(temp_2[, 1]))
result_osr_4_1 <- rep(2, length(temp_2[, 1]))
result_osr_4_1_k <- rep(2, length(temp_2[, 1]))
result_osr_5 <- rep(2, length(temp_2[, 1]))
result_osr_5_k <- rep(2, length(temp_2[, 1]))
result_osr_5_1 <- rep(2, length(temp_2[, 1]))
result_osr_5_1_k <- rep(2, length(temp_2[, 1]))

result_osr_4[which(ground_result_osr!=2)] <- result_nonground_osr_4
result_osr_4_k[which(ground_result_osr!=2)] <- result_nonground_osr_4_k
result_osr_4_1[which(ground_result_osr!=2)] <- result_nonground_osr_4_1
result_osr_4_1_k[which(ground_result_osr!=2)] <- result_nonground_osr_4_1_k
result_osr_5[which(ground_result_osr!=2)] <- result_nonground_osr_5
result_osr_5_k[which(ground_result_osr!=2)] <- result_nonground_osr_5_k
result_osr_5_1[which(ground_result_osr!=2)] <- result_nonground_osr_5_1
result_osr_5_1_k[which(ground_result_osr!=2)] <- result_nonground_osr_5_1_k

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4_1_k], decorate=(box=FALSE))

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5_1_k], decorate=(box=FALSE))

text4.17 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel.rds"
text5.17 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel.rds"
text4.18 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel_height.rds"
text5.18 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel_height.rds"
text4.19 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel_intensity.rds"
text5.19 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel_intensity.rds"
text4.20 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel_height_intensity.rds"
text5.20 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel_height_intensity.rds"

saveRDS(result_osr_4, text4.17)
saveRDS(result_osr_4_k, text5.17)
saveRDS(result_osr_4_1, text4.18)
saveRDS(result_osr_4_1_k, text5.18)
saveRDS(result_osr_5, text4.19)
saveRDS(result_osr_5_k, text5.19)
saveRDS(result_osr_5_1, text4.20)
saveRDS(result_osr_5_1_k, text5.20)
