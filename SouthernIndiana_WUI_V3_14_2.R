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
library(dplyr)

sourceCpp("C++ code/Ground_local_hyperbolic_patches.cpp")
sourceCpp("C++ code/Ground_local_quadratic_functions.cpp")
sourceCpp("C++ code/Gaussian_Gradient_Hessian.cpp")
sourceCpp("C++ code/House_detection.cpp")

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

hh_cand <- list(c(1,1), c(1,2), c(1,4), c(1,8))
adjust <- 2.0 
Size <- 50
More <- 20
outlier <- 1

i <- 71
indice_base <- which((lasw[, 1]>=region[i, 1]+360)&(lasw[, 2]>=region[i, 2])
                     &(lasw[, 1]<=region[i, 3]+360)&(lasw[, 2]<=region[i, 4]))
temp_2 <- lasw[indice_base, ]
intensity_2 <- lasw_intensity[indice_base]
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[las$Classification[indice_base]], decorate=(box=FALSE))

df_temp_2 <- data.frame(temp_2, las$Classification[indice_base])
names(df_temp_2) <- c("X", "Y", "Z", "Classification")
las.patch.base <- LAS(df_temp_2)

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

las_patch_csf <- classify_ground(las.patch.base, algorithm = csf(sloop_smooth = TRUE, class_threshold = 1, cloth_resolution = 1, time_step = 1))
ng_patch_csf <- filter_poi(las_patch_csf, Classification != 2L)
plot(las_patch_csf,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

las_patch_pmf <- classify_ground(las.patch.base, algorithm = pmf(ws = 5, th = 3))
ng_patch_pmf <- filter_poi(las_patch_pmf, Classification != 2L)
plot(las_patch_pmf,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

las_patch_mcc <- classify_ground(las.patch.base, algorithm = mcc(1.5,0.3))
ng_patch_mcc <- filter_poi(las_patch_mcc, Classification != 2L)
plot(las_patch_mcc,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

nonground_patch_csf <- temp_2[which(las_patch_csf$Classification!=2), ]
nonground_patch_pmf <- temp_2[which(las_patch_pmf$Classification!=2), ]
nonground_patch_mcc <- temp_2[which(las_patch_mcc$Classification!=2), ]
nonground_patch_ori <- temp_2[which(las.patch.base$Classification!=2), ]
intensity_nonground_patch_csf <- intensity_2[which(las_patch_csf$Classification!=2)]
intensity_nonground_patch_pmf <- intensity_2[which(las_patch_pmf$Classification!=2)]
intensity_nonground_patch_mcc <- intensity_2[which(las_patch_mcc$Classification!=2)]
intensity_nonground_patch_ori <- intensity_2[which(las.patch.base$Classification!=2)]
height_nonground_patch_csf <- nonground_patch_csf[, 3]-min(nonground_patch_csf[, 3])
height_nonground_patch_pmf <- nonground_patch_pmf[, 3]-min(nonground_patch_pmf[, 3])
height_nonground_patch_mcc <- nonground_patch_mcc[, 3]-min(nonground_patch_mcc[, 3])
height_nonground_patch_ori <- nonground_patch_ori[, 3]-min(nonground_patch_ori[, 3])

h2 <- c(15, 15, 30)
h_matrix_csf <- matrix(rep(h2, length(nonground_patch_csf[, 1])), length(nonground_patch_csf[, 1]), byrow = TRUE)
h_matrix_pmf <- matrix(rep(h2, length(nonground_patch_pmf[, 1])), length(nonground_patch_pmf[, 1]), byrow = TRUE)
h_matrix_mcc <- matrix(rep(h2, length(nonground_patch_mcc[, 1])), length(nonground_patch_mcc[, 1]), byrow = TRUE)
h_matrix_ori <- matrix(rep(h2, length(nonground_patch_ori[, 1])), length(nonground_patch_ori[, 1]), byrow = TRUE)

GMM_cluster_whole_csf_sectored_kernel_Hessg <- compute_hessi_eig_dif_4_sectored(length(height_nonground_patch_csf), nonground_patch_csf, h_matrix_csf, height_nonground_patch_csf)
GMM_cluster_whole_pmf_sectored_kernel_Hessg <- compute_hessi_eig_dif_4_sectored(length(height_nonground_patch_pmf), nonground_patch_pmf, h_matrix_pmf, height_nonground_patch_pmf)
GMM_cluster_whole_mcc_sectored_kernel_Hessg <- compute_hessi_eig_dif_4_sectored(length(height_nonground_patch_mcc), nonground_patch_mcc, h_matrix_mcc, height_nonground_patch_mcc)
GMM_cluster_whole_ori_sectored_kernel_Hessg <- compute_hessi_eig_dif_4_sectored(length(height_nonground_patch_ori), nonground_patch_ori, h_matrix_ori, height_nonground_patch_ori)
GMM_cluster_whole_csf_sectored_kernel_Hessg[which(is.infinite(GMM_cluster_whole_csf_sectored_kernel_Hessg))] <- 200
GMM_cluster_whole_pmf_sectored_kernel_Hessg[which(is.infinite(GMM_cluster_whole_pmf_sectored_kernel_Hessg))] <- 200
GMM_cluster_whole_mcc_sectored_kernel_Hessg[which(is.infinite(GMM_cluster_whole_mcc_sectored_kernel_Hessg))] <- 200
GMM_cluster_whole_ori_sectored_kernel_Hessg[which(is.infinite(GMM_cluster_whole_ori_sectored_kernel_Hessg))] <- 200

GMM_cluster_whole_csf_kernel_Hessg <- compute_hessi_eig_dif(length(height_nonground_patch_csf), nonground_patch_csf, h_matrix_csf, height_nonground_patch_csf)
GMM_cluster_whole_pmf_kernel_Hessg <- compute_hessi_eig_dif(length(height_nonground_patch_pmf), nonground_patch_pmf, h_matrix_pmf, height_nonground_patch_pmf)
GMM_cluster_whole_mcc_kernel_Hessg <- compute_hessi_eig_dif(length(height_nonground_patch_mcc), nonground_patch_mcc, h_matrix_mcc, height_nonground_patch_mcc)
GMM_cluster_whole_ori_kernel_Hessg <- compute_hessi_eig_dif(length(height_nonground_patch_ori), nonground_patch_ori, h_matrix_ori, height_nonground_patch_ori)
GMM_cluster_whole_csf_kernel_Hessg[which(is.infinite(GMM_cluster_whole_csf_kernel_Hessg))] <- 200
GMM_cluster_whole_pmf_kernel_Hessg[which(is.infinite(GMM_cluster_whole_pmf_kernel_Hessg))] <- 200
GMM_cluster_whole_mcc_kernel_Hessg[which(is.infinite(GMM_cluster_whole_mcc_kernel_Hessg))] <- 200
GMM_cluster_whole_ori_kernel_Hessg[which(is.infinite(GMM_cluster_whole_ori_kernel_Hessg))] <- 200

GMM_cluster_whole_csf_density <- kernel_density_knn(h_matrix_csf, nonground_patch_csf, height_nonground_patch_csf, k=20)
GMM_cluster_whole_pmf_density <- kernel_density_knn(h_matrix_pmf, nonground_patch_pmf, height_nonground_patch_pmf, k=20)
GMM_cluster_whole_mcc_density <- kernel_density_knn(h_matrix_mcc, nonground_patch_mcc, height_nonground_patch_mcc, k=20)
GMM_cluster_whole_ori_density <- kernel_density_knn(h_matrix_ori, nonground_patch_ori, height_nonground_patch_ori, k=20)

GMM_cluster_whole_csf_4 <- GMM_Cluster_4(GMM_cluster_whole_csf_sectored_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4 <- GMM_Cluster_4(GMM_cluster_whole_pmf_sectored_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4 <- GMM_Cluster_4(GMM_cluster_whole_mcc_sectored_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4 <- GMM_Cluster_4(GMM_cluster_whole_ori_sectored_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, 2)
GMM_cluster_whole_csf_4_k <- GMM_Cluster_4(GMM_cluster_whole_csf_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4_k <- GMM_Cluster_4(GMM_cluster_whole_pmf_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4_k <- GMM_Cluster_4(GMM_cluster_whole_mcc_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4_k <- GMM_Cluster_4(GMM_cluster_whole_ori_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, 2)
GMM_cluster_whole_csf_4_1 <- GMM_Cluster_4_1(GMM_cluster_whole_csf_sectored_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4_1 <- GMM_Cluster_4_1(GMM_cluster_whole_pmf_sectored_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4_1 <- GMM_Cluster_4_1(GMM_cluster_whole_mcc_sectored_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4_1 <- GMM_Cluster_4_1(GMM_cluster_whole_ori_sectored_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, height_nonground_patch_ori, 2)
GMM_cluster_whole_csf_4_1_k <- GMM_Cluster_4_1(GMM_cluster_whole_csf_sectored_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4_1_k <- GMM_Cluster_4_1(GMM_cluster_whole_pmf_sectored_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4_1_k <- GMM_Cluster_4_1(GMM_cluster_whole_mcc_sectored_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4_1_k <- GMM_Cluster_4_1(GMM_cluster_whole_ori_sectored_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, height_nonground_patch_ori, 2)

GMM_cluster_whole_csf_5 <- GMM_Cluster_5(GMM_cluster_whole_csf_sectored_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5 <- GMM_Cluster_5(GMM_cluster_whole_pmf_sectored_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5 <- GMM_Cluster_5(GMM_cluster_whole_mcc_sectored_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5 <- GMM_Cluster_5(GMM_cluster_whole_ori_sectored_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, 2)
GMM_cluster_whole_csf_5_k <- GMM_Cluster_5(GMM_cluster_whole_csf_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5_k <- GMM_Cluster_5(GMM_cluster_whole_pmf_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5_k <- GMM_Cluster_5(GMM_cluster_whole_mcc_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5_k <- GMM_Cluster_5(GMM_cluster_whole_ori_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, 2)
GMM_cluster_whole_csf_5_1 <- GMM_Cluster_5_1(GMM_cluster_whole_csf_sectored_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5_1 <- GMM_Cluster_5_1(GMM_cluster_whole_pmf_sectored_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5_1 <- GMM_Cluster_5_1(GMM_cluster_whole_mcc_sectored_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5_1 <- GMM_Cluster_5_1(GMM_cluster_whole_ori_sectored_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, height_nonground_patch_ori, 2)
GMM_cluster_whole_csf_5_1_k <- GMM_Cluster_5_1(GMM_cluster_whole_csf_sectored_kernel_Hessg, GMM_cluster_whole_csf_density, intensity_nonground_patch_csf, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5_1_k <- GMM_Cluster_5_1(GMM_cluster_whole_pmf_sectored_kernel_Hessg, GMM_cluster_whole_pmf_density, intensity_nonground_patch_pmf, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5_1_k <- GMM_Cluster_5_1(GMM_cluster_whole_mcc_sectored_kernel_Hessg, GMM_cluster_whole_mcc_density, intensity_nonground_patch_mcc, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5_1_k <- GMM_Cluster_5_1(GMM_cluster_whole_ori_sectored_kernel_Hessg, GMM_cluster_whole_ori_density, intensity_nonground_patch_ori, height_nonground_patch_ori, 2)

result_nonground_csf_4_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4, GMM_cluster_whole_csf_sectored_kernel_Hessg)
result_nonground_pmf_4_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4, GMM_cluster_whole_pmf_sectored_kernel_Hessg)
result_nonground_mcc_4_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_4, GMM_cluster_whole_mcc_sectored_kernel_Hessg)
result_nonground_ori_4_w <- label_fix_GMM_nonground_4(GMM_cluster_whole_ori_4, GMM_cluster_whole_ori_sectored_kernel_Hessg)
result_nonground_csf_4_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4_k, GMM_cluster_whole_csf_kernel_Hessg)
result_nonground_pmf_4_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4_k, GMM_cluster_whole_pmf_kernel_Hessg)
result_nonground_mcc_4_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_4_k, GMM_cluster_whole_mcc_kernel_Hessg)
result_nonground_ori_4_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_4_k, GMM_cluster_whole_ori_kernel_Hessg)
result_nonground_csf_4_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4_1, GMM_cluster_whole_csf_sectored_kernel_Hessg)
result_nonground_pmf_4_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4_1, GMM_cluster_whole_pmf_sectored_kernel_Hessg)
result_nonground_mcc_4_1_w <- label_fix_GMM_nonground_4(GMM_cluster_whole_mcc_4_1, GMM_cluster_whole_mcc_sectored_kernel_Hessg)
result_nonground_ori_4_1_w <- label_fix_GMM_nonground_4(GMM_cluster_whole_ori_4_1, GMM_cluster_whole_ori_sectored_kernel_Hessg)
result_nonground_csf_4_1_kw <- label_fix_GMM_nonground_2(GMM_cluster_whole_csf_4_1_k, GMM_cluster_whole_csf_kernel_Hessg)
result_nonground_pmf_4_1_kw <- label_fix_GMM_nonground_2(GMM_cluster_whole_pmf_4_1_k, GMM_cluster_whole_pmf_kernel_Hessg)
result_nonground_mcc_4_1_kw <- label_fix_GMM_nonground_4(GMM_cluster_whole_mcc_4_1_k, GMM_cluster_whole_mcc_kernel_Hessg)
result_nonground_ori_4_1_kw <- label_fix_GMM_nonground_4(GMM_cluster_whole_ori_4_1_k, GMM_cluster_whole_ori_kernel_Hessg)

result_nonground_csf_5_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_5, GMM_cluster_whole_csf_sectored_kernel_Hessg)
result_nonground_pmf_5_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_5, GMM_cluster_whole_pmf_sectored_kernel_Hessg)
result_nonground_mcc_5_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_5, GMM_cluster_whole_mcc_sectored_kernel_Hessg)
result_nonground_ori_5_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_5, GMM_cluster_whole_ori_sectored_kernel_Hessg)
result_nonground_csf_5_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_5_k, GMM_cluster_whole_csf_kernel_Hessg)
result_nonground_pmf_5_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_5_k, GMM_cluster_whole_pmf_kernel_Hessg)
result_nonground_mcc_5_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_5_k, GMM_cluster_whole_mcc_kernel_Hessg)
result_nonground_ori_5_kw <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_5_k, GMM_cluster_whole_ori_kernel_Hessg)
result_nonground_csf_5_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_5_1, GMM_cluster_whole_csf_sectored_kernel_Hessg)
result_nonground_pmf_5_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_5_1, GMM_cluster_whole_pmf_sectored_kernel_Hessg)
result_nonground_mcc_5_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_5_1, GMM_cluster_whole_mcc_sectored_kernel_Hessg)
result_nonground_ori_5_1_w <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_5_1, GMM_cluster_whole_ori_sectored_kernel_Hessg)
result_nonground_csf_5_1_kw <- label_fix_GMM_nonground_3(GMM_cluster_whole_csf_5_1_k, GMM_cluster_whole_csf_kernel_Hessg)
result_nonground_pmf_5_1_kw <- label_fix_GMM_nonground_4(GMM_cluster_whole_pmf_5_1_k, GMM_cluster_whole_pmf_kernel_Hessg)
result_nonground_mcc_5_1_kw <- label_fix_GMM_nonground_3(GMM_cluster_whole_mcc_5_1_k, GMM_cluster_whole_mcc_kernel_Hessg)
result_nonground_ori_5_1_kw <- label_fix_GMM_nonground_4(GMM_cluster_whole_ori_5_1_k, GMM_cluster_whole_ori_kernel_Hessg)

plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_w], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_w], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_w], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_w], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_kw], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_kw], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_kw], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_kw], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_1_kw], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_1_kw], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_1_kw], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_1_kw], decorate=(box=FALSE))

plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_w], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_w], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_w], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_w], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_kw], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_kw], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_kw], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_kw], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_1_w], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_1_kw], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_1_kw], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_1_kw], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_1_kw], decorate=(box=FALSE))

GMM_cluster_whole_csf_4_d <- Mclust(GMM_cluster_whole_csf_density, G=3)$classification
GMM_cluster_whole_pmf_4_d <- Mclust(GMM_cluster_whole_pmf_density, G=3)$classification
GMM_cluster_whole_mcc_4_d <- Mclust(GMM_cluster_whole_mcc_density, G=3)$classification
GMM_cluster_whole_ori_4_d <- Mclust(GMM_cluster_whole_ori_density, G=3)$classification

result_nonground_csf_4_d <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4_d, GMM_cluster_whole_csf_density)
result_nonground_pmf_4_d <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4_d, GMM_cluster_whole_pmf_density)
result_nonground_mcc_4_d <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_4_d, GMM_cluster_whole_mcc_density)
result_nonground_ori_4_d <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_4_d, GMM_cluster_whole_ori_density)

result_nonground_csf_4_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_4_w, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_4_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_4_w, result_nonground_pmf_4_d, k=20, tau=0.5)
result_nonground_mcc_4_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_4_w, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_4_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_4_w, result_nonground_ori_4_d, k=20, tau=0.5)
result_nonground_csf_4_k_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_4_kw, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_4_k_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_4_kw, result_nonground_pmf_4_d, k=20, tau=0.5)
result_nonground_mcc_4_k_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_4_kw, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_4_k_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_4_kw, result_nonground_ori_4_d, k=20, tau=0.5)
result_nonground_csf_4_1_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_4_1_w, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_4_1_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_4_1_w, result_nonground_pmf_4_d, k=20, tau=0.5)
result_nonground_mcc_4_1_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_4_1_w, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_4_1_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_4_1_w, result_nonground_ori_4_d, k=20, tau=0.5)
result_nonground_csf_4_1_k_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_4_1_kw, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_4_1_k_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_4_1_kw, result_nonground_pmf_4_d, k=20, tau=0.6)
result_nonground_mcc_4_1_k_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_4_1_kw, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_4_1_k_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_4_1_kw, result_nonground_ori_4_d, k=20, tau=0.5)

result_nonground_csf_5_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_5_w, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_5_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_5_w, result_nonground_pmf_4_d, k=20, tau=0.5)
result_nonground_mcc_5_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_5_w, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_5_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_5_w, result_nonground_ori_4_d, k=20, tau=0.5)
result_nonground_csf_5_k_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_5_kw, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_5_k_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_5_kw, result_nonground_pmf_4_d, k=20, tau=0.5)
result_nonground_mcc_5_k_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_5_kw, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_5_k_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_5_kw, result_nonground_ori_4_d, k=20, tau=0.5)
result_nonground_csf_5_1_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_5_1_w, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_5_1_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_5_1_w, result_nonground_pmf_4_d, k=20, tau=0.5)
result_nonground_mcc_5_1_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_5_1_w, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_5_1_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_5_1_w, result_nonground_ori_4_d, k=20, tau=0.5)
result_nonground_csf_5_1_k_int <- joint_knn_prob(nonground_patch_csf, result_nonground_csf_5_1_kw, result_nonground_csf_4_d, k=20, tau=0.5)
result_nonground_pmf_5_1_k_int <- joint_knn_prob(nonground_patch_pmf, result_nonground_pmf_5_1_kw, result_nonground_pmf_4_d, k=20, tau=0.5)
result_nonground_mcc_5_1_k_int <- joint_knn_prob(nonground_patch_mcc, result_nonground_mcc_5_1_kw, result_nonground_mcc_4_d, k=20, tau=0.5)
result_nonground_ori_5_1_k_int <- joint_knn_prob(nonground_patch_ori, result_nonground_ori_5_1_kw, result_nonground_ori_4_d, k=20, tau=0.5)

plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_int], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_1_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_1_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_1_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_1_k_int], decorate=(box=FALSE))

plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_int], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_1_int], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_1_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_1_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_1_k_int], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_1_k_int], decorate=(box=FALSE))

result_nonground_csf_4 <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_4_int, 
                                           (result_nonground_csf_4_int==3), GMM_cluster_whole_csf_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_4 <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_4_int, 
                                           (result_nonground_pmf_4_int==3), GMM_cluster_whole_pmf_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_4 <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_4_int, 
                                           (result_nonground_mcc_4_int==3), GMM_cluster_whole_mcc_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_4 <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_4_int, 
                                           (result_nonground_ori_4_int==3), GMM_cluster_whole_ori_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_csf_4_k <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_4_k_int, 
                                             (result_nonground_csf_4_k_int==3), GMM_cluster_whole_csf_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_4_k <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_4_k_int, 
                                             (result_nonground_pmf_4_k_int==3), GMM_cluster_whole_pmf_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_4_k <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_4_k_int, 
                                             (result_nonground_mcc_4_k_int==3), GMM_cluster_whole_mcc_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_4_k <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_4_k_int, 
                                             (result_nonground_ori_4_k_int==3), GMM_cluster_whole_ori_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_csf_4_1 <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_4_1_int, 
                                             (result_nonground_csf_4_1_int==3), GMM_cluster_whole_csf_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_4_1 <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_4_1_int, 
                                             (result_nonground_pmf_4_1_int==3), GMM_cluster_whole_pmf_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_4_1 <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_4_1_int, 
                                             (result_nonground_mcc_4_1_int==3), GMM_cluster_whole_mcc_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_4_1 <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_4_1_int, 
                                             (result_nonground_ori_4_1_int==3), GMM_cluster_whole_ori_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_csf_4_1_k <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_4_1_k_int, 
                                               (result_nonground_csf_4_1_k_int==3), GMM_cluster_whole_csf_sectored_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_4_1_k <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_4_1_k_int, 
                                               (result_nonground_pmf_4_1_k_int==3), GMM_cluster_whole_pmf_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_4_1_k <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_4_1_k_int, 
                                               (result_nonground_mcc_4_1_k_int==3), GMM_cluster_whole_mcc_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_4_1_k <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_4_1_k_int, 
                                               (result_nonground_ori_4_1_k_int==3), GMM_cluster_whole_ori_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)

result_nonground_csf_5 <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_5_int, 
                                           (result_nonground_csf_5_int==3), GMM_cluster_whole_csf_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_5 <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_5_int, 
                                           (result_nonground_pmf_5_int==3), GMM_cluster_whole_pmf_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_5 <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_5_int, 
                                           (result_nonground_mcc_5_int==3), GMM_cluster_whole_mcc_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_5 <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_5_int, 
                                           (result_nonground_ori_5_int==3), GMM_cluster_whole_ori_sectored_kernel_Hessg, 
                                           k=200, radius=10, metric_threshold=0.7)
result_nonground_csf_5_k <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_5_k_int, 
                                             (result_nonground_csf_5_k_int==3), GMM_cluster_whole_csf_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_5_k <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_5_k_int, 
                                             (result_nonground_pmf_5_k_int==3), GMM_cluster_whole_pmf_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_5_k <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_5_k_int, 
                                             (result_nonground_mcc_5_k_int==3), GMM_cluster_whole_mcc_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_5_k <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_5_k_int, 
                                             (result_nonground_ori_5_k_int==3), GMM_cluster_whole_ori_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_csf_5_1 <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_5_1_int, 
                                             (result_nonground_csf_5_1_int==3), GMM_cluster_whole_csf_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_5_1 <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_5_1_int, 
                                             (result_nonground_pmf_5_1_int==3), GMM_cluster_whole_pmf_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_5_1 <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_5_1_int, 
                                             (result_nonground_mcc_5_1_int==3), GMM_cluster_whole_mcc_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_5_1 <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_5_1_int, 
                                             (result_nonground_ori_5_1_int==3), GMM_cluster_whole_ori_sectored_kernel_Hessg, 
                                             k=200, radius=10, metric_threshold=0.7)
result_nonground_csf_5_1_k <- relabel_houses_1(FALSE, nonground_patch_csf, result_nonground_csf_5_1_k_int, 
                                               (result_nonground_csf_5_1_k_int==3), GMM_cluster_whole_csf_sectored_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)
result_nonground_pmf_5_1_k <- relabel_houses_1(FALSE, nonground_patch_pmf, result_nonground_pmf_5_1_k_int, 
                                               (result_nonground_pmf_5_1_k_int==3), GMM_cluster_whole_pmf_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)
result_nonground_mcc_5_1_k <- relabel_houses_1(FALSE, nonground_patch_mcc, result_nonground_mcc_5_1_k_int, 
                                               (result_nonground_mcc_5_1_k_int==3), GMM_cluster_whole_mcc_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)
result_nonground_ori_5_1_k <- relabel_houses_1(FALSE, nonground_patch_ori, result_nonground_ori_5_1_k_int, 
                                               (result_nonground_ori_5_1_k_int==3), GMM_cluster_whole_ori_kernel_Hessg, 
                                               k=200, radius=10, metric_threshold=0.7)

plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_k], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_k], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_k], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_k], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_4_1_k], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_4_1_k], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_4_1_k], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_4_1_k], decorate=(box=FALSE))

plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_k], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_k], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_k], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_k], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_csf_5_1_k], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_pmf_5_1_k], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_mcc_5_1_k], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[result_nonground_ori_5_1_k], decorate=(box=FALSE))

result_csf_4 <- rep(2, length(temp_2[, 1]))
result_pmf_4 <- rep(2, length(temp_2[, 1]))
result_mcc_4 <- rep(2, length(temp_2[, 1]))
result_ori_4 <- rep(2, length(temp_2[, 1]))
result_csf_4_k <- rep(2, length(temp_2[, 1]))
result_pmf_4_k <- rep(2, length(temp_2[, 1]))
result_mcc_4_k <- rep(2, length(temp_2[, 1]))
result_ori_4_k <- rep(2, length(temp_2[, 1]))
result_csf_4_1 <- rep(2, length(temp_2[, 1]))
result_pmf_4_1 <- rep(2, length(temp_2[, 1]))
result_mcc_4_1 <- rep(2, length(temp_2[, 1]))
result_ori_4_1 <- rep(2, length(temp_2[, 1]))
result_csf_4_1_k <- rep(2, length(temp_2[, 1]))
result_pmf_4_1_k <- rep(2, length(temp_2[, 1]))
result_mcc_4_1_k <- rep(2, length(temp_2[, 1]))
result_ori_4_1_k <- rep(2, length(temp_2[, 1]))
result_csf_5 <- rep(2, length(temp_2[, 1]))
result_pmf_5 <- rep(2, length(temp_2[, 1]))
result_mcc_5 <- rep(2, length(temp_2[, 1]))
result_ori_5 <- rep(2, length(temp_2[, 1]))
result_csf_5_k <- rep(2, length(temp_2[, 1]))
result_pmf_5_k <- rep(2, length(temp_2[, 1]))
result_mcc_5_k <- rep(2, length(temp_2[, 1]))
result_ori_5_k <- rep(2, length(temp_2[, 1]))
result_csf_5_1 <- rep(2, length(temp_2[, 1]))
result_pmf_5_1 <- rep(2, length(temp_2[, 1]))
result_mcc_5_1 <- rep(2, length(temp_2[, 1]))
result_ori_5_1 <- rep(2, length(temp_2[, 1]))
result_csf_5_1_k <- rep(2, length(temp_2[, 1]))
result_pmf_5_1_k <- rep(2, length(temp_2[, 1]))
result_mcc_5_1_k <- rep(2, length(temp_2[, 1]))
result_ori_5_1_k <- rep(2, length(temp_2[, 1]))

result_csf_4[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_4
result_pmf_4[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_4
result_mcc_4[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_4
result_ori_4[which(las.patch.base$Classification!=2)] <- result_nonground_ori_4
result_csf_4_k[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_4_k
result_pmf_4_k[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_4_k
result_mcc_4_k[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_4_k
result_ori_4_k[which(las.patch.base$Classification!=2)] <- result_nonground_ori_4_k
result_csf_4_1[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_4_1
result_pmf_4_1[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_4_1
result_mcc_4_1[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_4_1
result_ori_4_1[which(las.patch.base$Classification!=2)] <- result_nonground_ori_4_1
result_csf_4_1_k[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_4_1_k
result_pmf_4_1_k[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_4_1_k
result_mcc_4_1_k[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_4_1_k
result_ori_4_1_k[which(las.patch.base$Classification!=2)] <- result_nonground_ori_4_1_k
result_csf_5[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_5
result_pmf_5[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_5
result_mcc_5[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_5
result_ori_5[which(las.patch.base$Classification!=2)] <- result_nonground_ori_5
result_csf_5_k[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_5_k
result_pmf_5_k[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_5_k
result_mcc_5_k[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_5_k
result_ori_5_k[which(las.patch.base$Classification!=2)] <- result_nonground_ori_5_k
result_csf_5_1[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_5_1
result_pmf_5_1[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_5_1
result_mcc_5_1[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_5_1
result_ori_5_1[which(las.patch.base$Classification!=2)] <- result_nonground_ori_5_1
result_csf_5_1_k[which(las_patch_csf$Classification!=2)] <- result_nonground_csf_5_1_k
result_pmf_5_1_k[which(las_patch_pmf$Classification!=2)] <- result_nonground_pmf_5_1_k
result_mcc_5_1_k[which(las_patch_mcc$Classification!=2)] <- result_nonground_mcc_5_1_k
result_ori_5_1_k[which(las.patch.base$Classification!=2)] <- result_nonground_ori_5_1_k

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4_1_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4_1_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4_1_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4_1_k], decorate=(box=FALSE))

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5_1_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5_1_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5_1_k], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5_1_k], decorate=(box=FALSE))

text4.1 <- "Results/4/SouthernIndiana_WUI_region14_CSF_sectored_kernel.rds"
text4.2 <- "Results/4/SouthernIndiana_WUI_region14_PMF_sectored_kernel.rds"
text4.3 <- "Results/4/SouthernIndiana_WUI_region14_MCC_sectored_kernel.rds"
text4.4 <- "Results/4/SouthernIndiana_WUI_region14_ori_sectored_kernel.rds"

text5.1 <- "Results/4/SouthernIndiana_WUI_region14_CSF_kernel.rds"
text5.2 <- "Results/4/SouthernIndiana_WUI_region14_PMF_kernel.rds"
text5.3 <- "Results/4/SouthernIndiana_WUI_region14_MCC_kernel.rds"
text5.4 <- "Results/4/SouthernIndiana_WUI_region14_ori_kernel.rds"

text4.5 <- "Results/4/SouthernIndiana_WUI_region14_CSF_sectored_kernel_height.rds"
text4.6 <- "Results/4/SouthernIndiana_WUI_region14_PMF_sectored_kernel_height.rds"
text4.7 <- "Results/4/SouthernIndiana_WUI_region14_MCC_sectored_kernel_height.rds"
text4.8 <- "Results/4/SouthernIndiana_WUI_region14_ori_sectored_kernel_height.rds"

text5.5 <- "Results/4/SouthernIndiana_WUI_region14_CSF_kernel_height.rds"
text5.6 <- "Results/4/SouthernIndiana_WUI_region14_PMF_kernel_height.rds"
text5.7 <- "Results/4/SouthernIndiana_WUI_region14_MCC_kernel_height.rds"
text5.8 <- "Results/4/SouthernIndiana_WUI_region14_ori_kernel_height.rds"

text4.9 <- "Results/4/SouthernIndiana_WUI_region14_CSF_sectored_kernel_intensity.rds"
text4.10 <- "Results/4/SouthernIndiana_WUI_region14_PMF_sectored_kernel_intensity.rds"
text4.11 <- "Results/4/SouthernIndiana_WUI_region14_MCC_sectored_kernel_intensity.rds"
text4.12 <- "Results/4/SouthernIndiana_WUI_region14_ori_sectored_kernel_intensity.rds"

text5.9 <- "Results/4/SouthernIndiana_WUI_region14_CSF_kernel_intensity.rds"
text5.10 <- "Results/4/SouthernIndiana_WUI_region14_PMF_kernel_intensity.rds"
text5.11 <- "Results/4/SouthernIndiana_WUI_region14_MCC_kernel_intensity.rds"
text5.12 <- "Results/4/SouthernIndiana_WUI_region14_ori_kernel_intensity.rds"

text4.13 <- "Results/4/SouthernIndiana_WUI_region14_CSF_sectored_kernel_height_intensity.rds"
text4.14 <- "Results/4/SouthernIndiana_WUI_region14_PMF_sectored_kernel_height_intensity.rds"
text4.15 <- "Results/4/SouthernIndiana_WUI_region14_MCC_sectored_kernel_height_intensity.rds"
text4.16 <- "Results/4/SouthernIndiana_WUI_region14_ori_sectored_kernel_height_intensity.rds"

text5.13 <- "Results/4/SouthernIndiana_WUI_region14_CSF_kernel_height_intensity.rds"
text5.14 <- "Results/4/SouthernIndiana_WUI_region14_PMF_kernel_height_intensity.rds"
text5.15 <- "Results/4/SouthernIndiana_WUI_region14_MCC_kernel_height_intensity.rds"
text5.16 <- "Results/4/SouthernIndiana_WUI_region14_ori_kernel_height_intensity.rds"

saveRDS(result_csf_4, text4.1)
saveRDS(result_pmf_4, text4.2)
saveRDS(result_mcc_4, text4.3)
saveRDS(result_ori_4, text4.4)
saveRDS(result_csf_4_k, text5.1)
saveRDS(result_pmf_4_k, text5.2)
saveRDS(result_mcc_4_k, text5.3)
saveRDS(result_ori_4_k, text5.4)
saveRDS(result_csf_4_1, text4.5)
saveRDS(result_pmf_4_1, text4.6)
saveRDS(result_mcc_4_1, text4.7)
saveRDS(result_ori_4_1, text4.8)
saveRDS(result_csf_4_1_k, text5.5)
saveRDS(result_pmf_4_1_k, text5.6)
saveRDS(result_mcc_4_1_k, text5.7)
saveRDS(result_ori_4_1_k, text5.8)
saveRDS(result_csf_5, text4.9)
saveRDS(result_pmf_5, text4.10)
saveRDS(result_mcc_5, text4.11)
saveRDS(result_ori_5, text4.12)
saveRDS(result_csf_5_k, text5.9)
saveRDS(result_pmf_5_k, text5.10)
saveRDS(result_mcc_5_k, text5.11)
saveRDS(result_ori_5_k, text5.12)
saveRDS(result_csf_5_1, text4.13)
saveRDS(result_pmf_5_1, text4.14)
saveRDS(result_mcc_5_1, text4.15)
saveRDS(result_ori_5_1, text4.16)
saveRDS(result_csf_5_1_k, text5.13)
saveRDS(result_pmf_5_1_k, text5.14)
saveRDS(result_mcc_5_1_k, text5.15)
saveRDS(result_ori_5_1_k, text5.16)
