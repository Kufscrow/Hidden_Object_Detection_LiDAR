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

sourceCpp("C++ code/Ground_local_hyperbolic_patches.cpp")
sourceCpp("C++ code/Ground_local_quadratic_functions.cpp")
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

patches.ground.filtering <- function(patch, hh, adjust, Size, More, outlier){
  center <- rep(2)
  center[1] <- (region[patch,1]+region[patch,3])/2
  center[2] <- (region[patch,2]+region[patch,4])/2
  indice_base <- which((lasw[, 1]>=region[patch, 1])&(lasw[, 2]>=region[patch, 2])
                       &(lasw[, 1]<=region[patch, 3])&(lasw[, 2]<=region[patch, 4]))
  indice_extend <- which((lasw[, 1]>=region[patch, 5])&(lasw[, 2]>=region[patch, 6])
                         &(lasw[, 1]<=region[patch, 7])&(lasw[, 2]<=region[patch, 8]))
  las.patch.base <- lasw[indice_base, ]
  las.patch.extended <- lasw[indice_extend, ]
  temp_1 <- las.patch.base
  temp_2 <- las.patch.extended
  temp_1[, 1] <- las.patch.base[, 1]-center[1]
  temp_2[, 1] <- las.patch.extended[, 1]-center[1]
  temp_1[, 2] <- las.patch.base[, 2]-center[2]
  temp_2[, 2] <- las.patch.extended[, 2]-center[2]
  result.filtering.hyperbola <- ground_filtering_hyperbola_patches(temp_1,temp_2,hh=hh,Size=Size,More=More,adjust=adjust,threshold=8,outlier=outlier)
  patch.result <- result.filtering.hyperbola$measure[, 5]+1
  return(list("result"=patch.result, "indice"=indice_base))
}

transx.patches.ground.filtering <- function(patch, dist, hh, adjust, Size, More, outlier){
  center <- rep(2)
  center[1] <- (region[patch,1]+dist+region[patch+k1,3]-dist)/2
  center[2] <- (region[patch,2]+region[patch,4])/2
  indice_base <- which((lasw[, 1]>=region[patch, 1]+dist)&(lasw[, 2]>=region[patch, 2])
                       &(lasw[, 1]<=region[patch, 3]+dist)&(lasw[, 2]<=region[patch, 4]))
  indice_extend <- which((lasw[, 1]>=region[patch, 5]+dist)&(lasw[, 2]>=region[patch, 6])
                         &(lasw[, 1]<=region[patch, 7]+dist)&(lasw[, 2]<=region[patch, 8]))
  las.patch.base <- lasw[indice_base, ]
  las.patch.extended <- lasw[indice_extend, ]
  temp_1 <- las.patch.base
  temp_2 <- las.patch.extended
  temp_1[, 1] <- las.patch.base[, 1]-center[1]
  temp_2[, 1] <- las.patch.extended[, 1]-center[1]
  temp_1[, 2] <- las.patch.base[, 2]-center[2]
  temp_2[, 2] <- las.patch.extended[, 2]-center[2]
  result.filtering.hyperbola <- ground_filtering_hyperbola_patches(temp_1,temp_2,hh=hh,Size=Size,More=More,adjust=adjust,threshold=8,outlier=outlier)
  patch.result <- result.filtering.hyperbola$measure[, 5]+1
  return(list("result"=patch.result, "indice"=indice_base))
}

transy.patches.ground.filtering <- function(patch, dist, hh, adjust, Size, More, outlier){
  center <- rep(2)
  center[1] <- (region[patch,1]+region[patch,3])/2
  center[2] <- (region[patch,2]+dist+region[patch+1,4]-dist)/2
  indice_base <- which((lasw[, 1]>=region[patch, 1])&(lasw[, 2]>=region[patch, 2]+dist)
                       &(lasw[, 1]<=region[patch, 3])&(lasw[, 2]<=region[patch, 4]+dist))
  indice_extend <- which((lasw[, 1]>=region[patch, 5])&(lasw[, 2]>=region[patch, 6]+dist)
                         &(lasw[, 1]<=region[patch, 7])&(lasw[, 2]<=region[patch, 8]+dist))
  las.patch.base <- lasw[indice_base, ]
  las.patch.extended <- lasw[indice_extend, ]
  temp_1 <- las.patch.base
  temp_2 <- las.patch.extended
  temp_1[, 1] <- las.patch.base[, 1]-center[1]
  temp_2[, 1] <- las.patch.extended[, 1]-center[1]
  temp_1[, 2] <- las.patch.base[, 2]-center[2]
  temp_2[, 2] <- las.patch.extended[, 2]-center[2]
  result.filtering.hyperbola <- ground_filtering_hyperbola_patches(temp_1,temp_2,hh=hh,Size=Size,More=More,adjust=adjust,threshold=8,outlier=outlier)
  patch.result <- result.filtering.hyperbola$measure[, 5]+1
  return(list("result"=patch.result, "indice"=indice_base))
}

house_center_clean <- function(temp, k=100, p=0.3){
  nn <- get.knn(temp, k=k)
  avg_dist <- rowMeans(nn$nn.dist)
  density <- 1/(avg_dist+1e-6)
  thres <- quantile(density, p)
  is_center <- as.numeric(density<thres)
  return(is_center)
}

GMM_Cluster_4 <- function(h, point_data_high, height_high, intensity_high, N){
  Hessg <- compute_hessi_eig_dif_4_sectored(length(height_high), point_data_high, h, height_high)
  Hessg[which(is.infinite(Hessg))] <- 200
  orig_data_high <- cbind(Hessg, intensity_high)
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(list("label"=label, "Hessg"=Hessg))
}

GMM_Cluster_5 <- function(h, point_data_high, height_high, intensity_high, N){
  Hessg <- compute_hessi_eig_dif(length(height_high), point_data_high, h, height_high)
  Hessg[which(is.infinite(Hessg))] <- 200
  orig_data_high <- cbind(Hessg, intensity_high)
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(list("label"=label, "Hessg"=Hessg))
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

relabel_roof_edges_2 <- function(pts, is_center, metric, height, k = 100, radius = 2.0, 
                                 metric_threshold = 0.7, height_threshold = 2.0) {
  require(RANN)
  nn <- nn2(pts[,1:3], k=k, searchtype = "radius", radius = radius)
  is_edge <- rep(FALSE, length(pts[, 1]))
  progress <- txtProgressBar(min = 0, max = nrow(pts), style = 3)
  for (i in which(is_center)) {
    neighbors <- nn$nn.idx[i, nn$nn.idx[i,] > 0]
    metric_diff <- abs(metric[neighbors] - metric[i])
    height_diff <- abs(height[neighbors] - height[i])
    edge_candidates <- neighbors[
      (metric_diff > metric_threshold) &
        (height_diff < height_threshold)
    ]
    is_edge[edge_candidates[!is_center[edge_candidates]]] <- TRUE
    
    setTxtProgressBar(progress, i)
  }
  close(progress)
  is_edge[is_center] <- FALSE
  return(is_edge)
}

label_fix_GMM_nonground_1 <- function(GMM_label, intensity){
  if (length(GMM_label)!=length(intensity)){
    stop("Dimension Mismatch")
  }
  label <- -2*as.numeric(GMM_label!=GMM_label[which.max(intensity)])+3
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

result.patch.1 <- transx.patches.ground.filtering(71, 360, hh_cand[[3]], adjust, Size, More, outlier)

temp.patch.1.ground <- temp_2[which(result.patch.1$result==2), ]
temp.patch.1.ground.labels <- ground_fit_quadratic_local(temp.patch.1.ground, iter=0, num_windows_x1=5, num_windows_x2=5, overlap=0.2, noise=FALSE, coeff=4.5)
result.patch.2 <- result.patch.1$result
result.patch.2[which(result.patch.1$result==2)] <- -temp.patch.1.ground.labels$ground_label+4
result.patch.2[which((result.patch.2==3)&(temp_2[, 1]<=region[i, 1]+380))] <- 2
temp.patch.2.nonground <- temp_2[which(result.patch.2!=2), ]
result.patch.2.nonground <- result.patch.2[which(result.patch.2!=2)]
intensity.patch.2.nonground <- intensity_2[which(result.patch.2!=2)]
height.patch.2.nonground <- temp.patch.2.nonground[, 3]-min(temp.patch.2.nonground[, 3])

result.patch.3.nonground <- rep(1, length(result.patch.2.nonground))
result.patch.3.nonground.1 <- rep(1, length(result.patch.2.nonground))
result.patch.3.nonground.2 <- rep(1, length(result.patch.2.nonground))
result.patch.3.nonground.3 <- rep(1, length(result.patch.2.nonground))
result.patch.3.nonground.4 <- rep(1, length(result.patch.2.nonground))
result.patch.3.nonground.5 <- rep(1, length(result.patch.2.nonground))
result.patch.3.nonground.6 <- rep(1, length(result.patch.2.nonground))
result.patch.3.nonground.7 <- rep(1, length(result.patch.2.nonground))

result.patch.3 <- rep(2, length(indice_base))
result.patch.3.1 <- rep(2, length(indice_base))
result.patch.3.2 <- rep(2, length(indice_base))
result.patch.3.3 <- rep(2, length(indice_base))
result.patch.3.4 <- rep(2, length(indice_base))
result.patch.3.5 <- rep(2, length(indice_base))
result.patch.3.6 <- rep(2, length(indice_base))
result.patch.3.7 <- rep(2, length(indice_base))

indice_base_partial <- which((temp.patch.2.nonground[, 2]>=region[i, 4]-500))
temp.patch.2.nonground.partial <- temp.patch.2.nonground[indice_base_partial, ]
result.patch.2.nonground.partial <- result.patch.2.nonground[indice_base_partial]
temp.patch.2.house.center <- temp.patch.2.nonground.partial[which(result.patch.2.nonground.partial==3), ]
temp.patch.2.fix.indice <- house_center_clean(temp.patch.2.house.center, k=10, p=0.2)
temp.patch.2.center.label <- rep(FALSE, length(indice_base_partial))
temp.patch.2.center.label[(which(result.patch.2.nonground.partial==3))[which(temp.patch.2.fix.indice==0)]] <- TRUE
intensity.patch.2.nonground.partial <- intensity.patch.2.nonground[indice_base_partial]
height.patch.2.nonground.partial <- temp.patch.2.nonground.partial[, 3]-min(temp.patch.2.nonground.partial[, 3])

h2 <- c(15, 15, 30)
h_matrix_2 <- matrix(rep(h2, length(temp.patch.2.nonground.partial[, 1])), length(temp.patch.2.nonground.partial[, 1]), byrow = TRUE)
tt7 <- Sys.time()
GMM_cluster_whole_1_4 <- GMM_Cluster_4(h_matrix_2, temp.patch.2.nonground.partial, height.patch.2.nonground.partial, intensity.patch.2.nonground.partial, 2)
tt8 <- Sys.time()
difftime(tt8, tt7, units="mins")
temp.patch.2.edge.1 <- relabel_roof_edges_1(temp.patch.2.nonground.partial, temp.patch.2.center.label, GMM_cluster_whole_1_4$Hessg, k=200, radius=8, metric_threshold=2)
temp.patch.2.edge.3 <- relabel_roof_edges_2(temp.patch.2.nonground.partial, temp.patch.2.center.label, GMM_cluster_whole_1_4$Hessg, height.patch.2.nonground.partial,
                                            k=200, radius=8, metric_threshold=2, height_threshold=5)
tt7 <- Sys.time()
GMM_cluster_whole_1_5 <- GMM_Cluster_5(h_matrix_2, temp.patch.2.nonground.partial, height.patch.2.nonground.partial, intensity.patch.2.nonground.partial, 2)
tt8 <- Sys.time()
difftime(tt8, tt7, units="mins")
temp.patch.2.edge.2 <- relabel_roof_edges_1(temp.patch.2.nonground.partial, temp.patch.2.center.label, GMM_cluster_whole_1_5$Hessg, k=200, radius=8, metric_threshold=2)
temp.patch.2.edge.4 <- relabel_roof_edges_2(temp.patch.2.nonground.partial, temp.patch.2.center.label, GMM_cluster_whole_1_5$Hessg, height.patch.2.nonground.partial,
                                            k=200, radius=8, metric_threshold=2, height_threshold=5)

result.patch.3.nonground[indice_base_partial] <- 2*as.numeric(temp.patch.2.edge.1|temp.patch.2.center.label)+1
result.patch.3.nonground.1[indice_base_partial] <- 2*as.numeric(temp.patch.2.edge.2|temp.patch.2.center.label)+1
result.patch.3.nonground.2[indice_base_partial] <- 2*as.numeric(temp.patch.2.edge.3|temp.patch.2.center.label)+1
result.patch.3.nonground.3[indice_base_partial] <- 2*as.numeric(temp.patch.2.edge.4|temp.patch.2.center.label)+1

result.patch.3[which(result.patch.2!=2)] <- result.patch.3.nonground
result.patch.3.1[which(result.patch.2!=2)] <- result.patch.3.nonground.1
result.patch.3.2[which(result.patch.2!=2)] <- result.patch.3.nonground.2
result.patch.3.3[which(result.patch.2!=2)] <- result.patch.3.nonground.3

GMM_cluster_fix_1_4_label <- label_fix_GMM_nonground_1(GMM_cluster_whole_1_4$label, GMM_cluster_whole_1_4$Hessg)
GMM_cluster_1_4_di_label <- joint_knn_prob(temp.patch.2.nonground, result.patch.3.nonground, GMM_cluster_fix_1_4_label, k=40, tau=0.1)
GMM_cluster_1_4_dhi_label <- joint_knn_prob(temp.patch.2.nonground, result.patch.3.nonground.1, GMM_cluster_fix_1_4_label, k=40, tau=0.1)

GMM_cluster_fix_1_5_label <- label_fix_GMM_nonground_1(GMM_cluster_whole_1_5$label, GMM_cluster_whole_1_5$Hessg)
GMM_cluster_1_5_di_label <- joint_knn_prob(temp.patch.2.nonground, result.patch.3.nonground.2, GMM_cluster_fix_1_5_label, k=40, tau=0.1)
GMM_cluster_1_5_dhi_label <- joint_knn_prob(temp.patch.2.nonground, result.patch.3.nonground.3, GMM_cluster_fix_1_5_label, k=40, tau=0.1)

result.patch.3.nonground.4[indice_base_partial] <- GMM_cluster_1_4_di_label
result.patch.3.nonground.5[indice_base_partial] <- GMM_cluster_1_4_dhi_label
result.patch.3.nonground.6[indice_base_partial] <- GMM_cluster_1_5_di_label 
result.patch.3.nonground.7[indice_base_partial] <- GMM_cluster_1_5_dhi_label

result.patch.3[which(result.patch.2!=2)] <- result.patch.3.nonground
result.patch.3.1[which(result.patch.2!=2)] <- result.patch.3.nonground.1
result.patch.3.2[which(result.patch.2!=2)] <- result.patch.3.nonground.2
result.patch.3.3[which(result.patch.2!=2)] <- result.patch.3.nonground.3
result.patch.3.4[which(result.patch.2!=2)] <- result.patch.3.nonground.4
result.patch.3.5[which(result.patch.2!=2)] <- result.patch.3.nonground.5
result.patch.3.6[which(result.patch.2!=2)] <- result.patch.3.nonground.6
result.patch.3.7[which(result.patch.2!=2)] <- result.patch.3.nonground.7

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.1$result], decorate=(box=FALSE) )
plot3d(temp.patch.1.ground, aspect=c(1, 1, 0.1), decorate=(box=FALSE) )
plot3d(temp.patch.1.ground, aspect=c(1, 1, 0.1), decorate=(box=FALSE), col=label_to_color_1[temp.patch.1.ground.labels$ground_label])
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.2], decorate=(box=FALSE) )
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), decorate=(box=FALSE) )
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.2.nonground.partial], decorate=(box=FALSE) )
plot3d(temp.patch.2.house.center, aspect=c(1, 1, 0.1), decorate=(box=FALSE) )
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[as.numeric(temp.patch.2.center.label)+1], decorate=(box=FALSE))

plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_1_4$label], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[2*as.numeric(temp.patch.2.edge.1|temp.patch.2.center.label)+1], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[2*as.numeric(temp.patch.2.edge.3|temp.patch.2.center.label)+1], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.nonground[indice_base_partial]], decorate=(box=FALSE))

plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_1_5$label], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[2*as.numeric(temp.patch.2.edge.2|temp.patch.2.center.label)+1], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[2*as.numeric(temp.patch.2.edge.4|temp.patch.2.center.label)+1], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground.partial, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.nonground.1[indice_base_partial]], decorate=(box=FALSE))

plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.nonground], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.nonground.1], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.nonground.2], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.nonground.3], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_1_4_di_label], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_1_5_di_label], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_1_4_dhi_label], decorate=(box=FALSE))
plot3d(temp.patch.2.nonground, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_1_5_dhi_label], decorate=(box=FALSE))

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.3], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.4], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.5], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.6], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.3.7], decorate=(box=FALSE))

text3.1 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_sectored_kernel.rds"
text3.2 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_kernel.rds"
text3.3 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_sectored_kernel_and_height.rds"
text3.4 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_kernel_and_height.rds"
text3.5 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_sectored_kernel_and_intensity.rds"
text3.6 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_kernel_and_intensity.rds"
text3.7 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_sectored_kernel_and_height_and_intensity.rds"
text3.8 <- "Results/4/SouthernIndiana_WUI_region14_h(1,4)_Result_kernel_and_height_and_intensity.rds"

saveRDS(result.patch.3, text3.1)
saveRDS(result.patch.3.1, text3.2)
saveRDS(result.patch.3.2, text3.3)
saveRDS(result.patch.3.3, text3.4)
saveRDS(result.patch.3.4, text3.5)
saveRDS(result.patch.3.5, text3.6)
saveRDS(result.patch.3.6, text3.7)
saveRDS(result.patch.3.7, text3.8)
