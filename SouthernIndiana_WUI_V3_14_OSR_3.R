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

house_center_clean <- function(temp, k=100, p=0.3){
  nn <- get.knn(temp, k=k)
  avg_dist <- rowMeans(nn$nn.dist)
  density <- 1/(avg_dist+1e-6)
  thres <- quantile(density, p)
  is_center <- as.numeric(density<thres)
  return(is_center)
}

sectored_kernel_Hessg <- function(h, point_data_high, height_high){
  Hessg <- compute_hessi_eig_dif_4_sectored(length(height_high), point_data_high, h, height_high)
  Hessg[which(is.infinite(Hessg))] <- 200
  return(Hessg)
}

kernel_Hessg <- function(h, point_data_high, height_high){
  Hessg <- compute_hessi_eig_dif(length(height_high), point_data_high, h, height_high)
  Hessg[which(is.infinite(Hessg))] <- 200
  return(Hessg)
}

GMM_Cluster_4 <- function(Hessg, height_high, N){
  GMM <- Mclust(Hessg, G=N)
  label <- GMM$classification
  return(label)
}

GMM_Cluster_4_1 <- function(Hessg, height_high, intensity_high, N){
  orig_data_high <- cbind(Hessg, intensity_high)
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(label)
}

GMM_Cluster_5 <- function(Hessg, height_high, N){
  orig_data_high <- cbind(Hessg, height_high)
  orig_data_high_prep <- scale(orig_data_high, scale=apply(orig_data_high, 2, sd))
  GMM <- Mclust(orig_data_high_prep, G=N)
  label <- GMM$classification
  return(label)
}

GMM_Cluster_5_1 <- function(Hessg, height_high, intensity_high, N){
  orig_data_high <- cbind(Hessg, intensity_high, height_high)
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

i <- 71
indice_base <- which((lasw[, 1]>=region[i, 1]+360)&(lasw[, 2]>=region[i, 2])
                     &(lasw[, 1]<=region[i, 3]+360)&(lasw[, 2]<=region[i, 4]))
temp_2 <- lasw[indice_base, ]
intensity_2 <- lasw_intensity[indice_base]
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[las$Classification[indice_base]], decorate=(box=FALSE))

ground_result_osr <- readRDS("Results/4/SouthernIndiana_WUI_region14_OSR.rds")
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[ground_result_osr], decorate=(box=FALSE))

nonground_patch_osr <- temp_2[which(ground_result_osr!=2), ]
intensity_nonground_patch_osr <- intensity_2[which(ground_result_osr!=2)]
height_nonground_patch_osr <- nonground_patch_osr[, 3]-min(nonground_patch_osr[, 3])

h2 <- c(15, 15, 30)
h_matrix_osr <- matrix(rep(h2, length(nonground_patch_osr[, 1])), length(nonground_patch_osr[, 1]), byrow = TRUE)

GMM_cluster_whole_osr_Hessg_1 <- sectored_kernel_Hessg(h_matrix_osr, nonground_patch_osr, height_nonground_patch_osr)
GMM_cluster_whole_osr_Hessg_2 <- kernel_Hessg(h_matrix_osr, nonground_patch_osr, height_nonground_patch_osr)

GMM_cluster_whole_osr_4_1 <- GMM_Cluster_4(GMM_cluster_whole_osr_Hessg_1, height_nonground_patch_osr, 2)
GMM_cluster_whole_osr_4_2 <- GMM_Cluster_4(GMM_cluster_whole_osr_Hessg_2, height_nonground_patch_osr, 2)
GMM_cluster_whole_osr_4_1_i <- GMM_Cluster_4_1(GMM_cluster_whole_osr_Hessg_1, height_nonground_patch_osr, intensity_nonground_patch_osr, 2)
GMM_cluster_whole_osr_4_2_i <- GMM_Cluster_4_1(GMM_cluster_whole_osr_Hessg_2, height_nonground_patch_osr, intensity_nonground_patch_osr, 2)

GMM_cluster_whole_osr_5_1 <- GMM_Cluster_5(GMM_cluster_whole_osr_Hessg_1, height_nonground_patch_osr, 2)
GMM_cluster_whole_osr_5_2 <- GMM_Cluster_5(GMM_cluster_whole_osr_Hessg_2, height_nonground_patch_osr, 2)
GMM_cluster_whole_osr_5_1_i <- GMM_Cluster_5_1(GMM_cluster_whole_osr_Hessg_1, height_nonground_patch_osr, intensity_nonground_patch_osr, 2)
GMM_cluster_whole_osr_5_2_i <- GMM_Cluster_5_1(GMM_cluster_whole_osr_Hessg_2, height_nonground_patch_osr, intensity_nonground_patch_osr, 2)

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_4_2], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_4_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_4_2_i], decorate=(box=FALSE))

plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_5_2], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_5_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_osr, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_osr_5_2_i], decorate=(box=FALSE))

result_osr_4_1 <- rep(2, length(temp_2[, 1]))
result_osr_4_2 <- rep(2, length(temp_2[, 1]))
result_osr_4_1_i <- rep(2, length(temp_2[, 1]))
result_osr_4_2_i <- rep(2, length(temp_2[, 1]))

result_osr_5_1 <- rep(2, length(temp_2[, 1]))
result_osr_5_2 <- rep(2, length(temp_2[, 1]))
result_osr_5_1_i <- rep(2, length(temp_2[, 1]))
result_osr_5_2_i <- rep(2, length(temp_2[, 1]))

result_osr_4_1[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4_1, GMM_cluster_whole_osr_Hessg_1)
result_osr_4_2[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4_2, GMM_cluster_whole_osr_Hessg_2)
result_osr_4_1_i[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4_1_i, GMM_cluster_whole_osr_Hessg_1)
result_osr_4_2_i[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_4_2_i, GMM_cluster_whole_osr_Hessg_2)

result_osr_5_1[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_5_1, GMM_cluster_whole_osr_Hessg_1)
result_osr_5_2[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_5_2, GMM_cluster_whole_osr_Hessg_2)
result_osr_5_1_i[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_5_1_i, GMM_cluster_whole_osr_Hessg_1)
result_osr_5_2_i[which(ground_result_osr!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_osr_5_2_i, GMM_cluster_whole_osr_Hessg_2)

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_4_2_i], decorate=(box=FALSE))

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_osr_5_2_i], decorate=(box=FALSE))

text16.1 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel_only.rds"
text16.2 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel_only.rds"
text16.3 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel_only_and_height.rds"
text16.4 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel_only_and_height.rds"
text16.5 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel_only_intensity.rds"
text16.6 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel_only_intensity.rds"
text16.7 <- "Results/4/SouthernIndiana_WUI_region14_OSR_sectored_kernel_only_intensity_and_height.rds"
text16.8 <- "Results/4/SouthernIndiana_WUI_region14_OSR_kernel_only_intensity_and_height.rds"

saveRDS(result_osr_4_1, text16.1)
saveRDS(result_osr_4_2, text16.2)
saveRDS(result_osr_5_1, text16.3)
saveRDS(result_osr_5_2, text16.4)
saveRDS(result_osr_4_1_i, text16.5)
saveRDS(result_osr_4_2_i, text16.6)
saveRDS(result_osr_5_1_i, text16.7)
saveRDS(result_osr_5_2_i, text16.8)

