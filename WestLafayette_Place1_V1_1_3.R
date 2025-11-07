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
las <- readLAS("LiDAR Datasets/in2018_30001885_12.las", filter="-drop_withheld")
lasw <- cbind(las$X, las$Y, las$Z)
lasw_intensity <- las$Intensity
lasw_sample <- as.matrix(lasw)

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

i <- 59
indice_base <- which((lasw[, 1]>=region[i, 1]+440)&(lasw[, 2]>=region[i, 2]+330)
                     &(lasw[, 1]<=region[i, 3]+440)&(lasw[, 2]<=region[i, 4]+330))
temp_2 <- lasw[indice_base, ]
intensity_2 <- lasw_intensity[indice_base]
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[las$Classification[indice_base]], decorate=(box=FALSE))

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
  center[1] <- (region[patch,1]+dist+region[patch,3]+dist)/2
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
  center[2] <- (region[patch,2]+dist+region[patch,4]+dist)/2
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

transxy.patches.ground.filtering <- function(patch, distx, disty, hh, adjust, Size, More, outlier){
  center <- rep(2)
  center[1] <- (region[patch,1]+distx+region[patch,3]+distx)/2
  center[2] <- (region[patch,2]+disty+region[patch,4]+disty)/2
  indice_base <- which((lasw[, 1]>=region[patch, 1]+distx)&(lasw[, 2]>=region[patch, 2]+disty)
                       &(lasw[, 1]<=region[patch, 3]+distx)&(lasw[, 2]<=region[patch, 4]+disty))
  indice_extend <- which((lasw[, 1]>=region[patch, 5]+distx)&(lasw[, 2]>=region[patch, 6]+disty)
                         &(lasw[, 1]<=region[patch, 7]+distx)&(lasw[, 2]<=region[patch, 8]+disty))
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

hh_cand <- list(c(1,1), c(1,2), c(1,4), c(1,8))
adjust <- 2.0 
Size <- 50
More <- 20
outlier <- 1

i <- 59
indice_base <- which((lasw[, 1]>=region[i, 1]+440)&(lasw[, 2]>=region[i, 2]+330)
                     &(lasw[, 1]<=region[i, 3]+440)&(lasw[, 2]<=region[i, 4]+330))
temp_2 <- lasw[indice_base, ]
intensity_2 <- lasw_intensity[indice_base]
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[las$Classification[indice_base]], decorate=(box=FALSE))

df_temp_2 <- data.frame(temp_2, las$Classification[indice_base])
names(df_temp_2) <- c("X", "Y", "Z", "Classification")
las.patch.base <- LAS(df_temp_2)

hh_cand <- list(c(1,1), c(1,2), c(1,4), c(1,8))
adjust <- 2.0 
Size <- 50
More <- 20
outlier <- 1
result.patch.1 <- transxy.patches.ground.filtering(59, 440, 330, hh_cand[[3]], adjust, Size, More, outlier)
temp.patch.1.ground <- temp_2[which(result.patch.1$result==2), ]
temp.patch.1.ground.labels <- ground_fit_linear_local(temp.patch.1.ground, iter=0, num_windows_x1=3, num_windows_x2=3, overlap=0.2, noise=FALSE, coeff=1.5)
result.patch.2 <- result.patch.1$result
result.patch.2[which(result.patch.1$result==2)] <- temp.patch.1.ground.labels$ground_label
temp.patch.2.nonground <- temp_2[which(result.patch.2==1),]
df_nonground_2_om <- data.frame(temp.patch.2.nonground)
names(df_nonground_2_om) <- c("X", "Y", "Z")
ng_patch_om <- LAS(df_nonground_2_om)
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.2], decorate=(box=FALSE))

las_patch_csf <- classify_ground(las.patch.base, algorithm = csf(sloop_smooth = TRUE, class_threshold = 1, cloth_resolution = 1, time_step = 1))
plot(las_patch_csf,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

las_patch_pmf <- classify_ground(las.patch.base, algorithm = pmf(ws = 5, th = 3))
plot(las_patch_pmf,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

las_patch_mcc <- classify_ground(las.patch.base, algorithm = mcc(1.5,0.3))
plot(las_patch_mcc,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

nonground_patch_om <- temp_2[which(result.patch.2!=2), ]
nonground_patch_csf <- temp_2[which(las_patch_csf$Classification!=2), ]
nonground_patch_pmf <- temp_2[which(las_patch_pmf$Classification!=2), ]
nonground_patch_mcc <- temp_2[which(las_patch_mcc$Classification!=2), ]
nonground_patch_ori <- temp_2[which(las.patch.base$Classification!=2), ]
height_nonground_patch_om <- nonground_patch_om[, 3]-min(nonground_patch_om[, 3])
height_nonground_patch_csf <- nonground_patch_csf[, 3]-min(nonground_patch_csf[, 3])
height_nonground_patch_pmf <- nonground_patch_pmf[, 3]-min(nonground_patch_pmf[, 3])
height_nonground_patch_mcc <- nonground_patch_mcc[, 3]-min(nonground_patch_mcc[, 3])
height_nonground_patch_ori <- nonground_patch_ori[, 3]-min(nonground_patch_ori[, 3])
intensity_nonground_patch_om <- intensity_2[which(result.patch.2!=2)]
intensity_nonground_patch_csf <- intensity_2[which(las_patch_csf$Classification!=2)]
intensity_nonground_patch_pmf <- intensity_2[which(las_patch_pmf$Classification!=2)]
intensity_nonground_patch_mcc <- intensity_2[which(las_patch_mcc$Classification!=2)]
intensity_nonground_patch_ori <- intensity_2[which(las.patch.base$Classification!=2)]

h2 <- c(15, 15, 30)
h_matrix_om <- matrix(rep(h2, length(nonground_patch_om[, 1])), length(nonground_patch_om[, 1]), byrow = TRUE)
h_matrix_csf <- matrix(rep(h2, length(nonground_patch_csf[, 1])), length(nonground_patch_csf[, 1]), byrow = TRUE)
h_matrix_pmf <- matrix(rep(h2, length(nonground_patch_pmf[, 1])), length(nonground_patch_pmf[, 1]), byrow = TRUE)
h_matrix_mcc <- matrix(rep(h2, length(nonground_patch_mcc[, 1])), length(nonground_patch_mcc[, 1]), byrow = TRUE)
h_matrix_ori <- matrix(rep(h2, length(nonground_patch_ori[, 1])), length(nonground_patch_ori[, 1]), byrow = TRUE)

GMM_cluster_whole_om_Hessg_1 <- sectored_kernel_Hessg(h_matrix_om, nonground_patch_om, height_nonground_patch_om)
GMM_cluster_whole_csf_Hessg_1 <- sectored_kernel_Hessg(h_matrix_csf, nonground_patch_csf, height_nonground_patch_csf)
GMM_cluster_whole_pmf_Hessg_1 <- sectored_kernel_Hessg(h_matrix_pmf, nonground_patch_pmf, height_nonground_patch_pmf)
GMM_cluster_whole_mcc_Hessg_1 <- sectored_kernel_Hessg(h_matrix_mcc, nonground_patch_mcc, height_nonground_patch_mcc)
GMM_cluster_whole_ori_Hessg_1 <- sectored_kernel_Hessg(h_matrix_ori, nonground_patch_ori, height_nonground_patch_ori)
GMM_cluster_whole_om_Hessg_2 <- kernel_Hessg(h_matrix_om, nonground_patch_om, height_nonground_patch_om)
GMM_cluster_whole_csf_Hessg_2 <- kernel_Hessg(h_matrix_csf, nonground_patch_csf, height_nonground_patch_csf)
GMM_cluster_whole_pmf_Hessg_2 <- kernel_Hessg(h_matrix_pmf, nonground_patch_pmf, height_nonground_patch_pmf)
GMM_cluster_whole_mcc_Hessg_2 <- kernel_Hessg(h_matrix_mcc, nonground_patch_mcc, height_nonground_patch_mcc)
GMM_cluster_whole_ori_Hessg_2 <- kernel_Hessg(h_matrix_ori, nonground_patch_ori, height_nonground_patch_ori)

GMM_cluster_whole_om_4_1 <- GMM_Cluster_4(GMM_cluster_whole_om_Hessg_1, height_nonground_patch_om, 2)
GMM_cluster_whole_csf_4_1 <- GMM_Cluster_4(GMM_cluster_whole_csf_Hessg_1, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4_1 <- GMM_Cluster_4(GMM_cluster_whole_pmf_Hessg_1, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4_1 <- GMM_Cluster_4(GMM_cluster_whole_mcc_Hessg_1, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4_1 <- GMM_Cluster_4(GMM_cluster_whole_ori_Hessg_1, height_nonground_patch_ori, 2)
GMM_cluster_whole_om_4_2 <- GMM_Cluster_4(GMM_cluster_whole_om_Hessg_2, height_nonground_patch_om, 2)
GMM_cluster_whole_csf_4_2 <- GMM_Cluster_4(GMM_cluster_whole_csf_Hessg_2, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4_2 <- GMM_Cluster_4(GMM_cluster_whole_pmf_Hessg_2, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4_2 <- GMM_Cluster_4(GMM_cluster_whole_mcc_Hessg_2, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4_2 <- GMM_Cluster_4(GMM_cluster_whole_ori_Hessg_2, height_nonground_patch_ori, 2)
GMM_cluster_whole_om_4_1_i <- GMM_Cluster_4_1(GMM_cluster_whole_om_Hessg_1, height_nonground_patch_om, intensity_nonground_patch_om, 2)
GMM_cluster_whole_csf_4_1_i <- GMM_Cluster_4_1(GMM_cluster_whole_csf_Hessg_1, height_nonground_patch_csf, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4_1_i <- GMM_Cluster_4_1(GMM_cluster_whole_pmf_Hessg_1, height_nonground_patch_pmf, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4_1_i <- GMM_Cluster_4_1(GMM_cluster_whole_mcc_Hessg_1, height_nonground_patch_mcc, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4_1_i <- GMM_Cluster_4_1(GMM_cluster_whole_ori_Hessg_1, height_nonground_patch_ori, intensity_nonground_patch_ori, 2)
GMM_cluster_whole_om_4_2_i <- GMM_Cluster_4_1(GMM_cluster_whole_om_Hessg_2, height_nonground_patch_om, intensity_nonground_patch_om, 2)
GMM_cluster_whole_csf_4_2_i <- GMM_Cluster_4_1(GMM_cluster_whole_csf_Hessg_2, height_nonground_patch_csf, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_4_2_i <- GMM_Cluster_4_1(GMM_cluster_whole_pmf_Hessg_2, height_nonground_patch_pmf, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_4_2_i <- GMM_Cluster_4_1(GMM_cluster_whole_mcc_Hessg_2, height_nonground_patch_mcc, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_4_2_i <- GMM_Cluster_4_1(GMM_cluster_whole_ori_Hessg_2, height_nonground_patch_ori, intensity_nonground_patch_ori, 2)

GMM_cluster_whole_om_5_1 <- GMM_Cluster_5(GMM_cluster_whole_om_Hessg_1, height_nonground_patch_om, 2)
GMM_cluster_whole_csf_5_1 <- GMM_Cluster_5(GMM_cluster_whole_csf_Hessg_1, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5_1 <- GMM_Cluster_5(GMM_cluster_whole_pmf_Hessg_1, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5_1 <- GMM_Cluster_5(GMM_cluster_whole_mcc_Hessg_1, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5_1 <- GMM_Cluster_5(GMM_cluster_whole_ori_Hessg_1, height_nonground_patch_ori, 2)
GMM_cluster_whole_om_5_2 <- GMM_Cluster_5(GMM_cluster_whole_om_Hessg_2, height_nonground_patch_om, 2)
GMM_cluster_whole_csf_5_2 <- GMM_Cluster_5(GMM_cluster_whole_csf_Hessg_2, height_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5_2 <- GMM_Cluster_5(GMM_cluster_whole_pmf_Hessg_2, height_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5_2 <- GMM_Cluster_5(GMM_cluster_whole_mcc_Hessg_2, height_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5_2 <- GMM_Cluster_5(GMM_cluster_whole_ori_Hessg_2, height_nonground_patch_ori, 2)
GMM_cluster_whole_om_5_1_i <- GMM_Cluster_5_1(GMM_cluster_whole_om_Hessg_1, height_nonground_patch_om, intensity_nonground_patch_om, 2)
GMM_cluster_whole_csf_5_1_i <- GMM_Cluster_5_1(GMM_cluster_whole_csf_Hessg_1, height_nonground_patch_csf, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5_1_i <- GMM_Cluster_5_1(GMM_cluster_whole_pmf_Hessg_1, height_nonground_patch_pmf, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5_1_i <- GMM_Cluster_5_1(GMM_cluster_whole_mcc_Hessg_1, height_nonground_patch_mcc, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5_1_i <- GMM_Cluster_5_1(GMM_cluster_whole_ori_Hessg_1, height_nonground_patch_ori, intensity_nonground_patch_ori, 2)
GMM_cluster_whole_om_5_2_i <- GMM_Cluster_5_1(GMM_cluster_whole_om_Hessg_2, height_nonground_patch_om, intensity_nonground_patch_om, 2)
GMM_cluster_whole_csf_5_2_i <- GMM_Cluster_5_1(GMM_cluster_whole_csf_Hessg_2, height_nonground_patch_csf, intensity_nonground_patch_csf, 2)
GMM_cluster_whole_pmf_5_2_i <- GMM_Cluster_5_1(GMM_cluster_whole_pmf_Hessg_2, height_nonground_patch_pmf, intensity_nonground_patch_pmf, 2)
GMM_cluster_whole_mcc_5_2_i <- GMM_Cluster_5_1(GMM_cluster_whole_mcc_Hessg_2, height_nonground_patch_mcc, intensity_nonground_patch_mcc, 2)
GMM_cluster_whole_ori_5_2_i <- GMM_Cluster_5_1(GMM_cluster_whole_ori_Hessg_2, height_nonground_patch_ori, intensity_nonground_patch_ori, 2)

plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_4_1], decorate=(box=FALSE))
plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_4_2], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_4_2], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_4_2], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_4_2], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_4_2], decorate=(box=FALSE))
plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_4_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_4_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_4_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_4_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_4_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_4_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_4_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_4_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_4_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_4_2_i], decorate=(box=FALSE))

plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_5_1], decorate=(box=FALSE))
plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_5_2], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_5_2], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_5_2], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_5_2], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_5_2], decorate=(box=FALSE))
plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_5_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_5_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_5_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_5_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_5_1_i], decorate=(box=FALSE))
plot3d(nonground_patch_om, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_om_5_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_csf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_csf_5_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_pmf, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_pmf_5_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_mcc, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_mcc_5_2_i], decorate=(box=FALSE))
plot3d(nonground_patch_ori, aspect=c(1, 1, 0.1), col=label_to_color_1[GMM_cluster_whole_ori_5_2_i], decorate=(box=FALSE))

result_om_4_1 <- rep(2, length(temp_2[, 1]))
result_csf_4_1 <- rep(2, length(temp_2[, 1]))
result_pmf_4_1 <- rep(2, length(temp_2[, 1]))
result_mcc_4_1 <- rep(2, length(temp_2[, 1]))
result_ori_4_1 <- rep(2, length(temp_2[, 1]))
result_om_4_2 <- rep(2, length(temp_2[, 1]))
result_csf_4_2 <- rep(2, length(temp_2[, 1]))
result_pmf_4_2 <- rep(2, length(temp_2[, 1]))
result_mcc_4_2 <- rep(2, length(temp_2[, 1]))
result_ori_4_2 <- rep(2, length(temp_2[, 1]))
result_om_4_1_i <- rep(2, length(temp_2[, 1]))
result_csf_4_1_i <- rep(2, length(temp_2[, 1]))
result_pmf_4_1_i <- rep(2, length(temp_2[, 1]))
result_mcc_4_1_i <- rep(2, length(temp_2[, 1]))
result_ori_4_1_i <- rep(2, length(temp_2[, 1]))
result_om_4_2_i <- rep(2, length(temp_2[, 1]))
result_csf_4_2_i <- rep(2, length(temp_2[, 1]))
result_pmf_4_2_i <- rep(2, length(temp_2[, 1]))
result_mcc_4_2_i <- rep(2, length(temp_2[, 1]))
result_ori_4_2_i <- rep(2, length(temp_2[, 1]))

result_om_5_1 <- rep(2, length(temp_2[, 1]))
result_csf_5_1 <- rep(2, length(temp_2[, 1]))
result_pmf_5_1 <- rep(2, length(temp_2[, 1]))
result_mcc_5_1 <- rep(2, length(temp_2[, 1]))
result_ori_5_1 <- rep(2, length(temp_2[, 1]))
result_om_5_2 <- rep(2, length(temp_2[, 1]))
result_csf_5_2 <- rep(2, length(temp_2[, 1]))
result_pmf_5_2 <- rep(2, length(temp_2[, 1]))
result_mcc_5_2 <- rep(2, length(temp_2[, 1]))
result_ori_5_2 <- rep(2, length(temp_2[, 1]))
result_om_5_1_i <- rep(2, length(temp_2[, 1]))
result_csf_5_1_i <- rep(2, length(temp_2[, 1]))
result_pmf_5_1_i <- rep(2, length(temp_2[, 1]))
result_mcc_5_1_i <- rep(2, length(temp_2[, 1]))
result_ori_5_1_i <- rep(2, length(temp_2[, 1]))
result_om_5_2_i <- rep(2, length(temp_2[, 1]))
result_csf_5_2_i <- rep(2, length(temp_2[, 1]))
result_pmf_5_2_i <- rep(2, length(temp_2[, 1]))
result_mcc_5_2_i <- rep(2, length(temp_2[, 1]))
result_ori_5_2_i <- rep(2, length(temp_2[, 1]))

result_om_4_1[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_4_1, GMM_cluster_whole_om_Hessg_1)
result_csf_4_1[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4_1, GMM_cluster_whole_csf_Hessg_1)
result_pmf_4_1[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4_1, GMM_cluster_whole_pmf_Hessg_1)
result_mcc_4_1[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_4_1, GMM_cluster_whole_mcc_Hessg_1)
result_ori_4_1[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_4_1, GMM_cluster_whole_ori_Hessg_1)
result_om_4_2[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_4_2, GMM_cluster_whole_om_Hessg_2)
result_csf_4_2[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4_2, GMM_cluster_whole_csf_Hessg_2)
result_pmf_4_2[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4_2, GMM_cluster_whole_pmf_Hessg_2)
result_mcc_4_2[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_4_2, GMM_cluster_whole_mcc_Hessg_2)
result_ori_4_2[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_4_2, GMM_cluster_whole_ori_Hessg_2)
result_om_4_1_i[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_4_1_i, GMM_cluster_whole_om_Hessg_1)
result_csf_4_1_i[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4_1_i, GMM_cluster_whole_csf_Hessg_1)
result_pmf_4_1_i[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4_1_i, GMM_cluster_whole_pmf_Hessg_1)
result_mcc_4_1_i[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_4_1_i, GMM_cluster_whole_mcc_Hessg_1)
result_ori_4_1_i[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_4_1_i, GMM_cluster_whole_ori_Hessg_1)
result_om_4_2_i[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_4_2_i, GMM_cluster_whole_om_Hessg_2)
result_csf_4_2_i[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_4_2_i, GMM_cluster_whole_csf_Hessg_2)
result_pmf_4_2_i[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_4_2_i, GMM_cluster_whole_pmf_Hessg_2)
result_mcc_4_2_i[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_4_2_i, GMM_cluster_whole_mcc_Hessg_2)
result_ori_4_2_i[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_4_2_i, GMM_cluster_whole_ori_Hessg_2)

result_om_5_1[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_5_1, GMM_cluster_whole_om_Hessg_1)
result_csf_5_1[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_5_1, GMM_cluster_whole_csf_Hessg_1)
result_pmf_5_1[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_5_1, GMM_cluster_whole_pmf_Hessg_1)
result_mcc_5_1[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_3(GMM_cluster_whole_mcc_5_1, GMM_cluster_whole_mcc_Hessg_1)
result_ori_5_1[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_3(GMM_cluster_whole_ori_5_1, GMM_cluster_whole_ori_Hessg_1)
result_om_5_2[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_5_2, GMM_cluster_whole_om_Hessg_2)
result_csf_5_2[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_3(GMM_cluster_whole_csf_5_2, GMM_cluster_whole_csf_Hessg_2)
result_pmf_5_2[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_5_2, GMM_cluster_whole_pmf_Hessg_2)
result_mcc_5_2[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_3(GMM_cluster_whole_mcc_5_2, GMM_cluster_whole_mcc_Hessg_2)
result_ori_5_2[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_5_2, GMM_cluster_whole_ori_Hessg_2)
result_om_5_1_i[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_5_1_i, GMM_cluster_whole_om_Hessg_1)
result_csf_5_1_i[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_5_1_i, GMM_cluster_whole_csf_Hessg_1)
result_pmf_5_1_i[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_5_1_i, GMM_cluster_whole_pmf_Hessg_1)
result_mcc_5_1_i[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_5_1_i, GMM_cluster_whole_mcc_Hessg_1)
result_ori_5_1_i[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_5_1_i, GMM_cluster_whole_ori_Hessg_1)
result_om_5_2_i[which(result.patch.2!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_om_5_2_i, GMM_cluster_whole_om_Hessg_2)
result_csf_5_2_i[which(las_patch_csf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_csf_5_2_i, GMM_cluster_whole_csf_Hessg_2)
result_pmf_5_2_i[which(las_patch_pmf$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_pmf_5_2_i, GMM_cluster_whole_pmf_Hessg_2)
result_mcc_5_2_i[which(las_patch_mcc$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_mcc_5_2_i, GMM_cluster_whole_mcc_Hessg_2)
result_ori_5_2_i[which(las.patch.base$Classification!=2)] <- label_fix_GMM_nonground_1(GMM_cluster_whole_ori_5_2_i, GMM_cluster_whole_ori_Hessg_2)

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_4_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_4_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_4_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_4_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_4_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_4_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_4_2_i], decorate=(box=FALSE))

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5_1], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_5_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5_2], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_5_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5_1_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_om_5_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_csf_5_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_pmf_5_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_mcc_5_2_i], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result_ori_5_2_i], decorate=(box=FALSE))

text6.1 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_sectored_kernel_only.rds"
text6.2 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_kernel_only.rds"
text6.3 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_sectored_kernel_only_and_height.rds"
text6.4 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_kernel_only_and_height.rds"
text6.5 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_sectored_kernel_only_intensity.rds"
text6.6 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_kernel_only_intensity.rds"
text6.7 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_sectored_kernel_only_intensity_and_height.rds"
text6.8 <- "Results/4/WestLafayette_City_region1_h(1,4)_Result_kernel_only_intensity_and_height.rds"

text7.1 <- "Results/4/WestLafayette_City_region1_CSF_sectored_kernel_only.rds"
text7.2 <- "Results/4/WestLafayette_City_region1_CSF_kernel_only.rds"
text7.3 <- "Results/4/WestLafayette_City_region1_CSF_sectored_kernel_only_and_height.rds"
text7.4 <- "Results/4/WestLafayette_City_region1_CSF_kernel_only_and_height.rds"
text7.5 <- "Results/4/WestLafayette_City_region1_CSF_sectored_kernel_only_intensity.rds"
text7.6 <- "Results/4/WestLafayette_City_region1_CSF_kernel_only_intensity.rds"
text7.7 <- "Results/4/WestLafayette_City_region1_CSF_sectored_kernel_only_intensity_and_height.rds"
text7.8 <- "Results/4/WestLafayette_City_region1_CSF_kernel_only_intensity_and_height.rds"

text8.1 <- "Results/4/WestLafayette_City_region1_PMF_sectored_kernel_only.rds"
text8.2 <- "Results/4/WestLafayette_City_region1_PMF_kernel_only.rds"
text8.3 <- "Results/4/WestLafayette_City_region1_PMF_sectored_kernel_only_and_height.rds"
text8.4 <- "Results/4/WestLafayette_City_region1_PMF_kernel_only_and_height.rds"
text8.5 <- "Results/4/WestLafayette_City_region1_PMF_sectored_kernel_only_intensity.rds"
text8.6 <- "Results/4/WestLafayette_City_region1_PMF_kernel_only_intensity.rds"
text8.7 <- "Results/4/WestLafayette_City_region1_PMF_sectored_kernel_only_intensity_and_height.rds"
text8.8 <- "Results/4/WestLafayette_City_region1_PMF_kernel_only_intensity_and_height.rds"

text9.1 <- "Results/4/WestLafayette_City_region1_MCC_sectored_kernel_only.rds"
text9.2 <- "Results/4/WestLafayette_City_region1_MCC_kernel_only.rds"
text9.3 <- "Results/4/WestLafayette_City_region1_MCC_sectored_kernel_only_and_height.rds"
text9.4 <- "Results/4/WestLafayette_City_region1_MCC_kernel_only_and_height.rds"
text9.5 <- "Results/4/WestLafayette_City_region1_MCC_sectored_kernel_only_intensity.rds"
text9.6 <- "Results/4/WestLafayette_City_region1_MCC_kernel_only_intensity.rds"
text9.7 <- "Results/4/WestLafayette_City_region1_MCC_sectored_kernel_only_intensity_and_height.rds"
text9.8 <- "Results/4/WestLafayette_City_region1_MCC_kernel_only_intensity_and_height.rds"

text10.1 <- "Results/4/WestLafayette_City_region1_ori_sectored_kernel_only.rds"
text10.2 <- "Results/4/WestLafayette_City_region1_ori_kernel_only.rds"
text10.3 <- "Results/4/WestLafayette_City_region1_ori_sectored_kernel_only_and_height.rds"
text10.4 <- "Results/4/WestLafayette_City_region1_ori_kernel_only_and_height.rds"
text10.5 <- "Results/4/WestLafayette_City_region1_ori_sectored_kernel_only_intensity.rds"
text10.6 <- "Results/4/WestLafayette_City_region1_ori_kernel_only_intensity.rds"
text10.7 <- "Results/4/WestLafayette_City_region1_ori_sectored_kernel_only_intensity_and_height.rds"
text10.8 <- "Results/4/WestLafayette_City_region1_ori_kernel_only_intensity_and_height.rds"

saveRDS(result_om_4_1, text6.1)
saveRDS(result_om_4_2, text6.2)
saveRDS(result_om_5_1, text6.3)
saveRDS(result_om_5_2, text6.4)
saveRDS(result_om_4_1_i, text6.5)
saveRDS(result_om_4_2_i, text6.6)
saveRDS(result_om_5_1_i, text6.7)
saveRDS(result_om_5_2_i, text6.8)

saveRDS(result_csf_4_1, text7.1)
saveRDS(result_csf_4_2, text7.2)
saveRDS(result_csf_5_1, text7.3)
saveRDS(result_csf_5_2, text7.4)
saveRDS(result_csf_4_1_i, text7.5)
saveRDS(result_csf_4_2_i, text7.6)
saveRDS(result_csf_5_1_i, text7.7)
saveRDS(result_csf_5_2_i, text7.8)

saveRDS(result_pmf_4_1, text8.1)
saveRDS(result_pmf_4_2, text8.2)
saveRDS(result_pmf_5_1, text8.3)
saveRDS(result_pmf_5_2, text8.4)
saveRDS(result_pmf_4_1_i, text8.5)
saveRDS(result_pmf_4_2_i, text8.6)
saveRDS(result_pmf_5_1_i, text8.7)
saveRDS(result_pmf_5_2_i, text8.8)

saveRDS(result_mcc_4_1, text9.1)
saveRDS(result_mcc_4_2, text9.2)
saveRDS(result_mcc_5_1, text9.3)
saveRDS(result_mcc_5_2, text9.4)
saveRDS(result_mcc_4_1_i, text9.5)
saveRDS(result_mcc_4_2_i, text9.6)
saveRDS(result_mcc_5_1_i, text9.7)
saveRDS(result_mcc_5_2_i, text9.8)

saveRDS(result_ori_4_1, text10.1)
saveRDS(result_ori_4_2, text10.2)
saveRDS(result_ori_5_1, text10.3)
saveRDS(result_ori_5_2, text10.4)
saveRDS(result_ori_4_1_i, text10.5)
saveRDS(result_ori_4_2_i, text10.6)
saveRDS(result_ori_5_1_i, text10.7)
saveRDS(result_ori_5_2_i, text10.8)

