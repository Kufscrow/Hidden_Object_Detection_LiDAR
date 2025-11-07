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

df_temp_2 <- data.frame(temp_2, las$Classification[indice_base])
names(df_temp_2) <- c("X", "Y", "Z", "Classification")
las.patch.base <- LAS(df_temp_2)

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

detect_house_after_ground <- function(ng_patch, pts_cand=50){
  ng_patch <- segment_shapes(ng_patch, shp_plane(k = 12), "is_plane")
  ng_patch <- segment_shapes(ng_patch, shp_hplane(k = 12), "is_hplane")
  ng_patch <- segment_shapes(ng_patch, shp_line(k = 12), "is_line")
  planar_patch_pts <- filter_poi(ng_patch, (is_plane == TRUE)|(is_hplane == TRUE)|(is_line == TRUE))
  planar_patch_clusters <- segment_shapes(planar_patch_pts, shp_plane(k = 12),
                                          attribute = "plane2")$plane2
  planar_patch_pts <- add_attribute(planar_patch_pts, planar_patch_clusters, "clusterID")
  cluster_patch_summary <- planar_patch_pts@data %>%
    dplyr::group_by(clusterID) %>%
    dplyr::summarise(
      n_pts = dplyr::n(),
      mean_z = mean(Z, na.rm = TRUE),
      sd_z   = sd(Z, na.rm = TRUE)
    )
  valid_patch_clusters <- cluster_patch_summary$clusterID[
    (cluster_patch_summary$n_pts > min(cluster_patch_summary$n_pts)+pts_cand)
  ]
  candidates_patch <- planar_patch_pts$clusterID %in% valid_patch_clusters
  planar_nonground_ind <- which((ng_patch$is_plane==TRUE)|(ng_patch$is_hplane==TRUE)|(ng_patch$is_line==TRUE))
  return(list("candidate"=candidates_patch, "planar_ind"=planar_nonground_ind))
}

detect_house_after_ground_inv <- function(ng_patch, pts_cand=50){
  ng_patch <- segment_shapes(ng_patch, shp_plane(k = 12), "is_plane")
  ng_patch <- segment_shapes(ng_patch, shp_hplane(k = 12), "is_hplane")
  ng_patch <- segment_shapes(ng_patch, shp_line(k = 12), "is_line")
  planar_patch_pts <- filter_poi(ng_patch, (is_plane == TRUE)|(is_hplane == TRUE)|(is_line == TRUE))
  planar_patch_clusters <- segment_shapes(planar_patch_pts, shp_plane(k = 12),
                                          attribute = "plane2")$plane2
  planar_patch_pts <- add_attribute(planar_patch_pts, planar_patch_clusters, "clusterID")
  cluster_patch_summary <- planar_patch_pts@data %>%
    dplyr::group_by(clusterID) %>%
    dplyr::summarise(
      n_pts = dplyr::n(),
      mean_z = mean(Z, na.rm = TRUE),
      sd_z   = sd(Z, na.rm = TRUE)
    )
  valid_patch_clusters <- cluster_patch_summary$clusterID[
    (cluster_patch_summary$n_pts < min(cluster_patch_summary$n_pts)+pts_cand)
  ]
  candidates_patch <- planar_patch_pts$clusterID %in% valid_patch_clusters
  planar_nonground_ind <- which((ng_patch$is_plane==TRUE)|(ng_patch$is_hplane==TRUE)|(ng_patch$is_line==TRUE))
  return(list("candidate"=candidates_patch, "planar_ind"=planar_nonground_ind))
}

las_patch_ori <- las.patch.base
ng_patch_ori <- filter_poi(las_patch_ori, Classification!=2L)
plot(las_patch_ori,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

las_patch_csf <- classify_ground(las.patch.base, algorithm = csf(sloop_smooth = TRUE, class_threshold = 1, cloth_resolution = 1, time_step = 1))
ng_patch_csf <- filter_poi(las_patch_csf, Classification != 2L)
plot(las_patch_csf,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

las_patch_pmf <- classify_ground(las.patch.base, algorithm = pmf(ws = 5, th = 3))
ng_patch_pmf <- filter_poi(las_patch_pmf, Classification != 2L)
plot(las_patch_pmf,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

las_patch_mcc <- classify_ground(las.patch.base, algorithm = mcc(1.5,0.3))
ng_patch_mcc <- filter_poi(las_patch_mcc, Classification != 2L)
plot(las_patch_mcc,color ="Classification",bg = "black",pal=c("Yellow","Red"),axis=F)

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
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[result.patch.2], decorate=(box=FALSE))

temp.patch.2.nonground <- temp_2[which(result.patch.2==1),]
df_nonground_2_om <- data.frame(temp.patch.2.nonground)
names(df_nonground_2_om) <- c("X", "Y", "Z")
ng_patch_om <- LAS(df_nonground_2_om)

csf_other_result <- detect_house_after_ground(ng_patch_csf)
temp_2_nonground_csf_ind <- which(las_patch_csf$Classification==1)

pmf_other_result <- detect_house_after_ground_inv(ng_patch_pmf)
temp_2_nonground_pmf_ind <- which(las_patch_pmf$Classification==1)

mcc_other_result <- detect_house_after_ground(ng_patch_mcc)
temp_2_nonground_mcc_ind <- which(las_patch_mcc$Classification==1)

om_other_result <- detect_house_after_ground(ng_patch_om)
temp_2_nonground_om_ind <- which(result.patch.2==1)

ori_other_result <- detect_house_after_ground(ng_patch_ori)
temp_2_nonground_ori_ind <- which(las_patch_ori$Classification==1)

final_result_csf <- rep(2, length(indice_base))
final_result_csf[temp_2_nonground_csf_ind] <- 1
final_result_csf[temp_2_nonground_csf_ind[csf_other_result$planar_ind]] <- 
  2*as.numeric(csf_other_result$candidate)+1

final_result_pmf <- rep(2, length(indice_base))
final_result_pmf[temp_2_nonground_pmf_ind] <- 1
final_result_pmf[temp_2_nonground_pmf_ind[pmf_other_result$planar_ind]] <- 
  2*as.numeric(pmf_other_result$candidate)+1

final_result_mcc <- rep(2, length(indice_base))
final_result_mcc[temp_2_nonground_mcc_ind] <- 1
final_result_mcc[temp_2_nonground_mcc_ind[mcc_other_result$planar_ind]] <- 
  2*as.numeric(mcc_other_result$candidate)+1

final_result_om <- rep(2, length(indice_base))
final_result_om[temp_2_nonground_om_ind] <- 1
final_result_om[temp_2_nonground_om_ind[om_other_result$planar_ind]] <- 
  2*as.numeric(om_other_result$candidate)+1

final_result_ori <- rep(2, length(indice_base))
final_result_ori[temp_2_nonground_ori_ind] <- 1
final_result_ori[temp_2_nonground_ori_ind[ori_other_result$planar_ind]] <- 
  2*as.numeric(ori_other_result$candidate)+1

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[final_result_csf], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[final_result_pmf], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[final_result_mcc], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[final_result_om], decorate=(box=FALSE))
plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[final_result_ori], decorate=(box=FALSE))

text1 <- "Results/4/WestLafayette_City_region1_csf_tracluster.rds"
text2 <- "Results/4/WestLafayette_City_region1_pmf_tracluster.rds"
text3 <- "Results/4/WestLafayette_City_region1_mcc_tracluster.rds"
text4 <- "Results/4/WestLafayette_City_region1_h(1,4)_om_tracluster.rds"
text5 <- "Results/4/WestLafayette_City_region1_ori_trancluster.rds"
