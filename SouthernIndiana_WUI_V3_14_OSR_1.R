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

result.patch.osr <- readRDS("Results/4/SouthernIndiana_WUI_region14_OSR.rds")

temp.patch.osr.nonground <- temp_2[which(result.patch.osr==1),]
df_nonground_2_osr <- data.frame(temp.patch.osr.nonground)
names(df_nonground_2_osr) <- c("X", "Y", "Z")
ng_patch_osr <- LAS(df_nonground_2_osr)

osr_other_result <- detect_house_after_ground_inv(ng_patch_osr)
temp_2_nonground_osr_ind <- which(result.patch.osr==1)

final_result_osr <- rep(2, length(indice_base))
final_result_osr[temp_2_nonground_osr_ind] <- 1
final_result_osr[temp_2_nonground_osr_ind[osr_other_result$planar_ind]] <- 
  2*as.numeric(osr_other_result$candidate)+1

plot3d(temp_2, aspect=c(1, 1, 0.1), col=label_to_color_1[final_result_osr], decorate=(box=FALSE))

text6 <- "Results/4/SouthernIndiana_WUI_region14_osr_trancluster.rds"
