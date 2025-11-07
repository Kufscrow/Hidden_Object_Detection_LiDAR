# Hidden_Object_Detection_LiDAR
This project contains R codes that is used for hidden object detection from LiDAR datasets representing WUI/forested/city areas.

There are two example scenarios studied in this object, one WUI and one city area. The satelitte maps are shown in pic14.png (WUI) and pic1_c.png (City).

All the methods are unsupervised two-stage method, one being ground filtering, and one being detection in non-ground points. Detailed methods used for each code are shown in the following table. Here "Default" means the two files names "SouthernIndiana_WUI_V3_14" and "WestLafayette_Place1_V1_1"

## 
| Code Names | Ground Filtering | Object Detection |
|-----------------|-----------------|-----------------|
| **Default.r**   | [HKGF](Pending publication) | [LIE_V2](Pending publication)  |
| **Default_1.r** | [HKGF](Pending publication), [CSF](https://www.mdpi.com/2072-4292/8/6/501),[MCC](https://ieeexplore.ieee.org/document/4137852), [PMF](https://ieeexplore.ieee.org/document/1202973) | [NPD](https://doi.org/10.1016/j.patcog.2014.12.020) |
| **Default_2.r** | [CSF](https://www.mdpi.com/2072-4292/8/6/501),[MCC](https://ieeexplore.ieee.org/document/4137852), [PMF](https://ieeexplore.ieee.org/document/1202973) | [LIE_V2](Pending publication) |
| **Default_3.r** | [CSF](https://www.mdpi.com/2072-4292/8/6/501),[MCC](https://ieeexplore.ieee.org/document/4137852), [PMF](https://ieeexplore.ieee.org/document/1202973) | [LIE_V1](https://ieeexplore.ieee.org/document/10825112) |
| **Default_OSR_1.r** | [OSR](https://ieeexplore.ieee.org/document/10825112) | [NPD](https://doi.org/10.1016/j.patcog.2014.12.020) |
| **Default_OSR_2.r** | [OSR](https://ieeexplore.ieee.org/document/10825112) | [LIE_V2](Pending publication) |
| **Default_OSR_3.r** | [HKGF](Pending publication), [CSF](https://www.mdpi.com/2072-4292/8/6/501),[MCC](https://ieeexplore.ieee.org/document/4137852), [PMF](https://ieeexplore.ieee.org/document/1202973) | [LIE_V1](https://ieeexplore.ieee.org/document/10825112) |



