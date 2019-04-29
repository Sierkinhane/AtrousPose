# AtrousPose
codes for paper：real time multi-person pose estimation based on atrous convolution

## Network Structures
![](https://github.com/Sierkinhane/AtrousPose/blob/master/images/basicNet2.png)
## Spatial pyramid network for prediction
![](https://github.com/Sierkinhane/AtrousPose/blob/master/images/spatialpyramid.png)
### Properties of atrouspose network
|--------------|----------|-----------|-------|-----------|-----|
| Architecture |Input Size|Output Size| FLOPS |Num. param.| FPS |
|AtrousPose-512|  384×384 |   48×48   |  50G  |     26M   |42FPS|
|AtrousPose-128|  384×384 |   48×48   |  20G  |     13M   |67FPS|

### Results on MPII val
| Arch               | Head | Shoulder | Elbow | Wrist |  Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|------|----------|-------|-------|------|------|-------|------|----------|
| pose_resnet_50     | 96.4 |     95.3 |  89.0 |  83.2 | 88.4 | 84.0 |  79.6 | 88.5 |     34.0 |
| pose_resnet_101    | 96.9 |     95.9 |  89.5 |  84.4 | 88.4 | 84.5 |  80.7 | 89.1 |     34.0 |
| pose_resnet_152    | 97.0 |     95.9 |  90.0 |  85.0 | 89.2 | 85.3 |  81.3 | 89.6 |     35.0 |
| **pose_hrnet_w32** | 97.1 |     95.9 |  90.3 |  86.4 | 89.1 | 87.1 |  83.3 | 90.3 |     37.7 |

### Note:

###
## Test Results
![](https://github.com/Sierkinhane/AtrousPose/blob/master/images/demo2.png)
