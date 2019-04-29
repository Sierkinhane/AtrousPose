# AtrousPose
codes for paper：real time multi-person pose estimation based on atrous convolution

## Network Structures
![](https://github.com/Sierkinhane/AtrousPose/blob/master/images/basicNet2.png)
## Spatial pyramid network for prediction
![](https://github.com/Sierkinhane/AtrousPose/blob/master/images/spatialpyramid.png)

## Properties of atrouspose network
| Arch               |Input Size|Output Size| FLOPS |Num.param.|  FPS(C++)  |
|--------------------|----------|-----------|-------|----------|------------|
| AtrousPose-512     |  384×384 |   48×48   |  50G  |    26M   |     42     |
| AtrousPose-128     |  384×384 |   48×48   |  20G  |    13M   |     67     |

## Test Results
![](https://github.com/Sierkinhane/AtrousPose/blob/master/images/demo2.png)

## Evaluation

## Demo
   * (Python demo)run > demo.py results will be store in /images
   * (C++ version) 
   
## Training
   * download mpii dataset
      * [MPII](http://human-pose.mpi-inf.mpg.de/)
   * download mask for mpii
      * [MASK](http://posefs1.perception.cs.cmu.edu/Users/ZheCao/masks_for_mpii_pose.tgz)
   * download json for mpii
      * [JSON](http://posefs1.perception.cs.cmu.edu/Users/ZheCao/MPI.json)
   * train
      * run > train.py
### Loss curve
![](https://github.com/Sierkinhane/AtrousPose/blob/master/loss_log/loss.jpg)
### This repository was based on:
   * [ZheCao](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
   * [Tensorboy](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)
   * [last-one](https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/tree/master/training)
   * [michalfaber](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
   * [fregu856](https://github.com/fregu856/deeplabv3)
