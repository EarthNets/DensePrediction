#!/bin/bash

#bash ./tools/dist_train.sh configs/simipu/simipu_r50_scratch_nyu_24e.py 2 --work-dir work_dir/logs/scratch_nyu
bash ./tools/dist_train.sh configs/depthformer/depthformer_swint_w7_nyu.py 2 --work-dir work_dir/logs/depthformer
#bash ./tools/dist_train.sh configs/SimIPU/SimIPU_r50_supervise_imagenet_nyu.py 2 --work-dir nfs/saves/SimIPU/supervise_imagenet_nyu
#bash ./tools/dist_train.sh configs/SimIPU/SimIPU_r50_kitti_50e_nyu.py 2 --work-dir nfs/saves/SimIPU/kitti_50e_nyu
#bash ./tools/dist_train.sh configs/SimIPU/SimIPU_r50_waymo_50e_nyu.py 2 --work-dir nfs/saves/SimIPU/waymo_50e_nyu

#bash ./tools/dist_train.sh configs/SimIPU/SimIPU_r50_kitti_50e_kitti.py 2 --work-dir nfs/saves/SimIPU/kitti_50e_kitti
#bash ./tools/dist_train.sh configs/SimIPU/SimIPU_r50_waymo_50e_kitti.py 2 --work-dir nfs/saves/SimIPU/waymo_50e_kitti





