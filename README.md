# GRNet: Geometric Relation Network for 3D Object Detection from Point Clouds
Created by Li, Ying and Ma, Lingfei and Tan, Weikai and Sun, Chen and Cao, Dongpu and Li, Jonathan from University of Waterloo


# Install
Install Pytorch and Tensorflow (for TensorBoard). It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Ubuntu 16.04, Pytorch v1.0, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4. 

Compile the CUDA layers for PointNet++, which we used in the backbone network:

    cd pointnet2
    python setup.py install
To see if the compilation is successful, try to run python models/votenet.py to see if a forward pass works.

Install the following Python dependencies (with pip install):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

## Prepare SUN RGB-D Data

For SUN RGB-D, follow the VoteNet https://github.com/facebookresearch/votenet/tree/main/sunrgbd to prepare it:

Download SUNRGBD v2 data HERE http://rgbd.cs.princeton.edu/data/ (SUNRGBD.zip, SUNRGBDMeta2DBB_v2.mat, SUNRGBDMeta3DBB_v2.mat) and the toolkits (SUNRGBDtoolbox.zip). Move all the downloaded files under OFFICIAL_SUNRGBD. Unzip the zip files.

Extract point clouds and annotations (class, v2 2D -- xmin,ymin,xmax,ymax, and 3D bounding boxes -- centroids, size, 2D heading) by running extract_split.m, extract_rgbd_data_v2.m and extract_rgbd_data_v1.m under the matlab folder.

Prepare data by running 
    python3 sunrgbd_data.py --gen_v1_data

You can also examine and visualize the data with python sunrgbd_data.py --viz and use MeshLab to view the generated PLY files at data_viz_dump.

NOTE: SUNRGBDtoolbox.zip should have MD5 hash 18d22e1761d36352f37232cba102f91f (you can check the hash with md5 SUNRGBDtoolbox.zip on Mac OS or md5sum SUNRGBDtoolbox.zip on Linux)

## Train and Test
To train a new VoteNet model on SUN RGB-D data (depth images):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd
    
To test the trained model with its checkpoint:

    python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
    
Final evaluation results will be printed on screen and also written in the log_eval.txt file under the dump directory. In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on oriented boxes. A properly trained VoteNet should have around 57 mAP@0.25 and 32 mAP@0.5.


```
@article{li2020grnet,
  title={GRNet: Geometric relation network for 3D object detection from point clouds},
  author={Li, Ying and Ma, Lingfei and Tan, Weikai and Sun, Chen and Cao, Dongpu and Li, Jonathan},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={165},
  pages={43--53},
  year={2020},
  publisher={Elsevier}
}

```
