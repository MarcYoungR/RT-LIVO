# RT-LIVO

## RT-LIVO: Real-Time LiDAR-Inertial-Visual Odometry with Dynamic Object Removal

**A modified version of FAST-LIVO2 with RT-DETR integration for real-time dynamic object detection and removal.**

### 📢 News

- 🔓 **2025-01-23**: Code released!
- 🎉 **2024-10-01**: FAST-LIVO2 accepted by **T-RO '24**!

### 📬 Contact

- **Original FAST-LIVO2**: [zhengcr@connect.hku.hk](mailto:zhengcr@connect.hku.hk)
- **RT-LIVO Modifications**: Based on FAST-LIVO2 with RT-DETR integration

## 1. Introduction

RT-LIVO is built upon FAST-LIVO2, an efficient and accurate LiDAR-inertial-visual fusion localization and mapping system. This version adds real-time dynamic object detection capabilities using RT-DETR, enabling the system to detect and remove dynamic objects (e.g., people, vehicles) from the point cloud map.

**Original Developer**: [Chunran Zheng 郑纯然](https://github.com/xuankuzcr)

<div align="center">
    <img src="pics/Framework.png" width = 100% >
</div>

### 1.1 Related video

Our accompanying video is now available on [**Bilibili**](https://www.bilibili.com/video/BV1Ezxge7EEi) and [**YouTube**](https://youtu.be/6dF2DzgbtlY).

### 1.2 Related paper

[FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/pdf/2408.14035)  

[FAST-LIVO2 on Resource-Constrained Platforms](https://arxiv.org/pdf/2501.13876)  

[FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/pdf/2203.00893)

[FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second](https://www.arxiv.org/pdf/2507.17210)

### 1.3 Our hard-synchronized equipment

We open-source our handheld device, including CAD files, synchronization scheme, STM32 source code, wiring instructions, and sensor ROS driver. Access these resources at this repository: [**LIV_handhold**](https://github.com/xuankuzcr/LIV_handhold).

### 1.4 Our associate dataset: FAST-LIVO2-Dataset
Our associate dataset [**FAST-LIVO2-Dataset**](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/ErdFNQtjMxZOorYKDTtK4ugBkogXfq1OfDm90GECouuIQA?e=KngY9Z) used for evaluation is also available online.

### 1.5 Our LiDAR-camera calibration method
The [**FAST-Calib**](https://github.com/hku-mars/FAST-Calib) toolkit is recommended. Its output extrinsic parameters can be directly filled into the YAML file.

### 1.6 RT-DETR Integration

This version integrates RT-DETR (Real-Time Detection Transformer) for dynamic object detection:

- **Model**: RT-DETR ONNX model (place in `weights/` directory)
- **Configuration**: See `config/avia.yaml` for RT-DETR parameters
- **Key Features**:
  - Real-time object detection from camera images
  - Dynamic point cloud filtering based on detection masks
  - Configurable confidence threshold and padding
  - Support for filtering specific object classes

**Note**: Proper extrinsic calibration between LiDAR and camera is critical for accurate point cloud projection. Use FAST-Calib or similar tools for calibration.

## 2. Prerequisites

### 2.1 Ubuntu and ROS

Ubuntu 18.04~20.04.  [ROS Installation](http://wiki.ros.org/ROS/Installation).

### 2.2 PCL && Eigen && OpenCV

PCL>=1.8, Follow [PCL Installation](https://pointclouds.org/). 

Eigen>=3.3.4, Follow [Eigen Installation](https://eigen.tuxfamily.org/index.php?title=Main_Page).

OpenCV>=4.2, Follow [Opencv Installation](http://opencv.org/).

### 2.3 ONNX Runtime

Required for RT-DETR inference. Download and install ONNX Runtime:

```bash
# For GPU support (recommended)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.23.2.tgz
sudo mv onnxruntime-linux-x64-gpu-1.23.2 /opt/onnxruntime

# For CPU only
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
```

**Important**: Update the `ONNXRUNTIME_DIR` path in `CMakeLists.txt` to match your installation location.

### 2.4 Sophus

Sophus Installation for the non-templated/double-only version.

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build && cmake ..
make
sudo make install
```

### 2.5 Vikit

Vikit contains camera models, some math and interpolation functions that we need. Vikit is a catkin project, therefore, download it into your catkin workspace source folder.

```bash
# Different from the one used in fast-livo1
cd catkin_ws/src
git clone https://github.com/xuankuzcr/rpg_vikit.git 
```

## 3. Build

Clone the repository and catkin_make:

```
cd ~/catkin_ws/src
git clone https://github.com/hku-mars/FAST-LIVO2.git rt_livo
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

**Note**: This repository is named `rt_livo` to distinguish it from the original FAST-LIVO2.

## 4. Run our examples

### 4.1 Prepare RT-DETR Model

Download RT-DETR ONNX model and place it in the `weights/` directory:

```bash
cd ~/catkin_ws/src/rt_livo
mkdir -p weights
# Download RT-DETR model (e.g., rtdetr_r50vd_6x_coco.onnx)
# Place the model file as weights/model.onnx
```

### 4.2 Configure Extrinsic Parameters

Edit `config/avia.yaml` and update the `Rcl` and `Pcl` parameters with your calibrated extrinsic values:

```yaml
Rcl: [rotation_matrix_elements]
Pcl: [translation_vector_elements]
```

### 4.3 Launch the System

Download our collected rosbag files via OneDrive ([**FAST-LIVO2-Dataset**](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhengcr_connect_hku_hk/ErdFNQtjMxZOorYKDTtK4ugBkogXfq1OfDm90GECouuIQA?e=KngY9Z)).

```
roslaunch rt_livo mapping_avia.launch
rosbag play YOUR_DOWNLOADED.bag
```


## 5. License

The source code of this package is released under the [**GPLv2**](http://www.gnu.org/licenses/) license. For commercial use, please contact me at <zhengcr@connect.hku.hk> and Prof. Fu Zhang at <fuzhang@hku.hk> to discuss an alternative license.