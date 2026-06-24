/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "vio.h"
#include "preprocess.h"
#include "RTDETRDetector.h"
#include "MotionVerifier.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include <vikit/camera_loader.h>

class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it);
  void initializeComponents();
  void initializeFiles();
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
 
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void img_cbk(const sensor_msgs::ImageConstPtr &msg_in);
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_visual_sub_map(const ros::Publisher &pubSubVisualMap);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher pubPath);
  void readParameters(ros::NodeHandle &nh);
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  double outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  deque<cv::Mat> img_buffer;
  deque<double> img_time_buffer;
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  double IMG_POINT_COV;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr feats_removed_accumulated_;  // 累积的被剔除点云，用于同步发布

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Subscriber sub_img;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  image_transport::Publisher pubImage;
  ros::Publisher pubRtdetrDebugImg;     // 调试图像: 相机图上叠加 红点(删除)/白点(保留)/检测框
  ros::Publisher mavros_pose_publisher;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;

  RTDETRDetector* detector_ = nullptr;

  // 运动验证模块 (RT-DETR 输出 -> 语义动态候选 -> 运动验证)
  MotionVerifier* motion_verifier_ = nullptr;

  // 语义动态候选列表 (相机模型坐标系)
  std::vector<SemanticCandidate> semantic_candidates_;
  // 上一帧候选 (用于跨帧关联 + 时序状态保持)
  std::vector<SemanticCandidate> tracked_candidates_;

  // 掩码图像 (0=动态/剔除, 255=静态/保留)
  cv::Mat current_mask_;

  // --- 配置参数 (将在 YAML 中读取) ---
  bool rtdetr_en = false;              // 开关 (对应 yaml: rtdetr/enable)
  double rtdetr_conf_thresh = 0.45;    // 阈值 (对应 yaml: rtdetr/conf_thresh)
  int rtdetr_padding = 10;             // 固定 padding (fallback)
  std::string rtdetr_model_name;       // 模型名 (对应 yaml: rtdetr/model_name)
  double rtdetr_time_offset = 0.0;     // 时间戳偏移(秒)

  // --- 运动验证参数 (motion_verify) ---
  bool motion_verify_en = true;
  int  motion_verify_min_points = 12;
  double motion_verify_high_thresh = 1.5;
  double motion_verify_medium_thresh = 3.0;
  double motion_verify_low_thresh = 5.0;

  // --- 运动状态时序平滑 (抑制红/绿框频繁切换导致的地图残留) ---
  bool   motion_temporal_smooth_en = true; // 跨帧关联 + MOVING 状态保持
  int    motion_hold_frames = 5;           // 判过 MOVING 后持续保持的帧数
  double assoc_iou_min = 0.3;              // 跨帧候选匹配的 IoU 下限

  // --- 自适应 padding 参数 (adaptive_padding) ---
  bool   adaptive_padding_en = true;
  int    padding_min_px = 4;
  int    padding_max_px = 40;
  double padding_base_px = 3.0;
  double padding_vmax = 3.0;
  double padding_flow_gain = 0.5;
  double detection_time_delay = 0.05;

  // --- 深度门控参数 (depth_gate) ---
  bool   depth_gate_en = true;
  double depth_gate_min = 0.3;
  double depth_gate_ratio = 0.12;

  // --- 旧版自适应协方差 (adaptive_cov, 已弃用为默认关闭) ---
  bool adaptive_img_cov_en = false;
  double adaptive_img_cov_min = 200.0;
  double adaptive_img_cov_max = 2000.0;

  // --- 可靠性感知协方差 (reliability_cov) ---
  bool   reliability_cov_en = true;
  double reliability_cov_min = 200.0;
  double reliability_cov_max = 2000.0;
  int    expected_visual_points = 80;

  // --- 功能函数 ---
  void InitRTDETR(ros::NodeHandle &nh);   // 初始化：读取YAML并加载模型
  void DetectAndMask(const cv::Mat& img); // 核心：检测 + 运动验证 + 生成Mask (不修改原图)

  // 自适应 padding: 综合深度 / 时间延迟 / 运动得分
  int computeAdaptivePadding(double median_depth, double motion_score);

  // 基于语义候选 + 深度门控判断一个 LiDAR 点是否为动态点
  bool isDynamicPointByCandidate(const Eigen::Vector3d& p_cam, const Eigen::Vector2d& uv);

  // LIO 过滤前, 用当前 body 点云更新每个候选目标的深度中值
  void updateCandidateDepthFromCloud(const PointCloudXYZI::Ptr& cloud_body);

  // 可靠性感知的视觉协方差更新
  void updateVisualReliabilityCovariance();

  // 运动状态时序平滑: 跨帧 IoU 关联 + MOVING 状态保持, 抑制单帧闪烁
  void applyTemporalSmoothing();
  // ==========================================

};
#endif