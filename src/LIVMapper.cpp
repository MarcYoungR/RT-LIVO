/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include <ros/package.h>

#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <algorithm>

ros::Publisher pub_deleted_points;

LIVMapper::LIVMapper(ros::NodeHandle &nh)
    : extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);

  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  InitRTDETR(nh);

  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents();
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper() {}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/img_en", img_en, 1);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<string>("common/img_topic", img_topic, "/left_camera/image");

  nh.param<bool>("vio/normal_en", normal_en, true);
  nh.param<bool>("vio/inverse_composition_en", inverse_composition_en, false);
  nh.param<int>("vio/max_iterations", max_iterations, 5);
  nh.param<double>("vio/img_point_cov", IMG_POINT_COV, 100);
  nh.param<bool>("vio/raycast_en", raycast_en, false);
  nh.param<bool>("vio/exposure_estimate_en", exposure_estimate_en, true);
  nh.param<double>("vio/inv_expo_cov", inv_expo_cov, 0.2);
  nh.param<int>("vio/grid_size", grid_size, 5);
  nh.param<int>("vio/grid_n_height", grid_n_height, 17);
  nh.param<int>("vio/patch_pyrimid_level", patch_pyrimid_level, 3);
  nh.param<int>("vio/patch_size", patch_size, 8);
  nh.param<double>("vio/outlier_threshold", outlier_threshold, 1000);

  nh.param<double>("time_offset/exposure_time_init", exposure_time_init, 0.0);
  nh.param<double>("time_offset/img_time_offset", img_time_offset, 0.0);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<bool>("pcd_save/colmap_output_en", colmap_output_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("extrin_calib/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/Rcl", cameraextrinR, vector<double>());
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  if (!vk::camera_loader::loadFromRosNs("laserMapping", vio_manager->cam)) throw std::runtime_error("Camera model not correctly specified.");

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->setImuToLidarExtrinsic(extT, extR);
  vio_manager->setLidarToCameraExtrinsic(cameraextrinR, cameraextrinT);
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->initializeVIO();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh, image_transport::ImageTransport &it) 
{
  sub_pcl = p_pre->lidar_type == AVIA ? 
            nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  sub_img = nh.subscribe(img_topic, 200000, &LIVMapper::img_cbk, this);
  
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  pubNormal = nh.advertise<visualization_msgs::MarkerArray>("visualization_marker", 100);
  pubSubVisualMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  plane_pub = nh.advertise<visualization_msgs::Marker>("/planner_normal", 1);
  voxel_pub = nh.advertise<visualization_msgs::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_removed", 100);
  pubLaserCloudDynDbg = nh.advertise<sensor_msgs::PointCloud2>("/dyn_obj_dbg_hist", 100);

  pub_deleted_points = nh.advertise<sensor_msgs::PointCloud2>("/rtdetr_deleted_points", 100);

  pubRtdetrDebugImg = nh.advertise<sensor_msgs::Image>("/rtdetr_debug_img", 1);

  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImage = it.advertise("/rgb_img", 1);
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);
  voxelmap_manager->voxel_map_pub_= nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  switch (LidarMeasures.lio_vio_flg) 
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}

void LIVMapper::handleVIO() 
{
  euler_cur = RotMtoEuler(_state.rot_end);
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;
    
  if (pcl_w_wait_pub->empty() || (pcl_w_wait_pub == nullptr)) 
  {
    std::cout << "[ VIO ] No point!!!" << std::endl;
    return;
  }
    
  std::cout << "[ VIO ] Raw feature num: " << pcl_w_wait_pub->points.size() << std::endl;

  if (fabs((LidarMeasures.last_lio_update_time - _first_lidar_time) - plot_time) < (frame_cnt / 2 * 0.1)) 
  {
    vio_manager->plot_flag = true;
  } 
  else 
  {
    vio_manager->plot_flag = false;
  }

  // 1. 传递 Mask 指针
  if (rtdetr_en) {
      vio_manager->dynamic_mask = &current_mask_;
  }

  // -------------------------------------------------------------------
  // [修复版] 过滤 VIO 视觉特征点
  // -------------------------------------------------------------------
  // -------------------------------------------------------------------
  // [修复版] 过滤 VIO 视觉特征点 (带空指针检查)
  // -------------------------------------------------------------------
  if (rtdetr_en && !current_mask_.empty() && !pcl_w_wait_pub->empty()) 
  {
      // 【关键修复】如果 VIO 还没初始化(new_frame_为空)，直接跳过过滤，防止崩溃
      if (vio_manager->new_frame_ == nullptr) {
          // ROS_WARN_THROTTLE(1.0, "[VIO] Frame not ready yet, skipping filter.");
      } 
      else 
      {
          PointCloudXYZI::Ptr feats_vio_clean(new PointCloudXYZI());
          
          for (const auto& pt : pcl_w_wait_pub->points) 
          {
              V3D p_w(pt.x, pt.y, pt.z);
              
              // 1. 安全检查通过后再调用
              V3D p_c = vio_manager->new_frame_->w2f(p_w); 

              bool keep = true;
              if (p_c(2) > 0.1) 
              {
                  V2D uv = vio_manager->new_frame_->cam_->world2cam(p_c);
                  
                  if (uv(0) >= 0 && uv(0) < current_mask_.cols && 
                      uv(1) >= 0 && uv(1) < current_mask_.rows) 
                  {
                      if (current_mask_.at<uchar>((int)uv(1), (int)uv(0)) == 0) {
                          keep = false; 
                      }
                  }
              }
              if (keep) feats_vio_clean->points.push_back(pt);
          }
          *pcl_w_wait_pub = *feats_vio_clean; 
      }
  }
  // -------------------------------------------------------------------

  // 可靠性感知视觉协方差: 在 VIO 更新前设置当前帧视觉协方差
  // (使用上一帧 total_points 近似; 若 reliability_cov 关闭则不修改 img_point_cov)
  if (rtdetr_en && reliability_cov_en) {
      updateVisualReliabilityCovariance();
  }

  vio_manager->processFrame(LidarMeasures.measures.back().img, _pv_list, voxelmap_manager->voxel_map_, LidarMeasures.last_lio_update_time - _first_lidar_time);

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  publish_frame_world(pubLaserCloudFullRes, vio_manager);
  publish_img_rgb(pubImage, vio_manager);

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::handleLIO()
{
  euler_cur = RotMtoEuler(_state.rot_end);
  // ... (保留原本的日志打印代码)
  fout_pre << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << std::endl;

  // ==========================================
  // [关键修复] Mask已在主循环中更新（行703），这里只需要验证
  // ==========================================
  if (rtdetr_en) {
      if (!current_mask_.empty()) {
          double mean_val = cv::mean(current_mask_)[0];
          ROS_DEBUG("[LIO] Using current_mask: mean=%.2f, size=%dx%d", mean_val, current_mask_.rows, current_mask_.cols);
      } else {
          ROS_WARN_THROTTLE(1.0, "[LIO] current_mask_ is EMPTY! Check if image was processed in main loop.");
      }
  }
  // ==========================================

  if (feats_undistort->empty() || (feats_undistort == nullptr))
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();
  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);

  // ==========================================
  // [终极修正版] LIO 动态剔除 + 鬼影消除 + 红点调试
  // ==========================================
  if (rtdetr_en && !current_mask_.empty())
  {
      PointCloudXYZI::Ptr feats_static(new PointCloudXYZI());
      // 用于调试显示的红色点云
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr feats_removed_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
      feats_static->reserve(feats_down_body->size());

      Eigen::Matrix3d R_bc;
      R_bc << MAT_FROM_ARRAY(cameraextrinR);
      Eigen::Vector3d T_bc;
      T_bc << VEC_FROM_ARRAY(cameraextrinT);

      // 过滤前: 用当前 body 点云更新每个候选目标的深度中值 (供深度门控使用)
      updateCandidateDepthFromCloud(feats_down_body);

      int before_num = feats_down_body->points.size();
      int kept_points = 0;

      // 调试图像: 在相机图上叠加 删除点(红) / 保留点(白) / 检测框(按状态着色)
      int cam_w = vio_manager->cam->width();
      int cam_h = vio_manager->cam->height();
      cv::Mat debug_img;
      bool build_dbg = rtdetr_en && !LidarMeasures.measures.back().img.empty();
      if (build_dbg) {
          // 源图缩放到相机模型分辨率, 与 world2cam 投影 (uv) / expanded_box 同坐标系
          cv::resize(LidarMeasures.measures.back().img, debug_img, cv::Size(cam_w, cam_h));
          for (const auto& c : semantic_candidates_) {
              cv::Scalar col(0, 255, 255); // UNCERTAIN=黄
              if (c.state == MotionState::MOVING_OBJECT)      col = cv::Scalar(0, 0, 255);   // 红
              else if (c.state == MotionState::STATIC_OBJECT) col = cv::Scalar(0, 255, 0);   // 绿
              cv::rectangle(debug_img, c.expanded_box, col, 2);
          }
      }

      for (const auto& pt : feats_down_body->points)
      {
          Eigen::Vector3d p_body(pt.x, pt.y, pt.z);
          Eigen::Vector3d p_cam = R_bc * p_body + T_bc;

          bool keep = true;
          bool in_front = (p_cam(2) > 0.0);
          Eigen::Vector2d uv(-1.0, -1.0);

          // 动态点判定: 优先 semantic_candidates_ + 深度门控;
          // 候选为空时回退到 2D mask 逻辑
          if (in_front)
          {
              uv = vio_manager->cam->world2cam(p_cam);

              if (!semantic_candidates_.empty()) {
                  if (isDynamicPointByCandidate(p_cam, uv)) keep = false;
              } else if (uv(0) >= 0 && uv(0) < current_mask_.cols &&
                         uv(1) >= 0 && uv(1) < current_mask_.rows) {
                  if (current_mask_.at<uchar>((int)uv(1), (int)uv(0)) == 0) keep = false;
              }
          }

          // 在调试图上画点: 白=保留, 红=删除
          if (build_dbg && in_front) {
              int u = (int)uv.x(), v = (int)uv.y();
              if (u >= 0 && u < debug_img.cols && v >= 0 && v < debug_img.rows) {
                  if (keep) cv::circle(debug_img, cv::Point(u, v), 1, cv::Scalar(255, 255, 255), -1);
                  else      cv::circle(debug_img, cv::Point(u, v), 2, cv::Scalar(0, 0, 255), -1);
              }
          }

          if (keep) {
              feats_static->points.push_back(pt);
              kept_points++;
          } else {
              // 把剔除的点转成红色，发给 RViz 调试
              V3D p_body_vec(pt.x, pt.y, pt.z);
              V3D p_global(_state.rot_end * (extR * p_body_vec + extT) + _state.pos_end);
              pcl::PointXYZRGB pt_rgb;
              pt_rgb.x = p_global(0); pt_rgb.y = p_global(1); pt_rgb.z = p_global(2);
              pt_rgb.r = 255; pt_rgb.g = 0; pt_rgb.b = 0; // 红色
              feats_removed_rgb->points.push_back(pt_rgb);
          }
      }

      // 发布调试图像 (每帧)
      if (build_dbg && !debug_img.empty()) {
          cv_bridge::CvImage out;
          out.header.stamp = ros::Time::now();
          out.encoding = sensor_msgs::image_encodings::BGR8;
          out.image = debug_img;
          pubRtdetrDebugImg.publish(out.toImageMsg());
      }

      // LIO 动态点过滤统计 (每秒一次)
      int removed_num = before_num - kept_points;
      ROS_INFO_THROTTLE(1.0,
          "[RT-LIVO-LIO] before=%d, after=%d, removed=%d, depth_gate=%s",
          before_num, kept_points, removed_num, depth_gate_en ? "on" : "off");

      // 熔断保护：如果剔除后点云太少，则放弃剔除，防止定位飘飞
      if (feats_down_body->size() > 100 && kept_points < 10) {
          ROS_ERROR("[LIO] EMERGENCY: Mask removed nearly all points! Skipping removal to avoid drift.");
          // 不更新 feats_down_body，保持原样
      } else {
          // 1. 更新用于 SLAM 计算的点云
          *feats_down_body = *feats_static;

          // 2. 【核心修复】强制把 RViz 显示用的点云也替换成干净的！
          *feats_undistort = *feats_down_body;

          // 累积删除的点，延迟到publish_frame_world时同步发布
          if (!feats_removed_rgb->empty()) {
              if (feats_removed_accumulated_ == nullptr) {
                  feats_removed_accumulated_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
              }
              *feats_removed_accumulated_ += *feats_removed_rgb;
          }
      }
  }
  // ==========================================

  double t_down = omp_get_wtime();
  feats_down_size = feats_down_body->points.size();
  
  // ... (接后续代码: voxelmap_manager->feats_down_body_ = feats_down_body; 等等) ...
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }
  
  double t1 = omp_get_wtime();


  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  // ==========================================
  // [修复] LIO 模式下清理累积缓冲区，防止历史点云干扰
  // ==========================================
  if (img_en) {
      PointCloudXYZI().swap(*pcl_wait_pub);  // 清空 VIO 模式累积的点云
  }
  publish_frame_world(pubLaserCloudFullRes, vio_manager);

  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  euler_cur = RotMtoEuler(_state.rot_end);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run() 
{
  ros::Rate rate(5000);
  while (ros::ok()) 
  {
    ros::spinOnce();
    if (!sync_packages(LidarMeasures)) 
    {
      rate.sleep();
      continue;
    }

    if (!LidarMeasures.measures.empty() && !LidarMeasures.measures.back().img.empty()) {
        ROS_DEBUG("[Main] Processing image for mask detection: %dx%d",
                  LidarMeasures.measures.back().img.rows,
                  LidarMeasures.measures.back().img.cols);
        DetectAndMask(LidarMeasures.measures.back().img);
    } else if (rtdetr_en) {
        ROS_WARN_THROTTLE(1.0, "[Main] No image available! measures.empty=%d",
                         LidarMeasures.measures.empty());
    }

    handleFirstFrame();

    processImu();

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;
  }

  double cur_head_time = msg->header.stamp.toSec();
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;
  }

  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);
  double timestamp = msg->header.stamp.toSec();

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);

  mtx_buffer.lock();

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);
    return;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

void LIVMapper::img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{
  if (!img_en) return;
  sensor_msgs::Image::Ptr msg(new sensor_msgs::Image(*msg_in));

  // Hiliti2022 40Hz
  if (hilti_en)
  {
    static int frame_counter = 0;
    if (++frame_counter % 4 != 0) return;
  }
  // double msg_header_time =  msg->header.stamp.toSec();
  double msg_header_time = msg->header.stamp.toSec() + img_time_offset;
  if (abs(msg_header_time - last_timestamp_img) < 0.001) return;
  ROS_INFO("Get image, its header time: %.6f", msg_header_time);
  if (last_timestamp_lidar < 0) return;

  if (msg_header_time < last_timestamp_img)
  {
    ROS_ERROR("image loop back. \n");
    return;
  }

  mtx_buffer.lock();

  double img_time_correct = msg_header_time; // last_timestamp_lidar + 0.105;

  if (img_time_correct - last_timestamp_img < 0.02)
  {
    ROS_WARN("Image need Jumps: %.6f", img_time_correct);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    return;
  }

  cv::Mat img_cur = getImageFromMsg(msg);
  img_buffer.push_back(img_cur);
  img_time_buffer.push_back(img_time_correct);

  // ROS_INFO("Correct Image time: %.6f", img_time_correct);

  last_timestamp_img = img_time_correct;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (img_buffer.empty() && img_en) return false;
  if (imu_buffer.empty() && imu_en) return false;

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      meas.pcl_proc_cur = meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { // waiting imu message needs to be
      // larger than _lidar_frame_end_time,
      // make sure complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / double(1000);
      double imu_newest_time = imu_buffer.back()->header.stamp.toSec();

      if (img_capture_time < meas.last_lio_update_time + 0.00001)
      {
        img_buffer.pop_front();
        img_time_buffer.pop_front();
        ROS_ERROR("[ Data Cut ] Throw one image frame! \n");
        return false;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // ROS_ERROR("lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;

      // [关键修复] 保存图像并从buffer中移除，确保mask和点云时间戳匹配
      mtx_buffer.lock();
      m.img = img_buffer.front();
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      while (!imu_buffer.empty())
      {
        if (imu_buffer.front()->header.stamp.toSec() > m.lio_time) break;

        if (imu_buffer.front()->header.stamp.toSec() > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // imu_buffer.front()->header.stamp.toSec());
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      *(meas.pcl_proc_cur) = *(meas.pcl_proc_next);
      PointCloudXYZI().swap(*meas.pcl_proc_next);

      int lid_frame_num = lid_raw_data_buffer.size();
      int max_size = meas.pcl_proc_cur->size() + 24000 * lid_frame_num;
      meas.pcl_proc_cur->reserve(max_size);
      meas.pcl_proc_next->reserve(max_size);
      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();
      double imu_time = imu_buffer.front()->header.stamp.toSec();

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      m.img = img_buffer.front();
      mtx_buffer.lock();
      img_buffer.pop_front();
      img_time_buffer.pop_front();
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      meas.measures.push_back(m);
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = ros::Time::now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  if (img_en)
  {
    static int pub_num = 1;

    // ==========================================
    // [新增修复] 在累积前过滤当前帧的动态物体点云
    // ==========================================
    if (rtdetr_en && !current_mask_.empty() && !pcl_w_wait_pub->empty())
    {
        // 检查 new_frame_ 是否存在，如果不存在则跳过过滤（LIO 模式下的第一帧）
        if (vio_manager->new_frame_ != nullptr)
        {
            PointCloudXYZI::Ptr pcl_w_wait_clean(new PointCloudXYZI());
            pcl_w_wait_clean->reserve(pcl_w_wait_pub->size());

            int removed_count = 0;
            for (const auto& pt : pcl_w_wait_pub->points)
            {
                V3D p_w(pt.x, pt.y, pt.z);
                V3D p_c = vio_manager->new_frame_->w2f(p_w);

                bool keep = true;
                // [改进] 放宽深度限制，与LIO中的处理保持一致
                if (p_c(2) > 0.0)
                {
                    V2D uv = vio_manager->new_frame_->cam_->world2cam(p_c);

                    if (uv(0) >= 0 && uv(0) < current_mask_.cols &&
                        uv(1) >= 0 && uv(1) < current_mask_.rows)
                    {
                        if (current_mask_.at<uchar>((int)uv(1), (int)uv(0)) == 0) {
                            keep = false;
                            removed_count++;
                        }
                    }
                }
                if (keep) pcl_w_wait_clean->points.push_back(pt);
            }
            *pcl_w_wait_pub = *pcl_w_wait_clean;
        }
        else
        {
            ROS_WARN_THROTTLE(1.0, "[Publish] new_frame_ is NULL, skipping point filtering (LIO mode?)");
        }
    }
    // ==========================================

    *pcl_wait_pub += *pcl_w_wait_pub;

    // ==========================================
    // [新增修复] 定期清理历史地图中的动态点云
    // ==========================================
    // 每5帧（pub_scan_num）清理一次历史累积地图，防止动态点云堆积
    if (rtdetr_en && !current_mask_.empty() && !pcl_wait_pub->empty() && vio_manager->new_frame_ != nullptr)
    {
        static int cleanup_counter = 0;
        cleanup_counter++;

        // [改进] 每隔一定次数（例如每2次pub_scan_num周期，即每10帧）进行一次全面清理
        // 这样可以更频繁地清理历史地图中的动态点云
        if (cleanup_counter % 10 == 0)
        {
            PointCloudXYZI::Ptr pcl_wait_clean(new PointCloudXYZI());
            pcl_wait_clean->reserve(pcl_wait_pub->size());

            int history_removed = 0;
            int history_checked = 0;
            for (const auto& pt : pcl_wait_pub->points)
            {
                V3D p_w(pt.x, pt.y, pt.z);
                V3D p_c = vio_manager->new_frame_->w2f(p_w);

                bool keep = true;
                // [改进] 放宽深度限制
                if (p_c(2) > 0.0)
                {
                    history_checked++;
                    V2D uv = vio_manager->new_frame_->cam_->world2cam(p_c);

                    if (uv(0) >= 0 && uv(0) < current_mask_.cols &&
                        uv(1) >= 0 && uv(1) < current_mask_.rows)
                    {
                        if (current_mask_.at<uchar>((int)uv(1), (int)uv(0)) == 0) {
                            keep = false;
                            history_removed++;
                        }
                    }
                }
                if (keep) pcl_wait_clean->points.push_back(pt);
            }

            *pcl_wait_pub = *pcl_wait_clean;
        }
    }
    // ==========================================

    if(pub_num == pub_scan_num)
    {
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);

      cv::Mat img_rgb = vio_manager->img_rgb;

      int total_points = 0;
      int masked_out_points = 0;
      int out_of_fov_points = 0;
      int behind_camera_points = 0;
      int no_frame_points = 0;

      if (vio_manager->new_frame_ == nullptr) {
          ROS_WARN_THROTTLE(1.0, "[Publish] WARNING: new_frame_ is NULL! All %zu points will be skipped.", size);
      }
      if (rtdetr_en && current_mask_.empty()) {
          ROS_WARN_THROTTLE(1.0, "[Publish] WARNING: current_mask_ is EMPTY!");
      } else if (rtdetr_en) {
      }

      for (size_t i = 0; i < size; i++)
      {
        PointTypeRGB pointRGB;
        pointRGB.x = pcl_wait_pub->points[i].x;
        pointRGB.y = pcl_wait_pub->points[i].y;
        pointRGB.z = pcl_wait_pub->points[i].z;

        V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);

        // 【关键修复】检查 new_frame_ 是否存在
        if (vio_manager->new_frame_ == nullptr) {
            no_frame_points++;
            continue;
        }

        V3D pf(vio_manager->new_frame_->w2f(p_w));
        if (pf[2] < 0) {
            behind_camera_points++;
            continue;
        }

        // 【核心修复】直接使用 3D 点 pf 进行投影，避免使用 w2c 导致维度错误
        V2D pc = vio_manager->new_frame_->cam_->world2cam(pf);

        total_points++;

        if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3))
        {
          // -------------------------------------------------------
          // [新增修复] 防止从 Mask 黑区取色
          bool should_color = true;
          if (rtdetr_en && !current_mask_.empty()) {
              int u = (int)pc(0);
              int v = (int)pc(1);
              if (u >= 0 && u < current_mask_.cols && v >= 0 && v < current_mask_.rows) {
                  if (current_mask_.at<uchar>(v, u) == 0) {
                      should_color = false;
                      masked_out_points++;
                  }
              }
          }

          if (should_color) {
              V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
              pointRGB.r = pixel[2];
              pointRGB.g = pixel[1];
              pointRGB.b = pixel[0];
              if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
          } else {
              // [关键修复] 被mask过滤的点，标记为红色并添加到删除点云
              if (pf.norm() > blind_rgb_points) {
                  pointRGB.r = 255;
                  pointRGB.g = 0;
                  pointRGB.b = 0;
                  if (feats_removed_accumulated_ == nullptr) {
                      feats_removed_accumulated_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
                  }
                  feats_removed_accumulated_->points.push_back(pointRGB);
              }
          }
          // -------------------------------------------------------
        } else {
            out_of_fov_points++;
        }
      }

      // [调试] 输出过滤统计
      if (rtdetr_en && total_points > 0) {
          ROS_INFO_THROTTLE(1.0, "[Publish] Filter: %d total, %d masked (%.1f%%), %d FOV, %d behind, %d no_frame",
                           total_points, masked_out_points,
                           100.0 * masked_out_points / total_points,
                           out_of_fov_points, behind_camera_points, no_frame_points);
      }
    } // 【修复】这里之前漏掉了括号，导致 else 报错
    else
    {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::PointCloud2 laserCloudmsg;
  if (img_en)
  {
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else
  {
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg);
  }

  // [关键修复] 发布累积的被剔除点，与/cloud_registered同步
  // 这些点是之前被mask剔除的，但还没有发布
  if (rtdetr_en && feats_removed_accumulated_ != nullptr && !feats_removed_accumulated_->empty())
  {
      sensor_msgs::PointCloud2 deleted_msg;
      pcl::toROSMsg(*feats_removed_accumulated_, deleted_msg);

      // 【关键】使用延迟的时间戳，对齐/cloud_registered的显示
      // 减去偏移量，让红色点显示在"过去"的时间，与白色点同步
      double adjusted_time = LidarMeasures.last_lio_update_time - rtdetr_time_offset;
      deleted_msg.header.stamp = ros::Time(adjusted_time);
      deleted_msg.header.frame_id = "camera_init";
      pub_deleted_points.publish(deleted_msg);

      // 清空累积的点云
      feats_removed_accumulated_->clear();
  }

  // [关键修复] 使用激光雷达时间戳
  laserCloudmsg.header.stamp = ros::Time(LidarMeasures.last_lio_update_time);
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);

  // ==========================================
  // [PCD保存] 累积点云数据用于保存
  // ==========================================
  if (pcd_save_en)
  {
    if (img_en && laserCloudWorldRGB->size() > 0)
    {
      // 将彩色点云累积到pcl_wait_save
      *pcl_wait_save += *laserCloudWorldRGB;
    }
    else if (!img_en && pcl_w_wait_pub->size() > 0)
    {
      // 将强度点云累积到pcl_wait_save_intensity
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
  }
  // ==========================================
}

void LIVMapper::publish_visual_sub_map(const ros::Publisher &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap.publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}

// ==========================================
// [新增] RT-DETR 具体实现
// ==========================================
void LIVMapper::InitRTDETR(ros::NodeHandle &nh) {
    // 读取总开关 (对应 yaml: rtdetr/enable)
    nh.param<bool>("rtdetr/enable", rtdetr_en, false);

    if (rtdetr_en) {
        // 读取基础参数
        nh.param<std::string>("rtdetr/model_name", rtdetr_model_name, "model.onnx");
        nh.param<double>("rtdetr/conf_thresh", rtdetr_conf_thresh, 0.45);
        nh.param<int>("rtdetr/padding", rtdetr_padding, 10);
        nh.param<double>("rtdetr/time_offset", rtdetr_time_offset, 0.0);

        // 运动验证参数
        nh.param<bool>("rtdetr/motion_verify/enable", motion_verify_en, true);
        nh.param<int>("rtdetr/motion_verify/min_track_points", motion_verify_min_points, 12);
        nh.param<double>("rtdetr/motion_verify/high_prior_thresh", motion_verify_high_thresh, 1.5);
        nh.param<double>("rtdetr/motion_verify/medium_prior_thresh", motion_verify_medium_thresh, 3.0);
        nh.param<double>("rtdetr/motion_verify/low_prior_thresh", motion_verify_low_thresh, 5.0);

        // 运动状态时序平滑
        nh.param<bool>("rtdetr/motion_verify/temporal_smooth_enable", motion_temporal_smooth_en, true);
        nh.param<int>("rtdetr/motion_verify/hold_frames", motion_hold_frames, 5);
        nh.param<double>("rtdetr/motion_verify/assoc_iou_min", assoc_iou_min, 0.3);

        // 自适应 padding 参数
        nh.param<bool>("rtdetr/adaptive_padding/enable", adaptive_padding_en, true);
        nh.param<int>("rtdetr/adaptive_padding/min_px", padding_min_px, 4);
        nh.param<int>("rtdetr/adaptive_padding/max_px", padding_max_px, 40);
        nh.param<double>("rtdetr/adaptive_padding/base_px", padding_base_px, 3.0);
        nh.param<double>("rtdetr/adaptive_padding/vmax", padding_vmax, 3.0);
        nh.param<double>("rtdetr/adaptive_padding/flow_gain", padding_flow_gain, 0.5);
        nh.param<double>("rtdetr/adaptive_padding/detection_time_delay", detection_time_delay, 0.05);

        // 深度门控参数
        nh.param<bool>("rtdetr/depth_gate/enable", depth_gate_en, true);
        nh.param<double>("rtdetr/depth_gate/min", depth_gate_min, 0.3);
        nh.param<double>("rtdetr/depth_gate/ratio", depth_gate_ratio, 0.12);

        // 旧版自适应协方差 (默认关闭)
        nh.param<bool>("rtdetr/adaptive_cov/enable", adaptive_img_cov_en, false);
        nh.param<double>("rtdetr/adaptive_cov/min_value", adaptive_img_cov_min, 200.0);
        nh.param<double>("rtdetr/adaptive_cov/max_value", adaptive_img_cov_max, 2000.0);

        // 可靠性感知协方差
        nh.param<bool>("rtdetr/reliability_cov/enable", reliability_cov_en, true);
        nh.param<double>("rtdetr/reliability_cov/min_value", reliability_cov_min, 200.0);
        nh.param<double>("rtdetr/reliability_cov/max_value", reliability_cov_max, 2000.0);
        nh.param<int>("rtdetr/reliability_cov/expected_visual_points", expected_visual_points, 80);

        ROS_INFO("[RT-DETR] time_offset: %.3f s", rtdetr_time_offset);
        ROS_INFO("[RT-DETR] motion_verify: %s (min_pts=%d, H/M/L=%.2f/%.2f/%.2f)",
                 motion_verify_en ? "ON" : "OFF", motion_verify_min_points,
                 motion_verify_high_thresh, motion_verify_medium_thresh, motion_verify_low_thresh);
        ROS_INFO("[RT-DETR] adaptive_padding: %s (min/max=%d/%d)", adaptive_padding_en ? "ON" : "OFF", padding_min_px, padding_max_px);
        ROS_INFO("[RT-DETR] depth_gate: %s (min=%.2f, ratio=%.2f)", depth_gate_en ? "ON" : "OFF", depth_gate_min, depth_gate_ratio);
        ROS_INFO("[RT-DETR] reliability_cov: %s (min/max=%.0f/%.0f, exp_pts=%d)",
                 reliability_cov_en ? "ON" : "OFF", reliability_cov_min, reliability_cov_max, expected_visual_points);
        ROS_INFO("[RT-DETR] legacy adaptive_cov: %s (deprecated)", adaptive_img_cov_en ? "ON" : "OFF");

        // 创建并配置运动验证器
        motion_verifier_ = new MotionVerifier();
        motion_verifier_->configure(
            motion_verify_en, motion_verify_min_points,
            motion_verify_high_thresh, motion_verify_medium_thresh, motion_verify_low_thresh);

        // 类别 -> 先验策略统一由 getMotionPrior() 决定 (HIGH直接删 / MEDIUM光流验证 / LOW保留)
        ROS_INFO("[RT-DETR] policy: HIGH=person+animals(direct del), MEDIUM=vehicles(optflow), LOW=keep");

        // 获取模型绝对路径: src/RT-LIVO/weights/model.onnx
        std::string pkg_path = ros::package::getPath("rt_livo");
        std::string model_path = pkg_path + "/weights/" + rtdetr_model_name;

        ROS_INFO("\033[1;32m[RT-DETR] Model Path: %s\033[0m", model_path.c_str());

        try {
            detector_ = new RTDETRDetector(model_path, true); // true = Use CUDA
            ROS_INFO("\033[1;32m[RT-DETR] Initialized successfully!\033[0m");
        } catch (const std::exception& e) {
            ROS_ERROR("[RT-DETR] Init Failed: %s", e.what());
            rtdetr_en = false;
        }
    }
}

void LIVMapper::DetectAndMask(const cv::Mat& img)
{
    // 1. 候选列表清空; mask 初始化为全 255 (保留). img 本身绝不修改.
    semantic_candidates_.clear();
    if (vio_manager != nullptr && vio_manager->cam != nullptr) {
        current_mask_ = cv::Mat(vio_manager->cam->height(), vio_manager->cam->width(), CV_8UC1, 255);
    } else {
        current_mask_ = cv::Mat(img.size(), CV_8UC1, 255);
    }

    // rtdetr 关闭或无检测器: 直接返回, 不影响原始 FAST-LIVO2 流程
    if (!rtdetr_en || detector_ == nullptr) return;

    int cam_w = vio_manager->cam->width();
    int cam_h = vio_manager->cam->height();
    cv::Rect cam_bounds(0, 0, cam_w, cam_h);

    // 检测框从图像空间 -> 相机模型空间 (与 world2cam 投影一致)
    double sx = (double)cam_w / std::max(1, img.cols);
    double sy = (double)cam_h / std::max(1, img.rows);
    auto toCam = [&](const cv::Rect& r) -> cv::Rect {
        cv::Rect c(cvRound(r.x * sx), cvRound(r.y * sy),
                   cvRound(r.width * sx), cvRound(r.height * sy));
        return c & cam_bounds;
    };

    ROS_DEBUG("[Detect] img=%dx%d cam=%dx%d", img.cols, img.rows, cam_w, cam_h);

    // 2. RT-DETR 检测 (类别过滤已统一到此, detector 内部不再硬编码类别)
    auto results = detector_->detect(img, rtdetr_conf_thresh);

    // 3. 逐检测框: 先验分级 -> (HIGH直接删 / MEDIUM光流验证 / LOW保留) -> padding -> 候选保存
    for (const auto& det : results)
    {
        MotionPrior prior = getMotionPrior(det.class_id);
        if (prior == MotionPrior::LOW) continue;  // 树木/桌椅/其它: 保留, 不参与删除

        SemanticCandidate cand;
        cand.class_id   = det.class_id;
        cand.det_score  = det.score;
        cand.box        = toCam(det.box);   // 相机模型坐标系
        cand.prior      = prior;

        // 分级处置:
        //   HIGH  (人/动物)         -> 直接判 MOVING, 不经光流验证
        //   MEDIUM(车/自行车)       -> 光流验证后决定
        if (cand.prior == MotionPrior::HIGH) {
            cand.state = MotionState::MOVING_OBJECT;
            cand.motion_score = 0.0f;
        } else if (motion_verifier_ != nullptr) {
            float ms = 0.0f;
            cand.state = motion_verifier_->verify(img, det.box, cand.prior, ms);
            cand.motion_score = ms;
        } else {
            cand.state = MotionState::UNCERTAIN_OBJECT;
        }

        // 检测阶段无真实深度, 先置 -1; computeAdaptivePadding 内部回退到 5.0
        cand.median_depth = -1.0f;

        // 自适应 padding (相机模型坐标系像素)
        cand.adaptive_padding = computeAdaptivePadding((double)cand.median_depth, (double)cand.motion_score);

        cv::Rect eb(cand.box.x - cand.adaptive_padding,
                    cand.box.y - cand.adaptive_padding,
                    cand.box.width + 2 * cand.adaptive_padding,
                    cand.box.height + 2 * cand.adaptive_padding);
        cand.expanded_box = eb & cam_bounds;

        semantic_candidates_.push_back(cand);
    }

    // 3.5 时序平滑: 跨帧关联 + MOVING 状态保持, 抑制单帧判定造成的红/绿频繁切换
    //    (残留根因: 偶尔一帧判 STATIC 就把动态点永久写入地图)
    applyTemporalSmoothing();

    int moving_num = 0, static_num = 0, uncertain_num = 0;
    for (const auto& c : semantic_candidates_) {
        if (c.state == MotionState::MOVING_OBJECT) moving_num++;
        else if (c.state == MotionState::STATIC_OBJECT) static_num++;
        else uncertain_num++;
    }

    // 4. 仅对 MOVING_OBJECT 的 expanded_box 在 mask 上画 0
    for (const auto& cand : semantic_candidates_) {
        if (cand.state != MotionState::MOVING_OBJECT) continue;
        cv::rectangle(current_mask_, cand.expanded_box, cv::Scalar(0), -1);
    }

    // 5. 旧版自适应协方差 (默认关闭; 与 reliability_cov 互斥, reliability 优先)
    if (adaptive_img_cov_en && !reliability_cov_en && vio_manager != nullptr) {
        int dyn_px = cv::countNonZero(current_mask_ == 0);
        int tot_px = std::max(1, current_mask_.rows * current_mask_.cols);
        double m = (double)dyn_px / tot_px;
        double cov;
        if (m <= 0.0) {
            cov = adaptive_img_cov_min;
        } else {
            double norm = std::log10(1.0 + 100.0 * m) / std::log10(101.0);
            cov = adaptive_img_cov_min + (adaptive_img_cov_max - adaptive_img_cov_min) * norm;
        }
        vio_manager->img_point_cov = cov;
    }

    // 6. 候选统计日志 (每秒一次, 便于实验/论文记录)
    double mask_ratio = 0.0;
    if (!current_mask_.empty()) {
        int dyn_px = cv::countNonZero(current_mask_ == 0);
        int tot_px = std::max(1, current_mask_.rows * current_mask_.cols);
        mask_ratio = (double)dyn_px / tot_px;
    }
    ROS_INFO_THROTTLE(1.0,
        "[RT-LIVO] candidates=%zu, moving=%d, static=%d, uncertain=%d, mask_ratio=%.3f, img_cov=%.2f",
        semantic_candidates_.size(), moving_num, static_num, uncertain_num, mask_ratio,
        vio_manager ? vio_manager->img_point_cov : -1.0);

    // 7. 缓存当前帧灰度, 供下一帧运动验证使用
    if (motion_verifier_ != nullptr) {
        motion_verifier_->updatePreviousFrame(img);
    }
}

// ==========================================
// 运动状态时序平滑: 跨帧 IoU 关联 + MOVING 状态保持
//   一旦某目标被判过 MOVING, 在 hold_frames 内即使单帧抖动到 STATIC/UNCERTAIN
//   也仍按 MOVING 处理 (持续删除), 从根源上消除红/绿频繁切换造成的地图残留.
// ==========================================
void LIVMapper::applyTemporalSmoothing()
{
    auto IoU = [](const cv::Rect& a, const cv::Rect& b) -> double {
        cv::Rect inter = a & b;
        if (inter.area() <= 0) return 0.0;
        double uni = (double)a.area() + (double)b.area() - (double)inter.area();
        return uni > 0.0 ? (double)inter.area() / uni : 0.0;
    };

    // 关闭时: 仅初始化 hold 计数, 不做跨帧关联
    if (!motion_temporal_smooth_en || tracked_candidates_.empty()) {
        for (auto& c : semantic_candidates_) {
            c.moving_hold_frames = (c.state == MotionState::MOVING_OBJECT) ? motion_hold_frames : 0;
        }
        tracked_candidates_ = semantic_candidates_;
        return;
    }

    std::vector<bool> matched_tracked(tracked_candidates_.size(), false);

    for (auto& c : semantic_candidates_) {
        MotionState raw = c.state;  // 本帧运动验证原始结果

        // 在上一帧候选中找同类、IoU 最大的匹配
        int best_idx = -1;
        double best_iou = 0.0;
        for (size_t i = 0; i < tracked_candidates_.size(); ++i) {
            if (matched_tracked[i] || tracked_candidates_[i].class_id != c.class_id) continue;
            double iou = IoU(tracked_candidates_[i].box, c.box);
            if (iou > best_iou) { best_iou = iou; best_idx = (int)i; }
        }

        if (best_idx >= 0 && best_iou >= assoc_iou_min) {
            matched_tracked[best_idx] = true;
            int& hold = tracked_candidates_[best_idx].moving_hold_frames;
            if (raw == MotionState::MOVING_OBJECT) {
                hold = motion_hold_frames;          // 重新充满保持窗口
                c.state = MotionState::MOVING_OBJECT;
            } else if (hold > 0) {
                hold--;                             // 保持窗口内: 仍按 MOVING 删除
                c.state = MotionState::MOVING_OBJECT;
            } else {
                c.state = raw;                      // 保持窗口耗尽且本帧确实静止
            }
            c.moving_hold_frames = hold;
        } else {
            // 新目标 (上一帧未见): 直接采用本帧判定
            c.moving_hold_frames = (raw == MotionState::MOVING_OBJECT) ? motion_hold_frames : 0;
            c.state = raw;
        }
    }

    // 更新跟踪记忆为当前帧 (框已更新到当前位置, hold 计数随之延续)
    tracked_candidates_ = semantic_candidates_;
}

// ==========================================
// 自适应 padding: 深度/时间延迟 + 运动得分
// ==========================================
int LIVMapper::computeAdaptivePadding(double median_depth, double motion_score)
{
    if (!adaptive_padding_en || vio_manager == nullptr || vio_manager->cam == nullptr) {
        return rtdetr_padding;
    }

    double fx = vio_manager->cam->fx();
    double z = median_depth > 0.3 ? median_depth : 5.0;  // 无真实深度时回退 5.0

    double dt = std::max(0.0, rtdetr_time_offset + detection_time_delay);

    // 物体在检测/传输延迟内可能的像素位移 (近似)
    double depth_padding = fx * padding_vmax * dt / std::max(z, 0.5);
    // 光流残差带来的额外 padding
    double motion_padding = padding_flow_gain * std::max(0.0, motion_score);

    int pad = static_cast<int>(padding_base_px + depth_padding + motion_padding);
    pad = std::max(padding_min_px, std::min(padding_max_px, pad));
    return pad;
}

// ==========================================
// 基于语义候选 + 深度门控判断 LiDAR 点是否为动态点
// ==========================================
bool LIVMapper::isDynamicPointByCandidate(const Eigen::Vector3d& p_cam, const Eigen::Vector2d& uv)
{
    for (const auto& obj : semantic_candidates_) {
        if (obj.state != MotionState::MOVING_OBJECT) continue;

        if (!obj.expanded_box.contains(cv::Point((int)uv.x(), (int)uv.y()))) continue;

        // 深度门控: 点深度需与目标 median_depth 一致; 关闭时仅用 2D 框
        if (depth_gate_en && obj.median_depth > 0.0) {
            double gate = depth_gate_min + depth_gate_ratio * obj.median_depth;
            if (std::abs(p_cam.z() - obj.median_depth) > gate) continue;
        }

        return true;
    }
    return false;
}

// ==========================================
// LIO 过滤前: 用 body 点云更新每个候选目标的深度中值
// ==========================================
void LIVMapper::updateCandidateDepthFromCloud(const PointCloudXYZI::Ptr& cloud_body)
{
    if (cloud_body == nullptr || cloud_body->empty()) return;
    if (vio_manager == nullptr || vio_manager->cam == nullptr) return;

    Eigen::Matrix3d R_bc;
    R_bc << MAT_FROM_ARRAY(cameraextrinR);
    Eigen::Vector3d T_bc;
    T_bc << VEC_FROM_ARRAY(cameraextrinT);

    for (auto& obj : semantic_candidates_) {
        std::vector<double> depths;

        for (const auto& pt : cloud_body->points) {
            Eigen::Vector3d p_body(pt.x, pt.y, pt.z);
            Eigen::Vector3d p_cam = R_bc * p_body + T_bc;
            if (p_cam.z() <= 0.0) continue;

            Eigen::Vector2d uv = vio_manager->cam->world2cam(p_cam);
            if (obj.box.contains(cv::Point((int)uv.x(), (int)uv.y()))) {
                depths.push_back(p_cam.z());
            }
        }

        if (!depths.empty()) {
            std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
            obj.median_depth = static_cast<float>(depths[depths.size() / 2]);
        }
    }
}

// ==========================================
// 可靠性感知视觉协方差: 基于有效视觉约束质量而非动态目标面积
// ==========================================
void LIVMapper::updateVisualReliabilityCovariance()
{
    if (!reliability_cov_en || vio_manager == nullptr) return;

    // 注: 此处 total_points 为上一帧 processFrame 的结果 (本帧尚未更新), 属可接受近似
    int valid_visual_points = vio_manager->total_points;
    double valid_ratio = static_cast<double>(valid_visual_points) /
                         std::max(1, expected_visual_points);
    valid_ratio = std::max(0.0, std::min(1.0, valid_ratio));

    double moving_pixels = 0.0;
    double uncertain_pixels = 0.0;
    double total_pixels = current_mask_.empty() ? 1.0 :
                          (double)current_mask_.rows * current_mask_.cols;

    for (const auto& obj : semantic_candidates_) {
        double area = static_cast<double>(obj.expanded_box.area());
        if (obj.state == MotionState::MOVING_OBJECT) moving_pixels += area;
        if (obj.state == MotionState::UNCERTAIN_OBJECT) uncertain_pixels += area;
    }

    double moving_ratio = std::max(0.0, std::min(1.0, moving_pixels / total_pixels));
    double uncertain_ratio = std::max(0.0, std::min(1.0, uncertain_pixels / total_pixels));

    // 视觉约束质量: 有效点越多越好, 不确定/运动区域越少越好
    double visual_quality =
        0.6 * valid_ratio +
        0.3 * (1.0 - uncertain_ratio) +
        0.1 * (1.0 - moving_ratio);
    visual_quality = std::max(0.0, std::min(1.0, visual_quality));

    // 质量越低, 协方差越大 (视觉约束越不可信)
    double cov = reliability_cov_min +
                 (reliability_cov_max - reliability_cov_min) * (1.0 - visual_quality);

    vio_manager->img_point_cov = cov;
}