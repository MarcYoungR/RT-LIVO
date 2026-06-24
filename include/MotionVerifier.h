/*
 * MotionVerifier: 真实运动验证模块 (RT-LIVO)
 *
 * 设计目标:
 *   RT-DETR 的输出仅作为 semantic dynamic candidate (语义动态候选).
 *   本模块对 MEDIUM 先验(载具)做光流运动验证, 决定是否判为 MOVING_OBJECT.
 *
 * 与调用方 (LIVMapper::DetectAndMask) 的分工:
 *   HIGH 先验 (人/动物) 由调用方直接判 MOVING_OBJECT 并删除, 不进入本模块;
 *   LOW  先验 (其它)   由调用方 continue 跳过, 不参与删除, 不进入本模块;
 *   因此 verify() 实际只会收到 MEDIUM, 内部仅使用 medium_prior_thresh.
 *   (high/low_prior_thresh 为预留参数, 当前不生效, 见 config 注释.)
 *
 * 判定流程 (MEDIUM 候选):
 *   1. 若 motion_verify/enable=false, 兼容旧逻辑: 直接视为 MOVING_OBJECT;
 *   2. 无上一帧图像 -> UNCERTAIN_OBJECT (不删除);
 *   3. 框内可跟踪角点数 < min_track_points -> UNCERTAIN_OBJECT (不删除);
 *   4. 计算 ROI 内光流位移中值 median_roi_flow;
 *   5. 计算背景光流位移中值 median_bg_flow (剔除目标框区域);
 *   6. motion_score = median_roi_flow - median_bg_flow;
 *   7. motion_score > medium_prior_thresh -> MOVING_OBJECT, 否则 STATIC_OBJECT.
 */

#ifndef MOTION_VERIFIER_H
#define MOTION_VERIFIER_H

#include <opencv2/opencv.hpp>

// 运动状态: 静态 / 运动 / 不确定(不删除)
enum class MotionState {
    STATIC_OBJECT = 0,
    MOVING_OBJECT = 1,
    UNCERTAIN_OBJECT = 2
};

// 类别动态先验: 越高越倾向于"可能运动", 但仍需验证
enum class MotionPrior {
    HIGH = 0,
    MEDIUM = 1,
    LOW = 2
};

// 语义动态候选目标 (RT-DETR 输出 + 验证结果)
struct SemanticCandidate {
    cv::Rect box;                 // 相机模型坐标系下的检测框 (与 world2cam 投影一致)
    cv::Rect expanded_box;        // 自适应 padding 后的扩展框 (用于 mask / 深度门控)
    int class_id = -1;            // COCO 类别 id
    float det_score = 0.0f;       // 检测置信度
    float motion_score = 0.0f;    // 运动验证得分 (ROI - 背景光流中值)
    float median_depth = -1.0f;   // 框内 LiDAR 点深度中值 (LIO 阶段更新; <0 表示未知)
    int adaptive_padding = 0;     // 本次使用的自适应 padding (像素)
    int moving_hold_frames = 0;   // 时序保持: 仍需按 MOVING 处理的剩余帧数
    MotionPrior prior = MotionPrior::LOW;
    MotionState state = MotionState::UNCERTAIN_OBJECT;
};

// 类别 -> 动态先验 (COCO 0-79 连续索引; HIGH=人/动物, MEDIUM=载具1-8, LOW=其它; 详见 MotionVerifier.cpp)
MotionPrior getMotionPrior(int class_id);

class MotionVerifier {
public:
    MotionVerifier();

    void configure(
        bool enable,
        int min_track_points,
        double high_prior_thresh,
        double medium_prior_thresh,
        double low_prior_thresh
    );

    // 对单个候选框做光流运动验证, 输出 motion_score 并返回运动状态
    MotionState verify(
        const cv::Mat& curr_img,
        const cv::Rect& box,
        MotionPrior prior,
        float& motion_score
    );

    // 在帧处理结束后调用, 缓存当前帧灰度图供下一帧使用
    void updatePreviousFrame(const cv::Mat& curr_img);

private:
    bool enable_ = true;
    bool has_prev_ = false;
    cv::Mat prev_gray_;

    int min_track_points_ = 12;
    double medium_prior_thresh_ = 3.0; // 唯一生效的光流阈值 (仅 MEDIUM 进入 verify)
    double high_prior_thresh_ = 1.5;   // 预留: HIGH 在调用方直接删, 不进入 verify
    double low_prior_thresh_ = 5.0;    // 预留: LOW 在调用方被跳过, 不进入 verify

    // ROI 内光流位移中值, valid_track_num 输出有效跟踪点数
    double computeMedianFlowInBox(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const cv::Rect& box,
        int& valid_track_num
    );

    // 背景光流位移中值 (排除 excluded_box 区域)
    double computeBackgroundMedianFlow(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const cv::Rect& excluded_box
    );
};

#endif // MOTION_VERIFIER_H
