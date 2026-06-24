/*
 * MotionVerifier: 真实运动验证模块 (RT-LIVO) 实现
 */

#include "MotionVerifier.h"
#include <vector>
#include <algorithm>
#include <cmath>

MotionPrior getMotionPrior(int class_id) {
    // HIGH  : 人 / 动物 (自主运动生物) -> 直接删除, 不经光流验证
    if (class_id == 0) return MotionPrior::HIGH;                       // person
    if (class_id >= 14 && class_id <= 23) return MotionPrior::HIGH;    // bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe

    // MEDIUM: 载具 (陆/海/空, 自主运动但常停放) -> 由光流验证后剔除
    if (class_id >= 1 && class_id <= 8) return MotionPrior::MEDIUM;    // bicycle/car/motorcycle/airplane/bus/train/truck/boat

    // LOW   : 基础设施 / 家具 / 电器 / 随身物品 / 食物 / 其它 -> 默认保留, 不参与删除
    return MotionPrior::LOW;
}

MotionVerifier::MotionVerifier() {}

void MotionVerifier::configure(
    bool enable,
    int min_track_points,
    double high_prior_thresh,
    double medium_prior_thresh,
    double low_prior_thresh)
{
    enable_ = enable;
    min_track_points_ = min_track_points;
    high_prior_thresh_ = high_prior_thresh;
    medium_prior_thresh_ = medium_prior_thresh;
    low_prior_thresh_ = low_prior_thresh;
}

double MotionVerifier::computeMedianFlowInBox(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const cv::Rect& box,
    int& valid_track_num)
{
    valid_track_num = 0;
    if (prev_gray.empty() || curr_gray.empty()) return 0.0;
    if (prev_gray.size() != curr_gray.size()) return 0.0;

    cv::Rect roi = box & cv::Rect(0, 0, prev_gray.cols, prev_gray.rows);
    if (roi.width < 4 || roi.height < 4) return 0.0;

    // 在 ROI 子图上提取 good features
    cv::Mat prev_roi = prev_gray(roi);
    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(prev_roi, pts, 200, 0.01, 5.0, cv::noArray(), 3, true, 0.04);
    if (pts.empty()) return 0.0;

    // 还原到整图坐标系
    for (auto& p : pts) { p.x += static_cast<float>(roi.x); p.y += static_cast<float>(roi.y); }

    std::vector<cv::Point2f> next;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, next, status, err,
                             cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

    std::vector<double> mags;
    mags.reserve(pts.size());
    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) continue;
        double dx = next[i].x - pts[i].x;
        double dy = next[i].y - pts[i].y;
        mags.push_back(std::sqrt(dx * dx + dy * dy));
    }

    valid_track_num = static_cast<int>(mags.size());
    if (mags.empty()) return 0.0;

    // 中值 (nth_element)
    size_t mid = mags.size() / 2;
    std::nth_element(mags.begin(), mags.begin() + mid, mags.end());
    return mags[mid];
}

double MotionVerifier::computeBackgroundMedianFlow(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const cv::Rect& excluded_box)
{
    if (prev_gray.empty() || curr_gray.empty()) return 0.0;
    if (prev_gray.size() != curr_gray.size()) return 0.0;

    // 略微扩大排除区, 避免边界点干扰背景统计
    cv::Rect ex = excluded_box & cv::Rect(0, 0, prev_gray.cols, prev_gray.rows);
    cv::Rect ex_big(ex.x - 15, ex.y - 15, ex.width + 30, ex.height + 30);
    ex_big &= cv::Rect(0, 0, prev_gray.cols, prev_gray.rows);

    // 在整图上做网格采样, 跳过目标区域
    std::vector<cv::Point2f> pts;
    int step = 24;
    for (int y = step / 2; y < prev_gray.rows; y += step) {
        for (int x = step / 2; x < prev_gray.cols; x += step) {
            if (ex_big.contains(cv::Point(x, y))) continue;
            pts.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }
    if (pts.size() < 8) return 0.0; // 背景点不足, 视作无背景运动

    std::vector<cv::Point2f> next;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, next, status, err,
                             cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

    std::vector<double> mags;
    mags.reserve(pts.size());
    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) continue;
        double dx = next[i].x - pts[i].x;
        double dy = next[i].y - pts[i].y;
        mags.push_back(std::sqrt(dx * dx + dy * dy));
    }
    if (mags.empty()) return 0.0;

    size_t mid = mags.size() / 2;
    std::nth_element(mags.begin(), mags.begin() + mid, mags.end());
    return mags[mid];
}

MotionState MotionVerifier::verify(
    const cv::Mat& curr_img,
    const cv::Rect& box,
    MotionPrior prior,
    float& motion_score)
{
    motion_score = 0.0f;

    // 兼容旧逻辑: 关闭运动验证时, 候选类别直接视为运动目标
    if (!enable_) return MotionState::MOVING_OBJECT;

    if (!has_prev_ || prev_gray_.empty()) return MotionState::UNCERTAIN_OBJECT;

    cv::Mat curr_gray;
    if (curr_img.channels() == 1) {
        curr_gray = curr_img;
    } else {
        cv::cvtColor(curr_img, curr_gray, cv::COLOR_BGR2GRAY);
    }
    if (curr_gray.size() != prev_gray_.size()) return MotionState::UNCERTAIN_OBJECT;

    int roi_valid = 0;
    double roi_med = computeMedianFlowInBox(prev_gray_, curr_gray, box, roi_valid);

    // 框内可跟踪点不足 -> 不确定, 不删除
    if (roi_valid < min_track_points_) return MotionState::UNCERTAIN_OBJECT;

    double bg_med = computeBackgroundMedianFlow(prev_gray_, curr_gray, box);
    double score = roi_med - bg_med;
    motion_score = static_cast<float>(score);

    // 仅 MEDIUM 先验会进入本函数 (HIGH 由调用方直接删, LOW 由调用方跳过),
    // 故只使用 medium_prior_thresh_; prior 参数保留以维持接口稳定, 实际未使用.
    (void)prior;

    if (score > medium_prior_thresh_) return MotionState::MOVING_OBJECT;
    return MotionState::STATIC_OBJECT;
}

void MotionVerifier::updatePreviousFrame(const cv::Mat& curr_img)
{
    if (curr_img.empty()) return;
    cv::Mat g;
    if (curr_img.channels() == 1) {
        g = curr_img.clone();
    } else {
        cv::cvtColor(curr_img, g, cv::COLOR_BGR2GRAY);
    }
    prev_gray_ = g; // 浅拷贝 (引用计数), 下一帧只读使用
    has_prev_ = true;
}
