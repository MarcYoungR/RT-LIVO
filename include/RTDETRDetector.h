#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 检测结果结构体
struct Detection {
    cv::Rect box;     // 矩形框
    int class_id;     // 类别ID
    float score;      // 置信度
    std::string label; // 类别名称
};

class RTDETRDetector {
public:
    /**
     * @brief 构造函数
     * @param model_path ONNX 模型文件的绝对路径
     * @param use_cuda 是否使用 CUDA 加速 (建议为 true)
     */
    RTDETRDetector(const std::string& model_path, bool use_cuda = true);
    
    ~RTDETRDetector() = default;

    /**
     * @brief 检测接口
     * @param image 输入图像 (BGR 格式, cv::Mat)
     * @param conf_threshold 置信度阈值 (默认 0.45)
     * @return 检测结果列表
     */
    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.45f);

private:
    // ONNX Runtime 核心对象
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    // 模型节点信息
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<int64_t> input_dims; // [1, 3, 640, 640]

    // 内部预处理函数
    std::vector<float> preprocess(const cv::Mat& image, int target_h, int target_w);
};