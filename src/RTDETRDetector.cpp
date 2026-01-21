#include "RTDETRDetector.h"
#include <numeric>

RTDETRDetector::RTDETRDetector(const std::string& model_path, bool use_cuda) 
    : env(ORT_LOGGING_LEVEL_WARNING, "RTDETRv2"), 
      session(nullptr) { // 初始化 session 为 nullptr
    
    // 1. 配置 Session
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (...) {
            std::cerr << "[WARN] CUDA not available, using CPU." << std::endl;
        }
    }
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // 2. 加载模型
    session = std::unique_ptr<Ort::Session>(new Ort::Session(env, model_path.c_str(), session_options));

    // 3. 获取输入信息
    Ort::TypeInfo input_info = session->GetInputTypeInfo(0);
    auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
    input_dims = input_shape_info.GetShape();
    if (input_dims[0] == -1) input_dims[0] = 1;

    // 4. 定义节点名称 (适配 RT-DETRv2 官方导出)
    // v2 有两个输入: 图片 和 原始尺寸
    static const char* in_names[] = {"images", "orig_target_sizes"};
    // v2 有三个输出: 类别, 框, 分数
    static const char* out_names[] = {"labels", "boxes", "scores"};

    input_node_names.assign(in_names, in_names + 2);   // 2个输入
    output_node_names.assign(out_names, out_names + 3); // 3个输出
}

std::vector<float> RTDETRDetector::preprocess(const cv::Mat& image, int target_h, int target_w) {
    cv::Mat resized, rgb, float_img;
    cv::resize(image, resized, cv::Size(target_w, target_h));
    cv::cvtColor(resized, rgb, cv::ColorConversionCodes::COLOR_BGR2RGB);
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    std::vector<float> input_tensor_values(3 * target_h * target_w);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(target_h, target_w, CV_32FC1, input_tensor_values.data() + i * target_h * target_w);
    }
    cv::split(float_img, channels);
    return input_tensor_values;
}

std::vector<Detection> RTDETRDetector::detect(const cv::Mat& image, float conf_threshold) {
    if (image.empty()) return {};

    int target_h = input_dims[2];
    int target_w = input_dims[3];

    // --- 准备输入 1: 图像 ---
    std::vector<float> input_tensor_values = preprocess(image, target_h, target_w);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value img_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), 
        input_dims.data(), input_dims.size());

    // --- 准备输入 2: 原始尺寸 (这是 v2 必须的!) ---
    // 格式是 int64, shape [batch, 2], 内容是 [width, height]
    std::vector<int64_t> size_values = {static_cast<int64_t>(image.cols), static_cast<int64_t>(image.rows)};
    std::vector<int64_t> size_dims = {1, 2};
    Ort::Value size_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, size_values.data(), size_values.size(), 
        size_dims.data(), size_dims.size());

    // 将两个输入放入 vector
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(img_tensor));
    inputs.push_back(std::move(size_tensor));

    // --- 执行推理 ---
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr}, 
        input_node_names.data(), inputs.data(), 2, // 2个输入
        output_node_names.data(), 3 // 3个输出
    );

    // --- 解析输出 (v2 是分离的输出) ---
    // 输出 0: labels [batch, 300] (int64 或 int32)
    // 输出 1: boxes  [batch, 300, 4] (float32)
    // 输出 2: scores [batch, 300] (float32)

    // 注意：GetTensorData 返回的是展平的数组指针
    // 假设 batch=1, num_queries=300
    const int64_t* labels_ptr = output_tensors[0].GetTensorData<int64_t>(); // labels通常是int64
    const float* boxes_ptr    = output_tensors[1].GetTensorData<float>();
    const float* scores_ptr   = output_tensors[2].GetTensorData<float>();

    auto output_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    int num_queries = output_shape[1]; // 通常是 300

    std::vector<Detection> results;
    for (int i = 0; i < num_queries; ++i) {
        float score = scores_ptr[i];
        int label = static_cast<int>(labels_ptr[i]);

        // 过滤: 置信度 + 类别 (0=Person, 2=Car, 5=Bus, 7=Truck)
        if (score > conf_threshold && (label == 0 || label == 2 || label == 5 || label == 7)) {
            // RT-DETRv2 导出的 boxes 已经是绝对坐标 [x1, y1, x2, y2]
            // 或者是 [cx, cy, w, h]，具体取决于导出配置，v2 默认通常是绝对坐标 [x1, y1, x2, y2]
            
            // 指针偏移: boxes 是 [300, 4]，所以每个 query 占 4 个 float
            const float* box = boxes_ptr + i * 4;
            float x1 = box[0];
            float y1 = box[1];
            float x2 = box[2];
            float y2 = box[3];

            Detection det;
            // 转换为 cv::Rect (x, y, w, h)
            det.box = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), 
                               static_cast<int>(x2 - x1), static_cast<int>(y2 - y1));
            
            // 边界保护
            det.box &= cv::Rect(0, 0, image.cols, image.rows);

            det.class_id = label;
            det.score = score;
            det.label = std::to_string(label);
            results.push_back(det);
        }
    }
    return results;
}
