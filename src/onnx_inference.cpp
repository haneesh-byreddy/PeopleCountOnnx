#include "onnx_inference.h"

Detector::Detector(std::string &modelPath)
{
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");

    sessionOptions = Ort::SessionOptions();
    Ort::SessionOptions session_options;

    int gpu_device_id = 0;
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = gpu_device_id;
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    std::cout << "Model loaded successfully" << std::endl;
}

cv::Mat Detector::detect(cv::Mat &image, float confThreshold, float iouThreshold)
{

    std::vector<float> image_data;
    cv::Mat preprocessedImage = preprocess(image, image_data);

    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, image_data.data(), image_data.size(), input_shape.data(), input_shape.size());

    const char *input_name = "images";
    const char *output_name = "output0";
    std::vector<const char *> output_names = {output_name};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, output_names.data(), output_names.size());

    if (output_tensors.empty())
    {
        std::cerr << "Error: No output tensor returned from inference." << std::endl;
    }

    float *output_data = output_tensors.front().GetTensorMutableData<float>();
    std::vector<int64_t> output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    cv::Mat detection = postprocess(image, output_data, output_shape, confThreshold, iouThreshold);

    return detection;
}

float iou(const std::vector<float> &boxA, const std::vector<float> &boxB)
{
    const float eps = 1e-6;
    float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    float x1 = std::max(boxA[0], boxB[0]);
    float y1 = std::max(boxA[1], boxB[1]);
    float x2 = std::min(boxA[2], boxB[2]);
    float y2 = std::min(boxA[3], boxB[3]);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float inter = w * h;
    return inter / (areaA + areaB - inter + eps);
}

void nms(std::vector<std::vector<float>> &boxes, const float iou_threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](const std::vector<float> &boxA, const std::vector<float> &boxB)
              { return boxA[4] > boxB[4]; });
    for (int i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i][4] == 0.f)
            continue;
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (boxes[i][5] != boxes[j][5])
                continue;
            if (iou(boxes[i], boxes[j]) > iou_threshold)
                boxes[j][4] = 0.f;
        }
    }
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const std::vector<float> &box)
                               { return box[4] == 0.f; }),
                boxes.end());
}
cv::Mat Detector::preprocess(cv::Mat &frame, std::vector<float> &input_tensor_values)
{
    cv::Mat resizedImage;
    cv::resize(frame, resizedImage, cv::Size(640, 640));
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(resizedImage, channels);
    for (auto &ch : channels)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float *)ch.datastart, (float *)ch.dataend);
    }
    return resizedImage;
}

cv::Mat Detector::postprocess(cv::Mat &frame, float *output_data, std::vector<int64_t> output_shape, float confThreshold, float iouThreshold)
{
    std::vector<std::vector<float>> boxes;
    cv::Mat resizedImage;
    cv::resize(frame, resizedImage, cv::Size(640, 640));
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
    for (int i = 0; i < output_shape[2]; ++i)
    {
        float cx = output_data[i + output_shape[2] * 0];
        float cy = output_data[i + output_shape[2] * 1];
        float w = output_data[i + output_shape[2] * 2];
        float h = output_data[i + output_shape[2] * 3];
        float score_1 = round(output_data[i + output_shape[2] * 4] * 100) / 100.0;
        float score_2 = round(output_data[i + output_shape[2] * 5] * 100) / 100.0;
        float score_3 = round(output_data[i + output_shape[2] * 6] * 100) / 100.0;

        if (score_1 > 0.1 || score_2 > 0.1 || score_3 > 0.1)
        {
            int class_id = 1;
            float max_score = score_1;
            if (score_2 > score_1)
            {
                class_id = 2;
                max_score = score_2;
            }
            if (score_3 > score_2)
            {
                class_id = 3;
                max_score = score_3;
            }
            int left = static_cast<int>(cx - w / 2);
            int top = static_cast<int>(cy - h / 2);
            int right = static_cast<int>(cx + w / 2);
            int bottom = static_cast<int>(cy + h / 2);
            boxes.push_back({static_cast<float>(left), static_cast<float>(top), static_cast<float>(right), static_cast<float>(bottom), max_score, static_cast<float>(class_id)}); // 0 is a placeholder for class_id
        }
    }
    float iou_threshold = 0.3;
    nms(boxes, iou_threshold);
    std::cout << "Number of boxes :" << boxes.size() << std::endl;

    for (const auto &box : boxes)
    {
        int left = static_cast<int>(box[0]);
        int top = static_cast<int>(box[1]);
        int right = static_cast<int>(box[2]);
        int bottom = static_cast<int>(box[3]);
        float score = box[4];
        int class_id = box[5];

        auto color = cv::Scalar(255, 0, 0);
        if (class_id == 1)
        {
            color = cv::Scalar(0, 255, 0);
        }
        if (class_id == 2)
        {
            color = cv::Scalar(0, 0, 255);
        }
        cv::rectangle(resizedImage, cv::Point(left, top), cv::Point(right, bottom), color, 1);
        std::string label = "Score: " + std::to_string(score) + " Class : " + std::to_string(class_id);
        cv::putText(resizedImage, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 0, 0);
    }
    cv::Mat output_frame;
    resizedImage.convertTo(output_frame, CV_8U, 255.0);
    return output_frame;
}