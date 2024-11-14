const std::string onnx_model_path = "./data/best_multi_class.onnx";
cv::Mat resized;
void preProcess(const cv::Mat &frame, std::vector<float> &input_tensor_values)
{
    cv::resize(frame, resized, cv::Size(640, 640));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (auto &ch : channels)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float *)ch.datastart, (float *)ch.dataend);
    }
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
int runInference()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO-Inference");
    Ort::SessionOptions session_options;

    int gpu_device_id = 0;

    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = gpu_device_id;

    session_options.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session session(env, onnx_model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char *input_name = input_name_ptr.get();

    size_t num_output_nodes = session.GetOutputCount();
    cout << "Number of output nodes: " << num_output_nodes << endl;

    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
        const char *output_name = output_name_ptr.get();
        cout << "Output name " << i << ": " << output_name << endl;
    }

    // cv::VideoCapture cap("./data/07-11-2024-11.52.54.avi");
    cv::VideoCapture cap("./data/07-11-2024-12.25.54.avi");
    if (!cap.isOpened())
    {
        cerr << "Error: Unable to open video stream." << endl;
        return -1;
    }
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        auto start = std::chrono::high_resolution_clock::now();

        vector<float> input_tensor_values;
        preProcess(frame, input_tensor_values);

        array<int64_t, 4> input_shape = {1, 3, 640, 640};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        const char *output_name = "output0";
        vector<const char *> output_names = {output_name};
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, output_names.data(), output_names.size());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        if (output_tensors.empty())
        {
            cerr << "Error: No output tensor returned from inference." << endl;
            return -1;
        }

        float *output_data = output_tensors.front().GetTensorMutableData<float>();
        auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        int no_of_people = 0;
        std::vector<std::vector<float>> boxes;
        for (int i = 0; i < output_shape[2]; ++i)
        {
            float cx = output_data[i + output_shape[2] * 0];
            float cy = output_data[i + output_shape[2] * 1];
            float w = output_data[i + output_shape[2] * 2];
            float h = output_data[i + output_shape[2] * 3];
            float score_1 = round(output_data[i + output_shape[2] * 4] * 100) / 100.0;
            float score_2 = round(output_data[i + output_shape[2] * 5] * 100) / 100.0;
            float score_3 = round(output_data[i + output_shape[2] * 6] * 100) / 100.0;

            float score_threshold = 0.1;
            if (score_1 > score_threshold || score_2 > score_threshold || score_3 > score_threshold)
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
            cv::rectangle(resized, cv::Point(left, top), cv::Point(right, bottom), color, 1);
            string label = "Score: " + to_string(score).substr(0, 4) + " Class : " + to_string(class_id);
            cv::putText(resized, label, cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 0, 0);
        }
        cv::imshow("YOLO Inference", resized);
        if (cv::waitKey(1) == 'q')
            break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}