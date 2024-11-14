#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <chrono>
#include "logger.h"
#include <thread>
#include <atomic>
#include <condition_variable>

template <typename T>
class SafeQueue {
    public:
        SafeQueue() : q(), m(), c() {}
        void enqueue(T t) {
            std::lock_guard<std::mutex> lock(m);
            q.push(t);
            c.notify_one();
        }
        bool dequeue(T& t) {
            std::unique_lock<std::mutex> lock(m);
            while (q.empty()) {
                if (finished) return false;
                c.wait(lock);
            }
            t = q.front();
            q.pop();
            return true;
        }
        void setFinished() {
            std::lock_guard<std::mutex> lock(m);
            finished = true;
            c.notify_all();
        }
    private:
        std::queue<T> q;
        mutable std::mutex m;
        std::condition_variable c;
        bool finished = false;
};

class Detector {
public:
    Detector (std::string &modelPath);
    cv::Mat detect(cv::Mat &image, float confThreshold = 0.4f, float iouThreshold = 0.45f);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};
    cv::Size inputImageShape;

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<const char *> inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    std::vector<const char *> outputNames;
    cv::Mat preprocess(cv::Mat &image, std::vector<float> &input_tensor);
    cv::Mat postprocess(cv::Mat &image, float* data, std::vector<int64_t> shape, float confThreshold, float iouThreshold);
};