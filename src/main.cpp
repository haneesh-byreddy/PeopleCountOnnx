#include "onnx_inference.h"

int main()
{
    std::string videoPath = "./data/07-11-2024-11.52.54.avi";
    std::string modelPath = "./data/best_multi_class.onnx";
    std::string outputPath = videoPath.substr(0, videoPath.length()-4) + "_detection" + videoPath.substr(videoPath.length()-4, videoPath.length());

    Detector detector(modelPath);
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open or find the video file!\n";
        return -1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));

    cv::VideoWriter out(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
    if (!out.isOpened())
    {
        std::cerr << "Error: Could not open the output video file for writing!\n";
        return -1;
    }

    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<cv::Mat> processedQueue;
    std::atomic<bool> processingDone(false);

    std::thread captureThread([&]()
        {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame))
        {
            frameQueue.enqueue(frame.clone()); 
            frameCount++;
        }
        frameQueue.setFinished(); 
        });
    std::thread processingThread([&]()
        {
        cv::Mat frame;
        int frameIndex = 0;
        while (frameQueue.dequeue(frame))
        {

            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat result = detector.detect(frame);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
            processedQueue.enqueue(result);
        }
        processedQueue.setFinished(); 
        });
    std::thread writingThread([&]()
        {
        cv::Mat processedFrame;
        while (processedQueue.dequeue(processedFrame))
        {
            std::cout << "Displaying frame" << std::endl;
            out.write(processedFrame);
            cv::imshow("yolo11 inference", processedFrame);
            if (cv::waitKey(10) == 'q') {
                break;
            };
        }
        });
    captureThread.join();
    processingThread.join();

    cap.release();
    cv::destroyAllWindows();

    std::cout << "Video processing completed successfully." << std::endl;

    return 0;
}