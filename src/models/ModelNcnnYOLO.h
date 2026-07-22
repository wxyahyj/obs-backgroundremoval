#ifndef MODEL_NCNN_YOLO_H
#define MODEL_NCNN_YOLO_H

#ifdef _WIN32
#define NOMINMAX
#endif

#include "IYoloModel.h"
#include "Detection.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <atomic>
#include <ncnn/net.h>
#include <ncnn/mat.h>

class ModelNcnnYOLO : public IYoloModel {
public:
    explicit ModelNcnnYOLO(Version version);
    ~ModelNcnnYOLO() override;

    void loadModel(const std::string& modelPath, const std::string& useGPU = "cpu",
                   int numThreads = 1, int inputResolution = 640) override;

    std::vector<Detection> inference(const cv::Mat& input) override;
    std::future<std::vector<Detection>> asyncInference(const cv::Mat& input) override;

    void setConfidenceThreshold(float threshold) override;
    void setNMSThreshold(float threshold) override;
    void setTargetClass(int classId) override;
    void setTargetClasses(const std::vector<int>& classIds) override;
    void loadClassNames(const std::string& namesFile) override;
    void setInputResolution(int resolution) override;

    Version getVersion() const override { return version_; }
    int getInputWidth() const override { return inputWidth_; }
    int getInputHeight() const override { return inputHeight_; }
    int getNumClasses() const override { return numClasses_; }
    const std::vector<std::string>& getClassNames() const override { return classNames_; }

    // ncnn 不支持 DML/CUDA 纹理直推，返回 false/空
    bool isDmlTextureSupported() const override { return false; }
    std::vector<Detection> inferenceFromTextureDml(
        const DmlPreprocessedFrame& preprocessedFrame,
        int originalWidth, int originalHeight,
        InferenceLatency* outLatency = nullptr) override;

    bool isGpuTextureSupported() const override { return false; }
    std::vector<Detection> inferenceFromTexture(
        void* d3d11Texture, int width, int height,
        int originalWidth, int originalHeight,
        InferenceLatency* outLatency = nullptr) override;

    DmlPreprocessor* getDmlPreprocessor() const override { return nullptr; }

    std::string getLatencySummary() const override { return latencyStats_.getSummary(); }
    void resetLatencyStats() override { latencyStats_.reset(); }

private:
    struct LetterboxInfo {
        float scale;
        int padX;
        int padY;
    };

    struct InferenceTask {
        cv::Mat input;
        std::promise<std::vector<Detection>> promise;
    };

    std::vector<Detection> postprocessYOLOv5(
        const float* rawOutput, int numBoxes, int stride, int numClasses,
        const LetterboxInfo& letterboxInfo, const cv::Size& originalImageSize);
    std::vector<Detection> postprocessYOLOv8(
        const float* rawOutput, int numBoxes, int stride, int numClasses,
        const LetterboxInfo& letterboxInfo, const cv::Size& originalImageSize);
    std::vector<Detection> postprocessYOLOv11(
        const float* rawOutput, int numBoxes, int stride, int numClasses,
        const LetterboxInfo& letterboxInfo, const cv::Size& originalImageSize);

    std::vector<int> performNMS(
        const std::vector<cv::Rect2f>& boxes,
        const std::vector<float>& scores,
        float nmsThreshold);
    float calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b);

    LetterboxInfo letterbox(const cv::Mat& input, cv::Mat& output);
    static LetterboxInfo calculateLetterboxParams(int srcWidth, int srcHeight, int dstWidth, int dstHeight);
    std::vector<Detection> doInference(const cv::Mat& input);

    Version version_;
    float confidenceThreshold_;
    float nmsThreshold_;
    int targetClassId_;
    std::unordered_set<int> targetClasses_;

    int inputWidth_;
    int inputHeight_;
    int numClasses_;
    std::vector<std::string> classNames_;

    ncnn::Net net_;
    bool vulkanEnabled_;
    bool modelLoaded_;

    // 预分配letterbox缓冲区
    cv::Mat letterboxBuffer_;
    cv::Mat resizedBuffer_;

    // ncnn 输出转置缓冲区
    std::vector<float> ncnnFwdScratch_;

    // 延迟统计
    LatencyStats latencyStats_;
    std::chrono::steady_clock::time_point lastLatencyLogTime_;
    static constexpr int LATENCY_LOG_INTERVAL_MS = 5000;

    // 异步推理线程
    std::thread inferenceThread_;
    std::atomic<bool> inferenceThreadRunning_;
    std::queue<std::unique_ptr<InferenceTask>> inferenceTasks_;
    std::mutex inferenceTasksMutex_;
    std::condition_variable inferenceTasksCV_;
};

#endif
