#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <opencv2/core.hpp>
#include "models/Detection.h"

class ModelYOLO;

class YOLODetector {
public:
    struct Config {
        std::string modelPath;
        int modelVersion = 2;  // YOLOv8
        int inputResolution = 640;
        float confidenceThreshold = 0.5f;
        float nmsThreshold = 0.45f;
        int targetClassId = -1;
        std::vector<int> targetClasses;
        int numThreads = 4;
        bool useGPU = false;
        bool useGpuTextureInference = false;
        int inferenceIntervalFrames = 1;
    };

    struct Stats {
        uint64_t totalFrames = 0;
        uint64_t inferenceCount = 0;
        double avgInferenceTimeMs = 0.0;
        double currentFps = 0.0;
    };

    explicit YOLODetector(const Config& config);
    ~YOLODetector();

    YOLODetector(const YOLODetector&) = delete;
    YOLODetector& operator=(const YOLODetector&) = delete;
    YOLODetector(YOLODetector&&) noexcept = default;
    YOLODetector& operator=(YOLODetector&&) noexcept = default;

    bool initialize();
    void shutdown();

    std::vector<Detection> inference(const cv::Mat& frame);
    
#ifdef _WIN32
#if defined(HAVE_CUDA) || defined(HAVE_ONNXRUNTIME_DML_EP)
    std::vector<Detection> inferenceFromTexture(void* texture, int width, int height, int fullWidth, int fullHeight);
#endif
#endif

    void updateConfig(const Config& config);
    Config getConfig() const { return config_; }
    Stats getStats() const;

    bool isInitialized() const { return initialized_; }

private:
    Config config_;
    std::unique_ptr<ModelYOLO> model_;
    std::mutex modelMutex_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};
    
    Stats stats_;
    mutable std::mutex statsMutex_;
};
