#ifndef IYOLOMODEL_H
#define IYOLOMODEL_H

#include "Detection.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <mutex>
#include <chrono>
#include <atomic>
#include <cstdio>

#ifdef _WIN32
#include "DmlPreprocessor.h"
#endif

// 延迟统计结构体
struct InferenceLatency {
    double totalMs;
    double preprocessMs;
    double inferenceMs;
    double postprocessMs;
    double gpuCopyMs;
    double cudaKernelMs;
    bool isGpuPath;

    InferenceLatency() : totalMs(0), preprocessMs(0), inferenceMs(0),
                         postprocessMs(0), gpuCopyMs(0), cudaKernelMs(0), isGpuPath(false) {}

    InferenceLatency& operator+=(const InferenceLatency& other) {
        totalMs += other.totalMs;
        preprocessMs += other.preprocessMs;
        inferenceMs += other.inferenceMs;
        postprocessMs += other.postprocessMs;
        gpuCopyMs += other.gpuCopyMs;
        cudaKernelMs += other.cudaKernelMs;
        isGpuPath = isGpuPath || other.isGpuPath;
        return *this;
    }
};

// 延迟统计器
class LatencyStats {
public:
    void addSample(const InferenceLatency& latency) {
        std::lock_guard<std::mutex> lock(mutex_);
        count_++;
        sum_ += latency;
        if (count_ == 1 || latency.totalMs < min_.totalMs) min_ = latency;
        if (count_ == 1 || latency.totalMs > max_.totalMs) max_ = latency;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = 0;
        sum_ = InferenceLatency();
        min_ = InferenceLatency();
        max_ = InferenceLatency();
    }

    std::string getSummary() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == 0) return "无数据";

        char buf[1024];
        if (sum_.isGpuPath) {
            snprintf(buf, sizeof(buf),
                "=== 延迟统计 (GPU路径) ===\n"
                "总延迟: 平均 %.2fms | 最小 %.2fms | 最大 %.2fms\n"
                "  预处理: %.2fms (CUDA内核: %.2fms, GPU拷贝: %.2fms)\n"
                "  推理: %.2fms\n"
                "  后处理: %.2fms\n"
                "样本数: %zu",
                sum_.totalMs / count_, min_.totalMs, max_.totalMs,
                sum_.preprocessMs / count_, sum_.cudaKernelMs / count_, sum_.gpuCopyMs / count_,
                sum_.inferenceMs / count_,
                sum_.postprocessMs / count_,
                count_);
        } else {
            snprintf(buf, sizeof(buf),
                "=== 延迟统计 (CPU路径) ===\n"
                "总延迟: 平均 %.2fms | 最小 %.2fms | 最大 %.2fms\n"
                "  预处理: %.2fms\n"
                "  推理: %.2fms\n"
                "  后处理: %.2fms\n"
                "样本数: %zu",
                sum_.totalMs / count_, min_.totalMs, max_.totalMs,
                sum_.preprocessMs / count_,
                sum_.inferenceMs / count_,
                sum_.postprocessMs / count_,
                count_);
        }
        return std::string(buf);
    }

    size_t getCount() const { return count_; }

private:
    mutable std::mutex mutex_;
    size_t count_ = 0;
    InferenceLatency sum_;
    InferenceLatency min_;
    InferenceLatency max_;
};

// 抽象推理接口，支持 ONNX Runtime 和 ncnn 后端
class IYoloModel {
public:
    enum class Version {
        YOLOv5 = 0,
        YOLOv8 = 1,
        YOLOv11 = 2
    };

    virtual ~IYoloModel() = default;

    // 模型加载
    virtual void loadModel(const std::string& modelPath, const std::string& useGPU = "cpu",
                           int numThreads = 1, int inputResolution = 640) = 0;

    // 推理
    virtual std::vector<Detection> inference(const cv::Mat& input) = 0;
    virtual std::future<std::vector<Detection>> asyncInference(const cv::Mat& input) = 0;

    // 参数设置
    virtual void setConfidenceThreshold(float threshold) = 0;
    virtual void setNMSThreshold(float threshold) = 0;
    virtual void setTargetClass(int classId) = 0;
    virtual void setTargetClasses(const std::vector<int>& classIds) = 0;
    virtual void loadClassNames(const std::string& namesFile) = 0;
    virtual void setInputResolution(int resolution) = 0;

    // 查询
    virtual Version getVersion() const = 0;
    virtual int getInputWidth() const = 0;
    virtual int getInputHeight() const = 0;
    virtual int getNumClasses() const = 0;
    virtual const std::vector<std::string>& getClassNames() const = 0;

    // DML 纹理推理（仅 ORT 后端支持，ncnn 返回 false/空）
    virtual bool isDmlTextureSupported() const = 0;
    virtual std::vector<Detection> inferenceFromTextureDml(
        const DmlPreprocessedFrame& preprocessedFrame,
        int originalWidth, int originalHeight,
        InferenceLatency* outLatency = nullptr) = 0;

    // CUDA 纹理推理（仅 ORT+CUDA 后端支持）
    virtual bool isGpuTextureSupported() const = 0;
    virtual std::vector<Detection> inferenceFromTexture(
        void* d3d11Texture, int width, int height,
        int originalWidth, int originalHeight,
        InferenceLatency* outLatency = nullptr) = 0;

    // DML 预处理器（仅 ORT+DML 后端）
    virtual DmlPreprocessor* getDmlPreprocessor() const = 0;

    // 延迟统计
    virtual std::string getLatencySummary() const = 0;
    virtual void resetLatencyStats() = 0;
};

#endif
