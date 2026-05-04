#ifndef MODEL_YOLO_H
#define MODEL_YOLO_H

#ifdef _WIN32
#define NOMINMAX
#endif

#include "Model.h"
#include "Detection.h"
#include <vector>
#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <future>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <chrono>
#include <atomic>
#include <cstdio>

#ifdef _WIN32
#include "DmlPreprocessor.h"
#endif

// 前向声明CUDA类型
struct cudaGraphicsResource;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

// 延迟统计结构体
struct InferenceLatency {
    double totalMs;           // 总延迟
    double preprocessMs;      // 预处理延迟
    double inferenceMs;       // 推理延迟
    double postprocessMs;     // 后处理延迟
    double gpuCopyMs;         // GPU数据拷贝延迟（仅GPU路径）
    double cudaKernelMs;      // CUDA内核执行延迟（仅GPU路径）
    bool isGpuPath;           // 是否使用GPU路径
    
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

class ModelYOLO : public ModelBCHW {
public:
    enum class Version {
        YOLOv5 = 0,
        YOLOv8 = 1,
        YOLOv11 = 2
    };

    explicit ModelYOLO(Version version);
    ~ModelYOLO() override;

    void loadModel(const std::string& modelPath, const std::string& useGPU = "cpu", int numThreads = 1, int inputResolution = 640);
    void preprocessInput(const cv::Mat& input, float* outputBuffer);
    void setInputResolution(int resolution);

    std::vector<Detection> inference(const cv::Mat& input);
    std::future<std::vector<Detection>> asyncInference(const cv::Mat& input);

    // GPU纹理直接推理（CUDA/TensorRT）
    std::vector<Detection> inferenceFromTexture(void* d3d11Texture, int width, int height, 
                                                 int originalWidth, int originalHeight,
                                                 InferenceLatency* outLatency = nullptr);
    bool isGpuTextureSupported() const { return cudaInteropInitialized_; }
    
    // DML纹理直接推理
    std::vector<Detection> inferenceFromTextureDml(void* d3d11Texture, int width, int height,
                                                    int originalWidth, int originalHeight,
                                                    InferenceLatency* outLatency = nullptr);
    bool isDmlTextureSupported() const { return dmlInteropInitialized_; }
    
    // 延迟统计
    const LatencyStats& getLatencyStats() const { return latencyStats_; }
    void resetLatencyStats() { latencyStats_.reset(); }
    std::string getLatencySummary() const { return latencyStats_.getSummary(); }

    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void setTargetClass(int classId);
    void setTargetClasses(const std::vector<int>& classIds);
    void loadClassNames(const std::string& namesFile);

    Version getVersion() const { return version_; }
    int getInputWidth() const { return inputWidth_; }
    int getInputHeight() const { return inputHeight_; }
    int getNumClasses() const { return numClasses_; }
    const std::vector<std::string>& getClassNames() const { return classNames_; }

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
        const float* rawOutput,
        int numBoxes,
        int numClasses,
        const LetterboxInfo& letterboxInfo,
        const cv::Size& originalImageSize
    );

    std::vector<Detection> postprocessYOLOv8(
        const float* rawOutput,
        int numBoxes,
        int numClasses,
        const LetterboxInfo& letterboxInfo,
        const cv::Size& originalImageSize
    );

    std::vector<Detection> postprocessYOLOv11(
        const float* rawOutput,
        int numBoxes,
        int numClasses,
        const LetterboxInfo& letterboxInfo,
        const cv::Size& originalImageSize
    );

    std::vector<int> performNMS(
        const std::vector<cv::Rect2f>& boxes,
        const std::vector<float>& scores,
        float nmsThreshold
    );

    float calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b);

    void xywhToxyxy(float cx, float cy, float w, float h,
                    float& x1, float& y1, float& x2, float& y2);

    LetterboxInfo letterbox(const cv::Mat& input, cv::Mat& output);
    static LetterboxInfo calculateLetterboxParams(int srcWidth, int srcHeight, int dstWidth, int dstHeight);
    std::vector<Detection> doInference(const cv::Mat& input);
    
    // GPU内存初始化
    bool initializeGpuMemory();
    void releaseGpuMemory();
    
    // CUDA互操作初始化（阶段2）
    bool initializeCudaInterop();
    void releaseCudaInterop();
    
    // DML预处理器初始化
    bool initializeDmlPreprocessor();
    void releaseDmlInterop();

    Version version_;
    float confidenceThreshold_;
    float nmsThreshold_;
    int targetClassId_;
    std::unordered_set<int> targetClasses_;

    int inputWidth_;
    int inputHeight_;
    int numClasses_;

    std::vector<std::string> classNames_;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::IoBinding> ioBinding_;
    std::vector<Ort::AllocatedStringPtr> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNames_;
    std::vector<std::vector<int64_t>> inputDims_;
    std::vector<std::vector<int64_t>> outputDims_;
    std::vector<std::vector<float>> outputTensorValues_;
    std::vector<std::vector<float>> inputTensorValues_;
    std::vector<Ort::Value> inputTensor_;
    std::vector<Ort::Value> outputTensor_;
    
    size_t inputBufferSize_;
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffer_;
    std::vector<Ort::Float16_t> inputBufferFp16_;  // FP16输入缓冲区
    bool useIOBinding_;
    bool isFp16Model_;  // 是否为FP16模型
    
    // 预分配letterbox缓冲区
    cv::Mat letterboxBuffer_;
    cv::Mat resizedBuffer_;
    
    // === 阶段1：GPU持久内存 ===
    bool useGpuMemory_;
    std::string currentDevice_;
    Ort::Allocator* gpuAllocator_;
    Ort::Value gpuInputTensor_;
    Ort::Value gpuOutputTensor_;
    Ort::MemoryInfo* gpuMemInfo_;
    
    // === 阶段2：CUDA纹理共享 ===
    bool cudaInteropInitialized_;
    void* cudaStream_;
    cudaGraphicsResource_t cudaResource_;
    void* cudaInputBuffer_;
    
    // === DML纹理共享 ===
    bool dmlInteropInitialized_;
    class DmlPreprocessor* dmlPreprocessor_;
    
    // === 延迟统计 ===
    LatencyStats latencyStats_;
    std::chrono::steady_clock::time_point lastLatencyLogTime_;
    static constexpr int LATENCY_LOG_INTERVAL_MS = 5000;  // 每5秒输出一次延迟统计
    
    std::thread inferenceThread_;
    std::atomic<bool> inferenceThreadRunning_;
    std::queue<std::unique_ptr<InferenceTask>> inferenceTasks_;
    std::mutex inferenceTasksMutex_;
    std::condition_variable inferenceTasksCV_;
};

#endif
