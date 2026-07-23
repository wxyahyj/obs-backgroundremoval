#ifndef MODEL_YOLO_H
#define MODEL_YOLO_H

#ifdef _WIN32
#define NOMINMAX
#endif

#include "Model.h"
#include "Detection.h"
#include "IYoloModel.h"
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

// 前向声明CUDA类型
struct cudaGraphicsResource;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

class ModelYOLO : public ModelBCHW, public IYoloModel {
public:
    explicit ModelYOLO(Version version);
    ~ModelYOLO() override;

    void loadModel(const std::string& modelPath, const std::string& useGPU = "cpu", int numThreads = 1, int inputResolution = 640) override;
    void preprocessInput(const cv::Mat& input, float* outputBuffer);
    void setInputResolution(int resolution) override;

    std::vector<Detection> inference(const cv::Mat& input) override;
    std::future<std::vector<Detection>> asyncInference(const cv::Mat& input) override;

    // GPU纹理直接推理（CUDA/TensorRT）— D3D11 interop, device float, no D2H
    std::vector<Detection> inferenceFromTexture(void* d3d11Texture,
                                                 int cropX, int cropY, int cropW, int cropH,
                                                 int originalWidth, int originalHeight,
                                                 InferenceLatency* outLatency = nullptr) override;
    bool isGpuTextureSupported() const override { return cudaInteropInitialized_; }

    // DML纹理直接推理
    std::vector<Detection> inferenceFromTextureDml(const DmlPreprocessedFrame& preprocessedFrame,
                                                    int originalWidth, int originalHeight,
                                                    InferenceLatency* outLatency = nullptr) override;
    bool isDmlTextureSupported() const override { return dmlInteropInitialized_; }

    // 延迟统计
    const LatencyStats& getLatencyStats() const { return latencyStats_; }
    void resetLatencyStats() override { latencyStats_.reset(); }
    std::string getLatencySummary() const override { return latencyStats_.getSummary(); }

    void setConfidenceThreshold(float threshold) override;
    void setNMSThreshold(float threshold) override;
    void setTargetClass(int classId) override;
    void setTargetClasses(const std::vector<int>& classIds) override;
    void loadClassNames(const std::string& namesFile) override;

    Version getVersion() const override { return version_; }
    int getInputWidth() const override { return inputWidth_; }
    int getInputHeight() const override { return inputHeight_; }
    int getNumClasses() const override { return numClasses_; }
    const std::vector<std::string>& getClassNames() const override { return classNames_; }
    DmlPreprocessor* getDmlPreprocessor() const override { return dmlPreprocessor_.get(); }

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
        float nmsThreshold,
        const std::vector<int>& classIds = {}
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
    // Run 热路径复用，避免每帧建 vector<const char*>
    std::vector<const char*> inputNamesChar_;
    std::vector<const char*> outputNamesChar_;
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
    std::vector<float> outputFp32Scratch_;         // FP16 输出解码缓冲
    std::vector<Ort::Float16_t> outputBufferFp16_;  // 持久 FP16 输出绑定缓冲
    std::vector<int64_t> inputShapeCache_;          // {1,3,H,W}
    std::vector<int64_t> outputShapeCache_;
    size_t outputElementCount_ = 0;
    std::unique_ptr<Ort::MemoryInfo> cpuMemInfo_;   // 持久 CPU MemoryInfo
    Ort::Value cpuInputTensor_{nullptr};           // 复用，缓冲指针不变则不重建
    Ort::Value cpuOutputTensor_{nullptr};
    bool cpuInputTensorFp16_ = false;
    bool cpuOutputTensorFp16_ = false;
    size_t cpuInputTensorElems_ = 0;
    size_t cpuOutputTensorElems_ = 0;
    bool useIOBinding_;
    bool isFp16Model_;  // 是否为FP16模型
    bool isFp16Output_ = false;

    void ensureCpuMemInfo();
    void ensureCpuInputTensor();
    void ensureCpuOutputTensor();
    
    // 预分配letterbox缓冲区
    cv::Mat letterboxBuffer_;
    cv::Mat resizedBuffer_;
    int letterboxLastNewW_ = -1;
    int letterboxLastNewH_ = -1;
    int letterboxLastPadX_ = -1;
    int letterboxLastPadY_ = -1;
    
    // === 阶段1：GPU持久内存 ===
    bool useGpuMemory_;
    std::string currentDevice_;
    std::unique_ptr<Ort::Allocator> gpuAllocator_;
    Ort::Value gpuInputTensor_;
    Ort::Value gpuOutputTensor_;
    std::unique_ptr<Ort::MemoryInfo> gpuMemInfo_;
    
    // === 阶段2：CUDA纹理共享 ===
    bool cudaInteropInitialized_;
    void* cudaStream_;
    // Cached register: only valid while the same ID3D11Texture2D* is live
    void* cudaRegisteredTex_;                 // last registered ID3D11Texture2D*
    cudaGraphicsResource_t cudaResource_;
    void* cudaInputBuffer_;                   // device float CHW (cudaMalloc)
    size_t cudaInputBufferBytes_;
    void* cudaOutputBuffer_;                  // device float output (optional)
    size_t cudaOutputBufferBytes_;
    void* cudaBgraStaging_;                   // device BGRA for host fallback
    size_t cudaBgraStagingBytes_;
    std::unique_ptr<Ort::MemoryInfo> cudaMemInfo_; // "Cuda" device MemoryInfo
    Ort::Value cudaInputTensor_{nullptr};     // wraps cudaInputBuffer_
    
    // === DML纹理共享 ===
    bool dmlInteropInitialized_;
    std::unique_ptr<DmlPreprocessor> dmlPreprocessor_;
    
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
