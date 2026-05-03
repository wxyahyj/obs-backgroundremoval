#ifndef MODEL_TENSORRT_YOLO_H
#define MODEL_TENSORRT_YOLO_H

#ifdef USE_TENSORRT_YOLO

#ifdef _WIN32
#define NOMINMAX
#endif

#include "Detection.h"
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <mutex>
#include <opencv2/core.hpp>

// 前向声明 TensorRT-YOLO 类型
namespace trtyolo {
    class DetectModel;
    class InferOption;
    struct DetectRes;
    struct Box;
}

// 延迟统计结构体
struct TrtInferenceLatency {
    double totalMs;
    double preprocessMs;
    double inferenceMs;
    double postprocessMs;
    
    TrtInferenceLatency() : totalMs(0), preprocessMs(0), inferenceMs(0), postprocessMs(0) {}
};

// 延迟统计器
class TrtLatencyStats {
public:
    void addSample(const TrtInferenceLatency& latency);
    void reset();
    std::string getSummary() const;
    size_t getCount() const { return count_; }
    
private:
    mutable std::mutex mutex_;
    size_t count_ = 0;
    TrtInferenceLatency sum_;
    TrtInferenceLatency min_;
    TrtInferenceLatency max_;
};

class ModelTensorRTYOLO {
public:
    enum class Version {
        YOLOv5 = 0,
        YOLOv8 = 1,
        YOLOv11 = 2
    };

    explicit ModelTensorRTYOLO(Version version = Version::YOLOv11);
    ~ModelTensorRTYOLO();

    bool loadEngine(const std::string& enginePath, int inputResolution = 640);
    bool isLoaded() const { return loaded_; }
    
    std::vector<Detection> inference(const cv::Mat& input);
    
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
    
    const TrtLatencyStats& getLatencyStats() const { return latencyStats_; }
    void resetLatencyStats() { latencyStats_.reset(); }
    std::string getLatencySummary() const { return latencyStats_.getSummary(); }
    
    std::unique_ptr<ModelTensorRTYOLO> clone();
    int getBatchSize() const;

private:
    std::vector<Detection> convertResults(const trtyolo::DetectRes& trtResult, 
                                           int originalWidth, int originalHeight);
    
    Version version_;
    float confidenceThreshold_;
    float nmsThreshold_;
    int targetClassId_;
    std::vector<int> targetClasses_;
    
    int inputWidth_;
    int inputHeight_;
    int numClasses_;
    bool loaded_;
    
    std::vector<std::string> classNames_;
    
    std::unique_ptr<trtyolo::DetectModel> detector_;
    
    TrtLatencyStats latencyStats_;
};

#endif // USE_TENSORRT_YOLO

#endif // MODEL_TENSORRT_YOLO_H
