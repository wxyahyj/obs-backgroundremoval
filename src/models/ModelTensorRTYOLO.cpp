#ifdef USE_TENSORRT_YOLO

#include "ModelTensorRTYOLO.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <fstream>

// 包含 TensorRT-YOLO 头文件
#include "trtyolo.hpp"

// TrtLatencyStats 实现
void TrtLatencyStats::addSample(const TrtInferenceLatency& latency) {
    std::lock_guard<std::mutex> lock(mutex_);
    count_++;
    sum_.totalMs += latency.totalMs;
    sum_.preprocessMs += latency.preprocessMs;
    sum_.inferenceMs += latency.inferenceMs;
    sum_.postprocessMs += latency.postprocessMs;
    if (count_ == 1 || latency.totalMs < min_.totalMs) min_ = latency;
    if (count_ == 1 || latency.totalMs > max_.totalMs) max_ = latency;
}

void TrtLatencyStats::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    count_ = 0;
    sum_ = TrtInferenceLatency();
    min_ = TrtInferenceLatency();
    max_ = TrtInferenceLatency();
}

std::string TrtLatencyStats::getSummary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ == 0) return "无数据";
    
    std::ostringstream oss;
    oss << "=== 延迟统计 (TensorRT-YOLO) ===\n"
        << "总延迟: 平均 " << (sum_.totalMs / count_) << "ms | "
        << "最小 " << min_.totalMs << "ms | "
        << "最大 " << max_.totalMs << "ms\n"
        << "  预处理: " << (sum_.preprocessMs / count_) << "ms\n"
        << "  推理: " << (sum_.inferenceMs / count_) << "ms\n"
        << "  后处理: " << (sum_.postprocessMs / count_) << "ms\n"
        << "样本数: " << count_;
    return oss.str();
}

// ModelTensorRTYOLO 实现
ModelTensorRTYOLO::ModelTensorRTYOLO(Version version)
    : version_(version)
    , confidenceThreshold_(0.5f)
    , nmsThreshold_(0.45f)
    , targetClassId_(-1)
    , inputWidth_(640)
    , inputHeight_(640)
    , numClasses_(80)
    , loaded_(false)
{
}

ModelTensorRTYOLO::~ModelTensorRTYOLO() = default;

bool ModelTensorRTYOLO::loadEngine(const std::string& enginePath, int inputResolution) {
    try {
        inputWidth_ = inputResolution;
        inputHeight_ = inputResolution;
        
        trtyolo::InferOption option;
        option.enableSwapRB();
        
        detector_ = std::make_unique<trtyolo::DetectModel>(enginePath, option);
        
        loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ModelTensorRTYOLO] 加载引擎失败: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<Detection> ModelTensorRTYOLO::inference(const cv::Mat& input) {
    if (!loaded_ || !detector_) {
        return {};
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    TrtInferenceLatency latency;
    
    auto preprocessStart = std::chrono::high_resolution_clock::now();
    
    cv::Mat rgbImage;
    if (input.channels() == 3) {
        cv::cvtColor(input, rgbImage, cv::COLOR_BGR2RGB);
    } else {
        rgbImage = input;
    }
    
    trtyolo::Image img(rgbImage.data, rgbImage.cols, rgbImage.rows);
    
    auto preprocessEnd = std::chrono::high_resolution_clock::now();
    latency.preprocessMs = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
    
    auto inferenceStart = std::chrono::high_resolution_clock::now();
    
    trtyolo::DetectRes result = detector_->predict(img);
    
    auto inferenceEnd = std::chrono::high_resolution_clock::now();
    latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count();
    
    auto postprocessStart = std::chrono::high_resolution_clock::now();
    
    std::vector<Detection> detections = convertResults(result, input.cols, input.rows);
    
    auto postprocessEnd = std::chrono::high_resolution_clock::now();
    latency.postprocessMs = std::chrono::duration<double, std::milli>(postprocessEnd - postprocessStart).count();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    latency.totalMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    latencyStats_.addSample(latency);
    
    return detections;
}

std::vector<Detection> ModelTensorRTYOLO::convertResults(const trtyolo::DetectRes& trtResult,
                                                          int originalWidth, int originalHeight) {
    std::vector<Detection> detections;
    
    for (int i = 0; i < trtResult.num; ++i) {
        if (i >= static_cast<int>(trtResult.classes.size()) ||
            i >= static_cast<int>(trtResult.scores.size()) ||
            i >= static_cast<int>(trtResult.boxes.size())) {
            break;
        }
        
        int cls = trtResult.classes[i];
        float score = trtResult.scores[i];
        
        if (score < confidenceThreshold_) {
            continue;
        }
        
        if (targetClassId_ >= 0 && cls != targetClassId_) {
            continue;
        }
        
        if (!targetClasses_.empty()) {
            if (std::find(targetClasses_.begin(), targetClasses_.end(), cls) == targetClasses_.end()) {
                continue;
            }
        }
        
        Detection det;
        det.classId = cls;
        det.confidence = score;
        
        if (cls >= 0 && cls < static_cast<int>(classNames_.size())) {
            det.className = classNames_[cls];
        }
        
        const auto& box = trtResult.boxes[i];
        
        float x1 = box.left;
        float y1 = box.top;
        float x2 = box.right;
        float y2 = box.bottom;
        
        det.x = x1 / originalWidth;
        det.y = y1 / originalHeight;
        det.width = (x2 - x1) / originalWidth;
        det.height = (y2 - y1) / originalHeight;
        
        det.centerX = det.x + det.width / 2.0f;
        det.centerY = det.y + det.height / 2.0f;
        
        det.trackId = -1;
        det.lostFrames = 0;
        
        detections.push_back(det);
    }
    
    return detections;
}

void ModelTensorRTYOLO::setConfidenceThreshold(float threshold) {
    confidenceThreshold_ = threshold;
}

void ModelTensorRTYOLO::setNMSThreshold(float threshold) {
    nmsThreshold_ = threshold;
}

void ModelTensorRTYOLO::setTargetClass(int classId) {
    targetClassId_ = classId;
    targetClasses_.clear();
}

void ModelTensorRTYOLO::setTargetClasses(const std::vector<int>& classIds) {
    targetClasses_ = classIds;
    targetClassId_ = -1;
}

void ModelTensorRTYOLO::loadClassNames(const std::string& namesFile) {
    classNames_.clear();
    std::ifstream file(namesFile);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                classNames_.push_back(line);
            }
        }
        numClasses_ = static_cast<int>(classNames_.size());
    }
}

std::unique_ptr<ModelTensorRTYOLO> ModelTensorRTYOLO::clone() {
    if (!loaded_ || !detector_) {
        return nullptr;
    }
    
    auto cloned = std::make_unique<ModelTensorRTYOLO>(version_);
    cloned->confidenceThreshold_ = confidenceThreshold_;
    cloned->nmsThreshold_ = nmsThreshold_;
    cloned->targetClassId_ = targetClassId_;
    cloned->targetClasses_ = targetClasses_;
    cloned->inputWidth_ = inputWidth_;
    cloned->inputHeight_ = inputHeight_;
    cloned->numClasses_ = numClasses_;
    cloned->classNames_ = classNames_;
    
    try {
        cloned->detector_ = detector_->clone();
        cloned->loaded_ = true;
    } catch (const std::exception& e) {
        std::cerr << "[ModelTensorRTYOLO] 克隆失败: " << e.what() << std::endl;
        return nullptr;
    }
    
    return cloned;
}

int ModelTensorRTYOLO::getBatchSize() const {
    if (loaded_ && detector_) {
        return detector_->batch();
    }
    return 1;
}

#endif // USE_TENSORRT_YOLO
