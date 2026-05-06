#include "YOLODetector.h"
#include "models/ModelYOLO.h"
#include <chrono>
#include <plugin-support.h>

YOLODetector::YOLODetector(const Config& config) : config_(config) {
}

YOLODetector::~YOLODetector() {
    shutdown();
}

bool YOLODetector::initialize() {
    if (initialized_) {
        return true;
    }

    try {
        std::lock_guard<std::mutex> lock(modelMutex_);
        
        model_ = std::make_unique<ModelYOLO>(static_cast<ModelYOLO::Version>(config_.modelVersion));
        
        if (!model_) {
            obs_log(LOG_ERROR, "[YOLODetector] Failed to create ModelYOLO");
            return false;
        }

        std::string useGPU = config_.useGPU ? "cuda" : "cpu";
        model_->loadModel(config_.modelPath, useGPU, config_.numThreads, config_.inputResolution);

        model_->setConfidenceThreshold(config_.confidenceThreshold);
        model_->setNMSThreshold(config_.nmsThreshold);
        
        if (!config_.targetClasses.empty()) {
            model_->setTargetClasses(config_.targetClasses);
        } else if (config_.targetClassId >= 0) {
            model_->setTargetClass(config_.targetClassId);
        }

        initialized_ = true;
        running_ = true;
        obs_log(LOG_INFO, "[YOLODetector] Initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[YOLODetector] Exception during initialization: %s", e.what());
        return false;
    }
}

void YOLODetector::shutdown() {
    if (!initialized_) {
        return;
    }

    running_ = false;
    
    std::lock_guard<std::mutex> lock(modelMutex_);
    model_.reset();
    initialized_ = false;
    
    obs_log(LOG_INFO, "[YOLODetector] Shutdown complete");
}

std::vector<Detection> YOLODetector::inference(const cv::Mat& frame) {
    if (!initialized_ || !model_) {
        return {};
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<Detection> detections;
    
    {
        std::lock_guard<std::mutex> lock(modelMutex_);
        if (model_) {
            try {
                detections = model_->inference(frame);
            } catch (const std::exception& e) {
                obs_log(LOG_ERROR, "[YOLODetector] Inference error: %s", e.what());
                return {};
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.inferenceCount++;
        stats_.avgInferenceTimeMs = (stats_.avgInferenceTimeMs * (stats_.inferenceCount - 1) + duration.count()) / stats_.inferenceCount;
    }

    return detections;
}

#ifdef _WIN32
#if defined(HAVE_CUDA) || defined(HAVE_ONNXRUNTIME_DML_EP)
std::vector<Detection> YOLODetector::inferenceFromTexture(void* texture, int width, int height, int fullWidth, int fullHeight) {
    if (!initialized_ || !model_) {
        return {};
    }

    std::vector<Detection> detections;
    
    {
        std::lock_guard<std::mutex> lock(modelMutex_);
        if (model_) {
            try {
#ifdef HAVE_CUDA
                if (model_->isGpuTextureSupported()) {
                    detections = model_->inferenceFromTexture(texture, width, height, fullWidth, fullHeight);
                }
#endif
#ifdef HAVE_ONNXRUNTIME_DML_EP
                if (model_->isDmlTextureSupported() && detections.empty()) {
                    detections = model_->inferenceFromTextureDml(texture, width, height, fullWidth, fullHeight);
                }
#endif
            } catch (const std::exception& e) {
                obs_log(LOG_ERROR, "[YOLODetector] GPU texture inference error: %s", e.what());
                return {};
            }
        }
    }

    return detections;
}
#endif
#endif

void YOLODetector::updateConfig(const Config& config) {
    bool needReload = (config.modelPath != config_.modelPath ||
                       config.modelVersion != config_.modelVersion ||
                       config.useGPU != config_.useGPU ||
                       config.numThreads != config_.numThreads ||
                       config.inputResolution != config_.inputResolution);

    config_ = config;

    if (needReload && initialized_) {
        shutdown();
        initialize();
    } else if (model_) {
        std::lock_guard<std::mutex> lock(modelMutex_);
        if (model_) {
            model_->setConfidenceThreshold(config_.confidenceThreshold);
            model_->setNMSThreshold(config_.nmsThreshold);
            
            if (!config_.targetClasses.empty()) {
                model_->setTargetClasses(config_.targetClasses);
            } else if (config_.targetClassId >= 0) {
                model_->setTargetClass(config_.targetClassId);
            }
        }
    }
}

YOLODetector::Stats YOLODetector::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}
