#include "ModelYOLO.h"
#include <plugin-support.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <fstream>
#include <numeric>

ModelYOLO::ModelYOLO(Version version)
    : ModelBCHW(),
      version_(version),
      confidenceThreshold_(0.5f),
      nmsThreshold_(0.45f),
      targetClassId_(-1),
      inputWidth_(640),
      inputHeight_(640),
      numClasses_(80)
{
    obs_log(LOG_INFO, "[ModelYOLO] Initialized (Version: %d)", static_cast<int>(version));
    
    try {
        std::string instanceName{"YOLOModel"};
        env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str());
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Failed to initialize ORT: %s", e.what());
    }
}

ModelYOLO::~ModelYOLO() {
    obs_log(LOG_INFO, "[ModelYOLO] Destroyed");
}

void ModelYOLO::loadModel(const std::string& modelPath) {
    obs_log(LOG_INFO, "[ModelYOLO] Loading model: %s", modelPath.c_str());
    
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
#if _WIN32
        std::wstring modelPathW(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(*env_, modelPathW.c_str(), sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
#endif
        
        populateInputOutputNames(session_, inputNames_, outputNames_);
        populateInputOutputShapes(session_, inputDims_, outputDims_);
        allocateTensorBuffers(inputDims_, outputDims_, outputTensorValues_, inputTensorValues_,
                              inputTensor_, outputTensor_);
        
        if (!inputDims_.empty()) {
            auto shape = inputDims_[0];
            if (shape.size() >= 4) {
                inputHeight_ = static_cast<int>(shape[2]);
                inputWidth_ = static_cast<int>(shape[3]);
            }
        }
        
        if (!outputDims_.empty()) {
            auto shape = outputDims_[0];
            if (shape.size() >= 3) {
                int lastDim = static_cast<int>(shape[2]);
                numClasses_ = lastDim - 5;
            }
        }
        
        name = "YOLO";
        
        obs_log(LOG_INFO, "[ModelYOLO] Model loaded successfully");
        obs_log(LOG_INFO, "  Input size: %dx%d", inputWidth_, inputHeight_);
        obs_log(LOG_INFO, "  Num classes: %d", numClasses_);
        
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Failed to load model: %s", e.what());
        throw;
    }
}

void ModelYOLO::preprocessInput(const cv::Mat& input, float* outputBuffer) {
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(inputWidth_, inputHeight_));

    cv::Mat rgb;
    if (input.channels() == 4) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
    } else if (input.channels() == 3) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = resized.clone();
    }

    cv::Mat rgb8u;
    if (rgb.depth() != CV_8U) {
        rgb.convertTo(rgb8u, CV_8U);
    } else {
        rgb8u = rgb;
    }

    const int channelSize = inputWidth_ * inputHeight_;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < inputHeight_; ++h) {
            for (int w = 0; w < inputWidth_; ++w) {
                int outputIdx = c * channelSize + h * inputWidth_ + w;
                outputBuffer[outputIdx] = rgb8u.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}

std::vector<Detection> ModelYOLO::inference(const cv::Mat& input) {
    obs_log(LOG_DEBUG, "[ModelYOLO] inference called with image %dx%d", input.cols, input.rows);
    
    if (!session_) {
        obs_log(LOG_ERROR, "[ModelYOLO] Session is null, cannot run inference");
        return {};
    }
    
    try {
        std::vector<float> inputTensorData(1 * 3 * inputHeight_ * inputWidth_);
        preprocessInput(input, inputTensorData.data());
        
        std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
        
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, 
            OrtMemType::OrtMemTypeDefault
        );
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorData.data(),
            inputTensorData.size(),
            inputShape.data(),
            inputShape.size()
        );
        
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(inputTensor));
        
        std::vector<const char*> inputNamesChar;
        for (const auto& name : inputNames_) {
            inputNamesChar.push_back(name.get());
        }
        
        std::vector<const char*> outputNamesChar;
        for (const auto& name : outputNames_) {
            outputNamesChar.push_back(name.get());
        }
        
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNamesChar.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNamesChar.data(),
            outputNamesChar.size()
        );
        
        if (outputTensors.empty()) {
            obs_log(LOG_ERROR, "[ModelYOLO] No output tensors");
            return {};
        }
        
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        if (outputShape.size() < 3) {
            obs_log(LOG_ERROR, "[ModelYOLO] Invalid output shape");
            return {};
        }
        
        int numBoxes = static_cast<int>(outputShape[1]);
        int numElements = static_cast<int>(outputShape[2]);
        int detectedClasses = numElements - 5;
        
        obs_log(LOG_DEBUG, "[ModelYOLO] Output shape: [%lld, %lld, %lld]", 
                outputShape[0], outputShape[1], outputShape[2]);
        obs_log(LOG_DEBUG, "[ModelYOLO] Processing %d boxes", numBoxes);
        
        cv::Size modelSize(inputWidth_, inputHeight_);
        cv::Size originalSize(input.cols, input.rows);
        
        std::vector<Detection> detections;
        
        switch (version_) {
            case Version::YOLOv5:
            case Version::YOLOv8:
                detections = postprocessYOLOv5(outputData, numBoxes, detectedClasses, 
                                              modelSize, originalSize);
                break;
            case Version::YOLOv11:
                detections = postprocessYOLOv11(outputData, numBoxes, detectedClasses, 
                                               modelSize, originalSize);
                break;
        }
        
        obs_log(LOG_INFO, "[ModelYOLO] Inference completed, found %zu detections", detections.size());
        
        return detections;
        
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Inference exception: %s", e.what());
        return {};
    }
}

std::vector<Detection> ModelYOLO::postprocessYOLOv5(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    const int numElements = 5 + numClasses;

    float scaleX = static_cast<float>(originalImageSize.width) / modelInputSize.width;
    float scaleY = static_cast<float>(originalImageSize.height) / modelInputSize.height;

    for (int i = 0; i < numBoxes; ++i) {
        const float* detection = rawOutput + i * numElements;

        float objectness = detection[4];

        if (objectness < confidenceThreshold_) {
            continue;
        }

        int maxClassId = 0;
        float maxClassProb = detection[5];

        for (int c = 1; c < numClasses; ++c) {
            if (detection[5 + c] > maxClassProb) {
                maxClassProb = detection[5 + c];
                maxClassId = c;
            }
        }

        float confidence = objectness * maxClassProb;

        if (confidence < confidenceThreshold_) {
            continue;
        }

        if (targetClassId_ >= 0 && maxClassId != targetClassId_) {
            continue;
        }

        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];

        float x1 = (cx - w / 2.0f) * scaleX;
        float y1 = (cy - h / 2.0f) * scaleY;
        float x2 = (cx + w / 2.0f) * scaleX;
        float y2 = (cy + h / 2.0f) * scaleY;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(originalImageSize.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(originalImageSize.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(originalImageSize.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(originalImageSize.height)));

        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(confidence);
        classIds.push_back(maxClassId);
    }

    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);

    for (int idx : nmsIndices) {
        Detection det;
        det.classId = classIds[idx];
        det.className = (det.classId < classNames_.size())
                        ? classNames_[det.classId]
                        : "Class_" + std::to_string(det.classId);
        det.confidence = scores[idx];

        det.x = boxes[idx].x / originalImageSize.width;
        det.y = boxes[idx].y / originalImageSize.height;
        det.width = boxes[idx].width / originalImageSize.width;
        det.height = boxes[idx].height / originalImageSize.height;

        det.centerX = det.x + det.width / 2.0f;
        det.centerY = det.y + det.height / 2.0f;

        detections.push_back(det);
    }

    obs_log(LOG_DEBUG, "[ModelYOLO] Detected %zu objects after NMS", detections.size());

    return detections;
}

std::vector<Detection> ModelYOLO::postprocessYOLOv8(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    return postprocessYOLOv5(rawOutput, numBoxes, numClasses, modelInputSize, originalImageSize);
}

std::vector<Detection> ModelYOLO::postprocessYOLOv11(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    return postprocessYOLOv5(rawOutput, numBoxes, numClasses, modelInputSize, originalImageSize);
}

std::vector<int> ModelYOLO::performNMS(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float nmsThreshold
) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];

        if (suppressed[idx]) {
            continue;
        }

        keep.push_back(idx);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];

            if (suppressed[idx2]) {
                continue;
            }

            float iou = calculateIoU(boxes[idx], boxes[idx2]);

            if (iou > nmsThreshold) {
                suppressed[idx2] = true;
            }
        }
    }

    return keep;
}

float ModelYOLO::calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    if (x2 < x1 || y2 < y1) {
        return 0.0f;
    }

    float intersection = (x2 - x1) * (y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float unionArea = areaA + areaB - intersection;

    return intersection / unionArea;
}

void ModelYOLO::xywhToxyxy(float cx, float cy, float w, float h,
                            float& x1, float& y1, float& x2, float& y2) {
    x1 = cx - w / 2.0f;
    y1 = cy - h / 2.0f;
    x2 = cx + w / 2.0f;
    y2 = cy + h / 2.0f;
}

void ModelYOLO::loadClassNames(const std::string& namesFile) {
    std::ifstream file(namesFile);

    if (!file.is_open()) {
        obs_log(LOG_WARNING, "[ModelYOLO] Failed to open class names: %s",
                namesFile.c_str());
        return;
    }

    classNames_.clear();
    std::string line;

    while (std::getline(file, line)) {
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        if (!line.empty()) {
            classNames_.push_back(line);
        }
    }

    numClasses_ = static_cast<int>(classNames_.size());

    obs_log(LOG_INFO, "[ModelYOLO] Loaded %d class names", numClasses_);
}

void ModelYOLO::setConfidenceThreshold(float threshold) {
    confidenceThreshold_ = std::max(0.0f, std::min(threshold, 1.0f));
}

void ModelYOLO::setNMSThreshold(float threshold) {
    nmsThreshold_ = std::max(0.0f, std::min(threshold, 1.0f));
}

void ModelYOLO::setTargetClass(int classId) {
    targetClassId_ = classId;
}
