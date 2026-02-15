#include "ModelYOLO.h"
#include <plugin-support.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#ifdef HAVE_ONNXRUNTIME_DML_EP
#include <dml_provider_factory.h>
#endif

ModelYOLO::ModelYOLO(Version version)
    : ModelBCHW(),
      version_(version),
      confidenceThreshold_(0.5f),
      nmsThreshold_(0.45f),
      targetClassId_(-1),
      inputWidth_(640),
      inputHeight_(640),
      numClasses_(80),
      inputBufferSize_(0)
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

void ModelYOLO::loadModel(const std::string& modelPath, const std::string& useGPU, int numThreads, int inputResolution) {
    obs_log(LOG_INFO, "[ModelYOLO] Loading model: %s", modelPath.c_str());
    
    std::string currentUseGPU = useGPU;
    bool gpuFailed = false;
    
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        obs_log(LOG_INFO, "[ModelYOLO] Using device: %s", currentUseGPU.c_str());
        
        if (currentUseGPU != "cpu") {
            sessionOptions.DisableMemPattern();
            sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        } else {
            sessionOptions.SetInterOpNumThreads(numThreads);
            sessionOptions.SetIntraOpNumThreads(numThreads);
        }
        
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
        if (currentUseGPU == "cuda") {
            obs_log(LOG_INFO, "[ModelYOLO] Attempting to enable CUDA execution provider...");
            try {
                obs_log(LOG_INFO, "[ModelYOLO] Loading CUDA execution provider with device ID 0");
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
                obs_log(LOG_INFO, "[ModelYOLO] CUDA execution provider enabled successfully");
            } catch (const std::exception& e) {
                obs_log(LOG_WARNING, "[ModelYOLO] Failed to enable CUDA: %s, falling back to CPU", e.what());
                obs_log(LOG_INFO, "[ModelYOLO] CUDA execution provider fallback to CPU mode");
                obs_log(LOG_INFO, "[ModelYOLO] Possible reasons: missing cuDNN, incorrect CUDA version, or missing dependencies");
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif
#ifdef HAVE_ONNXRUNTIME_ROCM_EP
        if (currentUseGPU == "rocm" && !gpuFailed) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(sessionOptions, 0));
                obs_log(LOG_INFO, "[ModelYOLO] ROCM execution provider enabled");
            } catch (const std::exception& e) {
                obs_log(LOG_WARNING, "[ModelYOLO] Failed to enable ROCM: %s, falling back to CPU", e.what());
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif
#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
        if (currentUseGPU == "tensorrt" && !gpuFailed) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
                obs_log(LOG_INFO, "[ModelYOLO] TensorRT execution provider enabled");
            } catch (const std::exception& e) {
                obs_log(LOG_WARNING, "[ModelYOLO] Failed to enable TensorRT: %s, falling back to CPU", e.what());
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif

#ifdef HAVE_ONNXRUNTIME_DML_EP
        if (currentUseGPU == "dml" && !gpuFailed) {
            try {
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));
                obs_log(LOG_INFO, "[ModelYOLO] DirectML execution provider enabled");
            } catch (const std::exception& e) {
                obs_log(LOG_WARNING, "[ModelYOLO] Failed to enable DirectML: %s, falling back to CPU", e.what());
                gpuFailed = true;
                currentUseGPU = "cpu";
            }
        }
#endif
        
        if (gpuFailed) {
            sessionOptions.SetInterOpNumThreads(numThreads);
            sessionOptions.SetIntraOpNumThreads(numThreads);
            obs_log(LOG_INFO, "[ModelYOLO] Switched to CPU mode");
        }
        
#if _WIN32
        std::wstring modelPathW(modelPath.begin(), modelPath.end());
        session_ = std::make_unique<Ort::Session>(*env_, modelPathW.c_str(), sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), sessionOptions);
#endif
        
        populateInputOutputNames(session_, inputNames_, outputNames_);
        populateInputOutputShapes(session_, inputDims_, outputDims_);
        
        // 始终使用从模型读取的实际输入尺寸，而不是用户设置的 inputResolution
        if (!inputDims_.empty()) {
            auto shape = inputDims_[0];
            if (shape.size() >= 4) {
                inputHeight_ = static_cast<int>(shape[2]);
                inputWidth_ = static_cast<int>(shape[3]);
                obs_log(LOG_INFO, "[ModelYOLO] Using model actual input size: %dx%d", inputWidth_, inputHeight_);
            }
        }
        
        allocateTensorBuffers(inputDims_, outputDims_, outputTensorValues_, inputTensorValues_,
                              inputTensor_, outputTensor_);
        
        if (!outputDims_.empty()) {
            auto shape = outputDims_[0];
            obs_log(LOG_INFO, "[ModelYOLO] Output shape size: %zu", shape.size());
            for (size_t i = 0; i < shape.size(); ++i) {
                obs_log(LOG_INFO, "[ModelYOLO] Output shape[%zu]: %lld", i, shape[i]);
            }
            obs_log(LOG_INFO, "[ModelYOLO] Model version: %d", static_cast<int>(version_));
            
            int detectedClasses = 80; // default COCO classes
            
            if (version_ == Version::YOLOv5 && shape.size() >= 3) {
                int64_t lastDim = shape[2];
                if (lastDim > 5) {
                    detectedClasses = static_cast<int>(lastDim - 5);
                }
                obs_log(LOG_INFO, "[ModelYOLO] YOLOv5 mode: lastDim=%lld, detectedClasses=%d", lastDim, detectedClasses);
            } else if (shape.size() >= 3) {
                int64_t elementsDim = shape[1];
                if (elementsDim > 4) {
                    detectedClasses = static_cast<int>(elementsDim - 4);
                }
                obs_log(LOG_INFO, "[ModelYOLO] YOLOv8/v11 mode: elementsDim=%lld, detectedClasses=%d", elementsDim, detectedClasses);
            }
            
            // 验证 detectedClasses 是否合理（一般不会超过 1000 个类别）
            if (detectedClasses > 0 && detectedClasses < 1000) {
                numClasses_ = detectedClasses;
                obs_log(LOG_INFO, "[ModelYOLO] Using numClasses: %d (valid range)", numClasses_);
            } else {
                obs_log(LOG_WARNING, "[ModelYOLO] Detected numClasses %d is invalid, using default: 80", detectedClasses);
                numClasses_ = 80;
            }
        }
        
        // 预分配输入缓冲区
        inputBufferSize_ = 1 * 3 * inputHeight_ * inputWidth_;
        inputBuffer_.resize(inputBufferSize_);
        obs_log(LOG_INFO, "[ModelYOLO] Allocated input buffer size: %zu", inputBufferSize_);
        
        name = "YOLO";
        
        obs_log(LOG_INFO, "[ModelYOLO] Model loaded successfully");
        obs_log(LOG_INFO, "  Input size: %dx%d", inputWidth_, inputHeight_);
        obs_log(LOG_INFO, "  Num classes: %d", numClasses_);
        obs_log(LOG_INFO, "  Device: %s", currentUseGPU.c_str());
        
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
    
    // 输入验证
    if (input.empty()) {
        obs_log(LOG_ERROR, "[ModelYOLO] Input image is empty");
        return {};
    }
    
    if (input.cols <= 0 || input.rows <= 0) {
        obs_log(LOG_ERROR, "[ModelYOLO] Invalid input image size: %dx%d", input.cols, input.rows);
        return {};
    }
    
    if (!session_) {
        obs_log(LOG_ERROR, "[ModelYOLO] Session is null, cannot run inference");
        return {};
    }
    
    try {
        // 使用预分配的输入缓冲区
        preprocessInput(input, inputBuffer_.data());
        
        std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
        
        // 根据执行提供程序选择合适的内存分配器
        Ort::MemoryInfo memoryInfo;
        try {
            memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, 
                OrtMemType::OrtMemTypeDefault
            );
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to create memory info: %s", e.what());
            return {};
        }
        
        Ort::Value inputTensor;
        try {
            inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputBuffer_.data(),
                inputBufferSize_,
                inputShape.data(),
                inputShape.size()
            );
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to create input tensor: %s", e.what());
            return {};
        }
        
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
        
        obs_log(LOG_DEBUG, "[ModelYOLO] Running ONNX Runtime inference");
        
        // 添加详细的错误处理
        Ort::RunOptions runOptions;
        
        std::vector<Ort::Value> outputTensors;
        try {
            outputTensors = session_->Run(
                runOptions,
                inputNamesChar.data(),
                inputTensors.data(),
                inputTensors.size(),
                outputNamesChar.data(),
                outputNamesChar.size()
            );
        } catch (const Ort::Exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] ONNX Runtime exception during Run: %s", e.what());
            return {};
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Exception during Run: %s", e.what());
            return {};
        } catch (...) {
            obs_log(LOG_ERROR, "[ModelYOLO] Unknown exception during Run");
            return {};
        }
        
        if (outputTensors.empty()) {
            obs_log(LOG_ERROR, "[ModelYOLO] No output tensors from ONNX Runtime");
            return {};
        }
        
        if (!outputTensors[0].IsTensor()) {
            obs_log(LOG_ERROR, "[ModelYOLO] Output is not a tensor");
            return {};
        }
        
        float* outputData = nullptr;
        try {
            outputData = outputTensors[0].GetTensorMutableData<float>();
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to get output tensor data: %s", e.what());
            return {};
        }
        
        if (!outputData) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to get output tensor data");
            return {};
        }
        
        std::vector<int64_t> outputShape;
        try {
            outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to get output shape: %s", e.what());
            return {};
        }
        
        if (outputShape.size() < 3) {
            obs_log(LOG_ERROR, "[ModelYOLO] Invalid output shape size: %zu", outputShape.size());
            return {};
        }
        
        int numBoxes = 0, numElements = 0;
        
        try {
            if (version_ == Version::YOLOv5) {
                numBoxes = static_cast<int>(outputShape[1]);
                numElements = static_cast<int>(outputShape[2]);
            } else {
                numBoxes = static_cast<int>(outputShape[2]);
                numElements = static_cast<int>(outputShape[1]);
            }
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to parse output shape: %s", e.what());
            return {};
        }
        
        // 验证输出参数
        if (numBoxes <= 0 || numElements <= 0) {
            obs_log(LOG_ERROR, "[ModelYOLO] Invalid output parameters: numBoxes=%d, numElements=%d", numBoxes, numElements);
            return {};
        }
        
        obs_log(LOG_DEBUG, "[ModelYOLO] Using numClasses from model: %d", numClasses_);
	
	obs_log(LOG_DEBUG, "[ModelYOLO] Output shape: [%lld, %lld, %lld]", 
			outputShape[0], outputShape[1], outputShape[2]);
	obs_log(LOG_DEBUG, "[ModelYOLO] Processing %d boxes, %d classes", numBoxes, numClasses_);
        
        cv::Size modelSize(inputWidth_, inputHeight_);
        cv::Size originalSize(input.cols, input.rows);
        
        std::vector<Detection> detections;
        
        try {
            switch (version_) {
                case Version::YOLOv5:
                    detections = postprocessYOLOv5(outputData, numBoxes, numClasses_, 
                                                  modelSize, originalSize);
                    break;
                case Version::YOLOv8:
                    detections = postprocessYOLOv8(outputData, numBoxes, numClasses_, 
                                                  modelSize, originalSize);
                    break;
                case Version::YOLOv11:
                    detections = postprocessYOLOv11(outputData, numBoxes, numClasses_, 
                                                   modelSize, originalSize);
                    break;
            }
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Postprocessing exception: %s", e.what());
            return {};
        }
        
        // 只在有检测结果时输出日志，减少日志量
        if (!detections.empty()) {
            obs_log(LOG_INFO, "[ModelYOLO] Inference completed, found %zu detections", detections.size());
        } else {
            obs_log(LOG_DEBUG, "[ModelYOLO] Inference completed, found 0 detections");
        }
        
        return detections;
        
    } catch (const Ort::Exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] ONNX Runtime exception: %s", e.what());
        return {};
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Inference exception: %s", e.what());
        return {};
    } catch (...) {
        obs_log(LOG_ERROR, "[ModelYOLO] Unknown inference exception");
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
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    float scaleX = static_cast<float>(originalImageSize.width) / modelInputSize.width;
    float scaleY = static_cast<float>(originalImageSize.height) / modelInputSize.height;

    for (int i = 0; i < numBoxes; ++i) {
        float cx = rawOutput[0 * numBoxes + i];
        float cy = rawOutput[1 * numBoxes + i];
        float w = rawOutput[2 * numBoxes + i];
        float h = rawOutput[3 * numBoxes + i];

        int maxClassId = 0;
        float maxClassProb = rawOutput[4 * numBoxes + i];

        for (int c = 1; c < numClasses; ++c) {
            float prob = rawOutput[(4 + c) * numBoxes + i];
            if (prob > maxClassProb) {
                maxClassProb = prob;
                maxClassId = c;
            }
        }

        float confidence = maxClassProb;

        if (confidence < confidenceThreshold_) {
            continue;
        }

        if (targetClassId_ >= 0 && maxClassId != targetClassId_) {
            continue;
        }

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

std::vector<Detection> ModelYOLO::postprocessYOLOv11(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    return postprocessYOLOv8(rawOutput, numBoxes, numClasses, modelInputSize, originalImageSize);
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

void ModelYOLO::setInputResolution(int resolution) {
    // 禁用手动设置输入分辨率，始终使用模型实际输入尺寸
    obs_log(LOG_WARNING, "[ModelYOLO] setInputResolution is disabled. Input resolution is determined by model.");
    obs_log(LOG_WARNING, "[ModelYOLO] Current model input size: %dx%d", inputWidth_, inputHeight_);
    // 不执行任何修改
}
