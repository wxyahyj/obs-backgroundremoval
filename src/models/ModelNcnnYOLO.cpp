#include "ModelNcnnYOLO.h"
#include <plugin-support.h>
#include <util/base.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <chrono>

ModelNcnnYOLO::LetterboxInfo ModelNcnnYOLO::calculateLetterboxParams(int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    LetterboxInfo info;
    float scale = std::min((float)dstWidth / srcWidth, (float)dstHeight / srcHeight);
    int newWidth = (int)(srcWidth * scale);
    int newHeight = (int)(srcHeight * scale);
    info.scale = scale;
    info.padX = (dstWidth - newWidth) / 2;
    info.padY = (dstHeight - newHeight) / 2;
    return info;
}

ModelNcnnYOLO::LetterboxInfo ModelNcnnYOLO::letterbox(const cv::Mat& input, cv::Mat& output) {
    LetterboxInfo info = calculateLetterboxParams(input.cols, input.rows, inputWidth_, inputHeight_);
    int newWidth = (int)(input.cols * info.scale);
    int newHeight = (int)(input.rows * info.scale);
    cv::resize(input, resizedBuffer_, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    const cv::Scalar padColor(114, 114, 114, 114);
    if (letterboxBuffer_.rows != inputHeight_ || letterboxBuffer_.cols != inputWidth_ || letterboxBuffer_.type() != input.type()) {
        letterboxBuffer_.create(inputHeight_, inputWidth_, input.type());
        letterboxBuffer_.setTo(padColor);
    } else {
        letterboxBuffer_.setTo(padColor);
    }
    resizedBuffer_.copyTo(letterboxBuffer_(cv::Rect(info.padX, info.padY, newWidth, newHeight)));
    output = letterboxBuffer_;
    return info;
}

ModelNcnnYOLO::ModelNcnnYOLO(Version version)
    : version_(version),
      confidenceThreshold_(0.25f),
      nmsThreshold_(0.45f),
      targetClassId_(-1),
      inputWidth_(640),
      inputHeight_(640),
      numClasses_(80),
      vulkanEnabled_(false),
      modelLoaded_(false),
      lastLatencyLogTime_(std::chrono::steady_clock::now()),
      inferenceThreadRunning_(false) {
    obs_log(LOG_INFO, "[ModelNcnnYOLO] Initialized (Version: %d)", static_cast<int>(version));
    inferenceThreadRunning_ = true;
    inferenceThread_ = std::thread([this]() {
        while (inferenceThreadRunning_) {
            std::unique_ptr<InferenceTask> task;
            {
                std::unique_lock<std::mutex> lock(inferenceTasksMutex_);
                inferenceTasksCV_.wait(lock, [this]() {
                    return !inferenceThreadRunning_ || !inferenceTasks_.empty();
                });
                if (!inferenceThreadRunning_) break;
                if (!inferenceTasks_.empty()) {
                    task = std::move(inferenceTasks_.front());
                    inferenceTasks_.pop();
                }
            }
            if (task) {
                try {
                    std::vector<Detection> results = doInference(task->input);
                    task->promise.set_value(results);
                } catch (const std::exception& e) {
                    obs_log(LOG_ERROR, "[ModelNcnnYOLO] Async inference error: %s", e.what());
                    task->promise.set_value({});
                }
            }
        }
    });
}

ModelNcnnYOLO::~ModelNcnnYOLO() {
    inferenceThreadRunning_ = false;
    inferenceTasksCV_.notify_all();
    if (inferenceThread_.joinable()) {
        inferenceThread_.join();
    }
    obs_log(LOG_INFO, "[ModelNcnnYOLO] Destroyed");
}

void ModelNcnnYOLO::loadModel(const std::string& modelPath, const std::string& useGPU, int numThreads, int inputResolution) {
    obs_log(LOG_INFO, "[ModelNcnnYOLO] Loading model: %s", modelPath.c_str());
    std::string basePath = modelPath;
    if (basePath.size() > 5 && basePath.substr(basePath.size() - 5) == ".onnx") {
        basePath = basePath.substr(0, basePath.size() - 5);
    }
    std::string paramPath = basePath + ".param";
    std::string binPath = basePath + ".bin";
    obs_log(LOG_INFO, "[ModelNcnnYOLO] Param file: %s", paramPath.c_str());
    obs_log(LOG_INFO, "[ModelNcnnYOLO] Bin file: %s", binPath.c_str());
    ncnn::Option opt;
    opt.num_threads = numThreads;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.lightmode = false;
    vulkanEnabled_ = (useGPU == "ncnn" || useGPU == "vulkan");
    if (vulkanEnabled_) {
        opt.use_vulkan_compute = true;
        ncnn::create_gpu_instance();
        obs_log(LOG_INFO, "[ModelNcnnYOLO] Vulkan GPU acceleration enabled");
    } else {
        opt.use_vulkan_compute = false;
        obs_log(LOG_INFO, "[ModelNcnnYOLO] Using CPU mode, threads: %d", numThreads);
    }
    net_.opt = opt;
    int ret = net_.load_param(paramPath.c_str());
    if (ret != 0) {
        obs_log(LOG_ERROR, "[ModelNcnnYOLO] Failed to load param file: %s (ret=%d)", paramPath.c_str(), ret);
        throw std::runtime_error("Failed to load ncnn param file: " + paramPath);
    }
    ret = net_.load_model(binPath.c_str());
    if (ret != 0) {
        obs_log(LOG_ERROR, "[ModelNcnnYOLO] Failed to load bin file: %s (ret=%d)", binPath.c_str(), ret);
        throw std::runtime_error("Failed to load ncnn bin file: " + binPath);
    }
    modelLoaded_ = true;
    inputWidth_ = inputResolution > 0 ? inputResolution : 640;
    inputHeight_ = inputResolution > 0 ? inputResolution : 640;
    obs_log(LOG_INFO, "[ModelNcnnYOLO] Model loaded successfully");
    obs_log(LOG_INFO, "[ModelNcnnYOLO]   Input size: %dx%d", inputWidth_, inputHeight_);
    obs_log(LOG_INFO, "[ModelNcnnYOLO]   Num classes: %d", numClasses_);
    obs_log(LOG_INFO, "[ModelNcnnYOLO]   Device: %s", vulkanEnabled_ ? "Vulkan GPU" : "CPU");
}

std::vector<Detection> ModelNcnnYOLO::inference(const cv::Mat& input) {
    return doInference(input);
}

std::future<std::vector<Detection>> ModelNcnnYOLO::asyncInference(const cv::Mat& input) {
    auto task = std::make_unique<InferenceTask>();
    task->input = input.clone();
    auto future = task->promise.get_future();
    {
        std::lock_guard<std::mutex> lock(inferenceTasksMutex_);
        inferenceTasks_.push(std::move(task));
    }
    inferenceTasksCV_.notify_one();
    return future;
}

std::vector<Detection> ModelNcnnYOLO::doInference(const cv::Mat& input) {
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    InferenceLatency latency;
    if (input.empty() || input.cols <= 0 || input.rows <= 0) {
        obs_log(LOG_ERROR, "[ModelNcnnYOLO] Invalid input image");
        return {};
    }
    if (!modelLoaded_) {
        obs_log(LOG_ERROR, "[ModelNcnnYOLO] Model not loaded");
        return {};
    }
    try {
        auto preprocessStartTime = std::chrono::high_resolution_clock::now();
        cv::Mat preprocessed;
        LetterboxInfo letterboxInfo = letterbox(input, preprocessed);
        ncnn::Mat inputMat = ncnn::Mat::from_pixels(preprocessed.data, ncnn::Mat::PIXEL_BGR2RGB, inputWidth_, inputHeight_);
        const float normVals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
        inputMat.substract_mean_normalize(nullptr, normVals);
        auto preprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.preprocessMs = std::chrono::duration<double, std::milli>(preprocessEndTime - preprocessStartTime).count();
        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        ncnn::Extractor ex = net_.create_extractor();
        ex.input(0, inputMat);

        // 提取输出 blob：遍历所有注册的输出索引
        ncnn::Mat outputMat;
        int extractRet = -1;

        const std::vector<int>& outIdxs = net_.output_indexes();

        // 尝试每个 output_index（pnnx 有 4 个输出: out1, out2, out3, out0）
        for (size_t oi = 0; oi < outIdxs.size(); oi++) {
            extractRet = ex.extract(outIdxs[oi], outputMat);
            obs_log(LOG_INFO, "[ModelNcnnYOLO] extract output_index[%zu]=%d (ret=%d)",
                    oi, outIdxs[oi], extractRet);
            if (extractRet == 0 && outputMat.data != nullptr && outputMat.total() > 0)
                break;
        }

        // 回退1：按 blob 名 "out0" 提取
        if (extractRet != 0 || outputMat.data == nullptr || outputMat.total() == 0) {
            obs_log(LOG_INFO, "[ModelNcnnYOLO] trying name 'out0'");
            extractRet = ex.extract("out0", outputMat);
        }

        // 回退2：从末尾倒序提第一个有效 blob
        if (extractRet != 0 || outputMat.data == nullptr || outputMat.total() == 0) {
            int nb = (int)net_.blobs().size();
            obs_log(LOG_INFO, "[ModelNcnnYOLO] scanning %d blobs from end", nb);
            for (int bi = nb - 1; bi >= 0; bi--) {
                ncnn::Mat tmp;
                int r = ex.extract(bi, tmp);
                if (r == 0 && tmp.data != nullptr && tmp.total() > 0) {
                    outputMat = tmp;
                    extractRet = 0;
                    obs_log(LOG_INFO, "[ModelNcnnYOLO] fallback blob %d: dims=%d c=%d h=%d w=%d total=%d",
                            bi, tmp.dims, tmp.c, tmp.h, tmp.w, (int)tmp.total());
                    break;
                }
            }
        }

        if (extractRet != 0 || outputMat.data == nullptr || outputMat.total() == 0) {
            obs_log(LOG_ERROR, "[ModelNcnnYOLO] all extract methods failed");
            return {};
        }

        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();

        // pnnx 输出为 2D: [h, w] = [boxes, elements]
        // 或 3D: [c=1, h=boxes, w=elements]
        int dims = outputMat.dims;
        int boxDim = (dims >= 2) ? outputMat.h : outputMat.w;
        int elemDim = outputMat.w;
        if (dims >= 3 && outputMat.c > 1) {
            boxDim = outputMat.h * outputMat.w;
            elemDim = outputMat.c;
        }

        int numBoxes = boxDim;
        int numElements = elemDim;
        int detectedClasses = 80;
        if (version_ == Version::YOLOv5) {
            if (numElements > 5) detectedClasses = numElements - 5;
        } else {
            if (numElements > 4) detectedClasses = numElements - 4;
        }
        if (detectedClasses > 0 && detectedClasses < 1000) {
            numClasses_ = detectedClasses;
        }

        obs_log(LOG_INFO, "[ModelNcnnYOLO] Output: dims=%d c=%d h=%d w=%d boxes=%d elems=%d classes=%d",
                dims, outputMat.c, outputMat.h, outputMat.w, numBoxes, numElements, numClasses_);

        const float* outputData = (const float*)outputMat.data;

        if (numBoxes <= 0 || numElements <= 0) {
            obs_log(LOG_ERROR, "[ModelNcnnYOLO] Invalid output: boxes=%d, elements=%d", numBoxes, numElements);
            return {};
        }

        cv::Size originalSize(input.cols, input.rows);
        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        std::vector<Detection> detections;
        switch (version_) {
            case Version::YOLOv5:
                detections = postprocessYOLOv5(outputData, numBoxes, numClasses_, letterboxInfo, originalSize);
                break;
            case Version::YOLOv8:
                detections = postprocessYOLOv8(outputData, numBoxes, numClasses_, letterboxInfo, originalSize);
                break;
            case Version::YOLOv11:
                detections = postprocessYOLOv11(outputData, numBoxes, numClasses_, letterboxInfo, originalSize);
                break;
        }

        auto postprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.postprocessMs = std::chrono::duration<double, std::milli>(postprocessEndTime - postprocessStartTime).count();
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        latency.totalMs = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();
        latency.isGpuPath = vulkanEnabled_;
        latencyStats_.addSample(latency);

        obs_log(LOG_INFO, "[ModelNcnnYOLO] Detections: %zu", detections.size());

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLatencyLogTime_).count() >= LATENCY_LOG_INTERVAL_MS) {
            obs_log(LOG_INFO, "[ModelNcnnYOLO] 延迟统计:\n%s", latencyStats_.getSummary().c_str());
            lastLatencyLogTime_ = now;
        }
        return detections;
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelNcnnYOLO] Inference exception: %s", e.what());
        return {};
    } catch (...) {
        obs_log(LOG_ERROR, "[ModelNcnnYOLO] Unknown inference exception");
        return {};
    }
}

std::vector<Detection> ModelNcnnYOLO::postprocessYOLOv5(
    const float* rawOutput, int numBoxes, int numClasses,
    const LetterboxInfo& letterboxInfo, const cv::Size& originalImageSize) {
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    const int numElements = 5 + numClasses;
    for (int i = 0; i < numBoxes; ++i) {
        const float* detection = rawOutput + i * numElements;
        float objectness = detection[4];
        if (objectness < confidenceThreshold_) continue;
        int maxClassId = 0;
        float maxClassProb = detection[5];
        for (int c = 1; c < numClasses; ++c) {
            if (detection[5 + c] > maxClassProb) {
                maxClassProb = detection[5 + c];
                maxClassId = c;
            }
        }
        float confidence = objectness * maxClassProb;
        if (confidence < confidenceThreshold_) continue;
        bool isTargetClass = false;
        if (targetClassId_ >= 0) {
            isTargetClass = (maxClassId == targetClassId_);
        } else if (!targetClasses_.empty()) {
            isTargetClass = targetClasses_.count(maxClassId);
        } else {
            isTargetClass = true;
        }
        if (!isTargetClass) continue;
        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];
        float x1 = (cx - w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y1 = (cy - h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
        float x2 = (cx + w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y2 = (cy + h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
        x1 = std::max(0.0f, std::min(x1, (float)originalImageSize.width));
        y1 = std::max(0.0f, std::min(y1, (float)originalImageSize.height));
        x2 = std::max(0.0f, std::min(x2, (float)originalImageSize.width));
        y2 = std::max(0.0f, std::min(y2, (float)originalImageSize.height));
        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(confidence);
        classIds.push_back(maxClassId);
    }
    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);
    for (int idx : nmsIndices) {
        Detection det;
        det.classId = classIds[idx];
        det.className = (det.classId < (int)classNames_.size()) ? classNames_[det.classId] : "Class_" + std::to_string(det.classId);
        det.confidence = scores[idx];
        det.x = boxes[idx].x / originalImageSize.width;
        det.y = boxes[idx].y / originalImageSize.height;
        det.width = boxes[idx].width / originalImageSize.width;
        det.height = boxes[idx].height / originalImageSize.height;
        det.centerX = det.x + det.width / 2.0f;
        det.centerY = det.y + det.height / 2.0f;
        detections.push_back(det);
    }
    return detections;
}

std::vector<Detection> ModelNcnnYOLO::postprocessYOLOv8(
    const float* rawOutput, int numBoxes, int numClasses,
    const LetterboxInfo& letterboxInfo, const cv::Size& originalImageSize) {
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    const int numElements = 4 + numClasses;
    for (int i = 0; i < numBoxes; ++i) {
        const float* detection = rawOutput + i * numElements;
        float maxClassProb = detection[4];
        int maxClassId = 0;
        for (int c = 1; c < numClasses; ++c) {
            if (detection[4 + c] > maxClassProb) {
                maxClassProb = detection[4 + c];
                maxClassId = c;
            }
        }
        if (maxClassProb < confidenceThreshold_) continue;
        bool isTargetClass = false;
        if (targetClassId_ >= 0) {
            isTargetClass = (maxClassId == targetClassId_);
        } else if (!targetClasses_.empty()) {
            isTargetClass = targetClasses_.count(maxClassId);
        } else {
            isTargetClass = true;
        }
        if (!isTargetClass) continue;
        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];
        float x1 = (cx - w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y1 = (cy - h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
        float x2 = (cx + w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y2 = (cy + h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
        x1 = std::max(0.0f, std::min(x1, (float)originalImageSize.width));
        y1 = std::max(0.0f, std::min(y1, (float)originalImageSize.height));
        x2 = std::max(0.0f, std::min(x2, (float)originalImageSize.width));
        y2 = std::max(0.0f, std::min(y2, (float)originalImageSize.height));
        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(maxClassProb);
        classIds.push_back(maxClassId);
    }
    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);
    for (int idx : nmsIndices) {
        Detection det;
        det.classId = classIds[idx];
        det.className = (det.classId < (int)classNames_.size()) ? classNames_[det.classId] : "Class_" + std::to_string(det.classId);
        det.confidence = scores[idx];
        det.x = boxes[idx].x / originalImageSize.width;
        det.y = boxes[idx].y / originalImageSize.height;
        det.width = boxes[idx].width / originalImageSize.width;
        det.height = boxes[idx].height / originalImageSize.height;
        det.centerX = det.x + det.width / 2.0f;
        det.centerY = det.y + det.height / 2.0f;
        detections.push_back(det);
    }
    return detections;
}

std::vector<Detection> ModelNcnnYOLO::postprocessYOLOv11(
    const float* rawOutput, int numBoxes, int numClasses,
    const LetterboxInfo& letterboxInfo, const cv::Size& originalImageSize) {
    return postprocessYOLOv8(rawOutput, numBoxes, numClasses, letterboxInfo, originalImageSize);
}

std::vector<int> ModelNcnnYOLO::performNMS(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float nmsThreshold) {
    std::vector<int> indices;
    std::vector<int> sorted(scores.size());
    for (int i = 0; i < (int)scores.size(); i++) sorted[i] = i;
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<bool> suppressed(scores.size(), false);
    for (int i = 0; i < (int)sorted.size(); i++) {
        if (suppressed[sorted[i]]) continue;
        indices.push_back(sorted[i]);
        for (int j = i + 1; j < (int)sorted.size(); j++) {
            if (suppressed[sorted[j]]) continue;
            float iou = calculateIoU(boxes[sorted[i]], boxes[sorted[j]]);
            if (iou > nmsThreshold) suppressed[sorted[j]] = true;
        }
    }
    return indices;
}

float ModelNcnnYOLO::calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float interX1 = std::max(a.x, b.x);
    float interY1 = std::max(a.y, b.y);
    float interX2 = std::min(a.x + a.width, b.x + b.width);
    float interY2 = std::min(a.y + a.height, b.y + b.height);
    float interArea = std::max(0.0f, interX2 - interX1) * std::max(0.0f, interY2 - interY1);
    float unionArea = a.width * a.height + b.width * b.height - interArea;
    return unionArea > 0 ? interArea / unionArea : 0;
}

void ModelNcnnYOLO::loadClassNames(const std::string& namesFile) {
    std::ifstream file(namesFile);
    if (!file.is_open()) {
        obs_log(LOG_WARNING, "[ModelNcnnYOLO] Failed to open class names: %s", namesFile.c_str());
        return;
    }
    classNames_.clear();
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find(' ');
        if (pos != std::string::npos) {
            classNames_.push_back(line.substr(pos + 1));
        } else {
            classNames_.push_back(line);
        }
    }
    numClasses_ = static_cast<int>(classNames_.size());
    obs_log(LOG_INFO, "[ModelNcnnYOLO] Loaded %d class names", numClasses_);
}

void ModelNcnnYOLO::setConfidenceThreshold(float threshold) { confidenceThreshold_ = threshold; }
void ModelNcnnYOLO::setNMSThreshold(float threshold) { nmsThreshold_ = threshold; }
void ModelNcnnYOLO::setTargetClass(int classId) { targetClassId_ = classId; }
void ModelNcnnYOLO::setTargetClasses(const std::vector<int>& classIds) { targetClasses_ = std::unordered_set<int>(classIds.begin(), classIds.end()); }
void ModelNcnnYOLO::setInputResolution(int resolution) {
    if (resolution > 0) {
        inputWidth_ = resolution;
        inputHeight_ = resolution;
    }
}

std::vector<Detection> ModelNcnnYOLO::inferenceFromTextureDml(
    const DmlPreprocessedFrame&, int, int, InferenceLatency*) { return {}; }
std::vector<Detection> ModelNcnnYOLO::inferenceFromTexture(
    void*, int, int, int, int, InferenceLatency*) { return {}; }
