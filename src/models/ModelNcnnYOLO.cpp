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
    // pnnx 生成 xxx.ncnn.param/xxx.ncnn.bin，优先尝试
    std::string paramPath = basePath + ".ncnn.param";
    std::string binPath = basePath + ".ncnn.bin";
    bool foundNcnnSuffix = false;
    {
        FILE* fp = nullptr;
        fopen_s(&fp, paramPath.c_str(), "rb");
        if (fp) { fclose(fp); foundNcnnSuffix = true; }
    }
    if (!foundNcnnSuffix) {
        paramPath = basePath + ".param";
        binPath = basePath + ".bin";
    }
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

        // 扫描全部 blob，提取检测头（3D: [c, h, w]，c 为 3*(5+cls)）
        struct HeadBlob {
            ncnn::Mat mat;
            int idx;
            int channels;
            int headH;
            int headW;
            int total;
        };
        std::vector<HeadBlob> heads;
        int nBlobs = (int)net_.blobs().size();

        for (int bi = 0; bi < nBlobs; bi++) {
            ncnn::Mat tmp;
            int r = ex.extract(bi, tmp);
            if (r != 0 || tmp.data == nullptr || tmp.total() == 0) continue;
            if (tmp.dims != 3) continue;
            int c = tmp.c;
            if (c < 6 || c > 300 || c % 3 != 0) continue;
            // c = 3*(5+cls) → cls = c/3 - 5
            obs_log(LOG_INFO, "[ModelNcnnYOLO] head blob %d: c=%d h=%d w=%d total=%d",
                    bi, c, tmp.h, tmp.w, (int)tmp.total());
            heads.push_back({tmp, bi, c, tmp.h, tmp.w, (int)tmp.total()});
        }

        // 按 total 排序（大→小 = stride 8/16/32）
        std::sort(heads.begin(), heads.end(),
                  [](const HeadBlob& a, const HeadBlob& b) { return a.total > b.total; });

        if (heads.empty()) {
            obs_log(LOG_ERROR, "[ModelNcnnYOLO] no detection head blobs found");
            return {};
        }

        // 取头 3 个 head（或全部）作为检测头
        int nHeads = std::min((int)heads.size(), 3);
        int detectedClasses = heads[0].channels / 3 - 5;
        if (detectedClasses > 0 && detectedClasses < 1000) {
            numClasses_ = detectedClasses;
        }
        obs_log(LOG_INFO, "[ModelNcnnYOLO] using %d heads, classes=%d", nHeads, numClasses_);

        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();

        cv::Size originalSize(input.cols, input.rows);
        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        std::vector<Detection> detections;

        // 解析每个检测头
        int strideVals[3] = {8, 16, 32}; // 按 total 从大到小对应 stride
        for (int hi = 0; hi < nHeads; hi++) {
            const HeadBlob& hb = heads[hi];
            int stride = strideVals[hi];
            int h = hb.headH;
            int w = hb.headW;
            int c = hb.channels;
            int numAnchors = 3;
            int elemPerAnchor = c / numAnchors; // = 5+cls

            const float* data = (const float*)hb.mat.data;

            for (int ai = 0; ai < numAnchors; ai++) {
                int chBase = ai * elemPerAnchor;
                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        // 每个 anchor 在每个 grid 位置提取 9 个值
                        float cx = hb.mat.channel(chBase + 0)[i * w + j];
                        float cy = hb.mat.channel(chBase + 1)[i * w + j];
                        float bw = hb.mat.channel(chBase + 2)[i * w + j];
                        float bh = hb.mat.channel(chBase + 3)[i * w + j];
                        float obj = hb.mat.channel(chBase + 4)[i * w + j];
                        if (obj < confidenceThreshold_) continue;
                        int bestCls = 0;
                        float bestProb = hb.mat.channel(chBase + 5)[i * w + j];
                        for (int ci = 1; ci < numClasses_; ci++) {
                            float p = hb.mat.channel(chBase + 5 + ci)[i * w + j];
                            if (p > bestProb) { bestProb = p; bestCls = ci; }
                        }
                        float conf = obj * bestProb;
                        if (conf < confidenceThreshold_) continue;

                        bool isTargetClass = false;
                        if (targetClassId_ >= 0) {
                            isTargetClass = (bestCls == targetClassId_);
                        } else if (!targetClasses_.empty()) {
                            isTargetClass = targetClasses_.count(bestCls);
                        } else {
                            isTargetClass = true;
                        }
                        if (!isTargetClass) continue;

                        // 已经绝对坐标，做 letterbox 反算
                        float x1 = (cx - bw / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
                        float y1 = (cy - bh / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
                        float x2 = (cx + bw / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
                        float y2 = (cy + bh / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
                        x1 = std::max(0.0f, std::min(x1, (float)originalSize.width));
                        y1 = std::max(0.0f, std::min(y1, (float)originalSize.height));
                        x2 = std::max(0.0f, std::min(x2, (float)originalSize.width));
                        y2 = std::max(0.0f, std::min(y2, (float)originalSize.height));

                        Detection det;
                        det.classId = bestCls;
                        det.className = (bestCls < (int)classNames_.size()) ? classNames_[bestCls] : "Cls" + std::to_string(bestCls);
                        det.confidence = conf;
                        det.x = x1 / originalSize.width;
                        det.y = y1 / originalSize.height;
                        det.width = (x2 - x1) / originalSize.width;
                        det.height = (y2 - y1) / originalSize.height;
                        det.centerX = det.x + det.width / 2.0f;
                        det.centerY = det.y + det.height / 2.0f;
                        detections.push_back(det);
                    }
                }
            }
        }

        // NMS
        if (!detections.empty()) {
            std::vector<cv::Rect2f> boxes;
            std::vector<float> scores;
            std::vector<int> classIds;
            for (auto& d : detections) {
                boxes.push_back(cv::Rect2f(d.x * originalSize.width, d.y * originalSize.height,
                                           d.width * originalSize.width, d.height * originalSize.height));
                scores.push_back(d.confidence);
                classIds.push_back(d.classId);
            }
            std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);
            std::vector<Detection> filtered;
            for (int idx : nmsIndices) filtered.push_back(detections[idx]);
            detections = filtered;
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
