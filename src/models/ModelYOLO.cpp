#include "ModelYOLO.h"
#include <plugin-support.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#ifdef HAVE_ONNXRUNTIME_DML_EP
#include <dml_provider_factory.h>
#include <d3d11.h>
#endif
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/stat.h>
#endif

// CUDA头文件（阶段2）
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "CudaPreprocessor.cuh"
#endif

ModelYOLO::LetterboxInfo ModelYOLO::calculateLetterboxParams(int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    LetterboxInfo info;
    
    float scaleX = static_cast<float>(dstWidth) / srcWidth;
    float scaleY = static_cast<float>(dstHeight) / srcHeight;
    info.scale = std::min(scaleX, scaleY);
    
    int newWidth = static_cast<int>(srcWidth * info.scale);
    int newHeight = static_cast<int>(srcHeight * info.scale);
    
    info.padX = (dstWidth - newWidth) / 2;
    info.padY = (dstHeight - newHeight) / 2;
    
    return info;
}

ModelYOLO::LetterboxInfo ModelYOLO::letterbox(const cv::Mat& input, cv::Mat& output) {
    LetterboxInfo info;
    
    float scaleX = static_cast<float>(inputWidth_) / input.cols;
    float scaleY = static_cast<float>(inputHeight_) / input.rows;
    info.scale = std::min(scaleX, scaleY);
    
    int newWidth = static_cast<int>(input.cols * info.scale);
    int newHeight = static_cast<int>(input.rows * info.scale);
    
    info.padX = (inputWidth_ - newWidth) / 2;
    info.padY = (inputHeight_ - newHeight) / 2;
    
    // 使用预分配的resized缓冲区
    if (resizedBuffer_.rows != newHeight || resizedBuffer_.cols != newWidth) {
        resizedBuffer_ = cv::Mat(newHeight, newWidth, input.type());
    }
    cv::resize(input, resizedBuffer_, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    
    // 使用预分配的letterbox输出缓冲区
    if (letterboxBuffer_.rows != inputHeight_ || letterboxBuffer_.cols != inputWidth_) {
        letterboxBuffer_ = cv::Mat(inputHeight_, inputWidth_, input.type(), cv::Scalar(114, 114, 114, 114));
    } else {
        // 重置填充区域为灰色
        letterboxBuffer_.setTo(cv::Scalar(114, 114, 114, 114));
    }
    resizedBuffer_.copyTo(letterboxBuffer_(cv::Rect(info.padX, info.padY, newWidth, newHeight)));
    
    // 输出指向预分配缓冲区
    output = letterboxBuffer_;
    
    return info;
}

ModelYOLO::ModelYOLO(Version version)
    : ModelBCHW(),
      version_(version),
      confidenceThreshold_(0.5f),
      nmsThreshold_(0.45f),
      targetClassId_(-1),
      inputWidth_(640),
      inputHeight_(640),
      numClasses_(80),
      inputBufferSize_(0),
      useIOBinding_(false),
      useGpuMemory_(false),
      gpuAllocator_(nullptr),
      gpuMemInfo_(nullptr),
      cudaInteropInitialized_(false),
      cudaStream_(nullptr),
      cudaResource_(nullptr),
      cudaInputBuffer_(nullptr),
      dmlInteropInitialized_(false),
      dmlPreprocessor_(nullptr),
      lastLatencyLogTime_(std::chrono::steady_clock::now()),
      inferenceThreadRunning_(false)
{
    obs_log(LOG_INFO, "[ModelYOLO] Initialized (Version: %d)", static_cast<int>(version));
    
    try {
        std::string instanceName{"YOLOModel"};
        env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str());
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Failed to initialize ORT: %s", e.what());
    }
    
    // 启动推理线程
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
                    obs_log(LOG_ERROR, "[ModelYOLO] Async inference error: %s", e.what());
                    task->promise.set_value({});
                }
            }
        }
    });
}

ModelYOLO::~ModelYOLO() {
    // 停止推理线程
    inferenceThreadRunning_ = false;
    inferenceTasksCV_.notify_one();
    if (inferenceThread_.joinable()) {
        inferenceThread_.join();
    }
    
    // 释放DML互操作资源
    releaseDmlInterop();
    
    // 释放CUDA互操作资源
    releaseCudaInterop();
    
    // 释放GPU内存
    releaseGpuMemory();
    
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
            
            // Arena配置 - 减少内存预分配
            sessionOptions.AddConfigEntry("session.arena_extend_strategy", "kSameAsRequested");
            sessionOptions.AddConfigEntry("memory.enable_memory_arena_shrinkage", "1");
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
                OrtTensorRTProviderOptions trt_options;
                memset(&trt_options, 0, sizeof(trt_options));
                
                trt_options.trt_engine_cache_enable = 1;
                trt_options.trt_fp16_enable = 1;
                trt_options.trt_max_workspace_size = 1ULL << 30;  // 1GB工作空间
                trt_options.trt_dla_enable = 0;  // DLA核心（Jetson设备可用）
                trt_options.trt_dla_core = 0;
                trt_options.trt_int8_enable = 0;  // INT8需要校准，暂不启用
                trt_options.trt_max_partition_iterations = 1000;  // 最大分区迭代次数
                trt_options.trt_min_subgraph_size = 1;  // 最小子图大小
                
#ifdef _WIN32
                std::wstring modelPathW(modelPath.begin(), modelPath.end());
                size_t lastSlash = modelPathW.find_last_of(L"\\/");
                if (lastSlash != std::wstring::npos) {
                    std::wstring cachePathW = modelPathW.substr(0, lastSlash) + L"\\trt_cache";
                    CreateDirectoryW(cachePathW.c_str(), NULL);
                    
                    std::string cachePathNarrow;
                    int len = WideCharToMultiByte(CP_ACP, 0, cachePathW.c_str(), -1, NULL, 0, NULL, NULL);
                    cachePathNarrow.resize(len);
                    WideCharToMultiByte(CP_ACP, 0, cachePathW.c_str(), -1, &cachePathNarrow[0], len, NULL, NULL);
                    cachePathNarrow.pop_back();
                    
                    trt_options.trt_engine_cache_path = _strdup(cachePathNarrow.c_str());
                    
                    obs_log(LOG_INFO, "[ModelYOLO] TensorRT cache path: %s", cachePathNarrow.c_str());
                }
#else
                size_t lastSlash = modelPath.find_last_of("/");
                if (lastSlash != std::string::npos) {
                    char cachePath[1024];
                    snprintf(cachePath, sizeof(cachePath), "%s/trt_cache", modelPath.substr(0, lastSlash).c_str());
                    mkdir(cachePath, 0755);
                    trt_options.trt_engine_cache_path = strdup(cachePath);
                    obs_log(LOG_INFO, "[ModelYOLO] TensorRT cache path: %s", cachePath);
                }
#endif
                sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
                obs_log(LOG_INFO, "[ModelYOLO] TensorRT execution provider enabled with cache");
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
        
        // 初始化IOBinding（仅GPU模式）
        if (currentUseGPU != "cpu") {
            try {
                ioBinding_ = std::make_unique<Ort::IoBinding>(*session_);
                
                // 预分配输出缓冲区
                size_t outputSize = 1;
                for (auto dim : outputDims_[0]) {
                    outputSize *= dim;
                }
                outputBuffer_.resize(outputSize);
                
                useIOBinding_ = true;
                
                // 阶段1：初始化GPU持久内存
                currentDevice_ = currentUseGPU;
                if (initializeGpuMemory()) {
                    obs_log(LOG_INFO, "[ModelYOLO] GPU persistent memory initialized successfully");
                    
                    // 阶段2：初始化CUDA纹理共享（仅CUDA模式）
                    if (currentDevice_ == "cuda" || currentDevice_ == "tensorrt") {
                        if (initializeCudaInterop()) {
                            obs_log(LOG_INFO, "[ModelYOLO] CUDA texture interop initialized");
                        } else {
                            obs_log(LOG_WARNING, "[ModelYOLO] CUDA interop init failed, texture sharing disabled");
                        }
                    }
                    
                    // 初始化DML预处理器（仅DML模式）
                    if (currentDevice_ == "dml") {
                        if (initializeDmlPreprocessor()) {
                            obs_log(LOG_INFO, "[ModelYOLO] DML preprocessor initialized");
                        } else {
                            obs_log(LOG_WARNING, "[ModelYOLO] DML preprocessor init failed, texture sharing disabled");
                        }
                    }
                } else {
                    obs_log(LOG_WARNING, "[ModelYOLO] GPU memory init failed, using CPU fallback");
                }
                
                obs_log(LOG_INFO, "[ModelYOLO] IOBinding enabled for GPU optimization");
            } catch (const std::exception& e) {
                obs_log(LOG_WARNING, "[ModelYOLO] Failed to initialize IOBinding: %s, using standard inference", e.what());
                useIOBinding_ = false;
            }
        }
        
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
    const int channelSize = inputWidth_ * inputHeight_;
    
    if (input.channels() == 4) {
        const unsigned char* inputData = input.data;
        float* rChannel = outputBuffer;
        float* gChannel = outputBuffer + channelSize;
        float* bChannel = outputBuffer + channelSize * 2;
        
        for (int i = 0; i < channelSize; ++i) {
            rChannel[i] = inputData[i * 4 + 2] / 255.0f;
            gChannel[i] = inputData[i * 4 + 1] / 255.0f;
            bChannel[i] = inputData[i * 4 + 0] / 255.0f;
        }
    } else if (input.channels() == 3) {
        const unsigned char* inputData = input.data;
        float* rChannel = outputBuffer;
        float* gChannel = outputBuffer + channelSize;
        float* bChannel = outputBuffer + channelSize * 2;
        
        for (int i = 0; i < channelSize; ++i) {
            rChannel[i] = inputData[i * 3 + 2] / 255.0f;
            gChannel[i] = inputData[i * 3 + 1] / 255.0f;
            bChannel[i] = inputData[i * 3 + 0] / 255.0f;
        }
    } else {
        cv::Mat rgb;
        cv::cvtColor(input, rgb, cv::COLOR_GRAY2RGB);
        
        cv::Mat floatMat;
        rgb.convertTo(floatMat, CV_32F, 1.0f / 255.0f);
        
        std::vector<cv::Mat> channels(3);
        cv::split(floatMat, channels);
        
        for (int c = 0; c < 3; ++c) {
            std::memcpy(outputBuffer + c * channelSize, channels[c].data, channelSize * sizeof(float));
        }
    }
}

std::vector<Detection> ModelYOLO::inference(const cv::Mat& input) {
    return doInference(input);
}

std::future<std::vector<Detection>> ModelYOLO::asyncInference(const cv::Mat& input) {
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

std::vector<Detection> ModelYOLO::doInference(const cv::Mat& input) {
    
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    InferenceLatency latency;
    
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
        auto preprocessStartTime = std::chrono::high_resolution_clock::now();
        
        cv::Mat letterboxed;
        LetterboxInfo letterboxInfo = letterbox(input, letterboxed);
        
        preprocessInput(letterboxed, inputBuffer_.data());
        
        auto preprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.preprocessMs = std::chrono::duration<double, std::milli>(preprocessEndTime - preprocessStartTime).count();
        
        std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
        
        Ort::Value inputTensor;
        try {
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, 
                OrtMemType::OrtMemTypeDefault
            );
            
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

        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        
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
        
        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();
        
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
        
        if (numBoxes <= 0 || numElements <= 0) {
            obs_log(LOG_ERROR, "[ModelYOLO] Invalid output parameters: numBoxes=%d, numElements=%d", numBoxes, numElements);
            return {};
        }
        
        cv::Size originalSize(input.cols, input.rows);
        
        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        
        std::vector<Detection> detections;
        
        try {
            switch (version_) {
                case Version::YOLOv5:
                    detections = postprocessYOLOv5(outputData, numBoxes, numClasses_, 
                                                  letterboxInfo, originalSize);
                    break;
                case Version::YOLOv8:
                    detections = postprocessYOLOv8(outputData, numBoxes, numClasses_, 
                                                  letterboxInfo, originalSize);
                    break;
                case Version::YOLOv11:
                    detections = postprocessYOLOv11(outputData, numBoxes, numClasses_, 
                                                   letterboxInfo, originalSize);
                    break;
            }
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Postprocessing exception: %s", e.what());
            return {};
        }
        
        auto postprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.postprocessMs = std::chrono::duration<double, std::milli>(postprocessEndTime - postprocessStartTime).count();
        
        // 计算总延迟
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        latency.totalMs = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();
        latency.isGpuPath = false;
        
        // 添加到统计器
        latencyStats_.addSample(latency);
        
        // 定期输出延迟日志
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLatencyLogTime_).count() >= LATENCY_LOG_INTERVAL_MS) {
            lastLatencyLogTime_ = now;
            obs_log(LOG_INFO, "[ModelYOLO] 延迟统计 (CPU路径):\n%s", latencyStats_.getSummary().c_str());
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
    const LetterboxInfo& letterboxInfo,
    const cv::Size& originalImageSize
) {
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    const int numElements = 5 + numClasses;

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

        bool isTargetClass = false;
        if (targetClassId_ >= 0) {
            isTargetClass = (maxClassId == targetClassId_);
        } else if (!targetClasses_.empty()) {
            isTargetClass = targetClasses_.count(maxClassId);
        } else {
            isTargetClass = true;
        }
        
        if (!isTargetClass) {
            continue;
        }

        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];

        float x1 = (cx - w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y1 = (cy - h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
        float x2 = (cx + w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y2 = (cy + h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;

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

    return detections;
}

std::vector<Detection> ModelYOLO::postprocessYOLOv8(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const LetterboxInfo& letterboxInfo,
    const cv::Size& originalImageSize
) {
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

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

        bool isTargetClass = false;
        if (targetClassId_ >= 0) {
            isTargetClass = (maxClassId == targetClassId_);
        } else if (!targetClasses_.empty()) {
            isTargetClass = targetClasses_.count(maxClassId);
        } else {
            isTargetClass = true;
        }
        
        if (!isTargetClass) {
            continue;
        }

        float x1 = (cx - w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y1 = (cy - h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;
        float x2 = (cx + w / 2.0f - letterboxInfo.padX) / letterboxInfo.scale;
        float y2 = (cy + h / 2.0f - letterboxInfo.padY) / letterboxInfo.scale;

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

    return detections;
}

std::vector<Detection> ModelYOLO::postprocessYOLOv11(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const LetterboxInfo& letterboxInfo,
    const cv::Size& originalImageSize
) {
    return postprocessYOLOv8(rawOutput, numBoxes, numClasses, letterboxInfo, originalImageSize);
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
    targetClasses_.clear();
    if (classId >= 0) {
        targetClasses_.insert(classId);
    }
}

void ModelYOLO::setTargetClasses(const std::vector<int>& classIds) {
    targetClasses_.clear();
    targetClasses_.insert(classIds.begin(), classIds.end());
    if (classIds.size() == 1) {
        targetClassId_ = classIds[0];
    } else if (classIds.empty()) {
        targetClassId_ = -1;
    } else {
        targetClassId_ = -1;
    }
}

void ModelYOLO::setInputResolution(int resolution) {
    // 禁用手动设置输入分辨率，始终使用模型实际输入尺寸
    obs_log(LOG_WARNING, "[ModelYOLO] setInputResolution is disabled. Input resolution is determined by model.");
    obs_log(LOG_WARNING, "[ModelYOLO] Current model input size: %dx%d", inputWidth_, inputHeight_);
    // 不执行任何修改
}

// ============================================================================
// 阶段1：GPU持久内存管理
// ============================================================================

bool ModelYOLO::initializeGpuMemory() {
    if (!session_ || !ioBinding_) {
        obs_log(LOG_ERROR, "[ModelYOLO] Cannot init GPU memory: session or IOBinding not ready");
        return false;
    }
    
    try {
        // 创建GPU内存信息
        if (currentDevice_ == "cuda" || currentDevice_ == "tensorrt") {
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
            gpuMemInfo_ = new Ort::MemoryInfo(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)
            );
            
            // 创建CUDA分配器
            gpuAllocator_ = new Ort::Allocator(*session_, *gpuMemInfo_);
            
            // 预分配GPU输入张量
            std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
            gpuInputTensor_ = Ort::Value::CreateTensor<float>(
                *gpuAllocator_, inputShape.data(), inputShape.size()
            );
            
            // 预分配GPU输出张量
            if (!outputDims_.empty()) {
                gpuOutputTensor_ = Ort::Value::CreateTensor<float>(
                    *gpuAllocator_, outputDims_[0].data(), outputDims_[0].size()
                );
            }
            
            useGpuMemory_ = true;
            obs_log(LOG_INFO, "[ModelYOLO] CUDA persistent memory allocated: input %dx%d", 
                    inputWidth_, inputHeight_);
            return true;
#else
            obs_log(LOG_WARNING, "[ModelYOLO] CUDA EP not available, using CPU memory");
            return false;
#endif
        } else if (currentDevice_ == "dml") {
#ifdef HAVE_ONNXRUNTIME_DML_EP
            // DirectML使用CPU内存作为暂存，但IOBinding仍然有效
            gpuMemInfo_ = new Ort::MemoryInfo(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
            );
            
            useGpuMemory_ = false;  // DML不使用真正的GPU内存分配
            obs_log(LOG_INFO, "[ModelYOLO] DirectML mode: using IOBinding with CPU memory");
            return true;
#else
            return false;
#endif
        }
        
        return false;
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Failed to initialize GPU memory: %s", e.what());
        useGpuMemory_ = false;
        return false;
    }
}

void ModelYOLO::releaseGpuMemory() {
    if (gpuAllocator_) {
        delete gpuAllocator_;
        gpuAllocator_ = nullptr;
    }
    
    if (gpuMemInfo_) {
        delete gpuMemInfo_;
        gpuMemInfo_ = nullptr;
    }
    
    // Ort::Value会自动释放
    gpuInputTensor_ = Ort::Value(nullptr);
    gpuOutputTensor_ = Ort::Value(nullptr);
    
    useGpuMemory_ = false;
    obs_log(LOG_INFO, "[ModelYOLO] GPU memory released");
}

// ============================================================================
// 阶段2：CUDA纹理共享
// ============================================================================

bool ModelYOLO::initializeCudaInterop() {
#ifdef HAVE_CUDA
    if (!useGpuMemory_) {
        obs_log(LOG_WARNING, "[ModelYOLO] Cannot init CUDA interop: GPU memory not initialized");
        return false;
    }
    
    try {
        // 创建CUDA流
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to create CUDA stream: %s", 
                    cudaGetErrorString(err));
            return false;
        }
        cudaStream_ = stream;
        
        cudaInteropInitialized_ = true;
        obs_log(LOG_INFO, "[ModelYOLO] CUDA interop initialized");
        return true;
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] CUDA interop init failed: %s", e.what());
        return false;
    }
#else
    obs_log(LOG_WARNING, "[ModelYOLO] CUDA not available for texture interop");
    return false;
#endif
}

void ModelYOLO::releaseCudaInterop() {
#ifdef HAVE_CUDA
    if (cudaResource_) {
        cudaGraphicsUnregisterResource(cudaResource_);
        cudaResource_ = nullptr;
    }
    
    if (cudaStream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(cudaStream_));
        cudaStream_ = nullptr;
    }
    
    if (cudaInputBuffer_) {
        cudaFree(cudaInputBuffer_);
        cudaInputBuffer_ = nullptr;
    }
    
    cudaInteropInitialized_ = false;
    obs_log(LOG_INFO, "[ModelYOLO] CUDA interop released");
#endif
}

bool ModelYOLO::initializeDmlPreprocessor() {
#ifdef HAVE_ONNXRUNTIME_DML_EP
    if (dmlInteropInitialized_) {
        return true;
    }
    
    ID3D11Device* d3d11Device = nullptr;
    ID3D11DeviceContext* d3d11Context = nullptr;
    
    try {
        // 创建 D3D11 设备
        D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
        D3D_FEATURE_LEVEL obtainedFeatureLevel;
        
        HRESULT hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            featureLevels,
            1,
            D3D11_SDK_VERSION,
            &d3d11Device,
            &obtainedFeatureLevel,
            &d3d11Context
        );
        
        if (FAILED(hr) || !d3d11Device || !d3d11Context) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to create D3D11 device for DML: 0x%08X", hr);
            if (d3d11Context) d3d11Context->Release();
            if (d3d11Device) d3d11Device->Release();
            return false;
        }
        
        // DmlPreprocessor 使用 ComPtr 管理设备生命周期
        // initialize() 内部会 AddRef，我们释放本地引用
        dmlPreprocessor_ = new DmlPreprocessor();
        if (!dmlPreprocessor_->initialize(d3d11Device)) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to initialize DML preprocessor");
            delete dmlPreprocessor_;
            dmlPreprocessor_ = nullptr;
            d3d11Context->Release();
            d3d11Device->Release();
            return false;
        }
        
        // DmlPreprocessor 已通过 ComPtr 持有设备引用
        // 释放本地引用（ComPtr 已 AddRef）
        d3d11Context->Release();
        d3d11Device->Release();
        d3d11Device = nullptr;
        d3d11Context = nullptr;
        
        dmlInteropInitialized_ = true;
        obs_log(LOG_INFO, "[ModelYOLO] DML preprocessor initialized successfully (D3D11 Feature Level: 0x%X)", obtainedFeatureLevel);
        return true;
        
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] DML preprocessor init exception: %s", e.what());
        // 异常情况下确保资源释放
        if (d3d11Context) d3d11Context->Release();
        if (d3d11Device) d3d11Device->Release();
        if (dmlPreprocessor_) {
            delete dmlPreprocessor_;
            dmlPreprocessor_ = nullptr;
        }
        return false;
    }
#else
    obs_log(LOG_WARNING, "[ModelYOLO] DML not supported in this build");
    return false;
#endif
}

void ModelYOLO::releaseDmlInterop() {
#ifdef HAVE_ONNXRUNTIME_DML_EP
    if (dmlPreprocessor_) {
        dmlPreprocessor_->release();
        delete dmlPreprocessor_;
        dmlPreprocessor_ = nullptr;
    }
    
    dmlInteropInitialized_ = false;
    obs_log(LOG_INFO, "[ModelYOLO] DML interop released");
#endif
}

std::vector<Detection> ModelYOLO::inferenceFromTexture(void* d3d11Texture, int width, int height, 
                                                         int originalWidth, int originalHeight,
                                                         InferenceLatency* outLatency) {
#ifdef HAVE_CUDA
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    InferenceLatency latency;
    latency.isGpuPath = true;
    
    if (!cudaInteropInitialized_ || !session_) {
        return {};
    }
    
    std::vector<Detection> detections;
    
    try {
        auto cudaStartTime = std::chrono::high_resolution_clock::now();
        
        ID3D11Texture2D* d3dTex = static_cast<ID3D11Texture2D*>(d3d11Texture);
        if (!d3dTex) return {};
        
        cudaStream_t stream = static_cast<cudaStream_t>(cudaStream_);
        
        if (!cudaResource_) {
            cudaError_t err = cudaGraphicsD3D11RegisterResource(
                &cudaResource_, d3dTex, cudaGraphicsRegisterFlagsNone);
            if (err != cudaSuccess) {
                obs_log(LOG_ERROR, "[ModelYOLO] Failed to register D3D11 texture: %s",
                        cudaGetErrorString(err));
                return {};
            }
        }
        
        cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_, stream);
        if (err != cudaSuccess) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to map texture: %s", 
                    cudaGetErrorString(err));
            return {};
        }
        
        cudaArray_t cudaArray;
        err = cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource_, 0, 0);
        if (err != cudaSuccess) {
            cudaGraphicsUnmapResources(1, &cudaResource_, stream);
            return {};
        }
        
        size_t requiredSize = 3 * inputHeight_ * inputWidth_ * sizeof(float);
        if (!cudaInputBuffer_) {
            err = cudaMalloc(&cudaInputBuffer_, requiredSize);
            if (err != cudaSuccess) {
                obs_log(LOG_ERROR, "[ModelYOLO] Failed to allocate CUDA input buffer: %s",
                        cudaGetErrorString(err));
                cudaGraphicsUnmapResources(1, &cudaResource_, stream);
                return {};
            }
        }
        
        auto kernelStartTime = std::chrono::high_resolution_clock::now();
        
        bool preprocessSuccess = cudaLetterboxAndPreprocess(
            cudaArray,
            static_cast<float*>(cudaInputBuffer_),
            inputWidth_,
            inputHeight_,
            stream
        );
        
        auto kernelEndTime = std::chrono::high_resolution_clock::now();
        latency.cudaKernelMs = std::chrono::duration<double, std::milli>(kernelEndTime - kernelStartTime).count();
        
        cudaGraphicsUnmapResources(1, &cudaResource_, stream);
        
        if (!preprocessSuccess) {
            obs_log(LOG_ERROR, "[ModelYOLO] CUDA letterbox preprocessing failed");
            return {};
        }
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            obs_log(LOG_ERROR, "[ModelYOLO] CUDA stream sync failed: %s", 
                    cudaGetErrorString(err));
            return {};
        }
        
        auto copyStartTime = std::chrono::high_resolution_clock::now();
        cudaMemcpy(inputBuffer_.data(), cudaInputBuffer_, requiredSize, cudaMemcpyDeviceToHost);
        auto copyEndTime = std::chrono::high_resolution_clock::now();
        latency.gpuCopyMs = std::chrono::duration<double, std::milli>(copyEndTime - copyStartTime).count();
        
        latency.preprocessMs = std::chrono::duration<double, std::milli>(copyEndTime - cudaStartTime).count();
        
        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        
        std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputBuffer_.data(),
            inputBufferSize_,
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
        
        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputTensors = session_->Run(
            runOptions,
            inputNamesChar.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNamesChar.data(),
            outputNamesChar.size()
        );
        
        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();
        
        if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
            return {};
        }
        
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        if (!outputData) return {};
        
        std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() < 3) return {};
        
        int numBoxes = 0;
        if (version_ == Version::YOLOv5) {
            numBoxes = static_cast<int>(outputShape[1]);
        } else {
            numBoxes = static_cast<int>(outputShape[2]);
        }
        
        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        
        LetterboxInfo letterboxInfo = calculateLetterboxParams(width, height, inputWidth_, inputHeight_);
        cv::Size originalSize(originalWidth, originalHeight);
        
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
        
        latencyStats_.addSample(latency);
        
        if (outLatency) {
            *outLatency = latency;
        }
        
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLatencyLogTime_).count() >= LATENCY_LOG_INTERVAL_MS) {
            lastLatencyLogTime_ = now;
            obs_log(LOG_INFO, "[ModelYOLO] 延迟统计 (GPU路径):\n%s", latencyStats_.getSummary().c_str());
        }
        
        return detections;
        
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Texture inference failed: %s", e.what());
        return {};
    }
#else
    (void)d3d11Texture;
    (void)width;
    (void)height;
    (void)originalWidth;
    (void)originalHeight;
    (void)outLatency;
    return {};
#endif
}

std::vector<Detection> ModelYOLO::inferenceFromTextureDml(void* d3d11Texture, int width, int height,
                                                          int originalWidth, int originalHeight,
                                                          InferenceLatency* outLatency) {
#ifdef HAVE_ONNXRUNTIME_DML_EP
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    InferenceLatency latency;
    latency.isGpuPath = true;
    
    if (!session_) {
        obs_log(LOG_ERROR, "[ModelYOLO] DML inference: session not initialized");
        return {};
    }
    
    ID3D11Texture2D* d3dTex = static_cast<ID3D11Texture2D*>(d3d11Texture);
    if (!d3dTex) {
        obs_log(LOG_ERROR, "[ModelYOLO] DML inference: invalid texture");
        return {};
    }
    
    std::vector<Detection> detections;
    
    try {
        auto preprocessStartTime = std::chrono::high_resolution_clock::now();
        
        // 确保输入缓冲区大小正确
        size_t requiredSize = 3 * inputHeight_ * inputWidth_;
        if (inputBuffer_.size() < requiredSize) {
            inputBuffer_.resize(requiredSize);
            inputBufferSize_ = requiredSize * sizeof(float);
        }
        
        // 使用 DML 预处理器从纹理预处理
        bool preprocessSuccess = false;
        if (dmlPreprocessor_ && dmlPreprocessor_->isInitialized()) {
            DmlPreprocessParams params;
            preprocessSuccess = dmlPreprocessor_->preprocessFromTexture(
                d3dTex,
                inputBuffer_.data(),
                inputWidth_,
                inputHeight_,
                &params
            );
            
            if (preprocessSuccess) {
                // 计算 letterbox 参数用于后处理
                // params 包含 scale, padX, padY
            }
        }
        
        auto preprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.preprocessMs = std::chrono::duration<double, std::milli>(preprocessEndTime - preprocessStartTime).count();
        
        if (!preprocessSuccess) {
            obs_log(LOG_WARNING, "[ModelYOLO] DML preprocessing failed, falling back to CPU");
            return {};
        }
        
        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        
        // 创建输入张量
        std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputBuffer_.data(),
            inputBufferSize_,
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
        
        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputTensors = session_->Run(
            runOptions,
            inputNamesChar.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNamesChar.data(),
            outputNamesChar.size()
        );
        
        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();
        
        if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
            obs_log(LOG_ERROR, "[ModelYOLO] DML inference: invalid output tensor");
            return {};
        }
        
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        if (!outputData) {
            obs_log(LOG_ERROR, "[ModelYOLO] DML inference: null output data");
            return {};
        }
        
        std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() < 3) {
            obs_log(LOG_ERROR, "[ModelYOLO] DML inference: invalid output shape");
            return {};
        }
        
        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        
        // 计算 letterbox 参数
        LetterboxInfo letterboxInfo = calculateLetterboxParams(width, height, inputWidth_, inputHeight_);
        
        int numBoxes = 0;
        if (version_ == Version::YOLOv5) {
            numBoxes = static_cast<int>(outputShape[1]);
            detections = postprocessYOLOv5(
                outputData,
                numBoxes,
                numClasses_,
                letterboxInfo,
                cv::Size(originalWidth, originalHeight)
            );
        } else if (version_ == Version::YOLOv8) {
            numBoxes = static_cast<int>(outputShape[2]);
            detections = postprocessYOLOv8(
                outputData,
                numBoxes,
                numClasses_,
                letterboxInfo,
                cv::Size(originalWidth, originalHeight)
            );
        } else if (version_ == Version::YOLOv11) {
            numBoxes = static_cast<int>(outputShape[2]);
            detections = postprocessYOLOv11(
                outputData,
                numBoxes,
                numClasses_,
                letterboxInfo,
                cv::Size(originalWidth, originalHeight)
            );
        }
        
        auto postprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.postprocessMs = std::chrono::duration<double, std::milli>(postprocessEndTime - postprocessStartTime).count();
        
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        latency.totalMs = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();
        
        // 记录延迟统计
        latencyStats_.addSample(latency);
        
        if (outLatency) {
            *outLatency = latency;
        }
        
        obs_log(LOG_INFO, "[ModelYOLO] DML texture inference: %zu detections, total %.2fms (preprocess %.2fms, inference %.2fms, postprocess %.2fms)",
                detections.size(), latency.totalMs, latency.preprocessMs, latency.inferenceMs, latency.postprocessMs);
        
        return detections;
        
    } catch (const Ort::Exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] DML ONNX Runtime error: %s", e.what());
        return {};
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] DML inference error: %s", e.what());
        return {};
    }
#else
    (void)d3d11Texture;
    (void)width;
    (void)height;
    (void)originalWidth;
    (void)originalHeight;
    (void)outLatency;
    obs_log(LOG_WARNING, "[ModelYOLO] DML not supported in this build");
    return {};
#endif
}
