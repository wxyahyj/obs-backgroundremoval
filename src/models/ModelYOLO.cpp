#include "ModelYOLO.h"
#include <plugin-support.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

// SIMD头文件
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef HAVE_ONNXRUNTIME_DML_EP
#include <d3d11.h>
#include <dml_provider_factory.h>
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

#include <cstring>
#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

static bool cpuSupportsF16C()
{
	static int cached = -1;
	if (cached >= 0) return cached != 0;
	int info[4] = {};
#if defined(_MSC_VER)
	__cpuidex(info, 1, 0);
#else
	cached = 0; return false;
#endif
	// ECX bit 29 = F16C
	cached = ((info[2] & (1 << 29)) != 0) ? 1 : 0;
	return cached != 0;
}

static void convertFloatBufferToHalf(const float* src, Ort::Float16_t* dst, size_t n)
{
	size_t i = 0;
#if defined(_MSC_VER)
	if (cpuSupportsF16C()) {
		for (; i + 8 <= n; i += 8) {
			__m256 v = _mm256_loadu_ps(src + i);
			__m128i h = _mm256_cvtps_ph(v, 0);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
		}
		for (; i + 4 <= n; i += 4) {
			__m128 v = _mm_loadu_ps(src + i);
			__m128i h = _mm_cvtps_ph(v, 0);
			_mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), h);
		}
	}
#endif
	for (; i < n; ++i) {
		dst[i] = Ort::Float16_t(src[i]);
	}
}

static void convertHalfBufferToFloat(const Ort::Float16_t* src, float* dst, size_t n)
{
	size_t i = 0;
#if defined(_MSC_VER)
	if (cpuSupportsF16C()) {
		for (; i + 8 <= n; i += 8) {
			__m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
			__m256 f = _mm256_cvtph_ps(h);
			_mm256_storeu_ps(dst + i, f);
		}
		for (; i + 4 <= n; i += 4) {
			__m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i));
			__m128 f = _mm_cvtph_ps(h);
			_mm_storeu_ps(dst + i, f);
		}
	}
#endif
	for (; i < n; ++i) {
		dst[i] = static_cast<float>(src[i]);
	}
}


ModelYOLO::LetterboxInfo ModelYOLO::calculateLetterboxParams(int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    // Align with AiMod: scale + round unpad + round pad (training/export consistent)
    LetterboxInfo info;
    
    float scaleX = static_cast<float>(dstWidth) / srcWidth;
    float scaleY = static_cast<float>(dstHeight) / srcHeight;
    info.scale = std::min(scaleX, scaleY);
    
    int newWidth = static_cast<int>(std::round(srcWidth * info.scale));
    int newHeight = static_cast<int>(std::round(srcHeight * info.scale));
    newWidth = std::max(1, std::min(newWidth, dstWidth));
    newHeight = std::max(1, std::min(newHeight, dstHeight));
    
    float dw = (dstWidth - newWidth) * 0.5f;
    float dh = (dstHeight - newHeight) * 0.5f;
    info.padX = static_cast<int>(std::round(dw - 0.1f));
    info.padY = static_cast<int>(std::round(dh - 0.1f));
    
    return info;
}

ModelYOLO::LetterboxInfo ModelYOLO::letterbox(const cv::Mat& input, cv::Mat& output) {
    // Same AiMod letterbox math as calculateLetterboxParams (must match postprocess inverse)
    LetterboxInfo info = calculateLetterboxParams(input.cols, input.rows, inputWidth_, inputHeight_);
    
    int newWidth = static_cast<int>(std::round(input.cols * info.scale));
    int newHeight = static_cast<int>(std::round(input.rows * info.scale));
    newWidth = std::max(1, std::min(newWidth, inputWidth_));
    newHeight = std::max(1, std::min(newHeight, inputHeight_));
    
    if (resizedBuffer_.rows != newHeight || resizedBuffer_.cols != newWidth || resizedBuffer_.type() != input.type()) {
        resizedBuffer_.create(newHeight, newWidth, input.type());
    }
    cv::resize(input, resizedBuffer_, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_NEAREST);

    const cv::Scalar padColor(114, 114, 114, 114);
    if (letterboxBuffer_.rows != inputHeight_ || letterboxBuffer_.cols != inputWidth_ || letterboxBuffer_.type() != input.type()) {
        letterboxBuffer_.create(inputHeight_, inputWidth_, input.type());
        letterboxBuffer_.setTo(padColor);
        letterboxLastNewW_ = letterboxLastNewH_ = letterboxLastPadX_ = letterboxLastPadY_ = -1;
    } else if (letterboxLastNewW_ != newWidth || letterboxLastNewH_ != newHeight ||
               letterboxLastPadX_ != info.padX || letterboxLastPadY_ != info.padY) {
        letterboxBuffer_.setTo(padColor);
    }
    resizedBuffer_.copyTo(letterboxBuffer_(cv::Rect(info.padX, info.padY, newWidth, newHeight)));
    letterboxLastNewW_ = newWidth;
    letterboxLastNewH_ = newHeight;
    letterboxLastPadX_ = info.padX;
    letterboxLastPadY_ = info.padY;
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
      dmlInteropInitialized_(false),
      inferenceThreadRunning_(false)
{
    // Minimal ctor: no Ort::Env, no Ort::Value members, no worker thread.
    // Ort::Env + worker start in loadModel().
    lastLatencyLogTime_ = std::chrono::steady_clock::now();
    obs_log(LOG_INFO, "[ModelYOLO] Constructed (Version: %d)", static_cast<int>(version));
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

    // Lazy Ort::Env
    if (!env_) {
        try {
            env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "YOLOModel");
            obs_log(LOG_INFO, "[ModelYOLO] Ort::Env created");
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to create Ort::Env: %s", e.what());
            return;
        } catch (...) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to create Ort::Env (unknown)");
            return;
        }
    }

    // Lazy async inference worker
    if (!inferenceThreadRunning_) {
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
                        task->promise.set_value(doInference(task->input));
                    } catch (const std::exception& e) {
                        obs_log(LOG_ERROR, "[ModelYOLO] Async inference error: %s", e.what());
                        task->promise.set_value({});
                    } catch (...) {
                        task->promise.set_value({});
                    }
                }
            }
        });
    }
    
    try {
        Ort::SessionOptions sessionOptions;
        // AiMod: DML uses BASIC; ALL can hurt DML stability. Other EPs keep ALL.
        if (currentUseGPU == "dml") {
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        } else {
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }
        
        obs_log(LOG_INFO, "[ModelYOLO] Using device: %s", currentUseGPU.c_str());
        
        if (currentUseGPU != "cpu") {
            // AiMod pattern: DisableMemPattern + SEQUENTIAL for DML/GPU EPs
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
		                // 通过 Ort::GetApi 运行时 API 表调用, 不依赖 .lib 导出
		                OrtCUDAProviderOptions cuda_options{};
		                cuda_options.device_id = 0;
		                // MUST check OrtStatus — missing cudnn DLL often returns FAIL without C++ throw
		                OrtStatus *st = Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(
		                    static_cast<OrtSessionOptions *>(sessionOptions), &cuda_options);
		                if (st != nullptr) {
		                    const char *msg = Ort::GetApi().GetErrorMessage(st);
		                    obs_log(LOG_WARNING,
		                            "[ModelYOLO] Failed to enable CUDA EP: %s, falling back to CPU",
		                            msg ? msg : "(null)");
		                    Ort::GetApi().ReleaseStatus(st);
		                    gpuFailed = true;
		                    currentUseGPU = "cpu";
		                } else {
		                    obs_log(LOG_INFO, "[ModelYOLO] CUDA execution provider enabled successfully");
		                }
		            } catch (const std::exception &e) {
		                obs_log(LOG_WARNING, "[ModelYOLO] Failed to enable CUDA: %s, falling back to CPU", e.what());
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
            // AiMod-style: C API status check (does not rely on C++ exception alone)
            try {
                OrtStatus* st = OrtSessionOptionsAppendExecutionProvider_DML(
                    static_cast<OrtSessionOptions*>(sessionOptions), 0);
                if (st != nullptr) {
                    const char* msg = Ort::GetApi().GetErrorMessage(st);
                    obs_log(LOG_WARNING,
                            "[ModelYOLO] Failed to enable DirectML: %s, falling back to CPU",
                            msg ? msg : "(null)");
                    Ort::GetApi().ReleaseStatus(st);
                    gpuFailed = true;
                    currentUseGPU = "cpu";
                } else {
                    obs_log(LOG_INFO, "[ModelYOLO] DirectML execution provider enabled (AiMod-style)");
                }
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
        inputNamesChar_.clear();
        outputNamesChar_.clear();
        inputNamesChar_.reserve(inputNames_.size());
        outputNamesChar_.reserve(outputNames_.size());
        for (const auto& name : inputNames_)
            inputNamesChar_.push_back(name.get());
        for (const auto& name : outputNames_)
            outputNamesChar_.push_back(name.get());
        if (inputNamesChar_.empty() || outputNamesChar_.empty()) {
            obs_log(LOG_ERROR, "[ModelYOLO] Empty IO names after session create");
            throw std::runtime_error("Empty IO names");
        }
        
        // 检测模型是否为FP16
        isFp16Model_ = false;
        try {
            Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
            auto tensorTypeInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType inputType = tensorTypeInfo.GetElementType();
            
            if (inputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                isFp16Model_ = true;
                obs_log(LOG_INFO, "[ModelYOLO] Detected FP16 model - will use FP16 inference");
            } else {
                obs_log(LOG_INFO, "[ModelYOLO] Detected FP32 model - will use FP32 inference");
            }
        } catch (const std::exception& e) {
            obs_log(LOG_WARNING, "[ModelYOLO] Could not detect input type: %s, assuming FP32", e.what());
        }
        
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
            obs_log(LOG_INFO, "[ModelYOLO] Num graph outputs: %zu", outputDims_.size());
            for (size_t oi = 0; oi < outputDims_.size(); ++oi) {
                const auto &shape = outputDims_[oi];
                obs_log(LOG_INFO, "[ModelYOLO] Output[%zu] rank=%zu", oi, shape.size());
                for (size_t i = 0; i < shape.size(); ++i) {
                    obs_log(LOG_INFO, "[ModelYOLO]   shape[%zu]=%lld", i, (long long)shape[i]);
                }
            }
            obs_log(LOG_INFO, "[ModelYOLO] Model version (user): %d", static_cast<int>(version_));
            // Layout detection may override version_ (e.g. [1,6300,9]+heads → YOLOv5)
            detectOutputLayoutFromShape(outputDims_[0]);
            obs_log(LOG_INFO, "[ModelYOLO] Model version (effective): %d", static_cast<int>(version_));
        }
        
        // 预分配输入缓冲区 + 热路径缓存
        inputBufferSize_ = 1 * 3 * inputHeight_ * inputWidth_;
        inputShapeCache_ = {1, 3, static_cast<int64_t>(inputHeight_), static_cast<int64_t>(inputWidth_)};
        if (!cpuMemInfo_) {
            cpuMemInfo_ = std::make_unique<Ort::MemoryInfo>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        }
        inputBuffer_.resize(inputBufferSize_);
        if (isFp16Model_) {
            inputBufferFp16_.resize(inputBufferSize_);
        }
        inputBuffer_.resize(inputBufferSize_);
        obs_log(LOG_INFO, "[ModelYOLO] Allocated input buffer size: %zu", inputBufferSize_);
        
        currentDevice_ = currentUseGPU;

        // Pre-allocate host output buffer for postprocess
        if (!outputDims_.empty()) {
            size_t outputSize = 1;
            for (auto dim : outputDims_[0]) {
                if (dim > 0) outputSize *= static_cast<size_t>(dim);
            }
            outputBuffer_.resize(outputSize);
        }

        // ---- AiMod-style DML path: NO IoBinding, host float blob + session.Run ----
        if (currentUseGPU == "dml") {
            useIOBinding_ = false;
            ioBinding_.reset();
            // Float preprocess (render-thread letterbox) still useful
            if (initializeDmlPreprocessor()) {
                obs_log(LOG_INFO, "[ModelYOLO] AiMod-style DML path: host CreateTensor+Run, float preprocess ON");
            } else {
                obs_log(LOG_WARNING, "[ModelYOLO] DML float preprocess init failed (Mat path still works)");
            }
        } else if (currentUseGPU != "cpu") {
            // Non-DML GPU EPs: keep IoBinding optional for later
            try {
                ioBinding_ = std::make_unique<Ort::IoBinding>(*session_);
                useIOBinding_ = true;
                if (initializeGpuMemory()) {
                    obs_log(LOG_INFO, "[ModelYOLO] GPU persistent memory initialized");
                    if (currentDevice_ == "cuda" || currentDevice_ == "tensorrt") {
                        if (initializeCudaInterop()) {
                            obs_log(LOG_INFO, "[ModelYOLO] CUDA device path ready");
                        }
                    }
                    if (initializeDmlPreprocessor()) {
                        obs_log(LOG_INFO, "[ModelYOLO] Preprocessed float path enabled for %s",
                                currentDevice_.c_str());
                    }
                } else {
                    obs_log(LOG_WARNING, "[ModelYOLO] GPU memory init failed (EP may still run)");
                    if (currentUseGPU == "cuda" || currentUseGPU == "tensorrt") {
                        currentDevice_ = currentUseGPU + "+cpu_pre";
                    }
                }
                obs_log(LOG_INFO, "[ModelYOLO] IOBinding enabled for GPU optimization");
            } catch (const std::exception& e) {
                obs_log(LOG_WARNING, "[ModelYOLO] IOBinding init failed: %s", e.what());
                useIOBinding_ = false;
            }
        } else {
            currentDevice_ = "cpu";
            useIOBinding_ = false;
        }
        
        name = "YOLO";
        // Prefer composite currentDevice_ when set (e.g. cuda+cpu_pre); else currentUseGPU
        if (currentDevice_.empty())
            currentDevice_ = currentUseGPU;
        
        obs_log(LOG_INFO, "[ModelYOLO] Model loaded successfully");
        obs_log(LOG_INFO, "  Input size: %dx%d", inputWidth_, inputHeight_);
        obs_log(LOG_INFO, "  Num classes: %d", numClasses_);
        obs_log(LOG_INFO, "  Device requested/effective EP: %s  runtime=%s", currentUseGPU.c_str(),
               currentDevice_.c_str());
        
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Failed to load model: %s", e.what());
        throw;
    }
}

void ModelYOLO::preprocessInput(const cv::Mat& input, float* outputBuffer) {
    const int channelSize = inputWidth_ * inputHeight_;
    const float norm = 1.0f / 255.0f;  // 预计算归一化因子
    
    if (input.channels() == 4) {
        const unsigned char* inputData = input.data;
        float* rChannel = outputBuffer;
        float* gChannel = outputBuffer + channelSize;
        float* bChannel = outputBuffer + channelSize * 2;
        
        // SSE: 一次 4 像素 BGRA→planar RGB float（修复原 AVX 写 8 float 越界）
        int i = 0;
#if defined(_MSC_VER) || defined(__SSE2__)
        const __m128 normVec = _mm_set1_ps(norm);
        for (; i + 3 < channelSize; i += 4) {
            __m128i bgra = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inputData + i * 4));
            __m128i b = _mm_and_si128(bgra, _mm_set1_epi32(0xFF));
            __m128i g = _mm_and_si128(_mm_srli_epi32(bgra, 8), _mm_set1_epi32(0xFF));
            __m128i r = _mm_and_si128(_mm_srli_epi32(bgra, 16), _mm_set1_epi32(0xFF));
            __m128 rf = _mm_mul_ps(_mm_cvtepi32_ps(r), normVec);
            __m128 gf = _mm_mul_ps(_mm_cvtepi32_ps(g), normVec);
            __m128 bf = _mm_mul_ps(_mm_cvtepi32_ps(b), normVec);
            _mm_storeu_ps(rChannel + i, rf);
            _mm_storeu_ps(gChannel + i, gf);
            _mm_storeu_ps(bChannel + i, bf);
        }
#endif
        // 处理剩余像素
        // 处理剩余像素
        for (; i < channelSize; ++i) {
            rChannel[i] = inputData[i * 4 + 2] * norm;
            gChannel[i] = inputData[i * 4 + 1] * norm;
            bChannel[i] = inputData[i * 4 + 0] * norm;
        }
    } else if (input.channels() == 3) {
        const unsigned char* inputData = input.data;
        float* rChannel = outputBuffer;
        float* gChannel = outputBuffer + channelSize;
        float* bChannel = outputBuffer + channelSize * 2;
        
        // BGR格式处理
        int i = 0;
        for (; i < channelSize; ++i) {
            rChannel[i] = inputData[i * 3 + 2] * norm;
            gChannel[i] = inputData[i * 3 + 1] * norm;
            bChannel[i] = inputData[i * 3 + 0] * norm;
        }
    } else {
        cv::Mat rgb;
        cv::cvtColor(input, rgb, cv::COLOR_GRAY2RGB);
        
        cv::Mat floatMat;
        rgb.convertTo(floatMat, CV_32F, norm);
        
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

void ModelYOLO::ensureCpuMemInfo()
{
    if (!cpuMemInfo_) {
        cpuMemInfo_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
}

void ModelYOLO::ensureCpuInputTensor()
{
    ensureCpuMemInfo();
    if (inputShapeCache_.size() != 4) {
        inputShapeCache_ = {1, 3, static_cast<int64_t>(inputHeight_), static_cast<int64_t>(inputWidth_)};
    }
    const size_t elems = inputBufferSize_;
    const bool wantFp16 = isFp16Model_;
    if (wantFp16) {
        if (inputBufferFp16_.size() < elems) inputBufferFp16_.resize(elems);
    } else {
        if (inputBuffer_.size() < elems) inputBuffer_.resize(elems);
    }
    // 缓冲地址/类型/元素数不变则复用 Ort::Value，避免每帧 CreateTensor
    if (cpuInputTensor_ && cpuInputTensorElems_ == elems && cpuInputTensorFp16_ == wantFp16) {
        return;
    }
    if (wantFp16) {
        cpuInputTensor_ = std::make_unique<Ort::Value>(
            Ort::Value::CreateTensor<Ort::Float16_t>(
                *cpuMemInfo_, inputBufferFp16_.data(), elems,
                inputShapeCache_.data(), inputShapeCache_.size()));
    } else {
        cpuInputTensor_ = std::make_unique<Ort::Value>(
            Ort::Value::CreateTensor<float>(
                *cpuMemInfo_, inputBuffer_.data(), elems,
                inputShapeCache_.data(), inputShapeCache_.size()));
    }
    cpuInputTensorElems_ = elems;
    cpuInputTensorFp16_ = wantFp16;
}

void ModelYOLO::ensureCpuOutputTensor()
{
    ensureCpuMemInfo();
    if (outputElementCount_ == 0) {
        if (!outputDims_.empty()) {
            size_t n = 1;
            outputShapeCache_ = outputDims_[0];
            for (auto d : outputShapeCache_) {
                if (d < 0) d = 1;
                n *= static_cast<size_t>(d);
            }
            outputElementCount_ = n;
        } else {
            outputElementCount_ = 1;
            outputShapeCache_ = {1};
        }
    }
    // 探测输出类型（首次）
    if (!session_) return;
    try {
        auto ti = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        isFp16Output_ = (ti.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        auto sh = ti.GetShape();
        if (!sh.empty()) {
            size_t n = 1;
            for (auto d : sh) { if (d < 0) d = 1; n *= static_cast<size_t>(d); }
            if (n > 0) {
                outputElementCount_ = n;
                outputShapeCache_.assign(sh.begin(), sh.end());
                for (auto& d : outputShapeCache_) if (d < 0) d = 1;
            }
        }
    } catch (...) {}

    const size_t elems = outputElementCount_;
    if (isFp16Output_) {
        if (outputBufferFp16_.size() < elems) outputBufferFp16_.resize(elems);
    } else {
        if (outputBuffer_.size() < elems) outputBuffer_.resize(elems);
    }
    if (cpuOutputTensor_ && cpuOutputTensorElems_ == elems && cpuOutputTensorFp16_ == isFp16Output_) {
        return;
    }
    if (isFp16Output_) {
        cpuOutputTensor_ = std::make_unique<Ort::Value>(
            Ort::Value::CreateTensor<Ort::Float16_t>(
                *cpuMemInfo_, outputBufferFp16_.data(), elems,
                outputShapeCache_.data(), outputShapeCache_.size()));
    } else {
        cpuOutputTensor_ = std::make_unique<Ort::Value>(
            Ort::Value::CreateTensor<float>(
                *cpuMemInfo_, outputBuffer_.data(), elems,
                outputShapeCache_.data(), outputShapeCache_.size()));
    }
    cpuOutputTensorElems_ = elems;
    cpuOutputTensorFp16_ = isFp16Output_;
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

    if (inputNamesChar_.empty() || outputNamesChar_.empty()) {
        inputNamesChar_.clear();
        outputNamesChar_.clear();
        for (const auto& name : inputNames_) inputNamesChar_.push_back(name.get());
        for (const auto& name : outputNames_) outputNamesChar_.push_back(name.get());
        if (inputNamesChar_.empty() || outputNamesChar_.empty()) {
            obs_log(LOG_ERROR, "[ModelYOLO] doInference: empty IO names");
            return {};
        }
    }
    
    try {
        auto preprocessStartTime = std::chrono::high_resolution_clock::now();

        // AiMod path: letterbox (round pad) → CHW float host blob → CreateTensor → Run
        cv::Mat letterboxed;
        LetterboxInfo letterboxInfo = letterbox(input, letterboxed);
        if (inputBuffer_.size() < inputBufferSize_)
            inputBuffer_.resize(inputBufferSize_);
        preprocessInput(letterboxed, inputBuffer_.data());
        if (isFp16Model_) {
            if (inputBufferFp16_.size() < inputBufferSize_)
                inputBufferFp16_.resize(inputBufferSize_);
            convertFloatBufferToHalf(inputBuffer_.data(), inputBufferFp16_.data(), inputBufferSize_);
        }

        // AiMod-style for DML (and when IoBinding off): skip persistent Ort::Value
        const bool aimodSimpleRun = (currentDevice_.find("dml") != std::string::npos) || !useIOBinding_;
        if (!aimodSimpleRun) {
            ensureCpuInputTensor();
            ensureCpuOutputTensor();
        }

        auto preprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.preprocessMs =
            std::chrono::duration<double, std::milli>(preprocessEndTime - preprocessStartTime)
                .count();

        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputTensors;
        bool usedBoundOutput = false;
        try {
            // AiMod-style (DML): host CreateTensor + session.Run, no IoBinding
            if (aimodSimpleRun || !(useIOBinding_ && ioBinding_ && cpuInputTensor_ && cpuOutputTensor_)) {
                ensureCpuMemInfo();
                if (inputShapeCache_.size() != 4) {
                    inputShapeCache_ = {1, 3, static_cast<int64_t>(inputHeight_),
                                        static_cast<int64_t>(inputWidth_)};
                }
                Ort::Value inT{nullptr};
                if (isFp16Model_) {
                    inT = Ort::Value::CreateTensor<Ort::Float16_t>(
                        *cpuMemInfo_, inputBufferFp16_.data(), inputBufferSize_,
                        inputShapeCache_.data(), inputShapeCache_.size());
                } else {
                    inT = Ort::Value::CreateTensor<float>(
                        *cpuMemInfo_, inputBuffer_.data(), inputBufferSize_,
                        inputShapeCache_.data(), inputShapeCache_.size());
                }
                std::vector<Ort::Value> inputTensors;
                inputTensors.push_back(std::move(inT));
                outputTensors = session_->Run(
                    runOptions, inputNamesChar_.data(), inputTensors.data(), inputTensors.size(),
                    outputNamesChar_.data(), outputNamesChar_.size());
            } else {
                ioBinding_->ClearBoundInputs();
                ioBinding_->ClearBoundOutputs();
                ioBinding_->BindInput(inputNamesChar_[0], *cpuInputTensor_);
                ioBinding_->BindOutput(outputNamesChar_[0], *cpuOutputTensor_);
                session_->Run(runOptions, *ioBinding_);
                usedBoundOutput = true;
            }
        } catch (const Ort::Exception& e) {
            cpuInputTensorElems_ = 0;
            cpuOutputTensorElems_ = 0;
            obs_log(LOG_ERROR, "[ModelYOLO] ONNX Runtime exception during Run: %s", e.what());
            return {};
        } catch (const std::exception& e) {
            cpuInputTensorElems_ = 0;
            cpuOutputTensorElems_ = 0;
            obs_log(LOG_ERROR, "[ModelYOLO] Exception during Run: %s", e.what());
            return {};
        } catch (...) {
            cpuInputTensorElems_ = 0;
            cpuOutputTensorElems_ = 0;
            obs_log(LOG_ERROR, "[ModelYOLO] Unknown exception during Run");
            return {};
        }
        
        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();
        
        float* outputData = nullptr;
        std::vector<int64_t> outputShape;
        try {
            if (usedBoundOutput) {
                if (isFp16Output_) {
                    if (outputFp32Scratch_.size() < outputElementCount_)
                        outputFp32Scratch_.resize(outputElementCount_);
                    convertHalfBufferToFloat(outputBufferFp16_.data(), outputFp32Scratch_.data(), outputElementCount_);
                    outputData = outputFp32Scratch_.data();
                } else {
                    outputData = outputBuffer_.data();
                }
                outputShape = outputShapeCache_;
            } else {
                if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
                    obs_log(LOG_ERROR, "[ModelYOLO] No output tensors from ONNX Runtime");
                    return {};
                }
                auto outputTypeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
                ONNXTensorElementDataType outputType = outputTypeInfo.GetElementType();
                outputShape = outputTypeInfo.GetShape();
                if (outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                    const Ort::Float16_t* fp16Data = outputTensors[0].GetTensorData<Ort::Float16_t>();
                    size_t outputSize = 1;
                    for (auto dim : outputShape) outputSize *= static_cast<size_t>(dim > 0 ? dim : 1);
                    if (outputFp32Scratch_.size() < outputSize)
                        outputFp32Scratch_.resize(outputSize);
                    convertHalfBufferToFloat(fp16Data, outputFp32Scratch_.data(), outputSize);
                    outputData = outputFp32Scratch_.data();
                } else {
                    outputData = outputTensors[0].GetTensorMutableData<float>();
                }
            }
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to get output tensor data: %s", e.what());
            return {};
        }

        if (!outputData) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to get output tensor data");
            return {};
        }
        
        if (outputShape.size() < 3) {
            obs_log(LOG_ERROR, "[ModelYOLO] Invalid output shape size: %zu", outputShape.size());
            return {};
        }
        
        int numBoxes = 0, numElements = 0;
        try {
            resolveOutputLayout(outputShape, numBoxes, numElements);
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to parse output shape: %s", e.what());
            return {};
        }
        
        if (numBoxes <= 0 || numElements <= 0) {
            obs_log(LOG_ERROR, "[ModelYOLO] Invalid output parameters: numBoxes=%d, numElements=%d", numBoxes, numElements);
            return {};
        }

        // Runtime re-detect end2end if shape is [?, max_det, 6]
        if (!end2endNmsOutput_ && !outputChannelsFirst_ && numElements == 6 && numBoxes <= 1000) {
            end2endNmsOutput_ = true;
            obs_log(LOG_INFO, "[ModelYOLO] Runtime promote to end2end-NMS (boxes=%d elems=6)", numBoxes);
        }
        
        cv::Size originalSize(input.cols, input.rows);
        
        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        
        std::vector<Detection> detections;
        
        try {
            if (end2endNmsOutput_ && numElements == 6) {
                detections = postprocessEnd2End(outputData, numBoxes, letterboxInfo, originalSize);
            } else {
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
        {
            const std::string &dev = currentDevice_;
            const bool epGpu = dev.find("cuda") != std::string::npos ||
                               dev.find("tensorrt") != std::string::npos ||
                               dev.find("dml") != std::string::npos ||
                               dev.find("rocm") != std::string::npos;
            latency.isGpuPath = epGpu;
        }

        latencyStats_.addSample(latency);

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLatencyLogTime_).count() >=
            LATENCY_LOG_INTERVAL_MS) {
            lastLatencyLogTime_ = now;
            const std::string &dev = currentDevice_;
            const char *pathLabel = "CPU EP";
            if (dev.find("cuda") != std::string::npos || dev.find("tensorrt") != std::string::npos) {
                // After HAVE_CUDA: preprocess may be GPU (no +cpu_pre) or still CPU
                if (dev.find("cpu_pre") != std::string::npos)
                    pathLabel = "CUDA EP · 预处理CPU";
                else if (useGpuMemory_)
                    pathLabel = "CUDA EP · 预处理GPU";
                else
                    pathLabel = "CUDA EP";
            } else if (dev.find("dml") != std::string::npos) {
                pathLabel = "DML EP";
            }
            obs_log(LOG_INFO, "[ModelYOLO] 延迟统计 (%s) device=%s:\n%s", pathLabel, dev.c_str(),
                    latencyStats_.getSummary().c_str());
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

void ModelYOLO::detectOutputLayoutFromShape(const std::vector<int64_t>& shape)
{
    // Version is decided by the user's model_version setting (passed into the
    // constructor as Version), NOT guessed from the output shape. Guessing caused
    // mis-decodes (e.g. a YOLOv5 objectness model treated as v8 -> objectness read
    // as class0 -> thousands of air boxes). We only infer layout + numClasses here.
    outputChannelsFirst_ = true; // default Ultralytics export [1, 4+nc, N]
    end2endNmsOutput_ = false;
    int detectedClasses = 80;

    if (shape.size() < 3) {
        numClasses_ = 80;
        obs_log(LOG_WARNING, "[ModelYOLO] Output rank < 3, default numClasses=80 channels-first");
        return;
    }

    const int64_t d1 = shape[1];
    const int64_t d2 = shape[2];

    // --- End2end NMS export: [1, max_det, 6] (x1,y1,x2,y2,conf,cls) ---
    if (d2 == 6 && d1 >= 1 && d1 <= 1000) {
        end2endNmsOutput_ = true;
        outputChannelsFirst_ = false;
        detectedClasses = 80;
        numClasses_ = detectedClasses;
        obs_log(LOG_INFO,
                "[ModelYOLO] end2end-NMS export [1, max_det=%lld, 6] - skip anchor decode",
                (long long)d1);
        return;
    }

    // --- Layout: box-major [1,N,C] vs channels-first [1,C,N] ---
    // Decide by dimension magnitude only (no version heuristic).
    const bool looksBoxMajor = (d1 > d2 && d2 >= 5 && d2 <= 512);
    const bool looksChFirst = (d2 > d1 && d1 >= 5 && d1 <= 512);
    if (looksBoxMajor) {
        outputChannelsFirst_ = false;
    } else if (looksChFirst) {
        outputChannelsFirst_ = true;
    } else {
        // ambiguous (e.g. square dims) - default Ultralytics channels-first
        outputChannelsFirst_ = true;
    }

    // --- numClasses from shape, per user-chosen version ---
    if (!outputChannelsFirst_) {
        // box-major [1, N, C]
        detectedClasses = (version_ == Version::YOLOv5)
                              ? static_cast<int>(d2 - 5)  // 4 xywh + 1 objectness + nc
                              : static_cast<int>(d2 - 4); // v8/v11: 4 xywh + nc
    } else {
        // channels-first [1, C, N]
        detectedClasses = (version_ == Version::YOLOv5)
                              ? static_cast<int>(d1 - 5)
                              : static_cast<int>(d1 - 4);
    }

    // --- Multi-output raw heads [1,3,H,W,C] are strong evidence of v5/v7 anchor
    // heads. If present, force v5 regardless of user selection (the raw head
    // layout is unambiguous). ---
    if (outputDims_.size() >= 2) {
        for (size_t oi = 1; oi < outputDims_.size(); ++oi) {
            const auto &s = outputDims_[oi];
            if (s.size() == 5 && s[1] == 3 && s[4] >= 6) {
                version_ = Version::YOLOv5;
                outputChannelsFirst_ = false;
                detectedClasses = static_cast<int>(s[4] - 5);
                obs_log(LOG_WARNING,
                        "[ModelYOLO] Raw YOLOv5 heads detected (out[%zu] rank5 anchors=3 C=%lld) "
                        "-> force v5 postprocess nc=%d",
                        oi, (long long)s[4], detectedClasses);
                break;
            }
        }
    }

    if (detectedClasses > 0 && detectedClasses < 1000) {
        numClasses_ = detectedClasses;
    } else {
        obs_log(LOG_WARNING, "[ModelYOLO] Detected numClasses %d invalid, default 80", detectedClasses);
        numClasses_ = 80;
    }
    obs_log(LOG_INFO,
            "[ModelYOLO] version=%d (user-selected) layout=%s N=%lld C=%lld nc=%d",
            static_cast<int>(version_), outputChannelsFirst_ ? "channels-first" : "box-major",
            outputChannelsFirst_ ? (long long)d2 : (long long)d1,
            outputChannelsFirst_ ? (long long)d1 : (long long)d2, numClasses_);
}

void ModelYOLO::resolveOutputLayout(const std::vector<int64_t>& outputShape, int& numBoxes, int& numElements) const
{
    numBoxes = 0;
    numElements = 0;
    if (outputShape.size() < 3)
        return;
    if (outputChannelsFirst_) {
        numElements = static_cast<int>(outputShape[1]);
        numBoxes = static_cast<int>(outputShape[2]);
    } else {
        numBoxes = static_cast<int>(outputShape[1]);
        numElements = static_cast<int>(outputShape[2]);
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

        // Same as OBS: boxes are letterbox-pixel xywh (v5)
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

    // AiMod/OBS v5: class-aware NMS when classIds provided
    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_, classIds);

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
    // Pipeline aligned with reference AiMod yolodml (YoloV8/V11):
    //   1) GenerateProposals in LETTERBOX pixel space
    //   2) Class-aware NMS (only same label suppresses)  ← key vs class-agnostic
    //   3) ScaleBoxes: remove pad / scale → original image
    //   4) Normalize 0..1 for aim/UI
    // Layout: channels-first [1,C,N] (AiMod/OBS standard) OR box-major [1,N,C]
    //         auto-detected at load (your [1,6300,9] is box-major).
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes; // letterbox-pixel xywh for NMS
    std::vector<float> scores;
    std::vector<int> classIds;

    const int stride = 4 + numClasses;
    const bool chFirst = outputChannelsFirst_;
    const float lbW = static_cast<float>(inputWidth_ > 0 ? inputWidth_ : 640);
    const float lbH = static_cast<float>(inputHeight_ > 0 ? inputHeight_ : 640);

    for (int i = 0; i < numBoxes; ++i) {
        float cx, cy, w, h;
        int maxClassId = 0;
        float maxClassProb;

        float secondBest = 0.f;
        if (chFirst) {
            // AiMod YoloV8/V11: ptr[c * num_detections + d]
            cx = rawOutput[0 * numBoxes + i];
            cy = rawOutput[1 * numBoxes + i];
            w = rawOutput[2 * numBoxes + i];
            h = rawOutput[3 * numBoxes + i];
            maxClassProb = rawOutput[4 * numBoxes + i];
            maxClassId = 0;
            for (int c = 1; c < numClasses; ++c) {
                float prob = rawOutput[(4 + c) * numBoxes + i];
                if (prob > maxClassProb) {
                    secondBest = maxClassProb;
                    maxClassProb = prob;
                    maxClassId = c;
                } else if (prob > secondBest) {
                    secondBest = prob;
                }
            }
        } else {
            // box-major row: [cx,cy,w,h,cls0..]
            const float *row = rawOutput + static_cast<size_t>(i) * static_cast<size_t>(stride);
            cx = row[0];
            cy = row[1];
            w = row[2];
            h = row[3];
            maxClassProb = row[4];
            maxClassId = 0;
            for (int c = 1; c < numClasses; ++c) {
                float prob = row[4 + c];
                if (prob > maxClassProb) {
                    secondBest = maxClassProb;
                    maxClassProb = prob;
                    maxClassId = c;
                } else if (prob > secondBest) {
                    secondBest = prob;
                }
            }
        }

        const float confidence = maxClassProb;
        if (confidence < confidenceThreshold_)
            continue;
        // Reject ambiguous peaks (flat class scores → typical empty-region noise)
        if (numClasses > 1) {
            const float margin = confidence - secondBest;
            if (margin < 0.05f)
                continue;
        }

        // Optional class filter (empty = all classes, same as AiMod/OBS default)
        bool isTargetClass = true;
        if (targetClassId_ >= 0)
            isTargetClass = (maxClassId == targetClassId_);
        else if (!targetClasses_.empty())
            isTargetClass = targetClasses_.count(maxClassId) != 0;
        if (!isTargetClass)
            continue;

        // AiMod: keep boxes in letterbox pixel space for NMS (left/top/w/h)
        float lcx = cx, lcy = cy, lw = w, lh = h;
        // Some exports normalize 0..1 to input size
        if (std::max(std::max(std::fabs(cx), std::fabs(cy)), std::max(std::fabs(w), std::fabs(h))) <=
            2.0f) {
            lcx = cx * lbW;
            lcy = cy * lbH;
            lw = w * lbW;
            lh = h * lbH;
        }
        // Clamp to letterbox canvas — unclamped ghosts often sit just outside pad
        lcx = std::max(0.f, std::min(lcx, lbW));
        lcy = std::max(0.f, std::min(lcy, lbH));
        lw = std::max(1.f, std::min(lw, lbW));
        lh = std::max(1.f, std::min(lh, lbH));
        const float left = lcx - lw * 0.5f;
        const float top = lcy - lh * 0.5f;
        if (lw < 1.f || lh < 1.f)
            continue;

        boxes.emplace_back(left, top, lw, lh);
        scores.push_back(confidence);
        classIds.push_back(maxClassId);
    }

    // AiMod NMSBoxes: class-aware (pass classIds)
    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_, classIds);

    const float scale = std::max(letterboxInfo.scale, 1e-6f);
    const float padX = static_cast<float>(letterboxInfo.padX);
    const float padY = static_cast<float>(letterboxInfo.padY);
    const float imgW = static_cast<float>(std::max(1, originalImageSize.width));
    const float imgH = static_cast<float>(std::max(1, originalImageSize.height));

    for (int idx : nmsIndices) {
        // AiMod ScaleBoxes: (x - pad) / scale
        float x = (boxes[idx].x - padX) / scale;
        float y = (boxes[idx].y - padY) / scale;
        float w = boxes[idx].width / scale;
        float h = boxes[idx].height / scale;

        // clamp to image
        float x2 = x + w;
        float y2 = y + h;
        x = std::max(0.f, std::min(x, imgW));
        y = std::max(0.f, std::min(y, imgH));
        x2 = std::max(0.f, std::min(x2, imgW));
        y2 = std::max(0.f, std::min(y2, imgH));
        w = x2 - x;
        h = y2 - y;
        if (w < 1.f || h < 1.f)
            continue;
        // Drop tiny noise / full-frame garbage after scale (pixel space)
        const float areaFrac = (w * h) / (imgW * imgH);
        if (areaFrac < 0.0008f || areaFrac > 0.85f)
            continue;

        Detection det;
        det.classId = classIds[idx];
        det.className = (det.classId < static_cast<int>(classNames_.size()))
                            ? classNames_[static_cast<size_t>(det.classId)]
                            : "Class_" + std::to_string(det.classId);
        det.confidence = scores[idx];
        det.x = x / imgW;
        det.y = y / imgH;
        det.width = w / imgW;
        det.height = h / imgH;
        det.centerX = det.x + det.width * 0.5f;
        det.centerY = det.y + det.height * 0.5f;
        // Keep box fully inside [0,1]
        if (det.x < 0.f || det.y < 0.f || det.x + det.width > 1.02f ||
            det.y + det.height > 1.02f)
            continue;
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

std::vector<Detection> ModelYOLO::postprocessEnd2End(
    const float* rawOutput,
    int numBoxes,
    const LetterboxInfo& letterboxInfo,
    const cv::Size& originalImageSize
) {
    // Ultralytics / Bing_TF / cs2.onnx style:
    //   [1, max_det, 6] rows = x1, y1, x2, y2, conf, class_id  (already NMS'd)
    // Coords are in letterbox pixel space of the model input (320 or 640).
    std::vector<Detection> detections;
    if (!rawOutput || numBoxes <= 0)
        return detections;

    const float scale = std::max(letterboxInfo.scale, 1e-6f);
    const float padX = static_cast<float>(letterboxInfo.padX);
    const float padY = static_cast<float>(letterboxInfo.padY);
    const float imgW = static_cast<float>(std::max(1, originalImageSize.width));
    const float imgH = static_cast<float>(std::max(1, originalImageSize.height));
    const float lbW = static_cast<float>(inputWidth_ > 0 ? inputWidth_ : 640);
    const float lbH = static_cast<float>(inputHeight_ > 0 ? inputHeight_ : 640);

    detections.reserve(static_cast<size_t>(std::min(numBoxes, 300)));

    for (int i = 0; i < numBoxes; ++i) {
        const float *row = rawOutput + static_cast<size_t>(i) * 6u;
        float x1 = row[0];
        float y1 = row[1];
        float x2 = row[2];
        float y2 = row[3];
        float conf = row[4];
        int classId = static_cast<int>(std::lround(row[5]));

        // Padded / empty slots
        if (conf < confidenceThreshold_)
            continue;
        if (conf < 1e-6f)
            continue;

        // Class filter (empty = all)
        bool isTargetClass = true;
        if (targetClassId_ >= 0)
            isTargetClass = (classId == targetClassId_);
        else if (!targetClasses_.empty())
            isTargetClass = targetClasses_.count(classId) != 0;
        if (!isTargetClass)
            continue;

        // Some exports use xywh instead of xyxy — detect if x2,y2 look like w,h
        // Heuristic: if x2 < x1 or y2 < y1, or (x2,y2) much smaller than typical xyxy max
        bool looksXywh = (x2 < x1) || (y2 < y1);
        if (!looksXywh) {
            // also: if both x2,y2 are small relative to center and x2~w style
            // Prefer xyxy when x2>x1 && y2>y1 (cs2 / Bing_TF confirmed xyxy)
            looksXywh = false;
        }

        float bx1, by1, bw, bh;
        if (looksXywh) {
            // treat as cx,cy,w,h
            const float cx = x1, cy = y1, w = x2, h = y2;
            bx1 = cx - w * 0.5f;
            by1 = cy - h * 0.5f;
            bw = w;
            bh = h;
        } else {
            bx1 = x1;
            by1 = y1;
            bw = x2 - x1;
            bh = y2 - y1;
        }

        if (bw < 1.f || bh < 1.f)
            continue;

        // If coords look normalized 0..1, scale to letterbox pixels first
        if (std::max(std::max(std::fabs(bx1), std::fabs(by1)),
                     std::max(std::fabs(bw), std::fabs(bh))) <= 2.0f) {
            bx1 *= lbW;
            by1 *= lbH;
            bw *= lbW;
            bh *= lbH;
        }

        // Letterbox inverse → original image pixels (AiMod ScaleBoxes)
        float ox = (bx1 - padX) / scale;
        float oy = (by1 - padY) / scale;
        float ow = bw / scale;
        float oh = bh / scale;

        float ox2 = ox + ow;
        float oy2 = oy + oh;
        ox = std::max(0.f, std::min(ox, imgW));
        oy = std::max(0.f, std::min(oy, imgH));
        ox2 = std::max(0.f, std::min(ox2, imgW));
        oy2 = std::max(0.f, std::min(oy2, imgH));
        ow = ox2 - ox;
        oh = oy2 - oy;
        if (ow < 1.f || oh < 1.f)
            continue;

        Detection det;
        det.classId = classId;
        det.className = (classId >= 0 && classId < static_cast<int>(classNames_.size()))
                            ? classNames_[static_cast<size_t>(classId)]
                            : "Class_" + std::to_string(classId);
        det.confidence = conf;
        det.x = ox / imgW;
        det.y = oy / imgH;
        det.width = ow / imgW;
        det.height = oh / imgH;
        det.centerX = det.x + det.width * 0.5f;
        det.centerY = det.y + det.height * 0.5f;
        detections.push_back(det);
    }

    return detections;
}

std::vector<int> ModelYOLO::performNMS(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float nmsThreshold,
    const std::vector<int>& classIds
) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    // AiMod NMSBoxes: only suppress boxes of the SAME class (label).
    // When classIds empty → class-agnostic (legacy OBS path).
    const bool useClass = !classIds.empty() && classIds.size() == boxes.size();

    // Same-class "ghost" boxes next to a real target often have LOW IoU (side-by-side
    // partials / multi-anchor). Pure IoU NMS keeps them even at nms=0.05.
    // Also suppress if centers are very close relative to box size (soft-NMS style merge).
    auto centerDist2 = [](const cv::Rect2f &a, const cv::Rect2f &b) {
        const float acx = a.x + a.width * 0.5f;
        const float acy = a.y + a.height * 0.5f;
        const float bcx = b.x + b.width * 0.5f;
        const float bcy = b.y + b.height * 0.5f;
        const float dx = acx - bcx;
        const float dy = acy - bcy;
        return dx * dx + dy * dy;
    };

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];

        if (suppressed[idx]) {
            continue;
        }

        keep.push_back(idx);

        const cv::Rect2f &bi = boxes[idx];
        // Merge radius: fraction of larger box diagonal (letterbox pixels)
        const float diag =
            std::sqrt(bi.width * bi.width + bi.height * bi.height);
        // 0.45 * diag → kill near-duplicates that barely overlap
        const float mergeR = std::max(8.f, 0.45f * diag);
        const float mergeR2 = mergeR * mergeR;

        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];

            if (suppressed[idx2]) {
                continue;
            }

            // AiMod: if (objects[j].label != label_i) skip
            if (useClass && classIds[idx] != classIds[idx2]) {
                continue;
            }

            const cv::Rect2f &bj = boxes[idx2];
            float iou = this->calculateIoU(bi, bj);

            // IoU suppress (standard)
            if (iou > nmsThreshold) {
                suppressed[idx2] = true;
                continue;
            }

            // Center-distance suppress for same-class ghosts (IoU may be ~0)
            if (centerDist2(bi, bj) <= mergeR2) {
                // Prefer keeping larger / higher-score box (idx already higher score)
                // Only merge if sizes are comparable (don't swallow a tiny head into body)
                const float ai = bi.width * bi.height;
                const float aj = bj.width * bj.height;
                const float ratio = (ai > 1e-3f && aj > 1e-3f)
                                       ? (std::min(ai, aj) / std::max(ai, aj))
                                       : 0.f;
                if (ratio > 0.25f) { // similar size → ghost duplicate
                    suppressed[idx2] = true;
                }
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
    if (unionArea <= 1e-6f) return 0.f;

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

    // Only overwrite numClasses_ if we detected more classes from file,
    // or if the model hasn't been parsed yet (still default 80).
    const int fileCount = static_cast<int>(classNames_.size());
    if (fileCount > numClasses_ || numClasses_ == 80) {
        numClasses_ = fileCount;
    }

    obs_log(LOG_INFO, "[ModelYOLO] Loaded %d class names (numClasses=%d)", fileCount, numClasses_);
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
        if (currentDevice_ == "cuda" || currentDevice_ == "tensorrt") {
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
#ifdef HAVE_CUDA
            // MUST set device before any cudaMalloc / ORT Cuda MemoryInfo
            {
                int devCount = 0;
                if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount <= 0) {
                    obs_log(LOG_ERROR, "[ModelYOLO] No CUDA devices for GPU memory");
                    return false;
                }
                if (cudaSetDevice(0) != cudaSuccess) {
                    obs_log(LOG_ERROR, "[ModelYOLO] cudaSetDevice(0) failed in initializeGpuMemory");
                    return false;
                }
            }

            // Real CUDA device memory (NOT CreateCpu — that was a bug).
            cudaMemInfo_ = std::make_unique<Ort::MemoryInfo>(
                "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault
            );
            gpuMemInfo_ = std::make_unique<Ort::MemoryInfo>(
                "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault
            );

            size_t inputBytes = static_cast<size_t>(3) * inputHeight_ * inputWidth_ * sizeof(float);
            if (cudaInputBuffer_ && cudaInputBufferBytes_ < inputBytes) {
                cudaFree(cudaInputBuffer_);
                cudaInputBuffer_ = nullptr;
                cudaInputBufferBytes_ = 0;
                cudaInputTensor_.reset();
            }
            if (!cudaInputBuffer_) {
                if (cudaMalloc(&cudaInputBuffer_, inputBytes) != cudaSuccess) {
                    obs_log(LOG_ERROR, "[ModelYOLO] cudaMalloc input failed");
                    useGpuMemory_ = false;
                    return false;
                }
                cudaInputBufferBytes_ = inputBytes;
            }

            std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
            inputShapeCache_ = inputShape;
            cudaInputTensor_ = std::make_unique<Ort::Value>(
                Ort::Value::CreateTensor<float>(
                    *cudaMemInfo_,
                    static_cast<float*>(cudaInputBuffer_),
                    inputBytes / sizeof(float),
                    inputShape.data(),
                    inputShape.size()
                ));

            // Optional device output buffer for IoBinding
            if (!outputDims_.empty()) {
                size_t outElems = 1;
                for (auto d : outputDims_[0]) {
                    if (d > 0) outElems *= static_cast<size_t>(d);
                }
                size_t outBytes = outElems * sizeof(float);
                if (cudaOutputBuffer_ && cudaOutputBufferBytes_ < outBytes) {
                    cudaFree(cudaOutputBuffer_);
                    cudaOutputBuffer_ = nullptr;
                    cudaOutputBufferBytes_ = 0;
                    gpuOutputTensor_.reset();
                }
                if (!cudaOutputBuffer_ && outElems > 0) {
                    if (cudaMalloc(&cudaOutputBuffer_, outBytes) == cudaSuccess) {
                        cudaOutputBufferBytes_ = outBytes;
                        gpuOutputTensor_ = std::make_unique<Ort::Value>(
                            Ort::Value::CreateTensor<float>(
                                *cudaMemInfo_,
                                static_cast<float*>(cudaOutputBuffer_),
                                outElems,
                                outputDims_[0].data(),
                                outputDims_[0].size()
                            ));
                    }
                }
            }

            gpuInputTensor_.reset(); // prefer cudaInputTensor_
            useGpuMemory_ = true;
            obs_log(LOG_INFO, "[ModelYOLO] CUDA device memory allocated: input %dx%d (%zu bytes)",
                    inputWidth_, inputHeight_, inputBytes);
            return true;
#else
            obs_log(LOG_WARNING, "[ModelYOLO] HAVE_CUDA not defined, cannot allocate device memory");
            return false;
#endif
#else
            obs_log(LOG_WARNING, "[ModelYOLO] CUDA EP not available, using CPU memory");
            return false;
#endif
        } else if (currentDevice_ == "dml") {
#ifdef HAVE_ONNXRUNTIME_DML_EP
            // DirectML uses CPU staging; IOBinding still useful
            gpuMemInfo_ = std::make_unique<Ort::MemoryInfo>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
            );
            
            useGpuMemory_ = false;
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
    cudaInputTensor_.reset();
    gpuInputTensor_.reset();
    gpuOutputTensor_.reset();

    if (gpuAllocator_) {
        gpuAllocator_.reset();
        gpuAllocator_ = nullptr;
    }
    
    if (gpuMemInfo_) {
        gpuMemInfo_.reset();
        gpuMemInfo_ = nullptr;
    }
    if (cudaMemInfo_) {
        cudaMemInfo_.reset();
        cudaMemInfo_ = nullptr;
    }

#ifdef HAVE_CUDA
    // Device buffers are owned by cudaMalloc — free after Ort::Value release
    if (cudaInputBuffer_) {
        cudaFree(cudaInputBuffer_);
        cudaInputBuffer_ = nullptr;
        cudaInputBufferBytes_ = 0;
    }
    if (cudaOutputBuffer_) {
        cudaFree(cudaOutputBuffer_);
        cudaOutputBuffer_ = nullptr;
        cudaOutputBufferBytes_ = 0;
    }
    if (cudaBgraStaging_) {
        cudaFree(cudaBgraStaging_);
        cudaBgraStaging_ = nullptr;
        cudaBgraStagingBytes_ = 0;
    }
#endif
    
    useGpuMemory_ = false;
    obs_log(LOG_INFO, "[ModelYOLO] GPU memory released");
}

// ============================================================================
// 阶段2：CUDA纹理共享
// ============================================================================

bool ModelYOLO::initializeCudaInterop() {
#ifdef HAVE_CUDA
    if (!useGpuMemory_ || !cudaInputBuffer_) {
        obs_log(LOG_WARNING, "[ModelYOLO] Cannot init CUDA path: device input buffer not ready");
        return false;
    }
    
    try {
        // Bind CUDA to device 0 (must match ORT CUDA EP device_id and OBS D3D adapter).
        // Do NOT call cudaGraphicsD3D11RegisterResource on OBS texrender textures from
        // the inference thread — that causes invalid device ordinal + DXGI device-removed.
        int devCount = 0;
        cudaError_t err = cudaGetDeviceCount(&devCount);
        if (err != cudaSuccess || devCount <= 0) {
            obs_log(LOG_ERROR, "[ModelYOLO] No CUDA devices: %s",
                    err != cudaSuccess ? cudaGetErrorString(err) : "count=0");
            return false;
        }
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            obs_log(LOG_ERROR, "[ModelYOLO] cudaSetDevice(0) failed: %s", cudaGetErrorString(err));
            return false;
        }
        int cur = -1;
        cudaGetDevice(&cur);
        obs_log(LOG_INFO, "[ModelYOLO] CUDA active device=%d (count=%d)", cur, devCount);

        if (!cudaStream_) {
            cudaStream_t stream;
            err = cudaStreamCreate(&stream);
            if (err != cudaSuccess) {
                obs_log(LOG_ERROR, "[ModelYOLO] Failed to create CUDA stream: %s",
                        cudaGetErrorString(err));
                return false;
            }
            cudaStream_ = stream;
        }

        cudaRegisteredTex_ = nullptr;
        if (cudaResource_) {
            cudaGraphicsUnregisterResource(cudaResource_);
            cudaResource_ = nullptr;
        }
        cudaInteropInitialized_ = true;
        obs_log(LOG_INFO, "[ModelYOLO] CUDA path ready (D3D11 interop disabled; use float-buffer path)");
        return true;
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] CUDA path init failed: %s", e.what());
        return false;
    }
#else
    obs_log(LOG_WARNING, "[ModelYOLO] CUDA not available");
    return false;
#endif
}

void ModelYOLO::releaseCudaInterop() {
#ifdef HAVE_CUDA
    if (cudaResource_) {
        cudaGraphicsUnregisterResource(cudaResource_);
        cudaResource_ = nullptr;
    }
    cudaRegisteredTex_ = nullptr;
    
    if (cudaStream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(cudaStream_));
        cudaStream_ = nullptr;
    }
    // Device input/output buffers are owned by releaseGpuMemory()
    cudaInteropInitialized_ = false;
    obs_log(LOG_INFO, "[ModelYOLO] CUDA interop released");
#endif
}

bool ModelYOLO::initializeDmlPreprocessor() {
    // Name is historical: this enables the float CHW preprocessed path used by
    // filter GPU-texture mode for DML and CUDA/TRT (CPU letterbox on render thread).
    if (dmlInteropInitialized_) {
        return true;
    }
    
    try {
        dmlPreprocessor_ = std::make_unique<DmlPreprocessor>();
        if (!dmlPreprocessor_->initialize()) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to initialize float preprocessor");
            dmlPreprocessor_.reset();
            return false;
        }
        
        dmlInteropInitialized_ = true;
        obs_log(LOG_INFO, "[ModelYOLO] Float preprocess path initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Float preprocessor init exception: %s", e.what());
        if (dmlPreprocessor_) {
            dmlPreprocessor_.reset();
        }
        return false;
    }
}

void ModelYOLO::releaseDmlInterop() {
    if (dmlPreprocessor_) {
        dmlPreprocessor_->release();
        dmlPreprocessor_.reset();
    }
    
    dmlInteropInitialized_ = false;
    obs_log(LOG_INFO, "[ModelYOLO] Float preprocess path released");
}

std::vector<Detection> ModelYOLO::inferenceFromTexture(void* d3d11Texture,
                                                         int cropX, int cropY, int cropW, int cropH,
                                                         int originalWidth, int originalHeight,
                                                         InferenceLatency* outLatency) {
#ifdef HAVE_CUDA
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    InferenceLatency latency;
    latency.isGpuPath = true;
    latency.gpuCopyMs = 0.0;

    if (!cudaInteropInitialized_ || !session_ || !cudaInputBuffer_ || !cudaStream_) {
        return {};
    }
    if (!d3d11Texture || cropW <= 0 || cropH <= 0) {
        return {};
    }

    if (inputNamesChar_.empty() || outputNamesChar_.empty()) {
        inputNamesChar_.clear();
        outputNamesChar_.clear();
        for (const auto& name : inputNames_) inputNamesChar_.push_back(name.get());
        for (const auto& name : outputNames_) outputNamesChar_.push_back(name.get());
        if (inputNamesChar_.empty() || outputNamesChar_.empty()) return {};
    }

    std::vector<Detection> detections;

    try {
        auto cudaStartTime = std::chrono::high_resolution_clock::now();

        ID3D11Texture2D* d3dTex = static_cast<ID3D11Texture2D*>(d3d11Texture);
        cudaStream_t stream = static_cast<cudaStream_t>(cudaStream_);

        // Re-register if texture pointer changed (texrender may recycle or replace)
        if (cudaRegisteredTex_ != d3d11Texture) {
            if (cudaResource_) {
                cudaGraphicsUnregisterResource(cudaResource_);
                cudaResource_ = nullptr;
            }
            cudaError_t regErr = cudaGraphicsD3D11RegisterResource(
                &cudaResource_, d3dTex, cudaGraphicsRegisterFlagsNone);
            if (regErr != cudaSuccess) {
                obs_log(LOG_ERROR, "[ModelYOLO] D3D11 register failed: %s",
                        cudaGetErrorString(regErr));
                cudaRegisteredTex_ = nullptr;
                throw std::runtime_error(std::string("D3D11 register failed: ") +
                                         cudaGetErrorString(regErr));
            }
            cudaRegisteredTex_ = d3d11Texture;
            obs_log(LOG_INFO, "[ModelYOLO] D3D11 texture registered for CUDA interop");
        }

        cudaError_t err = cudaGraphicsMapResources(1, &cudaResource_, stream);
        if (err != cudaSuccess) {
            obs_log(LOG_ERROR, "[ModelYOLO] Failed to map texture: %s",
                    cudaGetErrorString(err));
            // Invalidate cache — texture may have been destroyed
            if (cudaResource_) {
                cudaGraphicsUnregisterResource(cudaResource_);
                cudaResource_ = nullptr;
            }
            cudaRegisteredTex_ = nullptr;
            throw std::runtime_error(std::string("D3D11 map failed: ") + cudaGetErrorString(err));
        }

        cudaArray_t cudaArray = nullptr;
        err = cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource_, 0, 0);
        if (err != cudaSuccess || !cudaArray) {
            cudaGraphicsUnmapResources(1, &cudaResource_, stream);
            throw std::runtime_error("cudaGraphicsSubResourceGetMappedArray failed");
        }

        size_t requiredBytes = static_cast<size_t>(3) * inputHeight_ * inputWidth_ * sizeof(float);
        if (cudaInputBufferBytes_ < requiredBytes) {
            cudaGraphicsUnmapResources(1, &cudaResource_, stream);
            throw std::runtime_error("CUDA input buffer too small");
        }

        auto kernelStartTime = std::chrono::high_resolution_clock::now();
        bool preprocessSuccess = cudaLetterboxAndPreprocessCrop(
            cudaArray,
            static_cast<float*>(cudaInputBuffer_),
            originalWidth,
            originalHeight,
            cropX, cropY, cropW, cropH,
            inputWidth_,
            inputHeight_,
            stream
        );
        auto kernelEndTime = std::chrono::high_resolution_clock::now();
        latency.cudaKernelMs = std::chrono::duration<double, std::milli>(kernelEndTime - kernelStartTime).count();

        cudaGraphicsUnmapResources(1, &cudaResource_, stream);

        if (!preprocessSuccess) {
            throw std::runtime_error("CUDA letterbox crop preprocess failed");
        }

        // Ensure device tensor wraps current buffer
        if (inputShapeCache_.size() != 4) {
            inputShapeCache_ = {1, 3, static_cast<int64_t>(inputHeight_), static_cast<int64_t>(inputWidth_)};
        }
        if (!cudaMemInfo_) {
            cudaMemInfo_ = std::make_unique<Ort::MemoryInfo>(
                "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        }
        if (!cudaInputTensor_) {
            cudaInputTensor_ = std::make_unique<Ort::Value>(
                Ort::Value::CreateTensor<float>(
                    *cudaMemInfo_,
                    static_cast<float*>(cudaInputBuffer_),
                    requiredBytes / sizeof(float),
                    inputShapeCache_.data(),
                    inputShapeCache_.size()
                ));
        }

        // FP16 models: D2H convert then CPU tensor (device FP16 path not wired yet)
        bool useDeviceInput = !isFp16Model_;
        if (isFp16Model_) {
            auto copyStart = std::chrono::high_resolution_clock::now();
            if (inputBuffer_.size() < inputBufferSize_) inputBuffer_.resize(inputBufferSize_);
            if (inputBufferFp16_.size() < inputBufferSize_) inputBufferFp16_.resize(inputBufferSize_);
            cudaMemcpyAsync(inputBuffer_.data(), cudaInputBuffer_, requiredBytes,
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            convertFloatBufferToHalf(inputBuffer_.data(), inputBufferFp16_.data(), inputBufferSize_);
            auto copyEnd = std::chrono::high_resolution_clock::now();
            latency.gpuCopyMs = std::chrono::duration<double, std::milli>(copyEnd - copyStart).count();
            useDeviceInput = false;
        } else {
            // Kernel wrote device buffer; sync before ORT may consume it
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                obs_log(LOG_ERROR, "[ModelYOLO] CUDA stream sync failed: %s",
                        cudaGetErrorString(err));
                return {};
            }
        }

        latency.preprocessMs = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - cudaStartTime).count();

        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputTensors;
        bool usedBoundOutput = false;

        if (useDeviceInput && useIOBinding_ && ioBinding_ && cudaInputTensor_) {
            try {
                ioBinding_->ClearBoundInputs();
                ioBinding_->ClearBoundOutputs();
                ioBinding_->BindInput(inputNamesChar_[0], *cudaInputTensor_);
                if (gpuOutputTensor_) {
                    ioBinding_->BindOutput(outputNamesChar_[0], *gpuOutputTensor_);
                } else {
                    // Let ORT allocate CPU output
                    ensureCpuMemInfo();
                    ioBinding_->BindOutput(outputNamesChar_[0], static_cast<const OrtMemoryInfo*>(*cpuMemInfo_));
                }
                session_->Run(runOptions, *ioBinding_);
                usedBoundOutput = true;
            } catch (const std::exception& e) {
                obs_log(LOG_WARNING, "[ModelYOLO] CUDA IoBinding Run failed: %s, fallback session->Run", e.what());
                usedBoundOutput = false;
                std::vector<Ort::Value> inTs;
                Ort::Value inT = Ort::Value::CreateTensor<float>(
                    *cudaMemInfo_,
                    static_cast<float*>(cudaInputBuffer_),
                    requiredBytes / sizeof(float),
                    inputShapeCache_.data(),
                    inputShapeCache_.size()
                );
                inTs.push_back(std::move(inT));
                outputTensors = session_->Run(
                    runOptions,
                    inputNamesChar_.data(), inTs.data(), inTs.size(),
                    outputNamesChar_.data(), outputNamesChar_.size()
                );
            }
        } else {
            ensureCpuMemInfo();
            Ort::Value inT{nullptr};
            if (isFp16Model_) {
                inT = Ort::Value::CreateTensor<Ort::Float16_t>(
                    *cpuMemInfo_, inputBufferFp16_.data(), inputBufferSize_,
                    inputShapeCache_.data(), inputShapeCache_.size());
            } else {
                inT = Ort::Value::CreateTensor<float>(
                    *cudaMemInfo_,
                    static_cast<float*>(cudaInputBuffer_),
                    requiredBytes / sizeof(float),
                    inputShapeCache_.data(),
                    inputShapeCache_.size()
                );
            }
            std::vector<Ort::Value> inTs;
            inTs.push_back(std::move(inT));
            outputTensors = session_->Run(
                runOptions,
                inputNamesChar_.data(), inTs.data(), inTs.size(),
                outputNamesChar_.data(), outputNamesChar_.size()
            );
        }

        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();

        float* outputData = nullptr;
        std::vector<int64_t> outputShape;

        if (usedBoundOutput && gpuOutputTensor_) {
            // Copy device output → host for postprocess
            size_t outElems = 1;
            for (auto d : outputDims_[0]) if (d > 0) outElems *= static_cast<size_t>(d);
            if (outputBuffer_.size() < outElems) outputBuffer_.resize(outElems);
            cudaMemcpy(outputBuffer_.data(), cudaOutputBuffer_, outElems * sizeof(float),
                       cudaMemcpyDeviceToHost);
            outputData = outputBuffer_.data();
            outputShape = outputDims_[0];
        } else if (usedBoundOutput && ioBinding_) {
            // BindOutput to CPU MemoryInfo — outputs via GetOutputValues
            try {
                std::vector<Ort::Value> boundOuts = ioBinding_->GetOutputValues();
                if (!boundOuts.empty() && boundOuts[0].IsTensor()) {
                    auto info = boundOuts[0].GetTensorTypeAndShapeInfo();
                    outputShape = info.GetShape();
                    ONNXTensorElementDataType t = info.GetElementType();
                    if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                        const Ort::Float16_t* fp16 = boundOuts[0].GetTensorData<Ort::Float16_t>();
                        size_t n = 1;
                        for (auto d : outputShape) if (d > 0) n *= static_cast<size_t>(d);
                        if (outputFp32Scratch_.size() < n) outputFp32Scratch_.resize(n);
                        convertHalfBufferToFloat(fp16, outputFp32Scratch_.data(), n);
                        outputData = outputFp32Scratch_.data();
                    } else {
                        outputData = boundOuts[0].GetTensorMutableData<float>();
                    }
                }
            } catch (...) {
                return {};
            }
        } else {
            if (outputTensors.empty() || !outputTensors[0].IsTensor()) return {};
            auto outputTypeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
            outputShape = outputTypeInfo.GetShape();
            ONNXTensorElementDataType outputType = outputTypeInfo.GetElementType();
            if (outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                const Ort::Float16_t* fp16Data = outputTensors[0].GetTensorData<Ort::Float16_t>();
                size_t outputSize = 1;
                for (auto dim : outputShape) if (dim > 0) outputSize *= static_cast<size_t>(dim);
                if (outputFp32Scratch_.size() < outputSize) outputFp32Scratch_.resize(outputSize);
                convertHalfBufferToFloat(fp16Data, outputFp32Scratch_.data(), outputSize);
                outputData = outputFp32Scratch_.data();
            } else {
                outputData = outputTensors[0].GetTensorMutableData<float>();
            }
        }

        if (!outputData || outputShape.size() < 3) return {};

        int numBoxes = 0, numElements = 0;
        resolveOutputLayout(outputShape, numBoxes, numElements);
        if (numBoxes <= 0) return {};

        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        // Letterbox was computed on crop region
        LetterboxInfo letterboxInfo = calculateLetterboxParams(cropW, cropH, inputWidth_, inputHeight_);
        // Postprocess in crop-local space; filter remaps using cropX/Y + full size
        cv::Size originalSize(cropW, cropH);

        if (end2endNmsOutput_ || (!outputChannelsFirst_ && numElements == 6 && numBoxes <= 1000)) {
            end2endNmsOutput_ = true;
            detections = postprocessEnd2End(outputData, numBoxes, letterboxInfo, originalSize);
        } else {
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
        }
        (void)numElements;

        auto postprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.postprocessMs = std::chrono::duration<double, std::milli>(postprocessEndTime - postprocessStartTime).count();
        latency.totalMs = std::chrono::duration<double, std::milli>(postprocessEndTime - totalStartTime).count();
        latencyStats_.addSample(latency);
        if (outLatency) *outLatency = latency;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastLatencyLogTime_).count() >= LATENCY_LOG_INTERVAL_MS) {
            lastLatencyLogTime_ = now;
            obs_log(LOG_INFO, "[ModelYOLO] 延迟统计 (CUDA纹理路径):\n%s", latencyStats_.getSummary().c_str());
        }

        (void)originalWidth;
        (void)originalHeight;
        return detections;

    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Texture inference failed: %s", e.what());
        return {};
    }
#else
    (void)d3d11Texture;
    (void)cropX; (void)cropY; (void)cropW; (void)cropH;
    (void)originalWidth;
    (void)originalHeight;
    (void)outLatency;
    return {};
#endif
}

std::vector<Detection> ModelYOLO::inferenceFromTextureDml(const DmlPreprocessedFrame& preprocessedFrame,
                                                          int originalWidth, int originalHeight,
                                                          InferenceLatency* outLatency) {
    // AiMod-style: preprocessed host CHW float → CreateTensor → Run (no IoBinding for DML)
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    InferenceLatency latency;
    latency.isGpuPath = true;
    latency.preprocessMs = 0.0;  // preprocessing already done on render thread
    
    if (!session_) {
        obs_log(LOG_ERROR, "[ModelYOLO] Preprocessed inference: session not initialized");
        return {};
    }
    
    if (!preprocessedFrame.valid()) {
        obs_log(LOG_ERROR, "[ModelYOLO] Preprocessed inference: invalid frame");
        return {};
    }
    
    if (preprocessedFrame.width != inputWidth_ || preprocessedFrame.height != inputHeight_) {
        obs_log(LOG_ERROR, "[ModelYOLO] DML inference: preprocessed frame size mismatch "
                "(got %dx%d, expected %dx%d)",
                preprocessedFrame.width, preprocessedFrame.height,
                inputWidth_, inputHeight_);
        return {};
    }

    if (inputNamesChar_.empty() || outputNamesChar_.empty()) {
        inputNamesChar_.clear();
        outputNamesChar_.clear();
        for (const auto& name : inputNames_) inputNamesChar_.push_back(name.get());
        for (const auto& name : outputNames_) outputNamesChar_.push_back(name.get());
        if (inputNamesChar_.empty() || outputNamesChar_.empty()) return {};
    }
    
    std::vector<Detection> detections;
    
    try {
        auto inferenceStartTime = std::chrono::high_resolution_clock::now();
        
        size_t dataSize = preprocessedFrame.data.size();
        if (dataSize != inputBufferSize_) {
            inputBufferSize_ = dataSize;
            cpuInputTensorElems_ = 0;
        }
        if (isFp16Model_) {
            if (inputBufferFp16_.size() < dataSize) inputBufferFp16_.resize(dataSize);
            convertFloatBufferToHalf(preprocessedFrame.data.data(), inputBufferFp16_.data(), dataSize);
        } else {
            if (inputBuffer_.size() < dataSize) inputBuffer_.resize(dataSize);
            std::memcpy(inputBuffer_.data(), preprocessedFrame.data.data(), dataSize * sizeof(float));
        }

        ensureCpuMemInfo();
        if (inputShapeCache_.size() != 4) {
            inputShapeCache_ = {1, 3, static_cast<int64_t>(inputHeight_),
                                static_cast<int64_t>(inputWidth_)};
        }

        Ort::RunOptions runOptions;
        std::vector<Ort::Value> outputTensors;
        bool usedBoundOutput = false;

        // Prefer AiMod simple Run for DML; optional IoBinding only if explicitly enabled
        const bool aimodSimple = (currentDevice_.find("dml") != std::string::npos) || !useIOBinding_;
        if (aimodSimple || !(ioBinding_ && cpuInputTensor_ && cpuOutputTensor_)) {
            Ort::Value inT{nullptr};
            if (isFp16Model_) {
                inT = Ort::Value::CreateTensor<Ort::Float16_t>(
                    *cpuMemInfo_, inputBufferFp16_.data(), dataSize,
                    inputShapeCache_.data(), inputShapeCache_.size());
            } else {
                inT = Ort::Value::CreateTensor<float>(
                    *cpuMemInfo_, inputBuffer_.data(), dataSize,
                    inputShapeCache_.data(), inputShapeCache_.size());
            }
            std::vector<Ort::Value> inputTensors;
            inputTensors.push_back(std::move(inT));
            outputTensors = session_->Run(
                runOptions,
                inputNamesChar_.data(),
                inputTensors.data(),
                inputTensors.size(),
                outputNamesChar_.data(),
                outputNamesChar_.size()
            );
        } else {
            ensureCpuInputTensor();
            ensureCpuOutputTensor();
            ioBinding_->ClearBoundInputs();
            ioBinding_->ClearBoundOutputs();
            ioBinding_->BindInput(inputNamesChar_[0], *cpuInputTensor_);
            ioBinding_->BindOutput(outputNamesChar_[0], *cpuOutputTensor_);
            session_->Run(runOptions, *ioBinding_);
            usedBoundOutput = true;
        }

        auto inferenceEndTime = std::chrono::high_resolution_clock::now();
        latency.inferenceMs = std::chrono::duration<double, std::milli>(inferenceEndTime - inferenceStartTime).count();

        float* outputData = nullptr;
        std::vector<int64_t> outputShape;
        if (usedBoundOutput) {
            if (isFp16Output_) {
                if (outputFp32Scratch_.size() < outputElementCount_)
                    outputFp32Scratch_.resize(outputElementCount_);
                convertHalfBufferToFloat(outputBufferFp16_.data(), outputFp32Scratch_.data(), outputElementCount_);
                outputData = outputFp32Scratch_.data();
            } else {
                outputData = outputBuffer_.data();
            }
            outputShape = outputShapeCache_;
        } else {
            if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
                obs_log(LOG_ERROR, "[ModelYOLO] DML inference: invalid output tensor");
                return {};
            }
            auto outputTypeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType outputType = outputTypeInfo.GetElementType();
            outputShape = outputTypeInfo.GetShape();
            if (outputType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                const Ort::Float16_t* fp16Data = outputTensors[0].GetTensorData<Ort::Float16_t>();
                size_t outputSize = 1;
                for (auto dim : outputShape) outputSize *= static_cast<size_t>(dim > 0 ? dim : 1);
                if (outputFp32Scratch_.size() < outputSize) outputFp32Scratch_.resize(outputSize);
                convertHalfBufferToFloat(fp16Data, outputFp32Scratch_.data(), outputSize);
                outputData = outputFp32Scratch_.data();
            } else {
                outputData = outputTensors[0].GetTensorMutableData<float>();
            }
        }

        if (!outputData) {
            obs_log(LOG_ERROR, "[ModelYOLO] DML inference: null output data");
            return {};
        }
        
        if (outputShape.size() < 3) {
            obs_log(LOG_ERROR, "[ModelYOLO] DML inference: invalid output shape");
            return {};
        }
        
        auto postprocessStartTime = std::chrono::high_resolution_clock::now();
        
        LetterboxInfo letterboxInfo = calculateLetterboxParams(
            preprocessedFrame.srcWidth, preprocessedFrame.srcHeight,
            inputWidth_, inputHeight_);
        
        int numBoxes = 0, numElements = 0;
        resolveOutputLayout(outputShape, numBoxes, numElements);
        if (numBoxes <= 0) {
            obs_log(LOG_ERROR, "[ModelYOLO] DML inference: invalid numBoxes from shape");
            return {};
        }
        if (end2endNmsOutput_ || (!outputChannelsFirst_ && numElements == 6 && numBoxes <= 1000)) {
            end2endNmsOutput_ = true;
            detections = postprocessEnd2End(
                outputData, numBoxes, letterboxInfo,
                cv::Size(originalWidth, originalHeight));
        } else if (version_ == Version::YOLOv5) {
            detections = postprocessYOLOv5(
                outputData, numBoxes, numClasses_,
                letterboxInfo,
                cv::Size(originalWidth, originalHeight)
            );
        } else if (version_ == Version::YOLOv8) {
            detections = postprocessYOLOv8(
                outputData, numBoxes, numClasses_,
                letterboxInfo,
                cv::Size(originalWidth, originalHeight)
            );
        } else if (version_ == Version::YOLOv11) {
            detections = postprocessYOLOv11(
                outputData, numBoxes, numClasses_,
                letterboxInfo,
                cv::Size(originalWidth, originalHeight)
            );
        }
        (void)numElements;
        
        auto postprocessEndTime = std::chrono::high_resolution_clock::now();
        latency.postprocessMs = std::chrono::duration<double, std::milli>(postprocessEndTime - postprocessStartTime).count();
        
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        latency.totalMs = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();
        
        latencyStats_.addSample(latency);
        
        if (outLatency) {
            *outLatency = latency;
        }
        
        obs_log(LOG_INFO, "[ModelYOLO] DML texture inference: %zu detections, total %.2fms "
                "(preprocess 0ms (render-thread), inference %.2fms, postprocess %.2fms)",
                detections.size(), latency.totalMs, latency.inferenceMs, latency.postprocessMs);
        
        return detections;
        
    } catch (const Ort::Exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Preprocessed ONNX Runtime error: %s", e.what());
        return {};
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[ModelYOLO] Preprocessed inference error: %s", e.what());
        return {};
    }
}
