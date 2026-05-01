#include "yolo-detector-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#define NOMINMAX
#include <wchar.h>
#include <windows.h>
#include <gdiplus.h>
#include <commdlg.h>
#pragma comment(lib, "gdiplus.lib")
#include "MouseController.hpp"
#include "MouseControllerFactory.hpp"
#include "ConfigManager.hpp"
#ifdef HAVE_CUDA
#include <d3d11.h>
#endif
#endif

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <thread>
#include <regex>
#include <thread>
#include <chrono>
#include <sstream>
#include <functional>
#include <deque>
#include <map>
#include <algorithm>

#include <plugin-support.h>
#include "models/ModelYOLO.h"
#include "models/Detection.h"
#include "HungarianAlgorithm.hpp"
#include "FilterData.h"
#include "obs-utils/obs-utils.h"
#include "consts.h"

// 目标重识别结构体
struct LostTarget {
    int trackId;
    float x, y, width, height;
    float centerX, centerY;
    int lostFrames;
    std::chrono::steady_clock::time_point lostTime;
};

struct yolo_detector_filter : public filter_data, public std::enable_shared_from_this<yolo_detector_filter> {
	std::unique_ptr<ModelYOLO> yoloModel;
	std::mutex yoloModelMutex;
	ModelYOLO::Version modelVersion;

	std::vector<Detection> detections;
	std::mutex detectionsMutex;

	std::vector<Detection> trackedTargets;
	std::mutex trackedTargetsMutex;
	int nextTrackId;
	int maxLostFrames;
	float iouThreshold;
	
	// 多指标融合追踪权重
	float trackingWeightIou;
	float trackingWeightCenter;
	float trackingWeightAspect;
	float trackingWeightArea;
	
	// 目标重识别缓冲区
	std::vector<LostTarget> lostTargets;
	std::mutex lostTargetsMutex;
	int maxReidentifyFrames;  // 重识别最大帧数
	float reidentifyCenterThreshold;  // 重识别中心点距离阈值

	std::string modelPath;
	int inputResolution;
	float confidenceThreshold;
	float nmsThreshold;
	int targetClassId;
	std::vector<int> targetClasses;
	int inferenceIntervalFrames;

	bool showBBox;
	bool showLabel;
	bool showConfidence;
	int bboxLineWidth;
	uint32_t bboxColor;

	bool exportCoordinates;
	std::string coordinateOutputPath;

	bool showFOV;
	int fovRadius;
	uint32_t fovColor;
	int fovCrossLineScale;
	int fovCrossLineThickness;
	int fovCircleThickness;
	bool showFOVCircle;
	bool showFOVCross;

	bool showFOV2;
	int fovRadius2;
	uint32_t fovColor2;
	bool useDynamicFOV;
	bool isInFOV2Mode;
	bool hasTargetInFOV2;

	bool showDetectionResults;
	float labelFontScale;

	int regionX;
	int regionY;
	int regionWidth;
	int regionHeight;
	bool useRegion;

	std::thread inferenceThread;
	std::atomic<bool> inferenceRunning;
	int frameCounter;

	int inferenceFrameWidth;
	int inferenceFrameHeight;
	int cropOffsetX;
	int cropOffsetY;
	std::mutex inferenceFrameSizeMutex;

	uint64_t totalFrames;
	uint64_t inferenceCount;
	double avgInferenceTimeMs;

	std::atomic<bool> isInferencing;

	// 异步推理统计
	std::atomic<int> framesSubmitted{0};
	std::atomic<int> framesInferred{0};
	std::atomic<int> framesConsumed{0};
	std::atomic<int> framesDropped{0};

	// === 四缓冲区异步推理 ===
	static constexpr int BUFFER_COUNT = 4;

	// 输入帧缓冲区（主线程 → 推理线程）
	cv::Mat inputFrames[BUFFER_COUNT];
	int inputFrameWidths[BUFFER_COUNT] = {0};
	int inputFrameHeights[BUFFER_COUNT] = {0};
	int inputCropX[BUFFER_COUNT] = {0};
	int inputCropY[BUFFER_COUNT] = {0};
	int inputCropWidth[BUFFER_COUNT] = {0};
	int inputCropHeight[BUFFER_COUNT] = {0};

	// 输出结果缓冲区（推理线程 → 主线程）
	std::vector<Detection> outputDetections[BUFFER_COUNT];
	std::vector<Detection> outputTrackedTargets[BUFFER_COUNT];
	int outputFrameWidths[BUFFER_COUNT] = {0};
	int outputFrameHeights[BUFFER_COUNT] = {0};
	int outputCropX[BUFFER_COUNT] = {0};
	int outputCropY[BUFFER_COUNT] = {0};

	// 无锁索引管理
	std::atomic<int> inputWriteIdx{0};      // 主线程写入位置
	std::atomic<int> inputReadIdx{0};       // 推理线程读取位置
	std::atomic<int> outputReadyIdx{-1};    // 推理完成位置（-1表示无新结果）
	std::atomic<int64_t> outputSequence{0};  // 输出序列号，用于判断结果是否更新
	std::atomic<int64_t> lastConsumedSeq{-1}; // 上次消费的序列号
	std::atomic<int64_t> lastResultTimestamp{0}; // 上次有新结果的时刻

	// 缓冲区状态：0=空闲, 1=有数据待推理, 2=正在推理, 3=推理完成
	std::atomic<uint8_t> bufferState[BUFFER_COUNT] = {};
	std::atomic<uint8_t> outputState[BUFFER_COUNT] = {};

	std::chrono::high_resolution_clock::time_point lastFpsTime;
	int fpsFrameCount;
	double currentFps;

	gs_effect_t *solidEffect;

	// 线程池相关成员
	std::vector<std::thread> threadPool;
	
#ifdef _WIN32
	// GPU纹理推理支持
	bool useGpuTextureInference = false;
	ID3D11Texture2D* cachedD3D11Texture = nullptr;
	int gpuTextureWidth = 0;
	int gpuTextureHeight = 0;
#endif
	
	std::queue<std::function<void()>> taskQueue;
	std::mutex taskQueueMutex;
	std::condition_variable taskCondition;
	std::atomic<bool> threadPoolRunning;

	// 内存池相关成员
	struct ImageBufferKey {
		int rows;
		int cols;
		int type;

		bool operator==(const ImageBufferKey& other) const {
			return rows == other.rows && cols == other.cols && type == other.type;
		}
	};

	struct ImageBufferKeyHash {
		size_t operator()(const ImageBufferKey& key) const {
			size_t h1 = std::hash<int>()(key.rows);
			size_t h2 = std::hash<int>()(key.cols);
			size_t h3 = std::hash<int>()(key.type);
			return h1 ^ (h2 << 1) ^ (h3 << 2);
		}
	};

	std::unordered_map<ImageBufferKey, std::vector<cv::Mat>, ImageBufferKeyHash> imageBufferPool;
	std::vector<std::vector<Detection>> detectionBufferPool;
	std::mutex bufferPoolMutex;
	const int MAX_BUFFER_POOL_SIZE = 3;
	const int THREAD_POOL_SIZE = 4;

#ifdef _WIN32
	bool showFloatingWindow;
	int floatingWindowWidth;
	int floatingWindowHeight;
	int floatingWindowX;
	int floatingWindowY;
	bool floatingWindowDragging;
	POINT floatingWindowDragOffset;
	HWND floatingWindowHandle;
	std::mutex floatingWindowMutex;
	cv::Mat floatingWindowFrame;
	bool showTrackIdInFloatingWindow;

	// 检测框平滑
	struct SmoothedDetection {
		float x, y, width, height;
		bool initialized;

		SmoothedDetection() : x(0), y(0), width(0), height(0), initialized(false) {}

		void update(float newX, float newY, float newW, float newH, float alpha) {
			if (!initialized) {
				x = newX; y = newY; width = newW; height = newH;
				initialized = true;
			} else {
				x = x + alpha * (newX - x);
				y = y + alpha * (newY - y);
				width = width + alpha * (newW - width);
				height = height + alpha * (newH - height);
			}
		}
	};
	bool detectionSmoothingEnabled;
	float detectionSmoothingAlpha;
	std::map<int, SmoothedDetection> smoothedDetections;
	std::mutex smoothedDetectionsMutex;

		// PID调试数据
	static const int PID_HISTORY_SIZE = 200;  // 保存最近200帧的PID数据
	struct PidDataPoint {
		// === 基础数据（原有） ===
		float errorX;
		float errorY;
		float outputX;
		float outputY;
		float targetX;
		float targetY;
		float targetVelocityX;  // 目标X速度（像素/帧）
		float targetVelocityY;  // 目标Y速度（像素/帧）
		float currentKp;        // 当前使用的Kp
		float currentKi;        // 当前使用的Ki
		float currentKd;        // 当前使用的Kd

		// === 新增：P/I/D 分项输出 ===
		float pTermX = 0;
		float pTermY = 0;
		float iTermX = 0;
		float iTermY = 0;
		float dTermX = 0;
		float dTermY = 0;

		// === 新增：积分状态 ===
		float integralAbsX = 0;
		float integralAbsY = 0;
		float integralRatioX = 0;
		float integralRatioY = 0;

		// === 新增：控制诊断 ===
		int controlMode = 0;    // 0=IDLE 1=TRACKING 2=LOCKED 3=I_SATURATION 4=OSCILLATING 5=PREDICTING
		bool isFiring = false;

		std::chrono::steady_clock::time_point timestamp;
	};
	std::deque<PidDataPoint> pidHistory;
	std::mutex pidHistoryMutex;
	bool showPidDebugWindow;
	HWND pidDebugWindowHandle;
	int pidDebugWindowWidth;
	int pidDebugWindowHeight;
	int pidDebugWindowX;
	int pidDebugWindowY;
	bool pidDebugWindowDragging;
	POINT pidDebugWindowDragOffset;
	std::mutex pidDebugWindowMutex;
	cv::Mat pidDebugWindowFrame;

	static const int MAX_CONFIGS = 5;

    // 全局标准PID参数（独立于各配置）
    int algorithmTypeGlobal;  // 0=高级PID, 1=标准PID
    float stdKpGlobal;
    float stdKiGlobal;
    float stdKdGlobal;
    float stdOutputLimitGlobal;
    float stdDeadZoneGlobal;
    float stdIntegralLimitGlobal;
    
    // 动态FOV参数
    float dynamicFovShrinkPercent;      // 缩放百分比 (0.1-1.0)
    float dynamicFovTransitionTime;     // 过渡时间（毫秒）
    float currentFovRadius;             // 当前实际FOV半径
    std::chrono::steady_clock::time_point fovTransitionStartTime;
    bool isFovTransitioning;
    float fovTransitionStartRadius;
    float fovTransitionEndRadius;
	float stdIntegralDeadzoneGlobal;
	float stdIntegralThresholdGlobal;
	float stdIntegralRateGlobal;
	float stdDerivativeFilterAlphaGlobal;

	struct MouseControlConfig {
		bool enabled;
		int hotkey;
		float pMin;
		float pMax;
		float pSlope;
		float d;
		float i;
		float aimSmoothingX;
		float aimSmoothingY;
		float maxPixelMove;
		float deadZonePixels;
		int screenOffsetX;
		int screenOffsetY;
		int screenWidth;
		int screenHeight;
		float derivativeFilterAlpha;
		float targetYOffset;
		float advTargetThreshold;
		float advMinCoefficient;
		float advMaxCoefficient;
		float advTransitionSharpness;
		float advTransitionMidpoint;
		float advOutputSmoothing;
		float advSpeedFactor;
		bool useOneEuroFilter;
		float oneEuroMinCutoff;
		float oneEuroBeta;
		float oneEuroDCutoff;
		int controllerType;
		std::string makcuPort;
		int makcuBaudRate;
		bool enableYAxisUnlock;
		int yAxisUnlockDelay;
		bool enableAutoTrigger;
		int triggerRadius;
		int triggerCooldown;
		int triggerFireDelay;
		int triggerFireDuration;
		int triggerInterval;
		bool enableTriggerDelayRandom;
		int triggerDelayRandomMin;
		int triggerDelayRandomMax;
		bool enableTriggerDurationRandom;
		int triggerDurationRandomMin;
		int triggerDurationRandomMax;
		int triggerMoveCompensation;
		// 新功能参数
		float integralLimit;
		float integralSeparationThreshold;
		float integralDeadZone;
		float integralRate;
		float pGainRampInitialScale;
		float pGainRampDuration;
		// DerivativePredictor参数
		bool useDerivativePredictor;
		float predictionWeightX;
		float predictionWeightY;
		float velocitySmoothFactor;
		float accelerationSmoothFactor;
		float maxPredictionTime;
		// 持续自瞄和自动压枪参数
		bool continuousAimEnabled;
		bool autoRecoilControlEnabled;
		float recoilStrength;
		int recoilSpeed;
		float recoilPidGainScale;  // 压枪时Y轴PID增益系数
		// 算法选择
		int algorithmType;  // 0=高级PID, 1=标准PID
		// 标准PID参数
		float stdKp;
		float stdKi;
		float stdKd;
		float stdOutputLimit;
		float stdDeadZone;
		float stdIntegralLimit;
		float stdIntegralDeadzone;
		float stdIntegralThreshold;
		float stdIntegralRate;
		// 贝塞尔曲线移动参数
		bool enableBezierMovement;
		float bezierCurvature;
		float bezierRandomness;

		MouseControlConfig() {
			enabled = false;
			hotkey = VK_XBUTTON1;
			pMin = 0.153f;
			pMax = 0.6f;
			pSlope = 1.0f;
			d = 0.007f;
			i = 0.01f;
			aimSmoothingX = 0.7f;
			aimSmoothingY = 0.5f;
			maxPixelMove = 128.0f;
			deadZonePixels = 5.0f;
			screenOffsetX = 0;
			screenOffsetY = 0;
			screenWidth = 0;
			screenHeight = 0;
			derivativeFilterAlpha = 0.2f;
			targetYOffset = 0.0f;
			useOneEuroFilter = false;
			oneEuroMinCutoff = 1.0f;
			oneEuroBeta = 0.0f;
			oneEuroDCutoff = 1.0f;
			controllerType = 0;
			makcuPort = "COM5";
			makcuBaudRate = 4000000;
			enableYAxisUnlock = false;
			yAxisUnlockDelay = 500;
			enableAutoTrigger = false;
			triggerRadius = 5;
			triggerCooldown = 200;
			triggerFireDelay = 0;
			triggerFireDuration = 50;
			triggerInterval = 50;
			enableTriggerDelayRandom = false;
			triggerDelayRandomMin = 0;
			triggerDelayRandomMax = 0;
			enableTriggerDurationRandom = false;
			triggerDurationRandomMin = 0;
			triggerDurationRandomMax = 0;
			triggerMoveCompensation = 0;
			// 新功能参数默认值
			integralLimit = 100.0f;
			integralSeparationThreshold = 50.0f;
			integralDeadZone = 5.0f;
			integralRate = 0.015f;
			pGainRampInitialScale = 0.6f;
			pGainRampDuration = 0.5f;
			predictionWeightX = 0.3f;
		predictionWeightY = 0.1f;
			useDerivativePredictor = true;
			maxPredictionTime = 0.1f;
			// 持续自瞄和自动压枪默认值
			continuousAimEnabled = false;
			autoRecoilControlEnabled = false;
			recoilStrength = 5.0f;
			recoilSpeed = 16;
			recoilPidGainScale = 0.3f;  // 压枪时Y轴PID增益系数默认30%
			// 算法选择默认值
			algorithmType = 0;  // 默认使用高级PID
			// 标准PID参数默认值
			stdKp = 0.3f;
			stdKi = 0.01f;
			stdKd = 0.005f;
			stdOutputLimit = 10.0f;
			stdDeadZone = 0.3f;
			stdIntegralLimit = 100.0f;
			stdIntegralDeadzone = 1.0f;
			stdIntegralThreshold = 50.0f;
			stdIntegralRate = 0.015f;
			// 贝塞尔曲线移动参数默认值
			enableBezierMovement = false;
			bezierCurvature = 0.3f;
			bezierRandomness = 0.2f;
		}
	};

	int targetSwitchDelayMs = 500;
	float targetSwitchTolerance = 0.15f;

	std::array<MouseControlConfig, MAX_CONFIGS> mouseConfigs;
	int currentConfigIndex;
	std::unique_ptr<MouseControllerInterface> mouseController;

	std::string configName;
	std::string configList;

	// ChrisPID参数
	float chrisKp;
	float chrisKi;
	float chrisKd;
	float chrisPredWeightX;
	float chrisPredWeightY;
	float chrisInitScale;
	float chrisRampTime;
	float chrisOutputMax;
	float chrisIMax;
	float chrisDFilterAlpha;

	// DynamicPID参数
	float dynamicKp;
	float dynamicKi;
	float dynamicKd;
	float dynamicTargetThreshold;
	float dynamicSpeedMultiplier;
	float dynamicMinCoefficient;
	float dynamicMaxCoefficient;
	float dynamicTransitionSharpness;
	float dynamicTransitionMidpoint;
	int   dynamicMinDataPoints;
	float dynamicErrorTolerance;
	float dynamicSmoothingFactor;
	
#endif

	~yolo_detector_filter() {
		obs_log(LOG_INFO, "YOLO detector filter destructor called");
#ifdef _WIN32
		if (cachedD3D11Texture) {
			cachedD3D11Texture->Release();
			cachedD3D11Texture = nullptr;
		}
#endif
	}
};

void inferenceThreadWorker(yolo_detector_filter *filter);
static void renderDetectionBoxes(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static bool toggleInference(obs_properties_t *props, obs_property_t *property, void *data);
static bool refreshStats(obs_properties_t *props, obs_property_t *property, void *data);
static bool testMAKCUConnection(obs_properties_t *props, obs_property_t *property, void *data);
#ifdef _WIN32
static bool saveConfigCallback(obs_properties_t *props, obs_property_t *property, void *data);
static bool loadConfigCallback(obs_properties_t *props, obs_property_t *property, void *data);
#endif


#ifdef _WIN32
static LRESULT CALLBACK FloatingWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
static void createFloatingWindow(yolo_detector_filter *filter);
static void destroyFloatingWindow(yolo_detector_filter *filter);
static void updateFloatingWindowFrame(yolo_detector_filter *filter, const cv::Mat &frame);
static void renderFloatingWindow(yolo_detector_filter *filter);
static void setupPidDataCallback(yolo_detector_filter *filter);
static void createPidDebugWindow(yolo_detector_filter *filter);
static void destroyPidDebugWindow(yolo_detector_filter *filter);
static void updatePidDebugWindow(yolo_detector_filter *filter);
#endif

const char *yolo_detector_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("YOLODetector");
}

static bool onPageChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool onConfigChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static void setConfigPropertiesVisible(obs_properties_t *props, int configIndex, bool visible);
static void setBezierMovementPropertiesVisible(obs_properties_t *props, int configIndex, bool visible);
static void setPredictorPropertiesVisible(obs_properties_t *props, int configIndex, bool visible);

obs_properties_t *yolo_detector_filter_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *toggleBtn = obs_properties_add_button(props, "toggle_inference", obs_module_text("ToggleInference"), toggleInference);
	obs_properties_add_text(props, "inference_status", obs_module_text("InferenceStatus"), OBS_TEXT_INFO);

	obs_property_t *pageList = obs_properties_add_list(props, "settings_page", "设置页面", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(pageList, "模型与检测", 0);
	obs_property_list_add_int(pageList, "视觉与区域", 1);
	obs_property_list_add_int(pageList, "鼠标控制 - 基础", 2);
	obs_property_list_add_int(pageList, "鼠标控制 - PID参数", 3);
	obs_property_list_add_int(pageList, "鼠标控制 - 扳机", 4);
	obs_property_list_add_int(pageList, "追踪与高级", 5);
	obs_property_list_add_int(pageList, "预测与滤波", 6);
	obs_property_set_modified_callback(pageList, onPageChanged);

	obs_properties_add_group(props, "model_group", obs_module_text("ModelConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *modelPathProp = obs_properties_add_path(props, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE, "ONNX Models (*.onnx)", nullptr);
	obs_property_set_long_description(modelPathProp, "选择YOLO ONNX模型文件路径");
	obs_property_t *modelVersion = obs_properties_add_list(props, "model_version", obs_module_text("ModelVersion"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(modelVersion, "YOLOv5", static_cast<int>(ModelYOLO::Version::YOLOv5));
	obs_property_list_add_int(modelVersion, "YOLOv8", static_cast<int>(ModelYOLO::Version::YOLOv8));
	obs_property_list_add_int(modelVersion, "YOLOv11", static_cast<int>(ModelYOLO::Version::YOLOv11));
	obs_property_set_long_description(modelVersion, "选择YOLO模型版本（V5/V8/V11等）");
	obs_property_t *useGPUList = obs_properties_add_list(props, "use_gpu", obs_module_text("UseGPU"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(useGPUList, "CPU", USEGPU_CPU);
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
	obs_property_list_add_string(useGPUList, "CUDA", USEGPU_CUDA);
#endif
#ifdef HAVE_ONNXRUNTIME_ROCM_EP
	obs_property_list_add_string(useGPUList, "ROCm", USEGPU_ROCM);
#endif
#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
	obs_property_list_add_string(useGPUList, "TensorRT", USEGPU_TENSORRT);
#endif
#ifdef HAVE_ONNXRUNTIME_COREML_EP
	obs_property_list_add_string(useGPUList, "CoreML", USEGPU_COREML);
#endif
#ifdef HAVE_ONNXRUNTIME_DML_EP
	obs_property_list_add_string(useGPUList, "DirectML", USEGPU_DML);
#endif
	obs_property_set_long_description(useGPUList, "选择推理设备（CUDA/GPU/DirectML/CPU）");
	
#ifdef _WIN32
	obs_property_t *useGpuTextureProp = obs_properties_add_bool(props, "use_gpu_texture_inference", "启用GPU纹理推理(实验性)");
	obs_property_set_long_description(useGpuTextureProp, "直接在GPU上处理纹理，避免GPU-CPU数据传输（支持CUDA/DML设备）");
#endif
	
	obs_property_t *resolutionList = obs_properties_add_list(props, "input_resolution", obs_module_text("InputResolution"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(resolutionList, "320x320", 320);
	obs_property_list_add_int(resolutionList, "416x416", 416);
	obs_property_list_add_int(resolutionList, "512x512", 512);
	obs_property_list_add_int(resolutionList, "640x640", 640);
	obs_property_set_long_description(resolutionList, "模型输入分辨率，影响精度和速度");
	obs_property_t *numThreadsProp = obs_properties_add_int_slider(props, "num_threads", obs_module_text("NumThreads"), 1, 16, 1);
	obs_property_set_long_description(numThreadsProp, "CPU推理线程数，建议设置为物理核心数");

	obs_properties_add_group(props, "detection_group", obs_module_text("DetectionConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *confThresholdProp = obs_properties_add_float_slider(props, "confidence_threshold", obs_module_text("ConfidenceThreshold"), 0.01, 1.0, 0.01);
	obs_property_set_long_description(confThresholdProp, "检测置信度阈值，低于此值的检测结果将被过滤");
	obs_property_t *nmsThresholdProp = obs_properties_add_float_slider(props, "nms_threshold", obs_module_text("NMSThreshold"), 0.01, 1.0, 0.01);
	obs_property_set_long_description(nmsThresholdProp, "NMS非极大值抑制阈值，用于去除重叠框");
	obs_property_t *targetClass = obs_properties_add_list(props, "target_class", obs_module_text("TargetClass"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(targetClass, obs_module_text("AllClasses"), -1);
	obs_property_set_long_description(targetClass, "要检测的目标类别");
	obs_property_t *targetClassesTextProp = obs_properties_add_text(props, "target_classes_text", "目标类别(多个用逗号分隔)", OBS_TEXT_DEFAULT);
	obs_property_set_long_description(targetClassesTextProp, "指定多个目标类别，用逗号分隔（如：0,1,2）");
	obs_property_t *inferenceIntervalProp = obs_properties_add_int_slider(props, "inference_interval_frames", obs_module_text("InferenceIntervalFrames"), 0, 10, 1);
	obs_property_set_long_description(inferenceIntervalProp, "每隔多少帧进行一次推理，0表示每帧都推理");

	obs_properties_add_group(props, "render_group", obs_module_text("RenderConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *showDetectionResultsProp = obs_properties_add_bool(props, "show_detection_results", obs_module_text("ShowDetectionResults"));
	obs_property_set_long_description(showDetectionResultsProp, "显示检测结果（边界框、类别标签、置信度）");
	obs_property_t *bboxLineWidthProp = obs_properties_add_int_slider(props, "bbox_line_width", obs_module_text("LineWidth"), 1, 5, 1);
	obs_property_set_long_description(bboxLineWidthProp, "边界框线宽");
	obs_property_t *bboxColorProp = obs_properties_add_color(props, "bbox_color", obs_module_text("BoxColor"));
	obs_property_set_long_description(bboxColorProp, "边界框颜色");
	obs_property_t *labelFontScaleProp = obs_properties_add_float_slider(props, "label_font_scale", obs_module_text("LabelFontScale"), 0.2, 1.0, 0.05);
	obs_property_set_long_description(labelFontScaleProp, "标签字体大小");

	obs_properties_add_group(props, "region_group", obs_module_text("RegionDetection"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *useRegionProp = obs_properties_add_bool(props, "use_region", obs_module_text("UseRegionDetection"));
	obs_property_set_long_description(useRegionProp, "只在指定区域内进行检测");
	obs_property_t *regionXProp = obs_properties_add_int(props, "region_x", obs_module_text("RegionX"), 0, 3840, 1);
	obs_property_set_long_description(regionXProp, "检测区域左上角X坐标");
	obs_property_t *regionYProp = obs_properties_add_int(props, "region_y", obs_module_text("RegionY"), 0, 2160, 1);
	obs_property_set_long_description(regionYProp, "检测区域左上角Y坐标");
	obs_property_t *regionWidthProp = obs_properties_add_int(props, "region_width", obs_module_text("RegionWidth"), 1, 3840, 1);
	obs_property_set_long_description(regionWidthProp, "检测区域宽度");
	obs_property_t *regionHeightProp = obs_properties_add_int(props, "region_height", obs_module_text("RegionHeight"), 1, 2160, 1);
	obs_property_set_long_description(regionHeightProp, "检测区域高度");

	obs_properties_add_group(props, "advanced_group", obs_module_text("AdvancedConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *exportCoordinatesProp = obs_properties_add_bool(props, "export_coordinates", obs_module_text("ExportCoordinates"));
	obs_property_set_long_description(exportCoordinatesProp, "将检测结果坐标导出到JSON文件");
	obs_property_t *coordinateOutputPathProp = obs_properties_add_path(props, "coordinate_output_path", obs_module_text("CoordinateOutputPath"), OBS_PATH_FILE_SAVE, "JSON Files (*.json)", nullptr);
	obs_property_set_long_description(coordinateOutputPathProp, "坐标输出文件路径");

	obs_properties_add_group(props, "fov_group", obs_module_text("FOVSettings"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *showFovProp = obs_properties_add_bool(props, "show_fov", obs_module_text("ShowFOV"));
	obs_property_set_long_description(showFovProp, "是否显示瞄准FOV区域");
	obs_property_t *fovRadiusProp = obs_properties_add_int_slider(props, "fov_radius", obs_module_text("FOVRadius"), 1, 500, 1);
	obs_property_set_long_description(fovRadiusProp, "FOV半径（像素）");
	obs_property_t *showFovCircleProp = obs_properties_add_bool(props, "show_fov_circle", obs_module_text("ShowFOVCircle"));
	obs_property_set_long_description(showFovCircleProp, "是否显示FOV圆圈");
	obs_property_t *showFovCrossProp = obs_properties_add_bool(props, "show_fov_cross", obs_module_text("ShowFOVCross"));
	obs_property_set_long_description(showFovCrossProp, "是否显示FOV十字线");
	obs_property_t *fovCrossLineScaleProp = obs_properties_add_int_slider(props, "fov_cross_line_scale", obs_module_text("FOVCrossLineScale"), 1, 300, 5);
	obs_property_set_long_description(fovCrossLineScaleProp, "FOV十字线长度");
	obs_property_t *fovCrossLineThicknessProp = obs_properties_add_int_slider(props, "fov_cross_line_thickness", obs_module_text("FOVCrossLineThickness"), 1, 10, 1);
	obs_property_set_long_description(fovCrossLineThicknessProp, "FOV十字线粗细");
	obs_property_t *fovCircleThicknessProp = obs_properties_add_int_slider(props, "fov_circle_thickness", obs_module_text("FOVCircleThickness"), 1, 10, 1);
	obs_property_set_long_description(fovCircleThicknessProp, "FOV圆圈粗细");
	obs_property_t *fovColorProp = obs_properties_add_color(props, "fov_color", obs_module_text("FOVColor"));
	obs_property_set_long_description(fovColorProp, "FOV颜色");

	obs_properties_add_group(props, "fov2_group", "动态FOV设置", OBS_GROUP_NORMAL, nullptr);
	obs_property_t *useDynamicFovProp = obs_properties_add_bool(props, "use_dynamic_fov", "启用动态FOV");
	obs_property_set_long_description(useDynamicFovProp, "启用动态FOV，根据目标距离自动调整");
	obs_property_t *showFov2Prop = obs_properties_add_bool(props, "show_fov2", "显示第二个FOV");
	obs_property_set_long_description(showFov2Prop, "是否显示第二个FOV区域");
	obs_property_t *fovRadius2Prop = obs_properties_add_int_slider(props, "fov_radius2", "第二个FOV半径", 1, 200, 1);
	obs_property_set_long_description(fovRadius2Prop, "第二个FOV半径（像素）");
	obs_property_t *fovColor2Prop = obs_properties_add_color(props, "fov_color2", "第二个FOV颜色");
    obs_property_set_long_description(fovColor2Prop, "第二个FOV颜色");

    // 动态FOV参数
    obs_property_t *dynamicFovShrinkPercentProp = obs_properties_add_int_slider(props, "dynamic_fov_shrink_percent", "动态FOV缩放百分比", 10, 100, 1);
    obs_property_set_long_description(dynamicFovShrinkPercentProp, "动态FOV缩放到原FOV的百分比（例如：50表示缩小到50%)");
    
    obs_property_t *dynamicFovTransitionTimeProp = obs_properties_add_int_slider(props, "dynamic_fov_transition_time", "动态FOV过渡时间", 0, 1000, 10);
    obs_property_set_long_description(dynamicFovTransitionTimeProp, "动态FOV过渡动画时间（毫秒）， 0表示立即切换，100表示线性过渡， 100-500表示缓动过渡");

	obs_property_t *detectionSmoothingEnabledProp = obs_properties_add_bool(props, "detection_smoothing_enabled", "启用检测框平滑");
	obs_property_set_long_description(detectionSmoothingEnabledProp, "启用检测框平滑，减少检测框抖动");
	obs_property_t *detectionSmoothingAlphaProp = obs_properties_add_float_slider(props, "detection_smoothing_alpha", "平滑系数", 0.01, 1.0, 0.01);
	 obs_property_set_long_description(detectionSmoothingAlphaProp, "平滑系数，值越小越平滑但延迟越高，值越大响应越快但平滑效果减弱");

#ifdef _WIN32
	obs_property_t *configSelectList = obs_properties_add_list(props, "mouse_config_select", "配置选择", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(configSelectList, "配置1", 0);
	obs_property_list_add_int(configSelectList, "配置2", 1);
	obs_property_list_add_int(configSelectList, "配置3", 2);
	obs_property_list_add_int(configSelectList, "配置4", 3);
	obs_property_list_add_int(configSelectList, "配置5", 4);
	obs_property_set_modified_callback(configSelectList, onConfigChanged);

	for (int i = 0; i < 5; i++) {
		char propName[64];

		snprintf(propName, sizeof(propName), "enable_config_%d", i);
		obs_property_t *enableConfigProp = obs_properties_add_bool(props, propName, "启用此配置");
		obs_property_set_long_description(enableConfigProp, "启用当前鼠标控制配置");

		snprintf(propName, sizeof(propName), "continuous_aim_%d", i);
		obs_property_t *continuousAimProp = obs_properties_add_bool(props, propName, "启用持续自瞄");
		obs_property_set_long_description(continuousAimProp, "启用后无需按住热键，自动持续瞄准目标");

		snprintf(propName, sizeof(propName), "hotkey_%d", i);
		obs_property_t *hotkeyList = obs_properties_add_list(props, propName, "热键", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(hotkeyList, "鼠标左键", VK_LBUTTON);
		obs_property_list_add_int(hotkeyList, "鼠标右键", VK_RBUTTON);
		obs_property_list_add_int(hotkeyList, "侧键1", VK_XBUTTON1);
		obs_property_list_add_int(hotkeyList, "侧键2", VK_XBUTTON2);
		obs_property_list_add_int(hotkeyList, "空格", VK_SPACE);
		obs_property_list_add_int(hotkeyList, "Shift", VK_SHIFT);
		obs_property_list_add_int(hotkeyList, "Control", VK_CONTROL);
		obs_property_list_add_int(hotkeyList, "A", 'A');
		obs_property_list_add_int(hotkeyList, "D", 'D');
		obs_property_list_add_int(hotkeyList, "W", 'W');
		obs_property_list_add_int(hotkeyList, "S", 'S');
		obs_property_list_add_int(hotkeyList, "F1", VK_F1);
		obs_property_list_add_int(hotkeyList, "F2", VK_F2);
		obs_property_set_long_description(hotkeyList, "激活此配置的热键");

		snprintf(propName, sizeof(propName), "controller_type_%d", i);
		obs_property_t *controllerTypeList = obs_properties_add_list(props, propName, "控制方式", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(controllerTypeList, "Windows API", 0);
		obs_property_list_add_int(controllerTypeList, "MAKCU", 1);
		obs_property_set_long_description(controllerTypeList, "鼠标控制方式：WindowsAPI使用系统API，MAKCU使用串口设备");

		snprintf(propName, sizeof(propName), "makcu_port_%d", i);
		obs_property_t *makcuPortProp = obs_properties_add_text(props, propName, "MAKCU 端口", OBS_TEXT_DEFAULT);
		obs_property_set_long_description(makcuPortProp, "MAKCU串口端口号（如COM5）");

		snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
		obs_property_t *baudRateList = obs_properties_add_list(props, propName, "波特率", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(baudRateList, "9600", 9600);
		obs_property_list_add_int(baudRateList, "19200", 19200);
		obs_property_list_add_int(baudRateList, "38400", 38400);
		obs_property_list_add_int(baudRateList, "57600", 57600);
		obs_property_list_add_int(baudRateList, "115200", 115200);
		obs_property_list_add_int(baudRateList, "4000000 (4Mbps)", 4000000);
		obs_property_set_long_description(baudRateList, "MAKCU串口波特率");

		snprintf(propName, sizeof(propName), "p_min_%d", i);
		obs_property_t *pMinProp = obs_properties_add_float_slider(props, propName, "P最小值", 0.00, 1.00, 0.01);
		obs_property_set_long_description(pMinProp, "最小比例增益，远距离时使用");
		snprintf(propName, sizeof(propName), "p_max_%d", i);
		obs_property_t *pMaxProp = obs_properties_add_float_slider(props, propName, "P最大值", 0.00, 1.00, 0.01);
		obs_property_set_long_description(pMaxProp, "最大比例增益，近距离时使用");
		snprintf(propName, sizeof(propName), "p_slope_%d", i);
		obs_property_t *pSlopeProp = obs_properties_add_float_slider(props, propName, "P增长斜率", 0.00, 10, 0.01);
		obs_property_set_long_description(pSlopeProp, "距离-增益曲线斜率，控制P值随距离变化的敏感度");
		snprintf(propName, sizeof(propName), "d_%d", i);
		obs_property_t *dProp = obs_properties_add_float_slider(props, propName, "微分系数", 0.0, 2.0, 0.01);
		obs_property_set_long_description(dProp, "微分增益，控制对误差变化率的响应。值越大响应越快但容易抖动，值越小越平滑但锁定感弱。注意：改回原来的计算方式后，数值范围已调整");
		snprintf(propName, sizeof(propName), "i_%d", i);
		obs_property_t *iProp = obs_properties_add_float_slider(props, propName, "积分系数", 0.0, 0.1, 0.001);
		obs_property_set_long_description(iProp, "积分增益，用于消除稳态误差。值越大消除误差越快但容易超调，值越小越稳定但可能有残留误差");
		snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
		obs_property_t *derivFilterProp = obs_properties_add_float_slider(props, propName, "微分滤波系数", 0.01, 1.00, 0.01);
		obs_property_set_long_description(derivFilterProp, "微分滤波系数，用于平滑D项，减少抖动");

		// 高级PID增强参数（状态机+动态阈值）
		snprintf(propName, sizeof(propName), "adv_target_threshold_%d", i);
		obs_property_t *advThreshProp = obs_properties_add_float_slider(props, propName, "达标误差阈值", 1.0, 30.0, 0.5);
		obs_property_set_long_description(advThreshProp, "误差小于此值时视为达标，激活全速PID");
		snprintf(propName, sizeof(propName), "adv_min_coefficient_%d", i);
		obs_property_t *advMinCoeffProp = obs_properties_add_float_slider(props, propName, "动态最小系数", 0.5, 3.0, 0.1);
		obs_property_set_long_description(advMinCoeffProp, "目标小时动态阈值最小系数");
		snprintf(propName, sizeof(propName), "adv_max_coefficient_%d", i);
		obs_property_t *advMaxCoeffProp = obs_properties_add_float_slider(props, propName, "动态最大系数", 1.0, 5.0, 0.1);
		obs_property_set_long_description(advMaxCoeffProp, "目标大时动态阈值最大系数");
		snprintf(propName, sizeof(propName), "adv_transition_sharpness_%d", i);
		obs_property_t *advSharpProp = obs_properties_add_float_slider(props, propName, "过渡锐度", 1.0, 15.0, 0.5);
		obs_property_set_long_description(advSharpProp, "Sigmoid过渡锐度，越大过渡越陡峭");
		snprintf(propName, sizeof(propName), "adv_transition_midpoint_%d", i);
		obs_property_t *advMidProp = obs_properties_add_float_slider(props, propName, "过渡中点", 0.0, 1.0, 0.05);
		obs_property_set_long_description(advMidProp, "Sigmoid过渡中点");
		snprintf(propName, sizeof(propName), "adv_output_smoothing_%d", i);
		obs_property_t *advSmoothProp = obs_properties_add_float_slider(props, propName, "输出平滑系数", 0.0, 1.0, 0.05);
		obs_property_set_long_description(advSmoothProp, "PID输出整体平滑系数，1.0=不平滑");
		snprintf(propName, sizeof(propName), "adv_speed_factor_%d", i);
		obs_property_t *advSpeedProp = obs_properties_add_float_slider(props, propName, "未达标速度因子", 0.1, 1.0, 0.05);
		obs_property_set_long_description(advSpeedProp, "未达标时P/I的速度因子，0.5=半速");

		// 一欧元滤波器（高级PID可选输出平滑）
		snprintf(propName, sizeof(propName), "use_one_euro_filter_%d", i);
		obs_properties_add_bool(props, propName, "使用一欧元滤波器");
		snprintf(propName, sizeof(propName), "one_euro_min_cutoff_%d", i);
		obs_property_t *oneEuroMinProp = obs_properties_add_float_slider(props, propName, "最小截止频率", 0.1, 10.0, 0.1);
		obs_property_set_long_description(oneEuroMinProp, "目标静止时的截止频率，越小越平滑");
		snprintf(propName, sizeof(propName), "one_euro_beta_%d", i);
		obs_property_t *oneEuroBetaProp = obs_properties_add_float_slider(props, propName, "速度因子", 0.0, 1.0, 0.01);
		obs_property_set_long_description(oneEuroBetaProp, "速度自适应系数，越大响应越快，0=固定EMA");
		snprintf(propName, sizeof(propName), "one_euro_d_cutoff_%d", i);
		obs_property_t *oneEuroDProp = obs_properties_add_float_slider(props, propName, "速度截止频率", 0.1, 10.0, 0.1);
		obs_property_set_long_description(oneEuroDProp, "速度信号的截止频率，越大越灵敏");

		snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", i);
		obs_property_t *aimSmoothXProp = obs_properties_add_float_slider(props, propName, "X轴平滑度", 0.00, 1.0, 0.01);
		obs_property_set_long_description(aimSmoothXProp, "X轴鼠标移动平滑系数，值越大越平滑但延迟越高");
		snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", i);
		obs_property_t *aimSmoothYProp = obs_properties_add_float_slider(props, propName, "Y轴平滑度", 0.00, 1.0, 0.01);
		obs_property_set_long_description(aimSmoothYProp, "Y轴鼠标移动平滑系数，值越大越平滑但延迟越高");
		snprintf(propName, sizeof(propName), "target_y_offset_%d", i);
		obs_property_t *targetYOffsetProp = obs_properties_add_float_slider(props, propName, "Y轴目标偏移(%)", -50.0, 50.0, 1.0);
		obs_property_set_long_description(targetYOffsetProp, "目标Y轴偏移量（相对于目标框高度的百分比），正值向上偏移，负值向下偏移");
		snprintf(propName, sizeof(propName), "max_pixel_move_%d", i);
		obs_property_t *maxPixelMoveProp = obs_properties_add_float_slider(props, propName, "最大移动量", 0.0, 200.0, 1.0);
		obs_property_set_long_description(maxPixelMoveProp, "单次最大移动像素数，限制最大移动速度");
		snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", i);
		obs_property_t *deadZoneProp = obs_properties_add_float_slider(props, propName, "瞄准死区", 0.0, 20.0, 0.5);
		obs_property_set_long_description(deadZoneProp, "瞄准死区，误差小于此值时不移动鼠标");

		snprintf(propName, sizeof(propName), "screen_offset_x_%d", i);
		obs_property_t *screenOffsetXProp = obs_properties_add_int(props, propName, "屏幕偏移X", 0, 3840, 1);
		obs_property_set_long_description(screenOffsetXProp, "屏幕中心X轴偏移量，用于校准瞄准点");
		snprintf(propName, sizeof(propName), "screen_offset_y_%d", i);
		obs_property_t *screenOffsetYProp = obs_properties_add_int(props, propName, "屏幕偏移Y", 0, 2160, 1);
		obs_property_set_long_description(screenOffsetYProp, "屏幕中心Y轴偏移量，用于校准瞄准点");
		snprintf(propName, sizeof(propName), "screen_width_%d", i);
		obs_property_t *screenWidthProp = obs_properties_add_int(props, propName, "屏幕宽度", 0, 3840, 1);
		obs_property_set_long_description(screenWidthProp, "屏幕分辨率宽度，0表示自动检测");
		snprintf(propName, sizeof(propName), "screen_height_%d", i);
		obs_property_t *screenHeightProp = obs_properties_add_int(props, propName, "屏幕高度", 0, 2160, 1);
		obs_property_set_long_description(screenHeightProp, "屏幕分辨率高度，0表示自动检测");

		snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", i);
		obs_property_t *enableYUnlockProp = obs_properties_add_bool(props, propName, "启用长按解锁Y轴");
		obs_property_set_long_description(enableYUnlockProp, "长按热键一段时间后临时解锁Y轴移动");
		snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", i);
		obs_property_t *yUnlockDelayProp = obs_properties_add_int_slider(props, propName, "Y 轴解锁延迟 (ms)", 100, 2000, 50);
		obs_property_set_long_description(yUnlockDelayProp, "Y轴解锁延迟时间（毫秒）");

		// 自动扳机分组（可折叠）
		snprintf(propName, sizeof(propName), "auto_trigger_group_%d", i);
		obs_properties_t *autoTriggerProps = obs_properties_create();
		snprintf(propName, sizeof(propName), "trigger_radius_%d", i);
		obs_property_t *triggerRadiusProp = obs_properties_add_int_slider(autoTriggerProps, propName, "扳机触发半径(像素)", 1, 50, 1);
		obs_property_set_long_description(triggerRadiusProp, "自动扳机触发半径（像素）");
		snprintf(propName, sizeof(propName), "trigger_cooldown_%d", i);
		obs_property_t *triggerCooldownProp = obs_properties_add_int_slider(autoTriggerProps, propName, "扳机冷却时间(ms)", 50, 1000, 50);
		obs_property_set_long_description(triggerCooldownProp, "两次自动点击之间的最小间隔（毫秒）");
		snprintf(propName, sizeof(propName), "trigger_fire_delay_%d", i);
		obs_property_t *triggerFireDelayProp = obs_properties_add_int_slider(autoTriggerProps, propName, "开火延时(ms)", 0, 500, 10);
		obs_property_set_long_description(triggerFireDelayProp, "检测到目标后延迟多久开火（毫秒）");
		snprintf(propName, sizeof(propName), "trigger_fire_duration_%d", i);
		obs_property_t *triggerFireDurationProp = obs_properties_add_int_slider(autoTriggerProps, propName, "开火时长(ms)", 10, 500, 10);
		obs_property_set_long_description(triggerFireDurationProp, "鼠标按下持续时间（毫秒）");
		snprintf(propName, sizeof(propName), "trigger_interval_%d", i);
		obs_property_t *triggerIntervalProp = obs_properties_add_int_slider(autoTriggerProps, propName, "间隔设置(ms)", 10, 500, 10);
		obs_property_set_long_description(triggerIntervalProp, "自动扳机触发间隔（毫秒）");
		snprintf(propName, sizeof(propName), "enable_trigger_delay_random_%d", i);
		obs_property_t *enableTriggerDelayRandomProp = obs_properties_add_bool(autoTriggerProps, propName, "启用随机延时");
		obs_property_set_long_description(enableTriggerDelayRandomProp, "启用随机开火延时，增加不可预测性");
		snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
		obs_property_t *triggerDelayRandomMinProp = obs_properties_add_int_slider(autoTriggerProps, propName, "随机延时下限(ms)", 0, 200, 5);
		obs_property_set_long_description(triggerDelayRandomMinProp, "随机开火延时的下限");
		snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
		obs_property_t *triggerDelayRandomMaxProp = obs_properties_add_int_slider(autoTriggerProps, propName, "随机延时上限(ms)", 0, 200, 5);
		obs_property_set_long_description(triggerDelayRandomMaxProp, "随机开火延时的上限");
		snprintf(propName, sizeof(propName), "enable_trigger_duration_random_%d", i);
		obs_property_t *enableTriggerDurationRandomProp = obs_properties_add_bool(autoTriggerProps, propName, "启用随机时长");
		obs_property_set_long_description(enableTriggerDurationRandomProp, "启用随机开火时长");
		snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
		obs_property_t *triggerDurationRandomMinProp = obs_properties_add_int_slider(autoTriggerProps, propName, "随机时长下限(ms)", 0, 200, 5);
		obs_property_set_long_description(triggerDurationRandomMinProp, "随机开火时长的下限");
		snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
		obs_property_t *triggerDurationRandomMaxProp = obs_properties_add_int_slider(autoTriggerProps, propName, "随机时长上限(ms)", 0, 200, 5);
		obs_property_set_long_description(triggerDurationRandomMaxProp, "随机开火时长的上限");
		snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
		obs_property_t *triggerMoveCompensationProp = obs_properties_add_int_slider(autoTriggerProps, propName, "移动补偿(像素)", 0, 100, 1);
		obs_property_set_long_description(triggerMoveCompensationProp, "移动补偿像素数，用于补偿鼠标移动时的延迟");
		snprintf(propName, sizeof(propName), "auto_trigger_group_%d", i);
		obs_properties_add_group(props, propName, "自动扳机", OBS_GROUP_CHECKABLE, autoTriggerProps);

		// 新功能参数设置
		snprintf(propName, sizeof(propName), "integral_limit_%d", i);
		obs_property_t *integralLimitProp = obs_properties_add_float_slider(props, propName, "积分限幅", 0.0, 500.0, 1.0);
		obs_property_set_long_description(integralLimitProp, "积分项上限，防止积分饱和");
		snprintf(propName, sizeof(propName), "integral_separation_threshold_%d", i);
		obs_property_t *integralSepThresholdProp = obs_properties_add_float_slider(props, propName, "积分分离阈值", 0.0, 200.0, 1.0);
		obs_property_set_long_description(integralSepThresholdProp, "积分分离阈值，误差超过此值时暂停积分");
		snprintf(propName, sizeof(propName), "integral_dead_zone_%d", i);
		obs_property_t *integralDeadZoneProp = obs_properties_add_float_slider(props, propName, "积分死区", 0.0, 50.0, 0.1);
		obs_property_set_long_description(integralDeadZoneProp, "积分死区，误差小于此值时不进行积分");
		snprintf(propName, sizeof(propName), "integral_rate_%d", i);
		obs_property_t *integralRateProp = obs_properties_add_float_slider(props, propName, "积分增益率", 0.0, 0.1, 0.001);
		obs_property_set_long_description(integralRateProp, "积分增益的变化速率，值越大积分增益增加越快");
		snprintf(propName, sizeof(propName), "p_gain_ramp_initial_scale_%d", i);
		obs_property_t *pGainRampInitialProp = obs_properties_add_float_slider(props, propName, "P-Gain Ramp初始比例", 0.0, 1.0, 0.1);
		obs_property_set_long_description(pGainRampInitialProp, "P-Gain Ramp初始比例，热键按下初期使用较低的P值");
		snprintf(propName, sizeof(propName), "p_gain_ramp_duration_%d", i);
		obs_property_t *pGainRampDurationProp = obs_properties_add_float_slider(props, propName, "P-Gain Ramp持续时间(秒)", 0.0, 2.0, 0.1);
		obs_property_set_long_description(pGainRampDurationProp, "P-Gain Ramp持续时间，从初始比例过渡到100%的时间");

		// 后坐力控制分组（可折叠）
		snprintf(propName, sizeof(propName), "recoil_group_%d", i);
		obs_properties_t *recoilProps = obs_properties_create();
		snprintf(propName, sizeof(propName), "recoil_strength_%d", i);
		obs_property_t *recoilStrengthProp = obs_properties_add_float_slider(recoilProps, propName, "压枪强度(像素)", 0.0, 50.0, 1.0);
		obs_property_set_long_description(recoilStrengthProp, "每次压枪移动的像素数，值越大压枪幅度越大");
		snprintf(propName, sizeof(propName), "recoil_speed_%d", i);
		obs_property_t *recoilSpeedProp = obs_properties_add_int_slider(recoilProps, propName, "压枪速度(ms)", 1, 100, 1);
		obs_property_set_long_description(recoilSpeedProp, "压枪移动的时间间隔（毫秒），值越小压枪频率越高");
		snprintf(propName, sizeof(propName), "recoil_pid_gain_scale_%d", i);
		obs_property_t *recoilPidGainScaleProp = obs_properties_add_float_slider(recoilProps, propName, "压枪时Y轴PID增益", 0.0, 1.0, 0.05);
		obs_property_set_long_description(recoilPidGainScaleProp, "压枪时Y轴PID控制的增益系数，0表示完全禁用Y轴PID，1表示保持原增益");
		snprintf(propName, sizeof(propName), "recoil_group_%d", i);
		obs_properties_add_group(props, propName, "后坐力控制", OBS_GROUP_CHECKABLE, recoilProps);
	}

	// 预测器配置组
	for (int i = 0; i < 5; i++) {
		char propName[64];
		
		// 导数预测器分组（可折叠）
		snprintf(propName, sizeof(propName), "derivative_predictor_group_%d", i);
		obs_properties_t *derivPredProps = obs_properties_create();
		
		snprintf(propName, sizeof(propName), "prediction_weight_x_%d", i);
		obs_property_t *predictionWeightXProp = obs_properties_add_float_slider(derivPredProps, propName, "导数预测权重X", 0.0f, 1.0f, 0.1f);
		obs_property_set_long_description(predictionWeightXProp, "导数预测器在X轴的融合权重，值越大预测效果越强");
		
		snprintf(propName, sizeof(propName), "prediction_weight_y_%d", i);
		obs_property_t *predictionWeightYProp = obs_properties_add_float_slider(derivPredProps, propName, "导数预测权重Y", 0.0f, 1.0f, 0.1f);
		obs_property_set_long_description(predictionWeightYProp, "导数预测器在Y轴的融合权重，值越大预测效果越强");
		
		snprintf(propName, sizeof(propName), "max_prediction_time_%d", i);
		obs_property_t *maxPredTimeProp = obs_properties_add_float_slider(derivPredProps, propName, "最大预测时间(秒)", 0.01f, 0.3f, 0.01f);
		obs_property_set_long_description(maxPredTimeProp, "预测的最大时间范围，值越大预测越远但误差越大，建议0.05-0.15");
		
		snprintf(propName, sizeof(propName), "derivative_predictor_group_%d", i);
		obs_properties_add_group(props, propName, "导数预测器", OBS_GROUP_CHECKABLE, derivPredProps);
	}

	// 贝塞尔曲线移动分组
	for (int i = 0; i < 5; i++) {
		char propName[64];
		
		// 贝塞尔曲线移动分组（可折叠）
		snprintf(propName, sizeof(propName), "bezier_movement_group_%d", i);
		obs_properties_t *bezierProps = obs_properties_create();
		
		snprintf(propName, sizeof(propName), "bezier_curvature_%d", i);
		obs_property_t *bezierCurvatureProp = obs_properties_add_float_slider(bezierProps, propName, "曲线弯曲程度", 0.0f, 1.0f, 0.05f);
		obs_property_set_long_description(bezierCurvatureProp, "贝塞尔曲线的弯曲程度，值越大曲线越弯曲");
		
		snprintf(propName, sizeof(propName), "bezier_randomness_%d", i);
		obs_property_t *bezierRandomnessProp = obs_properties_add_float_slider(bezierProps, propName, "随机程度", 0.0f, 0.5f, 0.05f);
		obs_property_set_long_description(bezierRandomnessProp, "曲线的随机程度，值越大每次移动的轨迹越不固定");
		
		snprintf(propName, sizeof(propName), "bezier_movement_group_%d", i);
		obs_properties_add_group(props, propName, "贝塞尔曲线移动", OBS_GROUP_CHECKABLE, bezierProps);
		
	}

	obs_properties_add_button(props, "test_makcu_connection", "测试MAKCU连接", testMAKCUConnection);

	obs_properties_add_group(props, "tracking_group", "目标追踪设置", OBS_GROUP_NORMAL, nullptr);
	obs_property_t *iouThresholdProp = obs_properties_add_float_slider(props, "iou_threshold", "IoU阈值", 0.1, 0.9, 0.05);
	obs_property_set_long_description(iouThresholdProp, "目标追踪的IoU阈值，用于判断是否是同一目标");
	obs_property_t *maxLostFramesProp = obs_properties_add_int_slider(props, "max_lost_frames", "最大丢失帧数", 0, 30, 1);
	obs_property_set_long_description(maxLostFramesProp, "目标丢失多少帧后放弃追踪");
	obs_property_t *targetSwitchDelayProp = obs_properties_add_int_slider(props, "target_switch_delay", "转火延迟(ms)", 0, 1500, 50);
	obs_property_set_long_description(targetSwitchDelayProp, "切换目标前的延迟时间（毫秒）");
	obs_property_t *targetSwitchToleranceProp = obs_properties_add_float_slider(props, "target_switch_tolerance", "切换容差", 0.0, 0.5, 0.05);
	obs_property_set_long_description(targetSwitchToleranceProp, "切换目标的容差，防止频繁切换");
	
	// 多指标融合追踪权重
	obs_property_t *trackingWeightIouProp = obs_properties_add_float_slider(props, "tracking_weight_iou", "IoU权重", 0.0, 1.0, 0.05);
	obs_property_set_long_description(trackingWeightIouProp, "IoU距离在目标匹配中的权重，值越大越重视检测框重叠度");
	obs_property_t *trackingWeightCenterProp = obs_properties_add_float_slider(props, "tracking_weight_center", "中心点权重", 0.0, 1.0, 0.05);
	obs_property_set_long_description(trackingWeightCenterProp, "中心点距离在目标匹配中的权重，值越大越重视目标位置");
	obs_property_t *trackingWeightAspectProp = obs_properties_add_float_slider(props, "tracking_weight_aspect", "宽高比权重", 0.0, 1.0, 0.05);
	obs_property_set_long_description(trackingWeightAspectProp, "宽高比距离在目标匹配中的权重，值越大越重视目标形状");
	obs_property_t *trackingWeightAreaProp = obs_properties_add_float_slider(props, "tracking_weight_area", "面积权重", 0.0, 1.0, 0.05);
	obs_property_set_long_description(trackingWeightAreaProp, "面积距离在目标匹配中的权重，值越大越重视目标大小");
	
	// 重识别设置
	obs_property_t *maxReidentifyFramesProp = obs_properties_add_int_slider(props, "max_reidentify_frames", "重识别帧数", 0, 60, 5);
	obs_property_set_long_description(maxReidentifyFramesProp, "目标丢失后保留重识别的最大帧数，超过则完全放弃");
	obs_property_t *reidentifyCenterThresholdProp = obs_properties_add_float_slider(props, "reidentify_center_threshold", "重识别距离阈值", 0.01, 0.3, 0.01);
	obs_property_set_long_description(reidentifyCenterThresholdProp, "重识别时中心点距离阈值，距离小于此值认为是同一目标");
	
	obs_properties_add_group(props, "floating_window_group", obs_module_text("FloatingWindow"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *showFloatingWindowProp = obs_properties_add_bool(props, "show_floating_window", obs_module_text("ShowFloatingWindow"));
	obs_property_set_long_description(showFloatingWindowProp, "显示独立的预览窗口");
	obs_property_t *floatingWindowWidthProp = obs_properties_add_int_slider(props, "floating_window_width", obs_module_text("WindowWidth"), 320, 1920, 10);
	obs_property_set_long_description(floatingWindowWidthProp, "浮动窗口的宽度");
	obs_property_t *floatingWindowHeightProp = obs_properties_add_int_slider(props, "floating_window_height", obs_module_text("WindowHeight"), 240, 1080, 10);
	obs_property_set_long_description(floatingWindowHeightProp, "浮动窗口的高度");
	obs_property_t *showPidDebugWindowProp = obs_properties_add_bool(props, "show_pid_debug_window", "显示PID调试曲线");
	obs_property_set_long_description(showPidDebugWindowProp, "在浮动窗口中显示PID调试曲线，方便调整参数");
	obs_property_t *showTrackIdProp = obs_properties_add_bool(props, "show_track_id_in_floating_window", "显示目标ID");
	obs_property_set_long_description(showTrackIdProp, "在浮动窗口中显示目标追踪ID");

	obs_properties_add_group(props, "config_management_group", "配置管理", OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_button(props, "save_config", "保存配置", saveConfigCallback);
	obs_properties_add_button(props, "load_config", "加载配置", loadConfigCallback);
	
#endif

	obs_properties_add_text(props, "avg_inference_time", obs_module_text("AvgInferenceTime"), OBS_TEXT_INFO);
	obs_properties_add_text(props, "detected_objects", obs_module_text("DetectedObjects"), OBS_TEXT_INFO);

	// 页面6: PID参数 - 算法选择放在最上面
	// 算法选择（全局）
	obs_property_t *algorithmTypeList = obs_properties_add_list(props, "algorithm_type_global", "控制算法", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(algorithmTypeList, "高级PID (自适应)", 0);
	obs_property_list_add_int(algorithmTypeList, "标准PID (经典)", 1);
	obs_property_list_add_int(algorithmTypeList, "ChrisPID (克里斯控制器)", 2);
	obs_property_list_add_int(algorithmTypeList, "动态PID (动态阈值)", 3);
	obs_property_set_long_description(algorithmTypeList, "选择控制算法：高级PID包含自适应P增益、预测等功能；标准PID是经典PID控制；ChrisPID是克里斯控制器；动态PID基于动态阈值和状态机");
	obs_property_set_modified_callback(algorithmTypeList, onPageChanged);
	
	// 标准PID配置分组
	obs_properties_add_group(props, "std_pid_group", "标准PID配置", OBS_GROUP_NORMAL, nullptr);
	
	// 标准PID参数
	obs_property_t *stdKpProp = obs_properties_add_float_slider(props, "std_kp_global", "标准PID-Kp", 0.0, 1.0, 0.01);
	obs_property_set_long_description(stdKpProp, "标准PID的比例系数");
	obs_property_t *stdKiProp = obs_properties_add_float_slider(props, "std_ki_global", "标准PID-Ki", 0.0, 1.0, 0.001);
	obs_property_set_long_description(stdKiProp, "标准PID的积分系数");
	obs_property_t *stdKdProp = obs_properties_add_float_slider(props, "std_kd_global", "标准PID-Kd", 0.0, 1.0, 0.001);
	obs_property_set_long_description(stdKdProp, "标准PID的微分系数");
	obs_property_t *stdOutputLimitProp = obs_properties_add_float_slider(props, "std_output_limit_global", "输出限幅", 1.0, 200.0, 1.0);
	obs_property_set_long_description(stdOutputLimitProp, "PID输出的最大值限制");
	obs_property_t *stdDeadZoneProp = obs_properties_add_float_slider(props, "std_dead_zone_global", "死区", 0.0, 10.0, 0.1);
	obs_property_set_long_description(stdDeadZoneProp, "误差小于此值时不输出");
	obs_property_t *stdIntegralLimitProp = obs_properties_add_float_slider(props, "std_integral_limit_global", "积分限幅", 0.0, 500.0, 10.0);
	obs_property_set_long_description(stdIntegralLimitProp, "积分项的最大值限制");
	obs_property_t *stdIntegralDeadzoneProp = obs_properties_add_float_slider(props, "std_integral_deadzone_global", "积分死区", 0.0, 50.0, 0.5);
	obs_property_set_long_description(stdIntegralDeadzoneProp, "积分小于此值时不计入积分");
	obs_property_t *stdIntegralThresholdProp = obs_properties_add_float_slider(props, "std_integral_threshold_global", "积分分离阈值", 0.0, 200.0, 1.0);
	obs_property_set_long_description(stdIntegralThresholdProp, "误差超过此值时暂停积分");
	obs_property_t *stdIntegralRateProp = obs_properties_add_float_slider(props, "std_integral_rate_global", "积分增益率", 0.0, 0.1, 0.001);
	obs_property_set_long_description(stdIntegralRateProp, "积分增益的变化速率");
	obs_property_t *stdDerivativeFilterProp = obs_properties_add_float_slider(props, "std_derivative_filter_alpha_global", "微分滤波系数", 0.01, 1.0, 0.01);
	obs_property_set_long_description(stdDerivativeFilterProp, "标准PID微分项低通滤波系数，值越大响应越快，值越小越平滑但延迟");

	// ChrisPID参数
	obs_properties_add_group(props, "chris_pid_group", "ChrisPID配置", OBS_GROUP_NORMAL, nullptr);
	obs_property_t *chrisKpProp = obs_properties_add_float_slider(props, "chris_kp", "ChrisPID-Kp", 0.0, 2.0, 0.01);
	obs_property_set_long_description(chrisKpProp, "ChrisPID比例系数");
	obs_property_t *chrisKiProp = obs_properties_add_float_slider(props, "chris_ki", "ChrisPID-Ki", 0.0, 2.0, 0.01);
	obs_property_set_long_description(chrisKiProp, "ChrisPID积分系数");
	obs_property_t *chrisKdProp = obs_properties_add_float_slider(props, "chris_kd", "ChrisPID-Kd", 0.0, 0.1, 0.001);
	obs_property_set_long_description(chrisKdProp, "ChrisPID微分系数");
	obs_property_t *chrisPredWeightXProp = obs_properties_add_float_slider(props, "chris_pred_weight_x", "X轴预测权重", 0.0, 2.0, 0.01);
	obs_property_set_long_description(chrisPredWeightXProp, "ChrisPID X轴预测权重");
	obs_property_t *chrisPredWeightYProp = obs_properties_add_float_slider(props, "chris_pred_weight_y", "Y轴预测权重", 0.0, 2.0, 0.01);
	obs_property_set_long_description(chrisPredWeightYProp, "ChrisPID Y轴预测权重");
	obs_property_t *chrisInitScaleProp = obs_properties_add_float_slider(props, "chris_init_scale", "P增益初始缩放", 0.0, 1.0, 0.01);
	obs_property_set_long_description(chrisInitScaleProp, "ChrisPID P增益初始缩放比例，用于P-Gain Ramp");
	obs_property_t *chrisRampTimeProp = obs_properties_add_float_slider(props, "chris_ramp_time", "P增益爬坡时间", 0.0, 2.0, 0.01);
	obs_property_set_long_description(chrisRampTimeProp, "ChrisPID P增益从初始值爬升到1.0的时间（秒）");
	obs_property_t *chrisOutputMaxProp = obs_properties_add_float_slider(props, "chris_output_max", "输出最大值", 0.0, 500.0, 1.0);
	obs_property_set_long_description(chrisOutputMaxProp, "ChrisPID输出最大值限制");
	obs_property_t *chrisIMaxProp = obs_properties_add_float_slider(props, "chris_i_max", "积分限幅", 0.0, 500.0, 1.0);
	obs_property_set_long_description(chrisIMaxProp, "ChrisPID积分项限幅，防止积分饱和");
	obs_property_t *chrisDFilterAlphaProp = obs_properties_add_float_slider(props, "chris_d_filter_alpha", "D项滤波系数", 0.1, 1.0, 0.05);
	obs_property_set_long_description(chrisDFilterAlphaProp, "ChrisPID D项滤波系数，1.0=无滤波，0.1=强滤波");

	obs_properties_add_group(props, "dynamic_pid_group", "动态PID配置", OBS_GROUP_NORMAL, nullptr);
	obs_property_t *dynamicKpProp = obs_properties_add_float_slider(props, "dynamic_kp", "动态PID-Kp", 0.0, 2.0, 0.01);
	obs_property_set_long_description(dynamicKpProp, "动态PID比例系数");
	obs_property_t *dynamicKiProp = obs_properties_add_float_slider(props, "dynamic_ki", "动态PID-Ki", 0.0, 2.0, 0.01);
	obs_property_set_long_description(dynamicKiProp, "动态PID积分系数");
	obs_property_t *dynamicKdProp = obs_properties_add_float_slider(props, "dynamic_kd", "动态PID-Kd", 0.0, 0.5, 0.001);
	obs_property_set_long_description(dynamicKdProp, "动态PID微分系数");
	obs_property_t *dynamicThresholdProp = obs_properties_add_float_slider(props, "dynamic_target_threshold", "达标误差阈值", 0.0, 20.0, 0.1);
	obs_property_set_long_description(dynamicThresholdProp, "误差小于此值时视为达标，激活完整PID");
	obs_property_t *dynamicSpeedProp = obs_properties_add_float_slider(props, "dynamic_speed_multiplier", "速度倍率", 0.0, 5.0, 0.1);
	obs_property_set_long_description(dynamicSpeedProp, "速度计算倍率系数");
	obs_property_t *dynamicMinCoeffProp = obs_properties_add_float_slider(props, "dynamic_min_coefficient", "最小系数", 0.0, 5.0, 0.1);
	obs_property_set_long_description(dynamicMinCoeffProp, "动态阈值计算最小系数（目标小时）");
	obs_property_t *dynamicMaxCoeffProp = obs_properties_add_float_slider(props, "dynamic_max_coefficient", "最大系数", 0.0, 5.0, 0.1);
	obs_property_set_long_description(dynamicMaxCoeffProp, "动态阈值计算最大系数（目标大时）");
	obs_property_t *dynamicSharpnessProp = obs_properties_add_float_slider(props, "dynamic_transition_sharpness", "过渡锐度", 0.0, 20.0, 0.5);
	obs_property_set_long_description(dynamicSharpnessProp, "Sigmoid过渡锐度，越大过渡越陡峭");
	obs_property_t *dynamicMidpointProp = obs_properties_add_float_slider(props, "dynamic_transition_midpoint", "过渡中点", 0.0, 1.0, 0.01);
	obs_property_set_long_description(dynamicMidpointProp, "动态阈值过渡中点");
	obs_property_t *dynamicMinDataProp = obs_properties_add_int_slider(props, "dynamic_min_data_points", "最小数据量", 1, 10, 1);
	obs_property_set_long_description(dynamicMinDataProp, "稳定计数达到此值时激活积分");
	obs_property_t *dynamicToleranceProp = obs_properties_add_float_slider(props, "dynamic_error_tolerance", "误差变化容限", 0.0, 20.0, 0.1);
	obs_property_set_long_description(dynamicToleranceProp, "误差变化小于此值时认为稳定");
	obs_property_t *dynamicSmoothingProp = obs_properties_add_float_slider(props, "dynamic_smoothing_factor", "输出平滑因子", 0.0, 1.0, 0.05);
	obs_property_set_long_description(dynamicSmoothingProp, "输出EMA平滑系数，1.0=不平滑");

	UNUSED_PARAMETER(data);
	return props;
}

// 设置鼠标控制-基础页面的控件可见性
static void setMouseBasicPropertiesVisible(obs_properties_t *props, int configIndex, bool visible)
{
	char propName[64];

	snprintf(propName, sizeof(propName), "enable_config_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "continuous_aim_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "hotkey_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "controller_type_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "makcu_port_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "max_pixel_move_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "screen_offset_x_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "screen_offset_y_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "screen_width_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "screen_height_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	// Y轴偏移与解锁（从页面3迁移）
	snprintf(propName, sizeof(propName), "target_y_offset_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
}

// 设置鼠标控制-PID页面的控件可见性
static void setMousePIDPropertiesVisible(obs_properties_t *props, int configIndex, bool visible)
{
	char propName[64];

	snprintf(propName, sizeof(propName), "p_min_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "p_max_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "p_slope_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "p_gain_ramp_initial_scale_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "p_gain_ramp_duration_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "integral_limit_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "integral_separation_threshold_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "integral_dead_zone_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "integral_rate_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "d_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "i_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adv_target_threshold_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adv_min_coefficient_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adv_max_coefficient_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adv_transition_sharpness_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adv_transition_midpoint_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adv_output_smoothing_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adv_speed_factor_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "use_one_euro_filter_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "one_euro_min_cutoff_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "one_euro_beta_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "one_euro_d_cutoff_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	// 以下参数已迁移到其他页面：
	// target_y_offset, enable_y_axis_unlock, y_axis_unlock_delay -> 页面2（基础）
	// prediction_weight_x, prediction_weight_y -> 页面6（预测与滤波）
	// auto_recoil, recoil_strength, recoil_speed, recoil_pid_gain_scale -> 页面4（扳机）
}

// 设置贝塞尔曲线移动页面的控件可见性
static void setBezierMovementPropertiesVisible(obs_properties_t *props, int configIndex, bool visible)
{
	char propName[64];
	// 贝塞尔曲线移动分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "bezier_movement_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
}

// 设置预测器配置页面的控件可见性
static void setPredictorPropertiesVisible(obs_properties_t *props, int configIndex, bool visible)
{
	char propName[64];
	// 导数预测器分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "derivative_predictor_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
}

// 设置鼠标控制-扳机页面的控件可见性
static void setMouseTriggerPropertiesVisible(obs_properties_t *props, int configIndex, bool visible)
{
	char propName[64];

	// 自动扳机分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "auto_trigger_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	// 后坐力控制分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "recoil_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
}

static bool onConfigChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	int currentConfig = (int)obs_data_get_int(settings, "mouse_config_select");
	int page = (int)obs_data_get_int(settings, "settings_page");
	int algorithm = (int)obs_data_get_int(settings, "algorithm_type_global");

	for (int i = 0; i < 5; i++) {
		bool isCurrentConfig = (i == currentConfig);
		setMouseBasicPropertiesVisible(props, i, isCurrentConfig && page == 2);
		// 高级PID参数只在algorithm == 0时显示
		setMousePIDPropertiesVisible(props, i, isCurrentConfig && page == 3 && algorithm == 0);
		setMouseTriggerPropertiesVisible(props, i, isCurrentConfig && page == 4);
		setPredictorPropertiesVisible(props, i, isCurrentConfig && page == 6);
		setBezierMovementPropertiesVisible(props, i, isCurrentConfig && page == 7);
	}

	obs_property_set_visible(obs_properties_get(props, "mouse_config_select"), page == 2 || page == 3 || page == 4 || page == 6 || page == 7);
	obs_property_set_visible(obs_properties_get(props, "test_makcu_connection"), page == 2);

	return true;
}

static bool onPageChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	int page = (int)obs_data_get_int(settings, "settings_page");

	// 页面0: 模型与检测 - 显示模型组和检测组
	obs_property_set_visible(obs_properties_get(props, "model_group"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "detection_group"), page == 0);

	// 页面0: 模型与检测参数
	obs_property_set_visible(obs_properties_get(props, "model_path"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "model_version"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "use_gpu"), page == 0);
#ifdef _WIN32
	obs_property_set_visible(obs_properties_get(props, "use_gpu_texture_inference"), page == 0);
#endif
	obs_property_set_visible(obs_properties_get(props, "input_resolution"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "num_threads"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "confidence_threshold"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "nms_threshold"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "target_class"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "target_classes_text"), page == 0);
	obs_property_set_visible(obs_properties_get(props, "inference_interval_frames"), page == 0);

	// 页面1: 视觉与区域 - 显示渲染组、区域组和FOV组
	obs_property_set_visible(obs_properties_get(props, "render_group"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "region_group"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_group"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov2_group"), page == 1);

	// 页面1: 视觉与区域参数
	obs_property_set_visible(obs_properties_get(props, "show_detection_results"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "bbox_line_width"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "bbox_color"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "label_font_scale"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "use_region"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "region_x"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "region_y"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "region_width"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "region_height"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "show_fov"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_radius"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "show_fov_circle"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "show_fov_cross"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_cross_line_scale"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_cross_line_thickness"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_circle_thickness"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_color"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "use_dynamic_fov"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "show_fov2"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_radius2"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "fov_color2"), page == 1);
	
	// 动态FOV参数只在FOV设置页面显示
	obs_property_set_visible(obs_properties_get(props, "dynamic_fov_shrink_percent"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_fov_transition_time"), page == 1);
	
	// 检测框平滑参数只在视觉与区域页面显示
	obs_property_set_visible(obs_properties_get(props, "detection_smoothing_enabled"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "detection_smoothing_alpha"), page == 1);

#ifdef _WIN32
	// 配置选择器在鼠标控制页面(2,3,4)和预测与滤波页面(6)显示
	obs_property_set_visible(obs_properties_get(props, "mouse_config_select"), page == 2 || page == 3 || page == 4 || page == 6);

	// 根据当前页面和配置设置鼠标控制参数可见性
	int currentConfig = (int)obs_data_get_int(settings, "mouse_config_select");
	int algorithm = (int)obs_data_get_int(settings, "algorithm_type_global");
	for (int i = 0; i < 5; i++) {
		bool isCurrentConfig = (i == currentConfig);
		setMouseBasicPropertiesVisible(props, i, isCurrentConfig && page == 2);
		// 高级PID参数只在algorithm == 0时显示
		setMousePIDPropertiesVisible(props, i, isCurrentConfig && page == 3 && algorithm == 0);
		setMouseTriggerPropertiesVisible(props, i, isCurrentConfig && page == 4);
		setPredictorPropertiesVisible(props, i, isCurrentConfig && page == 6);
		setBezierMovementPropertiesVisible(props, i, isCurrentConfig && page == 6);
	}

	// 测试连接按钮只在基础页面显示
	obs_property_set_visible(obs_properties_get(props, "test_makcu_connection"), page == 2);

	// 页面5: 追踪与高级
	obs_property_set_visible(obs_properties_get(props, "tracking_group"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "iou_threshold"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "max_lost_frames"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "target_switch_delay"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "target_switch_tolerance"), page == 5);
	
	// 多指标融合追踪权重（页面5）
	obs_property_set_visible(obs_properties_get(props, "tracking_weight_iou"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "tracking_weight_center"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "tracking_weight_aspect"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "tracking_weight_area"), page == 5);
	
	// 重识别设置（页面5）
	obs_property_set_visible(obs_properties_get(props, "max_reidentify_frames"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "reidentify_center_threshold"), page == 5);
	
	obs_property_set_visible(obs_properties_get(props, "floating_window_group"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "show_floating_window"), page == 5);
    obs_property_set_visible(obs_properties_get(props, "floating_window_width"), page == 5);
    obs_property_set_visible(obs_properties_get(props, "floating_window_height"), page == 5);
    obs_property_set_visible(obs_properties_get(props, "show_pid_debug_window"), page == 5);
    obs_property_set_visible(obs_properties_get(props, "config_management_group"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "save_config"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "load_config"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "advanced_group"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "export_coordinates"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "coordinate_output_path"), page == 5);

	// 页面3: 鼠标控制 - PID参数（整合所有控制算法）
	// 算法选择（在页面3始终显示）
	obs_property_set_visible(obs_properties_get(props, "algorithm_type_global"), page == 3);
	
	// 标准PID参数组（选择1时显示）
	obs_property_set_visible(obs_properties_get(props, "std_pid_group"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_kp_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_ki_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_kd_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_output_limit_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_dead_zone_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_integral_limit_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_integral_deadzone_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_integral_threshold_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_integral_rate_global"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "std_derivative_filter_alpha_global"), page == 3 && algorithm == 1);

	// ChrisPID参数组（选择2时显示）
	obs_property_set_visible(obs_properties_get(props, "chris_pid_group"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_kp"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_ki"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_kd"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_pred_weight_x"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_pred_weight_y"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_init_scale"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_ramp_time"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_output_max"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_i_max"), page == 3 && algorithm == 2);
	obs_property_set_visible(obs_properties_get(props, "chris_d_filter_alpha"), page == 3 && algorithm == 2);

	// 动态PID参数组（选择3时显示）
	obs_property_set_visible(obs_properties_get(props, "dynamic_pid_group"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kp"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_ki"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kd"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_target_threshold"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_speed_multiplier"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_min_coefficient"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_max_coefficient"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_transition_sharpness"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_transition_midpoint"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_min_data_points"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_error_tolerance"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_smoothing_factor"), page == 3 && algorithm == 3);

	// 页面6: 预测与滤波（整合预测器、贝塞尔）
	obs_property_set_visible(obs_properties_get(props, "predictor_group"), page == 6);
	obs_property_set_visible(obs_properties_get(props, "bezier_movement_group"), page == 6);
	
#else
	(void)page;
#endif

	return true;
}

void yolo_detector_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, "model_path", "");
	obs_data_set_default_int(settings, "model_version", static_cast<int>(ModelYOLO::Version::YOLOv8));
	obs_data_set_default_string(settings, "use_gpu", USEGPU_CPU);
#ifdef _WIN32
	obs_data_set_default_bool(settings, "use_gpu_texture_inference", false);
#endif
	obs_data_set_default_int(settings, "input_resolution", 640);
	obs_data_set_default_int(settings, "num_threads", 4);
	obs_data_set_default_double(settings, "confidence_threshold", 0.5);
	obs_data_set_default_double(settings, "nms_threshold", 0.45);
	obs_data_set_default_int(settings, "target_class", -1);
	obs_data_set_default_int(settings, "inference_interval_frames", 1);
	obs_data_set_default_bool(settings, "show_detection_results", true);
	obs_data_set_default_int(settings, "bbox_line_width", 2);
	obs_data_set_default_int(settings, "bbox_color", 0xFF00FF00);
	obs_data_set_default_bool(settings, "show_fov", false);
	obs_data_set_default_int(settings, "fov_radius", 200);
	obs_data_set_default_bool(settings, "show_fov_circle", true);
	obs_data_set_default_bool(settings, "show_fov_cross", true);
	obs_data_set_default_int(settings, "fov_cross_line_scale", 100);
	obs_data_set_default_int(settings, "fov_cross_line_thickness", 2);
	obs_data_set_default_int(settings, "fov_circle_thickness", 2);
	obs_data_set_default_int(settings, "fov_color", 0xFFFF0000);

	// 第二个FOV默认值
	obs_data_set_default_bool(settings, "use_dynamic_fov", false);
    obs_data_set_default_bool(settings, "show_fov2", true);
    obs_data_set_default_int(settings, "fov_radius2", 50);
    obs_data_set_default_int(settings, "fov_color2", 0xFF00FF00);
    
    // 动态FOV参数
    obs_data_set_default_int(settings, "dynamic_fov_shrink_percent", 50);
    obs_data_set_default_int(settings, "dynamic_fov_transition_time", 200);
    
    // 检测框平滑参数
    obs_data_set_default_bool(settings, "detection_smoothing_enabled", true);
    obs_data_set_default_double(settings, "detection_smoothing_alpha", 0.3);
    
    obs_data_set_default_double(settings, "label_font_scale", 0.35);
	obs_data_set_default_bool(settings, "use_region", false);
	obs_data_set_default_int(settings, "region_x", 0);
	obs_data_set_default_int(settings, "region_y", 0);
	obs_data_set_default_int(settings, "region_width", 640);
	obs_data_set_default_int(settings, "region_height", 480);
	obs_data_set_default_bool(settings, "export_coordinates", false);
	obs_data_set_default_string(settings, "coordinate_output_path", "");
#ifdef _WIN32
	obs_data_set_default_bool(settings, "show_floating_window", false);
	obs_data_set_default_int(settings, "floating_window_width", 640);
	obs_data_set_default_int(settings, "floating_window_height", 480);
	obs_data_set_default_bool(settings, "show_pid_debug_window", false);
	obs_data_set_default_bool(settings, "show_track_id_in_floating_window", false);
#endif

#ifdef _WIN32
	obs_data_set_default_int(settings, "mouse_config_select", 0);

	for (int i = 0; i < 5; i++) {
		char propName[64];

		snprintf(propName, sizeof(propName), "enable_config_%d", i);
		obs_data_set_default_bool(settings, propName, false);

		snprintf(propName, sizeof(propName), "hotkey_%d", i);
		obs_data_set_default_int(settings, propName, VK_XBUTTON1);

		snprintf(propName, sizeof(propName), "controller_type_%d", i);
		obs_data_set_default_int(settings, propName, 0);

		snprintf(propName, sizeof(propName), "makcu_port_%d", i);
		obs_data_set_default_string(settings, propName, "COM5");

		snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
		obs_data_set_default_int(settings, propName, 4000000);

		snprintf(propName, sizeof(propName), "p_min_%d", i);
		obs_data_set_default_double(settings, propName, 0.153);
		snprintf(propName, sizeof(propName), "p_max_%d", i);
		obs_data_set_default_double(settings, propName, 0.6);
		snprintf(propName, sizeof(propName), "p_slope_%d", i);
		obs_data_set_default_double(settings, propName, 1.0);
		snprintf(propName, sizeof(propName), "d_%d", i);
		obs_data_set_default_double(settings, propName, 0.007);
		snprintf(propName, sizeof(propName), "i_%d", i);
		obs_data_set_default_double(settings, propName, 0.01);
		snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
		obs_data_set_default_double(settings, propName, 0.2);

		snprintf(propName, sizeof(propName), "adv_target_threshold_%d", i);
		obs_data_set_default_double(settings, propName, 10.0);
		snprintf(propName, sizeof(propName), "adv_min_coefficient_%d", i);
		obs_data_set_default_double(settings, propName, 1.5);
		snprintf(propName, sizeof(propName), "adv_max_coefficient_%d", i);
		obs_data_set_default_double(settings, propName, 2.5);
		snprintf(propName, sizeof(propName), "adv_transition_sharpness_%d", i);
		obs_data_set_default_double(settings, propName, 5.0);
		snprintf(propName, sizeof(propName), "adv_transition_midpoint_%d", i);
		obs_data_set_default_double(settings, propName, 0.3);
		snprintf(propName, sizeof(propName), "adv_output_smoothing_%d", i);
		obs_data_set_default_double(settings, propName, 0.7);
		snprintf(propName, sizeof(propName), "adv_speed_factor_%d", i);
		obs_data_set_default_double(settings, propName, 0.5);

		snprintf(propName, sizeof(propName), "use_one_euro_filter_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "one_euro_min_cutoff_%d", i);
		obs_data_set_default_double(settings, propName, 1.0);
		snprintf(propName, sizeof(propName), "one_euro_beta_%d", i);
		obs_data_set_default_double(settings, propName, 0.0);
		snprintf(propName, sizeof(propName), "one_euro_d_cutoff_%d", i);
		obs_data_set_default_double(settings, propName, 1.0);

		snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", i);
		obs_data_set_default_double(settings, propName, 0.7);
		snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", i);
		obs_data_set_default_double(settings, propName, 0.5);
		snprintf(propName, sizeof(propName), "target_y_offset_%d", i);
		obs_data_set_default_double(settings, propName, 0.0);
		snprintf(propName, sizeof(propName), "max_pixel_move_%d", i);
		obs_data_set_default_double(settings, propName, 128.0);
		snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", i);
		obs_data_set_default_double(settings, propName, 5.0);

		snprintf(propName, sizeof(propName), "screen_offset_x_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "screen_offset_y_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "screen_width_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "screen_height_%d", i);
		obs_data_set_default_int(settings, propName, 0);

		snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", i);
		obs_data_set_default_int(settings, propName, 500);

		snprintf(propName, sizeof(propName), "auto_trigger_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "trigger_radius_%d", i);
		obs_data_set_default_int(settings, propName, 5);
		snprintf(propName, sizeof(propName), "trigger_cooldown_%d", i);
		obs_data_set_default_int(settings, propName, 200);
		snprintf(propName, sizeof(propName), "trigger_fire_delay_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_fire_duration_%d", i);
		obs_data_set_default_int(settings, propName, 50);
		snprintf(propName, sizeof(propName), "trigger_interval_%d", i);
		obs_data_set_default_int(settings, propName, 50);
		snprintf(propName, sizeof(propName), "enable_trigger_delay_random_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "enable_trigger_duration_random_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
		obs_data_set_default_int(settings, propName, 0);

		// 新功能参数默认值
		snprintf(propName, sizeof(propName), "integral_limit_%d", i);
		obs_data_set_default_double(settings, propName, 100.0);
		snprintf(propName, sizeof(propName), "integral_separation_threshold_%d", i);
		obs_data_set_default_double(settings, propName, 50.0);
		snprintf(propName, sizeof(propName), "integral_dead_zone_%d", i);
		obs_data_set_default_double(settings, propName, 5.0);
		snprintf(propName, sizeof(propName), "integral_rate_%d", i);
		obs_data_set_default_double(settings, propName, 0.015);
		snprintf(propName, sizeof(propName), "p_gain_ramp_initial_scale_%d", i);
		obs_data_set_default_double(settings, propName, 0.6);
		snprintf(propName, sizeof(propName), "p_gain_ramp_duration_%d", i);
		obs_data_set_default_double(settings, propName, 0.5);
		snprintf(propName, sizeof(propName), "prediction_weight_x_%d", i);
		obs_data_set_default_double(settings, propName, 0.3);
		snprintf(propName, sizeof(propName), "prediction_weight_y_%d", i);
		obs_data_set_default_double(settings, propName, 0.1);

		// 持续自瞄和后坐力控制默认值
		snprintf(propName, sizeof(propName), "continuous_aim_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "recoil_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "recoil_strength_%d", i);
		obs_data_set_default_double(settings, propName, 5.0);
		snprintf(propName, sizeof(propName), "recoil_speed_%d", i);
		obs_data_set_default_int(settings, propName, 16);
		snprintf(propName, sizeof(propName), "recoil_pid_gain_scale_%d", i);
		obs_data_set_default_double(settings, propName, 0.3);
		// DerivativePredictor参数默认值
		snprintf(propName, sizeof(propName), "derivative_predictor_group_%d", i);
		obs_data_set_default_bool(settings, propName, true);
		snprintf(propName, sizeof(propName), "prediction_weight_x_%d", i);
		obs_data_set_default_double(settings, propName, 0.5);
		snprintf(propName, sizeof(propName), "prediction_weight_y_%d", i);
		obs_data_set_default_double(settings, propName, 0.1);
		snprintf(propName, sizeof(propName), "max_prediction_time_%d", i);
		obs_data_set_default_double(settings, propName, 0.1);
		// 贝塞尔曲线移动参数默认值
		snprintf(propName, sizeof(propName), "bezier_movement_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "bezier_curvature_%d", i);
		obs_data_set_default_double(settings, propName, 0.3);
		snprintf(propName, sizeof(propName), "bezier_randomness_%d", i);
		obs_data_set_default_double(settings, propName, 0.2);
	}

    obs_data_set_default_string(settings, "config_name", "");
    obs_data_set_default_string(settings, "config_list", "");
    obs_data_set_default_double(settings, "iou_threshold", 0.3);
    obs_data_set_default_int(settings, "max_lost_frames", 10);
    obs_data_set_default_int(settings, "target_switch_delay", 500);
    obs_data_set_default_double(settings, "target_switch_tolerance", 0.15);
    
    // 多指标融合追踪权重默认值
    obs_data_set_default_double(settings, "tracking_weight_iou", 0.4);
    obs_data_set_default_double(settings, "tracking_weight_center", 0.3);
    obs_data_set_default_double(settings, "tracking_weight_aspect", 0.15);
    obs_data_set_default_double(settings, "tracking_weight_area", 0.15);
    
    // 重识别参数默认值
    obs_data_set_default_int(settings, "max_reidentify_frames", 30);
    obs_data_set_default_double(settings, "reidentify_center_threshold", 0.1);

    obs_data_set_default_int(settings, "settings_page", 0);
    
    // 全局标准PID参数默认值
    obs_data_set_default_int(settings, "algorithm_type_global", 0);  // 默认高级PID
    obs_data_set_default_double(settings, "std_kp_global", 0.3);
    obs_data_set_default_double(settings, "std_ki_global", 0.01);
    obs_data_set_default_double(settings, "std_kd_global", 0.005);
    obs_data_set_default_double(settings, "std_output_limit_global", 50.0);
    obs_data_set_default_double(settings, "std_dead_zone_global", 0.3);
    obs_data_set_default_double(settings, "std_integral_limit_global", 100.0);
    obs_data_set_default_double(settings, "std_integral_deadzone_global", 1.0);
    obs_data_set_default_double(settings, "std_integral_threshold_global", 50.0);
    obs_data_set_default_double(settings, "std_integral_rate_global", 0.015);
    obs_data_set_default_double(settings, "std_derivative_filter_alpha_global", 0.2);
    
    // ChrisPID参数默认值
    obs_data_set_default_double(settings, "chris_kp", 0.45);
    obs_data_set_default_double(settings, "chris_ki", 0.02);
    obs_data_set_default_double(settings, "chris_kd", 0.04);
    obs_data_set_default_double(settings, "chris_pred_weight_x", 0.5);
    obs_data_set_default_double(settings, "chris_pred_weight_y", 0.1);
    obs_data_set_default_double(settings, "chris_init_scale", 0.6);
    obs_data_set_default_double(settings, "chris_ramp_time", 0.5);
    obs_data_set_default_double(settings, "chris_output_max", 150.0);
    obs_data_set_default_double(settings, "chris_i_max", 100.0);
    obs_data_set_default_double(settings, "chris_d_filter_alpha", 0.3);

    // DynamicPID默认值
    obs_data_set_default_double(settings, "dynamic_kp", 0.5);
    obs_data_set_default_double(settings, "dynamic_ki", 0.1);
    obs_data_set_default_double(settings, "dynamic_kd", 0.05);
    obs_data_set_default_double(settings, "dynamic_target_threshold", 4.0);
    obs_data_set_default_double(settings, "dynamic_speed_multiplier", 1.0);
    obs_data_set_default_double(settings, "dynamic_min_coefficient", 1.6);
    obs_data_set_default_double(settings, "dynamic_max_coefficient", 2.7);
    obs_data_set_default_double(settings, "dynamic_transition_sharpness", 5.0);
    obs_data_set_default_double(settings, "dynamic_transition_midpoint", 0.0);
    obs_data_set_default_int(settings, "dynamic_min_data_points", 2);
    obs_data_set_default_double(settings, "dynamic_error_tolerance", 3.0);
    obs_data_set_default_double(settings, "dynamic_smoothing_factor", 0.8);
#endif
}

void yolo_detector_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "YOLO detector filter updated");

	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<yolo_detector_filter> tf = *ptr;
	if (!tf) {
		return;
	}

	tf->isDisabled = true;

	std::string newModelPath = obs_data_get_string(settings, "model_path");
	ModelYOLO::Version newModelVersion = static_cast<ModelYOLO::Version>(obs_data_get_int(settings, "model_version"));
	std::string newUseGPU = obs_data_get_string(settings, "use_gpu");
	uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "num_threads");
	int newInputResolution = (int)obs_data_get_int(settings, "input_resolution");
	
	bool needModelUpdate = false;
	{
		std::lock_guard<std::mutex> lock(tf->yoloModelMutex);
		needModelUpdate = (newModelPath != tf->modelPath || newModelVersion != tf->modelVersion || newUseGPU != tf->useGPU || newNumThreads != tf->numThreads || newInputResolution != tf->inputResolution || !tf->yoloModel);
	}
	
	if (needModelUpdate) {
		tf->modelPath = newModelPath;
		tf->modelVersion = newModelVersion;
		tf->useGPU = newUseGPU;
		tf->numThreads = newNumThreads;
		tf->inputResolution = newInputResolution;
		
		if (!tf->modelPath.empty()) {
			try {
				obs_log(LOG_INFO, "[YOLO Filter] Loading new model: %s", tf->modelPath.c_str());
				
				std::unique_ptr<ModelYOLO> newYoloModel = std::make_unique<ModelYOLO>(tf->modelVersion);
				
				newYoloModel->loadModel(tf->modelPath, tf->useGPU, (int)tf->numThreads, tf->inputResolution);
				
				obs_log(LOG_INFO, "[YOLO Filter] Model loaded successfully");
				
				std::lock_guard<std::mutex> lock(tf->yoloModelMutex);
				tf->yoloModel = std::move(newYoloModel);
				
			} catch (const std::exception& e) {
				obs_log(LOG_ERROR, "[YOLO Filter] Failed to load model: %s", e.what());
				std::lock_guard<std::mutex> lock(tf->yoloModelMutex);
				tf->yoloModel.reset();
			}
		} else {
			std::lock_guard<std::mutex> lock(tf->yoloModelMutex);
			tf->yoloModel.reset();
		}
	}
	
	tf->confidenceThreshold = (float)obs_data_get_double(settings, "confidence_threshold");
	
#ifdef _WIN32
	tf->useGpuTextureInference = obs_data_get_bool(settings, "use_gpu_texture_inference");
	// GPU纹理推理支持CUDA、TensorRT和DML设备
	if (tf->useGpuTextureInference && tf->useGPU != "cuda" && tf->useGPU != "tensorrt" && tf->useGPU != "dml") {
		obs_log(LOG_WARNING, "[YOLO Filter] GPU纹理推理需要CUDA、TensorRT或DML设备，已禁用");
		tf->useGpuTextureInference = false;
	}
#endif
	tf->nmsThreshold = (float)obs_data_get_double(settings, "nms_threshold");
	tf->targetClassId = (int)obs_data_get_int(settings, "target_class");
	tf->inferenceIntervalFrames = (int)obs_data_get_int(settings, "inference_interval_frames");
	
	{
		std::lock_guard<std::mutex> lock(tf->yoloModelMutex);
		if (tf->yoloModel) {
			tf->yoloModel->setConfidenceThreshold(tf->confidenceThreshold);
			tf->yoloModel->setNMSThreshold(tf->nmsThreshold);

			// 检查是否有多个目标类别设置
			std::string targetClassesText = obs_data_get_string(settings, "target_classes_text");
			if (!targetClassesText.empty()) {
				// 解析逗号分隔的类别ID
				std::vector<int> selectedClasses;
				std::stringstream ss(targetClassesText);
				std::string item;
				while (std::getline(ss, item, ',')) {
					try {
						int classId = std::stoi(item);
						selectedClasses.push_back(classId);
					} catch (...) {
						// 忽略无效的数字
					}
				}
				if (!selectedClasses.empty()) {
					tf->yoloModel->setTargetClasses(selectedClasses);
					tf->targetClasses = selectedClasses;
				} else {
					tf->yoloModel->setTargetClass(tf->targetClassId);
					tf->targetClasses.clear();
				}
			} else {
				// 使用单个目标类别
				tf->yoloModel->setTargetClass(tf->targetClassId);
				tf->targetClasses.clear();
			}
		}
	}
	
	bool showDetectionResults = obs_data_get_bool(settings, "show_detection_results");
	tf->showDetectionResults = showDetectionResults;
	tf->showBBox = showDetectionResults;
	tf->showLabel = showDetectionResults;
	tf->showConfidence = showDetectionResults;
	tf->bboxLineWidth = (int)obs_data_get_int(settings, "bbox_line_width");
	tf->bboxColor = (uint32_t)obs_data_get_int(settings, "bbox_color");
	
	tf->showFOV = obs_data_get_bool(settings, "show_fov");
	int newFovRadius = (int)obs_data_get_int(settings, "fov_radius");
	bool fovRadiusChanged = (newFovRadius != tf->fovRadius);
	tf->fovRadius = newFovRadius;
	
	// 如果FOV半径改变且不在过渡中，更新当前FOV半径
	if (fovRadiusChanged && !tf->isFovTransitioning) {
		tf->currentFovRadius = static_cast<float>(tf->fovRadius);
	}
	
	tf->showFOVCircle = obs_data_get_bool(settings, "show_fov_circle");
	tf->showFOVCross = obs_data_get_bool(settings, "show_fov_cross");
	tf->fovCrossLineScale = (int)obs_data_get_int(settings, "fov_cross_line_scale");
	tf->fovCrossLineThickness = (int)obs_data_get_int(settings, "fov_cross_line_thickness");
	tf->fovCircleThickness = (int)obs_data_get_int(settings, "fov_circle_thickness");
	tf->fovColor = (uint32_t)obs_data_get_int(settings, "fov_color");

	// 第二个FOV设置
	tf->useDynamicFOV = obs_data_get_bool(settings, "use_dynamic_fov");
	tf->showFOV2 = obs_data_get_bool(settings, "show_fov2");
	// 确保第二个FOV的半径不超过第一个FOV的半径
	int requestedFOV2 = (int)obs_data_get_int(settings, "fov_radius2");
	tf->fovRadius2 = std::min(requestedFOV2, tf->fovRadius);
	tf->fovColor2 = (uint32_t)obs_data_get_int(settings, "fov_color2");
	
	// 动态FOV参数
	tf->dynamicFovShrinkPercent = (float)obs_data_get_int(settings, "dynamic_fov_shrink_percent") / 100.0f;
	tf->dynamicFovTransitionTime = (float)obs_data_get_int(settings, "dynamic_fov_transition_time");

	// 检测框平滑参数
	tf->detectionSmoothingEnabled = obs_data_get_bool(settings, "detection_smoothing_enabled");
	tf->detectionSmoothingAlpha = (float)obs_data_get_double(settings, "detection_smoothing_alpha");

	tf->labelFontScale = (float)obs_data_get_double(settings, "label_font_scale");

	tf->useRegion = obs_data_get_bool(settings, "use_region");
	tf->regionX = (int)obs_data_get_int(settings, "region_x");
	tf->regionY = (int)obs_data_get_int(settings, "region_y");
	tf->regionWidth = (int)obs_data_get_int(settings, "region_width");
	tf->regionHeight = (int)obs_data_get_int(settings, "region_height");

	tf->exportCoordinates = obs_data_get_bool(settings, "export_coordinates");
	tf->coordinateOutputPath = obs_data_get_string(settings, "coordinate_output_path");

#ifdef _WIN32
	bool newShowFloatingWindow = obs_data_get_bool(settings, "show_floating_window");
	int newFloatingWindowWidth = (int)obs_data_get_int(settings, "floating_window_width");
	int newFloatingWindowHeight = (int)obs_data_get_int(settings, "floating_window_height");
	bool newShowPidDebugWindow = obs_data_get_bool(settings, "show_pid_debug_window");

	if (newShowFloatingWindow != tf->showFloatingWindow || 
	    newFloatingWindowWidth != tf->floatingWindowWidth || 
	    newFloatingWindowHeight != tf->floatingWindowHeight) {
		tf->showFloatingWindow = newShowFloatingWindow;
		tf->floatingWindowWidth = newFloatingWindowWidth;
		tf->floatingWindowHeight = newFloatingWindowHeight;

		if (tf->showFloatingWindow) {
			createFloatingWindow(tf.get());
		} else {
			destroyFloatingWindow(tf.get());
		}
	}

	if (newShowPidDebugWindow != tf->showPidDebugWindow) {
		tf->showPidDebugWindow = newShowPidDebugWindow;
		if (tf->showPidDebugWindow) {
			createPidDebugWindow(tf.get());
		} else {
			destroyPidDebugWindow(tf.get());
		}
	}

	tf->showTrackIdInFloatingWindow = obs_data_get_bool(settings, "show_track_id_in_floating_window");

	tf->currentConfigIndex = (int)obs_data_get_int(settings, "mouse_config_select");

	for (int i = 0; i < 5; i++) {
		char propName[64];

		snprintf(propName, sizeof(propName), "enable_config_%d", i);
		tf->mouseConfigs[i].enabled = obs_data_get_bool(settings, propName);

		snprintf(propName, sizeof(propName), "hotkey_%d", i);
		tf->mouseConfigs[i].hotkey = (int)obs_data_get_int(settings, propName);

		snprintf(propName, sizeof(propName), "controller_type_%d", i);
		tf->mouseConfigs[i].controllerType = (int)obs_data_get_int(settings, propName);

		snprintf(propName, sizeof(propName), "makcu_port_%d", i);
		tf->mouseConfigs[i].makcuPort = obs_data_get_string(settings, propName);

		snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
		tf->mouseConfigs[i].makcuBaudRate = (int)obs_data_get_int(settings, propName);

		snprintf(propName, sizeof(propName), "p_min_%d", i);
		tf->mouseConfigs[i].pMin = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "p_max_%d", i);
		tf->mouseConfigs[i].pMax = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "p_slope_%d", i);
		tf->mouseConfigs[i].pSlope = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "d_%d", i);
		tf->mouseConfigs[i].d = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "i_%d", i);
		tf->mouseConfigs[i].i = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
		tf->mouseConfigs[i].derivativeFilterAlpha = (float)obs_data_get_double(settings, propName);

		snprintf(propName, sizeof(propName), "adv_target_threshold_%d", i);
		tf->mouseConfigs[i].advTargetThreshold = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adv_min_coefficient_%d", i);
		tf->mouseConfigs[i].advMinCoefficient = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adv_max_coefficient_%d", i);
		tf->mouseConfigs[i].advMaxCoefficient = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adv_transition_sharpness_%d", i);
		tf->mouseConfigs[i].advTransitionSharpness = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adv_transition_midpoint_%d", i);
		tf->mouseConfigs[i].advTransitionMidpoint = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adv_output_smoothing_%d", i);
		tf->mouseConfigs[i].advOutputSmoothing = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adv_speed_factor_%d", i);
		tf->mouseConfigs[i].advSpeedFactor = (float)obs_data_get_double(settings, propName);

		snprintf(propName, sizeof(propName), "use_one_euro_filter_%d", i);
		tf->mouseConfigs[i].useOneEuroFilter = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "one_euro_min_cutoff_%d", i);
		tf->mouseConfigs[i].oneEuroMinCutoff = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "one_euro_beta_%d", i);
		tf->mouseConfigs[i].oneEuroBeta = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "one_euro_d_cutoff_%d", i);
		tf->mouseConfigs[i].oneEuroDCutoff = (float)obs_data_get_double(settings, propName);

		snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", i);
		tf->mouseConfigs[i].aimSmoothingX = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", i);
		tf->mouseConfigs[i].aimSmoothingY = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "target_y_offset_%d", i);
		tf->mouseConfigs[i].targetYOffset = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "max_pixel_move_%d", i);
		tf->mouseConfigs[i].maxPixelMove = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", i);
		tf->mouseConfigs[i].deadZonePixels = (float)obs_data_get_double(settings, propName);

		snprintf(propName, sizeof(propName), "screen_offset_x_%d", i);
		tf->mouseConfigs[i].screenOffsetX = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "screen_offset_y_%d", i);
		tf->mouseConfigs[i].screenOffsetY = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "screen_width_%d", i);
		tf->mouseConfigs[i].screenWidth = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "screen_height_%d", i);
		tf->mouseConfigs[i].screenHeight = (int)obs_data_get_int(settings, propName);

		snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", i);
		tf->mouseConfigs[i].enableYAxisUnlock = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", i);
		tf->mouseConfigs[i].yAxisUnlockDelay = (int)obs_data_get_int(settings, propName);

		snprintf(propName, sizeof(propName), "auto_trigger_group_%d", i);
		tf->mouseConfigs[i].enableAutoTrigger = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_radius_%d", i);
		tf->mouseConfigs[i].triggerRadius = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_cooldown_%d", i);
		tf->mouseConfigs[i].triggerCooldown = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_fire_delay_%d", i);
		tf->mouseConfigs[i].triggerFireDelay = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_fire_duration_%d", i);
		tf->mouseConfigs[i].triggerFireDuration = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_interval_%d", i);
		tf->mouseConfigs[i].triggerInterval = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "enable_trigger_delay_random_%d", i);
		tf->mouseConfigs[i].enableTriggerDelayRandom = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
		tf->mouseConfigs[i].triggerDelayRandomMin = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
		tf->mouseConfigs[i].triggerDelayRandomMax = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "enable_trigger_duration_random_%d", i);
		tf->mouseConfigs[i].enableTriggerDurationRandom = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
		tf->mouseConfigs[i].triggerDurationRandomMin = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
		tf->mouseConfigs[i].triggerDurationRandomMax = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
		tf->mouseConfigs[i].triggerMoveCompensation = (int)obs_data_get_int(settings, propName);

		// 新功能参数更新
		snprintf(propName, sizeof(propName), "integral_limit_%d", i);
		tf->mouseConfigs[i].integralLimit = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "integral_separation_threshold_%d", i);
		tf->mouseConfigs[i].integralSeparationThreshold = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "integral_dead_zone_%d", i);
		tf->mouseConfigs[i].integralDeadZone = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "integral_rate_%d", i);
		tf->mouseConfigs[i].integralRate = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "p_gain_ramp_initial_scale_%d", i);
		tf->mouseConfigs[i].pGainRampInitialScale = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "p_gain_ramp_duration_%d", i);
		tf->mouseConfigs[i].pGainRampDuration = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "prediction_weight_x_%d", i);
		tf->mouseConfigs[i].predictionWeightX = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "prediction_weight_y_%d", i);
		tf->mouseConfigs[i].predictionWeightY = (float)obs_data_get_double(settings, propName);

		// 读取持续自瞄和后坐力控制配置
		snprintf(propName, sizeof(propName), "continuous_aim_%d", i);
		tf->mouseConfigs[i].continuousAimEnabled = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "recoil_group_%d", i);
		tf->mouseConfigs[i].autoRecoilControlEnabled = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "recoil_strength_%d", i);
		tf->mouseConfigs[i].recoilStrength = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "recoil_speed_%d", i);
		tf->mouseConfigs[i].recoilSpeed = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "recoil_pid_gain_scale_%d", i);
		tf->mouseConfigs[i].recoilPidGainScale = (float)obs_data_get_double(settings, propName);
		// DerivativePredictor参数
		snprintf(propName, sizeof(propName), "derivative_predictor_group_%d", i);
		tf->mouseConfigs[i].useDerivativePredictor = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "prediction_weight_x_%d", i);
		tf->mouseConfigs[i].predictionWeightX = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "prediction_weight_y_%d", i);
		tf->mouseConfigs[i].predictionWeightY = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "max_prediction_time_%d", i);
		tf->mouseConfigs[i].maxPredictionTime = (float)obs_data_get_double(settings, propName);
		// 贝塞尔曲线移动参数
		snprintf(propName, sizeof(propName), "bezier_movement_group_%d", i);
		tf->mouseConfigs[i].enableBezierMovement = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "bezier_curvature_%d", i);
		tf->mouseConfigs[i].bezierCurvature = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "bezier_randomness_%d", i);
		tf->mouseConfigs[i].bezierRandomness = (float)obs_data_get_double(settings, propName);
	}

	tf->targetSwitchDelayMs = (int)obs_data_get_int(settings, "target_switch_delay");
	tf->targetSwitchTolerance = (float)obs_data_get_double(settings, "target_switch_tolerance");

	// 读取全局标准PID参数
	tf->algorithmTypeGlobal = (int)obs_data_get_int(settings, "algorithm_type_global");
	tf->stdKpGlobal = (float)obs_data_get_double(settings, "std_kp_global");
	tf->stdKiGlobal = (float)obs_data_get_double(settings, "std_ki_global");
	tf->stdKdGlobal = (float)obs_data_get_double(settings, "std_kd_global");
	tf->stdOutputLimitGlobal = (float)obs_data_get_double(settings, "std_output_limit_global");
	tf->stdDeadZoneGlobal = (float)obs_data_get_double(settings, "std_dead_zone_global");
	tf->stdIntegralLimitGlobal = (float)obs_data_get_double(settings, "std_integral_limit_global");
	tf->stdIntegralDeadzoneGlobal = (float)obs_data_get_double(settings, "std_integral_deadzone_global");
	tf->stdIntegralThresholdGlobal = (float)obs_data_get_double(settings, "std_integral_threshold_global");
	tf->stdIntegralRateGlobal = (float)obs_data_get_double(settings, "std_integral_rate_global");
	tf->stdDerivativeFilterAlphaGlobal = (float)obs_data_get_double(settings, "std_derivative_filter_alpha_global");

	// 读取ChrisPID参数
	tf->chrisKp = (float)obs_data_get_double(settings, "chris_kp");
	tf->chrisKi = (float)obs_data_get_double(settings, "chris_ki");
	tf->chrisKd = (float)obs_data_get_double(settings, "chris_kd");
	tf->chrisPredWeightX = (float)obs_data_get_double(settings, "chris_pred_weight_x");
	tf->chrisPredWeightY = (float)obs_data_get_double(settings, "chris_pred_weight_y");
	tf->chrisInitScale = (float)obs_data_get_double(settings, "chris_init_scale");
	tf->chrisRampTime = (float)obs_data_get_double(settings, "chris_ramp_time");
	tf->chrisOutputMax = (float)obs_data_get_double(settings, "chris_output_max");
	tf->chrisIMax = (float)obs_data_get_double(settings, "chris_i_max");
	tf->chrisDFilterAlpha = (float)obs_data_get_double(settings, "chris_d_filter_alpha");

	// DynamicPID参数
	tf->dynamicKp = (float)obs_data_get_double(settings, "dynamic_kp");
	tf->dynamicKi = (float)obs_data_get_double(settings, "dynamic_ki");
	tf->dynamicKd = (float)obs_data_get_double(settings, "dynamic_kd");
	tf->dynamicTargetThreshold = (float)obs_data_get_double(settings, "dynamic_target_threshold");
	tf->dynamicSpeedMultiplier = (float)obs_data_get_double(settings, "dynamic_speed_multiplier");
	tf->dynamicMinCoefficient = (float)obs_data_get_double(settings, "dynamic_min_coefficient");
	tf->dynamicMaxCoefficient = (float)obs_data_get_double(settings, "dynamic_max_coefficient");
	tf->dynamicTransitionSharpness = (float)obs_data_get_double(settings, "dynamic_transition_sharpness");
	tf->dynamicTransitionMidpoint = (float)obs_data_get_double(settings, "dynamic_transition_midpoint");
	tf->dynamicMinDataPoints = (int)obs_data_get_int(settings, "dynamic_min_data_points");
	tf->dynamicErrorTolerance = (float)obs_data_get_double(settings, "dynamic_error_tolerance");
	tf->dynamicSmoothingFactor = (float)obs_data_get_double(settings, "dynamic_smoothing_factor");

	bool hasEnabledConfig = false;
	for (int i = 0; i < 5; i++) {
		if (tf->mouseConfigs[i].enabled) {
			hasEnabledConfig = true;
			break;
		}
	}

	if (!tf->mouseController && hasEnabledConfig) {
		tf->mouseController = MouseControllerFactory::createController(ControllerType::WindowsAPI, "", 0);
		setupPidDataCallback(tf.get());
		obs_log(LOG_INFO, "Created mouse controller for multi-config mode");
	}

	tf->configName = obs_data_get_string(settings, "config_name");
	tf->configList = obs_data_get_string(settings, "config_list");
	tf->iouThreshold = (float)obs_data_get_double(settings, "iou_threshold");
	tf->maxLostFrames = (int)obs_data_get_int(settings, "max_lost_frames");
	
	// 多指标融合追踪权重
	tf->trackingWeightIou = (float)obs_data_get_double(settings, "tracking_weight_iou");
	tf->trackingWeightCenter = (float)obs_data_get_double(settings, "tracking_weight_center");
	tf->trackingWeightAspect = (float)obs_data_get_double(settings, "tracking_weight_aspect");
	tf->trackingWeightArea = (float)obs_data_get_double(settings, "tracking_weight_area");
	
	// 重识别参数
	tf->maxReidentifyFrames = (int)obs_data_get_int(settings, "max_reidentify_frames");
	tf->reidentifyCenterThreshold = (float)obs_data_get_double(settings, "reidentify_center_threshold");
	
#endif

	tf->isDisabled = false;
}

static bool toggleInference(obs_properties_t *props, obs_property_t *property, void *data)
{
	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return true;
	}

	std::shared_ptr<yolo_detector_filter> tf = *ptr;
	if (!tf) {
		return true;
	}

	tf->isInferencing = !tf->isInferencing;
	obs_log(LOG_INFO, "[YOLO Detector] Inference %s, isInferencing=%d", 
		tf->isInferencing ? "enabled" : "disabled",
		(int)tf->isInferencing);

	obs_property_t *statusText = obs_properties_get(props, "inference_status");
	if (statusText) {
		obs_property_set_description(statusText, tf->isInferencing ? obs_module_text("InferenceRunning") : obs_module_text("InferenceStopped"));
	}

	return true;
}

static bool refreshStats(obs_properties_t *props, obs_property_t *property, void *data)
{
	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return true;
	}

	std::shared_ptr<yolo_detector_filter> tf = *ptr;
	if (!tf) {
		return true;
	}

	// 更新平均推理时间
	obs_property_t *inferenceTimeText = obs_properties_get(props, "avg_inference_time");
	if (inferenceTimeText) {
		char timeStr[128];
		snprintf(timeStr, sizeof(timeStr), "%s: %.2f ms", obs_module_text("AvgInferenceTime"), tf->avgInferenceTimeMs);
		obs_property_set_description(inferenceTimeText, timeStr);
	}

	// 更新检测到的物体数量
	obs_property_t *detectedObjectsText = obs_properties_get(props, "detected_objects");
	if (detectedObjectsText) {
		size_t count = 0;
		{
			std::lock_guard<std::mutex> lock(tf->detectionsMutex);
			count = tf->detections.size();
		}
		char countStr[128];
		snprintf(countStr, sizeof(countStr), "%s: %zu", obs_module_text("DetectedObjects"), count);
		obs_property_set_description(detectedObjectsText, countStr);
	}

	return true;
}

static bool testMAKCUConnection(obs_properties_t *props, obs_property_t *property, void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
    if (!ptr) {
        return true;
    }

    std::shared_ptr<yolo_detector_filter> tf = *ptr;
    if (!tf) {
        return true;
    }

    int currentConfig = tf->currentConfigIndex;
    std::string port = tf->mouseConfigs[currentConfig].makcuPort;
    int baudRate = tf->mouseConfigs[currentConfig].makcuBaudRate;

    MAKCUMouseController tempController(port, baudRate);

    bool isConnected = tempController.isConnected();

    if (isConnected) {
        bool commSuccess = tempController.testCommunication();
        if (commSuccess) {
            MessageBoxA(NULL, "MAKCU连接成功，通信正常", "连接测试", MB_OK | MB_ICONINFORMATION);
        } else {
            MessageBoxA(NULL, "MAKCU连接成功，但通信失败", "连接测试", MB_OK | MB_ICONWARNING);
        }
    } else {
        MessageBoxA(NULL, "MAKCU连接失败", "连接测试", MB_OK | MB_ICONERROR);
    }

    return true;
}

static bool saveConfigCallback(obs_properties_t *props, obs_property_t *property, void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
    if (!ptr) {
        return true;
    }

    std::shared_ptr<yolo_detector_filter> tf = *ptr;
    if (!tf) {
        return true;
    }

    obs_data_t *settings = obs_source_get_settings(tf->source);
    if (!settings) {
        return true;
    }

    char szFile[MAX_PATH] = {0};
    
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "JSON Files (*.json)\0*.json\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = "json";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    
    if (!GetSaveFileNameA(&ofn)) {
        obs_data_release(settings);
        return true;
    }
    
    std::string filePath = szFile;
    FILE* f = fopen(filePath.c_str(), "w");
    if (!f) {
        obs_data_release(settings);
        MessageBoxA(NULL, "无法打开文件进行写入！", "错误", MB_OK | MB_ICONERROR);
        return true;
    }
    
    fprintf(f, "{\n");
    fprintf(f, "  \"configs\": [\n");
    
    for (int i = 0; i < 5; i++) {
        char propName[64];
        
        fprintf(f, "    {\n");
        
        snprintf(propName, sizeof(propName), "enable_config_%d", i);
        fprintf(f, "      \"enabled\": %s,\n", obs_data_get_bool(settings, propName) ? "true" : "false");
        
        snprintf(propName, sizeof(propName), "hotkey_%d", i);
        fprintf(f, "      \"hotkey\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "controller_type_%d", i);
        fprintf(f, "      \"controllerType\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "makcu_port_%d", i);
        fprintf(f, "      \"makcuPort\": \"%s\",\n", obs_data_get_string(settings, propName));
        
        snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
        fprintf(f, "      \"makcuBaudRate\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "p_min_%d", i);
        fprintf(f, "      \"pMin\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "p_max_%d", i);
        fprintf(f, "      \"pMax\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "p_slope_%d", i);
        fprintf(f, "      \"pSlope\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "d_%d", i);
        fprintf(f, "      \"d\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
        fprintf(f, "      \"derivativeFilterAlpha\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", i);
        fprintf(f, "      \"aimSmoothingX\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", i);
        fprintf(f, "      \"aimSmoothingY\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "max_pixel_move_%d", i);
        fprintf(f, "      \"maxPixelMove\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", i);
        fprintf(f, "      \"deadZonePixels\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "target_y_offset_%d", i);
        fprintf(f, "      \"targetYOffset\": %.4f,\n", obs_data_get_double(settings, propName));
        
        snprintf(propName, sizeof(propName), "screen_offset_x_%d", i);
        fprintf(f, "      \"screenOffsetX\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "screen_offset_y_%d", i);
        fprintf(f, "      \"screenOffsetY\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "screen_width_%d", i);
        fprintf(f, "      \"screenWidth\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "screen_height_%d", i);
        fprintf(f, "      \"screenHeight\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", i);
        fprintf(f, "      \"enableYAxisUnlock\": %s,\n", obs_data_get_bool(settings, propName) ? "true" : "false");
        
        snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", i);
        fprintf(f, "      \"yAxisUnlockDelay\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "auto_trigger_group_%d", i);
        fprintf(f, "      \"enableAutoTrigger\": %s,\n", obs_data_get_bool(settings, propName) ? "true" : "false");
        
        snprintf(propName, sizeof(propName), "trigger_radius_%d", i);
        fprintf(f, "      \"triggerRadius\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_cooldown_%d", i);
        fprintf(f, "      \"triggerCooldown\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_fire_delay_%d", i);
        fprintf(f, "      \"triggerFireDelay\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_fire_duration_%d", i);
        fprintf(f, "      \"triggerFireDuration\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_interval_%d", i);
        fprintf(f, "      \"triggerInterval\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
        fprintf(f, "      \"triggerDelayRandomMin\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
        fprintf(f, "      \"triggerDelayRandomMax\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
        fprintf(f, "      \"triggerDurationRandomMin\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
        fprintf(f, "      \"triggerDurationRandomMax\": %d,\n", (int)obs_data_get_int(settings, propName));
        
        snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
        fprintf(f, "      \"triggerMoveCompensation\": %d\n", (int)obs_data_get_int(settings, propName));
        
        fprintf(f, "    }%s\n", (i < 4) ? "," : "");
    }
    
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    obs_data_release(settings);
    
    MessageBoxA(NULL, ("配置已保存到:\n" + filePath).c_str(), "成功", MB_OK | MB_ICONINFORMATION);

    return true;
}

static bool loadConfigCallback(obs_properties_t *props, obs_property_t *property, void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
    if (!ptr) {
        return true;
    }

    std::shared_ptr<yolo_detector_filter> tf = *ptr;
    if (!tf) {
        return true;
    }

    char szFile[MAX_PATH] = {0};
    
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "JSON Files (*.json)\0*.json\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = "json";
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    
    if (!GetOpenFileNameA(&ofn)) {
        return true;
    }
    
    std::string filePath = szFile;
    FILE* f = fopen(filePath.c_str(), "r");
    if (!f) {
        MessageBoxA(NULL, "无法打开文件！", "错误", MB_OK | MB_ICONERROR);
        return true;
    }
    
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    std::string content(fileSize, '\0');
    fread(&content[0], 1, fileSize, f);
    fclose(f);
    
    obs_data_t *settings = obs_source_get_settings(tf->source);
    if (!settings) {
        return true;
    }
    
    auto findValueInConfig = [&content](int configIndex, const char* key, std::string& outValue) -> bool {
        char configStartKey[32];
        snprintf(configStartKey, sizeof(configStartKey), "\"configs\": [");
        size_t configsPos = content.find(configStartKey);
        if (configsPos == std::string::npos) return false;
        
        size_t searchStart = configsPos + strlen(configStartKey);
        for (int c = 0; c <= configIndex; c++) {
            char braceKey[8];
            snprintf(braceKey, sizeof(braceKey), "{");
            size_t bracePos = content.find(braceKey, searchStart);
            if (bracePos == std::string::npos) return false;
            
            if (c == configIndex) {
                size_t braceEnd = content.find("}", bracePos);
                if (braceEnd == std::string::npos) return false;
                
                std::string configBlock = content.substr(bracePos, braceEnd - bracePos);
                char searchKey[64];
                snprintf(searchKey, sizeof(searchKey), "\"%s\":", key);
                size_t keyPos = configBlock.find(searchKey);
                if (keyPos == std::string::npos) return false;
                
                size_t valueStart = keyPos + strlen(searchKey);
                while (valueStart < configBlock.size() && (configBlock[valueStart] == ' ' || configBlock[valueStart] == '\n' || configBlock[valueStart] == '\r')) {
                    valueStart++;
                }
                
                if (valueStart >= configBlock.size()) return false;
                
                if (configBlock[valueStart] == '"') {
                    valueStart++;
                    size_t valueEnd = configBlock.find('"', valueStart);
                    if (valueEnd == std::string::npos) return false;
                    outValue = configBlock.substr(valueStart, valueEnd - valueStart);
                } else {
                    size_t valueEnd = valueStart;
                    while (valueEnd < configBlock.size() && configBlock[valueEnd] != ',' && configBlock[valueEnd] != '\n' && configBlock[valueEnd] != '\r' && configBlock[valueEnd] != '}') {
                        valueEnd++;
                    }
                    outValue = configBlock.substr(valueStart, valueEnd - valueStart);
                    while (!outValue.empty() && (outValue.back() == ' ' || outValue.back() == '\n' || outValue.back() == '\r')) {
                        outValue.pop_back();
                    }
                }
                return true;
            }
            searchStart = content.find("}", bracePos);
            if (searchStart == std::string::npos) return false;
            searchStart++;
        }
        return false;
    };

    for (int i = 0; i < 5; i++) {
        char propName[64];
        std::string val;
        
        snprintf(propName, sizeof(propName), "enable_config_%d", i);
        if (findValueInConfig(i, "enabled", val)) {
            obs_data_set_bool(settings, propName, val.find("true") != std::string::npos);
        }
        
        snprintf(propName, sizeof(propName), "hotkey_%d", i);
        if (findValueInConfig(i, "hotkey", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "controller_type_%d", i);
        if (findValueInConfig(i, "controllerType", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "makcu_port_%d", i);
        if (findValueInConfig(i, "makcuPort", val)) {
            obs_data_set_string(settings, propName, val.c_str());
        }
        
        snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
        if (findValueInConfig(i, "makcuBaudRate", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "p_min_%d", i);
        if (findValueInConfig(i, "pMin", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "p_max_%d", i);
        if (findValueInConfig(i, "pMax", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "p_slope_%d", i);
        if (findValueInConfig(i, "pSlope", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "d_%d", i);
        if (findValueInConfig(i, "d", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
        if (findValueInConfig(i, "derivativeFilterAlpha", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", i);
        if (findValueInConfig(i, "aimSmoothingX", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", i);
        if (findValueInConfig(i, "aimSmoothingY", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "max_pixel_move_%d", i);
        if (findValueInConfig(i, "maxPixelMove", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", i);
        if (findValueInConfig(i, "deadZonePixels", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "target_y_offset_%d", i);
        if (findValueInConfig(i, "targetYOffset", val)) {
            obs_data_set_double(settings, propName, atof(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "screen_offset_x_%d", i);
        if (findValueInConfig(i, "screenOffsetX", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "screen_offset_y_%d", i);
        if (findValueInConfig(i, "screenOffsetY", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "screen_width_%d", i);
        if (findValueInConfig(i, "screenWidth", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "screen_height_%d", i);
        if (findValueInConfig(i, "screenHeight", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", i);
        if (findValueInConfig(i, "enableYAxisUnlock", val)) {
            obs_data_set_bool(settings, propName, val.find("true") != std::string::npos);
        }
        
        snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", i);
        if (findValueInConfig(i, "yAxisUnlockDelay", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "auto_trigger_group_%d", i);
        if (findValueInConfig(i, "enableAutoTrigger", val)) {
            obs_data_set_bool(settings, propName, val.find("true") != std::string::npos);
        }
        
        snprintf(propName, sizeof(propName), "trigger_radius_%d", i);
        if (findValueInConfig(i, "triggerRadius", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_cooldown_%d", i);
        if (findValueInConfig(i, "triggerCooldown", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_fire_delay_%d", i);
        if (findValueInConfig(i, "triggerFireDelay", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_fire_duration_%d", i);
        if (findValueInConfig(i, "triggerFireDuration", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_interval_%d", i);
        if (findValueInConfig(i, "triggerInterval", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
        if (findValueInConfig(i, "triggerDelayRandomMin", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
        if (findValueInConfig(i, "triggerDelayRandomMax", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
        if (findValueInConfig(i, "triggerDurationRandomMin", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
        if (findValueInConfig(i, "triggerDurationRandomMax", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
        
        snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
        if (findValueInConfig(i, "triggerMoveCompensation", val)) {
            obs_data_set_int(settings, propName, atoi(val.c_str()));
        }
    }
    
    obs_data_release(settings);
    MessageBoxA(NULL, ("配置已从:\n" + filePath + "\n加载").c_str(), "成功", MB_OK | MB_ICONINFORMATION);

    return true;
}



#ifdef _WIN32
static yolo_detector_filter *g_floatingWindowFilter = nullptr;

static LRESULT CALLBACK FloatingWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	yolo_detector_filter *filter = g_floatingWindowFilter;

	switch (msg) {
	case WM_CREATE: {
		CREATESTRUCT *cs = (CREATESTRUCT *)lParam;
		SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
		break;
	}
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);
		if (filter) {
			std::lock_guard<std::mutex> lock(filter->floatingWindowMutex);
			if (!filter->floatingWindowFrame.empty()) {
				// 实现双缓冲
				HDC memDC = CreateCompatibleDC(hdc);
				HBITMAP memBitmap = CreateCompatibleBitmap(hdc, filter->floatingWindowFrame.cols, filter->floatingWindowFrame.rows);
				HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);
				
				// 在内存DC中绘制
				BITMAPINFO bmi = {};
				bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bmi.bmiHeader.biWidth = filter->floatingWindowFrame.cols;
				bmi.bmiHeader.biHeight = -filter->floatingWindowFrame.rows;
				bmi.bmiHeader.biPlanes = 1;
				bmi.bmiHeader.biBitCount = 32;
				bmi.bmiHeader.biCompression = BI_RGB;
				SetDIBitsToDevice(memDC, 0, 0, filter->floatingWindowFrame.cols, filter->floatingWindowFrame.rows,
					0, 0, 0, filter->floatingWindowFrame.rows, filter->floatingWindowFrame.data, &bmi, DIB_RGB_COLORS);
				
				// 一次性复制到窗口DC
				BitBlt(hdc, 0, 0, filter->floatingWindowFrame.cols, filter->floatingWindowFrame.rows,
					memDC, 0, 0, SRCCOPY);
				
				// 清理资源
				SelectObject(memDC, oldBitmap);
				DeleteObject(memBitmap);
				DeleteDC(memDC);
			}
		}
		EndPaint(hwnd, &ps);
		break;
	}
	case WM_LBUTTONDOWN: {
		if (filter) {
			filter->floatingWindowDragging = true;
			POINT pt;
			GetCursorPos(&pt);
			RECT rect;
			GetWindowRect(hwnd, &rect);
			filter->floatingWindowDragOffset.x = pt.x - rect.left;
			filter->floatingWindowDragOffset.y = pt.y - rect.top;
			SetCapture(hwnd);
		}
		break;
	}
	case WM_LBUTTONUP: {
		if (filter) {
			filter->floatingWindowDragging = false;
			ReleaseCapture();
		}
		break;
	}
	case WM_MOUSEMOVE: {
		if (filter && filter->floatingWindowDragging) {
			POINT pt;
			GetCursorPos(&pt);
			SetWindowPos(hwnd, NULL, pt.x - filter->floatingWindowDragOffset.x, pt.y - filter->floatingWindowDragOffset.y,
				0, 0, SWP_NOSIZE | SWP_NOZORDER);
		}
		break;
	}
	case WM_CLOSE: {
			if (filter) {
				filter->showFloatingWindow = false;
				destroyFloatingWindow(filter);
				// 保存悬浮窗关闭状态
				obs_data_t *settings = obs_source_get_settings(filter->source);
				if (settings) {
					obs_data_set_bool(settings, "show_floating_window", false);
					obs_data_release(settings);
				}
			}
			break;
		}
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

static void createFloatingWindow(yolo_detector_filter *filter)
{
	if (filter->floatingWindowHandle) {
		return;
	}

	g_floatingWindowFilter = filter;

	WNDCLASS wc = {};
	wc.lpfnWndProc = FloatingWindowProc;
	wc.hInstance = GetModuleHandle(NULL);
	wc.lpszClassName = L"YOLODetectorFloatingWindow";
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClass(&wc);

	int x = GetSystemMetrics(SM_CXSCREEN) / 2 - filter->floatingWindowWidth / 2;
	int y = GetSystemMetrics(SM_CYSCREEN) / 2 - filter->floatingWindowHeight / 2;

	filter->floatingWindowHandle = CreateWindowEx(
		WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
		L"YOLODetectorFloatingWindow",
		L"YOLO Detector",
		WS_POPUP | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		x, y,
		filter->floatingWindowWidth, filter->floatingWindowHeight,
		NULL, NULL, GetModuleHandle(NULL), filter
	);

	filter->floatingWindowX = x;
	filter->floatingWindowY = y;
	filter->floatingWindowDragging = false;

	obs_log(LOG_INFO, "[YOLO Detector] Floating window created");
}

static void destroyFloatingWindow(yolo_detector_filter *filter)
{
	if (filter->floatingWindowHandle) {
		DestroyWindow(filter->floatingWindowHandle);
		filter->floatingWindowHandle = nullptr;
		g_floatingWindowFilter = nullptr;
		obs_log(LOG_INFO, "[YOLO Detector] Floating window destroyed");
	}
}

static void updateFloatingWindowFrame(yolo_detector_filter *filter, const cv::Mat &frame)
{
	std::lock_guard<std::mutex> lock(filter->floatingWindowMutex);
	frame.copyTo(filter->floatingWindowFrame);

#ifdef _WIN32
	int frameWidth = filter->floatingWindowFrame.cols;
	int frameHeight = filter->floatingWindowFrame.rows;

	// 检测框和trackId已在主渲染中绘制，这里不再重复绘制

	// 在底部绘制统计信息面板（透明背景）
	{
		int frameWidth = filter->floatingWindowFrame.cols;
		int frameHeight = filter->floatingWindowFrame.rows;
		int panelHeight = 50;
		int panelY = frameHeight - panelHeight;
		
		// 绘制统计信息（无背景，透明显示）
		int textY = panelY + 15;
		int lineHeight = 15;
		double fontScale = 0.4;
		int thickness = 1;
		cv::Scalar textColor(255, 255, 255);
		cv::Scalar shadowColor(0, 0, 0);
		
		// 第一行：FPS和推理时间
		char buf[256];
		snprintf(buf, sizeof(buf), "FPS: %.1f | Inference: %.1fms | Detections: %zu",
			filter->currentFps,
			filter->avgInferenceTimeMs,
			filter->detections.size());
		// 绘制阴影提高可读性
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(11, textY + 1), cv::FONT_HERSHEY_SIMPLEX, fontScale, shadowColor, thickness + 1);
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(10, textY), cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
		
		// 第二行：帧统计
		textY += lineHeight;
		snprintf(buf, sizeof(buf), "Submitted: %llu | Inferred: %llu | Consumed: %llu",
			(unsigned long long)filter->framesSubmitted.load(),
			(unsigned long long)filter->framesInferred.load(),
			(unsigned long long)filter->framesConsumed.load());
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(11, textY + 1), cv::FONT_HERSHEY_SIMPLEX, fontScale, shadowColor, thickness + 1);
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(10, textY), cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
		
		// 第三行：丢帧统计（如果有）
		int dropped = filter->framesDropped.load();
		if (dropped > 0) {
			textY += lineHeight;
			snprintf(buf, sizeof(buf), "Dropped: %d", dropped);
			cv::putText(filter->floatingWindowFrame, buf, 
				cv::Point(11, textY + 1), cv::FONT_HERSHEY_SIMPLEX, fontScale, shadowColor, thickness + 1);
			cv::putText(filter->floatingWindowFrame, buf, 
				cv::Point(10, textY), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 100, 255), thickness);
		}
	}

#endif
}

// ============================================================
// PID 调试面板 - 增强版绘制函数
// 6 区域布局: 目标速度 | 误差 | P/I/D分项 | 总输出 | 积分+占比 | 控制模式
// ============================================================

// 绘制单条曲线（复用工具函数）
static void drawSingleCurve(cv::Mat &canvas,
                             int baseY, int topMargin, int bottomMargin,
                             int width,
                             const std::deque<yolo_detector_filter::PidDataPoint> &history,
                             std::function<float(const yolo_detector_filter::PidDataPoint&)> getValue,
                             float maxValue,
                             const cv::Scalar &color,
                             int thickness = 2)
{
    if (history.empty() || maxValue < 1e-6f) return;

    int usableHeight = bottomMargin - topMargin - 6;
    float scale = usableHeight / maxValue;

    std::vector<cv::Point> points;
    int x = 0;
    int step = std::max(1, width / static_cast<int>(history.size()));

    for (const auto &data : history) {
        float value = getValue(data);
        int y = baseY - static_cast<int>(value * scale);
        y = std::clamp(y, topMargin + 3, bottomMargin - 3);
        points.push_back(cv::Point(x, y));
        x += step;
        if (x >= width) break;
    }

    if (points.size() > 1) {
        cv::polylines(canvas, points, false, color, thickness);
    }
}

// 绘制区域标题
static void drawRegionLabel(cv::Mat &canvas, int regionIndex, int regionHeight, const char *text)
{
    int y = regionIndex * regionHeight + 14;
    cv::putText(canvas, text, cv::Point(5, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(200, 200, 220), 1);
}

// 绘制区域分隔线（细线）
static void drawRegionSeparator(cv::Mat &canvas, int y, int width)
{
    cv::line(canvas, cv::Point(0, y), cv::Point(width, y), cv::Scalar(55, 55, 65), 1);
}

// 绘制零线（中线）
static void drawZeroLine(cv::Mat &canvas, int y, int width)
{
    cv::line(canvas, cv::Point(0, y), cv::Point(width, y), cv::Scalar(70, 70, 80), 1);
}

// ============================================================
// 核心: 积分仪表盘绘制（圆弧 + 颜色编码）
// ============================================================
static void drawIntegralGauge(cv::Mat &canvas, int cx, int cy, int radius,
                               float ratioX, float ratioY)
{
    // 取两轴中较大的积分占用率
    float ratio = std::max(ratioX, ratioY);

    // 背景圆环（暗灰）
    cv::ellipse(canvas, cv::Point(cx, cy), cv::Size(radius, radius),
                0, 180, 360, cv::Scalar(45, 45, 55), 4);

    // 根据比例选择颜色
    cv::Scalar gaugeColor;
    if (ratio > 0.85f) {
        // 危险区：红色闪烁效果用高亮红
        gaugeColor = cv::Scalar(20, 20, 240);   // BGR 红色
    } else if (ratio > 0.65f) {
        // 警告区：橙黄色
        gaugeColor = cv::Scalar(20, 180, 230);  // BGR 黄色
    } else {
        // 安全区：绿色
        gaugeColor = cv::Scalar(40, 210, 80);   // BGR 绿色
    }

    // 绘制进度弧线（从左到右的半圆，180°~0°）
    float angleEnd = 180.0f - ratio * 180.0f;
    if (ratio > 0.001f) {
        cv::ellipse(canvas, cv::Point(cx, cy), cv::Size(radius, radius),
                    0, angleEnd, 180, gaugeColor, 4);
    }

    // 中心百分比文字
    char buf[16];
    snprintf(buf, sizeof(buf), "%.0f%%", ratio * 100.0f);
    int fontBase = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, nullptr).height;

    // 文字阴影
    cv::putText(canvas, buf, cv::Point(cx + 1, cy + fontBase / 3 + 1),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(15, 15, 15), 1);
    cv::putText(canvas, buf, cv::Point(cx, cy + fontBase / 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, gaugeColor, 1);
}

// ============================================================
// 核心: P/I/D 占比条形图绘制
// ============================================================
static void drawPidBarChart(cv::Mat &canvas, int barX, int barY, int barWidth, int barHeight,
                            float pVal, float iVal, float dVal)
{
    float total = pVal + iVal + dVal;
    if (total < 0.01f) total = 1.0f;

    int pPx = static_cast<int>(barWidth * pVal / total);
    int iPxBg = pPx;
    int iPx = static_cast<int>(barWidth * iVal / total);
    int dPx = barWidth - pPx - iPx;

    // 背景
    cv::rectangle(canvas, cv::Rect(barX, barY, barWidth, barHeight),
                  cv::Scalar(38, 38, 48), -1);

    // P 段（红色）
    if (pPx > 0) {
        cv::rectangle(canvas, cv::Rect(barX, barY, pPx, barHeight),
                      cv::Scalar(90, 50, 55), -1);  // 暗红
    }
    // I 段（绿色）
    if (iPx > 0) {
        cv::rectangle(canvas, cv::Rect(barX + iPxBg, barY, iPx, barHeight),
                      cv::Scalar(45, 95, 55), -1);  // 暗绿
    }
    // D 段（蓝色）
    if (dPx > 0) {
        cv::rectangle(canvas, cv::Rect(barX + iPxBg + iPx, barY, dPx, barHeight),
                      cv::Scalar(50, 55, 95), -1);  // 暗蓝
    }

    // 边框
    cv::rectangle(canvas, cv::Rect(barX, barY, barWidth, barHeight),
                  cv::Scalar(100, 100, 110), 1);

    // 百分比标注
    char buf[64];
    int textY = barY + barHeight - 3;
    int pPct = static_cast<int>(pVal / total * 100.0f);
    int iPct = static_cast<int>(iVal / total * 100.0f);
    int dPct = static_cast<int>(dVal / total * 100.0f);

    // 在各段中心显示百分比（如果段足够宽）
    if (pPx > 22) {
        snprintf(buf, sizeof(buf), "P%d%%", pPct);
        cv::putText(canvas, buf, cv::Point(barX + pPx/2 - 12, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(200, 120, 130), 1);
    }
    if (iPx > 22) {
        snprintf(buf, sizeof(buf), "I%d%%", iPct);
        cv::putText(canvas, buf, cv::Point(barX + iPxBg + iPx/2 - 12, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(120, 220, 140), 1);
    }
    if (dPx > 22) {
        snprintf(buf, sizeof(buf), "D%d%%", dPct);
        cv::putText(canvas, buf, cv::Point(barX + iPxBg + iPx + dPx/2 - 12, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(130, 140, 220), 1);
    }
}

// ============================================================
// 控制 mode → 字符串映射
// ============================================================
static const char* controlModeToString(int mode)
{
    switch (mode) {
        case 0: return "IDLE";
        case 1: return "TRACKING";
        case 2: return "LOCKED";
        case 3: return "I-SAT!";
        case 4: return "OSCILLATE";
        case 5: return "PREDICT";
        default: return "UNKNOWN";
    }
}

static cv::Scalar controlModeColor(int mode)
{
    switch (mode) {
        case 0: return cv::Scalar(120, 120, 130);      // IDLE: 灰色
        case 1: return cv::Scalar(255, 255, 255);       // TRACKING: 白色
        case 2: return cv::Scalar(60, 220, 110);        // LOCKED: 绿色
        case 3: return cv::Scalar(20, 20, 250);         // I-SAT: 红色
        case 4: return cv::Scalar(20, 200, 240);        // OSCILLATE: 黄色
        case 5: return cv::Scalar(200, 160, 255);       // PREDICT: 粉紫
        default: return cv::Scalar(150, 150, 150);
    }
}

// ============================================================
// 主绘制入口: drawPidDebugGraph (增强版 6 区域布局)
// ============================================================
static void drawPidDebugGraph(yolo_detector_filter *filter, cv::Mat &canvas)
{
    if (!filter || filter->pidHistory.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(filter->pidHistoryMutex);

    int width = canvas.cols;
    int height = canvas.rows;
    const int NUM_REGIONS = 6;
    int regionH = height / NUM_REGIONS;

    // === 背景 ===
    canvas.setTo(cv::Scalar(25, 26, 33));

    // === 分隔线 ===
    for (int i = 1; i < NUM_REGIONS; ++i) {
        drawRegionSeparator(canvas, i * regionH, width);
    }

    // === 计算最大值用于自适应缩放 ===
    float maxVel = 500.0f;
    float maxErr = 100.0f;
    float maxComp = 50.0f;   // P/I/D 分项最大值
    float maxOut = 50.0f;
    float maxKp = 1.0f;

    for (const auto &data : filter->pidHistory) {
        maxVel = std::max(maxVel, std::max(std::abs(data.targetVelocityX), std::abs(data.targetVelocityY)));
        maxErr = std::max(maxErr, std::max(std::abs(data.errorX), std::abs(data.errorY)));
        maxOut = std::max(maxOut, std::max(std::abs(data.outputX), std::abs(data.outputY)));

        float pMag = std::max(std::abs(data.pTermX), std::abs(data.pTermY));
        float iMag = std::max(std::abs(data.iTermX), std::abs(data.iTermY));
        float dMag = std::max(std::abs(data.dTermX), std::abs(data.dTermY));
        maxComp = std::max(maxComp, std::max(pMag, std::max(iMag, dMag)));

        maxKp = std::max(maxKp, std::abs(data.currentKp));
    }

    // 保证非零下界
    maxVel = std::max(maxVel, 1.0f);
    maxErr = std::max(maxErr, 1.0f);
    maxComp = std::max(maxComp, 1.0f);
    maxOut = std::max(maxOut, 1.0f);
    maxKp = std::max(maxKp, 0.01f);

    // =====================================================================
    // 区域 0: 目标速度 Target Velocity X/Y
    // =====================================================================
    int r0Top = 0 * regionH;
    int r0Bot = 1 * regionH;
    int r0Mid = r0Top + regionH / 2;
    drawRegionLabel(canvas, 0, regionH, "Target Velocity");
    drawZeroLine(canvas, r0Mid, width);

    drawSingleCurve(canvas, r0Mid, r0Top, r0Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.targetVelocityX; },
                    maxVel, cv::Scalar(0, 0, 220), 2);  // X: 红

    drawSingleCurve(canvas, r0Mid, r0Top, r0Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.targetVelocityY; },
                    maxVel, cv::Scalar(0, 200, 0), 2);    // Y: 绿

    // =====================================================================
    // 区域 1: 误差 Error X/Y
    // =====================================================================
    int r1Top = 1 * regionH;
    int r1Bot = 2 * regionH;
    int r1Mid = r1Top + regionH / 2;
    drawRegionLabel(canvas, 1, regionH, "Error X/Y");
    drawZeroLine(canvas, r1Mid, width);

    drawSingleCurve(canvas, r1Mid, r1Top, r1Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.errorX; },
                    maxErr, cv::Scalar(0, 0, 220), 2);

    drawSingleCurve(canvas, r1Mid, r1Top, r1Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.errorY; },
                    maxErr, cv::Scalar(0, 200, 0), 2);

    // =====================================================================
    // 区域 2: P/I/D 分项输出（堆叠曲线图）★核心新增★
    // =====================================================================
    int r2Top = 2 * regionH;
    int r2Bot = 3 * regionH;
    int r2Mid = r2Top + regionH / 2;
    drawRegionLabel(canvas, 2, regionH, "PID Components (P=Red I=Green D=Blue)");
    drawZeroLine(canvas, r2Mid, width);

    // P项曲线（红色系）
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.pTermX; },
                    maxComp, cv::Scalar(100, 50, 55), 2);
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.pTermY; },
                    maxComp, cv::Scalar(140, 75, 80), 2);

    // I项曲线（绿色系）
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.iTermX; },
                    maxComp, cv::Scalar(50, 120, 55), 2);
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.iTermY; },
                    maxComp, cv::Scalar(70, 155, 75), 2);

    // D项曲线（蓝色系）
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.dTermX; },
                    maxComp, cv::Scalar(55, 60, 130), 2);
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.dTermY; },
                    maxComp, cv::Scalar(75, 80, 160), 2);

    // 图例
    cv::putText(canvas, "P", cv::Point(width - 18, r2Top + 13),
                cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(170, 90, 95), 1);
    cv::putText(canvas, "I", cv::Point(width - 36, r2Top + 13),
                cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(90, 190, 105), 1);
    cv::putText(canvas, "D", cv::Point(width - 54, r2Top + 13),
                cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(95, 100, 200), 1);

    // =====================================================================
    // 区域 3: 总输出 Output X/Y
    // =====================================================================
    int r3Top = 3 * regionH;
    int r3Bot = 4 * regionH;
    int r3Mid = r3Top + regionH / 2;
    drawRegionLabel(canvas, 3, regionH, "Total Output X/Y");
    drawZeroLine(canvas, r3Mid, width);

    drawSingleCurve(canvas, r3Mid, r3Top, r3Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.outputX; },
                    maxOut, cv::Scalar(0, 0, 220), 2);

    drawSingleCurve(canvas, r3Mid, r3Top, r3Bot, width, filter->pidHistory,
                    [](const yolo_detector_filter::PidDataPoint &d) { return d.outputY; },
                    maxOut, cv::Scalar(0, 200, 0), 2);

    // =====================================================================
    // 区域 4: 积分仪表盘 + P/I/D 占比条 ★核心新增★
    // =====================================================================
    int r4Top = 4 * regionH;
    drawRegionLabel(canvas, 4, regionH, "Integral Gauge & PID Ratio");

    if (!filter->pidHistory.empty()) {
        const auto &latest = filter->pidHistory.back();

        // --- 左侧: 积分仪表盘 ---
        int gaugeCX = 48;
        int gaugeCY = r4Top + regionH / 2;
        int gaugeR = std::min(regionH / 2 - 8, 28);
        drawIntegralGauge(canvas, gaugeCX, gaugeCY, gaugeR,
                          latest.integralRatioX, latest.integralRatioY);

        // 仪表盘下方标签
        char iBuf[24];
        snprintf(iBuf, sizeof(iBuf), "I-X:%.0f%% I-Y:%.0f%%",
                 latest.integralRatioX * 100.0f, latest.integralRatioY * 100.0f);
        cv::putText(canvas, iBuf, cv::Point(8, r4Top + regionH - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.30, cv::Scalar(160, 165, 175), 1);

        // --- 右侧: P/I/D 占比条 ---
        int barX = 115;
        int barY = r4Top + regionH / 2 - 9;
        int barW = width - barX - 8;
        int barH = 18;

        float pAbs = std::abs(latest.pTermX) + std::abs(latest.pTermY);
        float iAbs = std::abs(latest.iTermX) + std::abs(latest.iTermY);
        float dAbs = std::abs(latest.dTermX) + std::abs(latest.dTermY);

        drawPidBarChart(canvas, barX, barY, barW, barH, pAbs, iAbs, dAbs);

        // 占比条下方文字
        char ratioBuf[48];
        snprintf(ratioBuf, sizeof(ratioBuf), "P:%.1f I:%.1f D:%.1f",
                 latest.pTermX + latest.pTermY,
                 latest.iTermX + latest.iTermY,
                 latest.dTermX + latest.dTermY);
        cv::putText(canvas, ratioBuf, cv::Point(barX, r4Top + regionH - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.30, cv::Scalar(145, 150, 160), 1);
    }

    // =====================================================================
    // 区域 5: 控制模式 + 实时状态栏 ★核心新增★
    // =====================================================================
    int r5Top = 5 * regionH;
    int r5Bot = height;
    drawRegionLabel(canvas, 5, regionH, "Control Mode & Status");

    if (!filter->pidHistory.empty()) {
        const auto &latest = filter->pidHistory.back();

        // --- 第一行: 控制模式指示器 ---
        int modeY = r5Top + 22;
        const char *modeStr = controlModeToString(latest.controlMode);
        cv::Scalar modeCol = controlModeColor(latest.controlMode);

        // 模式名称（带背景框突出显示）
        int tw = cv::getTextSize(modeStr, cv::FONT_HERSHEY_SIMPLEX, 0.44, 1, nullptr).width;
        cv::rectangle(canvas, cv::Rect(5, modeY - 14, tw + 10, 18),
                      cv::Scalar(35, 36, 43), -1);
        cv::rectangle(canvas, cv::Rect(5, modeY - 14, tw + 10, 18),
                      modeCol, 1);
        cv::putText(canvas, modeStr, cv::Point(10, modeY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.44, modeCol, 1);

        // 射击状态指示器
        int firingX = tw + 20;
        if (latest.isFiring) {
            cv::putText(canvas, "[FIRE]", cv::Point(firingX, modeY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(20, 20, 240), 1);
        } else {
            cv::putText(canvas, "[IDLE]", cv::Point(firingX, modeY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(100, 100, 110), 1);
        }

        // 算法类型
        const char *algoNames[] = {"AdvPID", "StdPID", "Chris"};
        // algorithmType 存储在 PidDataPoint 中但此处未直接使用，显示默认值
        cv::putText(canvas, algoNames[0],
                    cv::Point(firingX + 52, modeY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(140, 140, 160), 1);

        // --- 第二行: 关键数值状态栏 ---
        int statY = r5Top + 43;
        char statBuf[160];
        snprintf(statBuf, sizeof(statBuf),
                 "eX:%.1f eY:%.1f | outX:%.1f outY:%.1f | Kp:%.3f Ki:%.4f Kd:%.4f",
                 latest.errorX, latest.errorY,
                 latest.outputX, latest.outputY,
                 latest.currentKp, latest.currentKi, latest.currentKd);
        cv::putText(canvas, statBuf, cv::Point(5, statY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(185, 190, 200), 1);

        // --- 第三行: 积分数值详情 ---
        int detailY = r5Top + 62;
        char detailBuf[140];
        snprintf(detailBuf, sizeof(detailBuf),
                 "iAbsX:%.1f iAbsY:%.1f | iRatioX:%.0f%% iRatioY:%.0f%%",
                 latest.integralAbsX, latest.integralAbsY,
                 latest.integralRatioX * 100.0f, latest.integralRatioY * 100.0f);
        cv::putText(canvas, detailBuf, cv::Point(5, detailY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.31, cv::Scalar(135, 140, 155), 1);
    }
}

static void renderFloatingWindow(yolo_detector_filter *filter)
{
	if (!filter->floatingWindowHandle || filter->floatingWindowFrame.empty()) {
		return;
	}
	InvalidateRect(filter->floatingWindowHandle, NULL, FALSE);
}

static void setupPidDataCallback(yolo_detector_filter *filter)
{
	if (!filter || !filter->mouseController) {
		return;
	}
	
	filter->mouseController->setPidDataCallback([filter](const PidDebugData &data) {
		std::lock_guard<std::mutex> lock(filter->pidHistoryMutex);

		yolo_detector_filter::PidDataPoint point;
		point.errorX = data.errorX;
		point.errorY = data.errorY;
		point.outputX = data.outputX;
		point.outputY = data.outputY;
		point.targetX = data.targetX;
		point.targetY = data.targetY;
		point.targetVelocityX = data.targetVelocityX;
		point.targetVelocityY = data.targetVelocityY;
		point.currentKp = data.currentKp;
		point.currentKi = data.currentKi;
		point.currentKd = data.currentKd;

		// 新增字段
		point.pTermX = data.pTermX;
		point.pTermY = data.pTermY;
		point.iTermX = data.iTermX;
		point.iTermY = data.iTermY;
		point.dTermX = data.dTermX;
		point.dTermY = data.dTermY;
		point.integralAbsX = data.integralAbsX;
		point.integralAbsY = data.integralAbsY;
		point.integralRatioX = data.integralRatioX;
		point.integralRatioY = data.integralRatioY;
		point.controlMode = data.controlMode;
		point.isFiring = data.isFiring;

		point.timestamp = std::chrono::steady_clock::now();
		
		filter->pidHistory.push_back(point);
		
		while (filter->pidHistory.size() > filter->PID_HISTORY_SIZE) {
			filter->pidHistory.pop_front();
		}
	});
}

static yolo_detector_filter *g_pidDebugWindowFilter = nullptr;

static LRESULT CALLBACK PidDebugWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	yolo_detector_filter *filter = g_pidDebugWindowFilter;

	switch (msg) {
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);
		
		if (filter) {
			std::lock_guard<std::mutex> lock(filter->pidDebugWindowMutex);
			if (!filter->pidDebugWindowFrame.empty()) {
				BITMAPINFO bmi = {};
				bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bmi.bmiHeader.biWidth = filter->pidDebugWindowFrame.cols;
				bmi.bmiHeader.biHeight = -filter->pidDebugWindowFrame.rows;
				bmi.bmiHeader.biPlanes = 1;
				bmi.bmiHeader.biBitCount = 24;
				bmi.bmiHeader.biCompression = BI_RGB;
				SetDIBitsToDevice(hdc, 0, 0, filter->pidDebugWindowFrame.cols, filter->pidDebugWindowFrame.rows,
					0, 0, 0, filter->pidDebugWindowFrame.rows, filter->pidDebugWindowFrame.data, &bmi, DIB_RGB_COLORS);
			}
		}
		
		EndPaint(hwnd, &ps);
		break;
	}
	case WM_LBUTTONDOWN: {
		if (filter) {
			filter->pidDebugWindowDragging = true;
			POINT pt;
			GetCursorPos(&pt);
			filter->pidDebugWindowDragOffset.x = pt.x - filter->pidDebugWindowX;
			filter->pidDebugWindowDragOffset.y = pt.y - filter->pidDebugWindowY;
			SetCapture(hwnd);
		}
		break;
	}
	case WM_LBUTTONUP: {
		if (filter) {
			filter->pidDebugWindowDragging = false;
			ReleaseCapture();
		}
		break;
	}
	case WM_MOUSEMOVE: {
		if (filter && filter->pidDebugWindowDragging) {
			POINT pt;
			GetCursorPos(&pt);
			SetWindowPos(hwnd, NULL, pt.x - filter->pidDebugWindowDragOffset.x, pt.y - filter->pidDebugWindowDragOffset.y,
				0, 0, SWP_NOSIZE | SWP_NOZORDER);
			filter->pidDebugWindowX = pt.x - filter->pidDebugWindowDragOffset.x;
			filter->pidDebugWindowY = pt.y - filter->pidDebugWindowDragOffset.y;
		}
		break;
	}
	case WM_CLOSE: {
		if (filter) {
			filter->showPidDebugWindow = false;
			destroyPidDebugWindow(filter);
			obs_data_t *settings = obs_source_get_settings(filter->source);
			if (settings) {
				obs_data_set_bool(settings, "show_pid_debug_window", false);
				obs_data_release(settings);
			}
		}
		break;
	}
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

static void createPidDebugWindow(yolo_detector_filter *filter)
{
	if (filter->pidDebugWindowHandle) {
		return;
	}

	g_pidDebugWindowFilter = filter;

	WNDCLASS wc = {};
	wc.lpfnWndProc = PidDebugWindowProc;
	wc.hInstance = GetModuleHandle(NULL);
	wc.lpszClassName = L"YOLODetectorPidDebugWindow";
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClass(&wc);

	int x = filter->pidDebugWindowX;
	int y = filter->pidDebugWindowY;
	if (x == 0 && y == 0) {
		x = GetSystemMetrics(SM_CXSCREEN) / 2 - filter->pidDebugWindowWidth / 2;
		y = GetSystemMetrics(SM_CYSCREEN) / 2 - filter->pidDebugWindowHeight / 2 + 100;
	}

	filter->pidDebugWindowHandle = CreateWindowEx(
		WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
		L"YOLODetectorPidDebugWindow",
		L"PID Debug",
		WS_POPUP | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		x, y,
		filter->pidDebugWindowWidth, filter->pidDebugWindowHeight,
		NULL, NULL, GetModuleHandle(NULL), filter
	);

	filter->pidDebugWindowX = x;
	filter->pidDebugWindowY = y;
	filter->pidDebugWindowDragging = false;

	obs_log(LOG_INFO, "[YOLO Detector] PID debug window created");
}

static void destroyPidDebugWindow(yolo_detector_filter *filter)
{
	if (filter->pidDebugWindowHandle) {
		DestroyWindow(filter->pidDebugWindowHandle);
		filter->pidDebugWindowHandle = nullptr;
		g_pidDebugWindowFilter = nullptr;
		obs_log(LOG_INFO, "[YOLO Detector] PID debug window destroyed");
	}
}

static void updatePidDebugWindow(yolo_detector_filter *filter)
{
	if (!filter->pidDebugWindowHandle) {
		return;
	}

	{
		std::lock_guard<std::mutex> lock(filter->pidDebugWindowMutex);
		filter->pidDebugWindowFrame.create(filter->pidDebugWindowHeight, filter->pidDebugWindowWidth, CV_8UC3);
		drawPidDebugGraph(filter, filter->pidDebugWindowFrame);
	}

	InvalidateRect(filter->pidDebugWindowHandle, NULL, FALSE);
}
#endif

// 线程池工作函数
void threadPoolWorker(yolo_detector_filter *filter)
{
	while (filter->threadPoolRunning) {
		std::function<void()> task;
		{
			std::unique_lock<std::mutex> lock(filter->taskQueueMutex);
			filter->taskCondition.wait(lock, [filter] { return !filter->threadPoolRunning || !filter->taskQueue.empty(); });
			if (!filter->threadPoolRunning && filter->taskQueue.empty()) {
				return;
			}
			task = std::move(filter->taskQueue.front());
			filter->taskQueue.pop();
		}
		task();
	}
}

// 提交任务到线程池
template<typename F>
void submitTask(yolo_detector_filter *filter, F &&task)
{
	std::unique_lock<std::mutex> lock(filter->taskQueueMutex);
	filter->taskQueue.push(std::function<void()>(std::forward<F>(task)));
	lock.unlock();
	filter->taskCondition.notify_one();
}

// 从内存池中获取图像缓冲区
cv::Mat getImageBuffer(yolo_detector_filter *filter, int rows, int cols, int type)
{
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	yolo_detector_filter::ImageBufferKey key{rows, cols, type};
	auto it = filter->imageBufferPool.find(key);
	
	if (it != filter->imageBufferPool.end() && !it->second.empty()) {
		cv::Mat buffer = std::move(it->second.back());
		it->second.pop_back();
		return buffer;
	}
	
	// 如果没有合适的缓冲区，创建一个新的
	return cv::Mat(rows, cols, type);
}

// 释放图像缓冲区到内存池
void releaseImageBuffer(yolo_detector_filter *filter, cv::Mat &&buffer)
{
	if (buffer.empty()) {
		return;
	}
	
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	yolo_detector_filter::ImageBufferKey key{buffer.rows, buffer.cols, buffer.type()};
	auto it = filter->imageBufferPool.find(key);
	
	if (it != filter->imageBufferPool.end()) {
		if (it->second.size() < filter->MAX_BUFFER_POOL_SIZE) {
			it->second.push_back(std::move(buffer));
		}
	} else {
		std::vector<cv::Mat> buffers;
		buffers.reserve(5);
		buffers.push_back(std::move(buffer));
		filter->imageBufferPool[key] = std::move(buffers);
	}
}

// 从内存池中获取检测结果缓冲区
std::vector<Detection> getDetectionBuffer(yolo_detector_filter *filter)
{
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	if (!filter->detectionBufferPool.empty()) {
		std::vector<Detection> buffer = std::move(filter->detectionBufferPool.back());
		filter->detectionBufferPool.pop_back();
		buffer.clear(); // 清空缓冲区内容
		return buffer;
	}
	
	// 如果没有可用的缓冲区，创建一个新的
	return std::vector<Detection>();
}

// 释放检测结果缓冲区到内存池
void releaseDetectionBuffer(yolo_detector_filter *filter, std::vector<Detection> &&buffer)
{
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	// 确保内存池不超过最大大小
	if (filter->detectionBufferPool.size() < filter->MAX_BUFFER_POOL_SIZE) {
		// 清空缓冲区内容，保留容量
		buffer.clear();
		filter->detectionBufferPool.push_back(std::move(buffer));
	}
}

void inferenceThreadWorker(yolo_detector_filter *filter)
{
	obs_log(LOG_INFO, "[YOLO Detector] Async inference thread started (4-buffer mode)");

	// 提高线程优先级以减少延迟
	#ifdef _WIN32
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	#endif

	int inferenceFrameCounter = 0;
	// 从UI配置读取推理间隔，0表示每帧都推理
	int inferenceInterval = filter->inferenceIntervalFrames <= 0 ? 1 : filter->inferenceIntervalFrames;

	while (filter->inferenceRunning) {
		if (!filter->isInferencing) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}

		// 帧间隔控制：不是每帧都推理
		inferenceFrameCounter++;
		if (inferenceFrameCounter < inferenceInterval) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		inferenceFrameCounter = 0;

		// 无锁获取待推理帧
		int readIdx = -1;
		int startIdx = filter->inputReadIdx.load(std::memory_order_acquire);
		
		for (int i = 0; i < filter->BUFFER_COUNT; i++) {
			int checkIdx = (startIdx + i) % filter->BUFFER_COUNT;
			uint8_t expected = 1;  // 期望状态为"有数据待推理"
			
			if (filter->bufferState[checkIdx].compare_exchange_strong(
				expected, 2, std::memory_order_acq_rel)) {
				readIdx = checkIdx;
				filter->inputReadIdx.store(checkIdx, std::memory_order_release);
				break;
			}
		}
		
		if (readIdx == -1) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// 读取帧数据（克隆以避免数据竞争）
		cv::Mat frame = filter->inputFrames[readIdx].clone();
		int fullWidth = filter->inputFrameWidths[readIdx];
		int fullHeight = filter->inputFrameHeights[readIdx];
		int cropX = filter->inputCropX[readIdx];
		int cropY = filter->inputCropY[readIdx];
		int cropWidth = filter->inputCropWidth[readIdx];
		int cropHeight = filter->inputCropHeight[readIdx];

		// 标记输入缓冲区为空闲（已读取完毕）
		filter->bufferState[readIdx].store(0, std::memory_order_release);
		
		// 安全检查：确保帧数据有效
		if (frame.empty() || fullWidth <= 0 || fullHeight <= 0) {
			continue;
		}
		
		// 安全检查：确保裁剪区域有效
		if (cropWidth <= 0 || cropHeight <= 0) {
			cropWidth = fullWidth;
			cropHeight = fullHeight;
			cropX = 0;
			cropY = 0;
		}
		
		// 安全检查：确保裁剪区域不超出边界
		if (cropX < 0) cropX = 0;
		if (cropY < 0) cropY = 0;
		if (cropX + cropWidth > fullWidth) cropWidth = fullWidth - cropX;
		if (cropY + cropHeight > fullHeight) cropHeight = fullHeight - cropY;
		if (cropWidth <= 0 || cropHeight <= 0) {
			continue;
		}

		auto startTime = std::chrono::high_resolution_clock::now();
		auto inferenceStartTime = startTime;

		// 如果需要裁切，提取裁切区域
		cv::Mat inferenceFrame;
		if (cropX > 0 || cropY > 0 || cropWidth < fullWidth || cropHeight < fullHeight) {
			inferenceFrame = frame(cv::Rect(cropX, cropY, cropWidth, cropHeight)).clone();
		} else {
			inferenceFrame = frame;
			cropX = 0;
			cropY = 0;
			cropWidth = fullWidth;
			cropHeight = fullHeight;
		}

		// 执行推理
		std::vector<Detection> newDetections;
		{
			std::lock_guard<std::mutex> lock(filter->yoloModelMutex);
			if (filter->yoloModel) {
#ifdef _WIN32
#if defined(HAVE_CUDA) || defined(HAVE_ONNXRUNTIME_DML_EP)
				if (filter->useGpuTextureInference && filter->cachedD3D11Texture &&
				    filter->gpuTextureWidth > 0 && filter->gpuTextureHeight > 0) {
					bool gpuInferenceSuccess = false;
#ifdef HAVE_CUDA
					if (filter->yoloModel->isGpuTextureSupported() && !gpuInferenceSuccess) {
						newDetections = filter->yoloModel->inferenceFromTexture(
							filter->cachedD3D11Texture,
							filter->gpuTextureWidth,
							filter->gpuTextureHeight,
							fullWidth, fullHeight
						);
						gpuInferenceSuccess = true;
					}
#endif
#ifdef HAVE_ONNXRUNTIME_DML_EP
					if (filter->yoloModel->isDmlTextureSupported() && !gpuInferenceSuccess) {
						newDetections = filter->yoloModel->inferenceFromTextureDml(
							filter->cachedD3D11Texture,
							filter->gpuTextureWidth,
							filter->gpuTextureHeight,
							fullWidth, fullHeight
						);
						gpuInferenceSuccess = true;
					}
#endif
					if (!gpuInferenceSuccess) {
						newDetections = filter->yoloModel->inference(inferenceFrame);
					}
				} else {
					newDetections = filter->yoloModel->inference(inferenceFrame);
				}
#else
				newDetections = filter->yoloModel->inference(inferenceFrame);
#endif
#else
				newDetections = filter->yoloModel->inference(inferenceFrame);
#endif
			}
		}

		// 记录推理时间
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
			endTime - inferenceStartTime).count();

		// 坐标转换（如果有裁切区域且有检测结果）
		if (!newDetections.empty() && cropWidth > 0 && cropHeight > 0 && 
			(cropX > 0 || cropY > 0 || cropWidth < fullWidth || cropHeight < fullHeight)) {
			for (auto& det : newDetections) {
				float pixelX = det.x * cropWidth + cropX;
				float pixelY = det.y * cropHeight + cropY;
				float pixelW = det.width * cropWidth;
				float pixelH = det.height * cropHeight;
				float pixelCenterX = det.centerX * cropWidth + cropX;
				float pixelCenterY = det.centerY * cropHeight + cropY;
				
				det.x = pixelX / fullWidth;
				det.y = pixelY / fullHeight;
				det.width = pixelW / fullWidth;
				det.height = pixelH / fullHeight;
				det.centerX = pixelCenterX / fullWidth;
				det.centerY = pixelCenterY / fullHeight;
			}
		}

		// 处理目标追踪（如果有检测结果或已有追踪目标）
		if (!newDetections.empty() || !filter->trackedTargets.empty()) {
			std::vector<Detection> trackedDetections;
			{
				std::lock_guard<std::mutex> trackLock(filter->trackedTargetsMutex);
				
				std::vector<Detection>& trackedTargets = filter->trackedTargets;
				
				if (trackedTargets.empty()) {
					for (auto& det : newDetections) {
						det.trackId = filter->nextTrackId++;
						det.lostFrames = 0;
						trackedDetections.push_back(det);
					}
				} else {
					int n = static_cast<int>(newDetections.size());
					int m = static_cast<int>(trackedTargets.size());
					
					std::vector<std::vector<float>> costMatrix(n, std::vector<float>(m, 1.0f));
					
					for (int i = 0; i < n; ++i) {
						cv::Rect2f detBox(
							newDetections[i].x,
							newDetections[i].y,
							newDetections[i].width,
							newDetections[i].height
						);
						cv::Point2f detCenter(newDetections[i].centerX, newDetections[i].centerY);
						
						for (int j = 0; j < m; ++j) {
							cv::Rect2f trackBox(
								trackedTargets[j].x,
								trackedTargets[j].y,
								trackedTargets[j].width,
								trackedTargets[j].height
							);
							cv::Point2f trackCenter(trackedTargets[j].centerX, trackedTargets[j].centerY);
							
							costMatrix[i][j] = HungarianAlgorithm::calculateFusedDistance(
								detBox, trackBox, detCenter, trackCenter,
								filter->trackingWeightIou,
								filter->trackingWeightCenter,
								filter->trackingWeightAspect,
								filter->trackingWeightArea);
						}
					}
					
					std::vector<int> assignment = HungarianAlgorithm::solve(costMatrix);
					
					std::vector<bool> detectionMatched(n, false);
					std::vector<bool> trackMatched(m, false);
					
					for (int i = 0; i < n; ++i) {
						int j = assignment[i];
						if (j >= 0 && j < m && costMatrix[i][j] < (1.0f - filter->iouThreshold)) {
							newDetections[i].trackId = trackedTargets[j].trackId;
							newDetections[i].lostFrames = 0;
							trackedDetections.push_back(newDetections[i]);
							detectionMatched[i] = true;
							trackMatched[j] = true;
						}
					}
					
					for (int i = 0; i < n; ++i) {
						if (!detectionMatched[i]) {
							newDetections[i].trackId = filter->nextTrackId++;
							newDetections[i].lostFrames = 0;
							trackedDetections.push_back(newDetections[i]);
						}
					}
					
					for (int j = 0; j < m; ++j) {
						if (!trackMatched[j]) {
							trackedTargets[j].lostFrames++;
							if (trackedTargets[j].lostFrames <= filter->maxLostFrames) {
								trackedDetections.push_back(trackedTargets[j]);
							} else {
								// 目标丢失超过阈值，添加到重识别缓冲区
								std::lock_guard<std::mutex> lostLock(filter->lostTargetsMutex);
								LostTarget lost;
								lost.trackId = trackedTargets[j].trackId;
								lost.x = trackedTargets[j].x;
								lost.y = trackedTargets[j].y;
								lost.width = trackedTargets[j].width;
								lost.height = trackedTargets[j].height;
								lost.centerX = trackedTargets[j].centerX;
								lost.centerY = trackedTargets[j].centerY;
								lost.lostFrames = 0;
								lost.lostTime = std::chrono::steady_clock::now();
								
								// 检查是否已存在相同trackId，更新而非添加
								bool found = false;
								for (auto& existing : filter->lostTargets) {
									if (existing.trackId == lost.trackId) {
										existing = lost;
										found = true;
										break;
									}
								}
								if (!found) {
									filter->lostTargets.push_back(lost);
								}
							}
						}
					}
					
					// 尝试重识别丢失目标
					{
						std::lock_guard<std::mutex> lostLock(filter->lostTargetsMutex);
						auto now = std::chrono::steady_clock::now();
						for (auto it = filter->lostTargets.begin(); it != filter->lostTargets.end(); ) {
							// 检查重识别时间是否超时
							auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->lostTime).count();
							if (elapsed > filter->maxReidentifyFrames * 33) {  // 约33ms每帧
								it = filter->lostTargets.erase(it);
								continue;
							}
							
							// 尝试与未匹配的检测进行重识别
							for (int i = 0; i < n; ++i) {
								if (detectionMatched[i]) continue;
								
								float dx = newDetections[i].centerX - it->centerX;
								float dy = newDetections[i].centerY - it->centerY;
								float centerDist = std::sqrt(dx * dx + dy * dy);
								
								// 中心点距离很近，认为是同一目标
								if (centerDist < filter->reidentifyCenterThreshold) {
									newDetections[i].trackId = it->trackId;
									newDetections[i].lostFrames = 0;
									trackedDetections.push_back(newDetections[i]);
									detectionMatched[i] = true;
									it = filter->lostTargets.erase(it);
									break;
								}
							}
							
							if (it != filter->lostTargets.end()) {
								++it;
							}
						}
					}
				}
				
				filter->trackedTargets = std::move(trackedDetections);
			}

			// 应用检测框平滑（如果有追踪目标）
			if (filter->detectionSmoothingEnabled && !filter->trackedTargets.empty()) {
				std::lock_guard<std::mutex> smoothLock(filter->smoothedDetectionsMutex);
				for (auto& det : filter->trackedTargets) {
					auto [it, inserted] = filter->smoothedDetections.try_emplace(det.trackId);
					it->second.update(
						det.x, det.y, det.width, det.height, filter->detectionSmoothingAlpha);
					
					det.x = it->second.x;
					det.y = it->second.y;
					det.width = it->second.width;
					det.height = it->second.height;
				}
			}
		}

		// 写入输出缓冲区（总是执行，确保主线程能收到空结果通知）
		{
			std::lock_guard<std::mutex> trackLock(filter->trackedTargetsMutex);
			filter->outputDetections[readIdx] = filter->trackedTargets;
			filter->outputTrackedTargets[readIdx] = filter->trackedTargets;
		}
		filter->outputFrameWidths[readIdx] = fullWidth;
		filter->outputFrameHeights[readIdx] = fullHeight;
		filter->outputCropX[readIdx] = cropX;
		filter->outputCropY[readIdx] = cropY;

		// 更新统计信息
		filter->inferenceCount++;
		filter->avgInferenceTimeMs = (filter->avgInferenceTimeMs * (filter->inferenceCount - 1) + duration) / filter->inferenceCount;
		filter->framesInferred.fetch_add(1, std::memory_order_relaxed);

		// 标记输出缓冲区为完成，递增序列号
		int64_t currentSeq = filter->outputSequence.fetch_add(1, std::memory_order_relaxed);
		filter->outputState[readIdx].store(1, std::memory_order_release);
		
		// 只有序列号更新时才更新就绪索引（防止重复消费）
		int64_t lastSeq = filter->lastConsumedSeq.load(std::memory_order_acquire);
		if (currentSeq > lastSeq) {
			filter->outputReadyIdx.store(readIdx, std::memory_order_release);
		}

		// 导出坐标（如果有检测结果）
		if (filter->exportCoordinates && !newDetections.empty()) {
			exportCoordinatesToFile(filter, fullWidth, fullHeight);
		}
	}

	obs_log(LOG_INFO, "[YOLO Detector] Async inference thread stopped");
}

static void renderDetectionBoxes(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	std::lock_guard<std::mutex> lock(filter->detectionsMutex);

	if (filter->detections.empty()) {
		return;
	}

	gs_effect_t *solid = filter->solidEffect;
	gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
	gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);

	for (const auto& det : filter->detections) {
		float x = det.x * frameWidth;
		float y = det.y * frameHeight;
		float w = det.width * frameWidth;
		float h = det.height * frameHeight;

		struct vec4 color;
		float r = ((filter->bboxColor >> 16) & 0xFF) / 255.0f;
		float g = ((filter->bboxColor >> 8) & 0xFF) / 255.0f;
		float b = (filter->bboxColor & 0xFF) / 255.0f;
		float a = ((filter->bboxColor >> 24) & 0xFF) / 255.0f;
		vec4_set(&color, r, g, b, a);
		gs_effect_set_vec4(colorParam, &color);

		gs_render_start(true);
		gs_vertex2f(x, y);
		gs_vertex2f(x + w, y);
		gs_vertex2f(x + w, y + h);
		gs_vertex2f(x, y + h);
		gs_vertex2f(x, y);
		gs_render_stop(GS_LINES);
	}

	gs_technique_end_pass(tech);
	gs_technique_end(tech);
}

static void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (!filter->showFOV) {
		return;
	}

	gs_effect_t *solid = filter->solidEffect;
	gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
	gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

	float centerX = frameWidth / 2.0f;
	float centerY = frameHeight / 2.0f;
	float radius = static_cast<float>(filter->fovRadius);

	struct vec4 color;
	float r = ((filter->fovColor >> 16) & 0xFF) / 255.0f;
	float g = ((filter->fovColor >> 8) & 0xFF) / 255.0f;
	float b = (filter->fovColor & 0xFF) / 255.0f;
	float a = ((filter->fovColor >> 24) & 0xFF) / 255.0f;
	vec4_set(&color, r, g, b, a);

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);
	gs_effect_set_vec4(colorParam, &color);

	gs_render_start(true);

	gs_vertex2f(centerX - radius, centerY);
	gs_vertex2f(centerX + radius, centerY);

	gs_vertex2f(centerX, centerY - radius);
	gs_vertex2f(centerX, centerY + radius);

	gs_render_stop(GS_LINES);

	const int circleSegments = 64;
	gs_render_start(true);
	for (int i = 0; i <= circleSegments; ++i) {
		float angle = 2.0f * 3.1415926f * static_cast<float>(i) / static_cast<float>(circleSegments);
		float x = centerX + radius * cosf(angle);
		float y = centerY + radius * sinf(angle);
		gs_vertex2f(x, y);
	}
	gs_render_stop(GS_LINESTRIP);

	gs_technique_end_pass(tech);
	gs_technique_end(tech);
}

static void renderLabelsWithOpenCV(cv::Mat &image, yolo_detector_filter *filter)
{
	std::vector<Detection> detectionsCopy;
	{
		std::lock_guard<std::mutex> lock(filter->detectionsMutex);
		if (filter->detections.empty()) {
			return;
		}
		detectionsCopy = filter->detections;
	}

	int frameWidth = image.cols;
	int frameHeight = image.rows;
	int fontFace = cv::FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.5;
	int thickness = 2;
	int baseline = 0;

	for (const auto& det : detectionsCopy) {
		int x = static_cast<int>(det.x * frameWidth);
		int y = static_cast<int>(det.y * frameHeight);
		int w = static_cast<int>(det.width * frameWidth);
		int h = static_cast<int>(det.height * frameHeight);

		// 构建标签文本：类别ID(0-10) + 置信度(浮点数)
		char labelText[64];
		snprintf(labelText, sizeof(labelText), "%d: %.2f", det.classId, det.confidence);

		// 获取文本大小
		cv::Size textSize = cv::getTextSize(labelText, fontFace, fontScale, thickness, &baseline);

		// 绘制标签背景
		cv::Point textOrg(x, y - 5);
		cv::rectangle(image, 
			cv::Point(textOrg.x, textOrg.y - textSize.height - 5),
			cv::Point(textOrg.x + textSize.width + 10, textOrg.y + baseline),
			cv::Scalar(0, 0, 0, 200),
			-1);

		// 绘制文本
		cv::putText(image, labelText, 
			cv::Point(textOrg.x + 5, textOrg.y),
			fontFace, fontScale, 
			cv::Scalar(0, 255, 0, 255), 
			thickness);
	}
}

static void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (filter->coordinateOutputPath.empty()) {
		return;
	}

	std::lock_guard<std::mutex> lock(filter->detectionsMutex);

	try {
		std::ofstream file(filter->coordinateOutputPath);
		if (!file.is_open()) {
			obs_log(LOG_ERROR, "[YOLO Filter] Failed to open coordinate file: %s", 
					filter->coordinateOutputPath.c_str());
			return;
		}

		auto now = std::chrono::system_clock::now();
		auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
			now.time_since_epoch()
		).count();

		file << "{\n";
		file << "  \"timestamp\": " << timestamp << ",\n";
		file << "  \"frame_width\": " << frameWidth << ",\n";
		file << "  \"frame_height\": " << frameHeight << ",\n";
		file << "  \"detections\": [\n";

		for (size_t i = 0; i < filter->detections.size(); ++i) {
			const auto& det = filter->detections[i];

			file << "    {\n";
			file << "      \"class_id\": " << det.classId << ",\n";
			file << "      \"class_name\": \"" << det.className << "\",\n";
			file << "      \"confidence\": " << det.confidence << ",\n";
			file << "      \"bbox\": {\n";
			file << "        \"x\": " << (det.x * frameWidth) << ",\n";
			file << "        \"y\": " << (det.y * frameHeight) << ",\n";
			file << "        \"width\": " << (det.width * frameWidth) << ",\n";
			file << "        \"height\": " << (det.height * frameHeight) << "\n";
			file << "      },\n";
			file << "      \"center\": {\n";
			file << "        \"x\": " << (det.centerX * frameWidth) << ",\n";
			file << "        \"y\": " << (det.centerY * frameHeight) << "\n";
			file << "      },\n";
			file << "      \"track_id\": " << det.trackId << "\n";
			file << "    }";

			if (i < filter->detections.size() - 1) {
				file << ",";
			}
			file << "\n";
		}

		file << "  ]\n";
		file << "}\n";

		file.close();

	} catch (const std::exception& e) {
		obs_log(LOG_ERROR, "[YOLO Filter] Error exporting coordinates: %s", e.what());
	}
}

void *yolo_detector_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "[YOLO Detector] Filter created");
	try {
		// Create the instance as a shared_ptr
		auto instance = std::make_shared<yolo_detector_filter>();

		instance->source = source;
		instance->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
		instance->stagesurface = nullptr;

		instance->inferenceRunning = false;
		instance->frameCounter = 0;
		instance->inferenceFrameWidth = 0;
    instance->inferenceFrameHeight = 0;
    instance->cropOffsetX = 0;
    instance->cropOffsetY = 0;
    instance->totalFrames = 0;
		instance->inferenceCount = 0;
		instance->avgInferenceTimeMs = 0.0;
		instance->isInferencing = false;
		instance->lastFpsTime = std::chrono::high_resolution_clock::now();
		instance->fpsFrameCount = 0;
		instance->currentFps = 0.0;
		instance->nextTrackId = 0;
		instance->maxLostFrames = 10;
		instance->iouThreshold = 0.3f;
		instance->threadPoolRunning = true;

		obs_enter_graphics();
		instance->solidEffect = obs_get_base_effect(OBS_EFFECT_SOLID);
		obs_leave_graphics();

#ifdef _WIN32
		instance->showFloatingWindow = false;
		instance->floatingWindowWidth = 640;
		instance->floatingWindowHeight = 480;
		instance->floatingWindowX = 0;
		instance->floatingWindowY = 0;
		instance->floatingWindowDragging = false;
		instance->floatingWindowHandle = nullptr;

	instance->showPidDebugWindow = false;
	instance->pidDebugWindowHandle = nullptr;
	instance->pidDebugWindowWidth = 700;   // 增宽以容纳积分仪表盘+占比条
	instance->pidDebugWindowHeight = 540;  // 增高以适配 6 区域布局(每区 90px)
		instance->pidDebugWindowX = 0;
		instance->pidDebugWindowY = 0;
		instance->pidDebugWindowDragging = false;

		// 动态FOV参数初始化
		instance->fovRadius = 0;  // 初始化为0，确保第一次update时触发更新
		instance->dynamicFovShrinkPercent = 0.5f;
		instance->dynamicFovTransitionTime = 200.0f;
		instance->currentFovRadius = 100.0f;  // 将在update中设置正确值
		instance->isFovTransitioning = false;
		instance->fovTransitionStartRadius = 100.0f;
		instance->fovTransitionEndRadius = 100.0f;

		for (int i = 0; i < 5; i++) {
			instance->mouseConfigs[i] = yolo_detector_filter::MouseControlConfig();
		}
		instance->currentConfigIndex = 0;
		instance->mouseController = MouseControllerFactory::createController(ControllerType::WindowsAPI, "", 0);
		setupPidDataCallback(instance.get());

		instance->configName = "";
		instance->configList = "";

		// ChrisPID参数初始化
		instance->chrisKp = 0.45f;
		instance->chrisKi = 0.02f;
		instance->chrisKd = 0.04f;
		instance->chrisPredWeightX = 0.5f;
		instance->chrisPredWeightY = 0.1f;
		instance->chrisInitScale = 0.6f;
		instance->chrisRampTime = 0.5f;
		instance->chrisOutputMax = 150.0f;
		instance->chrisIMax = 100.0f;
		instance->chrisDFilterAlpha = 0.3f;
		
#ifdef _WIN32
		// GPU纹理推理初始化
		instance->useGpuTextureInference = false;
		instance->cachedD3D11Texture = nullptr;
		instance->gpuTextureWidth = 0;
		instance->gpuTextureHeight = 0;
#endif
		
#endif

		// 强制关闭悬浮窗（每次启动OBS时）
		obs_data_set_bool(settings, "show_floating_window", false);
		obs_data_set_bool(settings, "show_pid_debug_window", false);

		// Create pointer to shared_ptr for the update call
		auto ptr = new std::shared_ptr<yolo_detector_filter>(instance);
		yolo_detector_filter_update(ptr, settings);

		// Start thread pool
		for (int i = 0; i < instance->THREAD_POOL_SIZE; ++i) {
			instance->threadPool.emplace_back(threadPoolWorker, instance.get());
		}

		// Start inference thread
		instance->inferenceRunning = true;
		instance->inferenceThread = std::thread(inferenceThreadWorker, instance.get());

		return ptr;
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "[YOLO Detector] Failed to create filter: %s", e.what());
		return nullptr;
	}
}

void yolo_detector_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "[YOLO Detector] Filter destroyed");

	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return;
	}

	auto &tf = *ptr;
	if (!tf) {
		delete ptr;
		return;
	}

	// Mark as disabled to prevent further processing
	tf->isDisabled = true;

	// Stop inference thread
	tf->inferenceRunning = false;
	if (tf->inferenceThread.joinable()) {
		tf->inferenceThread.join();
	}

	// Stop thread pool
	tf->threadPoolRunning = false;
	tf->taskCondition.notify_all();
	for (auto& thread : tf->threadPool) {
		if (thread.joinable()) {
			thread.join();
		}
	}

#ifdef _WIN32
	// Destroy floating window
	destroyFloatingWindow(tf.get());
	
	// 保存悬浮窗关闭状态
	obs_data_t *settings = obs_source_get_settings(tf->source);
	if (settings) {
		obs_data_set_bool(settings, "show_floating_window", false);
		obs_data_release(settings);
	}
#endif

	// Clean up graphics resources
	obs_enter_graphics();
	if (tf->texrender) {
		gs_texrender_destroy(tf->texrender);
		tf->texrender = nullptr;
	}
	if (tf->stagesurface) {
		gs_stagesurface_destroy(tf->stagesurface);
		tf->stagesurface = nullptr;
	}
	obs_leave_graphics();

	delete ptr;
}

void yolo_detector_filter_activate(void *data)
{
	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<yolo_detector_filter> tf = *ptr;
	if (!tf) {
		return;
	}

	obs_log(LOG_INFO, "[YOLO Detector] Filter activated");
}

void yolo_detector_filter_deactivate(void *data)
{
	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<yolo_detector_filter> tf = *ptr;
	if (!tf) {
		return;
	}

	obs_log(LOG_INFO, "[YOLO Detector] Filter deactivated");
}

void yolo_detector_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<yolo_detector_filter> tf = *ptr;
	if (!tf || tf->isDisabled) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	tf->totalFrames++;
	tf->frameCounter++;
	tf->fpsFrameCount++;

	auto now = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - tf->lastFpsTime).count();
	if (elapsed >= 1000) {
		tf->currentFps = (double)tf->fpsFrameCount * 1000.0 / (double)elapsed;
		tf->fpsFrameCount = 0;
		tf->lastFpsTime = now;
	}

	// === 四缓冲区：消费推理结果（带序列号防重复） ===
	int readyIdx = tf->outputReadyIdx.load(std::memory_order_acquire);
	if (readyIdx >= 0 && readyIdx < tf->BUFFER_COUNT && 
		tf->outputState[readyIdx].load(std::memory_order_acquire) == 1) {
		
		// 检查是否是新的结果（防止重复消费）
		int64_t currentSeq = tf->outputSequence.load(std::memory_order_acquire);
		int64_t lastSeq = tf->lastConsumedSeq.load(std::memory_order_acquire);
		
		if (currentSeq > lastSeq) {
			std::lock_guard<std::mutex> detLock(tf->detectionsMutex);
			tf->detections = tf->outputDetections[readyIdx];
			
			std::lock_guard<std::mutex> sizeLock(tf->inferenceFrameSizeMutex);
			tf->inferenceFrameWidth = tf->outputFrameWidths[readyIdx];
			tf->inferenceFrameHeight = tf->outputFrameHeights[readyIdx];
			tf->cropOffsetX = tf->outputCropX[readyIdx];
			tf->cropOffsetY = tf->outputCropY[readyIdx];

			// 标记输出缓冲区为已消费，更新已消费序列号
			tf->outputState[readyIdx].store(0, std::memory_order_release);
			tf->lastConsumedSeq.store(currentSeq, std::memory_order_release);
			tf->framesConsumed.fetch_add(1, std::memory_order_relaxed);
			
			// 更新最后有结果的时刻
			auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now().time_since_epoch()).count();
			tf->lastResultTimestamp.store(nowMs, std::memory_order_relaxed);
		} else {
			// 没有新结果，检查是否超时需要清空
			auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now().time_since_epoch()).count();
			int64_t lastTs = tf->lastResultTimestamp.load(std::memory_order_acquire);
			
			// 超过500ms没有新结果，自动清空检测框
			if (nowMs - lastTs > 500 && !tf->detections.empty()) {
				std::lock_guard<std::mutex> detLock(tf->detectionsMutex);
				tf->detections.clear();
			}
		}
	}

#ifdef _WIN32
	auto getActiveConfig = [&tf]() -> int {
		for (int i = 0; i < 5; i++) {
			if (tf->mouseConfigs[i].enabled) {
				// 持续自瞄模式：直接返回该配置
				if (tf->mouseConfigs[i].continuousAimEnabled) {
					return i;
				}
				// 热键模式：检查热键是否按下
				if ((GetAsyncKeyState(tf->mouseConfigs[i].hotkey) & 0x8000) != 0) {
					return i;
				}
			}
		}
		return -1;
	};

	auto applyConfigToController = [&tf](int configIndex) {
		if (configIndex < 0 || configIndex >= 5) return;
		
		const auto& cfg = tf->mouseConfigs[configIndex];
		
		ControllerType newType = static_cast<ControllerType>(cfg.controllerType);
		if (!tf->mouseController || tf->mouseController->getControllerType() != newType) {
			tf->mouseController = MouseControllerFactory::createController(newType, cfg.makcuPort, cfg.makcuBaudRate);
			setupPidDataCallback(tf.get());
		}

		MouseControllerConfig mcConfig;
		mcConfig.enableMouseControl = true;
		mcConfig.hotkeyVirtualKey = cfg.hotkey;
		mcConfig.fovRadiusPixels = tf->useDynamicFOV ? static_cast<int>(tf->currentFovRadius) : tf->fovRadius;
		mcConfig.pidPMin = cfg.pMin;
		mcConfig.pidPMax = cfg.pMax;
		mcConfig.pidPSlope = cfg.pSlope;
		mcConfig.pidD = cfg.d;
		mcConfig.pidI = cfg.i;
		mcConfig.aimSmoothingX = cfg.aimSmoothingX;
		mcConfig.aimSmoothingY = cfg.aimSmoothingY;
		mcConfig.maxPixelMove = cfg.maxPixelMove;
		mcConfig.deadZonePixels = cfg.deadZonePixels;
		mcConfig.sourceCanvasPosX = 0.0f;
		mcConfig.sourceCanvasPosY = 0.0f;
		mcConfig.sourceCanvasScaleX = 1.0f;
		mcConfig.sourceCanvasScaleY = 1.0f;
		mcConfig.sourceWidth = obs_source_get_base_width(tf->source);
		mcConfig.sourceHeight = obs_source_get_base_height(tf->source);
		mcConfig.screenOffsetX = cfg.screenOffsetX;
		mcConfig.screenOffsetY = cfg.screenOffsetY;
		mcConfig.screenWidth = cfg.screenWidth;
		mcConfig.screenHeight = cfg.screenHeight;
		mcConfig.targetYOffset = cfg.targetYOffset;
		mcConfig.derivativeFilterAlpha = cfg.derivativeFilterAlpha;
		mcConfig.advTargetThreshold = cfg.advTargetThreshold;
		mcConfig.advMinCoefficient = cfg.advMinCoefficient;
		mcConfig.advMaxCoefficient = cfg.advMaxCoefficient;
		mcConfig.advTransitionSharpness = cfg.advTransitionSharpness;
		mcConfig.advTransitionMidpoint = cfg.advTransitionMidpoint;
		mcConfig.advOutputSmoothing = cfg.advOutputSmoothing;
		mcConfig.advSpeedFactor = cfg.advSpeedFactor;
		mcConfig.useOneEuroFilter = cfg.useOneEuroFilter;
		mcConfig.oneEuroMinCutoff = cfg.oneEuroMinCutoff;
		mcConfig.oneEuroBeta = cfg.oneEuroBeta;
		mcConfig.oneEuroDCutoff = cfg.oneEuroDCutoff;
		mcConfig.controllerType = static_cast<ControllerType>(cfg.controllerType);
		mcConfig.makcuPort = cfg.makcuPort;
		mcConfig.makcuBaudRate = cfg.makcuBaudRate;
		mcConfig.yUnlockEnabled = cfg.enableYAxisUnlock;
		mcConfig.yUnlockDelayMs = cfg.yAxisUnlockDelay;
		mcConfig.autoTriggerEnabled = cfg.enableAutoTrigger;
		mcConfig.autoTriggerRadius = cfg.triggerRadius;
		mcConfig.autoTriggerCooldownMs = cfg.triggerCooldown;
		mcConfig.autoTriggerFireDelay = cfg.triggerFireDelay;
		mcConfig.autoTriggerFireDuration = cfg.triggerFireDuration;
		mcConfig.autoTriggerInterval = cfg.triggerInterval;
		mcConfig.autoTriggerDelayRandomEnabled = cfg.enableTriggerDelayRandom;
		mcConfig.autoTriggerDelayRandomMin = cfg.triggerDelayRandomMin;
		mcConfig.autoTriggerDelayRandomMax = cfg.triggerDelayRandomMax;
		mcConfig.autoTriggerDurationRandomEnabled = cfg.enableTriggerDurationRandom;
		mcConfig.autoTriggerDurationRandomMin = cfg.triggerDurationRandomMin;
		mcConfig.autoTriggerDurationRandomMax = cfg.triggerDurationRandomMax;
		mcConfig.autoTriggerMoveCompensation = cfg.triggerMoveCompensation;
		mcConfig.targetSwitchDelayMs = tf->targetSwitchDelayMs;
		mcConfig.targetSwitchTolerance = tf->targetSwitchTolerance;
		// 新功能参数
		mcConfig.integralLimit = cfg.integralLimit;
		mcConfig.integralSeparationThreshold = cfg.integralSeparationThreshold;
		mcConfig.integralDeadZone = cfg.integralDeadZone;
		mcConfig.pGainRampInitialScale = cfg.pGainRampInitialScale;
		mcConfig.pGainRampDuration = cfg.pGainRampDuration;
		mcConfig.predictionWeightX = cfg.predictionWeightX;
		mcConfig.predictionWeightY = cfg.predictionWeightY;
		// 持续自瞄和自动压枪参数
		mcConfig.continuousAimEnabled = cfg.continuousAimEnabled;
		mcConfig.autoRecoilControlEnabled = cfg.autoRecoilControlEnabled;
		mcConfig.recoilStrength = cfg.recoilStrength;
		mcConfig.recoilSpeed = cfg.recoilSpeed;
		mcConfig.recoilPidGainScale = cfg.recoilPidGainScale;
		// DerivativePredictor参数
		mcConfig.useDerivativePredictor = cfg.useDerivativePredictor;
		mcConfig.predictionWeightX = cfg.predictionWeightX;
		mcConfig.predictionWeightY = cfg.predictionWeightY;
		// 贝塞尔曲线移动参数
		mcConfig.enableBezierMovement = cfg.enableBezierMovement;
		mcConfig.bezierCurvature = cfg.bezierCurvature;
		mcConfig.bezierRandomness = cfg.bezierRandomness;
		// 算法选择（使用全局设置）
		mcConfig.algorithmType = static_cast<AlgorithmType>(tf->algorithmTypeGlobal);
		// 标准PID参数（使用全局设置）
		mcConfig.stdKp = tf->stdKpGlobal;
		mcConfig.stdKi = tf->stdKiGlobal;
		mcConfig.stdKd = tf->stdKdGlobal;
		mcConfig.stdOutputLimit = tf->stdOutputLimitGlobal;
		mcConfig.stdDeadZone = tf->stdDeadZoneGlobal;
		mcConfig.stdIntegralLimit = tf->stdIntegralLimitGlobal;
		mcConfig.stdIntegralDeadzone = tf->stdIntegralDeadzoneGlobal;
		mcConfig.stdIntegralThreshold = tf->stdIntegralThresholdGlobal;
		mcConfig.stdIntegralRate = tf->stdIntegralRateGlobal;
		mcConfig.stdDerivativeFilterAlpha = tf->stdDerivativeFilterAlphaGlobal;
		mcConfig.stdSmoothingX = 0.7f;
		mcConfig.stdSmoothingY = 0.5f;
		// ChrisPID参数（使用全局设置）
		mcConfig.chrisKp = tf->chrisKp;
		mcConfig.chrisKi = tf->chrisKi;
		mcConfig.chrisKd = tf->chrisKd;
		mcConfig.chrisPredWeightX = tf->chrisPredWeightX;
		mcConfig.chrisPredWeightY = tf->chrisPredWeightY;
		mcConfig.chrisInitScale = tf->chrisInitScale;
		mcConfig.chrisRampTime = tf->chrisRampTime;
		mcConfig.chrisOutputMax = tf->chrisOutputMax;
		mcConfig.chrisIMax = tf->chrisIMax;
		mcConfig.chrisDFilterAlpha = tf->chrisDFilterAlpha;
		mcConfig.dynamicKp = tf->dynamicKp;
		mcConfig.dynamicKi = tf->dynamicKi;
		mcConfig.dynamicKd = tf->dynamicKd;
		mcConfig.dynamicTargetThreshold = tf->dynamicTargetThreshold;
		mcConfig.dynamicSpeedMultiplier = tf->dynamicSpeedMultiplier;
		mcConfig.dynamicMinCoefficient = tf->dynamicMinCoefficient;
		mcConfig.dynamicMaxCoefficient = tf->dynamicMaxCoefficient;
		mcConfig.dynamicTransitionSharpness = tf->dynamicTransitionSharpness;
		mcConfig.dynamicTransitionMidpoint = tf->dynamicTransitionMidpoint;
		mcConfig.dynamicMinDataPoints = tf->dynamicMinDataPoints;
		mcConfig.dynamicErrorTolerance = tf->dynamicErrorTolerance;
		mcConfig.dynamicSmoothingFactor = tf->dynamicSmoothingFactor;
		tf->mouseController->updateConfig(mcConfig);
	};

	// 缓动函数 - ease-out cubic
	auto easeOutCubic = [](float t) -> float {
		return 1.0f - powf(1.0f - t, 3.0f);
	};

	// 更新FOV过渡
	auto updateFovTransition = [&tf, &easeOutCubic]() {
		if (!tf->isFovTransitioning) {
			return;
		}
		
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - tf->fovTransitionStartTime).count();
		float transitionTime = tf->dynamicFovTransitionTime;
		
		if (transitionTime <= 0.0f) {
			// 立即切换
			tf->currentFovRadius = tf->fovTransitionEndRadius;
			tf->isFovTransitioning = false;
			return;
		}
		
		float progress = static_cast<float>(elapsed) / transitionTime;
		if (progress >= 1.0f) {
			// 过渡完成
			tf->currentFovRadius = tf->fovTransitionEndRadius;
			tf->isFovTransitioning = false;
		} else {
			// 使用缓动函数计算当前半径
			float easedProgress = easeOutCubic(progress);
			tf->currentFovRadius = tf->fovTransitionStartRadius + 
				(tf->fovTransitionEndRadius - tf->fovTransitionStartRadius) * easedProgress;
		}
	};

	// 开始FOV过渡
	auto startFovTransition = [&tf](float targetRadius) {
		if (tf->currentFovRadius == targetRadius && !tf->isFovTransitioning) {
			return;  // 已经是目标半径，无需过渡
		}
		
		tf->fovTransitionStartRadius = tf->currentFovRadius;
		tf->fovTransitionEndRadius = targetRadius;
		tf->fovTransitionStartTime = std::chrono::steady_clock::now();
		tf->isFovTransitioning = true;
	};

	// 更新FOV过渡
	updateFovTransition();

	// 动态FOV切换逻辑
	if (tf->useDynamicFOV) {
		std::vector<Detection> detectionsCopy;
		{
			std::lock_guard<std::mutex> lock(tf->detectionsMutex);
			detectionsCopy = tf->detections;
		}

		int activeConfig = getActiveConfig();
		float shrinkedRadius = static_cast<float>(tf->fovRadius) * tf->dynamicFovShrinkPercent;
		
		float centerX = 0.5f;
		float centerY = 0.5f;
		float currentFOVRadius = tf->currentFovRadius / static_cast<float>(obs_source_get_base_width(tf->source));
		
		bool hasTargetInCurrentFOV = false;
		for (const auto& det : detectionsCopy) {
			float dx = det.centerX - centerX;
			float dy = det.centerY - centerY;
			float distance = sqrtf(dx * dx + dy * dy);
			if (distance <= currentFOVRadius) {
				hasTargetInCurrentFOV = true;
				break;
			}
		}
		
		bool shouldShrinkFOV = (activeConfig >= 0) && hasTargetInCurrentFOV;
		
		if (!tf->isInFOV2Mode) {
			if (shouldShrinkFOV) {
				tf->isInFOV2Mode = true;
				startFovTransition(shrinkedRadius);
			}
		} else {
			if (!shouldShrinkFOV) {
				tf->isInFOV2Mode = false;
				startFovTransition(static_cast<float>(tf->fovRadius));
			}
		}

		// 更新鼠标控制器 - 无论是否在推理，只要有鼠标控制器就调用tick()确保能释放自动扳机
		if (tf->mouseController) {
			if (tf->isInferencing) {
				int activeConfig = getActiveConfig();
				if (activeConfig >= 0) {
					applyConfigToController(activeConfig);
					int frameWidth = 0, frameHeight = 0, cropX = 0, cropY = 0;
					{
						std::lock_guard<std::mutex> lock(tf->inferenceFrameSizeMutex);
						frameWidth = tf->inferenceFrameWidth;
						frameHeight = tf->inferenceFrameHeight;
						cropX = tf->cropOffsetX;
						cropY = tf->cropOffsetY;
					}
					tf->mouseController->setDetectionsWithFrameSize(detectionsCopy, frameWidth, frameHeight, cropX, cropY);
					tf->mouseController->tick();
				} else {
					MouseControllerConfig mcConfig;
					mcConfig.enableMouseControl = false;
					tf->mouseController->updateConfig(mcConfig);
					tf->mouseController->tick();
				}
			} else {
				// 即使不在推理，也要确保自动扳机被释放
				MouseControllerConfig mcConfig;
				mcConfig.enableMouseControl = false;
				tf->mouseController->updateConfig(mcConfig);
				tf->mouseController->tick();
			}
		}
	} else {
		// 不使用动态FOV，正常处理 - 无论是否在推理，只要有鼠标控制器就调用tick()确保能释放自动扳机
		if (tf->mouseController) {
			if (tf->isInferencing) {
				int activeConfig = getActiveConfig();
				if (activeConfig >= 0) {
					applyConfigToController(activeConfig);
					std::vector<Detection> detectionsCopy;
					int frameWidth = 0, frameHeight = 0, cropX = 0, cropY = 0;
					{
						std::lock_guard<std::mutex> lock(tf->detectionsMutex);
						detectionsCopy = tf->detections;
					}
					{
						std::lock_guard<std::mutex> lock(tf->inferenceFrameSizeMutex);
						frameWidth = tf->inferenceFrameWidth;
						frameHeight = tf->inferenceFrameHeight;
						cropX = tf->cropOffsetX;
						cropY = tf->cropOffsetY;
					}
					tf->mouseController->setDetectionsWithFrameSize(detectionsCopy, frameWidth, frameHeight, cropX, cropY);
					tf->mouseController->tick();
				} else {
					MouseControllerConfig mcConfig;
					mcConfig.enableMouseControl = false;
					tf->mouseController->updateConfig(mcConfig);
					tf->mouseController->tick();
				}
			} else {
				// 即使不在推理，也要确保自动扳机被释放
				MouseControllerConfig mcConfig;
				mcConfig.enableMouseControl = false;
				tf->mouseController->updateConfig(mcConfig);
				tf->mouseController->tick();
			}
		}
	}
#endif
}

void yolo_detector_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
	if (!ptr) {
		return;
	}

	std::shared_ptr<yolo_detector_filter> tf = *ptr;
	if (!tf || tf->isDisabled) {
		if (tf && tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	obs_source_t *target = obs_filter_get_target(tf->source);
	if (!target) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width = obs_source_get_base_width(target);
	uint32_t height = obs_source_get_base_height(target);

	if (width == 0 || height == 0) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	bool needShowLabels = tf->showLabel || tf->showConfidence;
	bool needCapture = tf->showFloatingWindow || tf->isInferencing || needShowLabels;

	// 捕获原始帧（用于推理、悬浮窗和标签显示）
	cv::Mat originalImage;
	int originalWidth = width;
	int originalHeight = height;
	int cropOffsetX = 0;
	int cropOffsetY = 0;
	
	if (needCapture) {
		obs_enter_graphics();
		gs_texrender_reset(tf->texrender);
		if (gs_texrender_begin(tf->texrender, width, height)) {
			struct vec4 background;
			vec4_zero(&background);
			gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
			gs_ortho(0.0f, (float)width, 0.0f, (float)height, -100.0f, 100.0f);
			gs_blend_state_push();
			gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
			obs_source_video_render(target);
			gs_blend_state_pop();
			gs_texrender_end(tf->texrender);

			gs_texture_t *tex = gs_texrender_get_texture(tf->texrender);
			if (tex) {
#if defined(HAVE_CUDA) || defined(HAVE_ONNXRUNTIME_DML_EP)
				// GPU纹理推理路径（CUDA 或 DML）
				if (tf->useGpuTextureInference && tf->yoloModel) {
					bool canUseGpuTexture = false;
#ifdef HAVE_CUDA
					canUseGpuTexture = canUseGpuTexture || tf->yoloModel->isGpuTextureSupported();
#endif
#ifdef HAVE_ONNXRUNTIME_DML_EP
					canUseGpuTexture = canUseGpuTexture || tf->yoloModel->isDmlTextureSupported();
#endif
					if (canUseGpuTexture) {
						void* d3d11Texture = gs_texture_get_obj(tex);
						if (d3d11Texture) {
							ID3D11Texture2D* d3dTex = static_cast<ID3D11Texture2D*>(d3d11Texture);
							d3dTex->AddRef();
							
							if (tf->cachedD3D11Texture) {
								tf->cachedD3D11Texture->Release();
							}
							tf->cachedD3D11Texture = d3dTex;
							tf->gpuTextureWidth = width;
							tf->gpuTextureHeight = height;
						}
					}
				}
#endif
				
				if (!tf->stagesurface || 
				    gs_stagesurface_get_width(tf->stagesurface) != width || 
				    gs_stagesurface_get_height(tf->stagesurface) != height) {
					if (tf->stagesurface) {
						gs_stagesurface_destroy(tf->stagesurface);
					}
					tf->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
				}
				if (tf->stagesurface) {
					gs_stage_texture(tf->stagesurface, tex);
					uint8_t *video_data;
					uint32_t linesize;
					if (gs_stagesurface_map(tf->stagesurface, &video_data, &linesize)) {
						// 直接使用映射数据，避免克隆
						cv::Mat temp(height, width, CV_8UC4, video_data, linesize);
						
						// === 四缓冲区无锁帧提交 ===
						int currentWrite = tf->inputWriteIdx.load(std::memory_order_relaxed);
						bool submitted = false;

						// 计算裁切区域信息
						int frameCropX = 0, frameCropY = 0;
						int frameCropWidth = static_cast<int>(width), frameCropHeight = static_cast<int>(height);
						
						if (tf->useRegion) {
							frameCropX = std::max(0, tf->regionX);
							frameCropY = std::max(0, tf->regionY);
							frameCropWidth = std::min(tf->regionWidth, static_cast<int>(width) - frameCropX);
							frameCropHeight = std::min(tf->regionHeight, static_cast<int>(height) - frameCropY);
							if (frameCropWidth <= 0 || frameCropHeight <= 0) {
								frameCropX = 0;
								frameCropY = 0;
								frameCropWidth = width;
								frameCropHeight = height;
							}
						}

						for (int i = 0; i < tf->BUFFER_COUNT; i++) {
							int checkIdx = (currentWrite + i) % tf->BUFFER_COUNT;
							uint8_t expected = 0;  // 期望状态为空闲
							
							if (tf->bufferState[checkIdx].compare_exchange_strong(
								expected, 1, std::memory_order_acq_rel)) {
								// 成功获取空闲槽位，写入数据
								if (tf->inputFrames[checkIdx].rows != height || 
									tf->inputFrames[checkIdx].cols != width) {
									tf->inputFrames[checkIdx] = cv::Mat(height, width, CV_8UC4);
								}
								temp.copyTo(tf->inputFrames[checkIdx]);
								
								// 记录帧信息和裁切区域
								tf->inputFrameWidths[checkIdx] = width;
								tf->inputFrameHeights[checkIdx] = height;
								tf->inputCropX[checkIdx] = frameCropX;
								tf->inputCropY[checkIdx] = frameCropY;
								tf->inputCropWidth[checkIdx] = frameCropWidth;
								tf->inputCropHeight[checkIdx] = frameCropHeight;
								
								// 更新写入索引
								tf->inputWriteIdx.store((checkIdx + 1) % tf->BUFFER_COUNT, std::memory_order_release);
								tf->framesSubmitted.fetch_add(1, std::memory_order_relaxed);
								submitted = true;
								break;
							}
						}

						if (!submitted) {
							tf->framesDropped.fetch_add(1, std::memory_order_relaxed);
						}
						
						// 只在悬浮窗开启时才克隆裁切后的区域
						if (tf->showFloatingWindow) {
							int cropWidth = tf->floatingWindowWidth;
							int cropHeight = tf->floatingWindowHeight;
							int centerX = temp.cols / 2;
							int centerY = temp.rows / 2;
							cropOffsetX = std::max(0, centerX - cropWidth / 2);
							cropOffsetY = std::max(0, centerY - cropHeight / 2);
							int actualCropWidth = std::min(cropWidth, temp.cols - cropOffsetX);
							int actualCropHeight = std::min(cropHeight, temp.rows - cropOffsetY);
							
							if (actualCropWidth > 0 && actualCropHeight > 0) {
								originalImage = temp(cv::Rect(cropOffsetX, cropOffsetY, actualCropWidth, actualCropHeight)).clone();
							}
						}
						
						gs_stagesurface_unmap(tf->stagesurface);
					}
				}
			}
		}
		obs_leave_graphics();
	}

	// 开始滤镜处理 - 确保源画面绝对正常！
	if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	gs_blend_state_push();
	gs_reset_blend_state();

	// 只渲染源画面，100%保证不会黑屏！检测框、FOV、标签只在悬浮窗显示
	obs_source_process_filter_end(tf->source, obs_get_base_effect(OBS_EFFECT_DEFAULT), width, height);

	gs_blend_state_pop();

#ifdef _WIN32
	// 更新浮动窗口
	if (tf->showFloatingWindow && !originalImage.empty()) {
		// originalImage 已经是裁切后的区域，直接使用
		cv::Mat& croppedFrame = originalImage;

		size_t detectionCount = 0;
		std::vector<Detection> detectionsCopy;
		{
			std::lock_guard<std::mutex> lock(tf->detectionsMutex);
			detectionCount = tf->detections.size();
			detectionsCopy = tf->detections;
		}

		if (tf->showBBox) {
			int lineWidth = tf->bboxLineWidth;
			float r = ((tf->bboxColor >> 16) & 0xFF) / 255.0f;
			float g = ((tf->bboxColor >> 8) & 0xFF) / 255.0f;
			float b = (tf->bboxColor & 0xFF) / 255.0f;
			cv::Scalar bboxColor(b * 255, g * 255, r * 255, 255);

			for (const auto& det : detectionsCopy) {
				// 坐标转换：从原始帧坐标到裁切区域坐标
				int x = static_cast<int>(det.x * originalWidth) - cropOffsetX;
				int y = static_cast<int>(det.y * originalHeight) - cropOffsetY;
				int w = static_cast<int>(det.width * originalWidth);
				int h = static_cast<int>(det.height * originalHeight);
				
				if (x + w >= 0 && y + h >= 0 && x < croppedFrame.cols && y < croppedFrame.rows) {
					cv::rectangle(croppedFrame, 
						cv::Point(x, y), 
						cv::Point(x + w, y + h), 
						bboxColor, 
						lineWidth);
					
					// 绘制trackId
					if (tf->showTrackIdInFloatingWindow) {
						std::string idText = "ID:" + std::to_string(det.trackId);
						int baseline = 0;
						double fontScale = 0.5;
						int thickness = 1;
						cv::Size textSize = cv::getTextSize(idText, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
						cv::Point textOrg(x, y - 5);
						if (textOrg.y < textSize.height) {
							textOrg.y = y + textSize.height + 5;
						}
						cv::putText(croppedFrame, idText, textOrg,
							cv::FONT_HERSHEY_SIMPLEX, fontScale, bboxColor, thickness);
					}
				}
			}
		}

		// 如果需要显示 FOV
		if (tf->showFOV) {
			// FOV中心在裁切区域的中心
			float fovCenterX = croppedFrame.cols / 2.0f;
			float fovCenterY = croppedFrame.rows / 2.0f;
			// 使用动态FOV半径（如果启用）
			float fovRadius = tf->useDynamicFOV ? tf->currentFovRadius : static_cast<float>(tf->fovRadius);
			float crossLineLength = static_cast<float>(tf->fovCrossLineScale);
			
			float r = ((tf->fovColor >> 16) & 0xFF) / 255.0f;
			float g = ((tf->fovColor >> 8) & 0xFF) / 255.0f;
			float b = (tf->fovColor & 0xFF) / 255.0f;
			cv::Scalar fovColor(b * 255, g * 255, r * 255, 255);

			if (tf->showFOVCross) {
				cv::line(croppedFrame, 
					cv::Point(static_cast<int>(fovCenterX - crossLineLength), static_cast<int>(fovCenterY)),
					cv::Point(static_cast<int>(fovCenterX + crossLineLength), static_cast<int>(fovCenterY)),
					fovColor, tf->fovCrossLineThickness);
				cv::line(croppedFrame, 
					cv::Point(static_cast<int>(fovCenterX), static_cast<int>(fovCenterY - crossLineLength)),
					cv::Point(static_cast<int>(fovCenterX), static_cast<int>(fovCenterY + crossLineLength)),
					fovColor, tf->fovCrossLineThickness);
			}

			if (tf->showFOVCircle) {
				cv::circle(croppedFrame, 
					cv::Point(static_cast<int>(fovCenterX), static_cast<int>(fovCenterY)),
					static_cast<int>(fovRadius),
					fovColor, tf->fovCircleThickness);
			}
		}

		// 绘制从中心点到目标的连接线
		cv::Point centerPoint(croppedFrame.cols / 2, croppedFrame.rows / 2);
		
		// 使用绿色绘制连接线
		cv::Scalar lineColor(0, 255, 0, 255);
		int lineThickness = 1;
		
		for (const auto& det : detectionsCopy) {
			// 坐标转换：从原始帧坐标到裁切区域坐标
			int targetX = static_cast<int>(det.centerX * originalWidth) - cropOffsetX;
			int targetY = static_cast<int>(det.centerY * originalHeight) - cropOffsetY;
			cv::Point targetPoint(targetX, targetY);
			
			// 确保目标点在裁剪区域内
			if (targetX >= 0 && targetY >= 0 && targetX < croppedFrame.cols && targetY < croppedFrame.rows) {
				// 绘制从中心点到目标的连接线
				cv::line(croppedFrame, centerPoint, targetPoint, lineColor, lineThickness);
			}
		}

		// 如果需要显示标签和置信度，就在 croppedFrame 上绘制
		if (tf->showLabel || tf->showConfidence) {
			int fontFace = cv::FONT_HERSHEY_SIMPLEX;
			double fontScale = tf->labelFontScale;
			int thickness = 2;
			int baseline = 0;
			
			for (const auto& det : detectionsCopy) {
				// 坐标转换：从原始帧坐标到裁切区域坐标
				int x = static_cast<int>(det.x * originalWidth) - cropOffsetX;
				int y = static_cast<int>(det.y * originalHeight) - cropOffsetY;
				
				// 确保在裁剪区域内（包括文本绘制位置）
				int textY = y - 5;
				if (x >= 0 && textY >= 0 && x < croppedFrame.cols && y < croppedFrame.rows) {
					// 构建标签文本
					char labelText[64];
					snprintf(labelText, sizeof(labelText), "%d: %.2f", det.classId, det.confidence);
					
					// 只绘制文本，不绘制黑色背景
					cv::Point textOrg(x, textY);
					cv::putText(croppedFrame, labelText, 
						textOrg,
						fontFace, fontScale, 
						cv::Scalar(0, 255, 0, 255), 
						thickness);
				}
			}
		}

		// 绘制 FPS 和检测数量信息（无背景）
		int fontFace = cv::FONT_HERSHEY_SIMPLEX;
		double fontScale = 0.6;
		int thickness = 2;
		int baseline = 0;

		char fpsText[64];
		snprintf(fpsText, sizeof(fpsText), "FPS: %.0f", tf->currentFps);
		cv::Size fpsSize = cv::getTextSize(fpsText, fontFace, fontScale, thickness, &baseline);

		char detText[64];
		snprintf(detText, sizeof(detText), "Detected: %zu", detectionCount);
		cv::Size detSize = cv::getTextSize(detText, fontFace, fontScale, thickness, &baseline);

		cv::putText(croppedFrame, fpsText,
			cv::Point(10, 10 + fpsSize.height),
			fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);

		cv::putText(croppedFrame, detText,
			cv::Point(10, 10 + fpsSize.height + detSize.height + 10),
			fontFace, fontScale, cv::Scalar(0, 255, 255), thickness);

		// 如果裁切后的尺寸与悬浮窗尺寸不匹配，需要调整
		if (croppedFrame.cols != tf->floatingWindowWidth || croppedFrame.rows != tf->floatingWindowHeight) {
			cv::Mat resizedFrame;
			cv::resize(croppedFrame, resizedFrame, cv::Size(tf->floatingWindowWidth, tf->floatingWindowHeight));
			updateFloatingWindowFrame(tf.get(), resizedFrame);
		} else {
			updateFloatingWindowFrame(tf.get(), croppedFrame);
		}
		renderFloatingWindow(tf.get());

		if (tf->showPidDebugWindow && tf->pidDebugWindowHandle) {
			updatePidDebugWindow(tf.get());
		}
	}
#endif
}
