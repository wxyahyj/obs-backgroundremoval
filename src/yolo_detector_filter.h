#ifndef YOLO_DETECTOR_FILTER_H
#define YOLO_DETECTOR_FILTER_H

#include <obs-module.h>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "FilterData.h"
#include "models/ModelYOLO.h"
#include "models/Detection.h"
#include "KalmanFilter.hpp"
#include "CrosshairDetector.hpp"
#include "MouseControllerInterface.hpp"
#include "MouseControllerFactory.hpp"
#include "models/DmlPreprocessor.h"

struct LostTarget {
	int trackId;
	int classId = -1;
	float x, y, width, height;
	float centerX, centerY;
	float velX = 0.0f;
	float velY = 0.0f;
	int lostFrames;
	std::chrono::steady_clock::time_point lostTime;
};

struct yolo_detector_filter : public filter_data, public std::enable_shared_from_this<yolo_detector_filter> {
	yolo_detector_filter(const yolo_detector_filter&) = delete;
	yolo_detector_filter& operator=(const yolo_detector_filter&) = delete;
	yolo_detector_filter() = default;
	yolo_detector_filter(yolo_detector_filter&&) = default;
	yolo_detector_filter& operator=(yolo_detector_filter&&) = default;

	std::shared_ptr<ModelYOLO> yoloModel;
	std::mutex yoloModelMutex;
	ModelYOLO::Version modelVersion;

	std::vector<Detection> detections;
	std::mutex detectionsMutex;

	std::vector<Detection> trackedTargets;
	std::mutex trackedTargetsMutex;
	int nextTrackId;
	int maxLostFrames;
	float iouThreshold;

	bool useKalmanTracker = false;
	int kalmanGenerateThreshold = 2;
	int kalmanTerminateCount = 5;
	KalmanP kalmanTracker;

	struct KalmanPrediction { float x, y, width, height; int trackId; };
	std::vector<KalmanPrediction> kalmanPredictions;
	std::mutex kalmanPredictionsMutex;
	bool showKalmanPredictions = true;
	uint32_t kalmanPredictionColor = 0xFF00FFFF;

	int kalmanPredictionFrames = 5;
	std::vector<std::vector<std::pair<float, float>>> kalmanTrajectories;
	std::mutex kalmanTrajectoriesMutex;
	bool showKalmanTrajectories = true;
	uint32_t kalmanTrajectoryColor = 0xFFFFFF00;

	float trackingWeightIou;
	float trackingWeightCenter;
	float trackingWeightAspect;
	float trackingWeightArea;

	std::vector<LostTarget> lostTargets;
	std::mutex lostTargetsMutex;
	int maxReidentifyFrames;
	float reidentifyCenterThreshold;

	// 检测框 EMA 平滑（非 Kalman 路径；alpha 越大越跟检测、越小越稳）
	bool detectionSmoothingEnabled = false;
	float detectionSmoothingAlpha = 0.3f;

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

	int regionX, regionY, regionWidth, regionHeight;
	bool useRegion;

	std::thread inferenceThread;
	std::atomic<bool> inferenceRunning;
	int frameCounter;

	int inferenceFrameWidth, inferenceFrameHeight;
	int cropOffsetX, cropOffsetY;
	std::mutex inferenceFrameSizeMutex;

	uint64_t totalFrames;
	uint64_t inferenceCount;
	double avgInferenceTimeMs;

	std::atomic<bool> isInferencing;

	std::atomic<int> framesSubmitted{0};
	std::atomic<int> framesInferred{0};
	std::atomic<int> framesConsumed{0};
	std::atomic<int> framesDropped{0};
	std::atomic<int> dmlDirectFrames{0};   // DML直推成功帧数
	std::atomic<int> dmlFallbackFrames{0};  // DML直推失败回退CPU帧数

	static constexpr int BUFFER_COUNT = 4;
	cv::Mat inputFrames[BUFFER_COUNT];
	int inputFrameWidths[BUFFER_COUNT] = {0};
	int inputFrameHeights[BUFFER_COUNT] = {0};
	int inputCropX[BUFFER_COUNT] = {0};
	int inputCropY[BUFFER_COUNT] = {0};
	int inputCropWidth[BUFFER_COUNT] = {0};
	int inputCropHeight[BUFFER_COUNT] = {0};
	std::mutex inputFramesMutex;
	std::condition_variable frameReadyCv;
	std::atomic<int> inputWriteIdx{0};
	std::atomic<int> inputReadIdx{0};
	std::atomic<int64_t> lastResultTimestamp{0};
	std::atomic<uint8_t> bufferState[BUFFER_COUNT] = {};

	struct InferenceResult {
		std::vector<Detection> detections;
		std::vector<Detection> trackedTargets;
		int frameWidth = 0, frameHeight = 0, cropX = 0, cropY = 0;
		int64_t timestamp = 0;
	};
	std::shared_ptr<InferenceResult> inferenceResultPtr_{nullptr};
	mutable std::mutex inferenceResultMutex_;

	std::chrono::high_resolution_clock::time_point lastFpsTime;
	int fpsFrameCount;
	double currentFps;

	gs_effect_t *solidEffect;
	std::vector<std::thread> threadPool;

#ifdef _WIN32
	bool useGpuTextureInference = false;
#endif

#ifdef _WIN32
	// DML GPU直推：双缓冲，渲染写 / 推理 swap 取，避免每帧 1MB+ 深拷
	DmlPreprocessedFrame dmlPreprocessedFrames[2];
	std::atomic<int> dmlWriteIdx{0};
	std::atomic<int> dmlReadyIdx{-1};
	std::mutex dmlPreprocessedFrameMutex;
	DmlPreprocessor dmlPreprocessor;
	// 兼容旧字段名引用：指向当前 ready 槽（仅调试用，热路径用双缓冲 API）
	DmlPreprocessedFrame dmlPreprocessedFrame;
#endif
	std::queue<std::function<void()>> taskQueue;
	std::mutex taskQueueMutex;
	std::condition_variable taskCondition;
	std::atomic<bool> threadPoolRunning;

	struct ImageBufferKey {
		int rows, cols, type;
		bool operator==(const ImageBufferKey& o) const { return rows==o.rows && cols==o.cols && type==o.type; }
	};
	struct ImageBufferKeyHash {
		size_t operator()(const ImageBufferKey& k) const { return std::hash<int>()(k.rows) ^ (std::hash<int>()(k.cols)<<1) ^ (std::hash<int>()(k.type)<<2); }
	};
	std::unordered_map<ImageBufferKey, std::vector<cv::Mat>, ImageBufferKeyHash> imageBufferPool;
	std::vector<std::vector<Detection>> detectionBufferPool;
	std::mutex bufferPoolMutex;
	const int MAX_BUFFER_POOL_SIZE = 3;
	const int THREAD_POOL_SIZE = 4;

#ifdef _WIN32
	bool showFloatingWindow;
	int floatingWindowWidth, floatingWindowHeight, floatingWindowX, floatingWindowY;
	bool floatingWindowDragging;
	POINT floatingWindowDragOffset;
	HWND floatingWindowHandle;
	std::mutex floatingWindowMutex;
	cv::Mat floatingWindowFrame;
	bool showTrackIdInFloatingWindow;

	static const int PID_HISTORY_SIZE = 200;
	std::deque<PidDebugData> pidHistory;
	std::mutex pidHistoryMutex;
	bool showPidDebugWindow;
	HWND pidDebugWindowHandle;
	int pidDebugWindowWidth, pidDebugWindowHeight, pidDebugWindowX, pidDebugWindowY;
	bool pidDebugWindowDragging;
	POINT pidDebugWindowDragOffset;
	std::mutex pidDebugWindowMutex;
	cv::Mat pidDebugWindowFrame;

	static const int MAX_CONFIGS = 5;
	int algorithmTypeGlobal;

	float dynamicFovShrinkPercent;
	float dynamicFovTransitionTime;
	float currentFovRadius;
	std::chrono::steady_clock::time_point fovTransitionStartTime;
	bool isFovTransitioning;
	float fovTransitionStartRadius;
	float fovTransitionEndRadius;

	struct MouseControlConfig {
		bool enabled = false;
		int hotkey = 0;
		float pMin = 0.153f, pMax = 0.6f, pSlope = 1.0f;
		float d = 0.007f, i = 0.01f;
		float maxPixelMove = 128.0f, deadZonePixels = 5.0f;
		int screenOffsetX = 0, screenOffsetY = 0, screenWidth = 0, screenHeight = 0;
		float derivativeFilterAlpha = 0.2f;
		float adaptivePGainRate = 0.03f, dTermScale = 0.3f;
		float targetYOffset = 0.0f;
		int controllerType = 0;
	std::string makcuPort = "COM5";
	int makcuBaudRate = 4000000;
	int logiDriverType = 0;  // LogiDriver子类型：0=自动, 1=GHUB, 2=LGS, 3=Razer
		bool enableYAxisUnlock = false;
		int yAxisUnlockDelay = 500;
		bool enableAutoTrigger = false;
		int triggerRadius = 5, triggerCooldown = 200;
		int triggerFireDelay = 0, triggerFireDuration = 50, triggerInterval = 50;
		bool enableTriggerDelayRandom = false;
		int triggerDelayRandomMin = 0, triggerDelayRandomMax = 0;
		bool enableTriggerDurationRandom = false;
		int triggerDurationRandomMin = 0, triggerDurationRandomMax = 0;
		int triggerMoveCompensation = 0;
		float integralLimit = 100.0f, integralRate = 1.0f;
		float pGainRampInitialScale = 0.6f, pGainRampDuration = 0.5f;
		bool useDerivativePredictor = true;
		float predictionWeightX = 0.3f, predictionWeightY = 0.1f;
		float velocitySmoothFactor = 0.0f, accelerationSmoothFactor = 0.0f;
		float maxPredictionTime = 0.1f;
		// Smith预估器
		bool smithPredictorEnabled = false;
		float smithModelGain = 1.0f;
		float smithModelTau = 0.02f;
		bool smithAutoTau = true;

		// IMM交互多模型滤波器
		bool immFilterEnabled = false;
		float immProcessNoisePos = 0.1f;
		float immProcessNoiseVel = 0.5f;
		float immProcessNoiseAcc = 1.0f;
		float immProcessNoiseTurn = 0.1f;
		float immMeasurementNoiseX = 1.0f;
		float immMeasurementNoiseY = 1.0f;
		int immActiveModels = 3;

		// OneEuro 误差滤波
		bool useOneEuroFilter = false;
		float oneEuroMinCutoff = 1.0f;
		float oneEuroBeta = 0.007f;
		float oneEuroDCutoff = 1.0f;

		// SlewRate控制器（限速平滑趋近）
		bool slewRateEnabled = false;
		float slewRateOutputGain = 0.25f;
		float slewRateResponseSmoothing = 0.0008f;
		float slewRateApproachDamping = 5.0f;
		float slewRateUpdateIntervalMs = 5.0f;
		float slewRateNormalizationScale = 5.0f;

		// 自适应PID控制器（位置式+自适应积分增益+积分死区+双重抗饱和）
		float adaptivePidKp = 1.0f;
		float adaptivePidKi = 0.1f;
		float adaptivePidKd = 0.05f;
		float adaptivePidDeadZone = 0.3f;
		float adaptivePidIntegralLimit = 100.0f;
		float adaptivePidIntegralDeadzone = 1.0f;
		float adaptivePidIntegralGainThreshold = 50.0f;
		float adaptivePidIntegralGainRate = 0.015f;
		float adaptivePidOutputLimit = 10.0f;

		bool continuousAimEnabled = false;
		bool autoRecoilControlEnabled = false;
		float recoilStrength = 5.0f;
		int recoilSpeed = 16;
		float recoilPidGainScale = 0.3f;
		int algorithmType = 0;
		bool enableBezierMovement = false;
		float bezierCurvature = 0.3f, bezierRandomness = 0.2f;
		bool enableGhostTracker = false;
		float ghostCurvature = 0.5f, ghostNoiseIntensity = 12.0f;
		float ghostVerticalSnapRatio = 3.0f, ghostNoiseFreq = 0.8f;
		bool enableNeuralPath = false;
		int neuralPathPoints = 35;
		double neuralMouseStepSize = 8.0;
		int neuralTargetRadius = 8, neuralConsumePerFrame = 2;
		bool enableNeuralPathDebug = false;
		bool enableTimeBasedMovement = true;
		float targetFrameRate = 60.0f;
	};

	int targetSwitchDelayMs = 500;
	float targetSwitchTolerance = 0.15f;
	std::array<MouseControlConfig, MAX_CONFIGS> mouseConfigs;
	int currentConfigIndex;
	std::unique_ptr<MouseControllerInterface> mouseController;
	std::mutex mouseConfigsMutex;  // 保护 mouseConfigs + mouseController

	std::string configName;
	std::string configList;

	float externalKpX, externalKiX, externalKdX;
	float externalKpY, externalKiY, externalKdY;
	float externalPredictX, externalPredictY;
	float externalRateX, externalRateY;
	float externalKiMode, externalKpLimit, externalKiLimit, externalKdLimit;
	float externalOutputLimit, externalKiRate, externalKiDeadband;

	CrosshairDetector crosshairDetector;
	CrosshairDetectorConfig crosshairConfig;
	cv::Mat crosshairFrameBuf;
	std::mutex crosshairFrameMutex;
	bool crosshairNeedsPick = false;
	float crosshairPixelX = -1.0f, crosshairPixelY = -1.0f;
	bool crosshairDetected = false;
	// ??????????????????????? crosshairFrameBuf ??????????
	int crosshairCropOffsetX = 0, crosshairCropOffsetY = 0;
#endif

	~yolo_detector_filter() {
		obs_log(LOG_INFO, "YOLO detector filter destructor called");
#ifdef _WIN32
#endif
	}
};
// === 渲染 (filter_rendering.cpp) ===
void renderDetectionBoxes(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderKalmanPredictions(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderKalmanTrajectories(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderRegion(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);

// === UI (filter_properties.cpp) ===
// yolo_detector_filter_properties 声明在yolo-detector-filter.h (extern "C")
bool toggleInference(obs_properties_t *props, obs_property_t *property, void *data);
bool refreshStats(obs_properties_t *props, obs_property_t *property, void *data);
bool testMAKCUConnection(obs_properties_t *props, obs_property_t *property, void *data);

#ifdef _WIN32
bool saveConfigCallback(obs_properties_t *props, obs_property_t *property, void *data);
bool loadConfigCallback(obs_properties_t *props, obs_property_t *property, void *data);

// === 浮窗 (filter_floating.cpp) ===
LRESULT CALLBACK FloatingWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void createFloatingWindow(yolo_detector_filter *filter);
void destroyFloatingWindow(yolo_detector_filter *filter);
void updateFloatingWindowFrame(yolo_detector_filter *filter, const cv::Mat &frame);
void renderFloatingWindow(yolo_detector_filter *filter);
void setupPidDataCallback(yolo_detector_filter *filter);
void createPidDebugWindow(yolo_detector_filter *filter);
void destroyPidDebugWindow(yolo_detector_filter *filter);
void updatePidDebugWindow(yolo_detector_filter *filter);
#endif

#endif /* YOLO_DETECTOR_FILTER_H */