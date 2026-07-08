#include "yolo_detector_filter.h"  // 结构体+C++函数声明
#include "filter_inference.h"      // inferenceThreadWorker, threadPoolWorker

// OBS回调的extern "C"声明（yolo-detector-filter-info.c需要）
#ifdef __cplusplus
extern "C" {
#endif
const char *yolo_detector_filter_getname(void *unused);
void *yolo_detector_filter_create(obs_data_t *settings, obs_source_t *source);
void yolo_detector_filter_destroy(void *data);
void yolo_detector_filter_defaults(obs_data_t *settings);
obs_properties_t *yolo_detector_filter_properties(void *data);
void yolo_detector_filter_update(void *data, obs_data_t *settings);
void yolo_detector_filter_activate(void *data);
void yolo_detector_filter_deactivate(void *data);
void yolo_detector_filter_video_tick(void *data, float seconds);
void yolo_detector_filter_video_render(void *data, gs_effect_t *_effect);
#ifdef __cplusplus
}
#endif

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#include <commdlg.h>
#pragma comment(lib, "gdiplus.lib")
#include "MouseController.hpp"
#include "ConfigManager.hpp"
#ifdef HAVE_CUDA
#include <d3d11.h>
#endif
#endif

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>

#include <plugin-support.h>
#include "obs-utils/obs-utils.h"
#include "consts.h"

const char *yolo_detector_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("YOLODetector");
}

static bool onPageChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool onKalmanTrackerChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool onNeuralPathChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool onConfigChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static void setConfigPropertiesVisible(obs_properties_t *props, int configIndex, bool visible);
static void setBezierMovementPropertiesVisible(obs_properties_t *props, int configIndex, bool visible);
static void setPredictorPropertiesVisible(obs_properties_t *props, int configIndex, bool visible);

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

	// === 共享指针：消费推理结果 ===
	std::shared_ptr<yolo_detector_filter::InferenceResult> inferenceResult;
	{
		std::lock_guard<std::mutex> resultLock(tf->inferenceResultMutex_);
		inferenceResult = tf->inferenceResultPtr_;
	}
	if (inferenceResult) {
		{
			std::lock_guard<std::mutex> detLock(tf->detectionsMutex);
			tf->detections = inferenceResult->detections;
		}
		{
			std::lock_guard<std::mutex> sizeLock(tf->inferenceFrameSizeMutex);
			tf->inferenceFrameWidth = inferenceResult->frameWidth;
			tf->inferenceFrameHeight = inferenceResult->frameHeight;
			tf->cropOffsetX = inferenceResult->cropX;
			tf->cropOffsetY = inferenceResult->cropY;
		}
		tf->framesConsumed.fetch_add(1, std::memory_order_relaxed);
	} else {
		// 没有结果，检查是否超时需要清空
		auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now().time_since_epoch()).count();
		int64_t lastTs = tf->lastResultTimestamp.load(std::memory_order_acquire);
		
		// 超过500ms没有新结果，自动清空检测框
		if (nowMs - lastTs > 500 && !tf->detections.empty()) {
			std::lock_guard<std::mutex> detLock(tf->detectionsMutex);
			tf->detections.clear();
		}
	}

#ifdef _WIN32
	// === 准星检测：吸管取色 + HSV检测管线 ===
	if (tf->crosshairConfig.enabled) {
		cv::Mat bgrFrame;
		{
			std::lock_guard<std::mutex> lock(tf->crosshairFrameMutex);
			bgrFrame = tf->crosshairFrameBuf.clone();
		}

		// 吸管取色处理
		if (tf->crosshairNeedsPick && !bgrFrame.empty()) {
			bool picked = tf->crosshairDetector.pickColorFromCenter(
				bgrFrame,
				obs_source_get_base_width(tf->source),
				obs_source_get_base_height(tf->source),
				tf->cropOffsetX, tf->cropOffsetY
			);
			if (picked) {
				CrosshairDetectorConfig& chCfg = tf->crosshairConfig;
				chCfg.colorPicked = true;
				chCfg.pickedH = tf->crosshairDetector.getConfig().pickedH;
				chCfg.pickedS = tf->crosshairDetector.getConfig().pickedS;
				chCfg.pickedV = tf->crosshairDetector.getConfig().pickedV;
				chCfg.hMin = tf->crosshairDetector.getConfig().hMin;
				chCfg.hMax = tf->crosshairDetector.getConfig().hMax;
				chCfg.sMin = tf->crosshairDetector.getConfig().sMin;
				chCfg.sMax = tf->crosshairDetector.getConfig().sMax;
				chCfg.vMin = tf->crosshairDetector.getConfig().vMin;
				chCfg.vMax = tf->crosshairDetector.getConfig().vMax;
				tf->crosshairDetector.updateConfig(chCfg);

				// 回写OBS settings，确保UI滑块更新且不被_update覆盖
				obs_data_t *pickSettings = obs_source_get_settings(tf->source);
				if (pickSettings) {
					obs_data_set_int(pickSettings, "crosshair_h_min", chCfg.hMin);
					obs_data_set_int(pickSettings, "crosshair_h_max", chCfg.hMax);
					obs_data_set_int(pickSettings, "crosshair_s_min", chCfg.sMin);
					obs_data_set_int(pickSettings, "crosshair_s_max", chCfg.sMax);
					obs_data_set_int(pickSettings, "crosshair_v_min", chCfg.vMin);
					obs_data_set_int(pickSettings, "crosshair_v_max", chCfg.vMax);

					// 更新取色结果显示（RGB + HSV + 搜索范围）
					char infoText[256];
					snprintf(infoText, sizeof(infoText),
						"RGB(%d,%d,%d) HSV(%d,%d,%d) H[%d~%d] S[%d~%d] V[%d~%d]",
						chCfg.pickedR, chCfg.pickedG, chCfg.pickedB,
						chCfg.pickedH, chCfg.pickedS, chCfg.pickedV,
						chCfg.hMin, chCfg.hMax, chCfg.sMin, chCfg.sMax, chCfg.vMin, chCfg.vMax);
					obs_data_set_string(pickSettings, "crosshair_color_info", infoText);

					obs_source_update(tf->source, pickSettings);
					obs_data_release(pickSettings);
				}

				obs_log(LOG_INFO, "[Crosshair] 吸管取色成功: RGB(%d,%d,%d) HSV(%d,%d,%d), 范围 H[%d-%d] S[%d-%d] V[%d-%d]",
					chCfg.pickedR, chCfg.pickedG, chCfg.pickedB,
					chCfg.pickedH, chCfg.pickedS, chCfg.pickedV,
					chCfg.hMin, chCfg.hMax, chCfg.sMin, chCfg.sMax, chCfg.vMin, chCfg.vMax);
			}
			tf->crosshairNeedsPick = false;
		}

		// 帧间隔控制由CrosshairDetector内部处理，这里每帧都调用detect
		if (!bgrFrame.empty()) {

			// 帧已经是中心裁切区域，在裁切帧上全覆盖检测
			// fovCenterX/Y=0.5, fovRadiusNorm=1.0 让detect覆盖整个裁切帧
			std::vector<Detection> crosshairDets = tf->crosshairDetector.detect(
				bgrFrame,
				tf->crosshairFullFrameW, tf->crosshairFullFrameH,
				tf->crosshairCropOffsetX, tf->crosshairCropOffsetY,
				0.5f, 0.5f, 1.0f
			);

			// 保存调试掩码
			if (tf->crosshairConfig.showDebugMask) {
				cv::Mat debugMask = tf->crosshairDetector.getDebugMask();
				if (!debugMask.empty()) {
					std::lock_guard<std::mutex> lock(tf->crosshairDebugMutex);
					tf->crosshairDebugMask = debugMask.clone();
				}
			}

			// 准星检测结果：提取准星位置作为瞄准起点
			// detect返回的centerX/centerY是基于bgrFrame(裁切帧)的归一化坐标
			// 需要映射回完整帧像素坐标
			if (!crosshairDets.empty()) {
				const auto& chDet = crosshairDets[0];
				int cropPxX = static_cast<int>(chDet.centerX * bgrFrame.cols);
				int cropPxY = static_cast<int>(chDet.centerY * bgrFrame.rows);
				{
					std::lock_guard<std::mutex> lock(tf->crosshairFrameMutex);
					tf->crosshairPixelX = static_cast<float>(cropPxX + tf->crosshairCropOffsetX);
					tf->crosshairPixelY = static_cast<float>(cropPxY + tf->crosshairCropOffsetY);
					tf->crosshairDetected = true;
				}
			} else {
				std::lock_guard<std::mutex> lock(tf->crosshairFrameMutex);
				tf->crosshairDetected = false;
				// 检测失败，重置跟踪状态，下一帧重新从中心搜索
				tf->crosshairDetector.resetTracking();
			}
		}
	}

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
		mcConfig.adaptivePGainRate = cfg.adaptivePGainRate;
		mcConfig.dTermScale = cfg.dTermScale;
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
		// 积分参数
		mcConfig.integralLimit = cfg.integralLimit;
		mcConfig.integralRate = cfg.integralRate;
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
		// GhostTracker曲线轨迹参数
		mcConfig.enableGhostTracker = cfg.enableGhostTracker;
		mcConfig.ghostCurvature = cfg.ghostCurvature;
		mcConfig.ghostNoiseIntensity = cfg.ghostNoiseIntensity;
		mcConfig.ghostVerticalSnapRatio = cfg.ghostVerticalSnapRatio;
		mcConfig.ghostNoiseFreq = cfg.ghostNoiseFreq;
		// 算法选择（使用全局设置）
		// 0=AdvancedPID, 1=ExternalPID
		switch (tf->algorithmTypeGlobal) {
			case 0: mcConfig.algorithmType = AlgorithmType::AdvancedPID; break;
			case 1: mcConfig.algorithmType = AlgorithmType::ExternalPID; break;
			default: mcConfig.algorithmType = AlgorithmType::AdvancedPID; break;
		}
		// 专业PID参数
		mcConfig.externalKpX = tf->externalKpX;
		mcConfig.externalKiX = tf->externalKiX;
		mcConfig.externalKdX = tf->externalKdX;
		mcConfig.externalKpY = tf->externalKpY;
		mcConfig.externalKiY = tf->externalKiY;
		mcConfig.externalKdY = tf->externalKdY;
		mcConfig.externalPredictX = tf->externalPredictX;
		mcConfig.externalPredictY = tf->externalPredictY;
		mcConfig.externalRateX = tf->externalRateX;
		mcConfig.externalRateY = tf->externalRateY;
		mcConfig.externalKiMode = tf->externalKiMode;
		mcConfig.externalKpLimit = tf->externalKpLimit;
		mcConfig.externalKiLimit = tf->externalKiLimit;
		mcConfig.externalKdLimit = tf->externalKdLimit;
		mcConfig.externalOutputLimit = tf->externalOutputLimit;
		mcConfig.externalKiRate = tf->externalKiRate;
		mcConfig.externalKiDeadband = tf->externalKiDeadband;
		// 神经网络轨迹生成器参数
		mcConfig.enableNeuralPath = cfg.enableNeuralPath;
		mcConfig.neuralPathPoints = cfg.neuralPathPoints;
		mcConfig.neuralMouseStepSize = cfg.neuralMouseStepSize;
		mcConfig.neuralTargetRadius = cfg.neuralTargetRadius;
		mcConfig.neuralConsumePerFrame = cfg.neuralConsumePerFrame;
		mcConfig.enableNeuralPathDebug = cfg.enableNeuralPathDebug;
		// 时间相关移动参数（帧率补偿）
		mcConfig.enableTimeBasedMovement = cfg.enableTimeBasedMovement;
		mcConfig.targetFrameRate = cfg.targetFrameRate;
		
		static int syncCount = 0;
		syncCount++;
		if ((syncCount <= 3 || cfg.enableNeuralPath) && cfg.enableNeuralPathDebug) {
			obs_log(LOG_INFO, "[NeuralPath-Sync] PASS TO CONTROLLER: enableNeuralPath=%d, neuralPathPoints=%d, neuralMouseStepSize=%.1f, neuralTargetRadius=%d, debug=%d",
					mcConfig.enableNeuralPath ? 1 : 0,
					mcConfig.neuralPathPoints,
					mcConfig.neuralMouseStepSize,
					mcConfig.neuralTargetRadius,
					mcConfig.enableNeuralPathDebug ? 1 : 0);
		}
		
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

		// 锁定mouseConfigsMutex保护 config读取 + mouseController写入
		{
		std::lock_guard<std::mutex> cfgLock(tf->mouseConfigsMutex);

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
					{
						std::lock_guard<std::mutex> lock(tf->crosshairFrameMutex);
						if (tf->crosshairDetected) {
							tf->mouseController->setAimOrigin(tf->crosshairPixelX, tf->crosshairPixelY);
						} else {
							tf->mouseController->setAimOrigin(-1.0f, -1.0f);
						}
					}
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
		} // end mouseConfigsMutex lock
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
					{
						std::lock_guard<std::mutex> lock(tf->crosshairFrameMutex);
						if (tf->crosshairDetected) {
							tf->mouseController->setAimOrigin(tf->crosshairPixelX, tf->crosshairPixelY);
						} else {
							tf->mouseController->setAimOrigin(-1.0f, -1.0f);
						}
					}
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
	bool needCapture = tf->showFloatingWindow || tf->isInferencing || needShowLabels
#ifdef _WIN32
		|| tf->crosshairConfig.enabled
#endif
		;

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

							// 首次：缓存D3D11设备（推理线程需要用它做DML预处理）
							if (!tf->cachedD3d11Device) {
								d3dTex->GetDevice(&tf->cachedD3d11Device);
							}
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
								// 加锁保护 inputFrames 的重新分配和写入，防止分辨率变化时的竞态条件
								{
									std::lock_guard<std::mutex> lock(tf->inputFramesMutex);
									if (tf->inputFrames[checkIdx].rows != height || 
										tf->inputFrames[checkIdx].cols != width) {
										tf->inputFrames[checkIdx] = cv::Mat(height, width, CV_8UC4);
									}
									temp.copyTo(tf->inputFrames[checkIdx]);
								}
								
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

						// 准星检测帧捕获：只裁切中心区域（准星始终在屏幕中心附近），大幅减少处理面积
						if (tf->crosshairConfig.enabled) {
							// 搜索半径：配置值或自动（帧宽1/6）
							int searchRadius = tf->crosshairConfig.searchRadius > 0
								? tf->crosshairConfig.searchRadius
								: temp.cols / 6;
							int centerX = temp.cols / 2;
							int centerY = temp.rows / 2;
							int x0 = std::max(0, centerX - searchRadius);
							int y0 = std::max(0, centerY - searchRadius);
							int x1 = std::min(temp.cols, centerX + searchRadius);
							int y1 = std::min(temp.rows, centerY + searchRadius);

							cv::Mat bgrCrop;
							cv::cvtColor(temp(cv::Rect(x0, y0, x1 - x0, y1 - y0)), bgrCrop, cv::COLOR_BGRA2BGR);
							std::lock_guard<std::mutex> chLock(tf->crosshairFrameMutex);
							tf->crosshairFrameBuf = std::move(bgrCrop);
							tf->crosshairCropOffsetX = x0;
							tf->crosshairCropOffsetY = y0;
							tf->crosshairFullFrameW = temp.cols;
							tf->crosshairFullFrameH = temp.rows;
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

	// 渲染源画面
	obs_source_process_filter_end(tf->source, obs_get_base_effect(OBS_EFFECT_DEFAULT), width, height);

	// 在源画面上渲染检测框和FOV
	if (tf->showDetectionResults) {
		renderDetectionBoxes(tf.get(), width, height);
	}
	// 渲染卡尔曼预测位置（虚线青色框）
	renderKalmanPredictions(tf.get(), width, height);
	// 渲染卡尔曼多帧预测轨迹（黄色轨迹线）
	renderKalmanTrajectories(tf.get(), width, height);
	if (tf->showFOV) {
		renderFOV(tf.get(), width, height);
	}
	if (tf->useRegion) {
		renderRegion(tf.get(), width, height);
	}
#ifdef _WIN32
	// 渲染准星搜索半径（橙色圆圈）
	if (tf->crosshairConfig.enabled) {
		gs_effect_t *solid = tf->solidEffect;
		gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
		gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

		int searchRadius = tf->crosshairConfig.searchRadius > 0
			? tf->crosshairConfig.searchRadius
			: static_cast<int>(width) / 6;
		float cx = width / 2.0f;
		float cy = height / 2.0f;
		float r = static_cast<float>(searchRadius);

		struct vec4 color;
		vec4_set(&color, 1.0f, 0.5f, 0.0f, 0.8f); // 橙色

		gs_technique_begin(tech);
		gs_technique_begin_pass(tech, 0);
		gs_effect_set_vec4(colorParam, &color);

		// 渲染圆圈
		const int circleSegments = 64;
		gs_render_start(true);
		for (int i = 0; i <= circleSegments; ++i) {
			float angle = 2.0f * 3.1415926f * static_cast<float>(i) / static_cast<float>(circleSegments);
			float x = cx + r * cosf(angle);
			float y = cy + r * sinf(angle);
			gs_vertex2f(x, y);
		}
		gs_render_stop(GS_LINESTRIP);

		gs_technique_end_pass(tech);
		gs_technique_end(tech);
	}
#endif

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

		// 准星检测可视化
		if (tf->crosshairConfig.enabled) {
			// 颜色隔离视图：直接在悬浮窗图像上做HSV分割
			if (tf->crosshairConfig.colorIsolationView) {
				cv::Mat bgrCropped;
				cv::cvtColor(croppedFrame, bgrCropped, cv::COLOR_BGRA2BGR);
				cv::Mat hsvCropped;
				cv::cvtColor(bgrCropped, hsvCropped, cv::COLOR_BGR2HSV);
				cv::Scalar lower(tf->crosshairConfig.hMin, tf->crosshairConfig.sMin, tf->crosshairConfig.vMin);
				cv::Scalar upper(tf->crosshairConfig.hMax, tf->crosshairConfig.sMax, tf->crosshairConfig.vMax);
				cv::Mat mask;
				cv::inRange(hsvCropped, lower, upper, mask);
				
				// 形态学处理（和实际检测保持一致）
				if (tf->crosshairConfig.erodeIterations > 0 && tf->crosshairConfig.morphKernelSize > 0) {
					cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
						cv::Size(tf->crosshairConfig.morphKernelSize, tf->crosshairConfig.morphKernelSize));
					cv::erode(mask, mask, kernel, cv::Point(-1, -1), tf->crosshairConfig.erodeIterations);
				}
				if (tf->crosshairConfig.dilateIterations > 0 && tf->crosshairConfig.morphKernelSize > 0) {
					cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
						cv::Size(tf->crosshairConfig.morphKernelSize, tf->crosshairConfig.morphKernelSize));
					cv::dilate(mask, mask, kernel, cv::Point(-1, -1), tf->crosshairConfig.dilateIterations);
				}
				
				cv::Mat invMask;
				cv::bitwise_not(mask, invMask);
				croppedFrame.setTo(cv::Scalar(0, 0, 0, 255), invMask);
			}
			// 调试掩码：直接在悬浮窗图像上做HSV分割
			else if (tf->crosshairConfig.showDebugMask) {
				cv::Mat bgrCropped;
				cv::cvtColor(croppedFrame, bgrCropped, cv::COLOR_BGRA2BGR);
				cv::Mat hsvCropped;
				cv::cvtColor(bgrCropped, hsvCropped, cv::COLOR_BGR2HSV);
				cv::Scalar lower(tf->crosshairConfig.hMin, tf->crosshairConfig.sMin, tf->crosshairConfig.vMin);
				cv::Scalar upper(tf->crosshairConfig.hMax, tf->crosshairConfig.sMax, tf->crosshairConfig.vMax);
				cv::Mat mask;
				cv::inRange(hsvCropped, lower, upper, mask);
				// 形态学处理和准星检测保持一致
				if (tf->crosshairConfig.erodeIterations > 0 && tf->crosshairConfig.morphKernelSize > 0) {
					cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
						cv::Size(tf->crosshairConfig.morphKernelSize, tf->crosshairConfig.morphKernelSize));
					cv::erode(mask, mask, kernel, cv::Point(-1, -1), tf->crosshairConfig.erodeIterations);
				}
				if (tf->crosshairConfig.dilateIterations > 0 && tf->crosshairConfig.morphKernelSize > 0) {
					cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
						cv::Size(tf->crosshairConfig.morphKernelSize, tf->crosshairConfig.morphKernelSize));
					cv::dilate(mask, mask, kernel, cv::Point(-1, -1), tf->crosshairConfig.dilateIterations);
				}
				// 半透明叠加
				cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGRA);
				cv::addWeighted(croppedFrame, 0.5, mask, 0.5, 0, croppedFrame);
			}
		}

		// 准星检测：绘制准星位置（紫色十字标注）
		bool chDetected = false;
		float chPixelX = 0, chPixelY = 0;
		{
			std::lock_guard<std::mutex> lock(tf->crosshairFrameMutex);
			chDetected = tf->crosshairDetected;
			chPixelX = tf->crosshairPixelX;
			chPixelY = tf->crosshairPixelY;
		}
		if (tf->crosshairConfig.enabled && chDetected) {
			cv::Scalar crosshairColor(255, 0, 255, 255);
			int cx = static_cast<int>(chPixelX) - cropOffsetX;
			int cy = static_cast<int>(chPixelY) - cropOffsetY;
			if (cx >= 0 && cx < croppedFrame.cols && cy >= 0 && cy < croppedFrame.rows) {
				// 绘制十字准星标记
				int armLen = 12;
				cv::line(croppedFrame, cv::Point(cx - armLen, cy), cv::Point(cx + armLen, cy), crosshairColor, 2);
				cv::line(croppedFrame, cv::Point(cx, cy - armLen), cv::Point(cx, cy + armLen), crosshairColor, 2);
				cv::circle(croppedFrame, cv::Point(cx, cy), 6, crosshairColor, 1);
				cv::putText(croppedFrame, "CH", cv::Point(cx + 8, cy - 8),
					cv::FONT_HERSHEY_SIMPLEX, 0.4, crosshairColor, 1);
			}
		}

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
