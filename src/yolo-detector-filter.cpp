#include "yolo-detector-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#include <gdiplus.h>
#include <commdlg.h>
#pragma comment(lib, "gdiplus.lib")
#include "MouseController.hpp"
#include "MouseControllerFactory.hpp"
#include "LogitechMacroConverter.hpp"
#include "RecoilPatternManager.hpp"
#include "ConfigManager.hpp"
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

#include <plugin-support.h>
#include "models/ModelYOLO.h"
#include "models/Detection.h"
#include "HungarianAlgorithm.hpp"
#include "FilterData.h"
#include "ort-utils/ort-session-utils.h"
#include "obs-utils/obs-utils.h"
#include "consts.h"
#include "update-checker/update-checker.h"

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
	std::atomic<bool> shouldInference;
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

	std::chrono::high_resolution_clock::time_point lastFpsTime;
	int fpsFrameCount;
	double currentFps;

	gs_effect_t *solidEffect;

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

	static const int MAX_CONFIGS = 5;

	struct MouseControlConfig {
		bool enabled;
		int hotkey;
		float pMin;
		float pMax;
		float pSlope;
		float d;
		float baselineCompensation;
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
		int triggerDelayRandomMin;
		int triggerDelayRandomMax;
		int triggerDurationRandomMin;
		int triggerDurationRandomMax;
		int triggerMoveCompensation;
		std::string weaponName;

		MouseControlConfig() {
			enabled = false;
			hotkey = VK_XBUTTON1;
			pMin = 0.153f;
			pMax = 0.6f;
			pSlope = 1.0f;
			d = 0.007f;
			baselineCompensation = 0.85f;
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
			triggerDelayRandomMin = 0;
			triggerDelayRandomMax = 0;
			triggerDurationRandomMin = 0;
			triggerDurationRandomMax = 0;
			triggerMoveCompensation = 0;
		}
	};

	std::array<MouseControlConfig, MAX_CONFIGS> mouseConfigs;
	int currentConfigIndex;
	std::unique_ptr<MouseControllerInterface> mouseController;

	std::string configName;
	std::string configList;
#endif

	~yolo_detector_filter() { obs_log(LOG_INFO, "YOLO detector filter destructor called"); }
};

void inferenceThreadWorker(yolo_detector_filter *filter);
static void renderDetectionBoxes(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static bool toggleInference(obs_properties_t *props, obs_property_t *property, void *data);
static bool refreshStats(obs_properties_t *props, obs_property_t *property, void *data);
static bool testMAKCUConnection(obs_properties_t *props, obs_property_t *property, void *data);
#ifdef _WIN32
static bool importRecoilPattern(obs_properties_t *props, obs_property_t *property, void *data);
static bool deleteRecoilPattern(obs_properties_t *props, obs_property_t *property, void *data);
static bool refreshWeaponList(obs_properties_t *props, obs_property_t *property, void *data);
static bool saveConfigCallback(obs_properties_t *props, obs_property_t *property, void *data);
static bool loadConfigCallback(obs_properties_t *props, obs_property_t *property, void *data);
#endif


#ifdef _WIN32
static LRESULT CALLBACK FloatingWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
static void createFloatingWindow(yolo_detector_filter *filter);
static void destroyFloatingWindow(yolo_detector_filter *filter);
static void updateFloatingWindowFrame(yolo_detector_filter *filter, const cv::Mat &frame);
static void renderFloatingWindow(yolo_detector_filter *filter);
#endif

const char *yolo_detector_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("YOLODetector");
}

static bool onPageChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static bool onConfigChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings);
static void setConfigPropertiesVisible(obs_properties_t *props, int configIndex, bool visible);

obs_properties_t *yolo_detector_filter_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *toggleBtn = obs_properties_add_button(props, "toggle_inference", obs_module_text("ToggleInference"), toggleInference);
	obs_properties_add_text(props, "inference_status", obs_module_text("InferenceStatus"), OBS_TEXT_INFO);

	obs_property_t *pageList = obs_properties_add_list(props, "settings_page", "设置页面", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(pageList, "YOLO检测设置", 0);
	obs_property_list_add_int(pageList, "鼠标控制", 1);
	obs_property_list_add_int(pageList, "FOV设置", 2);
	obs_property_list_add_int(pageList, "目标追踪", 3);
	obs_property_set_modified_callback(pageList, onPageChanged);

	obs_properties_add_group(props, "model_group", obs_module_text("ModelConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_path(props, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE, "ONNX Models (*.onnx)", nullptr);
	obs_property_t *modelVersion = obs_properties_add_list(props, "model_version", obs_module_text("ModelVersion"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(modelVersion, "YOLOv5", static_cast<int>(ModelYOLO::Version::YOLOv5));
	obs_property_list_add_int(modelVersion, "YOLOv8", static_cast<int>(ModelYOLO::Version::YOLOv8));
	obs_property_list_add_int(modelVersion, "YOLOv11", static_cast<int>(ModelYOLO::Version::YOLOv11));
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
	obs_property_t *resolutionList = obs_properties_add_list(props, "input_resolution", obs_module_text("InputResolution"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(resolutionList, "320x320", 320);
	obs_property_list_add_int(resolutionList, "416x416", 416);
	obs_property_list_add_int(resolutionList, "512x512", 512);
	obs_property_list_add_int(resolutionList, "640x640", 640);
	obs_properties_add_int_slider(props, "num_threads", obs_module_text("NumThreads"), 1, 16, 1);

	obs_properties_add_group(props, "detection_group", obs_module_text("DetectionConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_float_slider(props, "confidence_threshold", obs_module_text("ConfidenceThreshold"), 0.01, 1.0, 0.01);
	obs_properties_add_float_slider(props, "nms_threshold", obs_module_text("NMSThreshold"), 0.01, 1.0, 0.01);
	obs_property_t *targetClass = obs_properties_add_list(props, "target_class", obs_module_text("TargetClass"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(targetClass, obs_module_text("AllClasses"), -1);
	obs_properties_add_text(props, "target_classes_text", "目标类别(多个用逗号分隔)", OBS_TEXT_DEFAULT);
	obs_properties_add_int_slider(props, "inference_interval_frames", obs_module_text("InferenceIntervalFrames"), 0, 10, 1);

	obs_properties_add_group(props, "render_group", obs_module_text("RenderConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_bool(props, "show_detection_results", obs_module_text("ShowDetectionResults"));
	obs_properties_add_bool(props, "show_bbox", obs_module_text("ShowBoundingBox"));
	obs_properties_add_bool(props, "show_label", obs_module_text("ShowLabel"));
	obs_properties_add_bool(props, "show_confidence", obs_module_text("ShowConfidence"));
	obs_properties_add_int_slider(props, "bbox_line_width", obs_module_text("LineWidth"), 1, 5, 1);
	obs_properties_add_color(props, "bbox_color", obs_module_text("BoxColor"));
	obs_properties_add_float_slider(props, "label_font_scale", obs_module_text("LabelFontScale"), 0.2, 1.0, 0.05);

	obs_properties_add_group(props, "region_group", obs_module_text("RegionDetection"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_bool(props, "use_region", obs_module_text("UseRegionDetection"));
	obs_properties_add_int(props, "region_x", obs_module_text("RegionX"), 0, 3840, 1);
	obs_properties_add_int(props, "region_y", obs_module_text("RegionY"), 0, 2160, 1);
	obs_properties_add_int(props, "region_width", obs_module_text("RegionWidth"), 1, 3840, 1);
	obs_properties_add_int(props, "region_height", obs_module_text("RegionHeight"), 1, 2160, 1);

	obs_properties_add_group(props, "advanced_group", obs_module_text("AdvancedConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_bool(props, "export_coordinates", obs_module_text("ExportCoordinates"));
	obs_properties_add_path(props, "coordinate_output_path", obs_module_text("CoordinateOutputPath"), OBS_PATH_FILE_SAVE, "JSON Files (*.json)", nullptr);

	obs_properties_add_group(props, "fov_group", obs_module_text("FOVSettings"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_bool(props, "show_fov", obs_module_text("ShowFOV"));
	obs_properties_add_int_slider(props, "fov_radius", obs_module_text("FOVRadius"), 1, 500, 1);
	obs_properties_add_bool(props, "show_fov_circle", obs_module_text("ShowFOVCircle"));
	obs_properties_add_bool(props, "show_fov_cross", obs_module_text("ShowFOVCross"));
	obs_properties_add_int_slider(props, "fov_cross_line_scale", obs_module_text("FOVCrossLineScale"), 1, 300, 5);
	obs_properties_add_int_slider(props, "fov_cross_line_thickness", obs_module_text("FOVCrossLineThickness"), 1, 10, 1);
	obs_properties_add_int_slider(props, "fov_circle_thickness", obs_module_text("FOVCircleThickness"), 1, 10, 1);
	obs_properties_add_color(props, "fov_color", obs_module_text("FOVColor"));

	obs_properties_add_group(props, "fov2_group", "动态FOV设置", OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_bool(props, "use_dynamic_fov", "启用动态FOV");
	obs_properties_add_bool(props, "show_fov2", "显示第二个FOV");
	obs_properties_add_int_slider(props, "fov_radius2", "第二个FOV半径", 1, 200, 1);
	obs_properties_add_color(props, "fov_color2", "第二个FOV颜色");

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
		obs_properties_add_bool(props, propName, "启用此配置");

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

		snprintf(propName, sizeof(propName), "controller_type_%d", i);
		obs_property_t *controllerTypeList = obs_properties_add_list(props, propName, "控制方式", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(controllerTypeList, "Windows API", 0);
		obs_property_list_add_int(controllerTypeList, "MAKCU", 1);

		snprintf(propName, sizeof(propName), "makcu_port_%d", i);
		obs_properties_add_text(props, propName, "MAKCU 端口", OBS_TEXT_DEFAULT);

		snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
		obs_property_t *baudRateList = obs_properties_add_list(props, propName, "波特率", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(baudRateList, "9600", 9600);
		obs_property_list_add_int(baudRateList, "19200", 19200);
		obs_property_list_add_int(baudRateList, "38400", 38400);
		obs_property_list_add_int(baudRateList, "57600", 57600);
		obs_property_list_add_int(baudRateList, "115200", 115200);
		obs_property_list_add_int(baudRateList, "4000000 (4Mbps)", 4000000);

		snprintf(propName, sizeof(propName), "p_min_%d", i);
		obs_properties_add_float_slider(props, propName, "P最小值", 0.00, 1.00, 0.01);
		snprintf(propName, sizeof(propName), "p_max_%d", i);
		obs_properties_add_float_slider(props, propName, "P最大值", 0.00, 1.00, 0.01);
		snprintf(propName, sizeof(propName), "p_slope_%d", i);
		obs_properties_add_float_slider(props, propName, "P增长斜率", 0.00, 10, 0.01);
		snprintf(propName, sizeof(propName), "baseline_compensation_%d", i);
		obs_properties_add_float_slider(props, propName, "基线补偿", 0.00, 1.00, 0.01);
		snprintf(propName, sizeof(propName), "d_%d", i);
		obs_properties_add_float_slider(props, propName, "微分系数", 0.000, 1.00, 0.001);
		snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
		obs_properties_add_float_slider(props, propName, "微分滤波系数", 0.01, 1.00, 0.01);

		snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", i);
		obs_properties_add_float_slider(props, propName, "X轴平滑度", 0.00, 1.0, 0.01);
		snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", i);
		obs_properties_add_float_slider(props, propName, "Y轴平滑度", 0.00, 1.0, 0.01);
		snprintf(propName, sizeof(propName), "target_y_offset_%d", i);
		obs_properties_add_float_slider(props, propName, "Y轴目标偏移", -50.0, 50.0, 1.0);
		snprintf(propName, sizeof(propName), "max_pixel_move_%d", i);
		obs_properties_add_float_slider(props, propName, "最大移动量", 0.0, 200.0, 1.0);
		snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", i);
		obs_properties_add_float_slider(props, propName, "瞄准死区", 0.0, 20.0, 0.5);

		snprintf(propName, sizeof(propName), "screen_offset_x_%d", i);
		obs_properties_add_int(props, propName, "屏幕偏移X", 0, 3840, 1);
		snprintf(propName, sizeof(propName), "screen_offset_y_%d", i);
		obs_properties_add_int(props, propName, "屏幕偏移Y", 0, 2160, 1);
		snprintf(propName, sizeof(propName), "screen_width_%d", i);
		obs_properties_add_int(props, propName, "屏幕宽度", 0, 3840, 1);
		snprintf(propName, sizeof(propName), "screen_height_%d", i);
		obs_properties_add_int(props, propName, "屏幕高度", 0, 2160, 1);

		snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", i);
		obs_properties_add_bool(props, propName, "启用长按解锁Y轴");
		snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", i);
		obs_properties_add_int_slider(props, propName, "Y轴解锁延迟(ms)", 100, 2000, 50);

		snprintf(propName, sizeof(propName), "enable_auto_trigger_%d", i);
		obs_properties_add_bool(props, propName, "启用自动扳机");
		snprintf(propName, sizeof(propName), "trigger_radius_%d", i);
		obs_properties_add_int_slider(props, propName, "扳机触发半径(像素)", 1, 50, 1);
		snprintf(propName, sizeof(propName), "trigger_cooldown_%d", i);
		obs_properties_add_int_slider(props, propName, "扳机冷却时间(ms)", 50, 1000, 50);
		snprintf(propName, sizeof(propName), "trigger_fire_delay_%d", i);
		obs_properties_add_int_slider(props, propName, "开火延时(ms)", 0, 500, 10);
		snprintf(propName, sizeof(propName), "trigger_fire_duration_%d", i);
		obs_properties_add_int_slider(props, propName, "开火时长(ms)", 10, 500, 10);
		snprintf(propName, sizeof(propName), "trigger_interval_%d", i);
		obs_properties_add_int_slider(props, propName, "间隔设置(ms)", 10, 500, 10);
		snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
		obs_properties_add_int_slider(props, propName, "随机延时下限(ms)", 0, 200, 5);
		snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
		obs_properties_add_int_slider(props, propName, "随机延时上限(ms)", 0, 200, 5);
		snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
		obs_properties_add_int_slider(props, propName, "随机时长下限(ms)", 0, 200, 5);
		snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
		obs_properties_add_int_slider(props, propName, "随机时长上限(ms)", 0, 200, 5);
		snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
		obs_properties_add_int_slider(props, propName, "移动补偿(像素)", 0, 100, 1);

		snprintf(propName, sizeof(propName), "weapon_select_%d", i);
		obs_properties_add_list(props, propName, "选择武器", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	}

	obs_properties_add_button(props, "test_makcu_connection", "测试MAKCU连接", testMAKCUConnection);
	obs_properties_add_text(props, "weapon_name_input", "武器名称", OBS_TEXT_DEFAULT);
	obs_properties_add_button(props, "import_recoil_pattern", "导入压枪宏", importRecoilPattern);
	obs_properties_add_button(props, "delete_recoil_pattern", "删除武器配置", deleteRecoilPattern);
	obs_properties_add_button(props, "refresh_weapon_list", "刷新武器列表", refreshWeaponList);

	obs_properties_add_group(props, "tracking_group", "目标追踪设置", OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_float_slider(props, "iou_threshold", "IoU阈值", 0.1, 0.9, 0.05);
	obs_properties_add_int_slider(props, "max_lost_frames", "最大丢失帧数", 1, 30, 1);

	obs_properties_add_group(props, "floating_window_group", obs_module_text("FloatingWindow"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_bool(props, "show_floating_window", obs_module_text("ShowFloatingWindow"));
	obs_properties_add_int_slider(props, "floating_window_width", obs_module_text("WindowWidth"), 320, 1920, 10);
	obs_properties_add_int_slider(props, "floating_window_height", obs_module_text("WindowHeight"), 240, 1080, 10);

	obs_properties_add_group(props, "config_management_group", "配置管理", OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_button(props, "save_config", "保存配置", saveConfigCallback);
	obs_properties_add_button(props, "load_config", "加载配置", loadConfigCallback);
#endif

	obs_properties_add_text(props, "avg_inference_time", obs_module_text("AvgInferenceTime"), OBS_TEXT_INFO);
	obs_properties_add_text(props, "detected_objects", obs_module_text("DetectedObjects"), OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

static void setConfigPropertiesVisible(obs_properties_t *props, int configIndex, bool visible)
{
	char propName[64];

	snprintf(propName, sizeof(propName), "enable_config_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "hotkey_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "controller_type_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "makcu_port_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "p_min_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "p_max_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "p_slope_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "baseline_compensation_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "d_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "aim_smoothing_x_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "aim_smoothing_y_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "target_y_offset_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "max_pixel_move_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "dead_zone_pixels_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "screen_offset_x_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "screen_offset_y_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "screen_width_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "screen_height_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "enable_y_axis_unlock_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "y_axis_unlock_delay_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "enable_auto_trigger_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_radius_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_cooldown_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_fire_delay_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_fire_duration_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_interval_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);

	snprintf(propName, sizeof(propName), "weapon_select_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
}

static bool onConfigChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	int currentConfig = (int)obs_data_get_int(settings, "mouse_config_select");

	for (int i = 0; i < 5; i++) {
		setConfigPropertiesVisible(props, i, i == currentConfig);
	}

	obs_property_set_visible(obs_properties_get(props, "mouse_config_select"), true);
	obs_property_set_visible(obs_properties_get(props, "test_makcu_connection"), true);
	obs_property_set_visible(obs_properties_get(props, "weapon_name_input"), true);
	obs_property_set_visible(obs_properties_get(props, "import_recoil_pattern"), true);
	obs_property_set_visible(obs_properties_get(props, "delete_recoil_pattern"), true);
	obs_property_set_visible(obs_properties_get(props, "refresh_weapon_list"), true);

	return true;
}

static bool onPageChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	int page = (int)obs_data_get_int(settings, "settings_page");

	obs_property_t *model_group = obs_properties_get(props, "model_group");
	obs_property_t *detection_group = obs_properties_get(props, "detection_group");
	obs_property_t *render_group = obs_properties_get(props, "render_group");
	obs_property_t *region_group = obs_properties_get(props, "region_group");
	obs_property_t *advanced_group = obs_properties_get(props, "advanced_group");
	obs_property_t *fov_group = obs_properties_get(props, "fov_group");
	obs_property_t *fov2_group = obs_properties_get(props, "fov2_group");

	obs_property_set_visible(model_group, page == 0);
	obs_property_set_visible(detection_group, page == 0);
	obs_property_set_visible(render_group, page == 0);
	obs_property_set_visible(region_group, page == 0);
	obs_property_set_visible(advanced_group, page == 0);

	obs_property_set_visible(fov_group, page == 2);
	obs_property_set_visible(fov2_group, page == 2);

	obs_property_t *model_path = obs_properties_get(props, "model_path");
	obs_property_t *model_version = obs_properties_get(props, "model_version");
	obs_property_t *use_gpu = obs_properties_get(props, "use_gpu");
	obs_property_t *input_resolution = obs_properties_get(props, "input_resolution");
	obs_property_t *num_threads = obs_properties_get(props, "num_threads");
	obs_property_t *confidence_threshold = obs_properties_get(props, "confidence_threshold");
	obs_property_t *nms_threshold = obs_properties_get(props, "nms_threshold");
	obs_property_t *target_class = obs_properties_get(props, "target_class");
	obs_property_t *target_classes_text = obs_properties_get(props, "target_classes_text");
	obs_property_t *inference_interval_frames = obs_properties_get(props, "inference_interval_frames");
	obs_property_t *show_detection_results = obs_properties_get(props, "show_detection_results");
	obs_property_t *show_bbox = obs_properties_get(props, "show_bbox");
	obs_property_t *show_label = obs_properties_get(props, "show_label");
	obs_property_t *show_confidence = obs_properties_get(props, "show_confidence");
	obs_property_t *bbox_line_width = obs_properties_get(props, "bbox_line_width");
	obs_property_t *bbox_color = obs_properties_get(props, "bbox_color");
	obs_property_t *label_font_scale = obs_properties_get(props, "label_font_scale");
	obs_property_t *use_region = obs_properties_get(props, "use_region");
	obs_property_t *region_x = obs_properties_get(props, "region_x");
	obs_property_t *region_y = obs_properties_get(props, "region_y");
	obs_property_t *region_width = obs_properties_get(props, "region_width");
	obs_property_t *region_height = obs_properties_get(props, "region_height");
	obs_property_t *export_coordinates = obs_properties_get(props, "export_coordinates");
	obs_property_t *coordinate_output_path = obs_properties_get(props, "coordinate_output_path");

	obs_property_set_visible(model_path, page == 0);
	obs_property_set_visible(model_version, page == 0);
	obs_property_set_visible(use_gpu, page == 0);
	obs_property_set_visible(input_resolution, page == 0);
	obs_property_set_visible(num_threads, page == 0);
	obs_property_set_visible(confidence_threshold, page == 0);
	obs_property_set_visible(nms_threshold, page == 0);
	obs_property_set_visible(target_class, page == 0);
	obs_property_set_visible(target_classes_text, page == 0);
	obs_property_set_visible(inference_interval_frames, page == 0);
	obs_property_set_visible(show_detection_results, page == 0);
	obs_property_set_visible(show_bbox, page == 0);
	obs_property_set_visible(show_label, page == 0);
	obs_property_set_visible(show_confidence, page == 0);
	obs_property_set_visible(bbox_line_width, page == 0);
	obs_property_set_visible(bbox_color, page == 0);
	obs_property_set_visible(label_font_scale, page == 0);
	obs_property_set_visible(use_region, page == 0);
	obs_property_set_visible(region_x, page == 0);
	obs_property_set_visible(region_y, page == 0);
	obs_property_set_visible(region_width, page == 0);
	obs_property_set_visible(region_height, page == 0);
	obs_property_set_visible(export_coordinates, page == 0);
	obs_property_set_visible(coordinate_output_path, page == 0);

	obs_property_t *show_fov = obs_properties_get(props, "show_fov");
	obs_property_t *fov_radius = obs_properties_get(props, "fov_radius");
	obs_property_t *show_fov_circle = obs_properties_get(props, "show_fov_circle");
	obs_property_t *show_fov_cross = obs_properties_get(props, "show_fov_cross");
	obs_property_t *fov_cross_line_scale = obs_properties_get(props, "fov_cross_line_scale");
	obs_property_t *fov_cross_line_thickness = obs_properties_get(props, "fov_cross_line_thickness");
	obs_property_t *fov_circle_thickness = obs_properties_get(props, "fov_circle_thickness");
	obs_property_t *fov_color = obs_properties_get(props, "fov_color");
	obs_property_t *use_dynamic_fov = obs_properties_get(props, "use_dynamic_fov");
	obs_property_t *show_fov2 = obs_properties_get(props, "show_fov2");
	obs_property_t *fov_radius2 = obs_properties_get(props, "fov_radius2");
	obs_property_t *fov_color2 = obs_properties_get(props, "fov_color2");

	obs_property_set_visible(show_fov, page == 2);
	obs_property_set_visible(fov_radius, page == 2);
	obs_property_set_visible(show_fov_circle, page == 2);
	obs_property_set_visible(show_fov_cross, page == 2);
	obs_property_set_visible(fov_cross_line_scale, page == 2);
	obs_property_set_visible(fov_cross_line_thickness, page == 2);
	obs_property_set_visible(fov_circle_thickness, page == 2);
	obs_property_set_visible(fov_color, page == 2);
	obs_property_set_visible(use_dynamic_fov, page == 2);
	obs_property_set_visible(show_fov2, page == 2);
	obs_property_set_visible(fov_radius2, page == 2);
	obs_property_set_visible(fov_color2, page == 2);

#ifdef _WIN32
	obs_property_t *tracking_group = obs_properties_get(props, "tracking_group");
	obs_property_set_visible(tracking_group, page == 3);

	obs_property_t *iou_threshold = obs_properties_get(props, "iou_threshold");
	obs_property_t *max_lost_frames = obs_properties_get(props, "max_lost_frames");
	obs_property_set_visible(iou_threshold, page == 3);
	obs_property_set_visible(max_lost_frames, page == 3);

	obs_property_set_visible(obs_properties_get(props, "mouse_config_select"), page == 1);

	int currentConfig = (int)obs_data_get_int(settings, "mouse_config_select");
	for (int i = 0; i < 5; i++) {
		setConfigPropertiesVisible(props, i, page == 1 && i == currentConfig);
	}

	obs_property_set_visible(obs_properties_get(props, "test_makcu_connection"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "weapon_name_input"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "import_recoil_pattern"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "delete_recoil_pattern"), page == 1);
	obs_property_set_visible(obs_properties_get(props, "refresh_weapon_list"), page == 1);
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
	obs_data_set_default_int(settings, "input_resolution", 640);
	obs_data_set_default_int(settings, "num_threads", 4);
	obs_data_set_default_double(settings, "confidence_threshold", 0.5);
	obs_data_set_default_double(settings, "nms_threshold", 0.45);
	obs_data_set_default_int(settings, "target_class", -1);
	obs_data_set_default_int(settings, "inference_interval_frames", 1);
	obs_data_set_default_bool(settings, "show_detection_results", true);
	obs_data_set_default_bool(settings, "show_bbox", true);
	obs_data_set_default_bool(settings, "show_label", true);
	obs_data_set_default_bool(settings, "show_confidence", true);
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
		snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
		obs_data_set_default_double(settings, propName, 0.2);
		snprintf(propName, sizeof(propName), "baseline_compensation_%d", i);
		obs_data_set_default_double(settings, propName, 0.85);

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

		snprintf(propName, sizeof(propName), "enable_auto_trigger_%d", i);
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
		snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
		obs_data_set_default_int(settings, propName, 0);
		snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
		obs_data_set_default_int(settings, propName, 0);
	}

	obs_data_set_default_string(settings, "config_name", "");
	obs_data_set_default_string(settings, "config_list", "");
	obs_data_set_default_double(settings, "iou_threshold", 0.3);
	obs_data_set_default_int(settings, "max_lost_frames", 10);
	obs_data_set_default_int(settings, "settings_page", 0);
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
	tf->fovRadius = (int)obs_data_get_int(settings, "fov_radius");
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
		snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", i);
		tf->mouseConfigs[i].derivativeFilterAlpha = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "baseline_compensation_%d", i);
		tf->mouseConfigs[i].baselineCompensation = (float)obs_data_get_double(settings, propName);

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

		snprintf(propName, sizeof(propName), "enable_auto_trigger_%d", i);
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
		snprintf(propName, sizeof(propName), "trigger_delay_random_min_%d", i);
		tf->mouseConfigs[i].triggerDelayRandomMin = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_delay_random_max_%d", i);
		tf->mouseConfigs[i].triggerDelayRandomMax = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_duration_random_min_%d", i);
		tf->mouseConfigs[i].triggerDurationRandomMin = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_duration_random_max_%d", i);
		tf->mouseConfigs[i].triggerDurationRandomMax = (int)obs_data_get_int(settings, propName);
		snprintf(propName, sizeof(propName), "trigger_move_compensation_%d", i);
		tf->mouseConfigs[i].triggerMoveCompensation = (int)obs_data_get_int(settings, propName);

		snprintf(propName, sizeof(propName), "weapon_select_%d", i);
		tf->mouseConfigs[i].weaponName = obs_data_get_string(settings, propName);
	}

	bool hasEnabledConfig = false;
	for (int i = 0; i < 5; i++) {
		if (tf->mouseConfigs[i].enabled) {
			hasEnabledConfig = true;
			break;
		}
	}

	if (!tf->mouseController && hasEnabledConfig) {
		tf->mouseController = MouseControllerFactory::createController(ControllerType::WindowsAPI, "", 0);
		obs_log(LOG_INFO, "Created mouse controller for multi-config mode");
	}

	tf->configName = obs_data_get_string(settings, "config_name");
	tf->configList = obs_data_get_string(settings, "config_list");
	tf->iouThreshold = (float)obs_data_get_double(settings, "iou_threshold");
	tf->maxLostFrames = (int)obs_data_get_int(settings, "max_lost_frames");
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
	if (tf->isInferencing) {
		tf->shouldInference = true;
	}
	obs_log(LOG_INFO, "[YOLO Detector] Inference %s, isInferencing=%d, shouldInference=%d", 
		tf->isInferencing ? "enabled" : "disabled",
		(int)tf->isInferencing,
		(int)tf->shouldInference);

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

static bool importRecoilPattern(obs_properties_t *props, obs_property_t *property, void *data)
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
    std::string weaponName;
    if (settings) {
        weaponName = obs_data_get_string(settings, "weapon_name_input");
        obs_data_release(settings);
    }
    
    if (weaponName.empty()) {
        MessageBoxA(NULL, "请先输入武器名称！", "提示", MB_OK | MB_ICONWARNING);
        return true;
    }

    OPENFILENAMEA ofn = {};
    char szFile[MAX_PATH] = "";

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "宏文件 (*.xml;*.lua)\0*.xml;*.lua\0XML Files (*.xml)\0*.xml\0LUA Files (*.lua)\0*.lua\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "选择罗技压枪宏文件";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        if (RecoilPatternManager::getInstance().importFromLogitechMacro(szFile, weaponName)) {
            const RecoilPattern* pattern = RecoilPatternManager::getInstance().getPattern(weaponName);
            
            char infoMsg[1024];
            snprintf(infoMsg, sizeof(infoMsg),
                "压枪配置导入成功！\n\n"
                "武器名称: %s\n"
                "移动步数: %zu\n"
                "总移动X: %d\n"
                "总移动Y: %d\n"
                "总持续时间: %d ms\n\n"
                "请在\"选择武器\"下拉框中选择此武器。",
                weaponName.c_str(),
                pattern->moves.size(),
                pattern->totalMoveX,
                pattern->totalMoveY,
                pattern->totalDurationMs);

            MessageBoxA(NULL, infoMsg, "导入成功", MB_OK | MB_ICONINFORMATION);
            
            refreshWeaponList(props, property, data);
        } else {
            MessageBoxA(NULL, "无法解析宏文件，请确保文件格式正确。", "解析失败", MB_OK | MB_ICONERROR);
        }
    }

    return true;
}

static bool deleteRecoilPattern(obs_properties_t *props, obs_property_t *property, void *data)
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
    std::string weaponName;
    if (settings) {
        weaponName = obs_data_get_string(settings, "weapon_select");
        obs_data_release(settings);
    }
    
    if (weaponName.empty()) {
        MessageBoxA(NULL, "请先选择要删除的武器！", "提示", MB_OK | MB_ICONWARNING);
        return true;
    }

    char confirmMsg[256];
    snprintf(confirmMsg, sizeof(confirmMsg), "确定要删除武器 \"%s\" 的压枪配置吗？", weaponName.c_str());
    
    int result = MessageBoxA(NULL, confirmMsg, "确认删除", MB_YESNO | MB_ICONQUESTION);
    if (result == IDYES) {
        RecoilPatternManager::getInstance().removePattern(weaponName);
        MessageBoxA(NULL, "删除成功！", "提示", MB_OK | MB_ICONINFORMATION);
        refreshWeaponList(props, property, data);
    }

    return true;
}

static bool refreshWeaponList(obs_properties_t *props, obs_property_t *property, void *data)
{
    obs_property_t *weaponList = obs_properties_get(props, "weapon_select");
    if (!weaponList) {
        return true;
    }
    
    obs_property_list_clear(weaponList);
    
    obs_property_list_add_string(weaponList, "-- 无武器 --", "");
    
    auto weaponNames = RecoilPatternManager::getInstance().getWeaponNames();
    for (const auto& name : weaponNames) {
        obs_property_list_add_string(weaponList, name.c_str(), name.c_str());
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

    std::thread([tf]() {
        obs_data_t *settings = obs_source_get_settings(tf->source);
        if (!settings) {
            return;
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
            return;
        }
        
        std::string filePath = szFile;
        size_t lastSlash = filePath.find_last_of("\\/");
        size_t lastDot = filePath.find_last_of(".");
        std::string configName = (lastDot != std::string::npos && lastDot > lastSlash) 
            ? filePath.substr(lastSlash + 1, lastDot - lastSlash - 1) 
            : filePath.substr(lastSlash + 1);
        std::string dir = (lastSlash != std::string::npos) ? filePath.substr(0, lastSlash) : ".";
        ConfigManager::getInstance().setConfigsDirectory(dir);

        ExtendedMouseControllerConfig config;
        config.configName = configName;
        config.enableMouseControl = obs_data_get_bool(settings, "enable_mouse_control");
        config.hotkeyVirtualKey = (int)obs_data_get_int(settings, "mouse_control_hotkey");
        config.screenOffsetX = (int)obs_data_get_int(settings, "screen_offset_x");
        config.screenOffsetY = (int)obs_data_get_int(settings, "screen_offset_y");
        config.screenWidth = (int)obs_data_get_int(settings, "screen_width");
        config.screenHeight = (int)obs_data_get_int(settings, "screen_height");
        config.pidPMin = (float)obs_data_get_double(settings, "mouse_control_p_min");
        config.pidPMax = (float)obs_data_get_double(settings, "mouse_control_p_max");
        config.pidPSlope = (float)obs_data_get_double(settings, "mouse_control_p_slope");
        config.pidD = (float)obs_data_get_double(settings, "mouse_control_d");
        config.baselineCompensation = (float)obs_data_get_double(settings, "baseline_compensation");
        config.derivativeFilterAlpha = (float)obs_data_get_double(settings, "derivative_filter_alpha");
        config.aimSmoothingX = (float)obs_data_get_double(settings, "aim_smoothing_x");
        config.aimSmoothingY = (float)obs_data_get_double(settings, "aim_smoothing_y");
        config.maxPixelMove = (float)obs_data_get_double(settings, "max_pixel_move");
        config.deadZonePixels = (float)obs_data_get_double(settings, "dead_zone_pixels");
        config.targetYOffset = (float)obs_data_get_double(settings, "target_y_offset");
        config.controllerType = (int)obs_data_get_int(settings, "controller_type") == 1 ? ControllerType::MAKCU : ControllerType::WindowsAPI;
        config.makcuPort = obs_data_get_string(settings, "makcu_port");
        config.makcuBaudRate = (int)obs_data_get_int(settings, "makcu_baud_rate");
        config.yUnlockEnabled = obs_data_get_bool(settings, "enable_y_axis_unlock");
        config.yUnlockDelayMs = (int)obs_data_get_int(settings, "y_axis_unlock_delay");
        config.autoTriggerEnabled = obs_data_get_bool(settings, "enable_auto_trigger");
        config.autoTriggerRadius = (float)obs_data_get_int(settings, "trigger_radius");
        config.autoTriggerCooldownMs = (int)obs_data_get_int(settings, "trigger_cooldown");

        obs_data_release(settings);

        if (ConfigManager::getInstance().saveConfig(config)) {
            MessageBoxA(NULL, ("配置已保存到:\n" + filePath).c_str(), "成功", MB_OK | MB_ICONINFORMATION);
        } else {
            MessageBoxA(NULL, "保存配置失败！", "错误", MB_OK | MB_ICONERROR);
        }
    }).detach();

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

    std::thread([tf]() {
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
            return;
        }
        
        std::string filePath = szFile;
        size_t lastSlash = filePath.find_last_of("\\/");
        size_t lastDot = filePath.find_last_of(".");
        std::string configName = (lastDot != std::string::npos && lastDot > lastSlash) 
            ? filePath.substr(lastSlash + 1, lastDot - lastSlash - 1) 
            : filePath.substr(lastSlash + 1);
        
        std::string dir = (lastSlash != std::string::npos) ? filePath.substr(0, lastSlash) : ".";
        ConfigManager::getInstance().setConfigsDirectory(dir);

        ExtendedMouseControllerConfig config;
        if (ConfigManager::getInstance().loadConfig(configName, config)) {
            obs_data_t *newSettings = obs_source_get_settings(tf->source);
            if (newSettings) {
                obs_data_set_bool(newSettings, "enable_mouse_control", config.enableMouseControl);
                obs_data_set_int(newSettings, "mouse_control_hotkey", config.hotkeyVirtualKey);
                obs_data_set_int(newSettings, "screen_offset_x", config.screenOffsetX);
                obs_data_set_int(newSettings, "screen_offset_y", config.screenOffsetY);
                obs_data_set_int(newSettings, "screen_width", config.screenWidth);
                obs_data_set_int(newSettings, "screen_height", config.screenHeight);
                obs_data_set_double(newSettings, "mouse_control_p_min", config.pidPMin);
                obs_data_set_double(newSettings, "mouse_control_p_max", config.pidPMax);
                obs_data_set_double(newSettings, "mouse_control_p_slope", config.pidPSlope);
                obs_data_set_double(newSettings, "mouse_control_d", config.pidD);
                obs_data_set_double(newSettings, "baseline_compensation", config.baselineCompensation);
                obs_data_set_double(newSettings, "derivative_filter_alpha", config.derivativeFilterAlpha);
                obs_data_set_double(newSettings, "aim_smoothing_x", config.aimSmoothingX);
                obs_data_set_double(newSettings, "aim_smoothing_y", config.aimSmoothingY);
                obs_data_set_double(newSettings, "max_pixel_move", config.maxPixelMove);
                obs_data_set_double(newSettings, "dead_zone_pixels", config.deadZonePixels);
                obs_data_set_double(newSettings, "target_y_offset", config.targetYOffset);
                obs_data_set_int(newSettings, "controller_type", config.controllerType == ControllerType::MAKCU ? 1 : 0);
                obs_data_set_string(newSettings, "makcu_port", config.makcuPort.c_str());
                obs_data_set_int(newSettings, "makcu_baud_rate", config.makcuBaudRate);
                obs_data_set_bool(newSettings, "enable_y_axis_unlock", config.yUnlockEnabled);
                obs_data_set_int(newSettings, "y_axis_unlock_delay", config.yUnlockDelayMs);
                obs_data_set_bool(newSettings, "enable_auto_trigger", config.autoTriggerEnabled);
                obs_data_set_int(newSettings, "trigger_radius", (int)config.autoTriggerRadius);
                obs_data_set_int(newSettings, "trigger_cooldown", config.autoTriggerCooldownMs);
                obs_data_release(newSettings);
            }
            MessageBoxA(NULL, ("配置 \"" + configName + "\" 加载成功！").c_str(), "成功", MB_OK | MB_ICONINFORMATION);
        } else {
            MessageBoxA(NULL, "加载配置失败！配置文件可能不存在或格式错误。", "错误", MB_OK | MB_ICONERROR);
        }
    }).detach();

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
}

static void renderFloatingWindow(yolo_detector_filter *filter)
{
	if (!filter->floatingWindowHandle || filter->floatingWindowFrame.empty()) {
		return;
	}
	InvalidateRect(filter->floatingWindowHandle, NULL, FALSE);
}
#endif

void inferenceThreadWorker(yolo_detector_filter *filter)
{
	obs_log(LOG_INFO, "[YOLO Detector] Inference thread started");

	int sleepTime = 5; // 初始休眠时间（毫秒）

	while (filter->inferenceRunning) {
		if (!filter->shouldInference) {
			// 动态调整休眠时间
			// 系统负载低时增加休眠时间，负载高时减少休眠时间
			std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
			continue;
		}

		filter->shouldInference = false;

		if (!filter->isInferencing) {
			// 推理被禁用了，增加休眠时间
			sleepTime = std::min(50, sleepTime + 5);
			continue;
		}

		// 推理启用时，减少休眠时间以提高响应速度
		sleepTime = std::max(1, sleepTime - 2);

		cv::Mat fullFrame;
		cv::Mat frame;
		int cropX = 0;
		int cropY = 0;
		int cropWidth = 0;
		int cropHeight = 0;

		{
			std::unique_lock<std::mutex> lock(filter->inputBGRALock, std::try_to_lock);
			if (!lock.owns_lock()) {
				continue;
			}
			if (filter->inputBGRA.empty()) {
				continue;
			}
			fullFrame = filter->inputBGRA.clone();
		}

		int fullWidth = fullFrame.cols;
		int fullHeight = fullFrame.rows;

		if (filter->useRegion) {
			cropX = std::max(0, filter->regionX);
			cropY = std::max(0, filter->regionY);
			cropWidth = std::min(filter->regionWidth, fullWidth - cropX);
			cropHeight = std::min(filter->regionHeight, fullHeight - cropY);

			if (cropWidth > 0 && cropHeight > 0) {
				frame = fullFrame(cv::Rect(cropX, cropY, cropWidth, cropHeight)).clone();
			} else {
				frame = std::move(fullFrame);
				cropX = 0;
				cropY = 0;
				cropWidth = fullWidth;
				cropHeight = fullHeight;
			}
		} else {
			frame = std::move(fullFrame);
		}

		auto startTime = std::chrono::high_resolution_clock::now();

		std::vector<Detection> newDetections;
		try {
			std::lock_guard<std::mutex> lock(filter->yoloModelMutex);
			if (!filter->yoloModel) {
				continue;
			}
			newDetections = filter->yoloModel->inference(frame);
		} catch (const std::exception& e) {
			obs_log(LOG_ERROR, "[YOLO Detector] Inference error: %s", e.what());
		}

		if (filter->useRegion && cropWidth > 0 && cropHeight > 0) {
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

		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
			endTime - startTime
		).count();

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
					
					for (int j = 0; j < m; ++j) {
						cv::Rect2f trackBox(
							trackedTargets[j].x,
							trackedTargets[j].y,
							trackedTargets[j].width,
							trackedTargets[j].height
						);
						
						costMatrix[i][j] = HungarianAlgorithm::calculateIoUDistance(detBox, trackBox);
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
						}
					}
				}
			}
			
			filter->trackedTargets = trackedDetections;
		}

		{
			std::lock_guard<std::mutex> lock(filter->detectionsMutex);
			filter->detections = std::move(trackedDetections);
		}

		{
			std::lock_guard<std::mutex> lock(filter->inferenceFrameSizeMutex);
			filter->inferenceFrameWidth = fullWidth;
			filter->inferenceFrameHeight = fullHeight;
			filter->cropOffsetX = cropX;
			filter->cropOffsetY = cropY;
		}

		filter->inferenceCount++;
		filter->avgInferenceTimeMs = (filter->avgInferenceTimeMs * (filter->inferenceCount - 1) + duration) / filter->inferenceCount;

		if (filter->exportCoordinates && !newDetections.empty()) {
			exportCoordinatesToFile(filter, fullWidth, fullHeight);
		}
	}

	obs_log(LOG_INFO, "[YOLO Detector] Inference thread stopped");
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
		gs_vertex2f(x + w, y);
		gs_vertex2f(x + w, y + h);
		gs_vertex2f(x + w, y + h);
		gs_vertex2f(x, y + h);
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

		// Initialize ORT environment
		std::string instanceName{"yolo-detector-inference"};
		instance->env.reset(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str()));

		instance->inferenceRunning = false;
		instance->shouldInference = false;
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

		for (int i = 0; i < 5; i++) {
			instance->mouseConfigs[i] = yolo_detector_filter::MouseControlConfig();
		}
		instance->currentConfigIndex = 0;
		instance->mouseController = MouseControllerFactory::createController(ControllerType::WindowsAPI, "", 0);

		instance->configName = "";
		instance->configList = "";
#endif

		// Create pointer to shared_ptr for the update call
		auto ptr = new std::shared_ptr<yolo_detector_filter>(instance);
		yolo_detector_filter_update(ptr, settings);

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

	if (tf->inferenceIntervalFrames == 0 || tf->frameCounter >= tf->inferenceIntervalFrames) {
		tf->frameCounter = 0;
		tf->shouldInference = true;
	}

#ifdef _WIN32
	auto getActiveConfig = [&tf]() -> int {
		for (int i = 0; i < 5; i++) {
			if (tf->mouseConfigs[i].enabled) {
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
			obs_log(LOG_INFO, "Created mouse controller type=%d, port=%s, baud=%d", newType, cfg.makcuPort.c_str(), cfg.makcuBaudRate);
		}

		MouseControllerConfig mcConfig;
		mcConfig.enableMouseControl = true;
		mcConfig.hotkeyVirtualKey = cfg.hotkey;
		mcConfig.fovRadiusPixels = tf->useDynamicFOV && tf->isInFOV2Mode ? tf->fovRadius2 : tf->fovRadius;
		mcConfig.pidPMin = cfg.pMin;
		mcConfig.pidPMax = cfg.pMax;
		mcConfig.pidPSlope = cfg.pSlope;
		mcConfig.pidD = cfg.d;
		mcConfig.baselineCompensation = cfg.baselineCompensation;
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
		mcConfig.autoTriggerDelayRandomMin = cfg.triggerDelayRandomMin;
		mcConfig.autoTriggerDelayRandomMax = cfg.triggerDelayRandomMax;
		mcConfig.autoTriggerDurationRandomMin = cfg.triggerDurationRandomMin;
		mcConfig.autoTriggerDurationRandomMax = cfg.triggerDurationRandomMax;
		mcConfig.autoTriggerMoveCompensation = cfg.triggerMoveCompensation;
		tf->mouseController->updateConfig(mcConfig);
		tf->mouseController->setCurrentWeapon(cfg.weaponName);
	};

	// 动态FOV切换逻辑
	if (tf->useDynamicFOV) {
		std::vector<Detection> detectionsCopy;
		{
			std::lock_guard<std::mutex> lock(tf->detectionsMutex);
			detectionsCopy = tf->detections;
		}

		// 画面中心
		float centerX = 0.5f;
		float centerY = 0.5f;

		// 当前使用的FOV半径
		float currentFOVRadius = tf->isInFOV2Mode ? 
			(static_cast<float>(tf->fovRadius2) / static_cast<float>(obs_source_get_base_width(tf->source))) :
			(static_cast<float>(tf->fovRadius) / static_cast<float>(obs_source_get_base_width(tf->source)));

		// 检查FOV内是否有目标
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

		// FOV切换逻辑
		if (!tf->isInFOV2Mode) {
			// 模式1：使用第一个FOV检测
			if (hasTargetInCurrentFOV) {
				// 检测到目标，切换到第二个FOV模式
				tf->isInFOV2Mode = true;
			}
		} else {
			// 模式2：使用第二个FOV锁敌
			if (!hasTargetInCurrentFOV) {
				// 第二个FOV内没有目标，切换回第一个FOV模式
				tf->isInFOV2Mode = false;
			}
		}

		// 更新鼠标控制器
		if (tf->mouseController && tf->isInferencing) {
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
			}
		}
	} else {
		// 不使用动态FOV，正常处理
		if (tf->mouseController && tf->isInferencing) {
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
						
						// 保存原始帧用于推理线程
						std::unique_lock<std::mutex> lock(tf->inputBGRALock, std::try_to_lock);
						if (lock.owns_lock()) {
							// 调整inputBGRA大小以匹配当前帧，避免重新分配
							if (tf->inputBGRA.rows != height || tf->inputBGRA.cols != width) {
								tf->inputBGRA = cv::Mat(height, width, CV_8UC4);
							}
							// 直接复制数据，避免克隆
							temp.copyTo(tf->inputBGRA);
						}
						
						// 保存原始帧用于悬浮窗显示
						originalImage = temp.clone();
						
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
		int cropWidth = tf->floatingWindowWidth;
		int cropHeight = tf->floatingWindowHeight;

		int centerX = originalImage.cols / 2;
		int centerY = originalImage.rows / 2;

		int cropX = std::max(0, centerX - cropWidth / 2);
		int cropY = std::max(0, centerY - cropHeight / 2);

		int actualCropWidth = std::min(cropWidth, originalImage.cols - cropX);
		int actualCropHeight = std::min(cropHeight, originalImage.rows - cropY);

		if (actualCropWidth > 0 && actualCropHeight > 0) {
			cv::Mat croppedFrame = originalImage(cv::Rect(cropX, cropY, actualCropWidth, actualCropHeight)).clone();

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
					int x = static_cast<int>(det.x * originalImage.cols) - cropX;
					int y = static_cast<int>(det.y * originalImage.rows) - cropY;
					int w = static_cast<int>(det.width * originalImage.cols);
					int h = static_cast<int>(det.height * originalImage.rows);
					
					if (x + w >= 0 && y + h >= 0 && x < croppedFrame.cols && y < croppedFrame.rows) {
						cv::rectangle(croppedFrame, 
							cv::Point(x, y), 
							cv::Point(x + w, y + h), 
							bboxColor, 
							lineWidth);
					}
				}
			}

			// 如果需要显示 FOV
			if (tf->showFOV) {
				float fovCenterX = (originalImage.cols / 2.0f) - cropX;
				float fovCenterY = (originalImage.rows / 2.0f) - cropY;
				float fovRadius = static_cast<float>(tf->fovRadius);
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

			if (tf->showFOV2 && tf->useDynamicFOV) {
				float fovCenterX = (originalImage.cols / 2.0f) - cropX;
				float fovCenterY = (originalImage.rows / 2.0f) - cropY;
				float fovRadius2 = static_cast<float>(tf->fovRadius2);

				float r2 = ((tf->fovColor2 >> 16) & 0xFF) / 255.0f;
				float g2 = ((tf->fovColor2 >> 8) & 0xFF) / 255.0f;
				float b2 = (tf->fovColor2 & 0xFF) / 255.0f;
				cv::Scalar fovColor2(b2 * 255, g2 * 255, r2 * 255, 255);

				cv::circle(croppedFrame, 
					cv::Point(static_cast<int>(fovCenterX), static_cast<int>(fovCenterY)),
					static_cast<int>(fovRadius2),
					fovColor2, 2);
			}

			// 绘制从中心点到目标的连接线
			float centerX = (originalImage.cols / 2.0f) - cropX;
			float centerY = (originalImage.rows / 2.0f) - cropY;
			cv::Point centerPoint(static_cast<int>(centerX), static_cast<int>(centerY));
			
			// 使用绿色绘制连接线
			cv::Scalar lineColor(0, 255, 0, 255);
			int lineThickness = 1;
			
			for (const auto& det : detectionsCopy) {
				// 计算目标在裁剪帧中的坐标
				int targetX = static_cast<int>(det.centerX * originalImage.cols) - cropX;
				int targetY = static_cast<int>(det.centerY * originalImage.rows) - cropY;
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
					int x = static_cast<int>(det.x * originalImage.cols) - cropX;
					int y = static_cast<int>(det.y * originalImage.rows) - cropY;
					
					// 确保在裁剪区域内
					if (x >= 0 && y >= 0 && x < croppedFrame.cols && y < croppedFrame.rows) {
						// 构建标签文本
						char labelText[64];
						snprintf(labelText, sizeof(labelText), "%d: %.2f", det.classId, det.confidence);
						
						// 只绘制文本，不绘制黑色背景
						cv::Point textOrg(x, y - 5);
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

			if (croppedFrame.cols != cropWidth || croppedFrame.rows != cropHeight) {
				cv::Mat resizedFrame;
				cv::resize(croppedFrame, resizedFrame, cv::Size(cropWidth, cropHeight));
				updateFloatingWindowFrame(tf.get(), resizedFrame);
			} else {
				updateFloatingWindowFrame(tf.get(), croppedFrame);
			}
			renderFloatingWindow(tf.get());
		}
	}
#endif
}
