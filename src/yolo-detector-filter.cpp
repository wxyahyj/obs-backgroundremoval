#include "yolo-detector-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
#endif

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>
#include <chrono>

#include <plugin-support.h>
#include "models/ModelYOLO.h"
#include "models/Detection.h"
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

	std::string modelPath;
	int inputResolution;
	float confidenceThreshold;
	float nmsThreshold;
	int targetClassId;
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

	int regionX;
	int regionY;
	int regionWidth;
	int regionHeight;
	bool useRegion;

	std::thread inferenceThread;
	std::atomic<bool> inferenceRunning;
	std::atomic<bool> shouldInference;
	int frameCounter;

	uint64_t totalFrames;
	uint64_t inferenceCount;
	double avgInferenceTimeMs;

	std::atomic<bool> isInferencing;

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
#endif

	~yolo_detector_filter() { obs_log(LOG_INFO, "YOLO detector filter destructor called"); }
};

void inferenceThreadWorker(yolo_detector_filter *filter);
static void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static bool toggleInference(obs_properties_t *props, obs_property_t *property, void *data);
static bool refreshStats(obs_properties_t *props, obs_property_t *property, void *data);

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

obs_properties_t *yolo_detector_filter_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_properties_add_group(props, "control_group", obs_module_text("Control"), OBS_GROUP_NORMAL, nullptr);

	obs_property_t *toggleBtn = obs_properties_add_button(props, "toggle_inference", obs_module_text("ToggleInference"), toggleInference);
	obs_properties_add_text(props, "inference_status", obs_module_text("InferenceStatus"), OBS_TEXT_INFO);

	obs_property_t *refreshBtn = obs_properties_add_button(props, "refresh_stats", obs_module_text("RefreshStats"), refreshStats);

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

	obs_properties_add_int_slider(props, "inference_interval_frames", obs_module_text("InferenceIntervalFrames"), 1, 10, 1);

	obs_properties_add_group(props, "render_group", obs_module_text("RenderConfiguration"), OBS_GROUP_NORMAL, nullptr);

	obs_properties_add_bool(props, "show_bbox", obs_module_text("ShowBoundingBox"));

	obs_properties_add_bool(props, "show_label", obs_module_text("ShowLabel"));

	obs_properties_add_bool(props, "show_confidence", obs_module_text("ShowConfidence"));

	obs_properties_add_int_slider(props, "bbox_line_width", obs_module_text("LineWidth"), 1, 5, 1);

	obs_properties_add_color(props, "bbox_color", obs_module_text("BoxColor"));

	obs_properties_add_group(props, "fov_group", obs_module_text("FOVSettings"), OBS_GROUP_NORMAL, nullptr);

	obs_properties_add_bool(props, "show_fov", obs_module_text("ShowFOV"));
	obs_properties_add_int_slider(props, "fov_radius", obs_module_text("FOVRadius"), 50, 500, 10);
	obs_properties_add_color(props, "fov_color", obs_module_text("FOVColor"));

	obs_properties_add_group(props, "region_group", obs_module_text("RegionDetection"), OBS_GROUP_NORMAL, nullptr);

	obs_properties_add_bool(props, "use_region", obs_module_text("UseRegionDetection"));
	obs_properties_add_int(props, "region_x", obs_module_text("RegionX"), 0, 3840, 1);
	obs_properties_add_int(props, "region_y", obs_module_text("RegionY"), 0, 2160, 1);
	obs_properties_add_int(props, "region_width", obs_module_text("RegionWidth"), 1, 3840, 1);
	obs_properties_add_int(props, "region_height", obs_module_text("RegionHeight"), 1, 2160, 1);

	obs_properties_add_group(props, "advanced_group", obs_module_text("AdvancedConfiguration"), OBS_GROUP_NORMAL, nullptr);

	obs_properties_add_bool(props, "export_coordinates", obs_module_text("ExportCoordinates"));

	obs_properties_add_path(props, "coordinate_output_path", obs_module_text("CoordinateOutputPath"), OBS_PATH_FILE_SAVE, "JSON Files (*.json)", nullptr);

	obs_properties_add_group(props, "stats_group", obs_module_text("Statistics"), OBS_GROUP_NORMAL, nullptr);

	obs_properties_add_text(props, "avg_inference_time", obs_module_text("AvgInferenceTime"), OBS_TEXT_INFO);

	obs_properties_add_text(props, "detected_objects", obs_module_text("DetectedObjects"), OBS_TEXT_INFO);

#ifdef _WIN32
	obs_properties_add_group(props, "floating_window_group", obs_module_text("FloatingWindow"), OBS_GROUP_NORMAL, nullptr);
	obs_properties_add_bool(props, "show_floating_window", obs_module_text("ShowFloatingWindow"));
	obs_properties_add_int_slider(props, "floating_window_width", obs_module_text("WindowWidth"), 320, 1920, 10);
	obs_properties_add_int_slider(props, "floating_window_height", obs_module_text("WindowHeight"), 240, 1080, 10);
#endif

	UNUSED_PARAMETER(data);
	return props;
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
	obs_data_set_default_int(settings, "inference_interval_frames", 3);
	obs_data_set_default_bool(settings, "show_bbox", true);
	obs_data_set_default_bool(settings, "show_label", true);
	obs_data_set_default_bool(settings, "show_confidence", true);
	obs_data_set_default_int(settings, "bbox_line_width", 2);
	obs_data_set_default_int(settings, "bbox_color", 0xFF00FF00);
	obs_data_set_default_bool(settings, "show_fov", false);
	obs_data_set_default_int(settings, "fov_radius", 200);
	obs_data_set_default_int(settings, "fov_color", 0xFFFF0000);
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
			tf->yoloModel->setTargetClass(tf->targetClassId);
		}
	}
	
	tf->showBBox = obs_data_get_bool(settings, "show_bbox");
	tf->showLabel = obs_data_get_bool(settings, "show_label");
	tf->showConfidence = obs_data_get_bool(settings, "show_confidence");
	tf->bboxLineWidth = (int)obs_data_get_int(settings, "bbox_line_width");
	tf->bboxColor = (uint32_t)obs_data_get_int(settings, "bbox_color");
	
	tf->showFOV = obs_data_get_bool(settings, "show_fov");
	tf->fovRadius = (int)obs_data_get_int(settings, "fov_radius");
	tf->fovColor = (uint32_t)obs_data_get_int(settings, "fov_color");

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
				BITMAPINFO bmi = {};
				bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bmi.bmiHeader.biWidth = filter->floatingWindowFrame.cols;
				bmi.bmiHeader.biHeight = -filter->floatingWindowFrame.rows;
				bmi.bmiHeader.biPlanes = 1;
				bmi.bmiHeader.biBitCount = 32;
				bmi.bmiHeader.biCompression = BI_RGB;
				SetDIBitsToDevice(hdc, 0, 0, filter->floatingWindowFrame.cols, filter->floatingWindowFrame.rows,
					0, 0, 0, filter->floatingWindowFrame.rows, filter->floatingWindowFrame.data, &bmi, DIB_RGB_COLORS);
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
	wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

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

	while (filter->inferenceRunning) {
		if (!filter->shouldInference) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}

		filter->shouldInference = false;

		if (!filter->isInferencing) {
			// 推理被禁用了，跳过
			continue;
		}

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

		if (filter->useRegion) {
			int fullWidth = fullFrame.cols;
			int fullHeight = fullFrame.rows;

			cropX = std::max(0, filter->regionX);
			cropY = std::max(0, filter->regionY);
			cropWidth = std::min(filter->regionWidth, fullWidth - cropX);
			cropHeight = std::min(filter->regionHeight, fullHeight - cropY);

			if (cropWidth > 0 && cropHeight > 0) {
				frame = fullFrame(cv::Rect(cropX, cropY, cropWidth, cropHeight)).clone();
			} else {
				frame = fullFrame.clone();
				cropX = 0;
				cropY = 0;
				cropWidth = fullWidth;
				cropHeight = fullHeight;
			}
		} else {
			frame = fullFrame.clone();
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
				int fullWidth = fullFrame.cols;
				int fullHeight = fullFrame.rows;
				
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

		{
			std::lock_guard<std::mutex> lock(filter->detectionsMutex);
			filter->detections = std::move(newDetections);
		}

		filter->inferenceCount++;
		filter->avgInferenceTimeMs = (filter->avgInferenceTimeMs * (filter->inferenceCount - 1) + duration) / filter->inferenceCount;

		if (filter->exportCoordinates && !newDetections.empty()) {
			exportCoordinatesToFile(filter, frame.cols, frame.rows);
		}
	}

	obs_log(LOG_INFO, "[YOLO Detector] Inference thread stopped");
}

static void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (!filter->showFOV) {
		return;
	}

	gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
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
		instance->totalFrames = 0;
		instance->inferenceCount = 0;
		instance->avgInferenceTimeMs = 0.0;
		instance->isInferencing = true;

#ifdef _WIN32
		instance->showFloatingWindow = false;
		instance->floatingWindowWidth = 640;
		instance->floatingWindowHeight = 480;
		instance->floatingWindowX = 0;
		instance->floatingWindowY = 0;
		instance->floatingWindowDragging = false;
		instance->floatingWindowHandle = nullptr;
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

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf.get(), width, height)) {
		return;
	}

	tf->totalFrames++;
	tf->frameCounter++;

	if (tf->frameCounter >= tf->inferenceIntervalFrames) {
			tf->frameCounter = 0;
			tf->shouldInference = true;
			// 减少日志输出，只在调试模式下输出
			// obs_log(LOG_DEBUG, "[YOLO Detector] Set shouldInference = true, isInferencing = %d, yoloModel valid = %d", 
			// 	(int)tf->isInferencing, 
			// 	tf->yoloModel ? 1 : 0);
		}
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

	if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	gs_blend_state_push();
	gs_reset_blend_state();

	cv::Mat outputImage;
	gs_texture_t *renderTexture = nullptr;
	bool hasDetections = false;

	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		if (!tf->inputBGRA.empty()) {
			outputImage = tf->inputBGRA.clone();

			if (tf->showBBox) {
				std::lock_guard<std::mutex> detLock(tf->detectionsMutex);
				if (!tf->detections.empty()) {
					hasDetections = true;
					for (const auto& det : tf->detections) {
						int x = static_cast<int>(det.x * width);
						int y = static_cast<int>(det.y * height);
						int w = static_cast<int>(det.width * width);
						int h = static_cast<int>(det.height * height);

						cv::Scalar bboxColor(
							(tf->bboxColor >> 16) & 0xFF,
							(tf->bboxColor >> 8) & 0xFF,
							tf->bboxColor & 0xFF
						);

						cv::rectangle(outputImage, cv::Rect(x, y, w, h), bboxColor, tf->bboxLineWidth);

						std::string labelText;
						if (tf->showLabel) {
							labelText = std::to_string(det.classId);
						}
						if (tf->showLabel && tf->showConfidence) {
							labelText += " ";
						}
						if (tf->showConfidence) {
							char confidenceStr[32];
							snprintf(confidenceStr, sizeof(confidenceStr), "%.2f", det.confidence);
							labelText += confidenceStr;
						}

						if (!labelText.empty()) {
							int fontFace = cv::FONT_HERSHEY_SIMPLEX;
							double fontScale = 0.5;
							int thickness = 2;
							int baseline = 0;

							cv::Size textSize = cv::getTextSize(labelText, fontFace, fontScale, thickness, &baseline);

							int textY = y - 5;
							if (textY < 0) {
								textY = y + textSize.height + 5;
							}

							cv::rectangle(outputImage,
								cv::Rect(x, textY - textSize.height - 5, textSize.width, textSize.height + 10),
								bboxColor,
								-1);

							cv::putText(outputImage,
								labelText,
								cv::Point(x, textY),
								fontFace,
								fontScale,
								cv::Scalar(255, 255, 255),
								thickness);
						}
					}
				}
			}
		}
	}

	if (!outputImage.empty()) {
		renderTexture = gs_texture_create(width, height, GS_BGRA, 1, (const uint8_t**)&outputImage.data, 0);
	}

	if (renderTexture) {
		gs_draw_sprite(renderTexture, 0, width, height);
	} else {
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
		obs_source_video_render(target);
	}

	if (tf->showFOV) {
		renderFOV(tf.get(), width, height);
	}

	obs_source_process_filter_end(tf->source, obs_get_base_effect(OBS_EFFECT_DEFAULT), width, height);
	gs_blend_state_pop();

	if (renderTexture) {
		gs_texture_destroy(renderTexture);
	}

#ifdef _WIN32
	if (tf->showFloatingWindow && !outputImage.empty()) {
		int cropWidth = tf->floatingWindowWidth;
		int cropHeight = tf->floatingWindowHeight;

		int centerX = outputImage.cols / 2;
		int centerY = outputImage.rows / 2;

		int cropX = std::max(0, centerX - cropWidth / 2);
		int cropY = std::max(0, centerY - cropHeight / 2);

		int actualCropWidth = std::min(cropWidth, outputImage.cols - cropX);
		int actualCropHeight = std::min(cropHeight, outputImage.rows - cropY);

		if (actualCropWidth > 0 && actualCropHeight > 0) {
			cv::Mat croppedFrame = outputImage(cv::Rect(cropX, cropY, actualCropWidth, actualCropHeight)).clone();

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
