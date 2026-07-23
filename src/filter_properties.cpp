#include "yolo_detector_filter.h"
#include "yolo-detector-filter.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
bool onPageChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *data);
bool onKalmanTrackerChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *data);
bool onNeuralPathChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *data);
bool onConfigChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *data);
#include <commdlg.h>
#include "MouseController.hpp"
#include "ConfigManager.hpp"
#endif

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <plugin-support.h>
#include "obs-utils/obs-utils.h"
#include "consts.h"

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
	obs_property_list_add_int(pageList, "准星检测", 7);
	obs_property_set_modified_callback(pageList, onPageChanged);

	obs_properties_add_group(props, "model_group", obs_module_text("ModelConfiguration"), OBS_GROUP_NORMAL, nullptr);
	obs_property_t *modelPathProp = obs_properties_add_path(props, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE, "ONNX Models (*.onnx)", nullptr);
	obs_property_set_long_description(modelPathProp, "选择YOLO ONNX模型文件路径");
	obs_property_t *modelVersion = obs_properties_add_list(props, "model_version", obs_module_text("ModelVersion"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(modelVersion, "YOLOv5", static_cast<int>(IYoloModel::Version::YOLOv5));
	obs_property_list_add_int(modelVersion, "YOLOv8", static_cast<int>(IYoloModel::Version::YOLOv8));
	obs_property_list_add_int(modelVersion, "YOLOv11", static_cast<int>(IYoloModel::Version::YOLOv11));
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
#ifdef HAVE_NCNN
	obs_property_list_add_string(useGPUList, "ncnn (Vulkan)", USEGPU_NCNN);
#endif
	obs_property_set_long_description(useGPUList, "选择推理设备（CUDA/GPU/DirectML/ncnn/CPU）");
	
#ifdef _WIN32
	obs_property_t *useGpuTextureProp = obs_properties_add_bool(props, "use_gpu_texture_inference", "启用GPU纹理推理");
	obs_property_set_long_description(useGpuTextureProp, "直接在GPU上处理纹理，避免GPU-CPU数据传输（支持DML设备）。渲染线程完成预处理，推理线程直接消费float buffer。");
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

	// KalmanFilter 追踪设置
	obs_property_t *useKalmanTrackerProp = obs_properties_add_bool(props, "use_kalman_tracker", "启用卡尔曼追踪");
	obs_property_set_long_description(useKalmanTrackerProp, "启用卡尔曼滤波器进行目标追踪，提供更稳定的目标ID和预测能力");
	obs_property_set_modified_callback(useKalmanTrackerProp, onKalmanTrackerChanged);
	obs_property_t *kalmanGenerateThresholdProp = obs_properties_add_int_slider(props, "kalman_generate_threshold", "追踪确认阈值", 1, 10, 1);
	obs_property_set_long_description(kalmanGenerateThresholdProp, "目标需要连续检测到的帧数才能被确认追踪");
	obs_property_t *kalmanTerminateCountProp = obs_properties_add_int_slider(props, "kalman_terminate_count", "追踪丢失阈值", 1, 10, 1);
	obs_property_set_long_description(kalmanTerminateCountProp, "目标丢失多少帧后停止追踪");
	obs_property_t *showKalmanPredictionsProp = obs_properties_add_bool(props, "show_kalman_predictions", "显示预测位置");
	obs_property_set_long_description(showKalmanPredictionsProp, "在画面上显示卡尔曼滤波器预测的目标位置（青色虚线框）");
	obs_property_t *kalmanPredictionFramesProp = obs_properties_add_int_slider(props, "kalman_prediction_frames", "预测帧数", 1, 20, 1);
	obs_property_set_long_description(kalmanPredictionFramesProp, "卡尔曼滤波器预测的未来帧数，用于显示预测轨迹");
	obs_property_t *showKalmanTrajectoriesProp = obs_properties_add_bool(props, "show_kalman_trajectories", "显示预测轨迹");
	obs_property_set_long_description(showKalmanTrajectoriesProp, "在画面上显示卡尔曼滤波器预测的轨迹线（黄色）");

	// 神经网络轨迹生成器设置
	obs_property_t *enableNeuralPathProp = obs_properties_add_bool(props, "enable_neural_path", "启用神经网络轨迹");
	obs_property_set_long_description(enableNeuralPathProp, "启用神经网络轨迹生成器，生成更自然的鼠标移动轨迹");
	obs_property_set_modified_callback(enableNeuralPathProp, onNeuralPathChanged);
	obs_property_t *neuralPathPointsProp = obs_properties_add_int_slider(props, "neural_path_points", "轨迹点数量", 10, 100, 5);
	obs_property_set_long_description(neuralPathPointsProp, "轨迹点数量，越多越平滑但移动越慢");
	obs_property_t *neuralMouseStepSizeProp = obs_properties_add_float_slider(props, "neural_mouse_step_size", "鼠标步长", 1.0, 20.0, 0.5);
	obs_property_set_long_description(neuralMouseStepSizeProp, "每次移动的步长大小");
	obs_property_t *neuralTargetRadiusProp = obs_properties_add_int_slider(props, "neural_target_radius", "目标半径", 1, 50, 1);
	obs_property_set_long_description(neuralTargetRadiusProp, "到达目标的判定半径");
	obs_property_t *neuralConsumeProp = obs_properties_add_int_slider(props, "neural_consume_per_frame", "每帧消费点数", 1, 5, 1);
	obs_property_set_long_description(neuralConsumeProp, "每帧消费的路径点数量，越大移动越快但曲线越粗略（1=拟人,2=平衡,3+=快速）");
	obs_property_t *enableNeuralPathDebugProp = obs_properties_add_bool(props, "enable_neural_path_debug", "神经网络调试日志");
	obs_property_set_long_description(enableNeuralPathDebugProp, "开启后会输出详细的神经网络轨迹运行日志，用于调试");

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
		obs_property_list_add_int(controllerTypeList, "罗技/雷蛇驱动", 2);
		obs_property_list_add_int(controllerTypeList, "UU remote GvInput", 3);
		obs_property_list_add_int(controllerTypeList, "NtUserSendInput", 5);
		obs_property_list_add_int(controllerTypeList, "NtUserInjectMouse", 6);
		obs_property_list_add_int(controllerTypeList, "NtUserInjectPointer", 7);
        obs_property_set_long_description(controllerTypeList, "mouse control: WindowsAPI=system API, MAKCU=serial, Logi/Razer=kernel driver, GvInput=Netease WHQL HID, NtUserSendInput=direct NtUserSendInput call, NtUserInjectMouse=virtual pointer device injection, NtUserInjectPointer=low-level pointer injection");
		obs_property_set_modified_callback(controllerTypeList, onConfigChanged);

		snprintf(propName, sizeof(propName), "logi_driver_type_%d", i);
		obs_property_t *logiDriverTypeList = obs_properties_add_list(props, propName, "驱动子类型", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
		obs_property_list_add_int(logiDriverTypeList, "自动检测", 0);
		obs_property_list_add_int(logiDriverTypeList, "Logitech G HUB", 1);
		obs_property_list_add_int(logiDriverTypeList, "Logitech LGS", 2);
		obs_property_list_add_int(logiDriverTypeList, "Razer Synapse", 3);
		obs_property_set_long_description(logiDriverTypeList, "驱动子类型：自动检测优先尝试所有驱动，或强制指定特定驱动");
		obs_property_set_modified_callback(logiDriverTypeList, onConfigChanged);

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
		
		snprintf(propName, sizeof(propName), "adaptive_p_gain_rate_%d", i);
		obs_property_t *adaptPRateProp = obs_properties_add_float_slider(props, propName, "自适应P变化率", 0.001, 0.5, 0.001);
		obs_property_set_long_description(adaptPRateProp, "自适应P增益变化率（对应专业PID的预测参数），值越大P增益调整越快");
		
		snprintf(propName, sizeof(propName), "d_term_scale_%d", i);
		obs_property_t *dScaleProp = obs_properties_add_float_slider(props, propName, "D项缩放因子", 0.0, 1.0, 0.01);
		obs_property_set_long_description(dScaleProp, "D项缩放因子（对应专业PID的采样率），值越大D项影响越大");

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

		// 积分参数设置
		snprintf(propName, sizeof(propName), "integral_limit_%d", i);
		obs_property_t *integralLimitProp = obs_properties_add_float_slider(props, propName, "积分限幅", 0.0, 500.0, 1.0);
		obs_property_set_long_description(integralLimitProp, "积分项上限，防止积分饱和");
		snprintf(propName, sizeof(propName), "integral_rate_%d", i);
		obs_property_t *integralRateProp = obs_properties_add_float_slider(props, propName, "积分速率", 0.0, 2.0, 0.1);
		obs_property_set_long_description(integralRateProp, "积分增长速率系数，值越大积分越快");
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
	
	// Smith预估器分组（可折叠）
	for (int i = 0; i < 5; i++) {
		char propName[64];
		snprintf(propName, sizeof(propName), "smith_predictor_group_%d", i);
		obs_properties_t *smithProps = obs_properties_create();

		snprintf(propName, sizeof(propName), "smith_enabled_%d", i);
		obs_property_t *smithEnabledProp = obs_properties_add_bool(smithProps, propName, "Smith预估器(纯滞后补偿)");
		obs_property_set_long_description(smithEnabledProp, "用过程模型从反馈回路剔除YOLO推理延迟，使PID可用更高增益而不振荡");

		snprintf(propName, sizeof(propName), "smith_model_gain_%d", i);
		obs_property_t *smithGainProp = obs_properties_add_float_slider(smithProps, propName, "模型增益K", 0.1f, 5.0f, 0.1f);
		obs_property_set_long_description(smithGainProp, "被控对象静态增益，默认1.0即可");

		snprintf(propName, sizeof(propName), "smith_model_tau_%d", i);
		obs_property_t *smithTauProp = obs_properties_add_float_slider(smithProps, propName, "预估纯滞后τ(秒)", 0.005f, 0.2f, 0.005f);
		obs_property_set_long_description(smithTauProp, "手动指定纯滞后时间。若开启自动τ则忽略此值");

		snprintf(propName, sizeof(propName), "smith_auto_tau_%d", i);
		obs_property_t *smithAutoTauProp = obs_properties_add_bool(smithProps, propName, "自动τ(使用实测推理延迟)");
		obs_property_set_long_description(smithAutoTauProp, "自动用 avgInferenceTimeMs 作为纯滞后τ，推荐开启");

		snprintf(propName, sizeof(propName), "smith_predictor_group_%d", i);
		obs_properties_add_group(props, propName, "Smith预估器", OBS_GROUP_CHECKABLE, smithProps);
	}

	// SlewRate控制器分组（限速平滑趋近+阻尼制动）
	for (int i = 0; i < 5; i++) {
		char propName[64];

		snprintf(propName, sizeof(propName), "slew_rate_group_%d", i);
		obs_properties_t *slewRateProps = obs_properties_create();

		snprintf(propName, sizeof(propName), "slew_rate_output_gain_%d", i);
		obs_property_t *slewGainProp = obs_properties_add_float_slider(slewRateProps, propName, "输出增益", 0.0f, 3.0f, 0.01f);
		obs_property_set_long_description(slewGainProp, "基础输出增益系数，默认0.25");

		snprintf(propName, sizeof(propName), "slew_rate_response_smoothing_%d", i);
		obs_property_t *slewSmoothProp = obs_properties_add_float_slider(slewRateProps, propName, "响应平滑系数", 0.0f, 0.01f, 0.0001f);
		obs_property_set_long_description(slewSmoothProp, "趋近项低通平滑系数，小值=更平滑，默认0.0008");

		snprintf(propName, sizeof(propName), "slew_rate_approach_damping_%d", i);
		obs_property_t *slewDampProp = obs_properties_add_float_slider(slewRateProps, propName, "趋近阻尼", 0.0f, 20.0f, 0.1f);
		obs_property_set_long_description(slewDampProp, "误差持续缩小时的阻尼强度，抑制超调，默认5.0");

		snprintf(propName, sizeof(propName), "slew_rate_update_interval_ms_%d", i);
		obs_property_t *slewIntervalProp = obs_properties_add_float_slider(slewRateProps, propName, "更新间隔(ms)", 1.0f, 50.0f, 0.5f);
		obs_property_set_long_description(slewIntervalProp, "控制器内部更新间隔，默认5.0ms");

		snprintf(propName, sizeof(propName), "slew_rate_normalization_scale_%d", i);
		obs_property_t *slewNormProp = obs_properties_add_float_slider(slewRateProps, propName, "归一化缩放", 1.0f, 30.0f, 0.5f);
		obs_property_set_long_description(slewNormProp, "对误差进行归一化的缩放系数，默认5.0");

		snprintf(propName, sizeof(propName), "slew_rate_group_%d", i);
		obs_properties_add_group(props, propName, "SlewRate控制器(限速平滑趋近)", OBS_GROUP_CHECKABLE, slewRateProps);
	}

	// 自适应PID控制器分组（位置式+自适应积分增益+积分死区+双重抗饱和）
	for (int i = 0; i < 5; i++) {
		char propName[64];

		snprintf(propName, sizeof(propName), "adaptive_pid_group_%d", i);
		obs_properties_t *adaptivePidProps = obs_properties_create();

		snprintf(propName, sizeof(propName), "adaptive_pid_kp_%d", i);
		obs_property_t *adaptKpProp = obs_properties_add_float_slider(adaptivePidProps, propName, "比例增益 Kp", 0.0f, 5.0f, 0.01f);
		obs_property_set_long_description(adaptKpProp, "位置式PID比例系数，默认1.0");

		snprintf(propName, sizeof(propName), "adaptive_pid_ki_%d", i);
		obs_property_t *adaptKiProp = obs_properties_add_float_slider(adaptivePidProps, propName, "积分增益 Ki", 0.0f, 1.0f, 0.001f);
		obs_property_set_long_description(adaptKiProp, "积分系数，消除稳态误差，默认0.1");

		snprintf(propName, sizeof(propName), "adaptive_pid_kd_%d", i);
		obs_property_t *adaptKdProp = obs_properties_add_float_slider(adaptivePidProps, propName, "微分增益 Kd", 0.0f, 1.0f, 0.001f);
		obs_property_set_long_description(adaptKdProp, "微分系数，抑制超调，默认0.05");

		snprintf(propName, sizeof(propName), "adaptive_pid_dead_zone_%d", i);
		obs_property_t *adaptDzProp = obs_properties_add_float_slider(adaptivePidProps, propName, "输入死区", 0.0f, 10.0f, 0.01f);
		obs_property_set_long_description(adaptDzProp, "误差小于此值时置零，避免微小抖动，默认0.3");

		snprintf(propName, sizeof(propName), "adaptive_pid_integral_limit_%d", i);
		obs_property_t *adaptILimitProp = obs_properties_add_float_slider(adaptivePidProps, propName, "积分限幅", 1.0f, 500.0f, 1.0f);
		obs_property_set_long_description(adaptILimitProp, "积分累积限幅，防止积分饱和，默认100.0");

		snprintf(propName, sizeof(propName), "adaptive_pid_integral_deadzone_%d", i);
		obs_property_t *adaptIDzProp = obs_properties_add_float_slider(adaptivePidProps, propName, "积分死区", 0.0f, 10.0f, 0.01f);
		obs_property_set_long_description(adaptIDzProp, "积分累积小于此值时忽略积分项，默认1.0");

		snprintf(propName, sizeof(propName), "adaptive_pid_integral_gain_threshold_%d", i);
		obs_property_t *adaptIGainThrProp = obs_properties_add_float_slider(adaptivePidProps, propName, "积分自适应阈值", 1.0f, 200.0f, 1.0f);
		obs_property_set_long_description(adaptIGainThrProp, "误差小于此值时积分增强，大于时衰减，默认50.0");

		snprintf(propName, sizeof(propName), "adaptive_pid_integral_gain_rate_%d", i);
		obs_property_t *adaptIGainRateProp = obs_properties_add_float_slider(adaptivePidProps, propName, "积分自适应速率", 0.001f, 0.1f, 0.001f);
		obs_property_set_long_description(adaptIGainRateProp, "积分增益自适应调整速率，默认0.015");

		snprintf(propName, sizeof(propName), "adaptive_pid_output_limit_%d", i);
		obs_property_t *adaptOutLimitProp = obs_properties_add_float_slider(adaptivePidProps, propName, "输出限幅", 1.0f, 200.0f, 1.0f);
		obs_property_set_long_description(adaptOutLimitProp, "单帧输出最大值，默认10.0");

		snprintf(propName, sizeof(propName), "adaptive_pid_group_%d", i);
		obs_properties_add_group(props, propName, "自适应PID控制器(位置式+自适应积分)", OBS_GROUP_CHECKABLE, adaptivePidProps);
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
		
		// GhostTracker曲线轨迹参数
		obs_properties_t *ghostProps = obs_properties_create();
		
		snprintf(propName, sizeof(propName), "ghost_curvature_%d", i);
		obs_property_t *ghostCurvProp = obs_properties_add_float_slider(ghostProps, propName, "曲线强度", 0.0f, 1.0f, 0.05f);
		obs_property_set_long_description(ghostCurvProp, "曲线轨迹的弯曲强度，值越大曲线越弯曲");
		
		snprintf(propName, sizeof(propName), "ghost_noise_intensity_%d", i);
		obs_property_t *ghostNoiseProp = obs_properties_add_float_slider(ghostProps, propName, "噪声强度", 0.0f, 30.0f, 1.0f);
		obs_property_set_long_description(ghostNoiseProp, "Perlin噪声的最大像素偏移，模拟手部抖动");
		
		snprintf(propName, sizeof(propName), "ghost_vertical_snap_%d", i);
		obs_property_t *ghostSnapProp = obs_properties_add_float_slider(ghostProps, propName, "垂直吸附比例", 1.0f, 10.0f, 0.5f);
		obs_property_set_long_description(ghostSnapProp, "垂直方向吸附阈值，值越大越容易锁定垂直轨道");
		
		snprintf(propName, sizeof(propName), "ghost_noise_freq_%d", i);
		obs_property_t *ghostFreqProp = obs_properties_add_float_slider(ghostProps, propName, "噪声频率", 0.1f, 2.0f, 0.1f);
		obs_property_set_long_description(ghostFreqProp, "Perlin噪声的变化频率，值越大抖动越快");
		
		snprintf(propName, sizeof(propName), "ghost_tracker_group_%d", i);
		obs_properties_add_group(props, propName, "曲线轨迹(GhostTracker)", OBS_GROUP_CHECKABLE, ghostProps);
		
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
	obs_properties_add_text(props, "dml_stats", "DML直推统计", OBS_TEXT_INFO);

	// 页面6: PID参数 - 算法选择放在最上面
	// 算法选择（全局）
	obs_property_t *algorithmTypeList = obs_properties_add_list(props, "algorithm_type_global", "控制算法", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(algorithmTypeList, "高级PID (动态P增益)", 0);
	obs_property_list_add_int(algorithmTypeList, "专业PID (卡尔曼滤波)", 1);
	obs_property_list_add_int(algorithmTypeList, "aim 控制器 (增量式PID+预测+噪声)", 2);
	obs_property_set_long_description(algorithmTypeList, "选择控制算法：高级PID包含动态P增益、预测等功能；专业PID内置卡尔曼滤波和自适应增益；aim 控制器集成增量式PID+运动预测+柏林噪声");
	obs_property_set_modified_callback(algorithmTypeList, onPageChanged);
	
	// 专业PID参数组
	obs_properties_add_group(props, "external_pid_group", "专业PID配置", OBS_GROUP_NORMAL, nullptr);
	obs_property_t *extKpXProp = obs_properties_add_float_slider(props, "external_kp_x", "X轴-Kp", 0.0, 10.0, 0.01);
	obs_property_set_long_description(extKpXProp, "专业PID X轴比例系数");
	obs_property_t *extKiXProp = obs_properties_add_float_slider(props, "external_ki_x", "X轴-Ki", 0.0, 5.0, 0.001);
	obs_property_set_long_description(extKiXProp, "专业PID X轴积分系数");
	obs_property_t *extKdXProp = obs_properties_add_float_slider(props, "external_kd_x", "X轴-Kd", 0.0, 10.0, 0.01);
	obs_property_set_long_description(extKdXProp, "专业PID X轴微分系数");
	obs_property_t *extKpYProp = obs_properties_add_float_slider(props, "external_kp_y", "Y轴-Kp", 0.0, 10.0, 0.01);
	obs_property_set_long_description(extKpYProp, "专业PID Y轴比例系数");
	obs_property_t *extKiYProp = obs_properties_add_float_slider(props, "external_ki_y", "Y轴-Ki", 0.0, 5.0, 0.001);
	obs_property_set_long_description(extKiYProp, "专业PID Y轴积分系数");
	obs_property_t *extKdYProp = obs_properties_add_float_slider(props, "external_kd_y", "Y轴-Kd", 0.0, 10.0, 0.01);
	obs_property_set_long_description(extKdYProp, "专业PID Y轴微分系数");
	obs_property_t *extPredictXProp = obs_properties_add_float_slider(props, "external_predict_x", "X轴-预测参数", 0.0, 5.0, 0.01);
	obs_property_set_long_description(extPredictXProp, "专业PID X轴预测参数");
	obs_property_t *extPredictYProp = obs_properties_add_float_slider(props, "external_predict_y", "Y轴-预测参数", 0.0, 5.0, 0.01);
	obs_property_set_long_description(extPredictYProp, "专业PID Y轴预测参数");
	obs_property_t *extRateXProp = obs_properties_add_float_slider(props, "external_rate_x", "X轴-采样率", 0.0, 1.0, 0.001);
	obs_property_set_long_description(extRateXProp, "专业PID X轴采样率");
	obs_property_t *extRateYProp = obs_properties_add_float_slider(props, "external_rate_y", "Y轴-采样率", 0.0, 1.0, 0.001);
	obs_property_set_long_description(extRateYProp, "专业PID Y轴采样率");
	obs_property_t *extKiModeProp = obs_properties_add_int_slider(props, "external_ki_mode", "积分模式", 0, 1, 1);
	obs_property_set_long_description(extKiModeProp, "积分模式: 0=标准积分, 1=自适应积分");
	obs_property_t *extKpLimitProp = obs_properties_add_float_slider(props, "external_kp_limit", "P项限幅", 0.0, 10000.0, 1.0);
	obs_property_set_long_description(extKpLimitProp, "P项输出限幅，0=不限幅");
	obs_property_t *extKiLimitProp = obs_properties_add_float_slider(props, "external_ki_limit", "I项限幅", 0.0, 10000.0, 1.0);
	obs_property_set_long_description(extKiLimitProp, "I项输出限幅，0=不限幅");
	obs_property_t *extKdLimitProp = obs_properties_add_float_slider(props, "external_kd_limit", "D项限幅", 0.0, 10000.0, 1.0);
	obs_property_set_long_description(extKdLimitProp, "D项输出限幅，0=不限幅");
	obs_property_t *extOutputLimitProp = obs_properties_add_float_slider(props, "external_output_limit", "总输出限幅", 0.0, 10000.0, 1.0);
	obs_property_set_long_description(extOutputLimitProp, "总输出限幅，0=不限幅");
	obs_property_t *extKiRateProp = obs_properties_add_float_slider(props, "external_ki_rate", "积分速率", 0.0, 1.0, 0.001);
	obs_property_set_long_description(extKiRateProp, "卡尔曼滤波器采样率，影响积分和平滑");
	obs_property_t *extKiDeadbandProp = obs_properties_add_float_slider(props, "external_ki_deadband", "积分死区", 0.0, 10.0, 0.01);
	obs_property_set_long_description(extKiDeadbandProp, "误差变化超过此值时重置状态");

	// aim 控制器参数组（增量式PID+运动预测+柏林噪声）
	// 使用子容器，组可见时子属性自动跟随
	obs_properties *aimProps = obs_properties_create();
	obs_property_t *aimKpProp = obs_properties_add_float_slider(aimProps, "aim_kp", "比例增益 Kp", 0.0, 3.0, 0.01);
	obs_property_set_long_description(aimKpProp, "aim 控制器比例增益，响应速度");
	obs_property_t *aimKiProp = obs_properties_add_float_slider(aimProps, "aim_ki", "积分增益 Ki", 0.0, 0.5, 0.001);
	obs_property_set_long_description(aimKiProp, "aim 控制器积分增益，消除稳态误差");
	obs_property_t *aimKdProp = obs_properties_add_float_slider(aimProps, "aim_kd", "微分增益 Kd", 0.0, 0.2, 0.001);
	obs_property_set_long_description(aimKdProp, "aim 控制器微分增益，抑制超调");
	obs_property_t *aimNoiseEnabledProp = obs_properties_add_bool(aimProps, "aim_noise_enabled", "启用人类化抖动");
	obs_property_set_long_description(aimNoiseEnabledProp, "启用柏林噪声模拟人类操作的自然抖动");
	obs_property_set_modified_callback(aimNoiseEnabledProp, onPageChanged);
	obs_property_t *aimNoiseAmpProp = obs_properties_add_float_slider(aimProps, "aim_noise_amplitude", "噪声幅度", 0.0, 20.0, 0.1);
	obs_property_set_long_description(aimNoiseAmpProp, "柏林噪声幅度（像素），仅启用抖动时生效");
	obs_property_t *aimPredWeightXProp = obs_properties_add_float_slider(aimProps, "aim_prediction_weight_x", "X轴预测权重", 0.0, 1.0, 0.01);
	obs_property_set_long_description(aimPredWeightXProp, "aim 控制器自带运动预测器的X轴权重");
	obs_property_t *aimPredWeightYProp = obs_properties_add_float_slider(aimProps, "aim_prediction_weight_y", "Y轴预测权重", 0.0, 1.0, 0.01);
	obs_property_set_long_description(aimPredWeightYProp, "aim 控制器自带运动预测器的Y轴权重");
	obs_property_t *aimRampTimeProp = obs_properties_add_float_slider(aimProps, "aim_ramp_time", "渐入时间(秒)", 0.0, 2.0, 0.01);
	obs_property_set_long_description(aimRampTimeProp, "从初始缩放到满输出的过渡时间");
	obs_property_t *aimInitScaleProp = obs_properties_add_float_slider(aimProps, "aim_init_scale", "初始缩放", 0.0, 1.0, 0.01);
	obs_property_set_long_description(aimInitScaleProp, "锁定瞬间的输出缩放比例，避免大幅移动");
	obs_property_t *aimOutputMaxProp = obs_properties_add_float_slider(aimProps, "aim_output_max", "最大输出", 1.0, 500.0, 1.0);
	obs_property_set_long_description(aimOutputMaxProp, "aim 控制器单帧最大输出幅度");
	obs_properties_add_group(props, "aim_controller_group", "aim 控制器配置", OBS_GROUP_NORMAL, aimProps);

	// ========== 页面7: 准星检测 ==========
#ifdef _WIN32
	obs_properties_add_group(props, "crosshair_group", "准星检测", OBS_GROUP_NORMAL, nullptr);
	obs_property_t *crosshairEnabledProp = obs_properties_add_bool(props, "crosshair_enabled", "启用准星检测");
	obs_property_set_long_description(crosshairEnabledProp, "使用HSV颜色+形态学+子矩阵分位数+模板匹配检测准星");
	
	// 吸管取色
	obs_property_t *crosshairPickColorProp = obs_properties_add_button(props, "crosshair_pick_color", "🎯 吸管取色（点击自动采集准星颜色）", [](obs_properties_t *, obs_property_t *, void *data) -> bool {
		auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
		if (!ptr) return false;
		auto tf = *ptr;
		if (!tf) return false;
		tf->crosshairConfig.pickingColor = true;
		return true;
	});
	obs_property_set_long_description(crosshairPickColorProp, "点击后自动从帧中心采样HSV颜色并设置范围");
	obs_property_t *crosshairColorInfoProp = obs_properties_add_text(props, "crosshair_color_info", "取色结果", OBS_TEXT_INFO);
	obs_property_set_long_description(crosshairColorInfoProp, "显示最近一次取色的RGB和HSV值");

	// 手动RGB输入（已知准星颜色时直接填入）
	obs_property_t *chManualRProp = obs_properties_add_int_slider(props, "crosshair_manual_r", "准星颜色 R", 0, 255, 1);
	obs_property_set_long_description(chManualRProp, "准星颜色的红色分量(0-255)，可从游戏设置中查看");
	obs_property_t *chManualGProp = obs_properties_add_int_slider(props, "crosshair_manual_g", "准星颜色 G", 0, 255, 1);
	obs_property_set_long_description(chManualGProp, "准星颜色的绿色分量(0-255)");
	obs_property_t *chManualBProp = obs_properties_add_int_slider(props, "crosshair_manual_b", "准星颜色 B", 0, 255, 1);
	obs_property_set_long_description(chManualBProp, "准星颜色的蓝色分量(0-255)");
	obs_property_t *crosshairApplyRgbProp = obs_properties_add_button(props, "crosshair_apply_rgb", "✅ 应用RGB颜色（转换为HSV范围）", [](obs_properties_t *, obs_property_t *, void *data) -> bool {
		auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
		if (!ptr) return false;
		auto tf = *ptr;
		if (!tf) return false;
		// 从settings读取RGB值
		obs_data_t *settings = obs_source_get_settings(tf->source);
		if (!settings) return false;
		int r = (int)obs_data_get_int(settings, "crosshair_manual_r");
		int g = (int)obs_data_get_int(settings, "crosshair_manual_g");
		int b = (int)obs_data_get_int(settings, "crosshair_manual_b");
		obs_data_release(settings);
		// 应用RGB→HSV
		tf->crosshairDetector.applyManualRgb(r, g, b);
		CrosshairDetectorConfig& chCfg = tf->crosshairConfig;
		chCfg.pickedR = r; chCfg.pickedG = g; chCfg.pickedB = b;
		chCfg.pickedH = tf->crosshairDetector.getConfig().pickedH;
		chCfg.pickedS = tf->crosshairDetector.getConfig().pickedS;
		chCfg.pickedV = tf->crosshairDetector.getConfig().pickedV;
		chCfg.hMin = tf->crosshairDetector.getConfig().hMin;
		chCfg.hMax = tf->crosshairDetector.getConfig().hMax;
		chCfg.sMin = tf->crosshairDetector.getConfig().sMin;
		chCfg.sMax = tf->crosshairDetector.getConfig().sMax;
		chCfg.vMin = tf->crosshairDetector.getConfig().vMin;
		chCfg.vMax = tf->crosshairDetector.getConfig().vMax;
		chCfg.colorPicked = true;
		tf->crosshairDetector.updateConfig(chCfg);
		// 回写settings
		obs_data_t *applySettings = obs_source_get_settings(tf->source);
		if (applySettings) {
			obs_data_set_int(applySettings, "crosshair_h_min", chCfg.hMin);
			obs_data_set_int(applySettings, "crosshair_h_max", chCfg.hMax);
			obs_data_set_int(applySettings, "crosshair_s_min", chCfg.sMin);
			obs_data_set_int(applySettings, "crosshair_s_max", chCfg.sMax);
			obs_data_set_int(applySettings, "crosshair_v_min", chCfg.vMin);
			obs_data_set_int(applySettings, "crosshair_v_max", chCfg.vMax);
			char infoText[256];
			snprintf(infoText, sizeof(infoText),
				"RGB(%d,%d,%d) HSV(%d,%d,%d) H[%d~%d] S[%d~%d] V[%d~%d]",
				r, g, b, chCfg.pickedH, chCfg.pickedS, chCfg.pickedV,
				chCfg.hMin, chCfg.hMax, chCfg.sMin, chCfg.sMax, chCfg.vMin, chCfg.vMax);
			obs_data_set_string(applySettings, "crosshair_color_info", infoText);
			obs_source_update(tf->source, applySettings);
			obs_data_release(applySettings);
		}
		return true;
	});
	obs_property_set_long_description(crosshairApplyRgbProp, "将手动输入的RGB颜色转换为HSV搜索范围并应用");

	// HSV手动微调
	obs_property_t *chHMinProp = obs_properties_add_int_slider(props, "crosshair_h_min", "H最小", 0, 180, 1);
	obs_property_set_long_description(chHMinProp, "HSV色调下限(0-180)");
	obs_property_t *chHMaxProp = obs_properties_add_int_slider(props, "crosshair_h_max", "H最大", 0, 180, 1);
	obs_property_set_long_description(chHMaxProp, "HSV色调上限(0-180)");
	obs_property_t *chSMinProp = obs_properties_add_int_slider(props, "crosshair_s_min", "S最小", 0, 255, 1);
	obs_property_set_long_description(chSMinProp, "HSV饱和度下限(0-255)");
	obs_property_t *chSMaxProp = obs_properties_add_int_slider(props, "crosshair_s_max", "S最大", 0, 255, 1);
	obs_property_set_long_description(chSMaxProp, "HSV饱和度上限(0-255)");
	obs_property_t *chVMinProp = obs_properties_add_int_slider(props, "crosshair_v_min", "V最小", 0, 255, 1);
	obs_property_set_long_description(chVMinProp, "HSV明度下限(0-255)");
	obs_property_t *chVMaxProp = obs_properties_add_int_slider(props, "crosshair_v_max", "V最大", 0, 255, 1);
	obs_property_set_long_description(chVMaxProp, "HSV明度上限(0-255)");

	// 取色容差
	obs_property_t *chHTolProp = obs_properties_add_int_slider(props, "crosshair_h_tolerance", "H容差", 1, 90, 1);
	obs_property_set_long_description(chHTolProp, "吸管取色时H通道的容差范围");
	obs_property_t *chSTolProp = obs_properties_add_int_slider(props, "crosshair_s_tolerance", "S容差", 1, 128, 1);
	obs_property_set_long_description(chSTolProp, "吸管取色时S通道的容差范围");
	obs_property_t *chVTolProp = obs_properties_add_int_slider(props, "crosshair_v_tolerance", "V容差", 1, 128, 1);
	obs_property_set_long_description(chVTolProp, "吸管取色时V通道的容差范围");

	// 形态学参数
	obs_property_t *chMorphKernelProp = obs_properties_add_int_slider(props, "crosshair_morph_kernel", "形态学核大小", 1, 15, 2);
	obs_property_set_long_description(chMorphKernelProp, "腐蚀/膨胀的核大小（仅奇数）");
	obs_property_t *chErodeIterProp = obs_properties_add_int_slider(props, "crosshair_erode_iter", "腐蚀迭代次数", 0, 5, 1);
	obs_property_set_long_description(chErodeIterProp, "腐蚀操作迭代次数，去噪点");
	obs_property_t *chDilateIterProp = obs_properties_add_int_slider(props, "crosshair_dilate_iter", "膨胀迭代次数", 0, 10, 1);
	obs_property_set_long_description(chDilateIterProp, "膨胀操作迭代次数，填充间隙");

	// 子矩阵分位数过滤
	obs_property_t *chGridRowsProp = obs_properties_add_int_slider(props, "crosshair_grid_rows", "网格行数", 2, 20, 1);
	obs_property_set_long_description(chGridRowsProp, "将二值图划分为多少行网格");
	obs_property_t *chGridColsProp = obs_properties_add_int_slider(props, "crosshair_grid_cols", "网格列数", 2, 20, 1);
	obs_property_set_long_description(chGridColsProp, "将二值图划分为多少列网格");
	obs_property_t *chQuantileProp = obs_properties_add_float_slider(props, "crosshair_quantile_threshold", "分位数阈值", 0.0, 1.0, 0.01);
	obs_property_set_long_description(chQuantileProp, "子区域白像素比例低于此值则清零，去除稀疏噪声");

	// 模板匹配
	obs_property_t *chTemplatePathProp = obs_properties_add_path(props, "crosshair_template_path", "模板图片路径", OBS_PATH_FILE, "Image Files (*.png *.jpg *.bmp)", nullptr);
	obs_property_set_long_description(chTemplatePathProp, "准星模板图片（灰度），留空则仅用轮廓定位");
	obs_property_t *chMatchThresholdProp = obs_properties_add_float_slider(props, "crosshair_match_threshold", "匹配阈值", 0.0, 1.0, 0.05);
	obs_property_set_long_description(chMatchThresholdProp, "模板匹配置信度阈值，低于此值不认为匹配成功");

	// 通用参数
	obs_property_t *chMinAreaProp = obs_properties_add_int_slider(props, "crosshair_min_area", "最小面积", 1, 10000, 10);
	obs_property_set_long_description(chMinAreaProp, "检测候选区域的最小像素面积");
	obs_property_t *chMaxAreaProp = obs_properties_add_int_slider(props, "crosshair_max_area", "最大面积", 1, 50000, 100);
	obs_property_set_long_description(chMaxAreaProp, "检测候选区域的最大像素面积");

	// 轮廓形状过滤
	obs_property_t *chShapeFilterProp = obs_properties_add_bool(props, "crosshair_shape_filter_enabled", "启用形状过滤");
	obs_property_set_long_description(chShapeFilterProp, "根据轮廓形状特征过滤，提高准心识别准确度");
	obs_property_t *chShapeTypeProp = obs_properties_add_list(props, "crosshair_shape_type", "准心形状", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(chShapeTypeProp, "任意形状", 0);
	obs_property_list_add_int(chShapeTypeProp, "十字形 (+字)", 1);
	obs_property_list_add_int(chShapeTypeProp, "点状 (圆点)", 2);
	obs_property_list_add_int(chShapeTypeProp, "T字形", 3);
	obs_property_set_long_description(chShapeTypeProp, "选择准心形状类型，CS:GO选十字形，三角洲选点状");
	obs_property_t *chMinFillRatioProp = obs_properties_add_float_slider(props, "crosshair_min_fill_ratio", "最小填充率", 0.0, 1.0, 0.05);
	obs_property_set_long_description(chMinFillRatioProp, "轮廓面积/包围盒面积的最小值，点状准心建议0.3以上");
	obs_property_t *chMaxFillRatioProp = obs_properties_add_float_slider(props, "crosshair_max_fill_ratio", "最大填充率", 0.0, 1.0, 0.05);
	obs_property_set_long_description(chMaxFillRatioProp, "轮廓面积/包围盒面积的最大值，十字准心建议0.5以下");
	obs_property_t *chMinAspectProp = obs_properties_add_float_slider(props, "crosshair_min_aspect_ratio", "最小纵横比", 0.1, 5.0, 0.1);
	obs_property_set_long_description(chMinAspectProp, "宽/高的最小值，十字和点状建议0.5以上");
	obs_property_t *chMaxAspectProp = obs_properties_add_float_slider(props, "crosshair_max_aspect_ratio", "最大纵横比", 0.1, 5.0, 0.1);
	obs_property_set_long_description(chMaxAspectProp, "宽/高的最大值，十字和点状建议2.0以下");

	obs_property_t *chSearchRadiusProp = obs_properties_add_int_slider(props, "crosshair_search_radius", "搜索半径", 0, 1920, 10);
	obs_property_set_long_description(chSearchRadiusProp, "准星搜索范围半径（像素，0=自动1/6帧宽），越小越快但可能漏检偏移大的准星");
	obs_property_t *chDetectIntervalProp = obs_properties_add_int_slider(props, "crosshair_detect_interval", "检测帧间隔", 1, 60, 1);
	obs_property_set_long_description(chDetectIntervalProp, "每隔多少帧检测一次，传统CV很快建议1");
	obs_property_t *chColorIsolationProp = obs_properties_add_bool(props, "crosshair_color_isolation", "🎨 颜色隔离视图");
	obs_property_set_long_description(chColorIsolationProp, "悬浮窗黑底只显示HSV匹配颜色+YOLO框，直观看到找色效果");
	obs_property_t *chDebugMaskProp = obs_properties_add_bool(props, "crosshair_debug_mask", "显示HSV掩码调试");
	obs_property_set_long_description(chDebugMaskProp, "在悬浮窗半透明叠加HSV二值化掩码用于调参");
#endif

	UNUSED_PARAMETER(data);

	// 初始化默认页面可见性：page==0时只显示page 0和1的控件
	// OBS加载属性窗口时不自动触发onPageChanged，手动初始化
	
	// 隐藏页面2-7的全局控件
	obs_property_t *p;
	
	// 动态FOV (page 1)
	p = obs_properties_get(props, "fov2_group"); if (p) obs_property_set_visible(p, false);
	p = obs_properties_get(props, "dynamic_fov_game_hfov"); if (p) obs_property_set_visible(p, false);
	p = obs_properties_get(props, "dynamic_fov_transition_time"); if (p) obs_property_set_visible(p, false);
	
	// page 2-7 配置选择器
	p = obs_properties_get(props, "mouse_config_select"); if (p) obs_property_set_visible(p, false);
	p = obs_properties_get(props, "algorthm_type_global"); if (p) obs_property_set_visible(p, false);
	p = obs_properties_get(props, "test_makcu_connection"); if (p) obs_property_set_visible(p, false);
	
	// page 3 PID全局
	p = obs_properties_get(props, "dynamic_pid_group"); if (p) obs_property_set_visible(p, false);
	p = obs_properties_get(props, "external_pid_group"); if (p) obs_property_set_visible(p, false);
	p = obs_properties_get(props, "aim_controller_group"); if (p) obs_property_set_visible(p, false);
	
	// page 5 追踪
	p = obs_properties_get(props, "tracking_group"); if (p) obs_property_set_visible(p, false);
	
	// page 7 准星
	p = obs_properties_get(props, "crosshair_group"); if (p) obs_property_set_visible(p, false);
	p = obs_properties_get(props, "crosshair_enabled"); if (p) obs_property_set_visible(p, false);
	
	// page 8 配置管理全局（如果存在）
	p = obs_properties_get(props, "config_management_group"); if (p) obs_property_set_visible(p, false);

	// 隐藏所有5组配置下的分组（它们在page 2-7中按需显示）
	for (int i = 0; i < 5; i++) {
		char propName[64];
		// 自动扳机
		snprintf(propName, sizeof(propName), "auto_trigger_group_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		// 后坐力控制
		snprintf(propName, sizeof(propName), "recoil_control_group_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		// 导数预测器
		snprintf(propName, sizeof(propName), "derivative_predictor_group_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		// Smith预估器
		snprintf(propName, sizeof(propName), "smith_predictor_group_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		// IMM交互多模型
		snprintf(propName, sizeof(propName), "imm_filter_group_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		// 贝塞尔曲线
		snprintf(propName, sizeof(propName), "bezier_movement_group_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		// 曲线轨迹
		snprintf(propName, sizeof(propName), "curve_trajectory_group_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		
		// ====== 配置级PID & 热键 ======
		snprintf(propName, sizeof(propName), "global_mouse_enabled_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "global_hotkey_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "global_target_lock_hotkey_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		
		// PID基础 (Kp/Ki/Kd + 高级)
		snprintf(propName, sizeof(propName), "global_kp_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "global_ki_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "global_kd_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		
		// 鼠标高级参数
		snprintf(propName, sizeof(propName), "intensity_mode_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "intensity_value_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "smooth_factor_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "anti_shake_threshold_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "mouse_method_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "logi_device_selector_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "razer_device_selector_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
		snprintf(propName, sizeof(propName), "gvinput_config_%d", i);
		p = obs_properties_get(props, propName); if (p) obs_property_set_visible(p, false);
	}

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
	snprintf(propName, sizeof(propName), "logi_driver_type_%d", configIndex);
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
	snprintf(propName, sizeof(propName), "integral_rate_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "d_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "i_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "derivative_filter_alpha_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "adaptive_p_gain_rate_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "d_term_scale_%d", configIndex);
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
	// GhostTracker曲线轨迹参数可见性
	snprintf(propName, sizeof(propName), "ghost_curvature_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "ghost_noise_intensity_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "ghost_vertical_snap_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "ghost_noise_freq_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	snprintf(propName, sizeof(propName), "ghost_tracker_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
}

// 设置预测器配置页面的控件可见性
static void setPredictorPropertiesVisible(obs_properties_t *props, int configIndex, bool visible)
{
	char propName[64];
	// 导数预测器分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "derivative_predictor_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	// Smith预估器分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "smith_predictor_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	// IMM 交互多模型
	snprintf(propName, sizeof(propName), "imm_filter_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	// SlewRate控制器分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "slew_rate_group_%d", configIndex);
	obs_property_set_visible(obs_properties_get(props, propName), visible);
	// 自适应PID控制器分组（CHECKABLE，勾选即启用）
	snprintf(propName, sizeof(propName), "adaptive_pid_group_%d", configIndex);
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

bool onConfigChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
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
		setBezierMovementPropertiesVisible(props, i, isCurrentConfig && page == 6);
	}

	// 根据控制器类型动态显示/隐藏专属字段
	for (int i = 0; i < 5; i++) {
		bool isCurrentConfig = (i == currentConfig);
		if (!isCurrentConfig || page != 2) continue;

		char propName[64];
		snprintf(propName, sizeof(propName), "controller_type_%d", i);
		int ctrlType = (int)obs_data_get_int(settings, propName);

		bool showMakcu = (ctrlType == 1);     // MAKCU
		bool showLogi = (ctrlType == 2);      // LogiDriver
		// showGvInput = (ctrlType == 3);    // GvInput has no exclusive fields to hide

		snprintf(propName, sizeof(propName), "makcu_port_%d", i);
		obs_property_set_visible(obs_properties_get(props, propName), showMakcu);
		snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
		obs_property_set_visible(obs_properties_get(props, propName), showMakcu);
		snprintf(propName, sizeof(propName), "logi_driver_type_%d", i);
		obs_property_set_visible(obs_properties_get(props, propName), showLogi);
	}

	// 动态PID参数只在algorithm == 3时显示
	obs_property_set_visible(obs_properties_get(props, "dynamic_pid_group"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kp"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_ki"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kd"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_target_threshold"), page == 3 && algorithm == 3);

	// 专业PID参数只在algorithm == 1时显示
	obs_property_set_visible(obs_properties_get(props, "external_pid_group"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kp_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kd_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kp_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kd_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_predict_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_predict_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_rate_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_rate_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_mode"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kp_limit"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_limit"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kd_limit"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_output_limit"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_rate"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_deadband"), page == 3 && algorithm == 1);

	// aim 控制器参数只在 algorithm == 2 时显示
	// 子属性在子容器中，组可见时自动跟随；仅 aim_noise_amplitude 需根据开关单独控制
	bool aimVisible = (page == 3 && algorithm == 2);
	bool aimNoiseVisible = aimVisible && obs_data_get_bool(settings, "aim_noise_enabled");
	obs_property_set_visible(obs_properties_get(props, "aim_controller_group"), aimVisible);
	obs_property_set_visible(obs_properties_get(props, "aim_noise_amplitude"), aimNoiseVisible);

	obs_property_set_visible(obs_properties_get(props, "mouse_config_select"), page == 2 || page == 3 || page == 4 || page == 6 || page == 7);
	obs_property_set_visible(obs_properties_get(props, "test_makcu_connection"), page == 2);

	return true;
}

bool onKalmanTrackerChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	bool useKalman = obs_data_get_bool(settings, "use_kalman_tracker");
	
	obs_property_set_visible(obs_properties_get(props, "kalman_generate_threshold"), useKalman);
	obs_property_set_visible(obs_properties_get(props, "kalman_terminate_count"), useKalman);
	obs_property_set_visible(obs_properties_get(props, "show_kalman_predictions"), useKalman);
	obs_property_set_visible(obs_properties_get(props, "kalman_prediction_frames"), useKalman);
	obs_property_set_visible(obs_properties_get(props, "show_kalman_trajectories"), useKalman);
	
	return true;
}

bool onNeuralPathChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
{
	bool useNeuralPath = obs_data_get_bool(settings, "enable_neural_path");
	
	obs_property_set_visible(obs_properties_get(props, "neural_path_points"), useNeuralPath);
	obs_property_set_visible(obs_properties_get(props, "neural_mouse_step_size"), useNeuralPath);
	obs_property_set_visible(obs_properties_get(props, "neural_target_radius"), useNeuralPath);
	obs_property_set_visible(obs_properties_get(props, "enable_neural_path_debug"), useNeuralPath);
	
	return true;
}

bool onPageChanged(obs_properties_t *props, obs_property_t *property, obs_data_t *settings)
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

#ifdef _WIN32
	// 配置选择器在鼠标控制页面(2,3,4)和预测与滤波页面(6)显示
	obs_property_set_visible(obs_properties_get(props, "mouse_config_select"), page == 2 || page == 3 || page == 4 || page == 6 || page == 7);

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

	// 根据控制器类型动态显示/隐藏专属字段
	for (int i = 0; i < 5; i++) {
		bool isCurrentConfig = (i == currentConfig);
		if (!isCurrentConfig || page != 2) continue;

		char propName[64];
		snprintf(propName, sizeof(propName), "controller_type_%d", i);
		int ctrlType = (int)obs_data_get_int(settings, propName);

		bool showMakcu = (ctrlType == 1);     // MAKCU
		bool showLogi = (ctrlType == 2);      // LogiDriver

		snprintf(propName, sizeof(propName), "makcu_port_%d", i);
		obs_property_set_visible(obs_properties_get(props, propName), showMakcu);
		snprintf(propName, sizeof(propName), "makcu_baud_rate_%d", i);
		obs_property_set_visible(obs_properties_get(props, propName), showMakcu);
		snprintf(propName, sizeof(propName), "logi_driver_type_%d", i);
		obs_property_set_visible(obs_properties_get(props, propName), showLogi);
	}

	// 动态PID参数只在algorithm == 3时显示
	obs_property_set_visible(obs_properties_get(props, "dynamic_pid_group"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kp"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_ki"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kd"), page == 3 && algorithm == 3);
	obs_property_set_visible(obs_properties_get(props, "dynamic_target_threshold"), page == 3 && algorithm == 3);

	// 专业PID参数只在algorithm == 1时显示
	obs_property_set_visible(obs_properties_get(props, "external_pid_group"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kp_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kd_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kp_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_ki_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_kd_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_predict_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_predict_y"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_rate_x"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "external_rate_y"), page == 3 && algorithm == 1);

	// aim 控制器参数只在 algorithm == 2 时显示
	// 子属性在子容器中，组可见时自动跟随；仅 aim_noise_amplitude 需根据开关单独控制
	{
		bool aimVis = (page == 3 && algorithm == 2);
		bool aimNoiseVis = aimVis && obs_data_get_bool(settings, "aim_noise_enabled");
		obs_property_set_visible(obs_properties_get(props, "aim_controller_group"), aimVis);
		obs_property_set_visible(obs_properties_get(props, "aim_noise_amplitude"), aimNoiseVis);
	}

	// 测试连接按钮只在基础页面显示
	obs_property_set_visible(obs_properties_get(props, "test_makcu_connection"), page == 2);

	// 页面5: 追踪与高级
	obs_property_set_visible(obs_properties_get(props, "tracking_group"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "iou_threshold"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "max_lost_frames"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "target_switch_delay"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "target_switch_tolerance"), page == 5);
	
	// 卡尔曼追踪设置（页面5）
	bool useKalman = obs_data_get_bool(settings, "use_kalman_tracker");
	obs_property_set_visible(obs_properties_get(props, "use_kalman_tracker"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "kalman_generate_threshold"), page == 5 && useKalman);
	obs_property_set_visible(obs_properties_get(props, "kalman_terminate_count"), page == 5 && useKalman);
	obs_property_set_visible(obs_properties_get(props, "show_kalman_predictions"), page == 5 && useKalman);
	obs_property_set_visible(obs_properties_get(props, "kalman_prediction_frames"), page == 5 && useKalman);
	obs_property_set_visible(obs_properties_get(props, "show_kalman_trajectories"), page == 5 && useKalman);
	
	// 神经网络轨迹生成器设置（页面5）
	bool useNeuralPath = obs_data_get_bool(settings, "enable_neural_path");
	obs_property_set_visible(obs_properties_get(props, "enable_neural_path"), page == 5);
	obs_property_set_visible(obs_properties_get(props, "neural_path_points"), page == 5 && useNeuralPath);
	obs_property_set_visible(obs_properties_get(props, "neural_mouse_step_size"), page == 5 && useNeuralPath);
	obs_property_set_visible(obs_properties_get(props, "neural_target_radius"), page == 5 && useNeuralPath);
	obs_property_set_visible(obs_properties_get(props, "neural_consume_per_frame"), page == 5 && useNeuralPath);
	obs_property_set_visible(obs_properties_get(props, "enable_neural_path_debug"), page == 5 && useNeuralPath);
	
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
	
	// 动态PID参数组（选择1时显示，因为现在只有两种算法：AdvancedPID=0, DynamicPID=1）
	obs_property_set_visible(obs_properties_get(props, "dynamic_pid_group"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kp"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_ki"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_kd"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_target_threshold"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_speed_multiplier"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_min_coefficient"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_max_coefficient"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_transition_sharpness"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_transition_midpoint"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_min_data_points"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_error_tolerance"), page == 3 && algorithm == 1);
	obs_property_set_visible(obs_properties_get(props, "dynamic_smoothing_factor"), page == 3 && algorithm == 1);

	// 页面6: 预测与滤波（整合预测器、贝塞尔）
	obs_property_set_visible(obs_properties_get(props, "predictor_group"), page == 6);
	obs_property_set_visible(obs_properties_get(props, "bezier_movement_group"), page == 6);

	// 页面7: 准星检测
	obs_property_set_visible(obs_properties_get(props, "crosshair_group"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_enabled"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_pick_color"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_color_info"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_manual_r"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_manual_g"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_manual_b"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_apply_rgb"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_h_min"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_h_max"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_s_min"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_s_max"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_v_min"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_v_max"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_h_tolerance"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_s_tolerance"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_v_tolerance"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_morph_kernel"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_erode_iter"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_dilate_iter"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_grid_rows"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_grid_cols"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_quantile_threshold"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_template_path"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_match_threshold"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_min_area"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_max_area"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_shape_filter_enabled"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_shape_type"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_min_fill_ratio"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_max_fill_ratio"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_min_aspect_ratio"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_max_aspect_ratio"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_detect_interval"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_search_radius"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_color_isolation"), page == 7);
	obs_property_set_visible(obs_properties_get(props, "crosshair_debug_mask"), page == 7);

#else
	(void)page;
#endif

	return true;
}

void yolo_detector_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, "model_path", "");
	obs_data_set_default_int(settings, "model_version", static_cast<int>(IYoloModel::Version::YOLOv8));
	obs_data_set_default_string(settings, "use_gpu", USEGPU_CPU);
#ifdef _WIN32
	obs_data_set_default_bool(settings, "use_gpu_texture_inference", false);
#endif
	obs_data_set_default_int(settings, "input_resolution", 640);
	obs_data_set_default_int(settings, "num_threads", 4);
	obs_data_set_default_double(settings, "confidence_threshold", 0.2);
	obs_data_set_default_double(settings, "nms_threshold", 0.45);
	obs_data_set_default_int(settings, "target_class", -1);
	obs_data_set_default_int(settings, "inference_interval_frames", 1);
	obs_data_set_default_bool(settings, "is_inferencing", false);
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
    
    // KalmanFilter 追踪参数
    obs_data_set_default_bool(settings, "use_kalman_tracker", false);
    obs_data_set_default_int(settings, "kalman_generate_threshold", 2);
    obs_data_set_default_int(settings, "kalman_terminate_count", 5);
    obs_data_set_default_bool(settings, "show_kalman_predictions", true);
    obs_data_set_default_int(settings, "kalman_prediction_frames", 5);
    obs_data_set_default_bool(settings, "show_kalman_trajectories", true);
    
    // MotionSimulator 人类行为模拟器参数
    obs_data_set_default_bool(settings, "enable_motion_simulator", false);
    obs_data_set_default_bool(settings, "motion_sim_random_pos", true);
    obs_data_set_default_bool(settings, "motion_sim_overshoot", true);
    obs_data_set_default_bool(settings, "motion_sim_micro_overshoot", true);
    obs_data_set_default_bool(settings, "motion_sim_inertia", true);
    obs_data_set_default_bool(settings, "motion_sim_left_btn_adaptive", true);
    obs_data_set_default_bool(settings, "motion_sim_spray_mode", true);
    obs_data_set_default_bool(settings, "motion_sim_tap_pause", true);
    obs_data_set_default_bool(settings, "motion_sim_retry", true);
    obs_data_set_default_int(settings, "motion_sim_max_retry", 2);
    obs_data_set_default_int(settings, "motion_sim_delay_ms", 80);
    obs_data_set_default_double(settings, "motion_sim_direct_prob", 0.85);
    obs_data_set_default_double(settings, "motion_sim_overshoot_prob", 0.10);
    obs_data_set_default_double(settings, "motion_sim_micro_ovshoot_prob", 0.05);
    
    // 神经网络轨迹生成器默认值
    obs_data_set_default_bool(settings, "enable_neural_path", false);
    obs_data_set_default_int(settings, "neural_path_points", 25);
    obs_data_set_default_double(settings, "neural_mouse_step_size", 8.0);
    obs_data_set_default_int(settings, "neural_target_radius", 8);
    obs_data_set_default_int(settings, "neural_consume_per_frame", 2);
    obs_data_set_default_bool(settings, "enable_neural_path_debug", false);
    
    // 稳定性检测默认值
    obs_data_set_default_bool(settings, "enable_stability_check", false);
    obs_data_set_default_int(settings, "stability_required_frames", 3);
    obs_data_set_default_double(settings, "stability_position_threshold", 5.0);
    obs_data_set_default_double(settings, "stability_size_threshold", 0.1);
    
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

		snprintf(propName, sizeof(propName), "logi_driver_type_%d", i);
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
		snprintf(propName, sizeof(propName), "adaptive_p_gain_rate_%d", i);
		obs_data_set_default_double(settings, propName, 0.03);
		snprintf(propName, sizeof(propName), "d_term_scale_%d", i);
		obs_data_set_default_double(settings, propName, 0.3);

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

		// 积分参数默认值
		snprintf(propName, sizeof(propName), "integral_limit_%d", i);
		obs_data_set_default_double(settings, propName, 100.0);
		snprintf(propName, sizeof(propName), "integral_rate_%d", i);
		obs_data_set_default_double(settings, propName, 1.0);
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
		// Smith预估器默认值
		snprintf(propName, sizeof(propName), "smith_predictor_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "smith_model_gain_%d", i);
		obs_data_set_default_double(settings, propName, 1.0);
		snprintf(propName, sizeof(propName), "smith_model_tau_%d", i);
		obs_data_set_default_double(settings, propName, 0.02);
		snprintf(propName, sizeof(propName), "smith_auto_tau_%d", i);
		obs_data_set_default_bool(settings, propName, true);
		// SlewRate控制器默认值
		snprintf(propName, sizeof(propName), "slew_rate_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "slew_rate_output_gain_%d", i);
		obs_data_set_default_double(settings, propName, 0.25);
		snprintf(propName, sizeof(propName), "slew_rate_response_smoothing_%d", i);
		obs_data_set_default_double(settings, propName, 0.0008);
		snprintf(propName, sizeof(propName), "slew_rate_approach_damping_%d", i);
		obs_data_set_default_double(settings, propName, 5.0);
		snprintf(propName, sizeof(propName), "slew_rate_update_interval_ms_%d", i);
		obs_data_set_default_double(settings, propName, 5.0);
		snprintf(propName, sizeof(propName), "slew_rate_normalization_scale_%d", i);
		obs_data_set_default_double(settings, propName, 5.0);
		// 自适应PID控制器默认值
		snprintf(propName, sizeof(propName), "adaptive_pid_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "adaptive_pid_kp_%d", i);
		obs_data_set_default_double(settings, propName, 1.0);
		snprintf(propName, sizeof(propName), "adaptive_pid_ki_%d", i);
		obs_data_set_default_double(settings, propName, 0.1);
		snprintf(propName, sizeof(propName), "adaptive_pid_kd_%d", i);
		obs_data_set_default_double(settings, propName, 0.05);
		snprintf(propName, sizeof(propName), "adaptive_pid_dead_zone_%d", i);
		obs_data_set_default_double(settings, propName, 0.3);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_limit_%d", i);
		obs_data_set_default_double(settings, propName, 100.0);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_deadzone_%d", i);
		obs_data_set_default_double(settings, propName, 1.0);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_gain_threshold_%d", i);
		obs_data_set_default_double(settings, propName, 50.0);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_gain_rate_%d", i);
		obs_data_set_default_double(settings, propName, 0.015);
		snprintf(propName, sizeof(propName), "adaptive_pid_output_limit_%d", i);
		obs_data_set_default_double(settings, propName, 10.0);
		// 贝塞尔曲线移动参数默认值
		snprintf(propName, sizeof(propName), "bezier_movement_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "bezier_curvature_%d", i);
		obs_data_set_default_double(settings, propName, 0.3);
		snprintf(propName, sizeof(propName), "bezier_randomness_%d", i);
		obs_data_set_default_double(settings, propName, 0.2);
		
		// GhostTracker曲线轨迹默认值
		snprintf(propName, sizeof(propName), "ghost_tracker_group_%d", i);
		obs_data_set_default_bool(settings, propName, false);
		snprintf(propName, sizeof(propName), "ghost_curvature_%d", i);
		obs_data_set_default_double(settings, propName, 0.5);
		snprintf(propName, sizeof(propName), "ghost_noise_intensity_%d", i);
		obs_data_set_default_double(settings, propName, 12.0);
		snprintf(propName, sizeof(propName), "ghost_vertical_snap_%d", i);
		obs_data_set_default_double(settings, propName, 3.0);
		snprintf(propName, sizeof(propName), "ghost_noise_freq_%d", i);
		obs_data_set_default_double(settings, propName, 0.8);
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
    
    // AdaptivePID默认值
    obs_data_set_default_double(settings, "adaptive_base_kp", 0.5);
    obs_data_set_default_double(settings, "adaptive_base_ki", 0.1);
    obs_data_set_default_double(settings, "adaptive_base_kd", 0.05);
    obs_data_set_default_double(settings, "adaptive_integral_threshold", 5.0);
    obs_data_set_default_double(settings, "adaptive_kp_threshold", 5.0);
    obs_data_set_default_double(settings, "adaptive_integral_rate", 0.1);
    obs_data_set_default_double(settings, "adaptive_kp_rate", 0.1);
    obs_data_set_default_double(settings, "adaptive_large_error_rate", 0.1);
    obs_data_set_default_double(settings, "adaptive_max_output", 1000.0);
    obs_data_set_default_double(settings, "adaptive_max_integral", 1000.0);
    obs_data_set_default_bool(settings, "adaptive_use_predictor", true);
    obs_data_set_default_double(settings, "adaptive_pred_weight_x", 0.5);
    obs_data_set_default_double(settings, "adaptive_pred_weight_y", 0.1);
    obs_data_set_default_double(settings, "adaptive_max_pred_time", 0.1);
    obs_data_set_default_double(settings, "adaptive_output_smoothing", 0.7);
    obs_data_set_default_double(settings, "adaptive_derivative_filter", 0.3);
    
    // IncrementalPID默认值
    obs_data_set_default_double(settings, "incremental_kp", 0.5);
    obs_data_set_default_double(settings, "incremental_ki", 0.1);
    obs_data_set_default_double(settings, "incremental_kd", 0.05);
    obs_data_set_default_double(settings, "incremental_speed_x", 1.0);
    obs_data_set_default_double(settings, "incremental_speed_y", 1.0);
    obs_data_set_default_int(settings, "incremental_aim_radius", 200);
    obs_data_set_default_bool(settings, "incremental_jitter", false);
    obs_data_set_default_bool(settings, "incremental_pid_enabled", true);
    obs_data_set_default_bool(settings, "incremental_side_comp", false);
    obs_data_set_default_double(settings, "incremental_side_comp_cap", 5.0);
    
    // 专业PID默认值
    obs_data_set_default_double(settings, "external_kp_x", 1.5);
    obs_data_set_default_double(settings, "external_ki_x", 0.0);
    obs_data_set_default_double(settings, "external_kd_x", 1.5);
    obs_data_set_default_double(settings, "external_kp_y", 1.5);
    obs_data_set_default_double(settings, "external_ki_y", 0.0);
    obs_data_set_default_double(settings, "external_kd_y", 1.5);
    obs_data_set_default_double(settings, "external_predict_x", 1.0);
    obs_data_set_default_double(settings, "external_predict_y", 1.0);
    obs_data_set_default_double(settings, "external_rate_x", 0.3);
    obs_data_set_default_double(settings, "external_rate_y", 0.3);
    obs_data_set_default_int(settings, "external_ki_mode", 1);
    obs_data_set_default_double(settings, "external_kp_limit", 9900.0);
    obs_data_set_default_double(settings, "external_ki_limit", 9900.0);
    obs_data_set_default_double(settings, "external_kd_limit", 9900.0);
    obs_data_set_default_double(settings, "external_output_limit", 0.0);
    obs_data_set_default_double(settings, "external_ki_rate", 0.05);
    obs_data_set_default_double(settings, "external_ki_deadband", 0.5);
    // aim 控制器参数默认值
    obs_data_set_default_double(settings, "aim_kp", 0.6);
    obs_data_set_default_double(settings, "aim_ki", 0.01);
    obs_data_set_default_double(settings, "aim_kd", 0.007);
    obs_data_set_default_bool(settings, "aim_noise_enabled", false);
    obs_data_set_default_double(settings, "aim_noise_amplitude", 2.0);
    obs_data_set_default_double(settings, "aim_prediction_weight_x", 0.3);
    obs_data_set_default_double(settings, "aim_prediction_weight_y", 0.1);
    obs_data_set_default_double(settings, "aim_ramp_time", 0.3);
    obs_data_set_default_double(settings, "aim_init_scale", 0.6);
    obs_data_set_default_double(settings, "aim_output_max", 128.0);
    obs_data_set_default_double(settings, "incremental_side_comp_denom", 1.0);
    obs_data_set_default_double(settings, "incremental_input_alpha", 0.3);
    obs_data_set_default_double(settings, "incremental_d_alpha", 0.2);
    obs_data_set_default_double(settings, "incremental_output_alpha", 0.4);

    // 准星检测参数默认值
    obs_data_set_default_bool(settings, "crosshair_enabled", false);
    obs_data_set_default_int(settings, "crosshair_h_min", 0);
    obs_data_set_default_int(settings, "crosshair_h_max", 180);
    obs_data_set_default_int(settings, "crosshair_s_min", 100);
    obs_data_set_default_int(settings, "crosshair_s_max", 255);
    obs_data_set_default_int(settings, "crosshair_v_min", 100);
    obs_data_set_default_int(settings, "crosshair_v_max", 255);
    obs_data_set_default_int(settings, "crosshair_manual_r", 0);
    obs_data_set_default_int(settings, "crosshair_manual_g", 255);
    obs_data_set_default_int(settings, "crosshair_manual_b", 0);
    obs_data_set_default_int(settings, "crosshair_h_tolerance", 10);
    obs_data_set_default_int(settings, "crosshair_s_tolerance", 40);
    obs_data_set_default_int(settings, "crosshair_v_tolerance", 40);
    obs_data_set_default_int(settings, "crosshair_morph_kernel", 3);
    obs_data_set_default_int(settings, "crosshair_erode_iter", 0);
    obs_data_set_default_int(settings, "crosshair_dilate_iter", 1);
    obs_data_set_default_int(settings, "crosshair_grid_rows", 4);
    obs_data_set_default_int(settings, "crosshair_grid_cols", 4);
    obs_data_set_default_double(settings, "crosshair_quantile_threshold", 0.01);
    obs_data_set_default_string(settings, "crosshair_template_path", "");
    obs_data_set_default_double(settings, "crosshair_match_threshold", 0.6);
    obs_data_set_default_int(settings, "crosshair_min_area", 10);
    obs_data_set_default_int(settings, "crosshair_max_area", 50000);
    obs_data_set_default_bool(settings, "crosshair_shape_filter_enabled", false);
    obs_data_set_default_int(settings, "crosshair_shape_type", 0);
    obs_data_set_default_double(settings, "crosshair_min_fill_ratio", 0.05);
    obs_data_set_default_double(settings, "crosshair_max_fill_ratio", 0.8);
    obs_data_set_default_double(settings, "crosshair_min_aspect_ratio", 0.3);
    obs_data_set_default_double(settings, "crosshair_max_aspect_ratio", 3.0);
    obs_data_set_default_int(settings, "crosshair_detect_interval", 1);
    obs_data_set_default_int(settings, "crosshair_search_radius", 0);
    obs_data_set_default_bool(settings, "crosshair_color_isolation", false);
    obs_data_set_default_bool(settings, "crosshair_debug_mask", false);
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
	IYoloModel::Version newModelVersion = static_cast<IYoloModel::Version>(obs_data_get_int(settings, "model_version"));
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
				obs_log(LOG_INFO, "[YOLO Filter] Loading new model: %s (backend: %s)", tf->modelPath.c_str(), tf->useGPU.c_str());

				std::shared_ptr<IYoloModel> newYoloModel;
				newYoloModel = std::make_shared<ModelYOLO>(tf->modelVersion);

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
	// 运行时检测DML直推能力：模型加载后检查preprocessor是否就绪
	if (tf->useGPU == "dml" && tf->yoloModel && tf->yoloModel->isDmlTextureSupported()) {
		obs_log(LOG_INFO, "[YOLO Filter] DML GPU直推可用：预处理在渲染线程完成，推理线程直接消费float buffer");
		if (!tf->useGpuTextureInference) {
			obs_log(LOG_INFO, "[YOLO Filter] 提示：可启用\"GPU纹理推理\"以使用DML直推路径，避免GPU-CPU数据搬运");
		}
	}

	// GPU纹理推理支持CUDA、TensorRT和DML设备
	if (tf->useGpuTextureInference && tf->useGPU != "dml") {
		obs_log(LOG_WARNING, "[YOLO Filter] GPU纹理推理需要DML设备，已禁用");
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
	
	// 推理状态 - 只有当设置中明确存在时才更新，避免参数调整时重置
	if (obs_data_has_user_value(settings, "is_inferencing")) {
		tf->isInferencing = obs_data_get_bool(settings, "is_inferencing");
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

	// KalmanFilter 追踪参数
	bool newUseKalmanTracker = obs_data_get_bool(settings, "use_kalman_tracker");
	int newKalmanGenerateThreshold = (int)obs_data_get_int(settings, "kalman_generate_threshold");
	int newKalmanTerminateCount = (int)obs_data_get_int(settings, "kalman_terminate_count");
	tf->showKalmanPredictions = obs_data_get_bool(settings, "show_kalman_predictions");
	tf->kalmanPredictionFrames = (int)obs_data_get_int(settings, "kalman_prediction_frames");
	tf->showKalmanTrajectories = obs_data_get_bool(settings, "show_kalman_trajectories");
	
	if (tf->useKalmanTracker != newUseKalmanTracker || 
	    tf->kalmanGenerateThreshold != newKalmanGenerateThreshold ||
	    tf->kalmanTerminateCount != newKalmanTerminateCount) {
		tf->useKalmanTracker = newUseKalmanTracker;
		tf->kalmanGenerateThreshold = newKalmanGenerateThreshold;
		tf->kalmanTerminateCount = newKalmanTerminateCount;
		if (tf->useKalmanTracker) {
			tf->kalmanTracker.init(tf->kalmanGenerateThreshold, tf->kalmanTerminateCount);
		} else {
			tf->kalmanTracker.reset();
		}
	}

	// 神经网络轨迹生成器参数（全局设置，应用到所有配置）
	for (int i = 0; i < yolo_detector_filter::MAX_CONFIGS; i++) {
		tf->mouseConfigs[i].enableNeuralPath = obs_data_get_bool(settings, "enable_neural_path");
		tf->mouseConfigs[i].neuralPathPoints = (int)obs_data_get_int(settings, "neural_path_points");
		tf->mouseConfigs[i].neuralMouseStepSize = obs_data_get_double(settings, "neural_mouse_step_size");
		tf->mouseConfigs[i].neuralTargetRadius = (int)obs_data_get_int(settings, "neural_target_radius");
		tf->mouseConfigs[i].neuralConsumePerFrame = (int)obs_data_get_int(settings, "neural_consume_per_frame");
		tf->mouseConfigs[i].enableNeuralPathDebug = obs_data_get_bool(settings, "enable_neural_path_debug");
	}
	
	// 日志：神经网络配置读取结果
	static int configReadCount = 0;
	configReadCount++;
	if ((configReadCount <= 3 || tf->mouseConfigs[0].enableNeuralPath) && tf->mouseConfigs[0].enableNeuralPathDebug) {
		obs_log(LOG_INFO, "[NeuralPath-Config] READ: enableNeuralPath=%d, neuralPathPoints=%d, neuralMouseStepSize=%.1f, neuralTargetRadius=%d, debug=%d",
				tf->mouseConfigs[0].enableNeuralPath ? 1 : 0,
				tf->mouseConfigs[0].neuralPathPoints,
				tf->mouseConfigs[0].neuralMouseStepSize,
				tf->mouseConfigs[0].neuralTargetRadius,
				tf->mouseConfigs[0].enableNeuralPathDebug ? 1 : 0);
	}

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

		snprintf(propName, sizeof(propName), "logi_driver_type_%d", i);
		tf->mouseConfigs[i].logiDriverType = (int)obs_data_get_int(settings, propName);

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
		snprintf(propName, sizeof(propName), "adaptive_p_gain_rate_%d", i);
		tf->mouseConfigs[i].adaptivePGainRate = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "d_term_scale_%d", i);
		tf->mouseConfigs[i].dTermScale = (float)obs_data_get_double(settings, propName);

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

		// 积分参数更新
		snprintf(propName, sizeof(propName), "integral_limit_%d", i);
		tf->mouseConfigs[i].integralLimit = (float)obs_data_get_double(settings, propName);
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
		// Smith预估器参数
		snprintf(propName, sizeof(propName), "smith_predictor_group_%d", i);
		tf->mouseConfigs[i].smithPredictorEnabled = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "smith_model_gain_%d", i);
		tf->mouseConfigs[i].smithModelGain = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "smith_model_tau_%d", i);
		tf->mouseConfigs[i].smithModelTau = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "smith_auto_tau_%d", i);
		tf->mouseConfigs[i].smithAutoTau = obs_data_get_bool(settings, propName);
		// SlewRate控制器参数
		snprintf(propName, sizeof(propName), "slew_rate_group_%d", i);
		tf->mouseConfigs[i].slewRateEnabled = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "slew_rate_output_gain_%d", i);
		tf->mouseConfigs[i].slewRateOutputGain = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "slew_rate_response_smoothing_%d", i);
		tf->mouseConfigs[i].slewRateResponseSmoothing = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "slew_rate_approach_damping_%d", i);
		tf->mouseConfigs[i].slewRateApproachDamping = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "slew_rate_update_interval_ms_%d", i);
		tf->mouseConfigs[i].slewRateUpdateIntervalMs = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "slew_rate_normalization_scale_%d", i);
		tf->mouseConfigs[i].slewRateNormalizationScale = (float)obs_data_get_double(settings, propName);
		// 自适应PID控制器参数
		snprintf(propName, sizeof(propName), "adaptive_pid_kp_%d", i);
		tf->mouseConfigs[i].adaptivePidKp = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_ki_%d", i);
		tf->mouseConfigs[i].adaptivePidKi = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_kd_%d", i);
		tf->mouseConfigs[i].adaptivePidKd = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_dead_zone_%d", i);
		tf->mouseConfigs[i].adaptivePidDeadZone = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_limit_%d", i);
		tf->mouseConfigs[i].adaptivePidIntegralLimit = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_deadzone_%d", i);
		tf->mouseConfigs[i].adaptivePidIntegralDeadzone = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_gain_threshold_%d", i);
		tf->mouseConfigs[i].adaptivePidIntegralGainThreshold = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_integral_gain_rate_%d", i);
		tf->mouseConfigs[i].adaptivePidIntegralGainRate = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "adaptive_pid_output_limit_%d", i);
		tf->mouseConfigs[i].adaptivePidOutputLimit = (float)obs_data_get_double(settings, propName);
		// 贝塞尔曲线移动参数
		snprintf(propName, sizeof(propName), "bezier_movement_group_%d", i);
		tf->mouseConfigs[i].enableBezierMovement = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "bezier_curvature_%d", i);
		tf->mouseConfigs[i].bezierCurvature = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "bezier_randomness_%d", i);
		tf->mouseConfigs[i].bezierRandomness = (float)obs_data_get_double(settings, propName);
		
		// GhostTracker曲线轨迹参数
		snprintf(propName, sizeof(propName), "ghost_tracker_group_%d", i);
		tf->mouseConfigs[i].enableGhostTracker = obs_data_get_bool(settings, propName);
		snprintf(propName, sizeof(propName), "ghost_curvature_%d", i);
		tf->mouseConfigs[i].ghostCurvature = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "ghost_noise_intensity_%d", i);
		tf->mouseConfigs[i].ghostNoiseIntensity = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "ghost_vertical_snap_%d", i);
		tf->mouseConfigs[i].ghostVerticalSnapRatio = (float)obs_data_get_double(settings, propName);
		snprintf(propName, sizeof(propName), "ghost_noise_freq_%d", i);
		tf->mouseConfigs[i].ghostNoiseFreq = (float)obs_data_get_double(settings, propName);
	}

	tf->targetSwitchDelayMs = (int)obs_data_get_int(settings, "target_switch_delay");
	tf->targetSwitchTolerance = (float)obs_data_get_double(settings, "target_switch_tolerance");

	// 读取全局算法类型参数
	tf->algorithmTypeGlobal = (int)obs_data_get_int(settings, "algorithm_type_global");

	// 专业PID参数
	tf->externalKpX = (float)obs_data_get_double(settings, "external_kp_x");
	tf->externalKiX = (float)obs_data_get_double(settings, "external_ki_x");
	tf->externalKdX = (float)obs_data_get_double(settings, "external_kd_x");
	tf->externalKpY = (float)obs_data_get_double(settings, "external_kp_y");
	tf->externalKiY = (float)obs_data_get_double(settings, "external_ki_y");
	tf->externalKdY = (float)obs_data_get_double(settings, "external_kd_y");
	tf->externalPredictX = (float)obs_data_get_double(settings, "external_predict_x");
	tf->externalPredictY = (float)obs_data_get_double(settings, "external_predict_y");
	tf->externalRateX = (float)obs_data_get_double(settings, "external_rate_x");
	tf->externalRateY = (float)obs_data_get_double(settings, "external_rate_y");
	tf->externalKiMode = (float)obs_data_get_int(settings, "external_ki_mode");
	tf->externalKpLimit = (float)obs_data_get_double(settings, "external_kp_limit");
	tf->externalKiLimit = (float)obs_data_get_double(settings, "external_ki_limit");
	tf->externalKdLimit = (float)obs_data_get_double(settings, "external_kd_limit");
	tf->externalOutputLimit = (float)obs_data_get_double(settings, "external_output_limit");
	tf->externalKiRate = (float)obs_data_get_double(settings, "external_ki_rate");
	tf->externalKiDeadband = (float)obs_data_get_double(settings, "external_ki_deadband");

	// aim 控制器参数读取
	tf->aimKp = (float)obs_data_get_double(settings, "aim_kp");
	tf->aimKi = (float)obs_data_get_double(settings, "aim_ki");
	tf->aimKd = (float)obs_data_get_double(settings, "aim_kd");
	tf->aimNoiseEnabled = obs_data_get_bool(settings, "aim_noise_enabled");
	tf->aimNoiseAmplitude = (float)obs_data_get_double(settings, "aim_noise_amplitude");
	tf->aimPredictionWeightX = (float)obs_data_get_double(settings, "aim_prediction_weight_x");
	tf->aimPredictionWeightY = (float)obs_data_get_double(settings, "aim_prediction_weight_y");
	tf->aimRampTime = (float)obs_data_get_double(settings, "aim_ramp_time");
	tf->aimInitScale = (float)obs_data_get_double(settings, "aim_init_scale");
	tf->aimOutputMax = (float)obs_data_get_double(settings, "aim_output_max");

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

	// 准星检测参数读取
	{
		CrosshairDetectorConfig& chCfg = tf->crosshairConfig;
		bool wasPickingColor = chCfg.pickingColor;  // 保留运行时吸管取色状态
		chCfg.enabled = obs_data_get_bool(settings, "crosshair_enabled");
		chCfg.hMin = (int)obs_data_get_int(settings, "crosshair_h_min");
		chCfg.hMax = (int)obs_data_get_int(settings, "crosshair_h_max");
		chCfg.sMin = (int)obs_data_get_int(settings, "crosshair_s_min");
		chCfg.sMax = (int)obs_data_get_int(settings, "crosshair_s_max");
		chCfg.vMin = (int)obs_data_get_int(settings, "crosshair_v_min");
		chCfg.vMax = (int)obs_data_get_int(settings, "crosshair_v_max");
		chCfg.hTolerance = (int)obs_data_get_int(settings, "crosshair_h_tolerance");
		chCfg.sTolerance = (int)obs_data_get_int(settings, "crosshair_s_tolerance");
		chCfg.vTolerance = (int)obs_data_get_int(settings, "crosshair_v_tolerance");
		chCfg.morphKernelSize = (int)obs_data_get_int(settings, "crosshair_morph_kernel");
		chCfg.erodeIterations = (int)obs_data_get_int(settings, "crosshair_erode_iter");
		chCfg.dilateIterations = (int)obs_data_get_int(settings, "crosshair_dilate_iter");
		chCfg.gridRows = (int)obs_data_get_int(settings, "crosshair_grid_rows");
		chCfg.gridCols = (int)obs_data_get_int(settings, "crosshair_grid_cols");
		chCfg.quantileThreshold = (float)obs_data_get_double(settings, "crosshair_quantile_threshold");
		std::string newTemplatePath = obs_data_get_string(settings, "crosshair_template_path");
		chCfg.templateImagePath = newTemplatePath;
		chCfg.matchThreshold = (float)obs_data_get_double(settings, "crosshair_match_threshold");
		chCfg.minArea = (int)obs_data_get_int(settings, "crosshair_min_area");
		chCfg.maxArea = (int)obs_data_get_int(settings, "crosshair_max_area");
		chCfg.shapeFilterEnabled = obs_data_get_bool(settings, "crosshair_shape_filter_enabled");
		chCfg.shapeType = (int)obs_data_get_int(settings, "crosshair_shape_type");
		chCfg.minFillRatio = (float)obs_data_get_double(settings, "crosshair_min_fill_ratio");
		chCfg.maxFillRatio = (float)obs_data_get_double(settings, "crosshair_max_fill_ratio");
		chCfg.minAspectRatio = (float)obs_data_get_double(settings, "crosshair_min_aspect_ratio");
		chCfg.maxAspectRatio = (float)obs_data_get_double(settings, "crosshair_max_aspect_ratio");
		chCfg.detectEveryNFrames = (int)obs_data_get_int(settings, "crosshair_detect_interval");
		chCfg.searchRadius = (int)obs_data_get_int(settings, "crosshair_search_radius");
		chCfg.colorIsolationView = obs_data_get_bool(settings, "crosshair_color_isolation");
		chCfg.showDebugMask = obs_data_get_bool(settings, "crosshair_debug_mask");

		// 如果模板路径改变，加载新模板
		tf->crosshairDetector.updateConfig(chCfg);
		if (!newTemplatePath.empty()) {
			tf->crosshairDetector.loadTemplate(newTemplatePath);
		}

		// 吸管取色处理
		if (chCfg.pickingColor || wasPickingColor) {
			chCfg.pickingColor = false;
			// 重新更新config，确保pickingColor为false
			tf->crosshairDetector.updateConfig(chCfg);
			// 取色将在video_tick中执行（需要当前帧数据）
			tf->crosshairNeedsPick = true;
		}
	}

#endif

	tf->isDisabled = false;
}

bool toggleInference(obs_properties_t *props, obs_property_t *property, void *data)
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

bool refreshStats(obs_properties_t *props, obs_property_t *property, void *data)
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

	// 更新DML直推统计
	obs_property_t *dmlStatsText = obs_properties_get(props, "dml_stats");
	if (dmlStatsText) {
		char dmlStr[128];
		snprintf(dmlStr, sizeof(dmlStr), "DML直推: %d 成功 / %d 回退CPU",
			tf->dmlDirectFrames.load(), tf->dmlFallbackFrames.load());
		obs_property_set_description(dmlStatsText, dmlStr);
	}

	return true;
}

bool testMAKCUConnection(obs_properties_t *props, obs_property_t *property, void *data)
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

bool saveConfigCallback(obs_properties_t *props, obs_property_t *property, void *data)
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

        snprintf(propName, sizeof(propName), "logi_driver_type_%d", i);
        fprintf(f, "      \"logiDriverType\": %d,\n", (int)obs_data_get_int(settings, propName));

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

bool loadConfigCallback(obs_properties_t *props, obs_property_t *property, void *data)
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

        snprintf(propName, sizeof(propName), "logi_driver_type_%d", i);
        if (findValueInConfig(i, "logiDriverType", val)) {
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


#endif // _WIN32
