#include "yolo-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <plugin-support.h>
#include "models/ModelYOLO.h"
#include "models/Detection.h"
#include "FilterData.h"
#include "ort-utils/ort-session-utils.h"
#include "obs-utils/obs-utils.h"
#include "consts.h"
#include "update-checker/update-checker.h"

// 定义新的滤镜结构体，继承自filter_data
struct yolo_filter_data : public filter_data, public std::enable_shared_from_this<yolo_filter_data> {
    // YOLO检测相关参数
    bool enableDetection = true;
    float confidenceThreshold = 0.5f;
    float nmsThreshold = 0.45f;
    int targetClassId = -1;  // -1表示检测所有类别
    std::string modelPath;   // 自定义模型路径
    std::string classNamesPath; // 类别名称文件路径
    ModelYOLO::Version yoloVersion = ModelYOLO::Version::YOLOv5;

    // 检测结果存储
    std::vector<Detection> detections;
    std::mutex detectionsMutex;

    // 绘制参数
    bool drawBoundingBoxes = true;
    bool drawLabels = true;
    float boundingBoxThickness = 2.0f;
    cv::Scalar boundingBoxColor{0, 255, 0, 255}; // BGR格式
    float fontSize = 0.5f;

    // 其他参数
    int detectionEveryXFrames = 1;
    int detectionFrameCounter = 0;

    ~yolo_filter_data() { obs_log(LOG_INFO, "YOLO filter destructor called"); }
};

void yolo_detection_thread(void *data); // Forward declaration

const char *yolo_filter_getname(void *unused)
{
    UNUSED_PARAMETER(unused);
    return obs_module_text("YoloDaWang");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings, const char *bool_prop, const char *prop_name)
{
    const bool enabled = obs_data_get_bool(settings, bool_prop);
    obs_property_t *p = obs_properties_get(ppts, prop_name);
    obs_property_set_visible(p, enabled);
    return true;
}

static bool draw_bounding_boxes_modified(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
    UNUSED_PARAMETER(p);
    return visible_on_bool(ppts, settings, "draw_bounding_boxes", "bounding_box_group");
}

static bool enable_detection_modified(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
    const bool enabled = obs_data_get_bool(settings, "enable_detection");
    obs_property_t *bbox_prop = obs_properties_get(ppts, "draw_bounding_boxes");
    obs_property_t *label_prop = obs_properties_get(ppts, "draw_labels");
    obs_property_set_visible(bbox_prop, enabled);
    obs_property_set_visible(label_prop, enabled);
    return true;
}

obs_properties_t *yolo_filter_properties(void *data)
{
    obs_properties_t *props = obs_properties_create();

    obs_property_t *p_enable_detection =
        obs_properties_add_bool(props, "enable_detection", obs_module_text("EnableYOLODetection"));
    obs_property_set_modified_callback(p_enable_detection, enable_detection_modified);

    // Detection settings
    obs_properties_t *detection_props = obs_properties_create();

    obs_properties_add_float_slider(detection_props, "confidence_threshold", 
                                   obs_module_text("ConfidenceThreshold"), 0.0, 1.0, 0.01);

    obs_properties_add_float_slider(detection_props, "nms_threshold", 
                                   obs_module_text("NMSThreshold"), 0.0, 1.0, 0.01);

    obs_properties_add_int_slider(detection_props, "target_class_id", 
                                 obs_module_text("TargetClassID"), -1, 1000, 1);

    obs_properties_add_path(detection_props, "model_path", 
                           obs_module_text("CustomYOLOModelPath"), OBS_PATH_FILE, 
                           "*.onnx", NULL);

    obs_properties_add_path(detection_props, "class_names_path", 
                           obs_module_text("ClassNamesFilePath"), OBS_PATH_FILE, 
                           "*.txt", NULL);

    obs_property_t *p_yolo_version = obs_properties_add_list(detection_props, "yolo_version",
                                    obs_module_text("YOLOVersion"),
                                    OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

    obs_property_list_add_string(p_yolo_version, obs_module_text("YOLOv5"), "yolov5");
    obs_property_list_add_string(p_yolo_version, obs_module_text("YOLOv8"), "yolov8");
    obs_property_list_add_string(p_yolo_version, obs_module_text("YOLOv11"), "yolov11");

    obs_properties_add_int(detection_props, "detection_every_x_frames", 
                          obs_module_text("DetectEveryXFrames"), 1, 300, 1);

    obs_properties_add_group(props, "detection_group", obs_module_text("Detection Settings"), 
                            OBS_GROUP_NORMAL, detection_props);

    // Bounding box settings
    obs_property_t *p_draw_bounding_boxes =
        obs_properties_add_bool(props, "draw_bounding_boxes", obs_module_text("DrawBoundingBoxes"));
    obs_property_set_modified_callback(p_draw_bounding_boxes, draw_bounding_boxes_modified);

    obs_properties_t *bbox_props = obs_properties_create();

    obs_properties_add_float_slider(bbox_props, "bounding_box_thickness", 
                                   obs_module_text("Bounding_Box_Thickness"), 1.0, 10.0, 0.5);

    obs_properties_add_color(bbox_props, "bounding_box_color", obs_module_text("Bounding_Box_Color"));

    obs_properties_add_bool(bbox_props, "draw_labels", obs_module_text("DrawLabels"));

    obs_properties_add_float_slider(bbox_props, "font_size", 
                                   obs_module_text("Font_Size"), 0.1, 2.0, 0.1);

    obs_properties_add_group(props, "bounding_box_group", obs_module_text("BoundingBoxSettings"), 
                            OBS_GROUP_NORMAL, bbox_props);

    // GPU, CPU and performance Props
    obs_property_t *p_use_gpu = obs_properties_add_list(props, "useGPU", obs_module_text("Inference Device"),
                                OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

    obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#ifdef HAVE_ONNXRUNTIME_CUDA_EP
    obs_property_list_add_string(p_use_gpu, obs_module_text("GPUCUDA"), USEGPU_CUDA);
#endif
#ifdef HAVE_ONNXRUNTIME_ROCM_EP
    obs_property_list_add_string(p_use_gpu, obs_module_text("GPUROCM"), USEGPU_ROCM);
#endif
#ifdef HAVE_ONNXRUNTIME_MIGRAPHX_EP
    obs_property_list_add_string(p_use_gpu, obs_module_text("GPUMIGRAPHX"), USEGPU_MIGRAPHX);
#endif
#ifdef HAVE_ONNXRUNTIME_TENSORRT_EP
    obs_property_list_add_string(p_use_gpu, obs_module_text("TENSORRT"), USEGPU_TENSORRT);
#endif
#if defined(__APPLE__)
    obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"), USEGPU_COREML);
#endif

    obs_properties_add_int_slider(props, "numThreads", obs_module_text("Num Threads"), 0, 8, 1);

    // Add a informative text about the plugin
    // replace the placeholder with the current version
    // use std::regex_replace instead of QString::arg because the latter doesn't work on Linux
    std::string basic_info = std::regex_replace(PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
    // Check for update
    if (get_latest_version() != nullptr) {
        basic_info += std::regex_replace(PLUGIN_INFO_TEMPLATE_UPDATE_AVAILABLE, std::regex("%1"),
                         get_latest_version());
    }
    obs_properties_add_text(props, "info", basic_info.c_str(), OBS_TEXT_INFO);

    UNUSED_PARAMETER(data);
    return props;
}

void yolo_filter_defaults(obs_data_t *settings)
{
    obs_data_set_default_bool(settings, "enable_detection", true);
    obs_data_set_default_double(settings, "confidence_threshold", 0.5);
    obs_data_set_default_double(settings, "nms_threshold", 0.45);
    obs_data_set_default_int(settings, "target_class_id", -1);
    obs_data_set_default_string(settings, "model_path", "");
    obs_data_set_default_string(settings, "class_names_path", "");
    obs_data_set_default_string(settings, "yolo_version", "yolov5");
    obs_data_set_default_int(settings, "detection_every_x_frames", 1);
    obs_data_set_default_bool(settings, "draw_bounding_boxes", true);
    obs_data_set_default_double(settings, "bounding_box_thickness", 2.0);
    obs_data_set_default_int(settings, "bounding_box_color", 0x00FF00); // Green
    obs_data_set_default_bool(settings, "draw_labels", true);
    obs_data_set_default_double(settings, "font_size", 0.5);
    
#if defined(__APPLE__)
    obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
    obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
    obs_data_set_default_int(settings, "numThreads", 1);
}

void yolo_filter_update(void *data, obs_data_t *settings)
{
    obs_log(LOG_INFO, "YOLO filter updated");

    // Cast to shared_ptr pointer and create a local shared_ptr
    auto *ptr = static_cast<std::shared_ptr<yolo_filter_data> *>(data);
    if (!ptr) {
        return;
    }

    std::shared_ptr<yolo_filter_data> tf = *ptr;
    if (!tf) {
        return;
    }

    tf->isDisabled = true;

    // Update YOLO-specific settings
    tf->enableDetection = obs_data_get_bool(settings, "enable_detection");
    tf->confidenceThreshold = (float)obs_data_get_double(settings, "confidence_threshold");
    tf->nmsThreshold = (float)obs_data_get_double(settings, "nms_threshold");
    tf->targetClassId = (int)obs_data_get_int(settings, "target_class_id");
    
    const char *modelPath = obs_data_get_string(settings, "model_path");
    if (modelPath && strlen(modelPath) > 0) {
        tf->modelPath = std::string(modelPath);
    }
    
    const char *classNamesPath = obs_data_get_string(settings, "class_names_path");
    if (classNamesPath && strlen(classNamesPath) > 0) {
        tf->classNamesPath = std::string(classNamesPath);
    }
    
    const std::string yoloVersionStr = obs_data_get_string(settings, "yolo_version");
    if (yoloVersionStr == "yolov8") {
        tf->yoloVersion = ModelYOLO::Version::YOLOv8;
    } else if (yoloVersionStr == "yolov11") {
        tf->yoloVersion = ModelYOLO::Version::YOLOv11;
    } else {
        tf->yoloVersion = ModelYOLO::Version::YOLOv5;
    }
    
    tf->detectionEveryXFrames = (int)obs_data_get_int(settings, "detection_every_x_frames");
    tf->detectionFrameCounter = 0;
    
    tf->drawBoundingBoxes = obs_data_get_bool(settings, "draw_bounding_boxes");
    tf->drawLabels = obs_data_get_bool(settings, "draw_labels");
    tf->boundingBoxThickness = (float)obs_data_get_double(settings, "bounding_box_thickness");
    
    int color = (int)obs_data_get_int(settings, "bounding_box_color");
    tf->boundingBoxColor = cv::Scalar(
        color & 0xFF,           // Blue
        (color >> 8) & 0xFF,    // Green  
        (color >> 16) & 0xFF,   // Red
        255                     // Alpha
    );
    
    tf->fontSize = (float)obs_data_get_double(settings, "font_size");

    // Handle model and inference settings
    const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
    const uint32_t newNumThreads = (uint32_t)obs_data_get_int(settings, "numThreads");

    if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads || tf->modelPath != "") {
        // lock modelMutex
        std::unique_lock<std::mutex> lock(tf->modelMutex);

        // Re-initialize model if switching inference device or model
        tf->useGPU = newUseGpu;
        tf->numThreads = newNumThreads;

        // Create YOLO model
        tf->model.reset(new ModelYOLO(tf->yoloVersion));

        // Set model parameters
        static_cast<ModelYOLO*>(tf->model.get())->setConfidenceThreshold(tf->confidenceThreshold);
        static_cast<ModelYOLO*>(tf->model.get())->setNMSThreshold(tf->nmsThreshold);
        static_cast<ModelYOLO*>(tf->model.get())->setTargetClass(tf->targetClassId);
        
        if (!tf->classNamesPath.empty()) {
            static_cast<ModelYOLO*>(tf->model.get())->loadClassNames(tf->classNamesPath);
        }

        // Load model from custom path if specified
        if (!tf->modelPath.empty()) {
            // For now, we'll use the model selection mechanism from the base class
            tf->modelSelection = tf->modelPath;
        } else {
            // Default model - could be a built-in YOLO model
            tf->modelSelection = "models/yolo_model.onnx"; // Placeholder
        }

        int ortSessionResult = createOrtSession(tf.get());
        if (ortSessionResult != OBS_BGREMOVAL_ORT_SESSION_SUCCESS) {
            obs_log(LOG_ERROR, "Failed to create ONNXRuntime session. Error code: %d", ortSessionResult);
            // disable filter
            tf->isDisabled = true;
            tf->model.reset();
            return;
        }
    }

    obs_enter_graphics();

    // Load effect files if needed
    // For now, we can reuse the existing effect or create a new one
    char *effect_path = obs_module_file(EFFECT_PATH);
    gs_effect_destroy(tf->effect);
    tf->effect = gs_effect_create_from_file(effect_path, NULL);
    bfree(effect_path);

    obs_leave_graphics();

    // Log the currently selected options
    obs_log(LOG_INFO, "YOLO Detection Filter Options:");
    obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
    obs_log(LOG_INFO, "  YOLO Version: %d", static_cast<int>(tf->yoloVersion));
    obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
    obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
    obs_log(LOG_INFO, "  Confidence Threshold: %f", tf->confidenceThreshold);
    obs_log(LOG_INFO, "  NMS Threshold: %f", tf->nmsThreshold);
    obs_log(LOG_INFO, "  Target Class ID: %d", tf->targetClassId);
    obs_log(LOG_INFO, "  Detect Every X Frames: %d", tf->detectionEveryXFrames);
    obs_log(LOG_INFO, "  Draw Bounding Boxes: %s", tf->drawBoundingBoxes ? "true" : "false");
    obs_log(LOG_INFO, "  Draw Labels: %s", tf->drawLabels ? "true" : "false");
    obs_log(LOG_INFO, "  Bounding Box Thickness: %f", tf->boundingBoxThickness);
    obs_log(LOG_INFO, "  Font Size: %f", tf->fontSize);
    obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");

    // enable
    tf->isDisabled = false;
}

void *yolo_filter_create(obs_data_t *settings, obs_source_t *source)
{
    auto tf = std::shared_ptr<yolo_filter_data>(new yolo_filter_data);
    tf->source = source;

    tf->env = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOFilter"));

    tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    tf->stagesurface = gs_stagesurface_create(640, 480, GS_BGRA);

    // Start the detection thread
    std::thread t(yolo_detection_thread, tf.get());
    t.detach();

    // Initialize as a shared_ptr in a void* container
    auto *ptr = new std::shared_ptr<yolo_filter_data>(tf);
    return ptr;
}

void yolo_filter_destroy(void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_filter_data> *>(data);
    if (!ptr) {
        return;
    }

    std::shared_ptr<yolo_filter_data> tf = *ptr;
    if (tf) {
        tf->isDisabled = true;
        
        // Clean up resources
        obs_enter_graphics();
        gs_texrender_destroy(tf->texrender);
        gs_stagesurface_destroy(tf->stagesurface);
        gs_effect_destroy(tf->effect);
        obs_leave_graphics();
    }

    delete ptr;
}

void yolo_filter_activate(void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_filter_data> *>(data);
    if (!ptr) {
        return;
    }

    std::shared_ptr<yolo_filter_data> tf = *ptr;
    if (!tf) {
        return;
    }

    obs_log(LOG_DEBUG, "YOLO filter activated");
}

void yolo_filter_deactivate(void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_filter_data> *>(data);
    if (!ptr) {
        return;
    }

    std::shared_ptr<yolo_filter_data> tf = *ptr;
    if (!tf) {
        return;
    }

    obs_log(LOG_DEBUG, "YOLO filter deactivated");
}

void yolo_filter_video_tick(void *data, float seconds)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_filter_data> *>(data);
    if (!ptr) {
        return;
    }

    std::shared_ptr<yolo_filter_data> tf = *ptr;
    if (!tf) {
        return;
    }

    if (tf->isDisabled) {
        return;
    }

    // Increment frame counter for detection frequency control
    tf->detectionFrameCounter++;
}

void yolo_filter_video_render(void *data, gs_effect_t *_effect)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_filter_data> *>(data);
    if (!ptr) {
        return;
    }

    std::shared_ptr<yolo_filter_data> tf = *ptr;
    if (!tf) {
        return;
    }

    if (tf->isDisabled) {
        obs_source_skip_video_filter(tf->source);
        return;
    }

    if (!tf->enableDetection) {
        obs_source_skip_video_filter(tf->source);
        return;
    }

    // Get the input frame
    uint32_t width = obs_source_get_base_width(tf->source);
    uint32_t height = obs_source_get_base_height(tf->source);

    if (width == 0 || height == 0) {
        obs_source_skip_video_filter(tf->source);
        return;
    }

    // Process the frame if it's time for detection
    if (tf->detectionFrameCounter >= tf->detectionEveryXFrames) {
        tf->detectionFrameCounter = 0;

        // Render the input to texture
        gs_texrender_reset(tf->texrender);

        if (gs_texrender_begin(tf->texrender, width, height)) {
            gs_blend_state_push();
            gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

            // Clear the texture
            struct vec4 background;
            vec4_zero(&background);
            gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);

            // Render the input
            obs_source_video_render(tf->source);

            gs_blend_state_pop();
            gs_texrender_end(tf->texrender);
        }

        // Get the rendered texture
        gs_texture_t *texture = gs_texrender_get_texture(tf->texrender);
        if (texture) {
            // Copy texture to CPU memory for processing
            if (gs_stagesurface_resize(tf->stagesurface, width, height)) {
                gs_stage_texture(tf->stagesurface, texture);

                // Map the staged texture
                uint8_t *data_ptr;
                uint32_t linesize;
                if (gs_stagesurface_map(tf->stagesurface, &data_ptr, &linesize)) {
                    // Create OpenCV Mat from the texture data
                    cv::Mat frame(height, width, CV_8UC4, data_ptr, linesize);

                    // Perform YOLO detection in a separate thread or queue it
                    // For now, we'll just copy the frame to be processed by the detection thread
                    {
                        std::lock_guard<std::mutex> lock(tf->inputBGRALock);
                        tf->inputBGRA = frame.clone();
                    }

                    gs_stagesurface_unmap(tf->stagesurface);
                }
            }
        }
    }

    // Render the original frame
    obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING);
    obs_source_process_filter_end(tf->source, _effect, width, height);

    // Draw detection results on top
    if (tf->drawBoundingBoxes || tf->drawLabels) {
        std::vector<Detection> currentDetections;
        {
            std::lock_guard<std::mutex> lock(tf->detectionsMutex);
            currentDetections = tf->detections;
        }

        // Draw bounding boxes using OBS drawing functions
        if (tf->drawBoundingBoxes) {
            gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
            gs_eparam_t *color_param = gs_effect_get_param_by_name(solid, "color");
            gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");

            for (const auto& detection : currentDetections) {
                if (detection.confidence < tf->confidenceThreshold) {
                    continue;
                }

                cv::Rect bbox = detection.getPixelBBox(width, height);

                // Draw bounding box
                struct vec4 color;
                vec4_set(&color, 
                         tf->boundingBoxColor[2]/255.0f,  // R
                         tf->boundingBoxColor[1]/255.0f,  // G
                         tf->boundingBoxColor[0]/255.0f,  // B
                         255.0f/255.0f);                  // A (fix alpha value)
                
                gs_effect_set_vec4(color_param, &color);
                
                gs_technique_begin(tech);
                gs_technique_begin_pass(tech, 0);

                // Draw rectangle outline using gs_draw_line
                gs_vertex2f(bbox.x, bbox.y);
                gs_vertex2f(bbox.x + bbox.width, bbox.y);
                
                gs_vertex2f(bbox.x + bbox.width, bbox.y);
                gs_vertex2f(bbox.x + bbox.width, bbox.y + bbox.height);
                
                gs_vertex2f(bbox.x + bbox.width, bbox.y + bbox.height);
                gs_vertex2f(bbox.x, bbox.y + bbox.height);
                
                gs_vertex2f(bbox.x, bbox.y + bbox.height);
                gs_vertex2f(bbox.x, bbox.y);

                gs_draw(GS_LINES, 0, 8);

                gs_technique_end_pass(tech);
                gs_technique_end(tech);
            }
        }

        // Note: For text labels, we would need to implement proper text rendering
        // which typically involves creating text textures or using OBS's text functions
        // For now, we'll log the labels that would be drawn
        if (tf->drawLabels) {
            for (const auto& detection : currentDetections) {
                if (detection.confidence < tf->confidenceThreshold) {
                    continue;
                }

                cv::Rect bbox = detection.getPixelBBox(width, height);
                std::string label = detection.className + " " + 
                                  std::to_string((int)(detection.confidence * 100)) + "%";
                
                obs_log(LOG_DEBUG, "Would draw label: %s at (%d, %d)", 
                       label.c_str(), bbox.x, bbox.y);
            }
        }
    }
}

// Detection thread function
void yolo_detection_thread(void *data)
{
    auto *tf_ptr = static_cast<yolo_filter_data*>(data);
    if (!tf_ptr) {
        return;
    }

    while (!tf_ptr->isDisabled) {
        // Wait a bit before next iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Adjust as needed

        // Check if we have a new frame to process
        {
            std::lock_guard<std::mutex> lock(tf_ptr->inputBGRALock);
            if (!tf_ptr->inputBGRA.empty()) {
                // Process the frame with YOLO
                try {
                    if (tf_ptr->model) {
                        auto detections = static_cast<ModelYOLO*>(tf_ptr->model.get())
                                            ->inference(tf_ptr->inputBGRA);
                        
                        // Update detections
                        {
                            std::lock_guard<std::mutex> lock_detections(tf_ptr->detectionsMutex);
                            tf_ptr->detections = detections;
                        }
                        
                        // Clear the input frame to indicate it's been processed
                        tf_ptr->inputBGRA = cv::Mat();
                    }
                } catch (const std::exception& e) {
                    obs_log(LOG_ERROR, "YOLO detection error: %s", e.what());
                }
            }
        }
    }
}