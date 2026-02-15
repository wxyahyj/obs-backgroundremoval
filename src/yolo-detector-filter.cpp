#include "yolo-detector-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
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
    ModelYOLO::Version modelVersion;

    std::vector<Detection> detections;
    std::mutex detectionsMutex;

    std::string modelPath;
    std::string classNamesPath;
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

    std::thread inferenceThread;
    std::atomic<bool> inferenceRunning;
    std::atomic<bool> shouldInference;
    int frameCounter;

    uint64_t totalFrames;
    uint64_t inferenceCount;
    double avgInferenceTimeMs;

    gs_effect_t *solidEffect;

    ~yolo_detector_filter() { obs_log(LOG_INFO, "YOLO detector filter destructor called"); }
};

void inferenceThreadWorker(yolo_detector_filter *filter);
static void renderDetectionBoxes(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
static void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);

const char *yolo_detector_filter_getname(void *unused)
{
    UNUSED_PARAMETER(unused);
    return obs_module_text("YOLODetector");
}

obs_properties_t *yolo_detector_filter_properties(void *data)
{
    obs_properties_t *props = obs_properties_create();

    obs_properties_add_group(props, "model_group", obs_module_text("ModelConfiguration"), OBS_GROUP_NORMAL, nullptr);

    obs_properties_add_path(props, "model_path", obs_module_text("ModelPath"), OBS_PATH_FILE, "ONNX Models (*.onnx)", nullptr);

    obs_property_t *modelVersion = obs_properties_add_list(props, "model_version", obs_module_text("ModelVersion"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(modelVersion, "YOLOv5", static_cast<int>(ModelYOLO::Version::YOLOv5));
    obs_property_list_add_int(modelVersion, "YOLOv8", static_cast<int>(ModelYOLO::Version::YOLOv8));
    obs_property_list_add_int(modelVersion, "YOLOv11", static_cast<int>(ModelYOLO::Version::YOLOv11));

    obs_properties_add_path(props, "class_names_path", obs_module_text("ClassNamesFile"), OBS_PATH_FILE, "Text Files (*.txt *.names)", nullptr);

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

    obs_properties_add_group(props, "advanced_group", obs_module_text("AdvancedConfiguration"), OBS_GROUP_NORMAL, nullptr);

    obs_properties_add_bool(props, "export_coordinates", obs_module_text("ExportCoordinates"));

    obs_properties_add_path(props, "coordinate_output_path", obs_module_text("CoordinateOutputPath"), OBS_PATH_FILE_SAVE, "JSON Files (*.json)", nullptr);

    obs_properties_add_group(props, "stats_group", obs_module_text("Statistics"), OBS_GROUP_NORMAL, nullptr);

    obs_properties_add_text(props, "avg_inference_time", obs_module_text("AvgInferenceTime"), OBS_TEXT_INFO);

    obs_properties_add_text(props, "detected_objects", obs_module_text("DetectedObjects"), OBS_TEXT_INFO);

    UNUSED_PARAMETER(data);
    return props;
}

void yolo_detector_filter_defaults(obs_data_t *settings)
{
    obs_data_set_default_string(settings, "model_path", "");
    obs_data_set_default_int(settings, "model_version", static_cast<int>(ModelYOLO::Version::YOLOv8));
    obs_data_set_default_string(settings, "class_names_path", "");
    obs_data_set_default_double(settings, "confidence_threshold", 0.5);
    obs_data_set_default_double(settings, "nms_threshold", 0.45);
    obs_data_set_default_int(settings, "target_class", -1);
    obs_data_set_default_int(settings, "inference_interval_frames", 3);
    obs_data_set_default_bool(settings, "show_bbox", true);
    obs_data_set_default_bool(settings, "show_label", true);
    obs_data_set_default_bool(settings, "show_confidence", true);
    obs_data_set_default_int(settings, "bbox_line_width", 2);
    obs_data_set_default_int(settings, "bbox_color", 0xFF00FF00);
    obs_data_set_default_bool(settings, "export_coordinates", false);
    obs_data_set_default_string(settings, "coordinate_output_path", "");
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
    
    if (newModelPath != tf->modelPath || !tf->yoloModel) {
        tf->modelPath = newModelPath;
        tf->modelVersion = newModelVersion;
        
        if (!tf->modelPath.empty()) {
            try {
                obs_log(LOG_INFO, "[YOLO Filter] Loading new model: %s", tf->modelPath.c_str());
                
                tf->yoloModel = std::make_unique<ModelYOLO>(tf->modelVersion);
                
                tf->yoloModel->loadModel(tf->modelPath);
                
                tf->classNamesPath = obs_data_get_string(settings, "class_names_path");
                if (!tf->classNamesPath.empty()) {
                    tf->yoloModel->loadClassNames(tf->classNamesPath);
                }
                
                obs_log(LOG_INFO, "[YOLO Filter] Model loaded successfully");
                
            } catch (const std::exception& e) {
                obs_log(LOG_ERROR, "[YOLO Filter] Failed to load model: %s", e.what());
                tf->yoloModel.reset();
            }
        }
    }
    
    tf->confidenceThreshold = (float)obs_data_get_double(settings, "confidence_threshold");
    tf->nmsThreshold = (float)obs_data_get_double(settings, "nms_threshold");
    tf->targetClassId = (int)obs_data_get_int(settings, "target_class");
    tf->inferenceIntervalFrames = (int)obs_data_get_int(settings, "inference_interval_frames");
    
    if (tf->yoloModel) {
        tf->yoloModel->setConfidenceThreshold(tf->confidenceThreshold);
        tf->yoloModel->setNMSThreshold(tf->nmsThreshold);
        tf->yoloModel->setTargetClass(tf->targetClassId);
    }
    
    tf->showBBox = obs_data_get_bool(settings, "show_bbox");
    tf->showLabel = obs_data_get_bool(settings, "show_label");
    tf->showConfidence = obs_data_get_bool(settings, "show_confidence");
    tf->bboxLineWidth = (int)obs_data_get_int(settings, "bbox_line_width");
    tf->bboxColor = (uint32_t)obs_data_get_int(settings, "bbox_color");
    
    tf->exportCoordinates = obs_data_get_bool(settings, "export_coordinates");
    tf->coordinateOutputPath = obs_data_get_string(settings, "coordinate_output_path");

    tf->isDisabled = false;
}

void inferenceThreadWorker(yolo_detector_filter *filter)
{
    obs_log(LOG_INFO, "[YOLO Detector] Inference thread started");

    while (filter->inferenceRunning) {
        if (!filter->shouldInference) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        filter->shouldInference = false;

        if (!filter->yoloModel) {
            continue;
        }

        uint8_t *data = nullptr;
        uint32_t linesize = 0;

        if (!gs_stagesurface_map(filter->stagesurface, &data, &linesize)) {
            continue;
        }

        uint32_t width = gs_stagesurface_get_width(filter->stagesurface);
        uint32_t height = gs_stagesurface_get_height(filter->stagesurface);

        cv::Mat frame(height, width, CV_8UC4, data, linesize);

        auto startTime = std::chrono::high_resolution_clock::now();

        std::vector<Detection> newDetections;
        try {
            newDetections = filter->yoloModel->inference(frame);
        } catch (const std::exception& e) {
            obs_log(LOG_ERROR, "[YOLO Detector] Inference error: %s", e.what());
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

        gs_stagesurface_unmap(filter->stagesurface);

        if (filter->exportCoordinates && !newDetections.empty()) {
            exportCoordinatesToFile(filter, width, height);
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
        vec4_set(&color, 0.0f, 1.0f, 0.0f, 1.0f);
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

        obs_log(LOG_DEBUG, "[YOLO Filter] Exported %zu detections to file", 
                filter->detections.size());

    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[YOLO Filter] Error exporting coordinates: %s", e.what());
    }
}

void *yolo_detector_filter_create(obs_data_t *settings, obs_source_t *source)
{
    auto filter = new std::shared_ptr<yolo_detector_filter>(std::make_shared<yolo_detector_filter>());
    auto &tf = *filter;

    tf->source = source;
    tf->inferenceRunning = false;
    tf->shouldInference = false;
    tf->frameCounter = 0;
    tf->totalFrames = 0;
    tf->inferenceCount = 0;
    tf->avgInferenceTimeMs = 0.0;

    obs_enter_graphics();
    tf->texrender = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
    tf->solidEffect = obs_get_base_effect(OBS_EFFECT_SOLID);
    obs_leave_graphics();

    yolo_detector_filter_update(filter, settings);

    tf->inferenceRunning = true;
    tf->inferenceThread = std::thread(inferenceThreadWorker, tf.get());

    obs_log(LOG_INFO, "[YOLO Detector] Filter created");

    return filter;
}

void yolo_detector_filter_destroy(void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
    if (!ptr) {
        return;
    }

    auto &tf = *ptr;
    if (!tf) {
        delete ptr;
        return;
    }

    tf->inferenceRunning = false;
    if (tf->inferenceThread.joinable()) {
        tf->inferenceThread.join();
    }

    obs_enter_graphics();
    if (tf->texrender) {
        gs_texrender_destroy(tf->texrender);
    }
    if (tf->stagesurface) {
        gs_stagesurface_destroy(tf->stagesurface);
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

    auto &tf = *ptr;
    if (!tf) {
        return;
    }

    obs_log(LOG_INFO, "YOLO detector filter activated");
}

void yolo_detector_filter_deactivate(void *data)
{
    auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
    if (!ptr) {
        return;
    }

    auto &tf = *ptr;
    if (!tf) {
        return;
    }

    obs_log(LOG_INFO, "YOLO detector filter deactivated");
}

void yolo_detector_filter_video_tick(void *data, float seconds)
{
    UNUSED_PARAMETER(seconds);

    auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
    if (!ptr) {
        return;
    }

    auto &tf = *ptr;
    if (!tf || tf->isDisabled) {
        return;
    }

    tf->totalFrames++;
    tf->frameCounter++;

    if (tf->frameCounter >= tf->inferenceIntervalFrames) {
        tf->frameCounter = 0;
        tf->shouldInference = true;
    }
}

void yolo_detector_filter_video_render(void *data, gs_effect_t *effect)
{
    UNUSED_PARAMETER(effect);

    auto *ptr = static_cast<std::shared_ptr<yolo_detector_filter> *>(data);
    if (!ptr) {
        return;
    }

    auto &tf = *ptr;
    if (!tf) {
        return;
    }

    if (!obs_source_process_filter_begin(tf->source, GS_RGBA, OBS_NO_DIRECT_RENDERING)) {
        return;
    }

    uint32_t width = obs_source_get_base_width(tf->source);
    uint32_t height = obs_source_get_base_height(tf->source);

    gs_texrender_reset(tf->texrender);
    if (gs_texrender_begin(tf->texrender, width, height)) {
        gs_blend_state_push();
        gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

        obs_source_process_filter_tech_end(tf->source, effect, width, height);

        gs_blend_state_pop();
        gs_texrender_end(tf->texrender);
    }

    if (!tf->stagesurface) {
        tf->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
    } else if (gs_stagesurface_get_width(tf->stagesurface) != width || gs_stagesurface_get_height(tf->stagesurface) != height) {
        gs_stagesurface_destroy(tf->stagesurface);
        tf->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);
    }

    gs_texture_t *tex = gs_texrender_get_texture(tf->texrender);
    if (tex) {
        gs_stage_texture(tf->stagesurface, tex);
    }

    obs_source_process_filter_end(tf->source, obs_get_base_effect(OBS_EFFECT_DEFAULT), width, height);

    if (tf->showBBox) {
        renderDetectionBoxes(tf.get(), width, height);
    }
}
