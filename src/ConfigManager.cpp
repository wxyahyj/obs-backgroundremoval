#ifdef _WIN32

#include "ConfigManager.hpp"
#include "plugin-support.h"

#include <obs-module.h>
#include <util/platform.h>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

ExtendedMouseControllerConfig::ExtendedMouseControllerConfig() {
    *this = getDefault();
}

ExtendedMouseControllerConfig ExtendedMouseControllerConfig::getDefault() {
    ExtendedMouseControllerConfig config;
    
    config.configName = "default";
    config.enableMouseControl = false;
    config.hotkeyVirtualKey = 0;
    config.fovRadiusPixels = 100;
    config.sourceCanvasPosX = 0.0f;
    config.sourceCanvasPosY = 0.0f;
    config.sourceCanvasScaleX = 1.0f;
    config.sourceCanvasScaleY = 1.0f;
    config.sourceWidth = 1920;
    config.sourceHeight = 1080;
    config.inferenceFrameWidth = 640;
    config.inferenceFrameHeight = 640;
    config.cropOffsetX = 0;
    config.cropOffsetY = 0;
    config.screenOffsetX = 0;
    config.screenOffsetY = 0;
    config.screenWidth = 1920;
    config.screenHeight = 1080;
    config.algorithmType = AlgorithmType::AdvancedPID;
    config.pidPMin = 0.15f;
    config.pidPMax = 0.6f;
    config.pidPSlope = 1.0f;
    config.pidD = 0.007f;
    config.pidI = 0.01f;
    config.aimSmoothingX = 0.7f;
    config.aimSmoothingY = 0.5f;
    config.maxPixelMove = 128.0f;
    config.deadZonePixels = 5.0f;
    config.targetYOffset = 0.0f;
    config.derivativeFilterAlpha = 0.2f;
    config.advTargetThreshold = 10.0f;
    config.advMinCoefficient = 1.5f;
    config.advMaxCoefficient = 2.5f;
    config.advTransitionSharpness = 5.0f;
    config.advTransitionMidpoint = 0.3f;
    config.advOutputSmoothing = 0.7f;
    config.advSpeedFactor = 0.5f;
    config.useOneEuroFilter = false;
    config.oneEuroMinCutoff = 1.0f;
    config.oneEuroBeta = 0.0f;
    config.oneEuroDCutoff = 1.0f;
    config.controllerType = ControllerType::WindowsAPI;
    config.makcuPort = "";
    config.makcuBaudRate = 115200;
    config.yUnlockDelayMs = 100;
    config.yUnlockEnabled = false;
    config.autoTriggerEnabled = false;
    config.autoTriggerRadius = 30;
    config.autoTriggerCooldownMs = 200;
    config.autoTriggerFireDelay = 0;
    config.autoTriggerFireDuration = 50;
    config.autoTriggerInterval = 50;
    config.autoTriggerDelayRandomEnabled = false;
    config.autoTriggerDelayRandomMin = 0;
    config.autoTriggerDelayRandomMax = 0;
    config.autoTriggerDurationRandomEnabled = false;
    config.autoTriggerDurationRandomMin = 0;
    config.autoTriggerDurationRandomMax = 0;
    config.autoTriggerMoveCompensation = 0;
    config.targetSwitchDelayMs = 500;
    config.targetSwitchTolerance = 0.15f;
    config.integralLimit = 100.0f;
    config.integralSeparationThreshold = 50.0f;
    config.integralDeadZone = 5.0f;
    config.integralRate = 0.015f;
    config.pGainRampInitialScale = 0.6f;
    config.pGainRampDuration = 0.5f;
    config.useDerivativePredictor = true;
    config.predictionWeightX = 0.3f;
    config.predictionWeightY = 0.1f;
    config.continuousAimEnabled = false;
    config.autoRecoilControlEnabled = false;
    config.recoilStrength = 5.0f;
    config.recoilSpeed = 16;
    config.recoilPidGainScale = 0.3f;
    config.enableBezierMovement = false;
    config.bezierCurvature = 0.3f;
    config.bezierRandomness = 0.2f;
    
    return config;
}

ConfigManager& ConfigManager::getInstance() {
    static ConfigManager instance;
    return instance;
}

ConfigManager::ConfigManager() {
    char* moduleConfigPath = obs_module_config_path("mouse_configs");
    if (moduleConfigPath) {
        configsDir = moduleConfigPath;
        bfree(moduleConfigPath);
    }
}

ConfigManager::~ConfigManager() {
}

void ConfigManager::setConfigsDirectory(const std::string& dir) {
    configsDir = dir;
}

std::string ConfigManager::getConfigsDirectory() {
    return configsDir;
}

bool ConfigManager::ensureConfigsDirectory() {
    if (configsDir.empty()) {
        obs_log(LOG_ERROR, "Config directory path is empty");
        return false;
    }
    
    std::error_code ec;
    if (!fs::exists(configsDir, ec)) {
        if (!fs::create_directories(configsDir, ec)) {
            obs_log(LOG_ERROR, "Failed to create config directory: %s", configsDir.c_str());
            return false;
        }
    }
    return true;
}

std::string ConfigManager::getConfigFilePath(const std::string& configName) {
    std::string safeName = configName;
    for (char& c : safeName) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || 
            c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return configsDir + "/" + safeName + ".json";
}

nlohmann::json ConfigManager::configToJson(const ExtendedMouseControllerConfig& config) {
    nlohmann::json j;
    
    j["configName"] = config.configName;
    j["enableMouseControl"] = config.enableMouseControl;
    j["hotkeyVirtualKey"] = config.hotkeyVirtualKey;
    j["fovRadiusPixels"] = config.fovRadiusPixels;
    j["sourceCanvasPosX"] = config.sourceCanvasPosX;
    j["sourceCanvasPosY"] = config.sourceCanvasPosY;
    j["sourceCanvasScaleX"] = config.sourceCanvasScaleX;
    j["sourceCanvasScaleY"] = config.sourceCanvasScaleY;
    j["sourceWidth"] = config.sourceWidth;
    j["sourceHeight"] = config.sourceHeight;
    j["inferenceFrameWidth"] = config.inferenceFrameWidth;
    j["inferenceFrameHeight"] = config.inferenceFrameHeight;
    j["cropOffsetX"] = config.cropOffsetX;
    j["cropOffsetY"] = config.cropOffsetY;
    j["screenOffsetX"] = config.screenOffsetX;
    j["screenOffsetY"] = config.screenOffsetY;
    j["screenWidth"] = config.screenWidth;
    j["screenHeight"] = config.screenHeight;
    j["algorithmType"] = static_cast<int>(config.algorithmType);
    j["pidPMin"] = config.pidPMin;
    j["pidPMax"] = config.pidPMax;
    j["pidPSlope"] = config.pidPSlope;
    j["pidD"] = config.pidD;
    j["pidI"] = config.pidI;
    j["aimSmoothingX"] = config.aimSmoothingX;
    j["aimSmoothingY"] = config.aimSmoothingY;
    j["maxPixelMove"] = config.maxPixelMove;
    j["deadZonePixels"] = config.deadZonePixels;
    j["targetYOffset"] = config.targetYOffset;
    j["derivativeFilterAlpha"] = config.derivativeFilterAlpha;
    j["advTargetThreshold"] = config.advTargetThreshold;
    j["advMinCoefficient"] = config.advMinCoefficient;
    j["advMaxCoefficient"] = config.advMaxCoefficient;
    j["advTransitionSharpness"] = config.advTransitionSharpness;
    j["advTransitionMidpoint"] = config.advTransitionMidpoint;
    j["advOutputSmoothing"] = config.advOutputSmoothing;
    j["advSpeedFactor"] = config.advSpeedFactor;
    j["useOneEuroFilter"] = config.useOneEuroFilter;
    j["oneEuroMinCutoff"] = config.oneEuroMinCutoff;
    j["oneEuroBeta"] = config.oneEuroBeta;
    j["oneEuroDCutoff"] = config.oneEuroDCutoff;
    j["controllerType"] = static_cast<int>(config.controllerType);
    j["makcuPort"] = config.makcuPort;
    j["makcuBaudRate"] = config.makcuBaudRate;
    j["yUnlockDelayMs"] = config.yUnlockDelayMs;
    j["yUnlockEnabled"] = config.yUnlockEnabled;
    j["autoTriggerEnabled"] = config.autoTriggerEnabled;
    j["autoTriggerRadius"] = config.autoTriggerRadius;
    j["autoTriggerCooldownMs"] = config.autoTriggerCooldownMs;
    j["autoTriggerFireDelay"] = config.autoTriggerFireDelay;
    j["autoTriggerFireDuration"] = config.autoTriggerFireDuration;
    j["autoTriggerInterval"] = config.autoTriggerInterval;
    j["autoTriggerDelayRandomEnabled"] = config.autoTriggerDelayRandomEnabled;
    j["autoTriggerDelayRandomMin"] = config.autoTriggerDelayRandomMin;
    j["autoTriggerDelayRandomMax"] = config.autoTriggerDelayRandomMax;
    j["autoTriggerDurationRandomEnabled"] = config.autoTriggerDurationRandomEnabled;
    j["autoTriggerDurationRandomMin"] = config.autoTriggerDurationRandomMin;
    j["autoTriggerDurationRandomMax"] = config.autoTriggerDurationRandomMax;
    j["autoTriggerMoveCompensation"] = config.autoTriggerMoveCompensation;
    j["targetSwitchDelayMs"] = config.targetSwitchDelayMs;
    j["targetSwitchTolerance"] = config.targetSwitchTolerance;
    j["integralLimit"] = config.integralLimit;
    j["integralSeparationThreshold"] = config.integralSeparationThreshold;
    j["integralDeadZone"] = config.integralDeadZone;
    j["integralRate"] = config.integralRate;
    j["pGainRampInitialScale"] = config.pGainRampInitialScale;
    j["pGainRampDuration"] = config.pGainRampDuration;
    j["useDerivativePredictor"] = config.useDerivativePredictor;
    j["predictionWeightX"] = config.predictionWeightX;
    j["predictionWeightY"] = config.predictionWeightY;
    j["continuousAimEnabled"] = config.continuousAimEnabled;
    j["autoRecoilControlEnabled"] = config.autoRecoilControlEnabled;
    j["recoilStrength"] = config.recoilStrength;
    j["recoilSpeed"] = config.recoilSpeed;
    j["recoilPidGainScale"] = config.recoilPidGainScale;
    j["enableBezierMovement"] = config.enableBezierMovement;
    j["bezierCurvature"] = config.bezierCurvature;
    j["bezierRandomness"] = config.bezierRandomness;
    
    j["dynamicKp"] = config.dynamicKp;
    j["dynamicKi"] = config.dynamicKi;
    j["dynamicKd"] = config.dynamicKd;
    j["dynamicTargetThreshold"] = config.dynamicTargetThreshold;
    j["dynamicSpeedMultiplier"] = config.dynamicSpeedMultiplier;
    j["dynamicMinCoefficient"] = config.dynamicMinCoefficient;
    j["dynamicMaxCoefficient"] = config.dynamicMaxCoefficient;
    j["dynamicTransitionSharpness"] = config.dynamicTransitionSharpness;
    j["dynamicTransitionMidpoint"] = config.dynamicTransitionMidpoint;
    j["dynamicMinDataPoints"] = config.dynamicMinDataPoints;
    j["dynamicErrorTolerance"] = config.dynamicErrorTolerance;
    j["dynamicSmoothingFactor"] = config.dynamicSmoothingFactor;
    
    return j;
}

bool ConfigManager::jsonToConfig(const nlohmann::json& j, ExtendedMouseControllerConfig& config) {
    config = ExtendedMouseControllerConfig::getDefault();
    
    try {
        if (j.contains("configName")) config.configName = j["configName"].get<std::string>();
        if (j.contains("enableMouseControl")) config.enableMouseControl = j["enableMouseControl"].get<bool>();
        if (j.contains("hotkeyVirtualKey")) config.hotkeyVirtualKey = j["hotkeyVirtualKey"].get<int>();
        if (j.contains("fovRadiusPixels")) config.fovRadiusPixels = j["fovRadiusPixels"].get<int>();
        if (j.contains("sourceCanvasPosX")) config.sourceCanvasPosX = j["sourceCanvasPosX"].get<float>();
        if (j.contains("sourceCanvasPosY")) config.sourceCanvasPosY = j["sourceCanvasPosY"].get<float>();
        if (j.contains("sourceCanvasScaleX")) config.sourceCanvasScaleX = j["sourceCanvasScaleX"].get<float>();
        if (j.contains("sourceCanvasScaleY")) config.sourceCanvasScaleY = j["sourceCanvasScaleY"].get<float>();
        if (j.contains("sourceWidth")) config.sourceWidth = j["sourceWidth"].get<int>();
        if (j.contains("sourceHeight")) config.sourceHeight = j["sourceHeight"].get<int>();
        if (j.contains("inferenceFrameWidth")) config.inferenceFrameWidth = j["inferenceFrameWidth"].get<int>();
        if (j.contains("inferenceFrameHeight")) config.inferenceFrameHeight = j["inferenceFrameHeight"].get<int>();
        if (j.contains("cropOffsetX")) config.cropOffsetX = j["cropOffsetX"].get<int>();
        if (j.contains("cropOffsetY")) config.cropOffsetY = j["cropOffsetY"].get<int>();
        if (j.contains("screenOffsetX")) config.screenOffsetX = j["screenOffsetX"].get<int>();
        if (j.contains("screenOffsetY")) config.screenOffsetY = j["screenOffsetY"].get<int>();
        if (j.contains("screenWidth")) config.screenWidth = j["screenWidth"].get<int>();
        if (j.contains("screenHeight")) config.screenHeight = j["screenHeight"].get<int>();
        if (j.contains("algorithmType")) config.algorithmType = static_cast<AlgorithmType>(j["algorithmType"].get<int>());
        if (j.contains("pidPMin")) config.pidPMin = j["pidPMin"].get<float>();
        if (j.contains("pidPMax")) config.pidPMax = j["pidPMax"].get<float>();
        if (j.contains("pidPSlope")) config.pidPSlope = j["pidPSlope"].get<float>();
        if (j.contains("pidD")) config.pidD = j["pidD"].get<float>();
        if (j.contains("pidI")) config.pidI = j["pidI"].get<float>();
        if (j.contains("aimSmoothingX")) config.aimSmoothingX = j["aimSmoothingX"].get<float>();
        if (j.contains("aimSmoothingY")) config.aimSmoothingY = j["aimSmoothingY"].get<float>();
        if (j.contains("maxPixelMove")) config.maxPixelMove = j["maxPixelMove"].get<float>();
        if (j.contains("deadZonePixels")) config.deadZonePixels = j["deadZonePixels"].get<float>();
        if (j.contains("targetYOffset")) config.targetYOffset = j["targetYOffset"].get<float>();
        if (j.contains("derivativeFilterAlpha")) config.derivativeFilterAlpha = j["derivativeFilterAlpha"].get<float>();
        if (j.contains("advTargetThreshold")) config.advTargetThreshold = j["advTargetThreshold"].get<float>();
        if (j.contains("advMinCoefficient")) config.advMinCoefficient = j["advMinCoefficient"].get<float>();
        if (j.contains("advMaxCoefficient")) config.advMaxCoefficient = j["advMaxCoefficient"].get<float>();
        if (j.contains("advTransitionSharpness")) config.advTransitionSharpness = j["advTransitionSharpness"].get<float>();
        if (j.contains("advTransitionMidpoint")) config.advTransitionMidpoint = j["advTransitionMidpoint"].get<float>();
        if (j.contains("advOutputSmoothing")) config.advOutputSmoothing = j["advOutputSmoothing"].get<float>();
        if (j.contains("advSpeedFactor")) config.advSpeedFactor = j["advSpeedFactor"].get<float>();
        if (j.contains("useOneEuroFilter")) config.useOneEuroFilter = j["useOneEuroFilter"].get<bool>();
        if (j.contains("oneEuroMinCutoff")) config.oneEuroMinCutoff = j["oneEuroMinCutoff"].get<float>();
        if (j.contains("oneEuroBeta")) config.oneEuroBeta = j["oneEuroBeta"].get<float>();
        if (j.contains("oneEuroDCutoff")) config.oneEuroDCutoff = j["oneEuroDCutoff"].get<float>();
        if (j.contains("controllerType")) config.controllerType = static_cast<ControllerType>(j["controllerType"].get<int>());
        if (j.contains("makcuPort")) config.makcuPort = j["makcuPort"].get<std::string>();
        if (j.contains("makcuBaudRate")) config.makcuBaudRate = j["makcuBaudRate"].get<int>();
        if (j.contains("yUnlockDelayMs")) config.yUnlockDelayMs = j["yUnlockDelayMs"].get<int>();
        if (j.contains("yUnlockEnabled")) config.yUnlockEnabled = j["yUnlockEnabled"].get<bool>();
        if (j.contains("autoTriggerEnabled")) config.autoTriggerEnabled = j["autoTriggerEnabled"].get<bool>();
        if (j.contains("autoTriggerRadius")) config.autoTriggerRadius = j["autoTriggerRadius"].get<int>();
        if (j.contains("autoTriggerCooldownMs")) config.autoTriggerCooldownMs = j["autoTriggerCooldownMs"].get<int>();
        if (j.contains("autoTriggerFireDelay")) config.autoTriggerFireDelay = j["autoTriggerFireDelay"].get<int>();
        if (j.contains("autoTriggerFireDuration")) config.autoTriggerFireDuration = j["autoTriggerFireDuration"].get<int>();
        if (j.contains("autoTriggerInterval")) config.autoTriggerInterval = j["autoTriggerInterval"].get<int>();
        if (j.contains("autoTriggerDelayRandomEnabled")) config.autoTriggerDelayRandomEnabled = j["autoTriggerDelayRandomEnabled"].get<bool>();
        if (j.contains("autoTriggerDelayRandomMin")) config.autoTriggerDelayRandomMin = j["autoTriggerDelayRandomMin"].get<int>();
        if (j.contains("autoTriggerDelayRandomMax")) config.autoTriggerDelayRandomMax = j["autoTriggerDelayRandomMax"].get<int>();
        if (j.contains("autoTriggerDurationRandomEnabled")) config.autoTriggerDurationRandomEnabled = j["autoTriggerDurationRandomEnabled"].get<bool>();
        if (j.contains("autoTriggerDurationRandomMin")) config.autoTriggerDurationRandomMin = j["autoTriggerDurationRandomMin"].get<int>();
        if (j.contains("autoTriggerDurationRandomMax")) config.autoTriggerDurationRandomMax = j["autoTriggerDurationRandomMax"].get<int>();
        if (j.contains("autoTriggerMoveCompensation")) config.autoTriggerMoveCompensation = j["autoTriggerMoveCompensation"].get<int>();
        if (j.contains("targetSwitchDelayMs")) config.targetSwitchDelayMs = j["targetSwitchDelayMs"].get<int>();
        if (j.contains("targetSwitchTolerance")) config.targetSwitchTolerance = j["targetSwitchTolerance"].get<float>();
        if (j.contains("integralLimit")) config.integralLimit = j["integralLimit"].get<float>();
        if (j.contains("integralSeparationThreshold")) config.integralSeparationThreshold = j["integralSeparationThreshold"].get<float>();
        if (j.contains("integralDeadZone")) config.integralDeadZone = j["integralDeadZone"].get<float>();
        if (j.contains("integralRate")) config.integralRate = j["integralRate"].get<float>();
        if (j.contains("pGainRampInitialScale")) config.pGainRampInitialScale = j["pGainRampInitialScale"].get<float>();
        if (j.contains("pGainRampDuration")) config.pGainRampDuration = j["pGainRampDuration"].get<float>();
        if (j.contains("useDerivativePredictor")) config.useDerivativePredictor = j["useDerivativePredictor"].get<bool>();
        if (j.contains("predictionWeightX")) config.predictionWeightX = j["predictionWeightX"].get<float>();
        if (j.contains("predictionWeightY")) config.predictionWeightY = j["predictionWeightY"].get<float>();
        if (j.contains("continuousAimEnabled")) config.continuousAimEnabled = j["continuousAimEnabled"].get<bool>();
        if (j.contains("autoRecoilControlEnabled")) config.autoRecoilControlEnabled = j["autoRecoilControlEnabled"].get<bool>();
        if (j.contains("recoilStrength")) config.recoilStrength = j["recoilStrength"].get<float>();
        if (j.contains("recoilSpeed")) config.recoilSpeed = j["recoilSpeed"].get<int>();
        if (j.contains("recoilPidGainScale")) config.recoilPidGainScale = j["recoilPidGainScale"].get<float>();
        if (j.contains("enableBezierMovement")) config.enableBezierMovement = j["enableBezierMovement"].get<bool>();
        if (j.contains("bezierCurvature")) config.bezierCurvature = j["bezierCurvature"].get<float>();
        if (j.contains("bezierRandomness")) config.bezierRandomness = j["bezierRandomness"].get<float>();
        
        if (j.contains("dynamicKp")) config.dynamicKp = j["dynamicKp"].get<float>();
        if (j.contains("dynamicKi")) config.dynamicKi = j["dynamicKi"].get<float>();
        if (j.contains("dynamicKd")) config.dynamicKd = j["dynamicKd"].get<float>();
        if (j.contains("dynamicTargetThreshold")) config.dynamicTargetThreshold = j["dynamicTargetThreshold"].get<float>();
        if (j.contains("dynamicSpeedMultiplier")) config.dynamicSpeedMultiplier = j["dynamicSpeedMultiplier"].get<float>();
        if (j.contains("dynamicMinCoefficient")) config.dynamicMinCoefficient = j["dynamicMinCoefficient"].get<float>();
        if (j.contains("dynamicMaxCoefficient")) config.dynamicMaxCoefficient = j["dynamicMaxCoefficient"].get<float>();
        if (j.contains("dynamicTransitionSharpness")) config.dynamicTransitionSharpness = j["dynamicTransitionSharpness"].get<float>();
        if (j.contains("dynamicTransitionMidpoint")) config.dynamicTransitionMidpoint = j["dynamicTransitionMidpoint"].get<float>();
        if (j.contains("dynamicMinDataPoints")) config.dynamicMinDataPoints = j["dynamicMinDataPoints"].get<int>();
        if (j.contains("dynamicErrorTolerance")) config.dynamicErrorTolerance = j["dynamicErrorTolerance"].get<float>();
        if (j.contains("dynamicSmoothingFactor")) config.dynamicSmoothingFactor = j["dynamicSmoothingFactor"].get<float>();
        
        return true;
    } catch (const nlohmann::json::exception& e) {
        obs_log(LOG_ERROR, "JSON parsing error: %s", e.what());
        return false;
    }
}

bool ConfigManager::saveConfig(const ExtendedMouseControllerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!ensureConfigsDirectory()) {
        return false;
    }
    
    std::string filePath = getConfigFilePath(config.configName);
    
    try {
        nlohmann::json j = configToJson(config);
        std::ofstream file(filePath);
        if (!file.is_open()) {
            obs_log(LOG_ERROR, "Failed to open config file for writing: %s", filePath.c_str());
            return false;
        }
        
        file << j.dump(2);
        file.close();
        
        obs_log(LOG_INFO, "Config saved: %s", config.configName.c_str());
        return true;
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "Failed to save config: %s", e.what());
        return false;
    }
}

bool ConfigManager::loadConfig(const std::string& configName, ExtendedMouseControllerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::string filePath = getConfigFilePath(configName);
    
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            obs_log(LOG_ERROR, "Failed to open config file for reading: %s", filePath.c_str());
            return false;
        }
        
        nlohmann::json j = nlohmann::json::parse(file);
        file.close();
        
        if (!jsonToConfig(j, config)) {
            obs_log(LOG_ERROR, "Failed to parse config file: %s", filePath.c_str());
            return false;
        }
        
        obs_log(LOG_INFO, "Config loaded: %s", configName.c_str());
        return true;
    } catch (const nlohmann::json::exception& e) {
        obs_log(LOG_ERROR, "JSON parsing error: %s", e.what());
        return false;
    }
}

bool ConfigManager::deleteConfig(const std::string& configName) {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::string filePath = getConfigFilePath(configName);
    
    std::error_code ec;
    if (!fs::exists(filePath, ec)) {
        obs_log(LOG_WARNING, "Config file does not exist: %s", filePath.c_str());
        return false;
    }
    
    if (!fs::remove(filePath, ec)) {
        obs_log(LOG_ERROR, "Failed to delete config file: %s", filePath.c_str());
        return false;
    }
    
    obs_log(LOG_INFO, "Config deleted: %s", configName.c_str());
    return true;
}

std::vector<std::string> ConfigManager::listConfigs() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<std::string> configs;
    
    if (!ensureConfigsDirectory()) {
        return configs;
    }
    
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(configsDir, ec)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.size() > 5 && filename.substr(filename.size() - 5) == ".json") {
                configs.push_back(filename.substr(0, filename.size() - 5));
            }
        }
    }
    
    std::sort(configs.begin(), configs.end());
    return configs;
}

bool ConfigManager::configExists(const std::string& configName) {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::string filePath = getConfigFilePath(configName);
    std::error_code ec;
    return fs::exists(filePath, ec);
}

#endif
