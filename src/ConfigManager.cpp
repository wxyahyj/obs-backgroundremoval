#ifdef _WIN32

#include "ConfigManager.hpp"
#include "plugin-support.h"

#include <obs-module.h>
#include <util/platform.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

ExtendedMouseControllerConfig::ExtendedMouseControllerConfig() {
    *this = getDefault();
}

ExtendedMouseControllerConfig ExtendedMouseControllerConfig::getDefault() {
    ExtendedMouseControllerConfig config;
    
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
    config.pidPMin = 0.1f;
    config.pidPMax = 0.5f;
    config.pidPSlope = 0.001f;
    config.pidD = 0.05f;
    config.baselineCompensation = 0.0f;
    config.aimSmoothingX = 0.0f;
    config.aimSmoothingY = 0.0f;
    config.maxPixelMove = 10.0f;
    config.deadZonePixels = 5.0f;
    config.targetYOffset = 0.0f;
    config.derivativeFilterAlpha = 0.5f;
    config.controllerType = ControllerType::WindowsAPI;
    config.makcuPort = "";
    config.makcuBaudRate = 115200;
    
    config.yUnlockDelayMs = 100;
    config.yUnlockEnabled = false;
    config.autoTriggerEnabled = false;
    config.autoTriggerRadius = 30.0f;
    config.autoTriggerCooldownMs = 200;
    config.configName = "default";
    
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

std::string ConfigManager::escapeJsonString(const std::string& str) {
    std::string result;
    result.reserve(str.size() * 2);
    
    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

std::string ConfigManager::unescapeJsonString(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 1 < str.size()) {
            switch (str[i + 1]) {
                case '"': result += '"'; ++i; break;
                case '\\': result += '\\'; ++i; break;
                case 'b': result += '\b'; ++i; break;
                case 'f': result += '\f'; ++i; break;
                case 'n': result += '\n'; ++i; break;
                case 'r': result += '\r'; ++i; break;
                case 't': result += '\t'; ++i; break;
                case 'u': {
                    if (i + 5 < str.size()) {
                        char buf[5] = {str[i+2], str[i+3], str[i+4], str[i+5], 0};
                        unsigned int codepoint = strtoul(buf, nullptr, 16);
                        if (codepoint < 0x80) {
                            result += static_cast<char>(codepoint);
                        } else if (codepoint < 0x800) {
                            result += static_cast<char>(0xC0 | (codepoint >> 6));
                            result += static_cast<char>(0x80 | (codepoint & 0x3F));
                        } else {
                            result += static_cast<char>(0xE0 | (codepoint >> 12));
                            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (codepoint & 0x3F));
                        }
                        i += 5;
                    }
                    break;
                }
                default: result += str[i]; break;
            }
        } else {
            result += str[i];
        }
    }
    return result;
}

std::string ConfigManager::configToJson(const ExtendedMouseControllerConfig& config) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"configName\": \"" << escapeJsonString(config.configName) << "\",\n";
    oss << "  \"enableMouseControl\": " << (config.enableMouseControl ? "true" : "false") << ",\n";
    oss << "  \"hotkeyVirtualKey\": " << config.hotkeyVirtualKey << ",\n";
    oss << "  \"fovRadiusPixels\": " << config.fovRadiusPixels << ",\n";
    oss << "  \"sourceCanvasPosX\": " << config.sourceCanvasPosX << ",\n";
    oss << "  \"sourceCanvasPosY\": " << config.sourceCanvasPosY << ",\n";
    oss << "  \"sourceCanvasScaleX\": " << config.sourceCanvasScaleX << ",\n";
    oss << "  \"sourceCanvasScaleY\": " << config.sourceCanvasScaleY << ",\n";
    oss << "  \"sourceWidth\": " << config.sourceWidth << ",\n";
    oss << "  \"sourceHeight\": " << config.sourceHeight << ",\n";
    oss << "  \"inferenceFrameWidth\": " << config.inferenceFrameWidth << ",\n";
    oss << "  \"inferenceFrameHeight\": " << config.inferenceFrameHeight << ",\n";
    oss << "  \"cropOffsetX\": " << config.cropOffsetX << ",\n";
    oss << "  \"cropOffsetY\": " << config.cropOffsetY << ",\n";
    oss << "  \"screenOffsetX\": " << config.screenOffsetX << ",\n";
    oss << "  \"screenOffsetY\": " << config.screenOffsetY << ",\n";
    oss << "  \"screenWidth\": " << config.screenWidth << ",\n";
    oss << "  \"screenHeight\": " << config.screenHeight << ",\n";
    oss << "  \"pidPMin\": " << config.pidPMin << ",\n";
    oss << "  \"pidPMax\": " << config.pidPMax << ",\n";
    oss << "  \"pidPSlope\": " << config.pidPSlope << ",\n";
    oss << "  \"pidD\": " << config.pidD << ",\n";
    oss << "  \"baselineCompensation\": " << config.baselineCompensation << ",\n";
    oss << "  \"aimSmoothingX\": " << config.aimSmoothingX << ",\n";
    oss << "  \"aimSmoothingY\": " << config.aimSmoothingY << ",\n";
    oss << "  \"maxPixelMove\": " << config.maxPixelMove << ",\n";
    oss << "  \"deadZonePixels\": " << config.deadZonePixels << ",\n";
    oss << "  \"targetYOffset\": " << config.targetYOffset << ",\n";
    oss << "  \"derivativeFilterAlpha\": " << config.derivativeFilterAlpha << ",\n";
    oss << "  \"controllerType\": " << static_cast<int>(config.controllerType) << ",\n";
    oss << "  \"makcuPort\": \"" << escapeJsonString(config.makcuPort) << "\",\n";
    oss << "  \"makcuBaudRate\": " << config.makcuBaudRate << ",\n";
    oss << "  \"yUnlockDelayMs\": " << config.yUnlockDelayMs << ",\n";
    oss << "  \"yUnlockEnabled\": " << (config.yUnlockEnabled ? "true" : "false") << ",\n";
    oss << "  \"autoTriggerEnabled\": " << (config.autoTriggerEnabled ? "true" : "false") << ",\n";
    oss << "  \"autoTriggerRadius\": " << config.autoTriggerRadius << ",\n";
    oss << "  \"autoTriggerCooldownMs\": " << config.autoTriggerCooldownMs << "\n";
    oss << "}\n";
    return oss.str();
}

bool ConfigManager::jsonToConfig(const std::string& json, ExtendedMouseControllerConfig& config) {
    config = ExtendedMouseControllerConfig::getDefault();
    
    auto extractString = [&json](const std::string& key) -> std::string {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos) return "";
        
        pos = json.find(':', pos);
        if (pos == std::string::npos) return "";
        
        pos = json.find('"', pos);
        if (pos == std::string::npos) return "";
        
        size_t endPos = json.find('"', pos + 1);
        if (endPos == std::string::npos) return "";
        
        return unescapeJsonString(json.substr(pos + 1, endPos - pos - 1));
    };
    
    auto extractNumber = [&json](const std::string& key) -> std::pair<bool, double> {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos) return {false, 0.0};
        
        pos = json.find(':', pos);
        if (pos == std::string::npos) return {false, 0.0};
        
        while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) {
            ++pos;
        }
        
        if (pos >= json.size()) return {false, 0.0};
        
        std::string numStr;
        bool hasDecimal = false;
        bool hasSign = false;
        
        while (pos < json.size()) {
            char c = json[pos];
            if (c == '-' && !hasSign && numStr.empty()) {
                numStr += c;
                hasSign = true;
            } else if (c >= '0' && c <= '9') {
                numStr += c;
            } else if (c == '.' && !hasDecimal) {
                numStr += c;
                hasDecimal = true;
            } else {
                break;
            }
            ++pos;
        }
        
        if (numStr.empty() || numStr == "-") return {false, 0.0};
        
        try {
            return {true, std::stod(numStr)};
        } catch (...) {
            return {false, 0.0};
        }
    };
    
    auto extractBool = [&json](const std::string& key) -> std::pair<bool, bool> {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = json.find(searchKey);
        if (pos == std::string::npos) return {false, false};
        
        pos = json.find(':', pos);
        if (pos == std::string::npos) return {false, false};
        
        size_t truePos = json.find("true", pos);
        size_t falsePos = json.find("false", pos);
        
        if (truePos != std::string::npos && (falsePos == std::string::npos || truePos < falsePos)) {
            return {true, true};
        } else if (falsePos != std::string::npos) {
            return {true, false};
        }
        return {false, false};
    };
    
    config.configName = extractString("configName");
    if (config.configName.empty()) {
        config.configName = "unnamed";
    }
    
    auto boolVal = extractBool("enableMouseControl");
    if (boolVal.first) config.enableMouseControl = boolVal.second;
    
    auto numVal = extractNumber("hotkeyVirtualKey");
    if (numVal.first) config.hotkeyVirtualKey = static_cast<int>(numVal.second);
    
    numVal = extractNumber("fovRadiusPixels");
    if (numVal.first) config.fovRadiusPixels = static_cast<int>(numVal.second);
    
    numVal = extractNumber("sourceCanvasPosX");
    if (numVal.first) config.sourceCanvasPosX = static_cast<float>(numVal.second);
    
    numVal = extractNumber("sourceCanvasPosY");
    if (numVal.first) config.sourceCanvasPosY = static_cast<float>(numVal.second);
    
    numVal = extractNumber("sourceCanvasScaleX");
    if (numVal.first) config.sourceCanvasScaleX = static_cast<float>(numVal.second);
    
    numVal = extractNumber("sourceCanvasScaleY");
    if (numVal.first) config.sourceCanvasScaleY = static_cast<float>(numVal.second);
    
    numVal = extractNumber("sourceWidth");
    if (numVal.first) config.sourceWidth = static_cast<int>(numVal.second);
    
    numVal = extractNumber("sourceHeight");
    if (numVal.first) config.sourceHeight = static_cast<int>(numVal.second);
    
    numVal = extractNumber("inferenceFrameWidth");
    if (numVal.first) config.inferenceFrameWidth = static_cast<int>(numVal.second);
    
    numVal = extractNumber("inferenceFrameHeight");
    if (numVal.first) config.inferenceFrameHeight = static_cast<int>(numVal.second);
    
    numVal = extractNumber("cropOffsetX");
    if (numVal.first) config.cropOffsetX = static_cast<int>(numVal.second);
    
    numVal = extractNumber("cropOffsetY");
    if (numVal.first) config.cropOffsetY = static_cast<int>(numVal.second);
    
    numVal = extractNumber("screenOffsetX");
    if (numVal.first) config.screenOffsetX = static_cast<int>(numVal.second);
    
    numVal = extractNumber("screenOffsetY");
    if (numVal.first) config.screenOffsetY = static_cast<int>(numVal.second);
    
    numVal = extractNumber("screenWidth");
    if (numVal.first) config.screenWidth = static_cast<int>(numVal.second);
    
    numVal = extractNumber("screenHeight");
    if (numVal.first) config.screenHeight = static_cast<int>(numVal.second);
    
    numVal = extractNumber("pidPMin");
    if (numVal.first) config.pidPMin = static_cast<float>(numVal.second);
    
    numVal = extractNumber("pidPMax");
    if (numVal.first) config.pidPMax = static_cast<float>(numVal.second);
    
    numVal = extractNumber("pidPSlope");
    if (numVal.first) config.pidPSlope = static_cast<float>(numVal.second);
    
    numVal = extractNumber("pidD");
    if (numVal.first) config.pidD = static_cast<float>(numVal.second);
    
    numVal = extractNumber("baselineCompensation");
    if (numVal.first) config.baselineCompensation = static_cast<float>(numVal.second);
    
    numVal = extractNumber("aimSmoothingX");
    if (numVal.first) config.aimSmoothingX = static_cast<float>(numVal.second);
    
    numVal = extractNumber("aimSmoothingY");
    if (numVal.first) config.aimSmoothingY = static_cast<float>(numVal.second);
    
    numVal = extractNumber("maxPixelMove");
    if (numVal.first) config.maxPixelMove = static_cast<float>(numVal.second);
    
    numVal = extractNumber("deadZonePixels");
    if (numVal.first) config.deadZonePixels = static_cast<float>(numVal.second);
    
    numVal = extractNumber("targetYOffset");
    if (numVal.first) config.targetYOffset = static_cast<float>(numVal.second);
    
    numVal = extractNumber("derivativeFilterAlpha");
    if (numVal.first) config.derivativeFilterAlpha = static_cast<float>(numVal.second);
    
    numVal = extractNumber("controllerType");
    if (numVal.first) config.controllerType = static_cast<ControllerType>(static_cast<int>(numVal.second));
    
    config.makcuPort = extractString("makcuPort");
    
    numVal = extractNumber("makcuBaudRate");
    if (numVal.first) config.makcuBaudRate = static_cast<int>(numVal.second);
    
    numVal = extractNumber("yUnlockDelayMs");
    if (numVal.first) config.yUnlockDelayMs = static_cast<int>(numVal.second);
    
    boolVal = extractBool("yUnlockEnabled");
    if (boolVal.first) config.yUnlockEnabled = boolVal.second;
    
    boolVal = extractBool("autoTriggerEnabled");
    if (boolVal.first) config.autoTriggerEnabled = boolVal.second;
    
    numVal = extractNumber("autoTriggerRadius");
    if (numVal.first) config.autoTriggerRadius = static_cast<float>(numVal.second);
    
    numVal = extractNumber("autoTriggerCooldownMs");
    if (numVal.first) config.autoTriggerCooldownMs = static_cast<int>(numVal.second);
    
    return true;
}

bool ConfigManager::saveConfig(const ExtendedMouseControllerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!ensureConfigsDirectory()) {
        return false;
    }
    
    std::string filePath = getConfigFilePath(config.configName);
    std::string json = configToJson(config);
    
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        obs_log(LOG_ERROR, "Failed to open config file for writing: %s", filePath.c_str());
        return false;
    }
    
    file << json;
    file.close();
    
    if (!file) {
        obs_log(LOG_ERROR, "Failed to write config file: %s", filePath.c_str());
        return false;
    }
    
    obs_log(LOG_INFO, "Config saved: %s", config.configName.c_str());
    return true;
}

bool ConfigManager::loadConfig(const std::string& configName, ExtendedMouseControllerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::string filePath = getConfigFilePath(configName);
    
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        obs_log(LOG_ERROR, "Failed to open config file for reading: %s", filePath.c_str());
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    std::string json = buffer.str();
    
    if (!jsonToConfig(json, config)) {
        obs_log(LOG_ERROR, "Failed to parse config file: %s", filePath.c_str());
        return false;
    }
    
    obs_log(LOG_INFO, "Config loaded: %s", configName.c_str());
    return true;
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
