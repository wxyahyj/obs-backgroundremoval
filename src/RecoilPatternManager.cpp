#ifdef _WIN32

#include "RecoilPatternManager.hpp"
#include "LogitechMacroConverter.hpp"
#include <obs-module.h>
#include <plugin-support.h>
#include <fstream>

RecoilPatternManager::RecoilPatternManager()
{
    std::string configPath = getConfigFilePath();
    loadFromFile(configPath);
}

RecoilPatternManager& RecoilPatternManager::getInstance()
{
    static RecoilPatternManager instance;
    return instance;
}

bool RecoilPatternManager::importFromLogitechMacro(const std::string& filePath, const std::string& weaponName)
{
    ParsedMacro macro;
    if (!LogitechMacroConverter::parseFile(filePath, macro)) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    RecoilPattern pattern;
    pattern.weaponName = weaponName;
    pattern.totalDurationMs = 0;
    pattern.totalMoveX = 0;
    pattern.totalMoveY = 0;
    
    int currentDelay = 0;
    
    for (const auto& event : macro.events) {
        if (event.type == MacroEvent::MouseMove) {
            RecoilMove move;
            move.dx = event.dx;
            move.dy = event.dy;
            move.delayMs = currentDelay > 0 ? currentDelay : 1;
            pattern.moves.push_back(move);
            
            pattern.totalMoveX += event.dx;
            pattern.totalMoveY += event.dy;
            pattern.totalDurationMs += move.delayMs;
            currentDelay = 0;
        }
        else if (event.type == MacroEvent::Delay) {
            currentDelay += event.delayMs;
        }
    }
    
    if (pattern.moves.empty()) {
        return false;
    }
    
    patterns_[weaponName] = pattern;
    
    saveToFile(getConfigFilePath());
    
    obs_log(LOG_INFO, "[RecoilPatternManager] Imported pattern '%s': %zu moves, total move (%d, %d), duration %dms",
            weaponName.c_str(), pattern.moves.size(), 
            pattern.totalMoveX, pattern.totalMoveY, pattern.totalDurationMs);
    
    return true;
}

bool RecoilPatternManager::importFromString(const std::string& xmlContent, const std::string& weaponName)
{
    ParsedMacro macro;
    if (!LogitechMacroConverter::parseString(xmlContent, macro)) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    RecoilPattern pattern;
    pattern.weaponName = weaponName;
    pattern.totalDurationMs = 0;
    pattern.totalMoveX = 0;
    pattern.totalMoveY = 0;
    
    int currentDelay = 0;
    
    for (const auto& event : macro.events) {
        if (event.type == MacroEvent::MouseMove) {
            RecoilMove move;
            move.dx = event.dx;
            move.dy = event.dy;
            move.delayMs = currentDelay > 0 ? currentDelay : 1;
            pattern.moves.push_back(move);
            
            pattern.totalMoveX += event.dx;
            pattern.totalMoveY += event.dy;
            pattern.totalDurationMs += move.delayMs;
            currentDelay = 0;
        }
        else if (event.type == MacroEvent::Delay) {
            currentDelay += event.delayMs;
        }
    }
    
    if (pattern.moves.empty()) {
        return false;
    }
    
    patterns_[weaponName] = pattern;
    return true;
}

bool RecoilPatternManager::hasPattern(const std::string& weaponName) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return patterns_.find(weaponName) != patterns_.end();
}

const RecoilPattern* RecoilPatternManager::getPattern(const std::string& weaponName) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = patterns_.find(weaponName);
    if (it != patterns_.end()) {
        return &(it->second);
    }
    return nullptr;
}

std::vector<std::string> RecoilPatternManager::getWeaponNames() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(patterns_.size());
    for (const auto& pair : patterns_) {
        names.push_back(pair.first);
    }
    return names;
}

void RecoilPatternManager::removePattern(const std::string& weaponName)
{
    std::lock_guard<std::mutex> lock(mutex_);
    patterns_.erase(weaponName);
    saveToFile(getConfigFilePath());
}

void RecoilPatternManager::clearAllPatterns()
{
    std::lock_guard<std::mutex> lock(mutex_);
    patterns_.clear();
    saveToFile(getConfigFilePath());
}

int RecoilPatternManager::getPatternCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(patterns_.size());
}

std::string RecoilPatternManager::getConfigFilePath()
{
    char* configPath = obs_module_config_path("recoil_patterns.json");
    std::string result(configPath);
    bfree(configPath);
    return result;
}

nlohmann::json RecoilPatternManager::patternToJson(const RecoilPattern& pattern)
{
    nlohmann::json j;
    j["weaponName"] = pattern.weaponName;
    j["totalDurationMs"] = pattern.totalDurationMs;
    j["totalMoveX"] = pattern.totalMoveX;
    j["totalMoveY"] = pattern.totalMoveY;
    
    nlohmann::json movesArray = nlohmann::json::array();
    for (const auto& move : pattern.moves) {
        nlohmann::json moveObj;
        moveObj["dx"] = move.dx;
        moveObj["dy"] = move.dy;
        moveObj["delayMs"] = move.delayMs;
        movesArray.push_back(moveObj);
    }
    j["moves"] = movesArray;
    
    return j;
}

RecoilPattern RecoilPatternManager::jsonToPattern(const nlohmann::json& j)
{
    RecoilPattern pattern;
    
    if (j.contains("weaponName")) {
        pattern.weaponName = j["weaponName"].get<std::string>();
    }
    if (j.contains("totalDurationMs")) {
        pattern.totalDurationMs = j["totalDurationMs"].get<int>();
    }
    if (j.contains("totalMoveX")) {
        pattern.totalMoveX = j["totalMoveX"].get<int>();
    }
    if (j.contains("totalMoveY")) {
        pattern.totalMoveY = j["totalMoveY"].get<int>();
    }
    
    if (j.contains("moves") && j["moves"].is_array()) {
        for (const auto& moveObj : j["moves"]) {
            RecoilMove move;
            if (moveObj.contains("dx")) move.dx = moveObj["dx"].get<int>();
            if (moveObj.contains("dy")) move.dy = moveObj["dy"].get<int>();
            if (moveObj.contains("delayMs")) move.delayMs = moveObj["delayMs"].get<int>();
            pattern.moves.push_back(move);
        }
    }
    
    return pattern;
}

bool RecoilPatternManager::saveToFile(const std::string& filePath)
{
    try {
        nlohmann::json root;
        nlohmann::json patternsJson = nlohmann::json::object();
        
        for (const auto& pair : patterns_) {
            patternsJson[pair.first] = patternToJson(pair.second);
        }
        
        root["patterns"] = patternsJson;
        
        std::ofstream file(filePath);
        if (!file.is_open()) {
            obs_log(LOG_ERROR, "[RecoilPatternManager] Failed to open file for writing: %s", filePath.c_str());
            return false;
        }
        
        file << root.dump(2);
        file.close();
        
        return true;
    } catch (const std::exception& e) {
        obs_log(LOG_ERROR, "[RecoilPatternManager] Failed to save file: %s", e.what());
        return false;
    }
}

bool RecoilPatternManager::loadFromFile(const std::string& filePath)
{
    try {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            obs_log(LOG_INFO, "[RecoilPatternManager] Config file not found, starting fresh: %s", filePath.c_str());
            return false;
        }
        
        nlohmann::json root = nlohmann::json::parse(file);
        file.close();
        
        patterns_.clear();
        
        if (root.contains("patterns") && root["patterns"].is_object()) {
            for (auto& [weaponName, patternJson] : root["patterns"].items()) {
                RecoilPattern pattern = jsonToPattern(patternJson);
                if (!pattern.moves.empty()) {
                    patterns_[weaponName] = pattern;
                }
            }
        }
        
        obs_log(LOG_INFO, "[RecoilPatternManager] Loaded %zu patterns from file", patterns_.size());
        return true;
    } catch (const nlohmann::json::exception& e) {
        obs_log(LOG_ERROR, "[RecoilPatternManager] JSON parsing error: %s", e.what());
        return false;
    }
}

#endif
