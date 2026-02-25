#ifdef _WIN32

#include "RecoilPatternManager.hpp"
#include "LogitechMacroConverter.hpp"
#include <obs-module.h>
#include <fstream>
#include <sstream>
#include <algorithm>

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

bool RecoilPatternManager::saveToFile(const std::string& filePath)
{
    std::ofstream file(filePath);
    if (!file.is_open()) {
        obs_log(LOG_ERROR, "[RecoilPatternManager] Failed to open file for writing: %s", filePath.c_str());
        return false;
    }
    
    file << "{\n";
    file << "  \"patterns\": {\n";
    
    bool first = true;
    for (const auto& pair : patterns_) {
        if (!first) {
            file << ",\n";
        }
        first = false;
        
        const RecoilPattern& pattern = pair.second;
        file << "    \"" << pattern.weaponName << "\": {\n";
        file << "      \"weaponName\": \"" << pattern.weaponName << "\",\n";
        file << "      \"totalDurationMs\": " << pattern.totalDurationMs << ",\n";
        file << "      \"totalMoveX\": " << pattern.totalMoveX << ",\n";
        file << "      \"totalMoveY\": " << pattern.totalMoveY << ",\n";
        file << "      \"moves\": [\n";
        
        for (size_t i = 0; i < pattern.moves.size(); ++i) {
            const RecoilMove& move = pattern.moves[i];
            file << "        {\"dx\": " << move.dx << ", \"dy\": " << move.dy << ", \"delayMs\": " << move.delayMs << "}";
            if (i < pattern.moves.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        
        file << "      ]\n";
        file << "    }";
    }
    
    file << "\n  }\n";
    file << "}\n";
    
    return true;
}

bool RecoilPatternManager::loadFromFile(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        obs_log(LOG_INFO, "[RecoilPatternManager] Config file not found, starting fresh: %s", filePath.c_str());
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();
    
    patterns_.clear();
    
    size_t pos = content.find("\"patterns\"");
    if (pos == std::string::npos) {
        return false;
    }
    
    while (true) {
        size_t weaponStart = content.find("\"", pos + 11);
        if (weaponStart == std::string::npos) break;
        
        size_t weaponEnd = content.find("\"", weaponStart + 1);
        if (weaponEnd == std::string::npos) break;
        
        std::string weaponName = content.substr(weaponStart + 1, weaponEnd - weaponStart - 1);
        
        RecoilPattern pattern;
        pattern.weaponName = weaponName;
        
        size_t movesStart = content.find("\"moves\"", weaponEnd);
        if (movesStart == std::string::npos) {
            pos = weaponEnd;
            continue;
        }
        
        size_t arrayStart = content.find("[", movesStart);
        size_t arrayEnd = content.find("]", arrayStart);
        
        std::string movesContent = content.substr(arrayStart, arrayEnd - arrayStart);
        
        size_t movePos = 0;
        while (true) {
            size_t dxPos = movesContent.find("\"dx\"", movePos);
            if (dxPos == std::string::npos) break;
            
            size_t colonPos = movesContent.find(":", dxPos);
            size_t commaPos = movesContent.find(",", colonPos);
            if (commaPos == std::string::npos) {
                commaPos = movesContent.find("}", colonPos);
            }
            
            std::string dxStr = movesContent.substr(colonPos + 1, commaPos - colonPos - 1);
            
            size_t dyPos = movesContent.find("\"dy\"", commaPos);
            colonPos = movesContent.find(":", dyPos);
            commaPos = movesContent.find(",", colonPos);
            if (commaPos == std::string::npos) {
                commaPos = movesContent.find("}", colonPos);
            }
            
            std::string dyStr = movesContent.substr(colonPos + 1, commaPos - colonPos - 1);
            
            size_t delayPos = movesContent.find("\"delayMs\"", commaPos);
            colonPos = movesContent.find(":", delayPos);
            size_t endBrace = movesContent.find("}", colonPos);
            
            std::string delayStr = movesContent.substr(colonPos + 1, endBrace - colonPos - 1);
            
            RecoilMove move;
            move.dx = std::stoi(dxStr);
            move.dy = std::stoi(dyStr);
            move.delayMs = std::stoi(delayStr);
            
            pattern.moves.push_back(move);
            pattern.totalMoveX += move.dx;
            pattern.totalMoveY += move.dy;
            pattern.totalDurationMs += move.delayMs;
            
            movePos = endBrace;
        }
        
        if (!pattern.moves.empty()) {
            patterns_[weaponName] = pattern;
        }
        
        pos = arrayEnd;
    }
    
    obs_log(LOG_INFO, "[RecoilPatternManager] Loaded %zu patterns from file", patterns_.size());
    return true;
}

#endif
