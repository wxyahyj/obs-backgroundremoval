#ifdef _WIN32

#include "LogitechMacroConverter.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <shlobj.h>
#include <windows.h>

MacroEvent::MacroEvent()
    : type(MouseMove)
    , dx(0), dy(0)
    , button(0)
    , delayMs(0)
    , keyCode(0)
    , wheelDelta(0)
{
}

ParsedMacro::ParsedMacro()
    : totalMoveX(0)
    , totalMoveY(0)
    , totalDurationMs(0)
    , clickCount(0)
    , mouseMoveCount(0)
    , avgMoveDistance(0.0f)
    , maxMoveDistance(0.0f)
{
}

void ParsedMacro::calculateStatistics()
{
    totalMoveX = 0;
    totalMoveY = 0;
    totalDurationMs = 0;
    clickCount = 0;
    mouseMoveCount = 0;
    maxMoveDistance = 0.0f;
    
    float totalDistance = 0.0f;
    
    for (const auto& event : events) {
        switch (event.type) {
            case MacroEvent::MouseMove:
                totalMoveX += event.dx;
                totalMoveY += event.dy;
                mouseMoveCount++;
                {
                    float dist = std::sqrt(static_cast<float>(event.dx * event.dx + event.dy * event.dy));
                    totalDistance += dist;
                    if (dist > maxMoveDistance) {
                        maxMoveDistance = dist;
                    }
                }
                break;
            case MacroEvent::MouseDown:
                clickCount++;
                break;
            case MacroEvent::Delay:
                totalDurationMs += event.delayMs;
                break;
            default:
                break;
        }
    }
    
    if (mouseMoveCount > 0) {
        avgMoveDistance = totalDistance / mouseMoveCount;
    } else {
        avgMoveDistance = 0.0f;
    }
}

void ParsedMacro::clear()
{
    name.clear();
    events.clear();
    totalMoveX = 0;
    totalMoveY = 0;
    totalDurationMs = 0;
    clickCount = 0;
    mouseMoveCount = 0;
    avgMoveDistance = 0.0f;
    maxMoveDistance = 0.0f;
}

std::string LogitechMacroConverter::trim(const std::string& str)
{
    size_t start = 0;
    while (start < str.length() && (str[start] == ' ' || str[start] == '\t' || 
           str[start] == '\n' || str[start] == '\r')) {
        start++;
    }
    
    if (start >= str.length()) {
        return "";
    }
    
    size_t end = str.length() - 1;
    while (end > start && (str[end] == ' ' || str[end] == '\t' || 
           str[end] == '\n' || str[end] == '\r')) {
        end--;
    }
    
    return str.substr(start, end - start + 1);
}

std::string LogitechMacroConverter::toLower(const std::string& str)
{
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string LogitechMacroConverter::extractAttribute(const std::string& element, const std::string& attrName)
{
    std::string searchStr = attrName + "=\"";
    size_t pos = element.find(searchStr);
    if (pos == std::string::npos) {
        searchStr = attrName + "='";
        pos = element.find(searchStr);
    }
    
    if (pos == std::string::npos) {
        return "";
    }
    
    pos += searchStr.length();
    char quoteChar = element[pos - 1];
    
    size_t endPos = element.find(quoteChar, pos);
    if (endPos == std::string::npos) {
        return "";
    }
    
    return element.substr(pos, endPos - pos);
}

int LogitechMacroConverter::parseMouseButton(const std::string& buttonStr)
{
    if (buttonStr.empty()) {
        return 0;
    }
    
    std::string lower = toLower(buttonStr);
    
    if (lower == "left" || lower == "1" || lower == "lmb") {
        return 1;
    } else if (lower == "right" || lower == "2" || lower == "rmb") {
        return 2;
    } else if (lower == "middle" || lower == "3" || lower == "mmb") {
        return 3;
    } else if (lower == "x1" || lower == "4" || lower == "back") {
        return 4;
    } else if (lower == "x2" || lower == "5" || lower == "forward") {
        return 5;
    }
    
    try {
        return std::stoi(buttonStr);
    } catch (...) {
        return 0;
    }
}

bool LogitechMacroConverter::parseEventElement(const std::string& eventStr, MacroEvent& event)
{
    std::string lowerEvent = toLower(eventStr);
    
    if (lowerEvent.find("mousemove") != std::string::npos || 
        lowerEvent.find("mouse_move") != std::string::npos ||
        lowerEvent.find("move") != std::string::npos) {
        event.type = MacroEvent::MouseMove;
        event.dx = 0;
        event.dy = 0;
        
        std::string dxStr = extractAttribute(eventStr, "dx");
        std::string dyStr = extractAttribute(eventStr, "dy");
        std::string xStr = extractAttribute(eventStr, "x");
        std::string yStr = extractAttribute(eventStr, "y");
        
        if (!dxStr.empty()) {
            try { event.dx = std::stoi(dxStr); } catch (...) {}
        } else if (!xStr.empty()) {
            try { event.dx = std::stoi(xStr); } catch (...) {}
        }
        
        if (!dyStr.empty()) {
            try { event.dy = std::stoi(dyStr); } catch (...) {}
        } else if (!yStr.empty()) {
            try { event.dy = std::stoi(yStr); } catch (...) {}
        }
        
        return true;
    }
    else if (lowerEvent.find("mousedown") != std::string::npos || 
             lowerEvent.find("mouse_down") != std::string::npos ||
             lowerEvent.find("buttondown") != std::string::npos ||
             lowerEvent.find("button_down") != std::string::npos) {
        event.type = MacroEvent::MouseDown;
        std::string buttonStr = extractAttribute(eventStr, "button");
        if (buttonStr.empty()) {
            buttonStr = extractAttribute(eventStr, "btn");
        }
        event.button = parseMouseButton(buttonStr);
        return true;
    }
    else if (lowerEvent.find("mouseup") != std::string::npos || 
             lowerEvent.find("mouse_up") != std::string::npos ||
             lowerEvent.find("buttonup") != std::string::npos ||
             lowerEvent.find("button_up") != std::string::npos) {
        event.type = MacroEvent::MouseUp;
        std::string buttonStr = extractAttribute(eventStr, "button");
        if (buttonStr.empty()) {
            buttonStr = extractAttribute(eventStr, "btn");
        }
        event.button = parseMouseButton(buttonStr);
        return true;
    }
    else if (lowerEvent.find("delay") != std::string::npos || 
             lowerEvent.find("wait") != std::string::npos ||
             lowerEvent.find("sleep") != std::string::npos) {
        event.type = MacroEvent::Delay;
        event.delayMs = 0;
        
        std::string timeStr = extractAttribute(eventStr, "time");
        std::string msStr = extractAttribute(eventStr, "ms");
        std::string durationStr = extractAttribute(eventStr, "duration");
        
        if (!timeStr.empty()) {
            try { event.delayMs = std::stoi(timeStr); } catch (...) {}
        } else if (!msStr.empty()) {
            try { event.delayMs = std::stoi(msStr); } catch (...) {}
        } else if (!durationStr.empty()) {
            try { event.delayMs = std::stoi(durationStr); } catch (...) {}
        }
        
        return true;
    }
    else if (lowerEvent.find("keydown") != std::string::npos || 
             lowerEvent.find("key_down") != std::string::npos) {
        event.type = MacroEvent::KeyDown;
        std::string keyStr = extractAttribute(eventStr, "key");
        std::string codeStr = extractAttribute(eventStr, "keycode");
        
        if (!codeStr.empty()) {
            try { event.keyCode = std::stoi(codeStr); } catch (...) {}
        } else if (!keyStr.empty()) {
            event.keyCode = VkKeyScanA(keyStr[0]) & 0xFF;
        }
        return true;
    }
    else if (lowerEvent.find("keyup") != std::string::npos || 
             lowerEvent.find("key_up") != std::string::npos) {
        event.type = MacroEvent::KeyUp;
        std::string keyStr = extractAttribute(eventStr, "key");
        std::string codeStr = extractAttribute(eventStr, "keycode");
        
        if (!codeStr.empty()) {
            try { event.keyCode = std::stoi(codeStr); } catch (...) {}
        } else if (!keyStr.empty()) {
            event.keyCode = VkKeyScanA(keyStr[0]) & 0xFF;
        }
        return true;
    }
    else if (lowerEvent.find("mousewheel") != std::string::npos || 
             lowerEvent.find("wheel") != std::string::npos ||
             lowerEvent.find("scroll") != std::string::npos) {
        event.type = MacroEvent::MouseWheel;
        std::string deltaStr = extractAttribute(eventStr, "delta");
        std::string amountStr = extractAttribute(eventStr, "amount");
        
        if (!deltaStr.empty()) {
            try { event.wheelDelta = std::stoi(deltaStr); } catch (...) {}
        } else if (!amountStr.empty()) {
            try { event.wheelDelta = std::stoi(amountStr); } catch (...) {}
        }
        return true;
    }
    
    return false;
}

bool LogitechMacroConverter::parseFile(const std::string& filePath, ParsedMacro& result)
{
    result.clear();
    
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    std::string content = buffer.str();
    
    if (filePath.size() > 4) {
        std::string ext = toLower(filePath.substr(filePath.size() - 4));
        if (ext == ".lua") {
            return parseLuaString(content, result);
        }
    }
    
    return parseString(content, result);
}

bool LogitechMacroConverter::parseLuaString(const std::string& luaContent, ParsedMacro& result)
{
    result.clear();
    
    std::string content = luaContent;
    
    size_t tableStart = content.find("{{");
    if (tableStart != std::string::npos) {
        size_t tableEnd = content.find("}}", tableStart);
        if (tableEnd == std::string::npos) {
            tableEnd = content.find("}", content.find("}", tableStart + 2) + 1);
        }
        
        if (tableEnd != std::string::npos) {
            std::string tableContent = content.substr(tableStart, tableEnd - tableStart + 2);
            
            size_t pos = 0;
            while ((pos = tableContent.find("{", pos)) != std::string::npos) {
                size_t endPos = tableContent.find("}", pos);
                if (endPos == std::string::npos) {
                    break;
                }
                
                std::string entry = tableContent.substr(pos + 1, endPos - pos - 1);
                
                int x = 0, y = 0, d = 0;
                
                size_t xPos = entry.find("x=");
                if (xPos != std::string::npos) {
                    size_t xEnd = entry.find(",", xPos);
                    if (xEnd == std::string::npos) xEnd = entry.length();
                    std::string xStr = trim(entry.substr(xPos + 2, xEnd - xPos - 2));
                    try { x = std::stoi(xStr); } catch (...) {}
                }
                
                size_t yPos = entry.find("y=");
                if (yPos != std::string::npos) {
                    size_t yEnd = entry.find(",", yPos);
                    if (yEnd == std::string::npos) yEnd = entry.length();
                    std::string yStr = trim(entry.substr(yPos + 2, yEnd - yPos - 2));
                    try { y = std::stoi(yStr); } catch (...) {}
                }
                
                size_t dPos = entry.find("d=");
                if (dPos != std::string::npos) {
                    size_t dEnd = entry.find(",", dPos);
                    if (dEnd == std::string::npos) dEnd = entry.length();
                    std::string dStr = trim(entry.substr(dPos + 2, dEnd - dPos - 2));
                    try { d = std::stoi(dStr); } catch (...) {}
                }
                
                if (x != 0 || y != 0) {
                    MacroEvent moveEvent;
                    moveEvent.type = MacroEvent::MouseMove;
                    moveEvent.dx = x;
                    moveEvent.dy = y;
                    result.events.push_back(moveEvent);
                }
                
                if (d > 0) {
                    MacroEvent delayEvent;
                    delayEvent.type = MacroEvent::Delay;
                    delayEvent.delayMs = d;
                    result.events.push_back(delayEvent);
                }
                
                pos = endPos + 1;
            }
            
            result.calculateStatistics();
            return !result.events.empty();
        }
    }
    
    std::istringstream stream(luaContent);
    std::string line;
    
    while (std::getline(stream, line)) {
        std::string trimmed = trim(line);
        
        if (trimmed.empty() || trimmed[0] == '-' || trimmed[0] == '/' || trimmed[0] == '#') {
            continue;
        }
        
        std::string lower = toLower(trimmed);
        
        if (lower.find("movemouserelative") != std::string::npos ||
            lower.find("move_mouse_relative") != std::string::npos ||
            lower.find("movemouse") != std::string::npos) {
            
            int dx = 0, dy = 0;
            
            size_t parenStart = trimmed.find('(');
            size_t parenEnd = trimmed.find(')');
            
            if (parenStart != std::string::npos && parenEnd != std::string::npos && parenEnd > parenStart) {
                std::string args = trimmed.substr(parenStart + 1, parenEnd - parenStart - 1);
                
                size_t comma = args.find(',');
                if (comma != std::string::npos) {
                    std::string dxStr = trim(args.substr(0, comma));
                    std::string dyStr = trim(args.substr(comma + 1));
                    
                    try { dx = std::stoi(dxStr); } catch (...) {}
                    try { dy = std::stoi(dyStr); } catch (...) {}
                }
            }
            
            MacroEvent event;
            event.type = MacroEvent::MouseMove;
            event.dx = dx;
            event.dy = dy;
            result.events.push_back(event);
        }
        else if (lower.find("sleep") != std::string::npos ||
                 lower.find("wait") != std::string::npos ||
                 lower.find("delay") != std::string::npos) {
            
            int delayMs = 0;
            
            size_t parenStart = trimmed.find('(');
            size_t parenEnd = trimmed.find(')');
            
            if (parenStart != std::string::npos && parenEnd != std::string::npos && parenEnd > parenStart) {
                std::string arg = trim(trimmed.substr(parenStart + 1, parenEnd - parenStart - 1));
                try { delayMs = std::stoi(arg); } catch (...) {}
            }
            
            if (delayMs > 0) {
                MacroEvent event;
                event.type = MacroEvent::Delay;
                event.delayMs = delayMs;
                result.events.push_back(event);
            }
        }
        else if (lower.find("presskey") != std::string::npos ||
                 lower.find("press_key") != std::string::npos ||
                 lower.find("keydown") != std::string::npos ||
                 lower.find("key_down") != std::string::npos) {
            
            MacroEvent event;
            event.type = MacroEvent::KeyDown;
            result.events.push_back(event);
        }
        else if (lower.find("releasekey") != std::string::npos ||
                 lower.find("release_key") != std::string::npos ||
                 lower.find("keyup") != std::string::npos ||
                 lower.find("key_up") != std::string::npos) {
            
            MacroEvent event;
            event.type = MacroEvent::KeyUp;
            result.events.push_back(event);
        }
        else if (lower.find("pressMouseButton") != std::string::npos ||
                 lower.find("press_mouse_button") != std::string::npos ||
                 lower.find("mousedown") != std::string::npos ||
                 lower.find("mouse_down") != std::string::npos) {
            
            MacroEvent event;
            event.type = MacroEvent::MouseDown;
            event.button = 1;
            
            size_t parenStart = trimmed.find('(');
            size_t parenEnd = trimmed.find(')');
            
            if (parenStart != std::string::npos && parenEnd != std::string::npos && parenEnd > parenStart) {
                std::string arg = trim(trimmed.substr(parenStart + 1, parenEnd - parenStart - 1));
                try { event.button = std::stoi(arg); } catch (...) {}
            }
            
            result.events.push_back(event);
        }
        else if (lower.find("releaseMouseButton") != std::string::npos ||
                 lower.find("release_mouse_button") != std::string::npos ||
                 lower.find("mouseup") != std::string::npos ||
                 lower.find("mouse_up") != std::string::npos) {
            
            MacroEvent event;
            event.type = MacroEvent::MouseUp;
            event.button = 1;
            
            size_t parenStart = trimmed.find('(');
            size_t parenEnd = trimmed.find(')');
            
            if (parenStart != std::string::npos && parenEnd != std::string::npos && parenEnd > parenStart) {
                std::string arg = trim(trimmed.substr(parenStart + 1, parenEnd - parenStart - 1));
                try { event.button = std::stoi(arg); } catch (...) {}
            }
            
            result.events.push_back(event);
        }
    }
    
    result.calculateStatistics();
    
    return !result.events.empty();
}

bool LogitechMacroConverter::parseString(const std::string& xmlContent, ParsedMacro& result)
{
    result.clear();
    
    std::string content = xmlContent;
    
    size_t macroStart = content.find("<macro");
    if (macroStart == std::string::npos) {
        macroStart = content.find("<Macro");
    }
    
    if (macroStart == std::string::npos) {
        size_t nameStart = content.find("name=\"");
        if (nameStart != std::string::npos) {
            size_t nameEnd = content.find("\"", nameStart + 6);
            if (nameEnd != std::string::npos) {
                result.name = content.substr(nameStart + 6, nameEnd - nameStart - 6);
            }
        }
    } else {
        size_t macroEnd = content.find(">", macroStart);
        if (macroEnd != std::string::npos) {
            std::string macroTag = content.substr(macroStart, macroEnd - macroStart);
            result.name = extractAttribute(macroTag, "name");
        }
    }
    
    size_t pos = 0;
    while ((pos = content.find("<event", pos)) != std::string::npos) {
        size_t eventEnd = content.find("/>", pos);
        if (eventEnd == std::string::npos) {
            eventEnd = content.find(">", pos);
            if (eventEnd == std::string::npos) {
                break;
            }
            
            size_t closeTag = content.find("</event>", eventEnd);
            if (closeTag != std::string::npos) {
                eventEnd = closeTag + 8;
            }
        } else {
            eventEnd += 2;
        }
        
        std::string eventStr = content.substr(pos, eventEnd - pos);
        
        MacroEvent event;
        if (parseEventElement(eventStr, event)) {
            result.events.push_back(event);
        }
        
        pos = eventEnd;
    }
    
    pos = 0;
    while ((pos = content.find("<action", pos)) != std::string::npos) {
        size_t actionEnd = content.find("/>", pos);
        if (actionEnd == std::string::npos) {
            actionEnd = content.find(">", pos);
            if (actionEnd == std::string::npos) {
                break;
            }
            actionEnd++;
        } else {
            actionEnd += 2;
        }
        
        std::string actionStr = content.substr(pos, actionEnd - pos);
        
        MacroEvent event;
        if (parseEventElement(actionStr, event)) {
            result.events.push_back(event);
        }
        
        pos = actionEnd;
    }
    
    pos = 0;
    while ((pos = content.find("<step", pos)) != std::string::npos) {
        size_t stepEnd = content.find("/>", pos);
        if (stepEnd == std::string::npos) {
            stepEnd = content.find(">", pos);
            if (stepEnd == std::string::npos) {
                break;
            }
            stepEnd++;
        } else {
            stepEnd += 2;
        }
        
        std::string stepStr = content.substr(pos, stepEnd - pos);
        
        MacroEvent event;
        if (parseEventElement(stepStr, event)) {
            result.events.push_back(event);
        }
        
        pos = stepEnd;
    }
    
    result.calculateStatistics();
    
    return !result.events.empty();
}

void LogitechMacroConverter::generateConfigSuggestions(const ParsedMacro& macro, 
                                                        float& suggestedP, 
                                                        float& suggestedD,
                                                        float& suggestedSmoothing)
{
    suggestedP = 0.5f;
    suggestedD = 0.1f;
    suggestedSmoothing = 0.3f;
    
    if (macro.events.empty()) {
        return;
    }
    
    float avgMovePerEvent = 0.0f;
    int moveEventCount = 0;
    int totalMoveDistance = 0;
    
    for (const auto& event : macro.events) {
        if (event.type == MacroEvent::MouseMove) {
            int dist = static_cast<int>(std::sqrt(
                static_cast<float>(event.dx * event.dx + event.dy * event.dy)));
            totalMoveDistance += dist;
            moveEventCount++;
        }
    }
    
    if (moveEventCount > 0) {
        avgMovePerEvent = static_cast<float>(totalMoveDistance) / moveEventCount;
    }
    
    float avgDelay = 0.0f;
    int delayCount = 0;
    int totalDelay = 0;
    
    for (const auto& event : macro.events) {
        if (event.type == MacroEvent::Delay) {
            totalDelay += event.delayMs;
            delayCount++;
        }
    }
    
    if (delayCount > 0) {
        avgDelay = static_cast<float>(totalDelay) / delayCount;
    }
    
    if (avgMovePerEvent > 50.0f) {
        suggestedP = 0.3f;
        suggestedSmoothing = 0.5f;
    } else if (avgMovePerEvent > 20.0f) {
        suggestedP = 0.4f;
        suggestedSmoothing = 0.4f;
    } else if (avgMovePerEvent > 5.0f) {
        suggestedP = 0.5f;
        suggestedSmoothing = 0.3f;
    } else {
        suggestedP = 0.6f;
        suggestedSmoothing = 0.2f;
    }
    
    if (avgDelay > 0.0f) {
        if (avgDelay < 10.0f) {
            suggestedD = 0.05f;
        } else if (avgDelay < 30.0f) {
            suggestedD = 0.1f;
        } else if (avgDelay < 50.0f) {
            suggestedD = 0.15f;
        } else {
            suggestedD = 0.2f;
        }
    }
    
    if (macro.clickCount > 0 && macro.mouseMoveCount > 0) {
        float clickToMoveRatio = static_cast<float>(macro.clickCount) / macro.mouseMoveCount;
        if (clickToMoveRatio > 0.5f) {
            suggestedSmoothing *= 0.8f;
        }
    }
}

std::vector<std::string> LogitechMacroConverter::findLogitechMacroFiles()
{
    std::vector<std::string> macroFiles;
    
    char appDataPath[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_LOCAL_APPDATA, NULL, 0, appDataPath))) {
        std::vector<std::string> searchPaths;
        
        searchPaths.push_back(std::string(appDataPath) + "\\Logitech\\Logitech Gaming Software\\profiles");
        searchPaths.push_back(std::string(appDataPath) + "\\LGHUB\\settings");
        searchPaths.push_back(std::string(appDataPath) + "\\Logitech\\G HUB\\profiles");
        
        WIN32_FIND_DATAA findData;
        
        for (const auto& searchPath : searchPaths) {
            std::string searchPattern = searchPath + "\\*.xml";
            
            HANDLE hFind = FindFirstFileA(searchPattern.c_str(), &findData);
            if (hFind != INVALID_HANDLE_VALUE) {
                do {
                    std::string filePath = searchPath + "\\" + findData.cFileName;
                    if (isLogitechMacroFile(filePath)) {
                        macroFiles.push_back(filePath);
                    }
                } while (FindNextFileA(hFind, &findData));
                FindClose(hFind);
            }
        }
    }
    
    return macroFiles;
}

bool LogitechMacroConverter::isLogitechMacroFile(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }
    
    std::string content;
    std::string line;
    int lineCount = 0;
    
    while (std::getline(file, line) && lineCount < 50) {
        content += line + " ";
        lineCount++;
    }
    file.close();
    
    std::string lower = toLower(content);
    
    if (lower.find("<macro") != std::string::npos ||
        lower.find("<event") != std::string::npos ||
        lower.find("<action") != std::string::npos ||
        (lower.find("mousedown") != std::string::npos && 
         lower.find("mousemove") != std::string::npos)) {
        return true;
    }
    
    return false;
}

#endif
