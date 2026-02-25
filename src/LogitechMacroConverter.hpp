#ifndef LOGITECH_MACRO_CONVERTER_HPP
#define LOGITECH_MACRO_CONVERTER_HPP

#ifdef _WIN32

#include <string>
#include <vector>

struct MacroEvent {
    enum Type { 
        MouseMove, 
        MouseDown, 
        MouseUp, 
        Delay, 
        KeyDown, 
        KeyUp,
        MouseWheel
    };
    
    Type type;
    int dx, dy;
    int button;
    int delayMs;
    int keyCode;
    int wheelDelta;
    
    MacroEvent();
};

struct ParsedMacro {
    std::string name;
    std::vector<MacroEvent> events;
    
    int totalMoveX;
    int totalMoveY;
    int totalDurationMs;
    int clickCount;
    int mouseMoveCount;
    float avgMoveDistance;
    float maxMoveDistance;
    
    ParsedMacro();
    void calculateStatistics();
    void clear();
};

class LogitechMacroConverter {
public:
    static bool parseFile(const std::string& filePath, ParsedMacro& result);
    
    static bool parseString(const std::string& xmlContent, ParsedMacro& result);
    
    static void generateConfigSuggestions(const ParsedMacro& macro, 
                                          float& suggestedP, 
                                          float& suggestedD,
                                          float& suggestedSmoothing);
    
    static std::vector<std::string> findLogitechMacroFiles();
    
    static bool isLogitechMacroFile(const std::string& filePath);

private:
    static std::string trim(const std::string& str);
    static std::string extractAttribute(const std::string& element, const std::string& attrName);
    static bool parseEventElement(const std::string& eventStr, MacroEvent& event);
    static int parseMouseButton(const std::string& buttonStr);
    static std::string toLower(const std::string& str);
};

#endif

#endif
