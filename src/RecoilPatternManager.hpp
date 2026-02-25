#ifndef RECOIL_PATTERN_MANAGER_HPP
#define RECOIL_PATTERN_MANAGER_HPP

#ifdef _WIN32

#include <string>
#include <vector>
#include <mutex>
#include <map>

struct RecoilMove {
    int dx;
    int dy;
    int delayMs;
    
    RecoilMove() : dx(0), dy(0), delayMs(0) {}
    RecoilMove(int x, int y, int delay) : dx(x), dy(y), delayMs(delay) {}
};

struct RecoilPattern {
    std::string weaponName;
    std::vector<RecoilMove> moves;
    int totalDurationMs;
    int totalMoveX;
    int totalMoveY;
    
    RecoilPattern() : totalDurationMs(0), totalMoveX(0), totalMoveY(0) {}
};

class RecoilPatternManager {
public:
    static RecoilPatternManager& getInstance();
    
    bool importFromLogitechMacro(const std::string& filePath, const std::string& weaponName);
    
    bool importFromString(const std::string& xmlContent, const std::string& weaponName);
    
    bool hasPattern(const std::string& weaponName) const;
    
    const RecoilPattern* getPattern(const std::string& weaponName) const;
    
    std::vector<std::string> getWeaponNames() const;
    
    void removePattern(const std::string& weaponName);
    
    void clearAllPatterns();
    
    int getPatternCount() const;
    
    bool saveToFile(const std::string& filePath);
    
    bool loadFromFile(const std::string& filePath);

private:
    RecoilPatternManager();
    ~RecoilPatternManager() = default;
    RecoilPatternManager(const RecoilPatternManager&) = delete;
    RecoilPatternManager& operator=(const RecoilPatternManager&) = delete;
    
    mutable std::mutex mutex_;
    std::map<std::string, RecoilPattern> patterns_;
    
    std::string getConfigFilePath();
};

#endif

#endif
