#ifndef CONFIG_MANAGER_HPP
#define CONFIG_MANAGER_HPP

#ifdef _WIN32

#include <string>
#include <vector>
#include <mutex>
#include <nlohmann/json.hpp>
#include "MouseControllerInterface.hpp"

struct ExtendedMouseControllerConfig : public MouseControllerConfig {
    std::string configName;
    
    ExtendedMouseControllerConfig();
    static ExtendedMouseControllerConfig getDefault();
};

class ConfigManager {
public:
    static ConfigManager& getInstance();
    
    void setConfigsDirectory(const std::string& dir);
    bool saveConfig(const ExtendedMouseControllerConfig& config);
    bool loadConfig(const std::string& configName, ExtendedMouseControllerConfig& config);
    bool deleteConfig(const std::string& configName);
    std::vector<std::string> listConfigs();
    bool configExists(const std::string& configName);
    std::string getConfigsDirectory();
    
private:
    ConfigManager();
    ~ConfigManager();
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    std::mutex mutex;
    std::string configsDir;
    
    bool ensureConfigsDirectory();
    std::string getConfigFilePath(const std::string& configName);
    
    nlohmann::json configToJson(const ExtendedMouseControllerConfig& config);
    bool jsonToConfig(const nlohmann::json& j, ExtendedMouseControllerConfig& config);
};

#endif

#endif
