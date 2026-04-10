#ifndef WEBSERVER_HPP
#define WEBSERVER_HPP

#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <algorithm>
#include "MouseControllerInterface.hpp"

struct mg_mgr;
struct mg_connection;
struct mg_http_message;
struct mg_ws_message;

using ConfigCallback = std::function<void(const MouseControllerConfig&)>;
using StatusProvider = std::function<MouseControllerConfig()>;

struct WebServerStatus {
    bool isRunning = false;
    float currentErrorX = 0.0f;
    float currentErrorY = 0.0f;
    float currentOutputX = 0.0f;
    float currentOutputY = 0.0f;
    float targetX = 0.0f;
    float targetY = 0.0f;
    int fps = 0;
    int controllerFrequency = 0;
};

class WebServer {
public:
    WebServer(int port = 8080);
    ~WebServer();

    void start();
    void stop();
    bool isRunning() const { return running_; }

    void setConfigCallback(ConfigCallback callback);
    void setStatusProvider(StatusProvider provider);
    void updateStatus(const WebServerStatus& status);

    int getPort() const { return port_; }
    std::string getUrl() const;

    void eventHandler(struct mg_connection* c, int ev, void* ev_data);

private:
    
    void handleGetConfig(struct mg_connection* c);
    void handlePostConfig(struct mg_connection* c, struct mg_http_message* hm);
    void handleGetStatus(struct mg_connection* c);
    void handleGetAlgorithms(struct mg_connection* c);
    void handleWebSocket(struct mg_connection* c, struct mg_ws_message* wm);
    void handleStaticFile(struct mg_connection* c, const char* filename);

    std::string statusToJson(const WebServerStatus& s);
    std::string configToJson(const MouseControllerConfig& config);

    void* mgr_;
    int port_;
    std::atomic<bool> running_;
    std::thread serverThread_;
    std::mutex mutex_;

    ConfigCallback configCallback_;
    StatusProvider statusProvider_;
    WebServerStatus status_;

    std::vector<struct mg_connection*> wsConnections_;
};

#endif
