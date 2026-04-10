#ifndef WEBSERVER_HPP
#define WEBSERVER_HPP

#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include "MouseControllerInterface.hpp"

struct mg_mgr;
struct mg_connection;
struct mg_http_message;

using ConfigCallback = std::function<void(const MouseControllerConfig& config)>;
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
    WebServer(int port = 8080, const std::string& host = "127.0.0.1");
    ~WebServer();

    void start();
    void stop();
    bool isRunning() const;

    void setConfigCallback(ConfigCallback callback);
    void setStatusProvider(StatusProvider provider);
    void updateStatus(const WebServerStatus& status);

    int getPort() const { return port_; }
    std::string getUrl() const;

private:
    void serverLoop();
    static void eventHandler(struct mg_connection* c, int ev, void* ev_data, void* fn_data);

    void handleHttpRequest(struct mg_connection* c, struct mg_http_message* hm);
    void handleGetConfig(struct mg_connection* c);
    void handlePostConfig(struct mg_connection* c, struct mg_http_message* hm);
    void handleGetStatus(struct mg_connection* c);
    void handleGetConfigs(struct mg_connection* c);
    void handleSaveConfig(struct mg_connection* c, struct mg_http_message* hm);
    void handleLoadConfig(struct mg_connection* c, struct mg_http_message* hm);
    void handleGetAlgorithms(struct mg_connection* c);
    void handleSetAlgorithm(struct mg_connection* c, struct mg_http_message* hm);
    void handleWebSocket(struct mg_connection* c, int ev, void* ev_data);
    void serveStaticFile(struct mg_connection* c, struct mg_http_message* hm);

    void sendJsonResponse(struct mg_connection* c, int status, const std::string& json);
    void sendErrorResponse(struct mg_connection* c, int status, const std::string& message);

    std::string configToJson(const MouseControllerConfig& config);
    bool jsonToConfig(const std::string& json, MouseControllerConfig& config);

    mg_mgr* mgr_;
    int port_;
    std::string host_;
    std::atomic<bool> running_;
    std::thread serverThread_;
    std::mutex mutex_;

    ConfigCallback configCallback_;
    StatusProvider statusProvider_;
    WebServerStatus currentStatus_;
    MouseControllerConfig currentConfig_;

    std::vector<struct mg_connection*> wsConnections_;
};

#endif
