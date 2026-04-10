#include "WebServer.hpp"
#include "mongoose.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <regex>

static WebServer* g_webServer = nullptr;

static void mongooseEventHandler(struct mg_connection* c, int ev, void* ev_data)
{
    if (g_webServer) {
        g_webServer->eventHandler(c, ev, ev_data);
    }
}

// JSON解析辅助函数
static std::string jsonGetString(const std::string& json, const std::string& key, const std::string& defaultVal)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str();
    }
    return defaultVal;
}

static float jsonGetFloat(const std::string& json, const std::string& key, float defaultVal)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stof(match[1].str());
    }
    return defaultVal;
}

static int jsonGetInt(const std::string& json, const std::string& key, int defaultVal)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return defaultVal;
}

static bool jsonGetBool(const std::string& json, const std::string& key, bool defaultVal)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return match[1].str() == "true";
    }
    return defaultVal;
}

static std::string extractJsonObject(const std::string& json, const std::string& key)
{
    std::regex pattern("\"" + key + "\"\\s*:\\s*\\{([^}]*)\\}");
    std::smatch match;
    if (std::regex_search(json, match, pattern)) {
        return "{" + match[1].str() + "}";
    }
    return "{}";
}

WebServer::WebServer(int port)
    : port_(port)
    , running_(false)
    , mgr_(nullptr)
{
}

WebServer::~WebServer()
{
    stop();
}

void WebServer::start()
{
    if (running_) return;
    
    running_ = true;
    g_webServer = this;
    
    serverThread_ = std::thread([this]() {
        struct mg_mgr mgr;
        mg_mgr_init(&mgr);
        mgr_ = &mgr;
        
        std::string listenAddr = "http://0.0.0.0:" + std::to_string(port_);
        
        struct mg_connection* c = mg_http_listen(&mgr, listenAddr.c_str(), mongooseEventHandler, nullptr);
        
        if (c == nullptr) {
            std::cerr << "[WebServer] 无法启动服务器在端口 " << port_ << std::endl;
            running_ = false;
            return;
        }
        
        std::cout << "[WebServer] 服务器启动成功，访问地址: http://127.0.0.1:" << port_ << std::endl;
        
        while (running_) {
            mg_mgr_poll(&mgr, 50);
        }
        
        mg_mgr_free(&mgr);
        mgr_ = nullptr;
    });
}

void WebServer::stop()
{
    if (!running_) return;
    
    running_ = false;
    g_webServer = nullptr;
    
    if (serverThread_.joinable()) {
        serverThread_.join();
    }
}

void WebServer::setConfigCallback(ConfigCallback callback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    configCallback_ = callback;
}

void WebServer::setStatusProvider(StatusProvider provider)
{
    std::lock_guard<std::mutex> lock(mutex_);
    statusProvider_ = provider;
}

void WebServer::updateStatus(const WebServerStatus& status)
{
    std::lock_guard<std::mutex> lock(mutex_);
    status_ = status;
    
    // 广播状态到所有WebSocket连接
    std::string json = statusToJson(status_);
    for (auto* conn : wsConnections_) {
        mg_ws_send(conn, json.c_str(), json.length(), WEBSOCKET_OP_TEXT);
    }
}

std::string WebServer::getUrl() const
{
    return "http://127.0.0.1:" + std::to_string(port_);
}

std::string WebServer::statusToJson(const WebServerStatus& s)
{
    std::ostringstream oss;
    oss << "{";
    oss << "\"isRunning\":" << (s.isRunning ? "true" : "false") << ",";
    oss << "\"currentError\":{\"x\":" << s.currentErrorX << ",\"y\":" << s.currentErrorY << "},";
    oss << "\"currentOutput\":{\"x\":" << s.currentOutputX << ",\"y\":" << s.currentOutputY << "},";
    oss << "\"target\":{\"x\":" << s.targetX << ",\"y\":" << s.targetY << "},";
    oss << "\"fps\":" << s.fps << ",";
    oss << "\"controllerFrequency\":" << s.controllerFrequency;
    oss << "}";
    return oss.str();
}

std::string WebServer::configToJson(const MouseControllerConfig& config)
{
    std::ostringstream oss;
    oss << std::fixed;
    oss.precision(4);
    
    oss << "{";
    oss << "\"algorithmType\":" << static_cast<int>(config.algorithmType) << ",";
    
    // AdvancedPID参数
    oss << "\"advancedPID\":{";
    oss << "\"pMin\":" << config.pidPMin << ",";
    oss << "\"pMax\":" << config.pidPMax << ",";
    oss << "\"pSlope\":" << config.pidPSlope << ",";
    oss << "\"d\":" << config.pidD << ",";
    oss << "\"i\":" << config.pidI << ",";
    oss << "\"derivativeFilterAlpha\":" << config.derivativeFilterAlpha << ",";
    oss << "\"predictionWeightX\":" << config.predictionWeightX << ",";
    oss << "\"predictionWeightY\":" << config.predictionWeightY << ",";
    oss << "\"aimSmoothingX\":" << config.aimSmoothingX << ",";
    oss << "\"aimSmoothingY\":" << config.aimSmoothingY << ",";
    oss << "\"maxPixelMove\":" << config.maxPixelMove << ",";
    oss << "\"deadZonePixels\":" << config.deadZonePixels << ",";
    oss << "\"maxPredictionTime\":" << config.maxPredictionTime;
    oss << "},";
    
    // StandardPID参数
    oss << "\"standardPID\":{";
    oss << "\"kp\":" << config.stdKp << ",";
    oss << "\"ki\":" << config.stdKi << ",";
    oss << "\"kd\":" << config.stdKd << ",";
    oss << "\"smoothingX\":" << config.stdSmoothingX << ",";
    oss << "\"smoothingY\":" << config.stdSmoothingY << ",";
    oss << "\"derivativeFilterAlpha\":" << config.stdDerivativeFilterAlpha << ",";
    oss << "\"outputLimit\":" << config.stdOutputLimit << ",";
    oss << "\"deadZone\":" << config.stdDeadZone << ",";
    oss << "\"integralLimit\":" << config.stdIntegralLimit << ",";
    oss << "\"integralThreshold\":" << config.stdIntegralThreshold;
    oss << "},";
    
    // ChrisPID参数
    oss << "\"chrisPID\":{";
    oss << "\"kp\":" << config.chrisKp << ",";
    oss << "\"ki\":" << config.chrisKi << ",";
    oss << "\"kd\":" << config.chrisKd << ",";
    oss << "\"predWeightX\":" << config.chrisPredWeightX << ",";
    oss << "\"predWeightY\":" << config.chrisPredWeightY << ",";
    oss << "\"initScale\":" << config.chrisInitScale << ",";
    oss << "\"rampTime\":" << config.chrisRampTime << ",";
    oss << "\"outputMax\":" << config.chrisOutputMax << ",";
    oss << "\"iMax\":" << config.chrisIMax << ",";
    oss << "\"dFilterAlpha\":" << config.chrisDFilterAlpha;
    oss << "}";
    
    oss << "}";
    return oss.str();
}

void WebServer::eventHandler(struct mg_connection* c, int ev, void* ev_data)
{
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message* hm = (struct mg_http_message*)ev_data;
        
        struct mg_str* uri = &hm->uri;
        
        if (uri->len > 0) {
            std::string uriStr(uri->buf, uri->len);
            
            if (uriStr == "/api/config") {
                if (hm->method.len == 3 && strncmp(hm->method.buf, "GET", 3) == 0) {
                    handleGetConfig(c);
                } else if (hm->method.len == 4 && strncmp(hm->method.buf, "POST", 4) == 0) {
                    handlePostConfig(c, hm);
                } else {
                    mg_http_reply(c, 405, "Content-Type: application/json\r\n", "{\"error\":\"Method Not Allowed\"}");
                }
            } else if (uriStr == "/api/status") {
                handleGetStatus(c);
            } else if (uriStr == "/api/algorithms") {
                handleGetAlgorithms(c);
            } else if (uriStr == "/ws") {
                mg_ws_upgrade(c, hm, nullptr);
            } else if (uriStr == "/" || uriStr == "/index.html") {
                handleStaticFile(c, "index.html");
            } else if (uriStr == "/style.css") {
                handleStaticFile(c, "style.css");
            } else if (uriStr == "/app.js") {
                handleStaticFile(c, "app.js");
            } else {
                mg_http_reply(c, 404, "Content-Type: application/json\r\n", "{\"error\":\"Not Found\"}");
            }
        }
    } else if (ev == MG_EV_WS_MSG) {
        struct mg_ws_message* wm = (struct mg_ws_message*)ev_data;
        handleWebSocket(c, wm);
    } else if (ev == MG_EV_CLOSE) {
        std::lock_guard<std::mutex> lock(mutex_);
        wsConnections_.erase(std::remove(wsConnections_.begin(), wsConnections_.end(), c), wsConnections_.end());
    }
}

void WebServer::handleGetConfig(struct mg_connection* c)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (statusProvider_) {
        MouseControllerConfig config = statusProvider_();
        std::string json = configToJson(config);
        mg_http_reply(c, 200, "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n", "%s", json.c_str());
    } else {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n", "{\"error\":\"No status provider\"}");
    }
}

void WebServer::handlePostConfig(struct mg_connection* c, struct mg_http_message* hm)
{
    std::string body(hm->body.buf, hm->body.len);
    
    MouseControllerConfig config;
    
    // 算法类型
    config.algorithmType = static_cast<AlgorithmType>(jsonGetInt(body, "algorithmType", 0));
    
    // AdvancedPID参数
    std::string advJson = extractJsonObject(body, "advancedPID");
    config.pidPMin = jsonGetFloat(advJson, "pMin", config.pidPMin);
    config.pidPMax = jsonGetFloat(advJson, "pMax", config.pidPMax);
    config.pidPSlope = jsonGetFloat(advJson, "pSlope", config.pidPSlope);
    config.pidD = jsonGetFloat(advJson, "d", config.pidD);
    config.pidI = jsonGetFloat(advJson, "i", config.pidI);
    config.derivativeFilterAlpha = jsonGetFloat(advJson, "derivativeFilterAlpha", config.derivativeFilterAlpha);
    config.predictionWeightX = jsonGetFloat(advJson, "predictionWeightX", config.predictionWeightX);
    config.predictionWeightY = jsonGetFloat(advJson, "predictionWeightY", config.predictionWeightY);
    config.aimSmoothingX = jsonGetFloat(advJson, "aimSmoothingX", config.aimSmoothingX);
    config.aimSmoothingY = jsonGetFloat(advJson, "aimSmoothingY", config.aimSmoothingY);
    config.maxPixelMove = jsonGetFloat(advJson, "maxPixelMove", config.maxPixelMove);
    config.deadZonePixels = jsonGetFloat(advJson, "deadZonePixels", config.deadZonePixels);
    config.maxPredictionTime = jsonGetFloat(advJson, "maxPredictionTime", config.maxPredictionTime);
    
    // StandardPID参数
    std::string stdJson = extractJsonObject(body, "standardPID");
    config.stdKp = jsonGetFloat(stdJson, "kp", config.stdKp);
    config.stdKi = jsonGetFloat(stdJson, "ki", config.stdKi);
    config.stdKd = jsonGetFloat(stdJson, "kd", config.stdKd);
    config.stdSmoothingX = jsonGetFloat(stdJson, "smoothingX", config.stdSmoothingX);
    config.stdSmoothingY = jsonGetFloat(stdJson, "smoothingY", config.stdSmoothingY);
    config.stdDerivativeFilterAlpha = jsonGetFloat(stdJson, "derivativeFilterAlpha", config.stdDerivativeFilterAlpha);
    config.stdOutputLimit = jsonGetFloat(stdJson, "outputLimit", config.stdOutputLimit);
    config.stdDeadZone = jsonGetFloat(stdJson, "deadZone", config.stdDeadZone);
    config.stdIntegralLimit = jsonGetFloat(stdJson, "integralLimit", config.stdIntegralLimit);
    config.stdIntegralThreshold = jsonGetFloat(stdJson, "integralThreshold", config.stdIntegralThreshold);
    
    // ChrisPID参数
    std::string chrisJson = extractJsonObject(body, "chrisPID");
    config.chrisKp = jsonGetFloat(chrisJson, "kp", config.chrisKp);
    config.chrisKi = jsonGetFloat(chrisJson, "ki", config.chrisKi);
    config.chrisKd = jsonGetFloat(chrisJson, "kd", config.chrisKd);
    config.chrisPredWeightX = jsonGetFloat(chrisJson, "predWeightX", config.chrisPredWeightX);
    config.chrisPredWeightY = jsonGetFloat(chrisJson, "predWeightY", config.chrisPredWeightY);
    config.chrisInitScale = jsonGetFloat(chrisJson, "initScale", config.chrisInitScale);
    config.chrisRampTime = jsonGetFloat(chrisJson, "rampTime", config.chrisRampTime);
    config.chrisOutputMax = jsonGetFloat(chrisJson, "outputMax", config.chrisOutputMax);
    config.chrisIMax = jsonGetFloat(chrisJson, "iMax", config.chrisIMax);
    config.chrisDFilterAlpha = jsonGetFloat(chrisJson, "dFilterAlpha", config.chrisDFilterAlpha);
    
    // 调用回调同步到控制器
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (configCallback_) {
            configCallback_(config);
        }
    }
    
    std::cout << "[WebServer] 配置已更新: algorithmType=" << static_cast<int>(config.algorithmType) << std::endl;
    
    mg_http_reply(c, 200, "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n", "{\"success\":true}");
}

void WebServer::handleGetStatus(struct mg_connection* c)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string json = statusToJson(status_);
    mg_http_reply(c, 200, "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n", "%s", json.c_str());
}

void WebServer::handleGetAlgorithms(struct mg_connection* c)
{
    std::string json = "[{\"id\":0,\"name\":\"高级PID\"},{\"id\":1,\"name\":\"标准PID\"},{\"id\":2,\"name\":\"ChrisPID\"}]";
    mg_http_reply(c, 200, "Content-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n", "%s", json.c_str());
}

void WebServer::handleWebSocket(struct mg_connection* c, struct mg_ws_message* wm)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(wsConnections_.begin(), wsConnections_.end(), c) == wsConnections_.end()) {
        wsConnections_.push_back(c);
    }
    std::string json = statusToJson(status_);
    mg_ws_send(c, json.c_str(), json.length(), WEBSOCKET_OP_TEXT);
}

void WebServer::handleStaticFile(struct mg_connection* c, const char* filename)
{
    if (strcmp(filename, "index.html") == 0) {
        const char* html = R"(<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OBS 瞄准控制器</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <div class="header-content">
                <h1>🎯 瞄准控制器</h1>
                <div class="status-bar">
                    <div class="status-item">
                        <span class="status-label">状态</span>
                        <span id="status" class="status-value">加载中...</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">FPS</span>
                        <span id="fps" class="status-value">--</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">误差</span>
                        <span id="error" class="status-value">--</span>
                    </div>
                </div>
            </div>
        </header>
        
        <nav class="tabs">
            <button class="tab-btn active" data-tab="advanced">高级PID</button>
            <button class="tab-btn" data-tab="standard">标准PID</button>
            <button class="tab-btn" data-tab="chris">ChrisPID</button>
        </nav>
        
        <main id="content">
            <div class="loading">
                <div class="spinner"></div>
                <p>加载配置中...</p>
            </div>
        </main>
        
        <footer>
            <button id="save-btn" class="btn-primary">💾 保存配置</button>
            <button id="reset-btn" class="btn-secondary">🔄 重置</button>
        </footer>
    </div>
    
    <div id="toast" class="toast"></div>
    
    <script src="app.js"></script>
</body>
</html>)";
        mg_http_reply(c, 200, "Content-Type: text/html; charset=utf-8\r\n", "%s", html);
    } else if (strcmp(filename, "style.css") == 0) {
        const char* css = R"(* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-dark: #0f0f1a;
    --bg-card: rgba(30, 30, 50, 0.8);
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: rgba(255, 255, 255, 0.1);
}

body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
}

.container {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

/* Header */
header {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

h1 {
    font-size: 28px;
    font-weight: 600;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.status-bar {
    display: flex;
    gap: 24px;
}

.status-item {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.status-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-value {
    font-size: 18px;
    font-weight: 600;
    color: var(--success);
}

/* Tabs */
.tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
}

.tab-btn {
    flex: 1;
    padding: 14px 24px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 500;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.tab-btn:hover {
    background: rgba(99, 102, 241, 0.1);
    border-color: var(--primary);
    color: var(--text);
}

.tab-btn.active {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    border-color: transparent;
    color: white;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}

/* Main Content */
main {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid var(--border);
    min-height: 400px;
}

/* Param Groups */
.param-group {
    margin-bottom: 24px;
    padding: 20px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
    border: 1px solid var(--border);
}

.param-group h3 {
    color: var(--primary);
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}

.param-row {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    gap: 12px;
}

.param-row label {
    flex: 0 0 140px;
    font-size: 14px;
    color: var(--text-muted);
}

.param-row input[type="range"] {
    flex: 1;
    height: 6px;
    -webkit-appearance: none;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    outline: none;
}

.param-row input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    background: var(--primary);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.5);
    transition: transform 0.2s;
}

.param-row input[type="range"]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

.param-value {
    width: 70px;
    text-align: right;
    font-size: 14px;
    font-weight: 600;
    color: var(--success);
    font-family: 'Consolas', monospace;
}

/* Footer */
footer {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 24px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
}

button {
    padding: 14px 32px;
    border: none;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 30px rgba(99, 102, 241, 0.5);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.1);
    color: var(--text);
    border: 1px solid var(--border);
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.15);
}

/* Loading */
.loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px;
    color: var(--text-muted);
}

.spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 16px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Toast */
.toast {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%) translateY(100px);
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    padding: 16px 32px;
    border-radius: 12px;
    border: 1px solid var(--border);
    font-weight: 500;
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 1000;
}

.toast.show {
    transform: translateX(-50%) translateY(0);
    opacity: 1;
}

.toast.success {
    border-color: var(--success);
    color: var(--success);
}

.toast.error {
    border-color: var(--danger);
    color: var(--danger);
}

/* Responsive */
@media (max-width: 600px) {
    .container {
        padding: 12px;
    }
    
    .header-content {
        flex-direction: column;
        text-align: center;
    }
    
    .status-bar {
        justify-content: center;
    }
    
    .tabs {
        flex-direction: column;
    }
    
    .param-row {
        flex-wrap: wrap;
    }
    
    .param-row label {
        flex: 0 0 100%;
        margin-bottom: 4px;
    }
    
    .param-row input[type="range"] {
        flex: 1;
    }
    
    footer {
        flex-direction: column;
    }
    
    footer button {
        width: 100%;
    }
})";
        mg_http_reply(c, 200, "Content-Type: text/css; charset=utf-8\r\n", "%s", css);
    } else if (strcmp(filename, "app.js") == 0) {
        const char* js = R"(let currentConfig = null;
let ws = null;
let reconnectTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    setupTabs();
    setupButtons();
    connectWebSocket();
});

function loadConfig() {
    fetch('/api/config')
        .then(r => r.json())
        .then(config => {
            currentConfig = config;
            renderParams();
            showToast('配置加载成功', 'success');
        })
        .catch(err => {
            console.error('加载配置失败:', err);
            showToast('加载配置失败', 'error');
        });
}

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelector('.tab-btn.active').classList.remove('active');
            btn.classList.add('active');
            renderParams();
        });
    });
}

function setupButtons() {
    document.getElementById('save-btn').addEventListener('click', saveConfig);
    document.getElementById('reset-btn').addEventListener('click', () => {
        loadConfig();
        showToast('配置已重置', 'success');
    });
}

function renderParams() {
    const content = document.getElementById('content');
    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    
    let html = '';
    if (activeTab === 'advanced') {
        html = renderAdvancedPID();
    } else if (activeTab === 'standard') {
        html = renderStandardPID();
    } else if (activeTab === 'chris') {
        html = renderChrisPID();
    }
    
    content.innerHTML = html;
    setupSliders();
}

function renderAdvancedPID() {
    const p = currentConfig.advancedPID;
    return `
        <div class="param-group">
            <h3>📊 P增益</h3>
            ${sliderRow('pMin', 'P最小值', p.pMin, 0, 1, 0.01)}
            ${sliderRow('pMax', 'P最大值', p.pMax, 0, 1, 0.01)}
            ${sliderRow('pSlope', 'P斜率', p.pSlope, 0, 3, 0.01)}
        </div>
        <div class="param-group">
            <h3>📉 D增益</h3>
            ${sliderRow('d', 'D系数', p.d, 0, 0.1, 0.001)}
            ${sliderRow('derivativeFilterAlpha', 'D滤波系数', p.derivativeFilterAlpha, 0, 1, 0.01)}
        </div>
        <div class="param-group">
            <h3>📈 I增益</h3>
            ${sliderRow('i', 'I系数', p.i, 0, 0.1, 0.001)}
        </div>
        <div class="param-group">
            <h3>🔮 预测</h3>
            ${sliderRow('predictionWeightX', '预测权重X', p.predictionWeightX, 0, 2, 0.01)}
            ${sliderRow('predictionWeightY', '预测权重Y', p.predictionWeightY, 0, 2, 0.01)}
            ${sliderRow('maxPredictionTime', '最大预测时间', p.maxPredictionTime, 0, 0.5, 0.01)}
        </div>
        <div class="param-group">
            <h3>🌊 平滑</h3>
            ${sliderRow('aimSmoothingX', 'X轴平滑', p.aimSmoothingX, 0, 1, 0.01)}
            ${sliderRow('aimSmoothingY', 'Y轴平滑', p.aimSmoothingY, 0, 1, 0.01)}
        </div>
        <div class="param-group">
            <h3>⚙️ 限制</h3>
            ${sliderRow('maxPixelMove', '最大移动像素', p.maxPixelMove, 0, 300, 1)}
            ${sliderRow('deadZonePixels', '死区像素', p.deadZonePixels, 0, 50, 1)}
        </div>
    `;
}

function renderStandardPID() {
    const p = currentConfig.standardPID;
    return `
        <div class="param-group">
            <h3>📊 PID参数</h3>
            ${sliderRow('kp', 'Kp', p.kp, 0, 1, 0.01)}
            ${sliderRow('ki', 'Ki', p.ki, 0, 0.1, 0.001)}
            ${sliderRow('kd', 'Kd', p.kd, 0, 0.1, 0.001)}
        </div>
        <div class="param-group">
            <h3>🌊 平滑</h3>
            ${sliderRow('smoothingX', 'X轴平滑', p.smoothingX, 0, 1, 0.01)}
            ${sliderRow('smoothingY', 'Y轴平滑', p.smoothingY, 0, 1, 0.01)}
            ${sliderRow('derivativeFilterAlpha', 'D滤波系数', p.derivativeFilterAlpha, 0, 1, 0.01)}
        </div>
        <div class="param-group">
            <h3>⚙️ 限制</h3>
            ${sliderRow('outputLimit', '输出限制', p.outputLimit, 0, 200, 1)}
            ${sliderRow('deadZone', '死区', p.deadZone, 0, 10, 0.1)}
            ${sliderRow('integralLimit', '积分限制', p.integralLimit, 0, 200, 1)}
            ${sliderRow('integralThreshold', '积分分离阈值', p.integralThreshold, 0, 100, 1)}
        </div>
    `;
}

function renderChrisPID() {
    const p = currentConfig.chrisPID;
    return `
        <div class="param-group">
            <h3>📊 PID参数</h3>
            ${sliderRow('kp', 'Kp', p.kp, 0, 1, 0.01)}
            ${sliderRow('ki', 'Ki', p.ki, 0, 0.1, 0.001)}
            ${sliderRow('kd', 'Kd', p.kd, 0, 0.2, 0.001)}
        </div>
        <div class="param-group">
            <h3>🔮 预测</h3>
            ${sliderRow('predWeightX', '预测权重X', p.predWeightX, 0, 2, 0.01)}
            ${sliderRow('predWeightY', '预测权重Y', p.predWeightY, 0, 2, 0.01)}
        </div>
        <div class="param-group">
            <h3>📈 缩放</h3>
            ${sliderRow('initScale', '初始缩放', p.initScale, 0, 1, 0.01)}
            ${sliderRow('rampTime', '爬升时间', p.rampTime, 0, 2, 0.01)}
        </div>
        <div class="param-group">
            <h3>⚙️ 限制</h3>
            ${sliderRow('outputMax', '最大输出', p.outputMax, 0, 300, 1)}
            ${sliderRow('iMax', '积分最大值', p.iMax, 0, 200, 1)}
            ${sliderRow('dFilterAlpha', 'D滤波系数', p.dFilterAlpha, 0, 1, 0.01)}
        </div>
    `;
}

function sliderRow(key, label, value, min, max, step) {
    return `
        <div class="param-row">
            <label>${label}</label>
            <input type="range" data-key="${key}" min="${min}" max="${max}" step="${step}" value="${value}">
            <span class="param-value">${formatValue(value, step)}</span>
        </div>
    `;
}

function formatValue(value, step) {
    if (step >= 1) return value.toFixed(0);
    if (step >= 0.01) return value.toFixed(2);
    return value.toFixed(3);
}

function setupSliders() {
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            const key = e.target.dataset.key;
            const step = parseFloat(e.target.step);
            
            e.target.nextElementSibling.textContent = formatValue(value, step);
            
            updateConfigValue(key, value);
        });
    });
}

function updateConfigValue(key, value) {
    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    
    if (activeTab === 'advanced') {
        currentConfig.advancedPID[key] = value;
    } else if (activeTab === 'standard') {
        currentConfig.standardPID[key] = value;
    } else if (activeTab === 'chris') {
        currentConfig.chrisPID[key] = value;
    }
}

function saveConfig() {
    currentConfig.algorithmType = getAlgorithmType();
    
    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentConfig)
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            showToast('配置保存成功！', 'success');
        } else {
            showToast('保存失败: ' + (data.error || '未知错误'), 'error');
        }
    })
    .catch(err => {
        showToast('保存失败: ' + err.message, 'error');
    });
}

function getAlgorithmType() {
    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    if (activeTab === 'advanced') return 0;
    if (activeTab === 'standard') return 1;
    if (activeTab === 'chris') return 2;
    return 0;
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateStatus(data);
        } catch (e) {
            console.error('WebSocket解析错误:', e);
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket断开，5秒后重连...');
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connectWebSocket, 5000);
    };
    
    ws.onerror = (err) => {
        console.error('WebSocket错误:', err);
    };
}

function updateStatus(data) {
    const statusEl = document.getElementById('status');
    const fpsEl = document.getElementById('fps');
    const errorEl = document.getElementById('error');
    
    if (statusEl) {
        statusEl.textContent = data.isRunning ? '运行中' : '已停止';
        statusEl.style.color = data.isRunning ? '#22c55e' : '#ef4444';
    }
    
    if (fpsEl) {
        fpsEl.textContent = data.fps || '--';
    }
    
    if (errorEl) {
        const errX = data.currentError?.x?.toFixed(1) || '--';
        const errY = data.currentError?.y?.toFixed(1) || '--';
        errorEl.textContent = `${errX}, ${errY}`;
    }
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast ' + type + ' show';
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
})";
        mg_http_reply(c, 200, "Content-Type: application/javascript; charset=utf-8\r\n", "%s", js);
    } else {
        mg_http_reply(c, 404, "Content-Type: text/plain\r\n", "Not Found");
    }
}
