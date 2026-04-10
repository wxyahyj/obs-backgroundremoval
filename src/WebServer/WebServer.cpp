#include "WebServer.hpp"
#include "mongoose.h"
#include <iostream>
#include <sstream>
#include <fstream>

static WebServer* g_webServer = nullptr;

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
        
        std::string listenAddr = "http://127.0.0.1:" + std::to_string(port_);
        
        struct mg_connection* c = mg_http_listen(&mgr, listenAddr.c_str(), eventHandler, nullptr);
        if (c == nullptr) {
            std::cerr << "[WebServer] 无法启动服务器在端口 " << port_ << std::endl;
            running_ = false;
            return;
        }
        
        std::cout << "[WebServer] 服务器启动成功，访问地址: " << listenAddr << std::endl;
        
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
    
    // 广播WebSocket消息
    if (!wsConnections_.empty()) {
        std::string json = statusToJson(status);
        for (auto* c : wsConnections_) {
            mg_ws_send((struct mg_connection*)c, json.c_str(), json.length(), WEBSOCKET_OP_TEXT);
        }
    }
}

std::string WebServer::statusToJson(const WebServerStatus& status)
{
    std::ostringstream oss;
    oss << "{";
    oss << "\"isRunning\":" << (status.isRunning ? "true" : "false") << ",";
    oss << "\"currentError\":{\"x\":" << status.currentErrorX << ",\"y\":" << status.currentErrorY << "},";
    oss << "\"currentOutput\":{\"x\":" << status.currentOutputX << ",\"y\":" << status.currentOutputY << "},";
    oss << "\"target\":{\"x\":" << status.targetX << ",\"y\":" << status.targetY << "},";
    oss << "\"fps\":" << status.fps << ",";
    oss << "\"controllerFrequency\":" << status.controllerFrequency;
    oss << "}";
    return oss.str();
}

std::string WebServer::configToJson(const MouseControllerConfig& config)
{
    std::ostringstream oss;
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
    oss << "\"aimSmoothingY\":" << config.aimSmoothingY;
    oss << "},";
    
    // StandardPID参数
    oss << "\"standardPID\":{";
    oss << "\"kp\":" << config.stdKp << ",";
    oss << "\"ki\":" << config.stdKi << ",";
    oss << "\"kd\":" << config.stdKd << ",";
    oss << "\"smoothingX\":" << config.stdSmoothingX << ",";
    oss << "\"smoothingY\":" << config.stdSmoothingY << ",";
    oss << "\"derivativeFilterAlpha\":" << config.stdDerivativeFilterAlpha;
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

void WebServer::eventHandler(struct mg_connection* c, int ev, void* ev_data, void* fn_data)
{
    if (ev == MG_EV_HTTP_MSG) {
        struct mg_http_message* hm = (struct mg_http_message*)ev_data;
        
        if (mg_http_match_uri(hm, "/api/config")) {
            if (mg_vcasecmp(&hm->method, "GET") == 0) {
                handleGetConfig(c);
            } else if (mg_vcasecmp(&hm->method, "POST") == 0) {
                handlePostConfig(c, hm);
            } else {
                mg_http_reply(c, 405, "", "{\"error\":\"Method Not Allowed\"}");
            }
        } else if (mg_http_match_uri(hm, "/api/status")) {
            handleGetStatus(c);
        } else if (mg_http_match_uri(hm, "/api/algorithms")) {
            handleGetAlgorithms(c);
        } else if (mg_http_match_uri(hm, "/ws")) {
            mg_ws_upgrade(c, hm, nullptr);
        } else if (mg_http_match_uri(hm, "/")) {
            handleStaticFile(c, "index.html");
        } else if (mg_http_match_uri(hm, "/style.css")) {
            handleStaticFile(c, "style.css");
        } else if (mg_http_match_uri(hm, "/app.js")) {
            handleStaticFile(c, "app.js");
        } else {
            mg_http_reply(c, 404, "", "{\"error\":\"Not Found\"}");
        }
    } else if (ev == MG_EV_WS_MSG) {
        handleWebSocket(c, ev_data);
    } else if (ev == MG_EV_CLOSE) {
        // 移除WebSocket连接
        if (g_webServer) {
            std::lock_guard<std::mutex> lock(g_webServer->mutex_);
            auto& conns = g_webServer->wsConnections_;
            conns.erase(std::remove(conns.begin(), conns.end(), c), conns.end());
        }
    }
}

void WebServer::handleGetConfig(struct mg_connection* c)
{
    if (!g_webServer) {
        mg_http_reply(c, 500, "", "{\"error\":\"Server not initialized\"}");
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_webServer->mutex_);
    if (g_webServer->statusProvider_) {
        MouseControllerConfig config = g_webServer->statusProvider_();
        std::string json = g_webServer->configToJson(config);
        mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", json.c_str());
    } else {
        mg_http_reply(c, 500, "", "{\"error\":\"No status provider\"}");
    }
}

void WebServer::handlePostConfig(struct mg_connection* c, struct mg_http_message* hm)
{
    if (!g_webServer) {
        mg_http_reply(c, 500, "", "{\"error\":\"Server not initialized\"}");
        return;
    }
    
    // 解析JSON并更新配置
    std::string body(hm->body.ptr, hm->body.len);
    
    // TODO: 解析JSON并调用configCallback_
    
    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"success\":true}");
}

void WebServer::handleGetStatus(struct mg_connection* c)
{
    if (!g_webServer) {
        mg_http_reply(c, 500, "", "{\"error\":\"Server not initialized\"}");
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_webServer->mutex_);
    std::string json = g_webServer->statusToJson(g_webServer->status_);
    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", json.c_str());
}

void WebServer::handleGetAlgorithms(struct mg_connection* c)
{
    std::string json = "[{\"id\":0,\"name\":\"高级PID\"},{\"id\":1,\"name\":\"标准PID\"},{\"id\":2,\"name\":\"ChrisPID\"}]";
    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", json.c_str());
}

void WebServer::handleWebSocket(struct mg_connection* c, void* ev_data)
{
    if (!g_webServer) return;
    
    struct mg_ws_message* wm = (struct mg_ws_message*)ev_data;
    
    // 添加到连接列表
    {
        std::lock_guard<std::mutex> lock(g_webServer->mutex_);
        if (std::find(g_webServer->wsConnections_.begin(), g_webServer->wsConnections_.end(), c) == g_webServer->wsConnections_.end()) {
            g_webServer->wsConnections_.push_back(c);
        }
    }
    
    // 发送当前状态
    std::lock_guard<std::mutex> lock(g_webServer->mutex_);
    std::string json = g_webServer->statusToJson(g_webServer->status_);
    mg_ws_send(c, json.c_str(), json.length(), WEBSOCKET_OP_TEXT);
}

void WebServer::handleStaticFile(struct mg_connection* c, const char* filename)
{
    // TODO: 从内存或文件系统读取静态文件
    std::string content;
    
    if (strcmp(filename, "index.html") == 0) {
        content = R"(<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OBS 瞄准控制器</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>瞄准控制器参数</h1>
            <div class="status">状态: <span id="status">运行中</span></div>
        </header>
        
        <nav>
            <button class="tab-btn active" data-tab="advanced">高级PID</button>
            <button class="tab-btn" data-tab="standard">标准PID</button>
            <button class="tab-btn" data-tab="chris">ChrisPID</button>
        </nav>
        
        <main id="content">
            <p>加载中...</p>
        </main>
        
        <footer>
            <button id="save-btn">保存配置</button>
            <button id="reset-btn">重置</button>
        </footer>
    </div>
    
    <script src="app.js"></script>
</body>
</html>)";
        mg_http_reply(c, 200, "Content-Type: text/html\r\n", "%s", content.c_str());
    } else if (strcmp(filename, "style.css") == 0) {
        content = R"(
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }
.container { max-width: 800px; margin: 0 auto; padding: 20px; }
header { text-align: center; margin-bottom: 20px; }
nav { display: flex; gap: 10px; margin-bottom: 20px; }
.tab-btn { flex: 1; padding: 10px; border: none; background: #333; color: #fff; cursor: pointer; }
.tab-btn.active { background: #4CAF50; }
main { background: #2a2a2a; padding: 20px; border-radius: 8px; }
footer { margin-top: 20px; display: flex; gap: 10px; }
button { padding: 10px 20px; border: none; background: #4CAF50; color: #fff; cursor: pointer; border-radius: 4px; }
button:hover { background: #45a049; }
.param-group { margin-bottom: 20px; }
.param-group h3 { margin-bottom: 10px; color: #4CAF50; }
.param-row { display: flex; align-items: center; margin-bottom: 10px; }
.param-row label { width: 150px; }
.param-row input[type="range"] { flex: 1; margin: 0 10px; }
.param-row span { width: 60px; text-align: right; }
)";
        mg_http_reply(c, 200, "Content-Type: text/css\r\n", "%s", content.c_str());
    } else if (strcmp(filename, "app.js") == 0) {
        content = R"(
let currentConfig = null;

document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    setupEventListeners();
    connectWebSocket();
});

function loadConfig() {
    fetch('/api/config')
        .then(r => r.json())
        .then(config => {
            currentConfig = config;
            renderParams();
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
}

function renderAdvancedPID() {
    const p = currentConfig.advancedPID;
    return `
        <div class="param-group">
            <h3>P增益</h3>
            <div class="param-row">
                <label>P最小值</label>
                <input type="range" min="0" max="1" step="0.01" value="${p.pMin}" data-param="pMin">
                <span>${p.pMin.toFixed(2)}</span>
            </div>
            <div class="param-row">
                <label>P最大值</label>
                <input type="range" min="0" max="1" step="0.01" value="${p.pMax}" data-param="pMax">
                <span>${p.pMax.toFixed(2)}</span>
            </div>
        </div>
        <div class="param-group">
            <h3>其他参数</h3>
            <div class="param-row">
                <label>D系数</label>
                <input type="range" min="0" max="2" step="0.01" value="${p.d}" data-param="d">
                <span>${p.d.toFixed(3)}</span>
            </div>
            <div class="param-row">
                <label>I系数</label>
                <input type="range" min="0" max="0.1" step="0.001" value="${p.i}" data-param="i">
                <span>${p.i.toFixed(3)}</span>
            </div>
        </div>
    `;
}

function renderStandardPID() {
    const p = currentConfig.standardPID;
    return `
        <div class="param-group">
            <h3>标准PID参数</h3>
            <div class="param-row">
                <label>Kp</label>
                <input type="range" min="0" max="1" step="0.01" value="${p.kp}" data-param="kp">
                <span>${p.kp.toFixed(2)}</span>
            </div>
            <div class="param-row">
                <label>Ki</label>
                <input type="range" min="0" max="0.1" step="0.001" value="${p.ki}" data-param="ki">
                <span>${p.ki.toFixed(3)}</span>
            </div>
            <div class="param-row">
                <label>Kd</label>
                <input type="range" min="0" max="0.1" step="0.001" value="${p.kd}" data-param="kd">
                <span>${p.kd.toFixed(3)}</span>
            </div>
        </div>
    `;
}

function renderChrisPID() {
    const p = currentConfig.chrisPID;
    return `
        <div class="param-group">
            <h3>ChrisPID参数</h3>
            <div class="param-row">
                <label>Kp</label>
                <input type="range" min="0" max="1" step="0.01" value="${p.kp}" data-param="kp">
                <span>${p.kp.toFixed(2)}</span>
            </div>
            <div class="param-row">
                <label>Ki</label>
                <input type="range" min="0" max="0.1" step="0.001" value="${p.ki}" data-param="ki">
                <span>${p.ki.toFixed(3)}</span>
            </div>
            <div class="param-row">
                <label>Kd</label>
                <input type="range" min="0" max="0.1" step="0.001" value="${p.kd}" data-param="kd">
                <span>${p.kd.toFixed(3)}</span>
            </div>
        </div>
    `;
}

function setupEventListeners() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelector('.tab-btn.active').classList.remove('active');
            btn.classList.add('active');
            renderParams();
        });
    });
    
    document.getElementById('save-btn').addEventListener('click', saveConfig);
}

function saveConfig() {
    fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(currentConfig)
    }).then(r => r.json()).then(data => {
        alert('配置已保存');
    });
}

function connectWebSocket() {
    const ws = new WebSocket('ws://' + location.host + '/ws');
    ws.onmessage = (event) => {
        const status = JSON.parse(event.data);
        document.getElementById('status').textContent = status.isRunning ? '运行中' : '已停止';
    };
}
)";
        mg_http_reply(c, 200, "Content-Type: application/javascript\r\n", "%s", content.c_str());
    } else {
        mg_http_reply(c, 404, "", "Not Found");
    }
}
