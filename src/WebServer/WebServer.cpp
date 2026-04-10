#include "WebServer.hpp"
#include "mongoose.h"
#include <iostream>
#include <sstream>
#include <cstring>

static WebServer* g_webServer = nullptr;

static void mongooseEventHandler(struct mg_connection* c, int ev, void* ev_data, void* fn_data)
{
    if (g_webServer) {
        g_webServer->eventHandler(c, ev, ev_data);
    }
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
        
        std::string listenAddr = "http://127.0.0.1:" + std::to_string(port_);
        
        struct mg_connection* c = mg_http_listen(&mgr, listenAddr.c_str(), mongooseEventHandler, nullptr);
        
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
    oss << "{";
    oss << "\"algorithmType\":" << static_cast<int>(config.algorithmType) << ",";
    
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
    
    oss << "\"standardPID\":{";
    oss << "\"kp\":" << config.stdKp << ",";
    oss << "\"ki\":" << config.stdKi << ",";
    oss << "\"kd\":" << config.stdKd << ",";
    oss << "\"smoothingX\":" << config.stdSmoothingX << ",";
    oss << "\"smoothingY\":" << config.stdSmoothingY << ",";
    oss << "\"derivativeFilterAlpha\":" << config.stdDerivativeFilterAlpha;
    oss << "},";
    
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
        mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", json.c_str());
    } else {
        mg_http_reply(c, 500, "Content-Type: application/json\r\n", "{\"error\":\"No status provider\"}");
    }
}

void WebServer::handlePostConfig(struct mg_connection* c, struct mg_http_message* hm)
{
    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{\"success\":true}");
}

void WebServer::handleGetStatus(struct mg_connection* c)
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::string json = statusToJson(status_);
    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", json.c_str());
}

void WebServer::handleGetAlgorithms(struct mg_connection* c)
{
    std::string json = "[{\"id\":0,\"name\":\"高级PID\"},{\"id\":1,\"name\":\"标准PID\"},{\"id\":2,\"name\":\"ChrisPID\"}]";
    mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", json.c_str());
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
            <div class="status">状态: <span id="status">加载中...</span></div>
        </header>
        <nav>
            <button class="tab-btn active" data-tab="advanced">高级PID</button>
            <button class="tab-btn" data-tab="standard">标准PID</button>
            <button class="tab-btn" data-tab="chris">ChrisPID</button>
        </nav>
        <main id="content"><p class="loading">加载中...</p></main>
        <footer>
            <button id="save-btn">保存配置</button>
            <button id="reset-btn">重置</button>
        </footer>
    </div>
    <script src="app.js"></script>
</body>
</html>)";
        mg_http_reply(c, 200, "Content-Type: text/html; charset=utf-8\r\n", "%s", html);
    } else if (strcmp(filename, "style.css") == 0) {
        const char* css = R"(*{margin:0;padding:0;box-sizing:border-box}body{font-family:Arial,sans-serif;background:#1a1a1a;color:#fff}.container{max-width:800px;margin:0 auto;padding:20px}header{text-align:center;padding:20px 0;border-bottom:1px solid #333}h1{margin:0 0 10px;font-size:24px}.status{color:#888;font-size:14px}nav{display:flex;gap:10px;padding:20px 0}.tab-btn{flex:1;padding:10px 20px;background:#333;border:1px solid #555;color:#fff;cursor:pointer;border-radius:5px}.tab-btn.active{background:#4CAF50;border-color:#4CAF50}main{padding:20px 0}.param-group{margin-bottom:20px}.param-group h3{color:#4CAF50;margin-bottom:10px;font-size:16px}.param-row{display:flex;align-items:center;margin-bottom:10px}.param-row label{flex:0 0 150px;font-size:14px}.param-row input[type=range]{flex:1;margin:0 10px}.param-value{width:60px;text-align:right;font-size:14px;color:#4CAF50}footer{text-align:center;padding:20px;border-top:1px solid #333;margin-top:20px}footer button{padding:10px 30px;margin:0 10px;background:#4CAF50;border:none;color:#fff;cursor:pointer;border-radius:5px;font-size:14px}.loading{text-align:center;padding:40px;color:#888})";
        mg_http_reply(c, 200, "Content-Type: text/css; charset=utf-8\r\n", "%s", css);
    } else if (strcmp(filename, "app.js") == 0) {
        const char* js = R"(let currentConfig=null;document.addEventListener('DOMContentLoaded',()=>{loadConfig();setupEventListeners();connectWebSocket()});function loadConfig(){fetch('/api/config').then(r=>r.json()).then(config=>{currentConfig=config;renderParams()})}function renderParams(){const content=document.getElementById('content');const activeTab=document.querySelector('.tab-btn.active').dataset.tab;let html='';if(activeTab==='advanced')html=renderAdvancedPID();else if(activeTab==='standard')html=renderStandardPID();else if(activeTab==='chris')html=renderChrisPID();content.innerHTML=html}function renderAdvancedPID(){const p=currentConfig.advancedPID;return `<div class="param-group"><h3>P增益</h3><div class="param-row"><label>P最小值</label><input type="range" min="0" max="1" step="0.01" value="${p.pMin}"><span class="param-value">${p.pMin.toFixed(2)}</span></div><div class="param-row"><label>P最大值</label><input type="range" min="0" max="1" step="0.01" value="${p.pMax}"><span class="param-value">${p.pMax.toFixed(2)}</span></div></div>`}function renderStandardPID(){const p=currentConfig.standardPID;return `<div class="param-group"><h3>标准PID参数</h3><div class="param-row"><label>Kp</label><input type="range" min="0" max="1" step="0.01" value="${p.kp}"><span class="param-value">${p.kp.toFixed(2)}</span></div></div>`}function renderChrisPID(){const p=currentConfig.chrisPID;return `<div class="param-group"><h3>ChrisPID参数</h3><div class="param-row"><label>Kp</label><input type="range" min="0" max="1" step="0.01" value="${p.kp}"><span class="param-value">${p.kp.toFixed(2)}</span></div></div>`}function setupEventListeners(){document.querySelectorAll('.tab-btn').forEach(btn=>{btn.addEventListener('click',()=>{document.querySelector('.tab-btn.active').classList.remove('active');btn.classList.add('active');renderParams()})});document.getElementById('save-btn').addEventListener('click',saveConfig)}function saveConfig(){fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentConfig)}).then(r=>r.json()).then(data=>{alert('配置已保存')})}function connectWebSocket(){const ws=new WebSocket('ws://'+location.host+'/ws');ws.onmessage=event=>{const status=JSON.parse(event.data);document.getElementById('status').textContent=status.isRunning?'运行中':'已停止'}})";
        mg_http_reply(c, 200, "Content-Type: application/javascript; charset=utf-8\r\n", "%s", js);
    } else {
        mg_http_reply(c, 404, "Content-Type: text/plain\r\n", "Not Found");
    }
}
