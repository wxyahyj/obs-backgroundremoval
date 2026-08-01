#pragma once

// API 路由层(M4 重写)— 薄 handler,业务逻辑在 Engine / ConfigStore。
// 挂载到 HttpServer;无 WS,状态用轮询(GET /api/status)。

#include "HttpServer.hpp"
#include "engine/Engine.hpp"

#include <string>

namespace ya {
namespace web {

struct ApiContext {
    Engine* engine = nullptr;
    std::string config_path; // user.json(不存在则不落盘)
    std::string exe_dir;     // 可执行目录(模型扫描用)
};

void register_api_routes(HttpServer& srv, ApiContext& ctx);

} // namespace web
} // namespace ya
