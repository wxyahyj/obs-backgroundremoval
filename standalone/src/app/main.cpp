// YoloAim standalone 入口 — M4:引擎 + Web API
// 用法: yolo_host.exe [config.json]
// 默认: <exe 目录>/config/default.json(损坏时自动备份并回退默认)
// WebUI: http://127.0.0.1:17890

#include "config/ConfigStore.hpp"
#include "engine/Engine.hpp"
#include "overlay/OverlayWindow.hpp"
#include "web/ApiRoutes.hpp"
#include "web/HttpServer.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace {

std::filesystem::path exe_dir()
{
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
}

// 模型路径:相对路径按 exe 目录解析(先直接相对,再 models/)
void resolve_model_path(std::string& path, const std::filesystem::path& base)
{
    std::filesystem::path p(path);
    if (p.is_absolute() || path.empty())
        return;
    if (std::filesystem::exists(base / p))
        path = (base / p).string();
    else if (std::filesystem::exists(base / "models" / p.filename()))
        path = (base / "models" / p.filename()).string();
}

volatile LONG g_stop = 0;

BOOL WINAPI ctrl_handler(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
        InterlockedExchange(&g_stop, 1);
        return TRUE;
    }
    return FALSE;
}

} // namespace

int main(int argc, char** argv)
{
    SetConsoleCtrlHandler(ctrl_handler, TRUE);

    const std::filesystem::path base = exe_dir();
    std::filesystem::path cfg_path = base / "config" / "default.json";
    if (argc > 1)
        cfg_path = argv[1];

    ya::config::ConfigDocument cfg;
    std::string err;
    if (!ya::config::ConfigStore::load_or_default(cfg_path.string(), cfg, &err)) {
        std::fprintf(stderr, "[main] config: %s\n", err.c_str());
    }
    resolve_model_path(cfg.infer.model_path, base);

    std::fprintf(stderr, "[main] config: %s\n", cfg_path.string().c_str());
    std::fprintf(stderr, "[main] model: %s device=%s\n", cfg.infer.model_path.c_str(),
                 cfg.infer.device.c_str());

    ya::Engine engine;
    if (!engine.start(cfg)) {
        std::fprintf(stderr, "[main] engine start failed\n");
        return 1;
    }

    // Web API + UI
    ya::web::HttpServer web;
    ya::web::ApiContext actx{&engine, (base / "config" / "user.json").string()};
    ya::web::register_api_routes(web, actx);
    if (!web.start(17890, (base / "webui").string())) {
        std::fprintf(stderr, "[main] web server start failed (port busy?)\n");
        engine.stop();
        return 1;
    }

    std::fprintf(stderr, "[main] running — WebUI http://127.0.0.1:17890 (Ctrl+C to stop)\n");

    // 叠加层(可选,vision.show_floating_window 开关)
    ya::OverlayWindow overlay;
    bool overlay_on = false;
    while (!InterlockedCompareExchange(&g_stop, 0, 0)) {
        Sleep(50);
        const bool want = engine.config().vision.show_floating_window;
        if (want != overlay_on) {
            if (want) {
                const auto ci = engine.capture_info();
                if (overlay.create(ci.width > 0 ? ci.width : 640,
                                   ci.height > 0 ? ci.height : 640)) {
                    overlay.set_position(ci.origin_x, ci.origin_y);
                    overlay.show();
                    std::fprintf(stderr, "[overlay] shown %dx%d @(%d,%d)\n", ci.width,
                                 ci.height, ci.origin_x, ci.origin_y);
                }
            } else {
                overlay.hide();
            }
            overlay_on = want;
        }
        if (overlay_on) {
            const auto s = engine.stats();
            const auto ci = engine.capture_info();
            overlay.update(engine.last_detections(), ci.width > 0 ? ci.width : 1,
                           ci.height > 0 ? ci.height : 1, s.aim_status.fov_px,
                           engine.config().aim.show_fov);
        }
        overlay.pump();
    }

    web.stop();
    engine.stop();
    std::fprintf(stderr, "[main] stopped\n");
    return 0;
}
