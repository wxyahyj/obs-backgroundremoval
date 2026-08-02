// YoloAim standalone 入口 — M4:引擎 + Web API
// 用法: yolo_host.exe [config.json]
// 默认: <exe 目录>/config/default.json(损坏时自动备份并回退默认)
// WebUI: http://127.0.0.1:17890

#include "config/ConfigStore.hpp"
#include "engine/Engine.hpp"
#include "overlay/OverlayWindow.hpp"
#include "util/CrashGuard.hpp"
#include "util/Log.hpp"
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
    const std::filesystem::path log_dir = base / "logs";
    ya::util::log_init(log_dir.string());
    ya::util::install_crash_handler(log_dir.string());

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
    ya::web::ApiContext actx{&engine, (base / "config" / "user.json").string(),
                             base.string()};
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
    int overlay_w_ = 0;
    int overlay_h_ = 0;
    while (!InterlockedCompareExchange(&g_stop, 0, 0)) {
        Sleep(50);
        const bool want = engine.config().vision.show_floating_window;
        const auto cfg_now = engine.config();
        const int fw = cfg_now.vision.floating_window_width > 0
                           ? cfg_now.vision.floating_window_width
                           : 480;
        const int fh = cfg_now.vision.floating_window_height > 0
                           ? cfg_now.vision.floating_window_height
                           : 360;
        if (want != overlay_on) {
            if (want) {
                if (overlay.create(fw, fh)) {
                    overlay.show();
                    std::fprintf(stderr, "[overlay] floating window shown %dx%d\n", fw,
                                 fh);
                } else {
                    std::fprintf(stderr, "[overlay] floating window create FAILED\n");
                }
            } else {
                overlay.hide();
            }
            overlay_on = want;
            overlay_w_ = fw;
            overlay_h_ = fh;
        } else if (overlay_on && (fw != overlay_w_ || fh != overlay_h_)) {
            // 宽高配置变更 → 重建窗口
            overlay.destroy();
            if (overlay.create(fw, fh)) {
                overlay.show();
                std::fprintf(stderr, "[overlay] floating window resized %dx%d\n", fw,
                             fh);
            }
            overlay_w_ = fw;
            overlay_h_ = fh;
        }
        if (overlay_on) {
            const auto s = engine.stats();
            const auto pf = engine.preview_frame();
            overlay.update(pf.bgr, pf.width > 0 ? pf.width : 1,
                           pf.height > 0 ? pf.height : 1, engine.last_detections(),
                           s.aim_status.fov_px, cfg_now.aim.show_fov);
        }
        overlay.pump();
    }

    web.stop();
    engine.stop();
    std::fprintf(stderr, "[main] stopped\n");
    return 0;
}
