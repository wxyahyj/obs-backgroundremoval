// YoloAim standalone 入口 — M1 最小闭环
// 用法: yolo_host.exe [config.json]
// 默认: <exe 目录>/config/default.json

#include "engine/Engine.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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

// 模型路径:相对路径按 exe 目录解析(先 models/ 后直接相对)
std::string resolve_model_path(const std::string& path, const std::filesystem::path& base)
{
    std::filesystem::path p(path);
    if (p.is_absolute())
        return path;
    std::filesystem::path cand = base / p;
    if (std::filesystem::exists(cand))
        return cand.string();
    cand = base / "models" / p.filename();
    if (std::filesystem::exists(cand))
        return cand.string();
    return p.string(); // 原样返回,由 InferEngine 报错
}

bool load_config(const std::filesystem::path& path, ya::EngineConfig& cfg,
                 const std::filesystem::path& base)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[main] cannot open config: %s\n", path.string().c_str());
        return false;
    }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[main] config parse failed: %s\n", e.what());
        return false;
    }

    if (j.contains("capture")) {
        const auto& c = j["capture"];
        const std::string mode = c.value("mode", "center");
        cfg.backend = ya::FrameSource::parse_backend(c.value("backend", "dxgi"));
        cfg.center_region = (mode != "region");
        if (c.contains("region")) {
            cfg.region_x = c["region"].value("x", 0);
            cfg.region_y = c["region"].value("y", 0);
            cfg.width = c["region"].value("w", 640);
            cfg.height = c["region"].value("h", 640);
        } else {
            cfg.region_x = c.value("region_x", 0);
            cfg.region_y = c.value("region_y", 0);
            cfg.width = c.value("width", c.value("region_width", 640));
            cfg.height = c.value("height", c.value("region_height", 640));
        }
    }

    if (j.contains("infer")) {
        const auto& c = j["infer"];
        cfg.infer.model_path = resolve_model_path(c.value("model_path", ""), base);
        cfg.infer.device = c.value("device", c.value("use_gpu", "cpu"));
        cfg.infer.model_version = c.value("model_version", 2);
        cfg.infer.confidence = c.value("confidence", c.value("confidence_threshold", 0.45f));
        cfg.infer.nms = c.value("nms", c.value("nms_threshold", 0.45f));
        cfg.infer.input_resolution = c.value("input_resolution", c.value("input_size", 640));
        cfg.infer.num_threads = c.value("num_threads", 4);
        cfg.infer.interval_frames = c.value("interval_frames",
                                            c.value("inference_interval_frames", 1));
        cfg.infer.target_classes = c.value("target_classes",
                                           std::vector<int>{});
    }

    return true;
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

    ya::EngineConfig cfg;
    if (!load_config(cfg_path, cfg, base))
        return 1;

    std::fprintf(stderr, "[main] config: %s\n", cfg_path.string().c_str());
    std::fprintf(stderr, "[main] model: %s device=%s\n", cfg.infer.model_path.c_str(),
                 cfg.infer.device.c_str());

    ya::Engine engine;
    if (!engine.start(cfg)) {
        std::fprintf(stderr, "[main] engine start failed\n");
        return 1;
    }

    std::fprintf(stderr, "[main] engine running (Ctrl+C to stop)\n");
    while (!InterlockedCompareExchange(&g_stop, 0, 0))
        Sleep(200);

    engine.stop();
    std::fprintf(stderr, "[main] stopped\n");
    return 0;
}
