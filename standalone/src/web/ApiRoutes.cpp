// API 路由实现。

#include "ApiRoutes.hpp"

#include "config/ConfigStore.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>

namespace ya {
namespace web {

namespace {

// 按文件名推断模型版本(与 D:/AI 模型库命名一致)
int guess_model_version(const std::string& name)
{
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (n.find("yolov5") != std::string::npos || n.find("_v5") != std::string::npos)
        return 0; // YOLOv5
    if (n.find("v11") != std::string::npos || n.find("v26") != std::string::npos ||
        n.find("v27") != std::string::npos)
        return 2; // YOLOv11
    return 1; // 默认 YOLOv8(v8n/v8s/nms 等)
}

nlohmann::json status_json(const Engine& e)
{
    const PipelineStats s = e.stats();
    const auto ci = e.capture_info();
    const config::ConfigDocument cfg = e.config();
    nlohmann::json j;
    j["running"] = e.running();
    j["fps"] = s.fps;
    j["frames"] = s.frames;
    j["detections"] = s.detections;
    j["grab_ms"] = s.grab_ms;
    j["infer_ms"] = s.infer_ms;
    j["post_ms"] = s.post_ms;
    j["last_error"] = s.last_error;
    // parity 契约字段(OBS 对齐)
    j["capture_ok"] = ci.width > 0;
    j["capture_fps"] = s.fps;
    j["infer_enabled"] = cfg.infer.enabled;
    j["infer_ok"] = e.running();
    j["device_requested"] = cfg.infer.device;
    j["device_actual"] = cfg.infer.device;
    j["mode"] = cfg.capture.mode;
    j["use_region"] = cfg.capture.use_region;
    j["region_x"] = cfg.capture.region_x;
    j["region_y"] = cfg.capture.region_y;
    j["region_width"] = cfg.capture.region_width;
    j["region_height"] = cfg.capture.region_height;
    j["w"] = ci.width;
    j["h"] = ci.height;
    j["origin"] = {ci.origin_x, ci.origin_y};
    j["num_classes"] = e.num_classes();
    j["aim"] = {
        {"slot", s.aim_status.active_slot},
        {"aiming", s.aim_status.aiming},
        {"controller_ok", s.aim_status.controller_ok},
        {"fov_px", s.aim_status.fov_px},
        {"algorithm", s.aim_status.algorithm},
        {"controller_type", s.aim_status.controller_type},
    };
    return j;
}

nlohmann::json detections_json(const std::vector<Detection>& dets)
{
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& d : dets) {
        arr.push_back({
            {"class_id", d.classId},
            {"class", d.className},
            {"confidence", d.confidence},
            {"x", d.x},
            {"y", d.y},
            {"width", d.width},
            {"height", d.height},
            {"center_x", d.centerX},
            {"center_y", d.centerY},
            {"track_id", d.trackId},
        });
    }
    return arr;
}

HttpResponse ok_json(const nlohmann::json& j)
{
    return HttpResponse::json(j.dump());
}

HttpResponse fail_json(const std::string& msg, int status = 400)
{
    nlohmann::json j = {{"ok", false}, {"error", msg}};
    return HttpResponse::json(j.dump(), status);
}

} // namespace

void register_api_routes(HttpServer& srv, ApiContext& ctx)
{
    Engine* engine = ctx.engine;

    srv.route("GET", "/api/health",
              [](const HttpRequest&) { return HttpResponse::json("{\"ok\":true}"); });

    // 模型扫描:D:/AI(存在时)+ exe/models
    srv.route("GET", "/api/models", [&ctx](const HttpRequest&) {
        nlohmann::json models = nlohmann::json::array();
        std::vector<std::filesystem::path> dirs;
        if (!ctx.exe_dir.empty())
            dirs.push_back(std::filesystem::path(ctx.exe_dir) / "models");
        const std::filesystem::path d_ai("D:/AI");
        if (std::filesystem::exists(d_ai))
            dirs.push_back(d_ai);
        for (const auto& dir : dirs) {
            std::error_code ec;
            if (!std::filesystem::is_directory(dir, ec))
                continue;
            for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
                if (!entry.is_regular_file(ec))
                    continue;
                const std::string ext = entry.path().extension().string();
                if (ext != ".onnx")
                    continue;
                // 中文文件名:必须 UTF-8(u8string),否则 JSON 解析坏
                const std::string name = entry.path().filename().u8string();
                models.push_back({
                    {"name", name},
                    {"path", entry.path().u8string()},
                    {"size", static_cast<uint64_t>(entry.file_size(ec))},
                    {"version", guess_model_version(name)},
                });
            }
        }
        return ok_json({{"ok", true}, {"models", models}});
    });

    srv.route("GET", "/api/status",
              [engine](const HttpRequest&) { return ok_json(status_json(*engine)); });

    srv.route("GET", "/api/config", [engine](const HttpRequest&) {
        const nlohmann::json j = config::document_to_json(engine->config());
        return HttpResponse::json(
            (nlohmann::json{{"ok", true}, {"config", j}}).dump());
    });

    srv.route("PUT", "/api/config", [engine, &ctx](const HttpRequest& req) {
        nlohmann::json patch;
        try {
            patch = nlohmann::json::parse(req.body);
        } catch (const std::exception& e) {
            return fail_json(std::string("bad json: ") + e.what());
        }
        // 合并到当前配置副本 → 热应用 → 落盘
        config::ConfigDocument next = engine->config();
        std::string err;
        if (!config::ConfigStore::merge(next, patch, &err))
            return fail_json(err);
        engine->update_config(next);
        if (!ctx.config_path.empty()) {
            if (!config::ConfigStore::save_file(ctx.config_path, next, &err)) {
                std::fprintf(stderr, "[api] save config failed: %s\n", err.c_str());
            }
        }
        return ok_json({{"ok", true}});
    });

    srv.route("POST", "/api/engine/start", [engine](const HttpRequest&) {
        if (!engine->running()) {
            if (!engine->start(engine->config()))
                return fail_json("engine start failed (see log)");
        }
        return ok_json({{"ok", true}});
    });

    srv.route("POST", "/api/engine/stop", [engine](const HttpRequest&) {
        engine->stop();
        return ok_json({{"ok", true}});
    });

    srv.route("POST", "/api/engine/reload_model", [engine](const HttpRequest&) {
        engine->request_reload_model();
        return ok_json({{"ok", true}});
    });

    srv.route("POST", "/api/engine/reload_capture", [engine](const HttpRequest&) {
        engine->request_reload_capture();
        return ok_json({{"ok", true}});
    });

    srv.route("GET", "/api/detections", [engine](const HttpRequest&) {
        return ok_json({{"ok", true}, {"detections", detections_json(engine->last_detections())}});
    });

    srv.route("GET", "/api/preview.bmp", [engine](const HttpRequest&) {
        const std::vector<uint8_t> bmp = engine->preview_bmp();
        if (bmp.empty())
            return HttpResponse::text("no frame", 404);
        HttpResponse r;
        r.status = 200;
        r.content_type = "image/bmp";
        r.body.assign(bmp.begin(), bmp.end());
        return r;
    });

    srv.route("POST", "/api/crosshair/pick", [engine](const HttpRequest& req) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(req.body);
        } catch (...) {
            return fail_json("bad json");
        }
        if (!j.contains("x") || !j.contains("y"))
            return fail_json("need x,y (normalized 0..1)");
        int r = 0, g = 0, b = 0;
        if (!engine->pick_color(j["x"].get<double>(), j["y"].get<double>(), r, g, b))
            return fail_json("pick failed (no frame or out of range)");
        return ok_json({{"ok", true}, {"r", r}, {"g", g}, {"b", b}});
    });

    // 导入 OBS 场景集合 JSON:提取 yolo-detector-filter 滤镜设置 → 扁平键合并
    srv.route("POST", "/api/config/import_obs", [engine, &ctx](const HttpRequest& req) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(req.body);
        } catch (...) {
            return fail_json("bad json");
        }
        // 收集所有 yolo-detector-filter 的 settings(扁平 obs 键)
        nlohmann::json flat = nlohmann::json::object();
        int found = 0;
        auto collect = [&](const nlohmann::json& node, auto& self) -> void {
            if (!node.is_object())
                return;
            if (node.contains("type") && node["type"].is_string() &&
                node["type"].get<std::string>().find("yolo-detector-filter") !=
                    std::string::npos) {
                if (node.contains("settings") && node["settings"].is_object()) {
                    for (auto it = node["settings"].begin(); it != node["settings"].end();
                         ++it)
                        flat[it.key()] = it.value();
                    ++found;
                }
            }
            if (node.contains("sources") && node["sources"].is_array()) {
                for (auto& s : node["sources"])
                    self(s, self);
            }
            if (node.contains("filters") && node["filters"].is_array()) {
                for (auto& f : node["filters"])
                    self(f, self);
            }
            if (node.contains("scenes") && node["scenes"].is_array()) {
                for (auto& s : node["scenes"])
                    self(s, self);
            }
        };
        collect(j, collect);
        if (found == 0)
            return fail_json("no yolo-detector-filter found in scene JSON");
        // 合并到当前配置
        config::ConfigDocument next = engine->config();
        std::string err;
        if (!config::ConfigStore::merge(next, flat, &err))
            return fail_json(err);
        engine->update_config(next);
        if (!ctx.config_path.empty())
            config::ConfigStore::save_file(ctx.config_path, next, &err);
        return ok_json({{"ok", true}, {"filters_imported", found}});
    });

    srv.route("POST", "/api/controller/test", [engine](const HttpRequest& req) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(req.body);
        } catch (...) {
            return fail_json("bad json");
        }
        const std::string type = j.value("type", "WindowsAPI");
        const std::string port = j.value("makcu_port", "");
        const int baud = j.value("makcu_baud_rate", 0);
        const int logi = j.value("logi_driver_type", 0);
        std::string err;
        const bool ok = engine->test_controller(type, port, baud, logi, &err);
        return ok ? ok_json({{"ok", true}}) : fail_json(err.empty() ? "test failed" : err);
    });
}

} // namespace web
} // namespace ya
