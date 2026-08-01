// API 路由实现。

#include "ApiRoutes.hpp"

#include "config/ConfigStore.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>

namespace ya {
namespace web {

namespace {

nlohmann::json status_json(const Engine& e)
{
    const PipelineStats s = e.stats();
    nlohmann::json j;
    j["running"] = e.running();
    j["fps"] = s.fps;
    j["frames"] = s.frames;
    j["detections"] = s.detections;
    j["grab_ms"] = s.grab_ms;
    j["infer_ms"] = s.infer_ms;
    j["post_ms"] = s.post_ms;
    j["last_error"] = s.last_error;
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
