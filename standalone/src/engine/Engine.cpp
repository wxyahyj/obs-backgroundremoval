// M3 引擎实现 — 双线程管线。

#include "Engine.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ya {

namespace {

TrackerConfig tracker_config_from(const config::TrackerSection& t)
{
    TrackerConfig c;
    c.iou_threshold = t.iou_threshold;
    c.max_lost_frames = t.max_lost_frames;
    c.max_reidentify_frames = t.max_reidentify_frames;
    c.reidentify_center_threshold = t.reidentify_center_threshold;
    c.weight_iou = t.weight_iou;
    c.weight_center = t.weight_center;
    c.weight_aspect = t.weight_aspect;
    c.weight_area = t.weight_area;
    c.use_kalman = t.use_kalman;
    c.kalman_generate_threshold = t.kalman_generate_threshold;
    c.kalman_terminate_count = t.kalman_terminate_count;
    c.kalman_prediction_frames = t.kalman_prediction_frames;
    c.show_kalman_predictions = t.show_kalman_predictions;
    c.show_kalman_trajectories = t.show_kalman_trajectories;
    return c;
}

InferConfig infer_config_from(const config::InferSection& i)
{
    InferConfig c;
    c.model_path = i.model_path;
    c.device = i.device;
    c.model_version = i.model_version;
    c.confidence = i.confidence;
    c.nms = i.nms;
    c.input_resolution = i.input_resolution;
    c.num_threads = i.num_threads;
    c.interval_frames = i.interval_frames;
    c.target_classes = i.target_classes;
    return c;
}

} // namespace

Engine::Engine() = default;

Engine::~Engine()
{
    stop();
}

bool Engine::open_capture()
{
    config::CaptureSection c;
    {
        std::lock_guard<std::mutex> lock(cfg_mu_);
        c = cfg_.capture;
    }
    capture_ = std::make_unique<FrameSource>();
    const CaptureBackend backend = FrameSource::parse_backend(c.backend);

    auto try_open = [&](CaptureBackend b) -> bool {
        if (c.mode == "region")
            return capture_->open_region(b, c.region_x, c.region_y, c.region_width,
                                         c.region_height);
        if (c.mode == "full")
            return capture_->open_region(b, 0, 0, c.region_width, c.region_height);
        return capture_->open_center(b, c.width, c.height);
    };

    if (!try_open(backend)) {
        // 回退:DXGI/WGC 失败 → GDI(兼容远程桌面/低端环境)
        const std::string first_err = capture_->last_error();
        if (backend != CaptureBackend::Gdi && try_open(CaptureBackend::Gdi)) {
            std::fprintf(stderr,
                         "[engine] capture fallback dxgi->gdi (%s)\n", first_err.c_str());
        } else {
            std::fprintf(stderr, "[engine] capture open failed: %s\n",
                         first_err.c_str());
            capture_.reset();
            return false;
        }
    }
    std::fprintf(stderr, "[engine] capture open: backend=%s %dx%d origin=(%d,%d)\n",
                 FrameSource::backend_name(capture_->backend()), capture_->width(),
                 capture_->height(), capture_->origin_x(), capture_->origin_y());
    cap_width_.store(capture_->width());
    cap_height_.store(capture_->height());
    cap_origin_x_.store(capture_->origin_x());
    cap_origin_y_.store(capture_->origin_y());
    return true;
}

bool Engine::load_infer()
{
    InferConfig ic;
    {
        std::lock_guard<std::mutex> lock(cfg_mu_);
        ic = infer_config_from(cfg_.infer);
    }
    infer_ = std::make_unique<InferEngine>();
    if (!infer_->load(ic)) {
        std::fprintf(stderr, "[engine] infer load failed: %s\n",
                     infer_->last_error().c_str());
        infer_.reset();
        return false;
    }
    return true;
}

void Engine::apply_aim_settings()
{
    config::AimSection a;
    {
        std::lock_guard<std::mutex> lock(cfg_mu_);
        a = cfg_.aim;
    }
    // 收敛:固定使用自适应PID(旧配置/导入值可能仍是 AdvancedPID 等)
    a.algorithm = AlgorithmType::AdaptivePID;
    if (aim_) {
        aim_->set_settings(aim_settings_from_config(a));
        aim_->ensure_controller();
    }
    if (tracker_) {
        config::TrackerSection t;
        {
            std::lock_guard<std::mutex> lock(cfg_mu_);
            t = cfg_.tracker;
        }
        tracker_->set_config(tracker_config_from(t));
    }
}

bool Engine::start(const config::ConfigDocument& cfg)
{
    if (running_.load())
        return false;
    {
        std::lock_guard<std::mutex> lock(cfg_mu_);
        cfg_ = cfg;
    }

    if (!open_capture())
        return false;
    if (!load_infer())
        return false;

    tracker_ = std::make_unique<TrackerEngine>();
    aim_ = std::make_unique<FullAimBridge>();
    apply_aim_settings();
    pipeline_.attach(infer_.get(), tracker_.get(), aim_.get());

    stop_.store(false);
    running_.store(true);
    capture_th_ = std::thread(&Engine::capture_loop, this);
    process_th_ = std::thread(&Engine::process_loop, this);
    return true;
}

void Engine::stop()
{
    if (!running_.load())
        return;
    stop_.store(true);
    frame_cv_.notify_all();
    if (capture_th_.joinable())
        capture_th_.join();
    if (process_th_.joinable())
        process_th_.join();
    running_.store(false);
    infer_.reset();
    capture_.reset();
    tracker_.reset();
    aim_.reset();
}

void Engine::update_config(const config::ConfigDocument& cfg)
{
    {
        std::lock_guard<std::mutex> lock(cfg_mu_);
        cfg_ = cfg;
    }
    apply_aim_settings(); // 瞄准/跟踪参数热生效;模型/截图走 reload
    // 推理阈值/目标类别热生效(无需 reload_model)
    if (infer_) {
        infer_->set_thresholds(cfg.infer.confidence, cfg.infer.nms);
        infer_->set_target_classes(cfg.infer.target_classes);
    }
}

config::ConfigDocument Engine::config() const
{
    std::lock_guard<std::mutex> lock(cfg_mu_);
    return cfg_;
}

void Engine::request_reload_model()
{
    reload_model_.store(true);
}

void Engine::request_reload_capture()
{
    reload_capture_.store(true);
}

bool Engine::test_controller(const std::string& type, const std::string& makcu_port, int baud,
                             int logi_type, std::string* err)
{
    if (!aim_)
        return false;
    return aim_->test_controller(FullAimBridge::parse_controller(type), makcu_port, baud,
                                 logi_type, err);
}

PipelineStats Engine::stats() const
{
    std::lock_guard<std::mutex> lock(stats_mu_);
    return stats_;
}

std::vector<Detection> Engine::last_detections() const
{
    std::lock_guard<std::mutex> lock(stats_mu_);
    return stats_.last_dets;
}

std::vector<uint8_t> Engine::preview_bmp() const
{
    PreviewFrame f;
    {
        std::lock_guard<std::mutex> lock(preview_mu_);
        if (preview_.bgr.empty())
            return {};
        f = preview_;
    }
    cv::Mat img(f.height, f.width, CV_8UC3, const_cast<uint8_t*>(f.bgr.data()),
                static_cast<size_t>(f.width) * 3);
    cv::Mat out = img.clone();
    for (const auto& d : f.dets) {
        // 防御:NaN/Inf/越界坐标 → 跳过(异常框画图可崩)
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.width) ||
            !std::isfinite(d.height))
            continue;
        cv::Rect box = d.getPixelBBox(f.width, f.height);
        if (box.width <= 0 || box.height <= 0)
            continue;
        box.x = std::max(0, std::min(box.x, f.width - 1));
        box.y = std::max(0, std::min(box.y, f.height - 1));
        box.width = std::max(1, std::min(box.width, f.width - box.x));
        box.height = std::max(1, std::min(box.height, f.height - box.y));
        cv::rectangle(out, box, cv::Scalar(0, 255, 0), 2);
        char label[64];
        std::snprintf(label, sizeof(label), "%s %.0f%%", d.className.c_str(),
                      d.confidence * 100.f);
        cv::putText(out, label,
                    cv::Point(box.x, box.y > 14 ? box.y - 4 : box.y + 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1,
                    cv::LINE_AA);
    }
    std::vector<uint8_t> buf;
    if (!cv::imencode(".bmp", out, buf))
        return {};
    return buf;
}

int Engine::num_classes() const
{
    if (!infer_)
        return 0;
    return infer_->num_classes();
}

Engine::CaptureInfo Engine::capture_info() const
{
    CaptureInfo info;
    info.width = cap_width_.load();
    info.height = cap_height_.load();
    info.origin_x = cap_origin_x_.load();
    info.origin_y = cap_origin_y_.load();
    return info;
}

Engine::PreviewFrameData Engine::preview_frame() const
{
    PreviewFrameData out;
    std::lock_guard<std::mutex> lock(preview_mu_);
    out.bgr = preview_.bgr;
    out.width = preview_.width;
    out.height = preview_.height;
    return out;
}

bool Engine::pick_color(double nx, double ny, int& r, int& g, int& b) const
{
    PreviewFrame f;
    {
        std::lock_guard<std::mutex> lock(preview_mu_);
        if (preview_.bgr.empty() || preview_.width <= 0 || preview_.height <= 0)
            return false;
        f = preview_;
    }
    if (!std::isfinite(nx) || !std::isfinite(ny))
        return false;
    const int cx = static_cast<int>(nx * f.width);
    const int cy = static_cast<int>(ny * f.height);
    if (cx < 0 || cy < 0 || cx >= f.width || cy >= f.height)
        return false;

    // 5x5 邻域均值(抗单点噪)
    const int radius = 2;
    long sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const int px = cx + dx;
            const int py = cy + dy;
            if (px < 0 || py < 0 || px >= f.width || py >= f.height)
                continue;
            const size_t idx = (static_cast<size_t>(py) * f.width + px) * 3;
            // BGR 存储
            sum_b += f.bgr[idx + 0];
            sum_g += f.bgr[idx + 1];
            sum_r += f.bgr[idx + 2];
            ++count;
        }
    }
    if (count == 0)
        return false;
    r = static_cast<int>(sum_r / count);
    g = static_cast<int>(sum_g / count);
    b = static_cast<int>(sum_b / count);
    return true;
}

void Engine::capture_loop()
{
    FramePacket frame;
    auto last_log = std::chrono::steady_clock::now();
    bool cap_reopen_pending = false;
    int max_fps = 60;
    auto last_grab = std::chrono::steady_clock::now();
    while (!stop_.load()) {
        // 限帧:max_fps > 0 时按帧间隔节流(降低 DXGI+推理 GPU 争用)
        {
            std::lock_guard<std::mutex> lock(cfg_mu_);
            max_fps = cfg_.capture.max_fps;
        }
        if (max_fps > 0) {
            const auto now = std::chrono::steady_clock::now();
            const double interval_ms = 1000.0 / max_fps;
            const double elapsed =
                std::chrono::duration<double, std::milli>(now - last_grab).count();
            if (elapsed < interval_ms) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(static_cast<int>(interval_ms - elapsed)));
            }
        }
        // 截图重开:先释放旧 DXGI(同一输出同时只能有一个 desktop duplication),
        // 再重建 FrameSource(region/backend 变更生效)
        if (reload_capture_.exchange(false) || cap_reopen_pending) {
            cap_reopen_pending = false;
            std::fprintf(stderr, "[engine] reload capture\n");
            capture_.reset(); // 释放旧 duplication → 新的才能 DuplicateOutput
            auto next = std::make_unique<FrameSource>();
            config::CaptureSection c;
            {
                std::lock_guard<std::mutex> lock(cfg_mu_);
                c = cfg_.capture;
            }
            const CaptureBackend backend = FrameSource::parse_backend(c.backend);
            bool ok = false;
            if (c.mode == "region") {
                ok = next->open_region(backend, c.region_x, c.region_y, c.region_width,
                                       c.region_height);
            } else if (c.mode == "full") {
                ok = next->open_region(backend, 0, 0, c.region_width, c.region_height);
            } else {
                ok = next->open_center(backend, c.width, c.height);
            }
            if (ok) {
                capture_ = std::move(next);
                cap_width_.store(capture_->width());
                cap_height_.store(capture_->height());
                cap_origin_x_.store(capture_->origin_x());
                cap_origin_y_.store(capture_->origin_y());
                std::fprintf(stderr, "[engine] capture reopened %dx%d\n",
                             capture_->width(), capture_->height());
            } else {
                std::fprintf(stderr, "[engine] capture reopen FAILED: %s (retry)\n",
                             next->last_error().c_str());
                cap_reopen_pending = true;
            }
        }
        if (!capture_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }
        last_grab = std::chrono::steady_clock::now();
        const auto t0 = std::chrono::steady_clock::now();
        const bool got = capture_->grab(frame);
        const auto t1 = std::chrono::steady_clock::now();
        if (!got) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (frame.width <= 0 || frame.height <= 0 || frame.bgr.empty())
            continue;

        {
            std::lock_guard<std::mutex> lock(stats_mu_);
            stats_.grab_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        {
            std::lock_guard<std::mutex> lock(frame_mu_);
            latest_ = std::move(frame);
            has_frame_ = true;
        }
        frame_cv_.notify_one();

        // 低频状态日志
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= 5) {
            PipelineStats s = stats();
            std::fprintf(stderr,
                         "[engine] fps=%.1f frames=%llu dets=%zu grab=%.2fms infer=%.2fms post=%.2fms slot=%d aim=%d\n",
                         s.fps, static_cast<unsigned long long>(s.frames), s.detections,
                         s.grab_ms, s.infer_ms, s.post_ms, s.aim_status.active_slot,
                         s.aim_status.aiming ? 1 : 0);
            last_log = now;
        }
    }
}

void Engine::process_loop()
{
    auto last_log = std::chrono::steady_clock::now();
    uint64_t frames_since_log = 0;
    double infer_accum = 0.0;

    while (!stop_.load()) {
        if (reload_model_.exchange(false)) {
            model_loading_.store(true);
            std::fprintf(stderr, "[engine] reload model\n");
            InferConfig ic;
            {
                std::lock_guard<std::mutex> lock(cfg_mu_);
                ic = infer_config_from(cfg_.infer);
            }
            auto next = std::make_unique<InferEngine>();
            if (next->load(ic)) {
                infer_ = std::move(next);
                pipeline_.attach(infer_.get(), tracker_.get(), aim_.get());
                std::fprintf(stderr, "[engine] model reloaded\n");
            } else {
                std::fprintf(stderr, "[engine] model reload FAILED: %s\n",
                             next->last_error().c_str());
            }
            model_loading_.store(false);
        }

        // 取最新帧
        FramePacket frame;
        {
            std::unique_lock<std::mutex> lock(frame_mu_);
            frame_cv_.wait_for(lock, std::chrono::milliseconds(100),
                               [this] { return has_frame_ || stop_.load(); });
            if (stop_.load())
                break;
            if (!has_frame_)
                continue;
            frame = std::move(latest_);
            has_frame_ = false;
        }

        const auto t0 = std::chrono::steady_clock::now();
        pipeline_.process(frame);
        const auto t1 = std::chrono::steady_clock::now();

        // 预览缓存(拷贝 BGR + 框)
        {
            std::lock_guard<std::mutex> lock(preview_mu_);
            preview_.bgr = frame.bgr;
            preview_.width = frame.width;
            preview_.height = frame.height;
            preview_.dets = pipeline_.stats().last_dets;
        }

        frames_since_log++;
        infer_accum += pipeline_.stats().infer_ms;
        {
            std::lock_guard<std::mutex> lock(stats_mu_);
            stats_.frames = pipeline_.stats().frames;
            stats_.detections = pipeline_.stats().detections;
            stats_.last_dets = pipeline_.stats().last_dets;
            stats_.aim_status = pipeline_.stats().aim_status;
            stats_.infer_ms = pipeline_.stats().infer_ms;
            stats_.track_ms = pipeline_.stats().track_ms;
            stats_.aim_ms = pipeline_.stats().aim_ms;
            stats_.total_ms = pipeline_.stats().total_ms;
            stats_.post_ms = pipeline_.stats().post_ms;
        }

        // 坐标导出(低频,约 1s 一次)
        if (frames_since_log % 60 == 0) {
            bool do_export = false;
            std::string out_path;
            {
                std::lock_guard<std::mutex> lock(cfg_mu_);
                do_export = cfg_.vision.export_coordinates;
                out_path = cfg_.vision.coordinate_output_path;
            }
            if (do_export) {
                const std::vector<Detection> dets = pipeline_.stats().last_dets;
                nlohmann::json arr = nlohmann::json::array();
                for (const auto& d : dets) {
                    arr.push_back({
                        {"class_id", d.classId},
                        {"confidence", d.confidence},
                        {"x", d.x},
                        {"y", d.y},
                        {"width", d.width},
                        {"height", d.height},
                        {"track_id", d.trackId},
                    });
                }
                const std::string path = out_path.empty()
                                             ? "detections.json"
                                             : out_path;
                std::ofstream f(path, std::ios::trunc);
                if (f.is_open())
                    f << (nlohmann::json{{"timestamp", time(nullptr)}, {"detections", arr}})
                             .dump(1);
            }
        }

        // FPS 按实际消费率统计
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= 2) {
            const double elapsed = std::chrono::duration<double>(now - last_log).count();
            {
                std::lock_guard<std::mutex> lock(stats_mu_);
                stats_.fps = elapsed > 0 ? frames_since_log / elapsed : 0.0;
            }
            frames_since_log = 0;
            infer_accum = 0.0;
            last_log = now;
        }
        (void)t0;
        (void)t1;
    }
}

} // namespace ya
