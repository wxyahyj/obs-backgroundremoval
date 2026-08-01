// M3 引擎实现 — 双线程管线。

#include "Engine.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

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
    bool ok = false;
    if (c.mode == "region") {
        ok = capture_->open_region(backend, c.region_x, c.region_y, c.region_width,
                                   c.region_height);
    } else if (c.mode == "full") {
        ok = capture_->open_region(backend, 0, 0, c.region_width, c.region_height);
    } else {
        ok = capture_->open_center(backend, c.width, c.height);
    }
    if (!ok) {
        std::fprintf(stderr, "[engine] capture open failed: %s\n",
                     capture_->last_error().c_str());
        capture_.reset();
        return false;
    }
    std::fprintf(stderr, "[engine] capture open: backend=%s %dx%d origin=(%d,%d)\n",
                 FrameSource::backend_name(capture_->backend()), capture_->width(),
                 capture_->height(), capture_->origin_x(), capture_->origin_y());
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
}

void Engine::request_reload_model()
{
    reload_model_.store(true);
}

void Engine::request_reload_capture()
{
    reload_capture_.store(true);
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

void Engine::capture_loop()
{
    FramePacket frame;
    auto last_log = std::chrono::steady_clock::now();
    while (!stop_.load()) {
        // 截图重开:重建 FrameSource(region/backend 变更生效)
        if (reload_capture_.exchange(false)) {
            std::fprintf(stderr, "[engine] reload capture\n");
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
                std::fprintf(stderr, "[engine] capture reopened %dx%d\n",
                             capture_->width(), capture_->height());
            } else {
                std::fprintf(stderr, "[engine] capture reopen FAILED: %s\n",
                             next->last_error().c_str());
            }
        }
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

        frames_since_log++;
        infer_accum += pipeline_.stats().infer_ms;
        {
            std::lock_guard<std::mutex> lock(stats_mu_);
            stats_.frames = pipeline_.stats().frames;
            stats_.detections = pipeline_.stats().detections;
            stats_.last_dets = pipeline_.stats().last_dets;
            stats_.aim_status = pipeline_.stats().aim_status;
            stats_.infer_ms = pipeline_.stats().infer_ms;
            stats_.post_ms = pipeline_.stats().post_ms;
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
