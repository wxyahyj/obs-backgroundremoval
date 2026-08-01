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

Engine::Engine() = default;

Engine::~Engine()
{
    stop();
}

bool Engine::open_capture(const EngineConfig& cfg)
{
    capture_ = std::make_unique<FrameSource>();
    bool ok = false;
    if (cfg.center_region) {
        ok = capture_->open_center(cfg.backend, cfg.width, cfg.height);
    } else {
        ok = capture_->open_region(cfg.backend, cfg.region_x, cfg.region_y, cfg.width,
                                   cfg.height);
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

bool Engine::start(const EngineConfig& cfg)
{
    if (running_.load())
        return false;
    cfg_ = cfg;

    infer_ = std::make_unique<InferEngine>();
    if (!infer_->load(cfg_.infer)) {
        std::fprintf(stderr, "[engine] infer load failed: %s\n",
                     infer_->last_error().c_str());
        infer_.reset();
        return false;
    }

    if (!open_capture(cfg_)) {
        infer_.reset();
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(stats_mu_);
        stats_ = EngineStats();
        stats_.running = true;
        stats_.state = "running";
    }
    stop_.store(false);
    running_.store(true);
    thread_ = std::thread(&Engine::loop, this);
    return true;
}

void Engine::stop()
{
    if (!running_.load())
        return;
    stop_.store(true);
    if (thread_.joinable())
        thread_.join();
    running_.store(false);
    {
        std::lock_guard<std::mutex> lock(stats_mu_);
        stats_.running = false;
        stats_.state = "stopped";
    }
    infer_.reset();
    capture_.reset();
}

EngineStats Engine::stats() const
{
    std::lock_guard<std::mutex> lock(stats_mu_);
    return stats_;
}

void Engine::loop()
{
    FramePacket frame;
    auto last_log = std::chrono::steady_clock::now();
    uint64_t frames = 0;
    double grab_ms_acc = 0.0;
    double infer_ms_acc = 0.0;

    while (!stop_.load()) {
        const auto t_grab0 = std::chrono::steady_clock::now();
        const bool got = capture_->grab(frame);
        const auto t_grab1 = std::chrono::steady_clock::now();

        if (!got) {
            // 捕获失败:低频重试,避免忙转
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (frame.width <= 0 || frame.height <= 0 || frame.bgr.empty())
            continue;

        InferResult r;
        if (infer_) {
            const auto t_inf0 = std::chrono::steady_clock::now();
            r = infer_->run_bgr(frame.bgr.data(), frame.width, frame.height,
                                frame.width * frame.channels);
            const auto t_inf1 = std::chrono::steady_clock::now();
            infer_ms_acc += std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count();
        }
        grab_ms_acc += std::chrono::duration<double, std::milli>(t_grab1 - t_grab0).count();
        ++frames;

        {
            std::lock_guard<std::mutex> lock(stats_mu_);
            stats_.frames = frames;
            stats_.detections = r.dets.size();
            stats_.grab_ms = grab_ms_acc / frames;
            stats_.infer_ms = infer_ms_acc / frames;
        }

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() >= 2) {
            const double elapsed = std::chrono::duration<double>(now - last_log).count();
            const double fps = elapsed > 0.0 ? frames / elapsed : 0.0;
            {
                std::lock_guard<std::mutex> lock(stats_mu_);
                stats_.fps = fps;
            }
            std::fprintf(stderr,
                         "[engine] fps=%.1f frame=%llu dets=%zu grab=%.2fms infer=%.2fms %s\n",
                         fps, static_cast<unsigned long long>(frames), r.dets.size(),
                         frames > 0 ? grab_ms_acc / frames : 0.0,
                         frames > 0 ? infer_ms_acc / frames : 0.0,
                         r.error.empty() ? "" : r.error.c_str());
            frames = 0;
            grab_ms_acc = infer_ms_acc = 0.0;
            last_log = now;
        }
    }

    {
        std::lock_guard<std::mutex> lock(stats_mu_);
        stats_.running = false;
        stats_.state = "stopped";
    }
}

} // namespace ya
