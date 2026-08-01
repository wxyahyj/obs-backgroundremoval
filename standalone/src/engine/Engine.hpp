#pragma once

#include "capture/FrameSource.hpp"
#include "core/InferEngine.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ya {

// M1 最小闭环引擎:单线程 grab→infer→stats。
// M3 将替换为 FramePipeline 多线程管线,此接口保持不变。
struct EngineConfig {
    CaptureBackend backend = CaptureBackend::Dxgi;
    bool center_region = true;
    int region_x = 0;
    int region_y = 0;
    int width = 640;
    int height = 640;
    InferConfig infer;
};

struct EngineStats {
    double fps = 0.0;
    double grab_ms = 0.0;
    double infer_ms = 0.0;
    uint64_t frames = 0;
    uint64_t detections = 0;
    bool running = false;
    std::string state = "stopped";
    std::string last_error;
};

class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // 打开截图 + 加载模型 + 启动循环线程。失败返回 false,last_error 可查。
    bool start(const EngineConfig& cfg);
    void stop();
    bool running() const { return running_.load(); }
    EngineStats stats() const;

private:
    void loop();
    bool open_capture(const EngineConfig& cfg);

    std::unique_ptr<FrameSource> capture_;
    std::unique_ptr<InferEngine> infer_;
    EngineConfig cfg_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::thread thread_;
    mutable std::mutex stats_mu_;
    EngineStats stats_;
};

} // namespace ya
