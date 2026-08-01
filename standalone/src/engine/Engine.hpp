#pragma once

// M3 引擎:双线程管线。
//   capture 线程:grab → 最新帧槽(不等待推理)
//   process 线程:消费帧 → FramePipeline(推理→跟踪→瞄准)
// 配置:M2 ConfigDocument(嵌套模型)。

#include "capture/FrameSource.hpp"
#include "config/ConfigModel.hpp"
#include "core/TrackerEngine.hpp"

#include "FramePipeline.hpp"
#include "InferEngine.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace ya {

class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // 打开截图 + 加载模型 + 起双线程。失败返回 false。
    bool start(const config::ConfigDocument& cfg);
    void stop();
    bool running() const { return running_.load(); }

    // 热更新配置(下次循环生效;模型/截图参数需 reload)
    void update_config(const config::ConfigDocument& cfg);

    // 冷重载(下一轮循环执行)
    void request_reload_model();
    void request_reload_capture();

    PipelineStats stats() const;

    // 最近一次检测(预览/Web 用;线程安全拷贝)
    std::vector<Detection> last_detections() const;

private:
    void capture_loop();
    void process_loop();
    bool open_capture();
    bool load_infer();
    void apply_aim_settings();

    std::unique_ptr<FrameSource> capture_;
    std::unique_ptr<InferEngine> infer_;
    std::unique_ptr<TrackerEngine> tracker_;
    std::unique_ptr<FullAimBridge> aim_;
    FramePipeline pipeline_;

    // 配置(双线程共享,热更新时替换)
    std::mutex cfg_mu_;
    config::ConfigDocument cfg_;

    // 最新帧槽
    std::mutex frame_mu_;
    std::condition_variable frame_cv_;
    FramePacket latest_;
    bool has_frame_ = false;

    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> reload_model_{false};
    std::atomic<bool> reload_capture_{false};
    std::thread capture_th_;
    std::thread process_th_;

    // 统计
    mutable std::mutex stats_mu_;
    PipelineStats stats_;
};

} // namespace ya
