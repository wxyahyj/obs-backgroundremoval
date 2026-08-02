#pragma once

// M3 帧管线:单帧处理状态机(推理 → 跟踪 → 瞄准 → 统计)。
// 在 infer 线程上执行;捕获线程只推帧。

#include "capture/FrameSource.hpp"
#include "config/ConfigModel.hpp"

#include "FullAimBridge.hpp"
#include "InferEngine.hpp"
#include "TrackerEngine.hpp"

#include <memory>

namespace ya {

// 分阶段延迟统计(线程安全由调用方保证:单写者)
struct PipelineStats {
    double fps = 0.0;
    double grab_ms = 0.0;
    double infer_ms = 0.0;
    double track_ms = 0.0;
    double aim_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;
    uint64_t frames = 0;
    uint64_t detections = 0;
    std::vector<Detection> last_dets; // 最近一次跟踪结果(预览/叠加用)
    FullAimStatus aim_status;
    std::string last_error;
};

// ConfigDocument.aim → FullAimSettings(主仓瞄准设置结构)
FullAimSettings aim_settings_from_config(const config::AimSection& a);

class FramePipeline {
public:
    FramePipeline();

    void attach(InferEngine* infer, TrackerEngine* tracker, FullAimBridge* aim);

    // 处理一帧;内部累积 stats。
    void process(const FramePacket& frame);

    PipelineStats stats() const { return stats_; }
    void reset_stats();

private:
    InferEngine* infer_ = nullptr;
    TrackerEngine* tracker_ = nullptr;
    FullAimBridge* aim_ = nullptr;
    PipelineStats stats_;
};

} // namespace ya
