#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ya {

/// 远程推理结果（与手机 JSON 反序列化后结构一致）
struct RemoteDet {
    int class_id = 0;
    float confidence = 0.f;
    float x = 0.f; // normalized 0..1
    float y = 0.f;
    float w = 0.f;
    float h = 0.f;
};

struct RemoteResult {
    std::vector<RemoteDet> dets;
    float infer_ms = 0.f;
    int seq = 0;
    std::string error;
};

/// PC 端远程推理 TCP 客户端
/// 发 JPEG → 等 JSON → 取检测
class RemoteInferClient {
public:
    RemoteInferClient();
    ~RemoteInferClient();

    RemoteInferClient(const RemoteInferClient&) = delete;
    RemoteInferClient& operator=(const RemoteInferClient&) = delete;

    /// 连接手机，target = "192.168.1.100:9999"
    bool connect(const std::string &host, int port);
    void disconnect();
    bool is_connected() const;

    /// 发送一帧 BGR 并等待检测结果
    /// timeout_ms: 等待响应超时毫秒
    RemoteResult infer(const uint8_t *bgr, int w, int h,
                       int quality = 60, int timeout_ms = 3000);

    /// 最近一次往返延迟（毫秒）
    double rtt_ms() const { return rtt_ms_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    double rtt_ms_ = 0.0;
};

} // namespace ya
