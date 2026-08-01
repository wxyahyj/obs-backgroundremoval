#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ya {

/// 简易 PC→手机 远程推理发送端
/// DXGI 截图 → JPEG 编码 → TCP 发送到手机
class RemoteSender {
public:
    RemoteSender();
    ~RemoteSender();

    /// 连接手机推理服务器
    /// target: "192.168.1.100:9999"
    bool connect(const std::string &host, int port);

    /// 断开
    void disconnect();

    bool is_connected() const;

    /// 截取屏幕区域并发送 (发送JPEG到手机)
    /// region: x, y, w, h.  空区域 = 全屏.
    /// 返回检测结果(json已解析)
    struct Result {
        bool ok = false;
        float infer_ms = 0;
        int det_count = 0;
        std::string error;
    };
    Result capture_and_send(int region_w = 0, int region_h = 0,
                            int region_x = 0, int region_y = 0,
                            int quality = 60);

    /// 设置截图回调(用来自己提供截图)
    using CaptureFn = std::function<bool(std::vector<uint8_t> &bgr, int &w, int &h)>;
    void set_capture_callback(CaptureFn fn);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ya
