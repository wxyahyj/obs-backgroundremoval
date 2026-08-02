#pragma once

// 悬浮窗(M5c 重做)— OBS 风格独立窗口:显示捕获画面 + 检测框 + FOV。
// 置顶、可拖动、可调整?固定尺寸(配置 floating_window_width/height)。
// 实现:WS_EX_LAYERED + UpdateLayeredWindow,GDI 双缓冲。

#include "models/Detection.h"

#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ya {

class OverlayWindow {
public:
    OverlayWindow() = default;
    ~OverlayWindow() { destroy(); }

    OverlayWindow(const OverlayWindow&) = delete;
    OverlayWindow& operator=(const OverlayWindow&) = delete;

    // 创建窗口(尺寸 = 悬浮窗显示尺寸);不显示。
    bool create(int width, int height);
    void show();
    void hide();
    void destroy();

    // 刷新内容:frame_bgr = 捕获帧(可能比窗口大,内部缩放);
    // dets 归一化坐标(相对 frame_w/h);fov_px 帧内像素。
    void update(const std::vector<uint8_t>& frame_bgr, int frame_w, int frame_h,
                const std::vector<Detection>& dets, int fov_px, bool show_fov);

    // 链路延迟(悬浮窗底部文字显示)
    void set_pipeline(double grab_ms, double infer_ms, double track_ms,
                      double aim_ms, double total_ms);

    // 处理窗口消息(main 循环低频调用)
    void pump();

    bool visible() const { return visible_; }

private:
    static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    void render();

    HWND hwnd_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    bool visible_ = false;

    // 显示内容(update 传入)
    std::vector<uint8_t> frame_bgr_;
    int frame_w_ = 0;
    int frame_h_ = 0;
    std::vector<Detection> dets_;
    int fov_px_ = 0;
    bool show_fov_ = true;
    bool dirty_ = true;

    // 链路延迟文字
    double grab_ms_ = 0, infer_ms_ = 0, track_ms_ = 0, aim_ms_ = 0, total_ms_ = 0;
};

} // namespace ya
