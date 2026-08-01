#pragma once

// 游戏内叠加层(M5c)— 透明置顶窗口,绘制 bbox/FOV。
// 实现:WS_EX_LAYERED + UpdateLayeredWindow,GDI 双缓冲,点击穿透。

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

    // 创建窗口(尺寸 = 捕获区域);不显示。
    bool create(int width, int height);
    void set_position(int x, int y); // 屏幕坐标(捕获区域 origin)
    void show();
    void hide();
    void destroy();

    // 刷新内容;dets 归一化坐标,frame_w/h 捕获尺寸。
    void update(const std::vector<Detection>& dets, int frame_w, int frame_h, int fov_px,
                bool show_fov);

    // 处理窗口消息(main 循环低频调用)
    void pump();

    bool visible() const { return visible_; }

private:
    static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
    void render();

    HWND hwnd_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int pos_x_ = 0;
    int pos_y_ = 0;
    bool visible_ = false;

    std::vector<Detection> dets_;
    int frame_w_ = 0;
    int frame_h_ = 0;
    int fov_px_ = 0;
    bool show_fov_ = true;
    bool dirty_ = true;
};

} // namespace ya
