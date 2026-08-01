// 透明叠加层实现 — GDI 双缓冲 + UpdateLayeredWindow。

#include "OverlayWindow.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace ya {

namespace {

constexpr int kFps = 30;
constexpr DWORD kUpdateIntervalMs = 1000 / kFps;

// 简易 Bresenham 画线(8-bit RGBA buffer)
void draw_line(std::vector<uint32_t>& px, int w, int h, int x0, int y0, int x1, int y1,
               uint32_t color)
{
    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        if (x0 >= 0 && y0 >= 0 && x0 < w && y0 < h)
            px[static_cast<size_t>(y0) * w + x0] = color;
        if (x0 == x1 && y0 == y1)
            break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

void draw_circle(std::vector<uint32_t>& px, int w, int h, int cx, int cy, int radius,
                 uint32_t color)
{
    if (radius <= 0)
        return;
    int x = radius;
    int y = 0;
    int err = 1 - radius;
    while (x >= y) {
        const int pts[8][2] = {
            {cx + x, cy + y}, {cx - x, cy + y}, {cx + x, cy - y}, {cx - x, cy - y},
            {cx + y, cy + x}, {cx - y, cy + x}, {cx + y, cy - x}, {cx - y, cy - x},
        };
        for (const auto& p : pts) {
            if (p[0] >= 0 && p[1] >= 0 && p[0] < w && p[1] < h)
                px[static_cast<size_t>(p[1]) * w + p[0]] = color;
        }
        if (err < 0) {
            err += 2 * y + 1;
        } else {
            --x;
            err += 2 * (y - x) + 1;
        }
        ++y;
    }
}

} // namespace

LRESULT CALLBACK OverlayWindow::wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg) {
    case WM_NCHITTEST:
        return HTTRANSPARENT; // 点击穿透
    case WM_DESTROY:
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

bool OverlayWindow::create(int width, int height)
{
    if (hwnd_)
        return true;
    width_ = width;
    height_ = height;

    const wchar_t kClass[] = L"YoloAimOverlay";
    HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512)); // IDC_ARROW
    wc.lpszClassName = kClass;
    if (!RegisterClassW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        return false;

    hwnd_ = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW, kClass,
        L"YoloAimOverlay", WS_POPUP, pos_x_, pos_y_, width_, height_, nullptr, nullptr,
        inst, nullptr);
    if (!hwnd_)
        return false;
    return true;
}

void OverlayWindow::set_position(int x, int y)
{
    pos_x_ = x;
    pos_y_ = y;
    if (hwnd_)
        SetWindowPos(hwnd_, HWND_TOPMOST, x, y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
}

void OverlayWindow::show()
{
    if (!hwnd_ || visible_)
        return;
    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
    visible_ = true;
    dirty_ = true;
}

void OverlayWindow::hide()
{
    if (!hwnd_ || !visible_)
        return;
    ShowWindow(hwnd_, SW_HIDE);
    visible_ = false;
}

void OverlayWindow::destroy()
{
    if (hwnd_) {
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
    }
    visible_ = false;
}

void OverlayWindow::update(const std::vector<Detection>& dets, int frame_w, int frame_h,
                           int fov_px, bool show_fov)
{
    dets_ = dets;
    frame_w_ = frame_w;
    frame_h_ = frame_h;
    fov_px_ = fov_px;
    show_fov_ = show_fov;
    dirty_ = true;
}

void OverlayWindow::render()
{
    if (!hwnd_ || !dirty_)
        return;
    dirty_ = false;

    const int w = width_;
    const int h = height_;
    if (w <= 0 || h <= 0)
        return;

    std::vector<uint32_t> px(static_cast<size_t>(w) * h, 0x00000000); // 全透明

    const uint32_t kBox = 0xFF00FF00; // 绿框
    const uint32_t kFov = 0x80FFFFFF; // 半透明白圆

    // FOV 圆(帧内像素坐标)
    if (show_fov_ && fov_px_ > 0 && frame_w_ > 0) {
        const int cx = w / 2;
        const int cy = h / 2;
        draw_circle(px, w, h, cx, cy, fov_px_ * w / frame_w_, kFov);
    }

    // bbox(归一化 → 帧像素)
    if (frame_w_ > 0 && frame_h_ > 0) {
        for (const auto& d : dets_) {
            const int x0 = static_cast<int>(d.x * w);
            const int y0 = static_cast<int>(d.y * h);
            const int x1 = static_cast<int>((d.x + d.width) * w);
            const int y1 = static_cast<int>((d.y + d.height) * h);
            draw_line(px, w, h, x0, y0, x1, y0, kBox);
            draw_line(px, w, h, x1, y0, x1, y1, kBox);
            draw_line(px, w, h, x1, y1, x0, y1, kBox);
            draw_line(px, w, h, x0, y1, x0, y0, kBox);
        }
    }

    // UpdateLayeredWindow 送显
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h; // top-down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    HDC mem_dc = CreateCompatibleDC(nullptr);
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(mem_dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (bmp && bits) {
        std::memcpy(bits, px.data(), px.size() * sizeof(uint32_t));
        HGDIOBJ old = SelectObject(mem_dc, bmp);
        BLENDFUNCTION blend{};
        blend.BlendOp = AC_SRC_OVER;
        blend.SourceConstantAlpha = 255;
        blend.AlphaFormat = AC_SRC_ALPHA;
        POINT src{0, 0};
        SIZE size{w, h};
        POINT dst{pos_x_, pos_y_};
        UpdateLayeredWindow(hwnd_, nullptr, &dst, &size, mem_dc, &src, 0, &blend,
                            ULW_ALPHA);
        SelectObject(mem_dc, old);
        DeleteObject(bmp);
    }
    DeleteDC(mem_dc);
}

void OverlayWindow::pump()
{
    if (!hwnd_)
        return;
    MSG msg;
    while (PeekMessageW(&msg, hwnd_, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    render();
}

} // namespace ya
