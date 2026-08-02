// 悬浮窗实现 — 显示捕获画面 + 检测框 + FOV(GDI 双缓冲 + UpdateLayeredWindow)。

#include "OverlayWindow.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace ya {

namespace {

constexpr int kFps = 30;
constexpr DWORD kUpdateIntervalMs = 1000 / kFps;

// 简易 Bresenham 画线(8-bit BGRA buffer,预乘 alpha)
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
            {cx + y, cy + x}, {cx - y, cy + x}, {cx + y, cy - y}, {cx - y, cy - y},
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

    const wchar_t kClass[] = L"YoloAimFloating";
    HINSTANCE inst = GetModuleHandleW(nullptr);
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512)); // IDC_ARROW
    wc.lpszClassName = kClass;
    if (!RegisterClassW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        return false;

    // 置顶工具窗 + 可拖动;初始放屏幕右上角
    const int sw = GetSystemMetrics(SM_CXSCREEN);
    int x = sw - width - 16;
    int y = 64;
    hwnd_ = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW, kClass, L"YoloAim 悬浮窗",
        WS_POPUP | WS_CAPTION | WS_SYSMENU, x, y, width_, height_, nullptr, nullptr,
        inst, nullptr);
    if (!hwnd_)
        return false;
    return true;
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

void OverlayWindow::update(const std::vector<uint8_t>& frame_bgr, int frame_w,
                           int frame_h, const std::vector<Detection>& dets, int fov_px,
                           bool show_fov)
{
    frame_bgr_ = frame_bgr;
    frame_w_ = frame_w;
    frame_h_ = frame_h;
    dets_ = dets;
    fov_px_ = fov_px;
    show_fov_ = show_fov;
    dirty_ = true;
}

void OverlayWindow::set_pipeline(double grab_ms, double infer_ms, double track_ms,
                                 double aim_ms, double total_ms)
{
    grab_ms_ = grab_ms;
    infer_ms_ = infer_ms;
    track_ms_ = track_ms;
    aim_ms_ = aim_ms;
    total_ms_ = total_ms;
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

    std::vector<uint32_t> px(static_cast<size_t>(w) * h, 0xFF000000); // 黑底不透明

    // 1. 捕获画面(最近邻缩放 BGR → BGRA)
    if (frame_w_ > 0 && frame_h_ > 0 && frame_bgr_.size() >=
                                             static_cast<size_t>(frame_w_) * frame_h_ * 3) {
        for (int dy = 0; dy < h; ++dy) {
            const int sy = std::min(frame_h_ - 1, dy * frame_h_ / h);
            const uint8_t* row = frame_bgr_.data() + static_cast<size_t>(sy) * frame_w_ * 3;
            uint32_t* out = px.data() + static_cast<size_t>(dy) * w;
            for (int dx = 0; dx < w; ++dx) {
                const int sx = std::min(frame_w_ - 1, dx * frame_w_ / w);
                const uint8_t* p = row + static_cast<size_t>(sx) * 3;
                // BGR(p[0]=B,p[1]=G,p[2]=R) → BGRA uint32 小端:byte0=B,byte1=G,byte2=R
                out[dx] = 0xFF000000u | (static_cast<uint32_t>(p[2]) << 16) |
                          (static_cast<uint32_t>(p[1]) << 8) | p[0]; // BGRA
            }
        }
    }

    const uint32_t kBox = 0xFF00FF00; // 绿框(不透明)
    const uint32_t kFov = 0x80FFFFFF; // 半透明白圆

    // 2. FOV 圆(帧坐标 → 窗口坐标缩放)
    if (show_fov_ && fov_px_ > 0 && frame_w_ > 0 && frame_h_ > 0) {
        const int cx = w / 2;
        const int cy = h / 2;
        const int radius = std::max(1, fov_px_ * w / frame_w_);
        draw_circle(px, w, h, cx, cy, radius, kFov);
    }

    // 3. 检测框(归一化 → 窗口像素)
    if (frame_w_ > 0 && frame_h_ > 0) {
        for (const auto& d : dets_) {
            if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.width) ||
                !std::isfinite(d.height))
                continue;
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

    // UpdateLayeredWindow 送显(窗口客户区坐标)
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

        // 链路延迟文字(左下角)
        wchar_t buf[160];
        std::swprintf(buf, 160,
                      L"截图 %.1fms \u2192 推理 %.2fms \u2192 跟踪 %.2fms \u2192 瞄准 %.2fms \u2192 总计 %.1fms",
                      grab_ms_, infer_ms_, track_ms_, aim_ms_, total_ms_);
        SetBkMode(mem_dc, TRANSPARENT);
        SetTextColor(mem_dc, RGB(0, 255, 0));
        HFONT font = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                 DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                 CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                 DEFAULT_PITCH, L"Consolas");
        HGDIOBJ oldFont = SelectObject(mem_dc, font);
        RECT tr{4, h - 24, w - 4, h - 4};
        DrawTextW(mem_dc, buf, -1, &tr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        SelectObject(mem_dc, oldFont);
        DeleteObject(font);

        BLENDFUNCTION blend{};
        blend.BlendOp = AC_SRC_OVER;
        blend.SourceConstantAlpha = 255;
        blend.AlphaFormat = AC_SRC_ALPHA;
        POINT src{0, 0};
        SIZE size{w, h};
        POINT dst{0, 0}; // 客户区坐标(UpdateLayeredWindow 用窗口坐标)
        // UpdateLayeredWindow 需要屏幕坐标
        RECT rc{};
        GetWindowRect(hwnd_, &rc);
        dst.x = rc.left;
        dst.y = rc.top;
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
