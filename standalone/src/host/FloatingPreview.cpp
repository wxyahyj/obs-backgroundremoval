#include "FloatingPreview.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ya {

struct FloatingPreviewAccess {
	static std::mutex &mu(FloatingPreview *s) { return s->mu_; }
	static FloatingPreview::Settings &cfg(FloatingPreview *s) { return s->cfg_; }
	static std::vector<uint8_t> &bgra(FloatingPreview *s) { return s->bgra_; }
	static int &frame_w(FloatingPreview *s) { return s->frame_w_; }
	static int &frame_h(FloatingPreview *s) { return s->frame_h_; }
	static std::string &stats(FloatingPreview *s) { return s->stats_; }
	static void *&hwnd(FloatingPreview *s) { return s->hwnd_; }
	static bool &drag(FloatingPreview *s) { return s->drag_; }
	static int &drag_dx(FloatingPreview *s) { return s->drag_dx_; }
	static int &drag_dy(FloatingPreview *s) { return s->drag_dy_; }
	static bool &class_registered(FloatingPreview *s) { return s->class_registered_; }
	static bool &user_closed(FloatingPreview *s) { return s->user_closed_; }
};

#ifdef _WIN32
namespace {
const wchar_t *kClassName = L"YoloAimFloatingPreview";

struct CreateCtx {
	FloatingPreview *self = nullptr;
};
} // namespace

LRESULT CALLBACK FloatingPreviewProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	auto *self = reinterpret_cast<FloatingPreview *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
	switch (msg) {
	case WM_NCCREATE: {
		auto *cs = reinterpret_cast<CREATESTRUCTW *>(lParam);
		auto *ctx = static_cast<CreateCtx *>(cs->lpCreateParams);
		if (ctx && ctx->self) {
			SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(ctx->self));
			self = ctx->self;
		}
		return TRUE;
	}
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);
		if (self) {
			// Lock only to snapshot dimensions and get pointer — no O(n) vector copy
			int fw = 0, fh = 0;
			std::string stats;
			const uint8_t *src = nullptr;
			{
				std::lock_guard<std::mutex> lock(FloatingPreviewAccess::mu(self));
				fw = FloatingPreviewAccess::frame_w(self);
				fh = FloatingPreviewAccess::frame_h(self);
				stats = FloatingPreviewAccess::stats(self);
				const auto &b = FloatingPreviewAccess::bgra(self);
				if (!b.empty() && fw > 0 && fh > 0 &&
				    b.size() >= static_cast<size_t>(fw) * static_cast<size_t>(fh) * 4u) {
					src = b.data();
				}
			}
			RECT rc{};
			GetClientRect(hwnd, &rc);
			const int cw = std::max(1, static_cast<int>(rc.right - rc.left));
			const int ch = std::max(1, static_cast<int>(rc.bottom - rc.top));
			HBRUSH br = CreateSolidBrush(RGB(8, 10, 14));
			FillRect(hdc, &rc, br);
			DeleteObject(br);
			if (src && fw > 0 && fh > 0) {
				BITMAPINFO bmi{};
				bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bmi.bmiHeader.biWidth = fw;
				bmi.bmiHeader.biHeight = -fh;
				bmi.bmiHeader.biPlanes = 1;
				bmi.bmiHeader.biBitCount = 32;
				bmi.bmiHeader.biCompression = BI_RGB;
				const float sx = static_cast<float>(cw) / static_cast<float>(fw);
				const float sy = static_cast<float>(ch) / static_cast<float>(fh);
				const float s = std::min(sx, sy);
				const int dw = std::max(1, static_cast<int>(fw * s));
				const int dh = std::max(1, static_cast<int>(fh * s));
				const int ox = (cw - dw) / 2;
				const int oy = (ch - dh) / 2;
				SetStretchBltMode(hdc, COLORONCOLOR);
				StretchDIBits(hdc, ox, oy, dw, dh, 0, 0, fw, fh, src, &bmi,
				              DIB_RGB_COLORS, SRCCOPY);
			}
			if (!stats.empty()) {
				SetBkMode(hdc, TRANSPARENT);
				SetTextColor(hdc, RGB(230, 240, 255));
				RECT tr{8, 8, cw - 8, 48};
				DrawTextA(hdc, stats.c_str(), -1, &tr, DT_LEFT | DT_TOP | DT_WORDBREAK);
			}
		}
		EndPaint(hwnd, &ps);
		return 0;
	}
	case WM_LBUTTONDOWN: {
		if (!self)
			break;
		FloatingPreviewAccess::drag(self) = true;
		POINT pt{};
		GetCursorPos(&pt);
		RECT rect{};
		GetWindowRect(hwnd, &rect);
		FloatingPreviewAccess::drag_dx(self) = pt.x - rect.left;
		FloatingPreviewAccess::drag_dy(self) = pt.y - rect.top;
		SetCapture(hwnd);
		return 0;
	}
	case WM_LBUTTONUP: {
		if (self) {
			FloatingPreviewAccess::drag(self) = false;
			ReleaseCapture();
		}
		return 0;
	}
	case WM_MOUSEMOVE: {
		if (self && FloatingPreviewAccess::drag(self)) {
			POINT pt{};
			GetCursorPos(&pt);
			SetWindowPos(hwnd, nullptr, pt.x - FloatingPreviewAccess::drag_dx(self),
			             pt.y - FloatingPreviewAccess::drag_dy(self), 0, 0,
			             SWP_NOSIZE | SWP_NOZORDER);
		}
		return 0;
	}
	case WM_CLOSE: {
		// User closed window → flag for engine to clear config (avoid re-open loop).
		if (self) {
			std::lock_guard<std::mutex> lock(FloatingPreviewAccess::mu(self));
			FloatingPreviewAccess::cfg(self).enabled = false;
			FloatingPreviewAccess::user_closed(self) = true;
			FloatingPreviewAccess::drag(self) = false;
		}
		ReleaseCapture();
		DestroyWindow(hwnd);
		return 0;
	}
	case WM_DESTROY: {
		if (self) {
			std::lock_guard<std::mutex> lock(FloatingPreviewAccess::mu(self));
			FloatingPreviewAccess::drag(self) = false;
			if (FloatingPreviewAccess::hwnd(self) == hwnd)
				FloatingPreviewAccess::hwnd(self) = nullptr;
		}
		return 0;
	}
	default:
		break;
	}
	return DefWindowProcW(hwnd, msg, wParam, lParam);
}
#endif

FloatingPreview::FloatingPreview() = default;

FloatingPreview::~FloatingPreview() { destroy_window(); }

void FloatingPreview::set_settings(const Settings &s)
{
	std::lock_guard<std::mutex> lock(mu_);
	cfg_ = s;
	cfg_.width = std::clamp(cfg_.width, 160, 1920);
	cfg_.height = std::clamp(cfg_.height, 120, 1080);
}

FloatingPreview::Settings FloatingPreview::settings() const
{
	std::lock_guard<std::mutex> lock(mu_);
	return cfg_;
}

bool FloatingPreview::is_open() const
{
	std::lock_guard<std::mutex> lock(mu_);
	return hwnd_ != nullptr && cfg_.enabled;
}

bool FloatingPreview::consume_user_closed()
{
	std::lock_guard<std::mutex> lock(mu_);
	const bool v = user_closed_;
	user_closed_ = false;
	return v;
}

void FloatingPreview::push_bgr(const uint8_t *bgr, int w, int h, const std::string &stats_line)
{
	if (!bgr || w <= 0 || h <= 0)
		return;
#ifdef _WIN32
	HWND hwnd = nullptr;
#endif
	{
		std::lock_guard<std::mutex> lock(mu_);
		if (!cfg_.enabled)
			return;
		const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
		bgra_.resize(n * 4u);
		for (size_t i = 0; i < n; ++i) {
			bgra_[i * 4 + 0] = bgr[i * 3 + 0];
			bgra_[i * 4 + 1] = bgr[i * 3 + 1];
			bgra_[i * 4 + 2] = bgr[i * 3 + 2];
			bgra_[i * 4 + 3] = 255;
		}
		frame_w_ = w;
		frame_h_ = h;
		stats_ = cfg_.show_stats ? stats_line : std::string{};
#ifdef _WIN32
		hwnd = static_cast<HWND>(hwnd_);
#endif
	}
#ifdef _WIN32
	// Invalidate outside the lock to avoid re-entrant paint deadlocks
	if (hwnd && IsWindow(hwnd))
		InvalidateRect(hwnd, nullptr, FALSE);
#endif
}

void FloatingPreview::ensure_window()
{
#ifdef _WIN32
	if (hwnd_ && IsWindow(static_cast<HWND>(hwnd_)))
		return;
	hwnd_ = nullptr;

	if (!class_registered_) {
		WNDCLASSW wc{};
		wc.lpfnWndProc = FloatingPreviewProc;
		wc.hInstance = GetModuleHandleW(nullptr);
		wc.lpszClassName = kClassName;
		wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
		wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
		// Ignore "already registered" — safe for re-open after crash path
		if (!RegisterClassW(&wc)) {
			const DWORD err = GetLastError();
			if (err != ERROR_CLASS_ALREADY_EXISTS) {
				return;
			}
		}
		class_registered_ = true;
	}
	int ww = 640, wh = 480;
	{
		std::lock_guard<std::mutex> lock(mu_);
		ww = cfg_.width;
		wh = cfg_.height;
	}
	// Client size → outer size
	RECT wr{0, 0, ww, wh};
	AdjustWindowRectEx(&wr, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_TOPMOST | WS_EX_TOOLWINDOW);
	const int ow = wr.right - wr.left;
	const int oh = wr.bottom - wr.top;
	const int x = GetSystemMetrics(SM_CXSCREEN) / 2 - ow / 2;
	const int y = GetSystemMetrics(SM_CYSCREEN) / 2 - oh / 2;
	CreateCtx ctx{this};
	HWND hwnd = CreateWindowExW(
	    WS_EX_TOPMOST | WS_EX_TOOLWINDOW, kClassName, L"YOLO Aim Floating Preview",
	    WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_VISIBLE, x, y, ow, oh,
	    nullptr, nullptr, GetModuleHandleW(nullptr), &ctx);
	if (!hwnd)
		return;
	{
		std::lock_guard<std::mutex> lock(mu_);
		hwnd_ = hwnd;
	}
	ShowWindow(hwnd, SW_SHOWNOACTIVATE);
	UpdateWindow(hwnd);
#endif
}

void FloatingPreview::destroy_window()
{
#ifdef _WIN32
	HWND hwnd = nullptr;
	{
		std::lock_guard<std::mutex> lock(mu_);
		hwnd = static_cast<HWND>(hwnd_);
		hwnd_ = nullptr;
	}
	if (hwnd && IsWindow(hwnd)) {
		// Release any active capture first
		ReleaseCapture();
		// DestroyWindow must run on the creating thread (engine thread) — OK here
		DestroyWindow(hwnd);
		// Do NOT PeekMessage with hwnd after DestroyWindow — handle is invalid.
		// Windows will clean up queued messages automatically.
	}
#endif
}

void FloatingPreview::resize_window(int w, int h)
{
#ifdef _WIN32
	HWND hwnd = nullptr;
	{
		std::lock_guard<std::mutex> lock(mu_);
		hwnd = static_cast<HWND>(hwnd_);
	}
	if (hwnd && IsWindow(hwnd)) {
		RECT wr{0, 0, w, h};
		AdjustWindowRectEx(&wr, WS_OVERLAPPEDWINDOW, FALSE, WS_EX_TOPMOST | WS_EX_TOOLWINDOW);
		SetWindowPos(hwnd, nullptr, 0, 0, wr.right - wr.left, wr.bottom - wr.top,
		             SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
	}
#else
	(void)w;
	(void)h;
#endif
}

void FloatingPreview::pump()
{
#ifdef _WIN32
	const Settings s = settings();
	if (s.enabled) {
		if (!is_open())
			ensure_window();
		else {
			HWND hwnd = nullptr;
			{
				std::lock_guard<std::mutex> lock(mu_);
				hwnd = static_cast<HWND>(hwnd_);
			}
			if (hwnd && IsWindow(hwnd)) {
				RECT rc{};
				if (GetWindowRect(hwnd, &rc)) {
					const int cw = rc.right - rc.left;
					const int ch = rc.bottom - rc.top;
					// only resize if user didn't drag-resize much; skip aggressive churn
					if (std::abs(cw - s.width) > 80 || std::abs(ch - s.height) > 80) {
						// do not force-resize while user may be resizing
					}
				}
			} else {
				// hwnd stale
				std::lock_guard<std::mutex> lock(mu_);
				hwnd_ = nullptr;
			}
		}
		// CRITICAL: only pump messages for OUR window — never thread-global PeekMessage
		// (that steals Qt/Web/console messages and can crash the process).
		HWND hwnd = nullptr;
		{
			std::lock_guard<std::mutex> lock(mu_);
			hwnd = static_cast<HWND>(hwnd_);
		}
		if (hwnd && IsWindow(hwnd)) {
			MSG msg;
			int budget = 32;
			while (budget-- > 0 && PeekMessageW(&msg, hwnd, 0, 0, PM_REMOVE)) {
				TranslateMessage(&msg);
				DispatchMessageW(&msg);
			}
		}
	} else if (is_open()) {
		destroy_window();
	}
#endif
}

} // namespace ya
