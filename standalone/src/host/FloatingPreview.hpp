#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ya {

// OBS-style always-on-top floating preview (Win32 HWND), independent of the WebUI.
class FloatingPreview {
public:
	struct Settings {
		bool enabled = false;
		int width = 640;
		int height = 480;
		bool show_track_id = false;
		bool show_stats = true;
	};

	FloatingPreview();
	~FloatingPreview();

	FloatingPreview(const FloatingPreview &) = delete;
	FloatingPreview &operator=(const FloatingPreview &) = delete;

	void set_settings(const Settings &s);
	Settings settings() const;

	// BGR contiguous, size w*h*3. Thread-safe.
	void push_bgr(const uint8_t *bgr, int w, int h, const std::string &stats_line = {});

	// Create/destroy window + drain messages (call periodically from host/engine thread).
	void pump();

	bool is_open() const;

	// True if user closed the HWND (X button) while config still wanted it enabled.
	// Engine should clear vision.show_floating_window once.
	bool consume_user_closed();

private:
	void ensure_window();
	void destroy_window();
	void resize_window(int w, int h);

	mutable std::mutex mu_;
	Settings cfg_{};
	std::vector<uint8_t> bgra_;
	int frame_w_ = 0;
	int frame_h_ = 0;
	std::string stats_;
	void *hwnd_ = nullptr;
	bool class_registered_ = false;
	bool drag_ = false;
	int drag_dx_ = 0;
	int drag_dy_ = 0;
	bool user_closed_ = false; // set by WM_CLOSE; consumed by engine

	// Win32 WndProc (defined in .cpp) needs private access
	friend struct FloatingPreviewAccess;
};

} // namespace ya
