#pragma once

#include "yolo_aim/types.h"

#include <mutex>
#include <string>
#include <vector>

namespace ya {

struct AimConfig {
	bool enabled = false;
	int hotkey_vk = 0x02; // VK_RBUTTON
	int fov_radius = 120;
	float pid_p_min = 0.15f;
	float pid_p_max = 0.6f;
	float pid_d = 0.007f;
	float pid_i = 0.01f;
	float dead_zone_px = 5.0f;
	float max_pixel_move = 128.0f;
	float target_y_offset = 0.0f; // fraction of box height (0=center, -0.2=upper)
	std::string controller = "WindowsAPI";
};

class AimEngine {
public:
	AimEngine();
	~AimEngine();

	void set_config(const AimConfig &cfg);
	AimConfig config() const;

	// Call after detections are ready for this frame.
	// When enabled and hotkey held, selects target and moves mouse (relative).
	void tick(const std::vector<YoloDet> &dets, const AimFrameMeta &meta, AimDebug *debug_out);

	void reset();

	bool last_hotkey_down() const { return last_hotkey_down_; }
	bool last_moved() const { return last_moved_; }
	const AimDebug &last_debug() const { return last_debug_; }

private:
	bool hotkey_down(int vk) const;
	void move_relative(int dx, int dy);
	bool pick_target(const std::vector<YoloDet> &dets, const AimFrameMeta &meta, YoloDet *out,
	                 float *err_x, float *err_y) const;

	mutable std::mutex mu_;
	AimConfig cfg_;

	float integral_x_ = 0.f;
	float integral_y_ = 0.f;
	float prev_err_x_ = 0.f;
	float prev_err_y_ = 0.f;
	bool has_prev_ = false;

	bool last_hotkey_down_ = false;
	bool last_moved_ = false;
	AimDebug last_debug_{};
};

} // namespace ya
