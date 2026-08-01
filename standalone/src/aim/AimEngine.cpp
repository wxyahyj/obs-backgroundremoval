#include "AimEngine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ya {
namespace {

float clampf(float v, float lo, float hi)
{
	return std::max(lo, std::min(hi, v));
}

} // namespace

AimEngine::AimEngine()
{
	std::memset(&last_debug_, 0, sizeof(last_debug_));
}

AimEngine::~AimEngine() = default;

void AimEngine::set_config(const AimConfig &cfg)
{
	std::lock_guard<std::mutex> lock(mu_);
	cfg_ = cfg;
}

AimConfig AimEngine::config() const
{
	std::lock_guard<std::mutex> lock(mu_);
	return cfg_;
}

void AimEngine::reset()
{
	std::lock_guard<std::mutex> lock(mu_);
	integral_x_ = integral_y_ = 0.f;
	prev_err_x_ = prev_err_y_ = 0.f;
	has_prev_ = false;
	last_moved_ = false;
	last_hotkey_down_ = false;
	std::memset(&last_debug_, 0, sizeof(last_debug_));
}

bool AimEngine::hotkey_down(int vk) const
{
#ifdef _WIN32
	if (vk <= 0)
		return false;
	return (GetAsyncKeyState(vk) & 0x8000) != 0;
#else
	(void)vk;
	return false;
#endif
}

void AimEngine::move_relative(int dx, int dy)
{
#ifdef _WIN32
	if (dx == 0 && dy == 0)
		return;
	INPUT input = {};
	input.type = INPUT_MOUSE;
	input.mi.dx = dx;
	input.mi.dy = dy;
	input.mi.dwFlags = MOUSEEVENTF_MOVE;
	SendInput(1, &input, sizeof(INPUT));
#else
	(void)dx;
	(void)dy;
#endif
}

bool AimEngine::pick_target(const std::vector<YoloDet> &dets, const AimFrameMeta &meta,
                            YoloDet *out, float *err_x, float *err_y) const
{
	if (!out || dets.empty() || meta.frame_w <= 0 || meta.frame_h <= 0)
		return false;

	const float cx = 0.5f;
	const float cy = 0.5f;
	const float fw = static_cast<float>(meta.frame_w);
	const float fh = static_cast<float>(meta.frame_h);
	const float fov = static_cast<float>(cfg_.fov_radius);

	float best_dist2 = 1e30f;
	const YoloDet *best = nullptr;
	float best_ex = 0.f, best_ey = 0.f;

	for (const auto &d : dets) {
		// aim point: box center + vertical offset fraction of height
		const float aim_nx = d.x + d.w * 0.5f;
		const float aim_ny = d.y + d.h * (0.5f + cfg_.target_y_offset);
		const float px = (aim_nx - cx) * fw;
		const float py = (aim_ny - cy) * fh;
		const float dist2 = px * px + py * py;
		if (fov > 0.f && dist2 > fov * fov)
			continue;
		if (dist2 < best_dist2) {
			best_dist2 = dist2;
			best = &d;
			best_ex = px;
			best_ey = py;
		}
	}

	if (!best)
		return false;
	*out = *best;
	if (err_x)
		*err_x = best_ex;
	if (err_y)
		*err_y = best_ey;
	return true;
}

void AimEngine::tick(const std::vector<YoloDet> &dets, const AimFrameMeta &meta, AimDebug *debug_out)
{
	std::lock_guard<std::mutex> lock(mu_);
	AimDebug dbg{};
	dbg.det_count = static_cast<int32_t>(dets.size());
	last_moved_ = false;

	if (!cfg_.enabled) {
		last_hotkey_down_ = false;
		last_debug_ = dbg;
		if (debug_out)
			*debug_out = dbg;
		return;
	}

	const bool key = hotkey_down(cfg_.hotkey_vk);
	last_hotkey_down_ = key;
	if (!key) {
		// release: clear I term so we don't wind up while idle
		integral_x_ = integral_y_ = 0.f;
		has_prev_ = false;
		last_debug_ = dbg;
		if (debug_out)
			*debug_out = dbg;
		return;
	}

	YoloDet tgt{};
	float err_x = 0.f, err_y = 0.f;
	if (!pick_target(dets, meta, &tgt, &err_x, &err_y)) {
		has_prev_ = false;
		last_debug_ = dbg;
		if (debug_out)
			*debug_out = dbg;
		return;
	}

	dbg.target_track_id = tgt.track_id;
	dbg.error_x = err_x;
	dbg.error_y = err_y;

	const float abs_err = std::sqrt(err_x * err_x + err_y * err_y);
	if (abs_err <= cfg_.dead_zone_px) {
		// settled
		integral_x_ = integral_y_ = 0.f;
		has_prev_ = false;
		last_debug_ = dbg;
		if (debug_out)
			*debug_out = dbg;
		return;
	}

	// Dynamic P: larger error → higher P (clamped)
	const float span = std::max(1.f, static_cast<float>(cfg_.fov_radius));
	const float t = clampf(abs_err / span, 0.f, 1.f);
	const float kp = cfg_.pid_p_min + (cfg_.pid_p_max - cfg_.pid_p_min) * t;
	const float ki = cfg_.pid_i;
	const float kd = cfg_.pid_d;

	integral_x_ = clampf(integral_x_ + err_x * 0.016f, -50.f, 50.f);
	integral_y_ = clampf(integral_y_ + err_y * 0.016f, -50.f, 50.f);

	float d_x = 0.f, d_y = 0.f;
	if (has_prev_) {
		d_x = err_x - prev_err_x_;
		d_y = err_y - prev_err_y_;
	}
	prev_err_x_ = err_x;
	prev_err_y_ = err_y;
	has_prev_ = true;

	float out_x = kp * err_x + ki * integral_x_ + kd * d_x;
	float out_y = kp * err_y + ki * integral_y_ + kd * d_y;

	out_x = clampf(out_x, -cfg_.max_pixel_move, cfg_.max_pixel_move);
	out_y = clampf(out_y, -cfg_.max_pixel_move, cfg_.max_pixel_move);

	const int dx = static_cast<int>(std::lround(out_x));
	const int dy = static_cast<int>(std::lround(out_y));

	dbg.out_x = static_cast<float>(dx);
	dbg.out_y = static_cast<float>(dy);
	dbg.kp = kp;
	dbg.ki = ki;
	dbg.kd = kd;

	if (dx != 0 || dy != 0) {
		move_relative(dx, dy);
		last_moved_ = true;
	}

	last_debug_ = dbg;
	if (debug_out)
		*debug_out = dbg;
}

} // namespace ya
