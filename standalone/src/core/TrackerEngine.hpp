#pragma once

#include "models/Detection.h"

#include <vector>

namespace ya {

struct TrackerConfig {
	float iou_threshold = 0.3f;
	int max_lost_frames = 30;
	int max_reidentify_frames = 45;
	float reidentify_center_threshold = 0.08f;
	float weight_iou = 0.5f;
	float weight_center = 0.3f;
	float weight_aspect = 0.1f;
	float weight_area = 0.1f;
	bool use_kalman = false;
	// OBS Kalman UI fields (persisted; association already uses lost/re-id thresholds)
	int kalman_generate_threshold = 3;
	int kalman_terminate_count = 5;
	bool show_kalman_predictions = false;
	int kalman_prediction_frames = 5;
	bool show_kalman_trajectories = false;
};

// Lightweight multi-object tracker (Hungarian fused distance + lost-coast + re-id).
// Mirrors OBS filter_inference association without OBS coupling.
class TrackerEngine {
public:
	void set_config(const TrackerConfig &cfg) { cfg_ = cfg; }
	TrackerConfig config() const { return cfg_; }

	// Assign/update trackId on detections in-place; returns tracked list.
	std::vector<Detection> update(const std::vector<Detection> &dets);

	void reset();

private:
	struct Track {
		Detection det;
		int id = -1;
		int lost = 0;
		bool active = true;
	};

	float fused_cost(const Detection &a, const Detection &b) const;
	static float iou(const Detection &a, const Detection &b);

	TrackerConfig cfg_;
	std::vector<Track> tracks_;
	int next_id_ = 1;
};

} // namespace ya
