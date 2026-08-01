#include "TrackerEngine.hpp"

#include "HungarianAlgorithm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>

namespace ya {
namespace {

float clampf(float v, float lo, float hi)
{
	return std::max(lo, std::min(hi, v));
}

} // namespace

float TrackerEngine::iou(const Detection &a, const Detection &b)
{
	const float ax2 = a.x + a.width;
	const float ay2 = a.y + a.height;
	const float bx2 = b.x + b.width;
	const float by2 = b.y + b.height;
	const float ix1 = std::max(a.x, b.x);
	const float iy1 = std::max(a.y, b.y);
	const float ix2 = std::min(ax2, bx2);
	const float iy2 = std::min(ay2, by2);
	const float iw = std::max(0.f, ix2 - ix1);
	const float ih = std::max(0.f, iy2 - iy1);
	const float inter = iw * ih;
	const float uni = a.width * a.height + b.width * b.height - inter;
	if (uni <= 1e-6f)
		return 0.f;
	return inter / uni;
}

float TrackerEngine::fused_cost(const Detection &a, const Detection &b) const
{
	// Prefer parent HungarianAlgorithm when available
	const float iou_v = iou(a, b);
	const float cx = a.centerX - b.centerX;
	const float cy = a.centerY - b.centerY;
	const float center_dist = std::sqrt(cx * cx + cy * cy);
	const float aspect_a = a.height > 1e-6f ? a.width / a.height : 1.f;
	const float aspect_b = b.height > 1e-6f ? b.width / b.height : 1.f;
	const float aspect_diff = std::fabs(aspect_a - aspect_b);
	const float area_a = a.width * a.height;
	const float area_b = b.width * b.height;
	const float area_ratio =
	    (std::max(area_a, area_b) > 1e-6f)
	        ? std::fabs(area_a - area_b) / std::max(area_a, area_b)
	        : 1.f;

	// cost: lower is better; convert similarities
	const float cost_iou = 1.f - clampf(iou_v, 0.f, 1.f);
	const float cost_center = clampf(center_dist / 0.5f, 0.f, 1.f);
	const float cost_aspect = clampf(aspect_diff, 0.f, 1.f);
	const float cost_area = clampf(area_ratio, 0.f, 1.f);

	return cfg_.weight_iou * cost_iou + cfg_.weight_center * cost_center +
	       cfg_.weight_aspect * cost_aspect + cfg_.weight_area * cost_area;
}

void TrackerEngine::reset()
{
	tracks_.clear();
	next_id_ = 1;
}

std::vector<Detection> TrackerEngine::update(const std::vector<Detection> &dets)
{
	if (dets.empty()) {
		for (auto &t : tracks_) {
			if (!t.active)
				continue;
			++t.lost;
			if (t.lost > cfg_.max_lost_frames)
				t.active = false;
		}
		// Do NOT emit coasted predictions as draw boxes when frame has zero
		// detections — that leaves "ghost" boxes floating where nothing is.
		// Aim/Kalman can still use tracks_ internally if needed later.
		return {};
	}

	// prune dead
	tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
	                             [](const Track &t) { return !t.active; }),
	              tracks_.end());

	const int nT = static_cast<int>(tracks_.size());
	const int nD = static_cast<int>(dets.size());

	std::vector<int> assign_track(nD, -1);
	std::vector<bool> track_used(nT, false);

	if (nT > 0 && nD > 0) {
		// OBS filter_inference: hard class gate — different classId never associate
		std::vector<std::vector<float>> cost(nT, std::vector<float>(nD, 1e3f));
		for (int i = 0; i < nT; ++i) {
			const auto &td = tracks_[i].det;
			const cv::Rect2f tb(td.x, td.y, td.width, td.height);
			const cv::Point2f tc(td.centerX, td.centerY);
			for (int j = 0; j < nD; ++j) {
				const auto &dd = dets[j];
				// OBS: if (det.classId != trk.classId) cost = huge
				if (dd.classId != td.classId) {
					cost[i][j] = 1e6f;
					continue;
				}
				// OBS coarse gate on center distance
				const float gdx = dd.centerX - td.centerX;
				const float gdy = dd.centerY - td.centerY;
				const float gate =
				    0.35f + 0.5f * std::max(dd.width + td.width, dd.height + td.height);
				if (gdx * gdx + gdy * gdy > gate * gate) {
					cost[i][j] = 1e6f;
					continue;
				}
				const cv::Rect2f db(dd.x, dd.y, dd.width, dd.height);
				const cv::Point2f dc(dd.centerX, dd.centerY);
				cost[i][j] = HungarianAlgorithm::calculateFusedDistance(
				    db, tb, dc, tc, cfg_.weight_iou, cfg_.weight_center, cfg_.weight_aspect,
				    cfg_.weight_area);
			}
		}
		std::vector<int> track_to_det = HungarianAlgorithm::solve(cost);
		for (int i = 0; i < nT && i < static_cast<int>(track_to_det.size()); ++i) {
			const int j = track_to_det[i];
			if (j < 0 || j >= nD)
				continue;
			if (cost[i][static_cast<size_t>(j)] > 0.85f)
				continue;
			// OBS double-check same class
			if (tracks_[i].det.classId != dets[static_cast<size_t>(j)].classId)
				continue;
			if (assign_track[j] >= 0)
				continue;
			track_used[i] = true;
			assign_track[j] = i;
		}
	}

	std::vector<Detection> out;
	out.reserve(dets.size());

	for (int j = 0; j < nD; ++j) {
		Detection d = dets[j];
		if (assign_track[j] >= 0) {
			Track &t = tracks_[assign_track[j]];
			// velocity
			d.velX = d.centerX - t.det.centerX;
			d.velY = d.centerY - t.det.centerY;
			d.trackId = t.id;
			d.lostFrames = 0;
			t.det = d;
			t.lost = 0;
			t.active = true;
		} else {
// try re-id against recently lost (OBS: same class only)
				// Find best match first, THEN modify track state (avoid mid-iteration corruption)
				int re_id = -1;
				float best = cfg_.reidentify_center_threshold;
				int best_track_idx = -1;
				for (int ti = 0; ti < nT; ++ti) {
					auto &t = tracks_[ti];
					if (t.active && t.lost == 0)
						continue;
					if (t.lost > cfg_.max_reidentify_frames)
						continue;
					if (t.det.classId >= 0 && d.classId != t.det.classId)
						continue;
					const float dx = t.det.centerX - d.centerX;
					const float dy = t.det.centerY - d.centerY;
					const float dist = std::sqrt(dx * dx + dy * dy);
					if (dist < best) {
						best = dist;
						re_id = t.id;
						best_track_idx = ti;
					}
				}
				if (re_id >= 0 && best_track_idx >= 0) {
					auto &t = tracks_[best_track_idx];
					d.trackId = t.id;
					d.lostFrames = 0;
					t.det = d;
					t.lost = 0;
					t.active = true;
				}
			if (re_id < 0) {
				Track nt;
				nt.id = next_id_++;
				nt.det = d;
				nt.lost = 0;
				nt.active = true;
				d.trackId = nt.id;
				d.lostFrames = 0;
				tracks_.push_back(nt);
			}
		}
		out.push_back(d);
	}

// mark unmatched tracks lost — do not append them to `out` (only real dets draw)
		for (int i = 0; i < nT; ++i) {
			if (!track_used[i] && tracks_[i].active) {
				++tracks_[i].lost;
				// coast internally for re-id only — only update center (velocity is center delta)
				tracks_[i].det.centerX += tracks_[i].det.velX;
				tracks_[i].det.centerY += tracks_[i].det.velY;
				if (tracks_[i].lost > cfg_.max_lost_frames)
					tracks_[i].active = false;
			}
		}

		return out; // only detections that exist this frame
	}

} // namespace ya
