#pragma once

#include "models/Detection.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ya {

struct InferConfig {
	std::string model_path;
	std::string device = "cpu"; // cpu | cuda | tensorrt | dml
	int model_version = -1;      // -1=自动(黑图冒烟检测) 0=v5 1=v8 2=v11
	int input_resolution = 640;
	float confidence = 0.45f;
	float nms = 0.45f;
	int num_threads = 4;
	std::vector<int> target_classes; // empty = all
	std::string target_classes_text; // "0,1,2" OBS form; empty = all
	int interval_frames = 1;         // run every N capture frames
};

struct InferResult {
	std::vector<Detection> dets;
	double infer_ms = 0.0;
	bool ok = false;
	std::string error;
};

class InferEngine {
public:
	InferEngine();
	~InferEngine();

	bool load(const InferConfig &cfg);
	void unload();
	bool ready() const { return ready_; }

	// 当前模型类别数(0 = 未加载)
	int num_classes() const;

	// BGR tightly packed (from FrameSource)
	InferResult run_bgr(const uint8_t *bgr, int w, int h, int stride);

	InferConfig config() const;
	void set_thresholds(float conf, float nms);
	// Hot-apply target class filter (empty = all classes). No model reload.
	void set_target_classes(const std::vector<int> &classes);
	const std::string &last_error() const { return last_error_; }

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
	mutable std::mutex mu_;
	InferConfig cfg_;
	bool ready_ = false;
	std::string last_error_;
};

} // namespace ya
