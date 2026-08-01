#include "InferEngine.hpp"

#include "models/ModelYOLO.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <opencv2/core.hpp>

namespace ya {

struct InferEngine::Impl {
	std::unique_ptr<ModelYOLO> model;
};

InferEngine::InferEngine() : impl_(std::make_unique<Impl>()) {}

InferEngine::~InferEngine()
{
	std::lock_guard<std::mutex> lock(mu_);
	impl_->model.reset();
	ready_ = false;
}

bool InferEngine::load(const InferConfig &cfg)
{
	std::lock_guard<std::mutex> lock(mu_);
	impl_->model.reset();
	ready_ = false;
	cfg_ = cfg;
	last_error_.clear();

	if (cfg_.model_path.empty()) {
		last_error_ = "model_path empty";
		return false;
	}

	auto version = static_cast<ModelYOLO::Version>(cfg_.model_version);
	// clamp
	if (cfg_.model_version < 0 || cfg_.model_version > 2) {
		version = ModelYOLO::Version::YOLOv11;
	}

	std::string device = cfg_.device.empty() ? "cpu" : cfg_.device;
	// normalize aliases
	if (device == "gpu")
		device = "cuda";

	const std::string requested = device;
		try {
			impl_->model = std::make_unique<ModelYOLO>(version);
			impl_->model->loadModel(cfg_.model_path, device, cfg_.num_threads,
			                        cfg_.input_resolution);
			impl_->model->setConfidenceThreshold(cfg_.confidence);
			impl_->model->setNMSThreshold(cfg_.nms);
			if (!cfg_.target_classes.empty()) {
				impl_->model->setTargetClasses(cfg_.target_classes);
			}
			// Prefer ModelYOLO runtime label (detects CUDA EP / cpu_pre honestly)
const std::string runtime = impl_->model->getRuntimeDevice();
				cfg_.device = runtime.empty() ? device : runtime;
				ready_ = true;
				last_error_.clear();
				// True EP failure only: actual is pure cpu (or starts with "cpu") while GPU was requested.
				// "cuda+cpu_pre" means CUDA EP is active; preprocess still on CPU (no HAVE_CUDA) — not a fallback.
				auto ep_base = [](std::string s) {
					const auto p = s.find('+');
					if (p != std::string::npos)
						s = s.substr(0, p);
					return s;
				};
				const std::string act_ep = ep_base(cfg_.device);
				const std::string req_ep = ep_base(requested);
				if (requested != "cpu" && act_ep == "cpu" && req_ep != "cpu") {
					last_error_ = "requested " + requested + " → actual " + cfg_.device +
					              " (check cudnn / CUDA toolkit PATH)";
				} else if (req_ep != act_ep && act_ep != "cpu" &&
				           !(req_ep == "cuda" && act_ep.find("cuda") == 0)) {
					// mild note only for unexpected EP swap (e.g. tensorrt→cuda)
					last_error_ = "EP note: requested " + requested + " actual " + cfg_.device;
				}
			std::fprintf(stderr,
			             "[infer] model loaded path=%s requested=%s actual=%s input=%d threads=%d conf=%.2f nms=%.2f\n",
			             cfg_.model_path.c_str(), requested.c_str(), cfg_.device.c_str(),
			             cfg_.input_resolution, cfg_.num_threads, cfg_.confidence, cfg_.nms);
			return true;
		} catch (const std::exception &e) {
			// CPU fallback
			if (device != "cpu") {
				std::fprintf(stderr, "[infer] device=%s FAILED (%s) → trying CPU fallback\n",
				             device.c_str(), e.what());
				try {
					impl_->model = std::make_unique<ModelYOLO>(version);
					impl_->model->loadModel(cfg_.model_path, "cpu", cfg_.num_threads,
					                        cfg_.input_resolution);
					impl_->model->setConfidenceThreshold(cfg_.confidence);
					impl_->model->setNMSThreshold(cfg_.nms);
					if (!cfg_.target_classes.empty()) {
						impl_->model->setTargetClasses(cfg_.target_classes);
					}
					const std::string runtime = impl_->model->getRuntimeDevice();
					cfg_.device = runtime.empty() ? "cpu" : runtime;
					ready_ = true;
					last_error_ = std::string("GPU failed, CPU ok: ") + e.what();
					std::fprintf(stderr,
					             "[infer] CPU FALLBACK ok path=%s requested=%s actual=%s note=%s\n",
					             cfg_.model_path.c_str(), requested.c_str(), cfg_.device.c_str(),
					             e.what());
					return true;
				} catch (const std::exception &e2) {
					last_error_ = std::string("load failed: ") + e2.what();
					impl_->model.reset();
					ready_ = false;
					std::fprintf(stderr, "[infer] load FAILED path=%s err=%s\n",
					             cfg_.model_path.c_str(), e2.what());
					return false;
				}
			}
			last_error_ = std::string("load failed: ") + e.what();
			impl_->model.reset();
			ready_ = false;
			std::fprintf(stderr, "[infer] load FAILED path=%s err=%s\n", cfg_.model_path.c_str(),
			             e.what());
			return false;
		}
}

void InferEngine::unload()
{
	std::lock_guard<std::mutex> lock(mu_);
	impl_->model.reset();
	ready_ = false;
}

InferConfig InferEngine::config() const
{
	std::lock_guard<std::mutex> lock(mu_);
	return cfg_;
}

void InferEngine::set_thresholds(float conf, float nms)
	{
		std::lock_guard<std::mutex> lock(mu_);
		cfg_.confidence = conf;
		cfg_.nms = nms;
		if (impl_->model && ready_) {
			impl_->model->setConfidenceThreshold(conf);
			impl_->model->setNMSThreshold(nms);
		}
	}

	void InferEngine::set_target_classes(const std::vector<int> &classes)
	{
		std::lock_guard<std::mutex> lock(mu_);
		cfg_.target_classes = classes;
		if (impl_->model && ready_) {
			// empty clears filter → all classes (ModelYOLO setTargetClasses)
			impl_->model->setTargetClasses(classes);
		}
	}

InferResult InferEngine::run_bgr(const uint8_t *bgr, int w, int h, int stride)
{
	InferResult r;
	std::lock_guard<std::mutex> lock(mu_);
	if (!ready_ || !impl_->model) {
		r.error = "not ready";
		return r;
	}
	if (!bgr || w <= 0 || h <= 0) {
		r.error = "bad frame";
		return r;
	}

	cv::Mat frame;
	if (stride == w * 3) {
		frame = cv::Mat(h, w, CV_8UC3, const_cast<uint8_t *>(bgr),
		                static_cast<size_t>(stride));
	} else {
		frame.create(h, w, CV_8UC3);
		for (int y = 0; y < h; ++y) {
			std::memcpy(frame.ptr(y), bgr + static_cast<size_t>(y) * stride,
			            static_cast<size_t>(w) * 3);
		}
	}

	const auto t0 = std::chrono::steady_clock::now();
	try {
		r.dets = impl_->model->inference(frame);
		r.ok = true;
	} catch (const std::exception &e) {
		r.error = e.what();
		r.ok = false;
	}
	const auto t1 = std::chrono::steady_clock::now();
	r.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	return r;
}

} // namespace ya
