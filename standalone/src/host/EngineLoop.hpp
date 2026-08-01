#pragma once

#include "FrameSource.hpp"
#include "FloatingPreview.hpp"
#include "InferEngine.hpp"
#include "TrackerEngine.hpp"
#include "yolo_aim/types.h"

#ifdef YA_WITH_AIM
#include "FullAimBridge.hpp"
#else
#include "AimEngine.hpp"
#endif

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ya {

struct VisionConfig {
			bool show_detection_results = true;
			int bbox_line_width = 2;
			float label_font_scale = 0.5f;
			bool export_coordinates = false;
			std::string coordinate_output_path; // relative or absolute; empty → exe dir detections.json
			// OBS floating window (page 5)
			bool show_floating_window = false;
			int floating_window_width = 640;
			int floating_window_height = 480;
			bool show_track_id_in_floating_window = false;
			bool preview_enabled = true; // web preview on/off (off = skip draw/encode)
		};

struct EngineConfig {
			CaptureBackend capture_backend = CaptureBackend::Dxgi;
			// center = screen-center ROI crop (default, OBS-like detection region)
			// region = fixed (region_x, region_y, width, height) ROI crop
			// full   = full desktop capture then software crop if use_region
			std::string capture_mode = "center";
			int width = 640;
			int height = 640;
			// OBS use_region + region_* (detection ROI). For center/region capture these
			// match the DXGI crop; for full-desktop capture, software crop uses them.
			bool use_region = true; // OBS default false on full source; standalone defaults true (ROI)
			int region_x = 0;
			int region_y = 0;
			int region_width = 640;  // OBS region_width (may differ from capture width when full)
			int region_height = 640;
			bool infer_enabled = false;
			bool aim_enabled = false;
		InferConfig infer;
		TrackerConfig tracker;
		VisionConfig vision;
	#ifdef YA_WITH_AIM
		FullAimSettings aim_full;
	#else
		AimConfig aim;
	#endif
std::string preview_dir;
			// Web/preview BMP encode rate (seconds). Default ~30 FPS.
			double preview_interval_sec = 0.033;
		// When set, PUT /api/config writes full document here (OBS-style user override).
		std::string user_config_path;
		// ONNX scan roots (exe/models, optional NCNN, custom).
		std::vector<std::string> model_search_dirs;
		// Directory used for WebUI uploads (usually exe/models).
		std::string models_dir;
	};

	struct ModelEntry {
		std::string name; // filename
		std::string path; // absolute or relative usable path
		std::string dir;  // parent folder label
		uint64_t size = 0;
	};

class EngineLoop {
public:
	EngineLoop();
	~EngineLoop();

	void set_config(const EngineConfig &cfg);
	EngineConfig config() const;

bool apply_config_json(const std::string &json, std::string *err_out = nullptr);
		std::string config_json() const;

		// Cold reloads (applied on engine thread next loop).
		bool request_reload_model(std::string *err_out = nullptr);
		bool request_reload_capture(std::string *err_out = nullptr);

	// Scan model_search_dirs / models_dir for *.onnx (OBS model_path picker equivalent).
			// If folder non-empty, scan that folder (optionally recursive) in addition / prefer it.
			std::vector<ModelEntry> list_models(const std::string &folder = {},
			                                    bool recursive = true) const;
			// Set primary models folder (persisted via models_dir + search list).
			bool set_model_folder(const std::string &folder, bool recursive = true,
			                      std::string *err_out = nullptr);
			// Save uploaded bytes under models_dir; returns absolute path.
			bool save_model_file(const std::string &filename, const std::vector<uint8_t> &data,
			                     std::string *out_path, std::string *err_out = nullptr);

bool start();
			void stop();
			// Flush current in-memory config to user_config_path (call on quit / Ctrl+C).
			bool save_user_config(std::string *err_out = nullptr);
			bool running() const { return running_.load(); }

		EngineSnapshot snapshot() const;
		AimDebug last_aim_debug() const;

		std::vector<uint8_t> preview_bmp() const;
		std::string preview_path() const;
		uint64_t frame_count() const { return frame_count_.load(); }

		std::vector<YoloDet> last_detections() const;

	#ifdef YA_WITH_AIM
		FullAimStatus aim_status() const;
		bool test_controller(const std::string &type, const std::string &makcu_port, int baud,
		                     int logi_type, std::string *err);
	#endif

private:
		void thread_main();
		// AiMod-style: capture never waits on YOLO (own thread + latest-frame slot)
		void capture_loop(FrameSource *source);
		// Preview thread (AiMod Display thread): draw boxes + BMP encode + floating
		// window + json export, off the hot infer loop. FloatingPreview HWND is
		// created/pumped here (Win32 thread affinity — must stay on this thread).
		void preview_loop();

		mutable std::mutex cfg_mu_;
		EngineConfig cfg_;

		std::atomic<bool> running_{false};
		std::atomic<bool> stop_{false};
		std::atomic<bool> reload_model_{false};
		std::atomic<bool> reload_capture_{false};
		std::thread th_;
		std::thread cap_th_;
		std::atomic<bool> cap_stop_{false};

		// Preview offload (AiMod-style third thread: Display)
		std::thread preview_th_;
		std::atomic<bool> preview_stop_{false};
		std::atomic<bool> floating_user_closed_{false}; // preview flags; main clears cfg
		mutable std::mutex preview_job_mu_;
		std::atomic<uint64_t> preview_seq_{0};
		struct PreviewJob {
			std::vector<uint8_t> bgr;          // moved from packet (no copy on hot path)
			int width = 0, height = 0;
			int origin_x = 0, origin_y = 0;
			std::vector<YoloDet> dets;         // tracked detections snapshot
			double fps = 0.0, infer_ms = 0.0;
			int fov_px = 0;
			double preview_interval = 0.033;
			VisionConfig vision;
		#ifdef YA_WITH_AIM
			FullAimSettings aim_full;
			bool aim_enabled = false;
		#endif
		} preview_job_;

			// Latest frame from capture thread (always overwrite = drop old frames)
			mutable std::mutex frame_mu_;
			FramePacket latest_frame_{};
			std::atomic<uint64_t> latest_seq_{0};
			std::atomic<uint64_t> capture_frames_{0};
			std::atomic<uint32_t> capture_fps_fixed_{0}; // fps*10 as uint32 (avoids atomic<double>)
			uint64_t last_seen_seq_ = 0; // reset per thread_main cycle

			mutable std::mutex snap_mu_;
			EngineSnapshot snap_{};

			mutable std::mutex preview_mu_;
			std::vector<uint8_t> preview_bmp_;
			std::string preview_path_;
			std::atomic<uint64_t> frame_count_{0};

			mutable std::mutex det_mu_;
			std::vector<YoloDet> last_dets_;

			mutable std::mutex aim_dbg_mu_;
			AimDebug last_aim_dbg_{};

			InferEngine infer_;
			TrackerEngine tracker_;
			FloatingPreview floating_;
		#ifdef YA_WITH_AIM
			FullAimBridge aim_full_;
		#else
			AimEngine aim_;
		#endif
		};

} // namespace ya
