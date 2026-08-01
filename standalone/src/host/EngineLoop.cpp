#include "EngineLoop.hpp"
#include "BmpUtil.hpp"
#include "ConfigStore.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#ifdef YA_WITH_AIM
#include "CrosshairDetector.hpp"
#include <opencv2/core.hpp>
#endif

namespace ya {
namespace {

// OBS filter_inference: crop ROI → YOLO → remap dets to full-frame normalized coords.
// Returns dets normalized to full_w x full_h; crop_x/y are the ROI origin in the frame.
void remap_dets_roi_to_full(std::vector<Detection> &dets, int crop_x, int crop_y, int crop_w,
                            int crop_h, int full_w, int full_h)
{
	if (dets.empty() || crop_w <= 0 || crop_h <= 0 || full_w <= 0 || full_h <= 0)
		return;
	// No-op when ROI is the whole frame
	if (crop_x == 0 && crop_y == 0 && crop_w == full_w && crop_h == full_h)
		return;
	const float fw = static_cast<float>(full_w);
	const float fh = static_cast<float>(full_h);
	const float cw = static_cast<float>(crop_w);
	const float ch = static_cast<float>(crop_h);
	const float ox = static_cast<float>(crop_x);
	const float oy = static_cast<float>(crop_y);
	for (auto &det : dets) {
		const float pixelX = det.x * cw + ox;
		const float pixelY = det.y * ch + oy;
		const float pixelW = det.width * cw;
		const float pixelH = det.height * ch;
		const float pixelCX = det.centerX * cw + ox;
		const float pixelCY = det.centerY * ch + oy;
		det.x = pixelX / fw;
		det.y = pixelY / fh;
		det.width = pixelW / fw;
		det.height = pixelH / fh;
		det.centerX = pixelCX / fw;
		det.centerY = pixelCY / fh;
	}
}

// Open capture like OBS detection region:
//  - center: DXGI center crop width×height (default)
//  - region: DXGI fixed ROI at (region_x, region_y)
//  - full:   full desktop (width/height ignored for open; software crop if use_region)
bool open_capture_obs_style(FrameSource &source, const EngineConfig &cfg, std::ostream &log)
{
	const int rw = cfg.region_width > 0 ? cfg.region_width : cfg.width;
	const int rh = cfg.region_height > 0 ? cfg.region_height : cfg.height;
	const int cap_w = cfg.width > 0 ? cfg.width : rw;
	const int cap_h = cfg.height > 0 ? cfg.height : rh;

	if (cfg.capture_mode == "full" || cfg.capture_mode == "fullscreen") {
#ifdef _WIN32
		const int sw = GetSystemMetrics(SM_CXSCREEN);
		const int sh = GetSystemMetrics(SM_CYSCREEN);
#else
		const int sw = 1920, sh = 1080;
#endif
		log << "[engine] open_full desktop " << sw << "x" << sh
		    << (cfg.use_region ? " + software use_region crop" : " (full-frame infer)")
		    << std::endl;
		// Capture entire primary display as region (0,0,sw,sh)
		return source.open_region(cfg.capture_backend, 0, 0, sw, sh);
	}
	if (cfg.capture_mode == "region") {
		log << "[engine] open_region x=" << cfg.region_x << " y=" << cfg.region_y << " w=" << rw
		    << " h=" << rh << " (OBS use_region ROI at capture)" << std::endl;
		return source.open_region(cfg.capture_backend, cfg.region_x, cfg.region_y, rw, rh);
	}
	// center (default): screen-center crop — standalone's primary "detection region"
	log << "[engine] open_center (OBS-style detection region / screen center crop) w=" << cap_w
	    << " h=" << cap_h << std::endl;
	return source.open_center(cfg.capture_backend, cap_w, cap_h);
}

// Resolve software crop rectangle relative to captured packet (OBS use_region on full frame).
// When capture already is the ROI (center/region modes), crop is 0,0,packet.w,packet.h.
void resolve_infer_roi(const EngineConfig &cfg, int packet_w, int packet_h, int origin_x,
                       int origin_y, int &crop_x, int &crop_y, int &crop_w, int &crop_h)
{
	crop_x = 0;
	crop_y = 0;
	crop_w = packet_w;
	crop_h = packet_h;

	// Capture already ROI-sized (center/region DXGI crop) → infer whole packet
	if (cfg.capture_mode != "full" && cfg.capture_mode != "fullscreen") {
		return;
	}
	// Full desktop capture: apply use_region like OBS filter_inference
	if (!cfg.use_region) {
		return;
	}
	const int rw = cfg.region_width > 0 ? cfg.region_width : cfg.width;
	const int rh = cfg.region_height > 0 ? cfg.region_height : cfg.height;
	// region_* are absolute screen coords; packet is full desktop starting at origin
	crop_x = std::max(0, cfg.region_x - origin_x);
	crop_y = std::max(0, cfg.region_y - origin_y);
	crop_w = std::min(rw, packet_w - crop_x);
	crop_h = std::min(rh, packet_h - crop_y);
	if (crop_w <= 0 || crop_h <= 0) {
		crop_x = 0;
		crop_y = 0;
		crop_w = packet_w;
		crop_h = packet_h;
	}
}

void draw_box_bgr(std::vector<uint8_t> &bgr, int w, int h, int x1, int y1, int x2, int y2,
                  uint8_t r, uint8_t g, uint8_t b, int thickness = 2)
{
	x1 = std::clamp(x1, 0, w - 1);
	x2 = std::clamp(x2, 0, w - 1);
	y1 = std::clamp(y1, 0, h - 1);
	y2 = std::clamp(y2, 0, h - 1);
	if (x2 < x1)
		std::swap(x1, x2);
	if (y2 < y1)
		std::swap(y1, y2);

	auto put = [&](int x, int y) {
		if (x < 0 || y < 0 || x >= w || y >= h)
			return;
		uint8_t *p = bgr.data() + (static_cast<size_t>(y) * w + x) * 3;
		p[0] = b;
		p[1] = g;
		p[2] = r;
	};

	for (int t = 0; t < thickness; ++t) {
		for (int x = x1; x <= x2; ++x) {
			put(x, y1 + t);
			put(x, y2 - t);
		}
		for (int y = y1; y <= y2; ++y) {
			put(x1 + t, y);
			put(x2 - t, y);
		}
	}
}

// Tiny 5x7 digit/char stamp so class id is burned into preview BMP (not only Web overlay)
void draw_label_bgr(std::vector<uint8_t> &bgr, int w, int h, int x, int y, const char *text,
                    uint8_t r, uint8_t g, uint8_t b)
{
	if (!text || !*text)
		return;
	// Prefer label inside box top; if y near 0, push down so text is not clipped
	int base_y = y;
	if (base_y < 10)
		base_y = y + 12;
	else
		base_y = y - 2;

	auto put = [&](int px, int py) {
		if (px < 0 || py < 0 || px >= w || py >= h)
			return;
		uint8_t *p = bgr.data() + (static_cast<size_t>(py) * w + px) * 3;
		p[0] = b;
		p[1] = g;
		p[2] = r;
	};

	// Background strip for readability
	const int tw = static_cast<int>(std::strlen(text)) * 6 + 4;
	const int th = 10;
	const int bx0 = std::max(0, x);
	const int by0 = std::max(0, base_y - 8);
	for (int py = by0; py < by0 + th && py < h; ++py)
		for (int px = bx0; px < bx0 + tw && px < w; ++px) {
			uint8_t *p = bgr.data() + (static_cast<size_t>(py) * w + px) * 3;
			p[0] = 20;
			p[1] = 20;
			p[2] = 20;
		}

	// Minimal 3x5 font for digits and 'c'/'%'
	static const uint8_t glyphs[16][5] = {
	    {0x7, 0x5, 0x5, 0x5, 0x7}, // 0
	    {0x2, 0x6, 0x2, 0x2, 0x7}, // 1
	    {0x7, 0x1, 0x7, 0x4, 0x7}, // 2
	    {0x7, 0x1, 0x7, 0x1, 0x7}, // 3
	    {0x5, 0x5, 0x7, 0x1, 0x1}, // 4
	    {0x7, 0x4, 0x7, 0x1, 0x7}, // 5
	    {0x7, 0x4, 0x7, 0x5, 0x7}, // 6
	    {0x7, 0x1, 0x1, 0x1, 0x1}, // 7
	    {0x7, 0x5, 0x7, 0x5, 0x7}, // 8
	    {0x7, 0x5, 0x7, 0x1, 0x7}, // 9
	    {0x6, 0x5, 0x7, 0x5, 0x5}, // A ~ c uses index 10 as 'c'
	    {0x0, 0x6, 0x4, 0x4, 0x6}, // c
	    {0x0, 0x0, 0x7, 0x0, 0x0}, // -
	    {0x0, 0x0, 0x0, 0x0, 0x2}, // .
	    {0x2, 0x5, 0x2, 0x0, 0x2}, // %
	    {0x0, 0x0, 0x0, 0x0, 0x0},
	};
	int cx = bx0 + 2;
	const int cy = by0 + 2;
	for (const char *p = text; *p; ++p) {
		int gi = 15;
		if (*p >= '0' && *p <= '9')
			gi = *p - '0';
		else if (*p == 'c' || *p == 'C')
			gi = 11;
		else if (*p == '-')
			gi = 12;
		else if (*p == '.')
			gi = 13;
		else if (*p == '%')
			gi = 14;
		else if (*p == ' ') {
			cx += 4;
			continue;
		}
		for (int row = 0; row < 5; ++row) {
			uint8_t bits = glyphs[gi][row];
			for (int col = 0; col < 3; ++col) {
				if (bits & (1 << (2 - col)))
					put(cx + col, cy + row);
			}
		}
		cx += 6;
	}
}

void draw_circle_bgr(std::vector<uint8_t> &bgr, int w, int h, int cx, int cy, int radius,
                     uint8_t r, uint8_t g, uint8_t b)
{
	if (radius <= 0)
		return;
	const int r2 = radius * radius;
	for (int y = -radius; y <= radius; ++y) {
		for (int x = -radius; x <= radius; ++x) {
			const int d = x * x + y * y;
			if (d >= r2 - radius && d <= r2 + radius) {
				const int px = cx + x, py = cy + y;
				if (px < 0 || py < 0 || px >= w || py >= h)
					continue;
				uint8_t *p = bgr.data() + (static_cast<size_t>(py) * w + px) * 3;
				p[0] = b;
				p[1] = g;
				p[2] = r;
			}
		}
	}
}

YoloDet to_c_det(const Detection &d)
{
	YoloDet o{};
	o.class_id = d.classId;
	o.confidence = d.confidence;
	o.x = d.x;
	o.y = d.y;
	o.w = d.width;
	o.h = d.height;
	o.cx = d.centerX;
	o.cy = d.centerY;
	o.track_id = d.trackId;
	return o;
}

bool json_num(const std::string &s, const char *key, double *out)
{
	const std::string pat = std::string("\"") + key + "\"";
	auto p = s.find(pat);
	if (p == std::string::npos)
		return false;
	p = s.find(':', p);
	if (p == std::string::npos)
		return false;
	++p;
	while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\r' || s[p] == '\n'))
		++p;
	try {
		size_t idx = 0;
		*out = std::stod(s.substr(p), &idx);
		return true;
	} catch (...) {
		return false;
	}
}

bool json_str(const std::string &s, const char *key, std::string *out)
{
	const std::string pat = std::string("\"") + key + "\"";
	auto p = s.find(pat);
	if (p == std::string::npos)
		return false;
	p = s.find(':', p);
	if (p == std::string::npos)
		return false;
	p = s.find('"', p);
	if (p == std::string::npos)
		return false;
	auto e = s.find('"', p + 1);
	if (e == std::string::npos)
		return false;
	*out = s.substr(p + 1, e - p - 1);
	return true;
}

bool json_bool_field(const std::string &s, const char *key, bool *out)
{
	const std::string pat = std::string("\"") + key + "\"";
	auto p = s.find(pat);
	if (p == std::string::npos)
		return false;
	p = s.find(':', p);
	if (p == std::string::npos)
		return false;
	++p;
	while (p < s.size() && (s[p] == ' ' || s[p] == '\t'))
		++p;
	if (s.compare(p, 4, "true") == 0) {
		*out = true;
		return true;
	}
	if (s.compare(p, 5, "false") == 0) {
		*out = false;
		return true;
	}
	return false;
}

std::string esc_json(const std::string &s)
{
	std::string o;
	o.reserve(s.size() + 8);
	for (char c : s) {
		if (c == '\\' || c == '"') {
			o.push_back('\\');
			o.push_back(c);
		} else if (static_cast<unsigned char>(c) >= 0x20) {
			o.push_back(c);
		}
	}
	return o;
}

} // namespace

EngineLoop::EngineLoop()
{
	std::memset(&snap_, 0, sizeof(snap_));
	std::snprintf(snap_.backend, sizeof(snap_.backend), "dxgi");
}

EngineLoop::~EngineLoop() { stop(); }

void EngineLoop::set_config(const EngineConfig &cfg)
{
	std::lock_guard<std::mutex> lock(cfg_mu_);
	cfg_ = cfg;
}

EngineConfig EngineLoop::config() const
{
	std::lock_guard<std::mutex> lock(cfg_mu_);
	return cfg_;
}

std::string EngineLoop::config_json() const
{
	return ConfigStore::to_json(config()).dump(2);
}

bool EngineLoop::apply_config_json(const std::string &json_body, std::string *err_out)
{
	if (json_body.empty()) {
		if (err_out)
			*err_out = "empty body";
		return false;
	}
	nlohmann::json patch;
	try {
		patch = nlohmann::json::parse(json_body);
	} catch (const std::exception &e) {
		if (err_out)
			*err_out = std::string("json parse: ") + e.what();
		return false;
	}

	EngineConfig c = config();
	if (!ConfigStore::merge_into(c, patch, err_out))
		return false;

	c.infer.confidence = std::clamp(c.infer.confidence, 0.01f, 0.99f);
	c.infer.nms = std::clamp(c.infer.nms, 0.01f, 0.99f);
	c.width = std::clamp(c.width, 64, 4096);
	c.height = std::clamp(c.height, 64, 4096);
	if (c.infer.interval_frames < 1)
		c.infer.interval_frames = 1;
	if (c.infer.interval_frames > 30)
		c.infer.interval_frames = 30;

{
			std::lock_guard<std::mutex> lock(cfg_mu_);
			cfg_ = c;
		}
infer_.set_thresholds(c.infer.confidence, c.infer.nms);
			infer_.set_target_classes(c.infer.target_classes);
			tracker_.set_config(c.tracker);
	#ifdef YA_WITH_AIM
		aim_full_.set_settings(c.aim_full);
	#else
		aim_.set_config(c.aim);
	#endif

// Persist full document immediately after every successful merge (OBS-style user override).
			if (!c.user_config_path.empty()) {
				std::string save_err;
				if (!ConfigStore::save_file(c.user_config_path, c, &save_err)) {
					std::cerr << "[engine] save user.json failed: " << save_err
					          << " path=" << c.user_config_path << std::endl;
				} else {
					std::cout << "[engine] user.json saved: " << c.user_config_path << std::endl;
				}
			} else {
				std::cerr << "[engine] WARN: user_config_path empty — config NOT persisted\n";
			}
			return true;
		}

	bool EngineLoop::save_user_config(std::string *err_out)
	{
		EngineConfig c = config();
		if (c.user_config_path.empty()) {
			if (err_out)
				*err_out = "user_config_path empty";
			return false;
		}
		std::string save_err;
		if (!ConfigStore::save_file(c.user_config_path, c, &save_err)) {
			if (err_out)
				*err_out = save_err;
			std::cerr << "[engine] save_user_config failed: " << save_err << std::endl;
			return false;
		}
		std::cout << "[engine] user.json flushed: " << c.user_config_path << std::endl;
		return true;
	}

	bool EngineLoop::request_reload_model(std::string *err_out)
	{
		if (!running_.load()) {
			if (err_out)
				*err_out = "engine not running";
			return false;
		}
		reload_model_ = true;
		return true;
	}

	bool EngineLoop::request_reload_capture(std::string *err_out)
	{
		if (!running_.load()) {
			if (err_out)
				*err_out = "engine not running";
			return false;
		}
		reload_capture_ = true;
		return true;
	}

	std::vector<ModelEntry> EngineLoop::list_models(const std::string &folder,
		                                                bool recursive) const
		{
			namespace fs = std::filesystem;
			EngineConfig c = config();
			std::vector<std::string> roots;
			if (!folder.empty()) {
				roots.push_back(folder);
			} else {
				roots = c.model_search_dirs;
				if (!c.models_dir.empty())
					roots.insert(roots.begin(), c.models_dir);
				// always try relative models/ next to cwd as last resort
				roots.push_back("models");
			}

			std::vector<ModelEntry> out;
			std::set<std::string> seen;

			auto add_file = [&](const fs::path &file, const fs::path &root_dir) {
				std::error_code ec;
				if (!fs::is_regular_file(file, ec))
					return;
				const auto ext = file.extension().string();
				std::string ext_l = ext;
				std::transform(ext_l.begin(), ext_l.end(), ext_l.begin(),
				               [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
				// Recognize common model packages (ONNX primary; NCNN param/bin pair as name)
				if (ext_l != ".onnx" && ext_l != ".param" && ext_l != ".engine" &&
				    ext_l != ".trt")
					return;
				fs::path abs = fs::weakly_canonical(file, ec);
				if (ec)
					abs = file;
				const std::string key = abs.string();
				if (!seen.insert(key).second)
					return;
				ModelEntry e;
				e.name = file.filename().string();
				e.path = key;
				// relative subdir label under scan root
				std::error_code ec2;
				fs::path rel = fs::relative(file.parent_path(), root_dir, ec2);
				if (!ec2 && !rel.empty() && rel != ".")
					e.dir = rel.string();
				else
					e.dir = root_dir.filename().string();
				if (e.dir.empty())
					e.dir = root_dir.string();
				e.size = static_cast<uint64_t>(fs::file_size(file, ec));
				out.push_back(std::move(e));
			};

			for (const auto &root : roots) {
				if (root.empty())
					continue;
				std::error_code ec;
				fs::path dir(root);
				if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec))
					continue;
				if (recursive) {
					for (auto it = fs::recursive_directory_iterator(
					         dir, fs::directory_options::skip_permission_denied, ec);
					     !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
						if (ec)
							break;
						// skip deep junk / huge trees
						if (it.depth() > 6)
							continue;
						add_file(it->path(), dir);
					}
				} else {
					for (auto it = fs::directory_iterator(dir, ec);
					     !ec && it != fs::directory_iterator(); it.increment(ec)) {
						if (ec)
							break;
						add_file(it->path(), dir);
					}
				}
			}
			std::sort(out.begin(), out.end(),
			          [](const ModelEntry &a, const ModelEntry &b) {
				          if (a.dir != b.dir)
					          return a.dir < b.dir;
				          return a.name < b.name;
			          });
			return out;
		}

		bool EngineLoop::set_model_folder(const std::string &folder, bool recursive,
		                                  std::string *err_out)
		{
			namespace fs = std::filesystem;
			if (folder.empty()) {
				if (err_out)
					*err_out = "empty folder";
				return false;
			}
			std::error_code ec;
			fs::path dir(folder);
			if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
				if (err_out)
					*err_out = "not a directory: " + folder;
				return false;
			}
			fs::path abs = fs::weakly_canonical(dir, ec);
			const std::string path = ec ? dir.string() : abs.string();
			{
				std::lock_guard<std::mutex> lock(cfg_mu_);
				cfg_.models_dir = path;
				auto &dirs = cfg_.model_search_dirs;
				dirs.erase(std::remove(dirs.begin(), dirs.end(), path), dirs.end());
				dirs.insert(dirs.begin(), path);
			}
			// persist if user.json configured
			EngineConfig c = config();
			if (!c.user_config_path.empty()) {
				std::string save_err;
				if (!ConfigStore::save_file(c.user_config_path, c, &save_err)) {
					std::cerr << "[engine] save user.json after set_model_folder: " << save_err
					          << std::endl;
				}
			}
			(void)recursive;
			std::cout << "[engine] model_folder set: " << path << std::endl;
			return true;
		}

	bool EngineLoop::save_model_file(const std::string &filename, const std::vector<uint8_t> &data,
	                                 std::string *out_path, std::string *err_out)
	{
		namespace fs = std::filesystem;
		if (filename.empty() || data.empty()) {
			if (err_out)
				*err_out = "empty filename or body";
			return false;
		}
		// strip path components — only basename
		fs::path name_path(filename);
		std::string base = name_path.filename().string();
		if (base.empty() || base == "." || base == "..") {
			if (err_out)
				*err_out = "invalid filename";
			return false;
		}
		// force .onnx
		std::string lower = base;
		std::transform(lower.begin(), lower.end(), lower.begin(),
		               [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
		if (lower.size() < 5 || lower.substr(lower.size() - 5) != ".onnx")
			base += ".onnx";

		EngineConfig c = config();
		fs::path dir = c.models_dir.empty() ? fs::path("models") : fs::path(c.models_dir);
		std::error_code ec;
		fs::create_directories(dir, ec);
		fs::path dest = dir / base;
		{
			std::ofstream ofs(dest, std::ios::binary);
			if (!ofs) {
				if (err_out)
					*err_out = "cannot open for write: " + dest.string();
				return false;
			}
			ofs.write(reinterpret_cast<const char *>(data.data()),
			          static_cast<std::streamsize>(data.size()));
			if (!ofs) {
				if (err_out)
					*err_out = "write failed";
				return false;
			}
		}
		const std::string abs = fs::weakly_canonical(dest, ec).string();
		if (out_path)
			*out_path = abs.empty() ? dest.string() : abs;
		return true;
	}

bool EngineLoop::start()
	{
		// Re-enable inference when user hits Start (OBS "toggle inference" style).
		{
			std::lock_guard<std::mutex> lock(cfg_mu_);
			cfg_.infer_enabled = true;
		}
		if (running_.load()) {
			reload_model_ = true;
			std::cout << "[engine] start: already running → infer_enabled=1 + reload_model\n";
			return true;
		}
	stop_ = false;
	cap_stop_ = false;
	preview_stop_ = false;
	floating_user_closed_ = false;
	preview_seq_ = 0;
	reload_model_ = true;
	reload_capture_ = false;
last_seen_seq_ = 0;
		capture_frames_ = 0;
	th_ = std::thread([this] { thread_main(); });
	if (preview_th_.joinable())
		preview_th_.join();
	preview_th_ = std::thread([this] { preview_loop(); });
	for (int i = 0; i < 200 && !running_.load(); ++i)
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	std::cout << "[engine] start: thread launched running=" << running_.load()
	          << " (AiMod-style capture + preview threads)\n";
	return true;
	}

	void EngineLoop::capture_loop(FrameSource *source)
	{
		// AiMod CaptureLoop: only grab + publish latest frame; never run YOLO here.
		using clock = std::chrono::steady_clock;
		auto window_start = clock::now();
		uint64_t frames = 0;
		FramePacket tmp;
		while (!cap_stop_.load() && !stop_.load()) {
			if (!source || !source->is_open()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				continue;
			}
			if (!source->grab(tmp)) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}
			{
				std::lock_guard<std::mutex> lock(frame_mu_);
				latest_frame_ = std::move(tmp);
			}
			latest_seq_.fetch_add(1, std::memory_order_release);
			++frames;
			capture_frames_.fetch_add(1, std::memory_order_relaxed);

			const auto now = clock::now();
			const double elapsed = std::chrono::duration<double>(now - window_start).count();
			if (elapsed >= 1.0) {
				capture_fps_fixed_.store(static_cast<uint32_t>((frames / elapsed) * 10.0),
			                                        std::memory_order_relaxed);
				frames = 0;
				window_start = now;
			}
		}
	}

	void EngineLoop::stop()
	{
		std::cout << "[engine] stop requested…\n";
		{
			std::lock_guard<std::mutex> lock(cfg_mu_);
			cfg_.infer_enabled = false;
		}
		cap_stop_ = true;
		stop_ = true;
		preview_stop_ = true;
		// Set running=false BEFORE joining so UI immediately shows STOPPED
		running_ = false;
		if (cap_th_.joinable())
			cap_th_.join();
		if (th_.joinable())
			th_.join();
		// Preview thread joined last so it can destroy the floating HWND on its
		// own thread (Win32 affinity) before FloatingPreview is touched again.
		if (preview_th_.joinable())
			preview_th_.join();
			// Drop model + detections so GPU work fully stops and UI shows OFF.
			infer_.unload();
			{
				std::lock_guard<std::mutex> lock(det_mu_);
				last_dets_.clear();
			}
			{
				std::lock_guard<std::mutex> lock(preview_mu_);
				preview_bmp_.clear();
			}
			{
				std::lock_guard<std::mutex> lock(snap_mu_);
				snap_.running = 0;
				snap_.infer_ok = 0;
				snap_.capture_ok = 0;
				snap_.last_det_count = 0;
				snap_.infer_ms = 0;
				snap_.capture_fps = 0;
				snap_.infer_fps = 0;
			}
		// Floating preview HWND was already destroyed by the preview thread on
		// its way out (Win32 thread affinity). This only flips the stored settings
		// to disabled — no pump() here, the window is gone and must not be touched
		// from this thread.
		{
			FloatingPreview::Settings fs = floating_.settings();
			fs.enabled = false;
			floating_.set_settings(fs);
		}
			// Always persist last runtime config on stop (close / Ctrl+C / destructor).
			std::string err;
			if (!save_user_config(&err) && !err.empty()) {
				// already logged inside save_user_config when path set
			}
			std::cout << "[engine] stop complete (infer unloaded)\n";
		}

EngineSnapshot EngineLoop::snapshot() const
{
	std::lock_guard<std::mutex> lock(snap_mu_);
	return snap_;
}

AimDebug EngineLoop::last_aim_debug() const
{
	std::lock_guard<std::mutex> lock(aim_dbg_mu_);
	return last_aim_dbg_;
}

std::vector<uint8_t> EngineLoop::preview_bmp() const
{
	std::lock_guard<std::mutex> lock(preview_mu_);
	return preview_bmp_;
}

std::string EngineLoop::preview_path() const
{
	std::lock_guard<std::mutex> lock(preview_mu_);
	return preview_path_;
}

std::vector<YoloDet> EngineLoop::last_detections() const
{
	std::lock_guard<std::mutex> lock(det_mu_);
	return last_dets_;
}

#ifdef YA_WITH_AIM
FullAimStatus EngineLoop::aim_status() const { return aim_full_.status(); }

bool EngineLoop::test_controller(const std::string &type, const std::string &makcu_port, int baud,
                                 int logi_type, std::string *err)
{
	return aim_full_.test_controller(FullAimBridge::parse_controller(type), makcu_port, baud,
	                                 logi_type, err);
}
#endif

void EngineLoop::thread_main()
{
	FrameSource source;
	EngineConfig cfg = config();

	const std::string device_requested = cfg.infer.device.empty() ? "cpu" : cfg.infer.device;
std::cout << "[engine] ========== START ==========\n"
			          << "[engine] capture backend=" << FrameSource::backend_name(cfg.capture_backend)
			          << " mode=" << cfg.capture_mode
			          << (cfg.capture_mode == "full"
			                  ? " (全屏捕获)"
			                  : (cfg.capture_mode == "region" ? " (固定区域 ROI)"
			                                                 : " (居中裁切=检测区域/非全屏)"))
			          << " use_region=" << (cfg.use_region ? 1 : 0)
			          << " size=" << cfg.width << "x" << cfg.height
			          << " region=(" << cfg.region_x << "," << cfg.region_y << " "
			          << cfg.region_width << "x" << cfg.region_height << ")\n"
			          << "[engine] infer_enabled=" << (cfg.infer_enabled ? 1 : 0)
			          << " model=" << cfg.infer.model_path
			          << " device_requested=" << device_requested
			          << " version=" << cfg.infer.model_version
			          << " input=" << cfg.infer.input_resolution
			          << " conf=" << cfg.infer.confidence << " nms=" << cfg.infer.nms
			          << " interval=" << cfg.infer.interval_frames << "\n"
			          << "[engine] aim_enabled=" << (cfg.aim_enabled ? 1 : 0) << std::endl;

const bool ok = open_capture_obs_style(source, cfg, std::cout);

			{
				std::lock_guard<std::mutex> lock(snap_mu_);
				snap_.running = 1;
				snap_.capture_ok = ok ? 1 : 0;
				std::snprintf(snap_.backend, sizeof(snap_.backend), "%s",
				              FrameSource::backend_name(cfg.capture_backend));
				std::snprintf(snap_.capture_mode, sizeof(snap_.capture_mode), "%s",
				              cfg.capture_mode.c_str());
				std::snprintf(snap_.device_req, sizeof(snap_.device_req), "%s",
				              device_requested.c_str());
				std::snprintf(snap_.device_act, sizeof(snap_.device_act), "%s",
				              device_requested.c_str());
				std::snprintf(snap_.model_path, sizeof(snap_.model_path), "%s",
				              cfg.infer.model_path.c_str());
				snap_.capture_w = ok ? source.width() : cfg.width;
				snap_.capture_h = ok ? source.height() : cfg.height;
				snap_.region_x = cfg.region_x;
				snap_.region_y = cfg.region_y;
			if (!ok)
				std::snprintf(snap_.last_error, sizeof(snap_.last_error), "%s",
				              source.last_error().c_str());
		}

		if (!ok) {
			std::cerr << "[engine] CAPTURE FAILED: " << source.last_error() << std::endl;
			running_ = false;
			std::lock_guard<std::mutex> lock(snap_mu_);
			snap_.running = 0;
			return;
		}

std::cout << "[engine] capture opened OK " << source.width() << "x" << source.height()
			          << " mode=" << cfg.capture_mode << " use_region=" << (cfg.use_region ? 1 : 0)
			          << " → DXGI center/region ROI (GPU CopySubresourceRegion)"
			          << " region=" << cfg.region_width << "x" << cfg.region_height
			          << std::endl;

			// AiMod: dedicated capture thread — grab never blocked by YOLO
			cap_stop_ = false;
			if (cap_th_.joinable())
				cap_th_.join();
			cap_th_ = std::thread([this, &source] { capture_loop(&source); });
			std::cout << "[engine] capture thread started (AiMod dual-thread)\n";

			bool infer_ok = false;
		if (cfg.infer_enabled) {
			std::cout << "[engine] loading model: " << cfg.infer.model_path
			          << " device_requested=" << device_requested << std::endl;
if (infer_.load(cfg.infer)) {
					infer_ok = true;
					const auto ic = infer_.config();
					auto ep_base = [](std::string s) {
						const auto p = s.find('+');
						if (p != std::string::npos)
							s = s.substr(0, p);
						return s;
					};
					const bool ep_ok = ep_base(ic.device) == ep_base(device_requested) ||
					                   (device_requested == "cuda" &&
					                    ic.device.find("cuda") == 0);
					std::cout << "[engine] model ready actual_device=" << ic.device
					          << (ep_ok ? " (as requested)"
					                    : " (FALLBACK from " + device_requested + ")")
					          << " note=" << (infer_.last_error().empty() ? "ok" : infer_.last_error())
					          << std::endl;
				std::lock_guard<std::mutex> lock(snap_mu_);
				std::snprintf(snap_.device_act, sizeof(snap_.device_act), "%s", ic.device.c_str());
				std::snprintf(snap_.device_req, sizeof(snap_.device_req), "%s",
				              device_requested.c_str());
				std::snprintf(snap_.model_path, sizeof(snap_.model_path), "%s",
				              ic.model_path.c_str());
				if (!infer_.last_error().empty())
					std::snprintf(snap_.last_error, sizeof(snap_.last_error), "%s",
					              infer_.last_error().c_str());
			} else {
				std::cerr << "[engine] model load FAILED: " << infer_.last_error() << std::endl;
				std::lock_guard<std::mutex> lock(snap_mu_);
				std::snprintf(snap_.last_error, sizeof(snap_.last_error), "%s",
				              infer_.last_error().c_str());
			}
		} else {
			std::cout << "[engine] infer DISABLED (infer_enabled=0)" << std::endl;
		}

	tracker_.set_config(cfg.tracker);
	tracker_.reset();

#ifdef YA_WITH_AIM
		{
			FullAimSettings as = cfg.aim_full;
			as.enabled = cfg.aim_enabled;
			aim_full_.set_settings(as);
			aim_full_.ensure_controller();
			std::cout << "[engine] FULL AIM stack enabled=" << (cfg.aim_enabled ? 1 : 0)
			          << " algo=" << FullAimBridge::algorithm_name(as.algorithm)
			          << " (8 backends linked)" << std::endl;
		}
		CrosshairDetector crosshair;
		CrosshairDetectorConfig last_xh_cfg{};
		int crosshair_frame_i = 0;
	#else
		{
			AimConfig ac = cfg.aim;
			ac.enabled = cfg.aim_enabled;
			aim_.set_config(ac);
			aim_.reset();
		}
	#endif

		{
			std::lock_guard<std::mutex> lock(snap_mu_);
			snap_.infer_ok = infer_ok ? 1 : 0;
			snap_.aim_ok = cfg.aim_enabled ? 1 : 0;
		}

		running_ = true;
		frame_count_ = 0;

		using clock = std::chrono::steady_clock;
		auto window_start = clock::now();
		int frames = 0;
		int infer_frames = 0;
		double fps = 0.0;
		double infer_fps = 0.0;
		double last_infer_ms = 0.0;
		int interval = std::max(1, cfg.infer.interval_frames);
		std::vector<Detection> last_tracked;
		int fov_draw = cfg.aim_enabled ?
	#ifdef YA_WITH_AIM
		                                 cfg.aim_full.fov_radius
	#else
		                                 cfg.aim.fov_radius
	#endif
		                               : 0;

		FramePacket packet;
		while (!stop_.load()) {
			// hot reload
			{
				EngineConfig live = config();
				cfg.aim_enabled = live.aim_enabled;
// Hot thresholds / class filter every loop (cheap; model already loaded)
					if (std::fabs(cfg.infer.confidence - live.infer.confidence) > 1e-6f ||
					    std::fabs(cfg.infer.nms - live.infer.nms) > 1e-6f) {
						infer_.set_thresholds(live.infer.confidence, live.infer.nms);
					}
					if (cfg.infer.target_classes != live.infer.target_classes) {
						infer_.set_target_classes(live.infer.target_classes);
					}
					cfg.infer.confidence = live.infer.confidence;
					cfg.infer.nms = live.infer.nms;
					cfg.infer.interval_frames = live.infer.interval_frames;
					cfg.infer.model_path = live.infer.model_path;
					cfg.infer.device = live.infer.device;
					cfg.infer.model_version = live.infer.model_version;
					cfg.infer.input_resolution = live.infer.input_resolution;
					cfg.infer.num_threads = live.infer.num_threads;
					cfg.infer.target_classes = live.infer.target_classes;
					cfg.infer.target_classes_text = live.infer.target_classes_text;
					cfg.infer_enabled = live.infer_enabled;
					cfg.tracker = live.tracker;
					cfg.vision = live.vision;
					cfg.capture_backend = live.capture_backend;
					cfg.capture_mode = live.capture_mode;
					cfg.width = live.width;
					cfg.height = live.height;
cfg.use_region = live.use_region;
						cfg.region_x = live.region_x;
						cfg.region_y = live.region_y;
						cfg.region_width = live.region_width;
						cfg.region_height = live.region_height;
						cfg.preview_interval_sec = live.preview_interval_sec;
					interval = std::max(1, cfg.infer.interval_frames);
					tracker_.set_config(cfg.tracker);
		#ifdef YA_WITH_AIM
					cfg.aim_full = live.aim_full;
					cfg.aim_full.enabled = live.aim_enabled;
					aim_full_.set_settings(cfg.aim_full);
					fov_draw = cfg.aim_full.fov_radius;
					if (!cfg.aim_full.show_fov)
						fov_draw = 0;
		#else
					cfg.aim = live.aim;
					cfg.aim.enabled = live.aim_enabled;
					aim_.set_config(cfg.aim);
					fov_draw = cfg.aim.fov_radius;
		#endif
				}

			// Cold reloads (model / capture)
				if (reload_model_.exchange(false)) {
					const std::string req =
					    cfg.infer.device.empty() ? "cpu" : cfg.infer.device;
					std::cout << "[engine] reload_model: " << cfg.infer.model_path
					          << " device_requested=" << req << std::endl;
					infer_.unload();
					infer_ok = false;
					if (cfg.infer_enabled) {
if (infer_.load(cfg.infer)) {
								infer_ok = true;
								const auto ic = infer_.config();
								auto ep_base = [](std::string s) {
									const auto p = s.find('+');
									if (p != std::string::npos)
										s = s.substr(0, p);
									return s;
								};
								const bool ep_ok = ep_base(ic.device) == ep_base(req) ||
								                   (req == "cuda" && ic.device.find("cuda") == 0);
								std::cout << "[engine] model reloaded OK actual_device=" << ic.device
								          << (ep_ok ? " (as requested)" : " (FALLBACK)")
								          << " note="
								          << (infer_.last_error().empty() ? "ok" : infer_.last_error())
								          << std::endl;
							std::lock_guard<std::mutex> lock(snap_mu_);
							std::snprintf(snap_.device_req, sizeof(snap_.device_req), "%s",
							              req.c_str());
							std::snprintf(snap_.device_act, sizeof(snap_.device_act), "%s",
							              ic.device.c_str());
							std::snprintf(snap_.model_path, sizeof(snap_.model_path), "%s",
							              ic.model_path.c_str());
							if (!infer_.last_error().empty())
								std::snprintf(snap_.last_error, sizeof(snap_.last_error), "%s",
								              infer_.last_error().c_str());
						} else {
							std::cerr << "[engine] model reload FAILED: " << infer_.last_error()
							          << std::endl;
						}
					}
					{
						std::lock_guard<std::mutex> lock(snap_mu_);
						snap_.infer_ok = infer_ok ? 1 : 0;
						if (!infer_ok)
							std::snprintf(snap_.last_error, sizeof(snap_.last_error), "%s",
							              infer_.last_error().c_str());
					}
				}

if (reload_capture_.exchange(false)) {
					std::cout << "[engine] reload_capture backend="
					          << FrameSource::backend_name(cfg.capture_backend)
					          << " mode=" << cfg.capture_mode
					          << " use_region=" << (cfg.use_region ? 1 : 0)
					          << " size=" << cfg.width << "x" << cfg.height
					          << " region=(" << cfg.region_x << "," << cfg.region_y << ","
					          << cfg.region_width << "x" << cfg.region_height << ")"
					          << std::endl;
					// Pause capture thread while reopening DXGI
					cap_stop_ = true;
					if (cap_th_.joinable())
						cap_th_.join();
					source.close();
					const bool cap_ok = open_capture_obs_style(source, cfg, std::cout);
					{
						std::lock_guard<std::mutex> lock(snap_mu_);
						snap_.capture_ok = cap_ok ? 1 : 0;
						std::snprintf(snap_.backend, sizeof(snap_.backend), "%s",
						              FrameSource::backend_name(cfg.capture_backend));
						std::snprintf(snap_.capture_mode, sizeof(snap_.capture_mode), "%s",
						              cfg.capture_mode.c_str());
						snap_.capture_w = source.width();
						snap_.capture_h = source.height();
						snap_.region_x = cfg.region_x;
						snap_.region_y = cfg.region_y;
						if (!cap_ok)
							std::snprintf(snap_.last_error, sizeof(snap_.last_error), "%s",
							              source.last_error().c_str());
					}
					if (!cap_ok) {
						std::cerr << "[engine] capture reload FAILED: " << source.last_error()
						          << std::endl;
						std::this_thread::sleep_for(std::chrono::milliseconds(50));
						continue;
					}
					std::cout << "[engine] capture reloaded OK " << source.width() << "x"
					          << source.height() << std::endl;
					cap_stop_ = false;
					cap_th_ = std::thread([this, &source] { capture_loop(&source); });
				}

				// Take latest frame from capture thread (drop intermediate frames)
				const uint64_t seq = latest_seq_.load(std::memory_order_acquire);
				if (seq == 0 || seq == last_seen_seq_) {
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					continue;
				}
				last_seen_seq_ = seq;
				{
					std::lock_guard<std::mutex> lock(frame_mu_);
					packet = latest_frame_; // copy for process (capture may overwrite next)
				}
				if (packet.bgr.empty() || packet.width <= 0 || packet.height <= 0) {
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					continue;
				}

				const uint64_t n = ++frame_count_;
				++frames;

// Hot-toggle: infer_enabled must gate every frame (Stop / Web checkbox).
					// infer_ok only means model is loaded — without this, turning off still runs YOLO.
					if (!cfg.infer_enabled) {
						if (!last_tracked.empty() || last_infer_ms > 0) {
							last_tracked.clear();
							last_infer_ms = 0;
							std::lock_guard<std::mutex> lock(det_mu_);
							last_dets_.clear();
						}
					} else if (infer_ok && (n % static_cast<uint64_t>(interval) == 0)) {
						// OBS filter_inference path:
						//   1) resolve ROI (use_region crop on full frame, or whole packet if already ROI)
						//   2) YOLO on ROI pixels only
						//   3) remap dets back to full packet coords when software-cropped
						int crop_x = 0, crop_y = 0, crop_w = packet.width, crop_h = packet.height;
						resolve_infer_roi(cfg, packet.width, packet.height, packet.origin_x,
						                  packet.origin_y, crop_x, crop_y, crop_w, crop_h);

						const int stride = packet.width * 3;
						const uint8_t *roi_ptr =
						    packet.bgr.data() +
						    (static_cast<size_t>(crop_y) * static_cast<size_t>(packet.width) +
						     static_cast<size_t>(crop_x)) *
						        3u;
						// When ROI is a sub-rect, rows are not contiguous → pass full stride
						// InferEngine copies row-by-row when stride != w*3.
						auto ir = infer_.run_bgr(roi_ptr, crop_w, crop_h, stride);
						if (ir.ok) {
							// Remap ROI-normalized dets → full-frame (packet) normalized (OBS)
							remap_dets_roi_to_full(ir.dets, crop_x, crop_y, crop_w, crop_h,
							                       packet.width, packet.height);
							last_tracked = tracker_.update(ir.dets);
							last_infer_ms = ir.infer_ms;
							++infer_frames;
							// After remap, dets live in packet frame; crop offsets for aim are 0
							// (same as OBS after coord transform zeros effective crop for controller).
							std::vector<YoloDet> cdets;
							cdets.reserve(last_tracked.size());
							for (const auto &d : last_tracked)
								cdets.push_back(to_c_det(d));
							std::lock_guard<std::mutex> lock(det_mu_);
							last_dets_ = std::move(cdets);
						}
					}

			// Aim
			AimDebug adbg{};
			if (cfg.aim_enabled) {
	#ifdef YA_WITH_AIM
				// CrosshairDetector → aim origin (pixel coords; -1 = frame center)
				if (cfg.aim_full.crosshair_enabled) {
					CrosshairDetectorConfig xcfg;
					xcfg.enabled = true;
					xcfg.hMin = cfg.aim_full.crosshair_h_min;
					xcfg.hMax = cfg.aim_full.crosshair_h_max;
					xcfg.sMin = cfg.aim_full.crosshair_s_min;
					xcfg.sMax = cfg.aim_full.crosshair_s_max;
					xcfg.vMin = cfg.aim_full.crosshair_v_min;
					xcfg.vMax = cfg.aim_full.crosshair_v_max;
					xcfg.hTolerance = cfg.aim_full.crosshair_h_tolerance;
					xcfg.sTolerance = cfg.aim_full.crosshair_s_tolerance;
					xcfg.vTolerance = cfg.aim_full.crosshair_v_tolerance;
					xcfg.manualR = cfg.aim_full.crosshair_manual_r;
					xcfg.manualG = cfg.aim_full.crosshair_manual_g;
					xcfg.manualB = cfg.aim_full.crosshair_manual_b;
					xcfg.searchRadius = cfg.aim_full.crosshair_search_radius;
					xcfg.detectEveryNFrames =
					    std::max(1, cfg.aim_full.crosshair_detect_interval);
					xcfg.morphKernelSize = cfg.aim_full.crosshair_morph_kernel;
					xcfg.erodeIterations = cfg.aim_full.crosshair_erode_iter;
					xcfg.dilateIterations = cfg.aim_full.crosshair_dilate_iter;
					xcfg.gridRows = cfg.aim_full.crosshair_grid_rows;
					xcfg.gridCols = cfg.aim_full.crosshair_grid_cols;
					xcfg.quantileThreshold = cfg.aim_full.crosshair_quantile_threshold;
					xcfg.templateImagePath = cfg.aim_full.crosshair_template_path;
					xcfg.matchThreshold = cfg.aim_full.crosshair_match_threshold;
					xcfg.minArea = cfg.aim_full.crosshair_min_area;
					xcfg.maxArea = cfg.aim_full.crosshair_max_area;
					xcfg.shapeFilterEnabled = cfg.aim_full.crosshair_shape_filter_enabled;
					xcfg.shapeType = cfg.aim_full.crosshair_shape_type;
					xcfg.minFillRatio = cfg.aim_full.crosshair_min_fill_ratio;
					xcfg.maxFillRatio = cfg.aim_full.crosshair_max_fill_ratio;
					xcfg.minAspectRatio = cfg.aim_full.crosshair_min_aspect_ratio;
					xcfg.maxAspectRatio = cfg.aim_full.crosshair_max_aspect_ratio;
xcfg.colorIsolationView = cfg.aim_full.crosshair_color_isolation;
						xcfg.showDebugMask = cfg.aim_full.crosshair_debug_mask;
						// Prefer explicit HSV; seed from manual RGB only when HSV is still default
						// and RGB just changed / first enable.
						const bool hsv_is_default =
						    cfg.aim_full.crosshair_h_min == 0 && cfg.aim_full.crosshair_h_max == 180 &&
						    cfg.aim_full.crosshair_s_min == 100 && cfg.aim_full.crosshair_s_max == 255 &&
						    cfg.aim_full.crosshair_v_min == 100 && cfg.aim_full.crosshair_v_max == 255;
						const bool rgb_changed =
						    xcfg.manualR != last_xh_cfg.manualR || xcfg.manualG != last_xh_cfg.manualG ||
						    xcfg.manualB != last_xh_cfg.manualB || !last_xh_cfg.enabled;
						if (hsv_is_default && rgb_changed &&
						    (xcfg.manualR | xcfg.manualG | xcfg.manualB) != 0) {
							crosshair.applyManualRgb(xcfg.manualR, xcfg.manualG, xcfg.manualB);
							auto seeded = crosshair.getConfig();
							xcfg.hMin = seeded.hMin;
							xcfg.hMax = seeded.hMax;
							xcfg.sMin = seeded.sMin;
							xcfg.sMax = seeded.sMax;
							xcfg.vMin = seeded.vMin;
							xcfg.vMax = seeded.vMax;
						}
						crosshair.updateConfig(xcfg);
						last_xh_cfg = xcfg;

					++crosshair_frame_i;
					if (crosshair_frame_i % xcfg.detectEveryNFrames == 0) {
						const int stride = packet.width * 3;
						cv::Mat frame(packet.height, packet.width, CV_8UC3,
						              const_cast<uint8_t *>(packet.bgr.data()),
						              static_cast<size_t>(stride));
						const float fov_norm =
						    (packet.width > 0)
						        ? (static_cast<float>(fov_draw) / static_cast<float>(packet.width))
						        : 0.5f;
						auto xh = crosshair.detect(frame, packet.width, packet.height, 0, 0, 0.5f,
						                           0.5f, fov_norm);
						if (!xh.empty()) {
							// detect returns normalized center → pixel for setAimOrigin
							cfg.aim_full.aim_origin_x = xh[0].centerX * packet.width;
							cfg.aim_full.aim_origin_y = xh[0].centerY * packet.height;
						} else {
							cfg.aim_full.aim_origin_x = -1.f;
							cfg.aim_full.aim_origin_y = -1.f;
						}
						aim_full_.set_settings(cfg.aim_full);
					}
				} else if (cfg.aim_full.aim_origin_x >= 0.f || cfg.aim_full.aim_origin_y >= 0.f) {
					cfg.aim_full.aim_origin_x = -1.f;
					cfg.aim_full.aim_origin_y = -1.f;
					aim_full_.set_settings(cfg.aim_full);
				}

				aim_full_.tick(last_tracked, packet.width, packet.height, 0, 0,
				               static_cast<float>(last_infer_ms));
				auto st = aim_full_.status();
				adbg.det_count = static_cast<int32_t>(last_tracked.size());
				adbg.target_track_id = st.active_slot;
adbg.capture_fps =
					    static_cast<float>(capture_fps_fixed_.load(std::memory_order_relaxed)) / 10.f;
					if (adbg.capture_fps <= 0.f)
						adbg.capture_fps = static_cast<float>(fps);
					adbg.infer_ms = last_infer_ms;
				fov_draw = st.fov_px > 0 ? st.fov_px : fov_draw;
	#else
				std::vector<YoloDet> cdets;
				for (const auto &d : last_tracked)
					cdets.push_back(to_c_det(d));
				AimFrameMeta meta{};
				meta.frame_w = packet.width;
				meta.frame_h = packet.height;
				meta.origin_x = packet.origin_x;
				meta.origin_y = packet.origin_y;
				aim_.tick(cdets, meta, &adbg);
	#endif
				std::lock_guard<std::mutex> lock(aim_dbg_mu_);
				last_aim_dbg_ = adbg;
			}

const auto now = clock::now();
			const double elapsed = std::chrono::duration<double>(now - window_start).count();
			if (elapsed >= 1.0) {
				// process-thread rate vs real capture rate (AiMod dual-thread)
				const double proc_fps = frames / elapsed;
				const double cap_fps = static_cast<double>(capture_fps_fixed_.load(std::memory_order_relaxed)) / 10.0;
				fps = cap_fps > 0.0 ? cap_fps : proc_fps;
				infer_fps = infer_frames / elapsed;
				frames = 0;
				infer_frames = 0;
				window_start = now;
				std::cout << "[engine] frames=" << n << " cap_fps=" << fps
				          << " proc_fps=" << proc_fps
				          << " infer_fps=" << infer_fps << " infer_ms=" << last_infer_ms
				          << " dets=" << last_tracked.size()
				          << " roi=" << packet.width << "x" << packet.height
				          << "@(" << packet.origin_x << "," << packet.origin_y << ")"
				          << " device=" << (infer_ok ? infer_.config().device : "n/a");
#ifdef YA_WITH_AIM
			if (cfg.aim_enabled) {
				auto st = aim_full_.status();
				std::cout << " aim_slot=" << st.active_slot
				          << " ctrl=" << FullAimBridge::controller_name(
				                           static_cast<ControllerType>(st.controller_type));
			}
#endif
			std::cout << " size=" << packet.width << "x" << packet.height << std::endl;
		}

			// --- Preview offload: hand frame + dets to the preview thread (AiMod Display
			//     thread). No draw / encode / pump / json on the hot infer loop. ---
			{
				// Floating window closed by user? (flagged by preview thread's pump)
				if (floating_user_closed_.exchange(false)) {
					cfg.vision.show_floating_window = false;
					{
						std::lock_guard<std::mutex> lock(cfg_mu_);
						cfg_.vision.show_floating_window = false;
					}
					save_user_config(nullptr);
				}
				PreviewJob job;
				job.bgr = std::move(packet.bgr); // packet.bgr unused after this point
				job.width = packet.width;
				job.height = packet.height;
				job.origin_x = packet.origin_x;
				job.origin_y = packet.origin_y;
				{
					std::vector<YoloDet> det_c;
					det_c.reserve(last_tracked.size());
					for (auto &d : last_tracked)
						det_c.push_back(to_c_det(d));
					job.dets = std::move(det_c);
				}
				job.fps = fps;
				job.infer_ms = last_infer_ms;
				job.fov_px = fov_draw;
				job.preview_interval = cfg.preview_interval_sec;
				job.vision = cfg.vision;
#ifdef YA_WITH_AIM
				job.aim_full = cfg.aim_full;
				job.aim_enabled = cfg.aim_enabled;
#endif
				{
					std::lock_guard<std::mutex> lock(preview_job_mu_);
					preview_job_ = std::move(job);
				}
				preview_seq_.fetch_add(1, std::memory_order_release);
			}

		{
				std::lock_guard<std::mutex> lock(snap_mu_);
snap_.running = 1;
					snap_.capture_ok = 1;
					// infer_ok = model loaded; infer_enabled gates actual YOLO each frame
					snap_.infer_ok = (infer_ok && cfg.infer_enabled) ? 1 : 0;
					snap_.aim_ok = cfg.aim_enabled ? 1 : 0;
				snap_.capture_fps = fps;
				snap_.infer_fps = infer_fps;
				snap_.infer_ms = last_infer_ms;
				snap_.last_det_count = static_cast<int32_t>(last_tracked.size());
				snap_.capture_w = packet.width > 0 ? packet.width : source.width();
				snap_.capture_h = packet.height > 0 ? packet.height : source.height();
				snap_.origin_x = packet.origin_x;
				snap_.origin_y = packet.origin_y;
				std::snprintf(snap_.capture_mode, sizeof(snap_.capture_mode), "%s",
				              cfg.capture_mode.c_str());
#ifdef YA_WITH_AIM
				auto st = aim_full_.status();
				snap_.aim_hotkey_down = st.hotkey_down ? 1 : 0;
				snap_.aim_moved = st.aiming ? 1 : 0;
#else
				snap_.aim_hotkey_down = aim_.last_hotkey_down() ? 1 : 0;
				snap_.aim_moved = aim_.last_moved() ? 1 : 0;
				snap_.aim_err_x = adbg.error_x;
				snap_.aim_err_y = adbg.error_y;
				snap_.aim_out_x = adbg.out_x;
				snap_.aim_out_y = adbg.out_y;
#endif
				// Keep GPU-fallback note; only clear empty/hard failures when healthy.
				if (infer_ok && snap_.last_error[0] &&
				    std::strncmp(snap_.last_error, "GPU failed", 10) != 0) {
					snap_.last_error[0] = '\0';
				}
			}
	}

// Stop capture thread before tearing down DXGI (AiMod order)
		cap_stop_ = true;
		if (cap_th_.joinable())
			cap_th_.join();
		infer_.unload();
		source.close();
		{
			std::lock_guard<std::mutex> lock(snap_mu_);
			snap_.running = 0;
		}
		std::cout << "[engine] stopped, total frames=" << frame_count_.load()
		          << " capture_frames=" << capture_frames_.load() << std::endl;
		running_ = false;
	}

void EngineLoop::preview_loop()
{
	using clock = std::chrono::steady_clock;
	auto last_encode = clock::now();
	bool first_saved = false;
	uint64_t last_seq = 0;

	while (!preview_stop_.load() && !stop_.load()) {
		// Win32 HWND thread affinity: the floating window must be created, pumped
		// and destroyed on THIS thread. pump() drains only our window's messages.
		floating_.pump();
		if (floating_.consume_user_closed())
			floating_user_closed_.store(true, std::memory_order_release);

		const uint64_t seq = preview_seq_.load(std::memory_order_acquire);
		if (seq == last_seq) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		PreviewJob job;
		{
			std::lock_guard<std::mutex> lock(preview_job_mu_);
			job = preview_job_;
		}
		last_seq = seq;

		if (job.bgr.empty() || job.width <= 0 || job.height <= 0)
			continue;

		// Throttle encode to ~preview_interval (default 30 FPS).
		const double iv = std::clamp(
		    job.preview_interval > 0.0 ? job.preview_interval : 0.033, 0.016, 0.5);
		const auto now = clock::now();
		if (std::chrono::duration<double>(now - last_encode).count() < iv)
			continue;
		last_encode = now;

		// Mirror floating-window settings from the latest config snapshot.
		FloatingPreview::Settings fs;
		fs.enabled = job.vision.show_floating_window;
		fs.width = job.vision.floating_window_width;
		fs.height = job.vision.floating_window_height;
		fs.show_track_id = job.vision.show_track_id_in_floating_window;
		fs.show_stats = true;
		floating_.set_settings(fs);

		// Skip expensive draw/encode when neither web preview nor floating window
		// is enabled — saves CPU when the user closed the web preview.
		const bool need_vis = job.vision.preview_enabled || job.vision.show_floating_window;
		if (!need_vis)
			continue;

		std::vector<uint8_t> vis = std::move(job.bgr); // draw on the moved buffer
		const int line_w = std::max(1, std::min(10, job.vision.bbox_line_width));
		if (job.vision.show_detection_results) {
			const size_t max_draw = 64;
			const size_t n_draw = std::min(max_draw, job.dets.size());
			for (size_t i = 0; i < n_draw; ++i) {
				const auto &d = job.dets[i];
				const int x1 = static_cast<int>(d.x * job.width);
				const int y1 = static_cast<int>(d.y * job.height);
				const int x2 = static_cast<int>((d.x + d.w) * job.width);
				const int y2 = static_cast<int>((d.y + d.h) * job.height);
				const uint8_t cr = static_cast<uint8_t>((d.class_id * 67) % 200 + 55);
				const uint8_t cg = static_cast<uint8_t>((d.class_id * 97) % 200 + 55);
				const uint8_t cb = static_cast<uint8_t>((d.class_id * 37) % 200 + 55);
				draw_box_bgr(vis, job.width, job.height, x1, y1, x2, y2, cr, cg, cb,
				             line_w);
				char lab[48];
				std::snprintf(lab, sizeof(lab), "c%d %.0f%%",
				              d.class_id, d.confidence * 100.f);
				draw_label_bgr(vis, job.width, job.height, x1, y1, lab, 255, 255, 255);
			}
		}
#ifdef YA_WITH_AIM
		const bool draw_fov_circle =
		    job.fov_px > 0 && job.aim_full.show_fov && job.aim_full.show_fov_circle;
		const bool draw_fov_cross =
		    job.fov_px > 0 && job.aim_full.show_fov && job.aim_full.show_fov_cross;
		if (draw_fov_circle) {
			draw_circle_bgr(vis, job.width, job.height, job.width / 2,
			                job.height / 2, job.fov_px, 61, 139, 253);
		}
		if (draw_fov_cross) {
			const int cx = job.width / 2;
			const int cy = job.height / 2;
			const int arm = std::max(
			    8, job.fov_px * std::max(1, job.aim_full.fov_cross_line_scale) / 100);
			const int t = std::max(1, job.aim_full.fov_cross_line_thickness);
			draw_box_bgr(vis, job.width, job.height, cx - arm, cy - t, cx + arm, cy + t,
			             61, 139, 253, 1);
			draw_box_bgr(vis, job.width, job.height, cx - t, cy - arm, cx + t, cy + arm,
			             61, 139, 253, 1);
		}
#else
		if (job.fov_px > 0) {
			draw_circle_bgr(vis, job.width, job.height, job.width / 2, job.height / 2,
			                job.fov_px, 61, 139, 253);
		}
#endif
		// Optional coordinate export (throttled to preview rate). Resolved relative
		// to cwd here (main loop's user_config_path not forwarded); absolute paths
		// work as before.
		if (job.vision.export_coordinates && !job.dets.empty()) {
			namespace fs2 = std::filesystem;
			fs2::path out = job.vision.coordinate_output_path.empty()
			                    ? fs2::path("detections.json")
			                    : fs2::path(job.vision.coordinate_output_path);
			std::ofstream ofs(out, std::ios::trunc);
			if (ofs) {
				ofs << "{\"count\":" << job.dets.size() << ",\"items\":[";
				for (size_t i = 0; i < job.dets.size(); ++i) {
					const auto &d = job.dets[i];
					if (i)
						ofs << ',';
ofs << "{\"class_id\":" << d.class_id
				    << ",\"confidence\":" << d.confidence << ",\"x\":" << d.x
				    << ",\"y\":" << d.y << ",\"w\":" << d.w << ",\"h\":" << d.h
				    << ",\"cx\":" << d.cx << ",\"cy\":" << d.cy
				    << ",\"track_id\":" << d.track_id << '}';
				}
				ofs << "]}";
			}
		}
		if (job.vision.preview_enabled) {
			std::vector<uint8_t> bmp;
			if (bgr_to_bmp_bytes(vis.data(), job.width, job.height, bmp)) {
				std::lock_guard<std::mutex> lock(preview_mu_);
				preview_bmp_ = std::move(bmp);
				if (!first_saved) {
					std::cout << "[engine] FIRST PREVIEW ready (preview thread, " << job.width
					          << "x" << job.height << ")" << std::endl;
					first_saved = true;
				}
			}
		}
		if (job.vision.show_floating_window) {
			char stats[192];
			std::snprintf(stats, sizeof(stats),
			              "FPS:%.1f | Infer:%.1fms | Dets:%zu | %dx%d @(%d,%d)", job.fps,
			              job.infer_ms, job.dets.size(), job.width, job.height,
			              job.origin_x, job.origin_y);
			floating_.push_bgr(vis.data(), job.width, job.height, stats);
		}
	}

	// Cleanup: destroy the floating HWND on THIS thread (Win32 affinity) before
	// stop() returns, so ~FloatingPreview never touches it from another thread.
	FloatingPreview::Settings off;
	off.enabled = false;
	floating_.set_settings(off);
	floating_.pump();
}

} // namespace ya
