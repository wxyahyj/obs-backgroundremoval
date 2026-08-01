#include "EngineLoop.hpp"
#include "ConfigStore.hpp"
#include "WebServer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <shellapi.h>
#endif

namespace fs = std::filesystem;

static fs::path exe_dir()
{
#ifdef _WIN32
	wchar_t buf[MAX_PATH];
	GetModuleFileNameW(nullptr, buf, MAX_PATH);
	return fs::path(buf).parent_path();
#else
	return fs::current_path();
#endif
}

#ifdef _WIN32
// ORT CUDA EP needs cudnn64_9 + cublas/cudart on the DLL search path.
// Prefer env override, then common local toolkit layouts, then exe-local cuda/.
static void prepend_dll_dir(const fs::path &dir)
{
	std::error_code ec;
	if (dir.empty() || !fs::is_directory(dir, ec))
		return;
	const std::wstring w = fs::weakly_canonical(dir, ec).wstring();
	if (w.empty())
		return;
// PATH prepend is reliable for ORT provider_bridge (multi-dir). Avoid SetDllDirectory —
		// it only allows a single directory and would clobber prior adds.
		wchar_t oldPath[32768];
		const DWORD n = GetEnvironmentVariableW(L"PATH", oldPath, 32768);
		std::wstring neu = w + L";";
		if (n > 0 && n < 32768)
			neu += oldPath;
		SetEnvironmentVariableW(L"PATH", neu.c_str());
		std::wcout << L"[host] CUDA DLL dir: " << w << L"\n";
	}

static void bootstrap_cuda_runtime_path(const fs::path &base)
{
	// 1) explicit env
	if (const char *e = std::getenv("YA_CUDA_PATH")) {
		if (e[0])
			prepend_dll_dir(fs::path(e));
	}
	if (const char *e = std::getenv("CUDNN_PATH")) {
		if (e[0]) {
			fs::path p(e);
			// allow pointing at cudnn root or bin/x64
			if (fs::exists(p / "cudnn64_9.dll"))
				prepend_dll_dir(p);
			else if (fs::exists(p / "bin" / "cudnn64_9.dll"))
				prepend_dll_dir(p / "bin");
			else if (fs::exists(p / "bin" / "12.9" / "x64" / "cudnn64_9.dll"))
				prepend_dll_dir(p / "bin" / "12.9" / "x64");
			else
				prepend_dll_dir(p);
		}
	}

// 2) next to exe (POST_BUILD stages cudnn/cudart here) + cuda/cudnn subdirs
		const fs::path local_candidates[] = {
		    base,
		    base / "cuda",
		    base / "cudnn",
		    base / "cuda" / "bin",
		    base / "cudnn" / "bin",
		};

	// 3) well-known developer layouts on this machine / CUDAToolkit
	const fs::path kit_candidates[] = {
	    fs::path("D:/CUDATool/bin"),
	    fs::path("D:/CUDATool/cudnn/bin/12.9/x64"),
	    fs::path("D:/CUDATool/cudnn/bin/13.1/x64"),
	    fs::path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin"),
	    fs::path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"),
	    fs::path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin"),
	};

	auto try_add = [](const fs::path &p) {
		std::error_code ec;
		if (fs::exists(p / "cudnn64_9.dll", ec) || fs::exists(p / "cudart64_12.dll", ec) ||
		    fs::exists(p / "cublas64_12.dll", ec)) {
			prepend_dll_dir(p);
			return true;
		}
		return false;
	};

	bool got_cudnn = false;
	bool got_cudart = false;
	for (const auto &p : local_candidates) {
		if (try_add(p)) {
			if (fs::exists(p / "cudnn64_9.dll"))
				got_cudnn = true;
			if (fs::exists(p / "cudart64_12.dll") || fs::exists(p / "cublas64_12.dll"))
				got_cudart = true;
		}
	}
	for (const auto &p : kit_candidates) {
		if (try_add(p)) {
			if (fs::exists(p / "cudnn64_9.dll"))
				got_cudnn = true;
			if (fs::exists(p / "cudart64_12.dll") || fs::exists(p / "cublas64_12.dll"))
				got_cudart = true;
		}
	}

	// Probe: can we load cudnn now?
	HMODULE h = LoadLibraryW(L"cudnn64_9.dll");
	if (h) {
		std::cout << "[host] cudnn64_9.dll LOAD OK → CUDA EP should work\n";
		FreeLibrary(h);
	} else {
		std::cerr << "[host] WARNING: cudnn64_9.dll still missing (GetLastError="
		          << GetLastError()
		          << "). CUDA will fall back to CPU. Install cuDNN 9 or set CUDNN_PATH / "
		             "YA_CUDA_PATH, or copy cudnn*.dll next to yolo_host.exe\\cuda\\\n";
		(void)got_cudnn;
		(void)got_cudart;
	}
}
#endif

static fs::path find_web_root(const fs::path &base)
{
	const fs::path candidates[] = {
	    base / "webui",
	    base / ".." / "webui",
	    base / ".." / ".." / "webui",
	    fs::current_path() / "webui",
	    fs::current_path() / "standalone" / "webui",
	};
	for (const auto &p : candidates) {
		std::error_code ec;
		if (fs::exists(p / "index.html", ec))
			return fs::weakly_canonical(p, ec);
	}
	return base / "webui";
}

static fs::path find_config(const fs::path &base)
{
	// Prefer user override (written by PUT /api/config), then defaults.
	const fs::path candidates[] = {
	    base / "config" / "user.json",
	    base / "config" / "default.json",
	    fs::current_path() / "config" / "user.json",
	    fs::current_path() / "config" / "default.json",
	    fs::current_path() / "standalone" / "config" / "default.json",
	};
	for (const auto &p : candidates) {
		if (fs::exists(p))
			return p;
	}
	return base / "config" / "default.json";
}

static bool json_has(const std::string &content, const char *needle)
{
	return content.find(needle) != std::string::npos;
}

static std::string json_string_field(const std::string &content, const char *key,
                                     const std::string &fallback = {})
{
	// very small helper: "key": "value"
	const std::string pat = std::string("\"") + key + "\"";
	auto p = content.find(pat);
	if (p == std::string::npos)
		return fallback;
	p = content.find(':', p);
	if (p == std::string::npos)
		return fallback;
	p = content.find('"', p);
	if (p == std::string::npos)
		return fallback;
	auto e = content.find('"', p + 1);
	if (e == std::string::npos)
		return fallback;
	return content.substr(p + 1, e - p - 1);
}

static double json_number_field(const std::string &content, const char *key, double fallback)
{
	const std::string pat = std::string("\"") + key + "\"";
	auto p = content.find(pat);
	if (p == std::string::npos)
		return fallback;
	p = content.find(':', p);
	if (p == std::string::npos)
		return fallback;
	++p;
	while (p < content.size() && (content[p] == ' ' || content[p] == '\t'))
		++p;
	try {
		size_t idx = 0;
		double v = std::stod(content.substr(p), &idx);
		return v;
	} catch (...) {
		return fallback;
	}
}

static fs::path resolve_model_path(const fs::path &base, const std::string &raw)
{
	fs::path p(raw);
	if (p.is_absolute() && fs::exists(p))
		return p;
	const fs::path candidates[] = {
	    base / raw,
	    base / "models" / p.filename(),
	    fs::current_path() / raw,
	    base / ".." / ".." / "NCNN" / p.filename(),
	    base / ".." / "NCNN" / p.filename(),
	};
	for (const auto &c : candidates) {
		std::error_code ec;
		if (fs::exists(c, ec))
			return fs::weakly_canonical(c, ec);
	}
	return base / raw;
}

static ya::EngineConfig load_config_simple(const fs::path &json_path, const fs::path &preview_dir,
                                           const fs::path &base)
{
ya::EngineConfig cfg;
		cfg.preview_dir.clear(); // memory preview only — no disk BMP
		cfg.preview_interval_sec = 0.033;
		(void)preview_dir;
	cfg.infer.model_path = (base / "models" / "yolo.onnx").string();
	cfg.infer.device = "cuda";
	cfg.infer.model_version = 2;
	cfg.infer.input_resolution = 320;
	cfg.infer.confidence = 0.35f;
	cfg.infer.nms = 0.45f;
	cfg.infer_enabled = true;

	if (!fs::exists(json_path)) {
		std::cerr << "[host] config not found, using defaults: " << json_path << "\n";
		cfg.infer.model_path = resolve_model_path(base, "models/yolo.onnx").string();
		return cfg;
	}
	std::ifstream ifs(json_path);
	std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

	// capture
	if (json_has(content, "\"gdi\""))
		cfg.capture_backend = ya::CaptureBackend::Gdi;
	else if (json_has(content, "\"wgc\""))
		cfg.capture_backend = ya::CaptureBackend::Wgc;
	else
		cfg.capture_backend = ya::CaptureBackend::Dxgi;
	if (json_has(content, "\"mode\": \"region\"") || json_has(content, "\"mode\":\"region\""))
		cfg.capture_mode = "region";

	cfg.width = static_cast<int>(json_number_field(content, "width", cfg.width));
	cfg.height = static_cast<int>(json_number_field(content, "height", cfg.height));
	// prefer capture width inside capture block — simple parse may pick first width; OK for M2

	// infer
	if (json_has(content, "\"enabled\": false") || json_has(content, "\"enabled\":false")) {
		// ambiguous; check infer section roughly: if "infer" has enabled false near it
		auto ip = content.find("\"infer\"");
		if (ip != std::string::npos) {
			auto sub = content.substr(ip, 400);
			if (json_has(sub, "\"enabled\": false") || json_has(sub, "\"enabled\":false"))
				cfg.infer_enabled = false;
		}
	}
	if (json_has(content, "\"enabled\": true") || json_has(content, "\"enabled\":true")) {
		auto ip = content.find("\"infer\"");
		if (ip != std::string::npos) {
			auto sub = content.substr(ip, 400);
			if (json_has(sub, "\"enabled\": true") || json_has(sub, "\"enabled\":true"))
				cfg.infer_enabled = true;
		}
	}

	std::string model = json_string_field(content, "model_path", "models/yolo.onnx");
	cfg.infer.model_path = resolve_model_path(base, model).string();
	cfg.infer.device = json_string_field(content, "device", "cuda");
	cfg.infer.model_version =
	    static_cast<int>(json_number_field(content, "model_version", cfg.infer.model_version));
	cfg.infer.input_resolution =
	    static_cast<int>(json_number_field(content, "input_size", cfg.infer.input_resolution));
	cfg.infer.confidence =
	    static_cast<float>(json_number_field(content, "confidence", cfg.infer.confidence));
	cfg.infer.nms = static_cast<float>(json_number_field(content, "nms", cfg.infer.nms));
		cfg.infer.interval_frames =
		    static_cast<int>(json_number_field(content, "interval_frames", 1));
		cfg.infer.num_threads = static_cast<int>(json_number_field(content, "num_threads", 4));

// tracking
	cfg.tracker.iou_threshold =
	    static_cast<float>(json_number_field(content, "iou_threshold", cfg.tracker.iou_threshold));
	cfg.tracker.max_lost_frames =
	    static_cast<int>(json_number_field(content, "max_lost_frames", cfg.tracker.max_lost_frames));

	// aim section — only parse top-level aim keys (before nested "configs")
	{
		auto ap = content.find("\"aim\"");
		if (ap != std::string::npos) {
			// Cut at nested configs array so profile "enabled":false does not kill global aim
			size_t cut = 800;
			auto configs_pos = content.find("\"configs\"", ap);
			if (configs_pos != std::string::npos && configs_pos > ap)
				cut = std::min<size_t>(cut, configs_pos - ap);
			auto sub = content.substr(ap, cut);

			// First "enabled" after "aim" is the global flag
			auto ep = sub.find("\"enabled\"");
			if (ep != std::string::npos) {
				auto colon = sub.find(':', ep);
				if (colon != std::string::npos) {
					auto val = sub.substr(colon + 1, 16);
					if (val.find("true") != std::string::npos)
						cfg.aim_enabled = true;
					else if (val.find("false") != std::string::npos)
						cfg.aim_enabled = false;
				}
			}
#ifdef YA_WITH_AIM
			cfg.aim_full.enabled = cfg.aim_enabled;
			cfg.aim_full.fov_radius =
			    static_cast<int>(json_number_field(sub, "fov_radius", cfg.aim_full.fov_radius));
			{
				const int algo = static_cast<int>(
				    json_number_field(sub, "algorithm",
				                      static_cast<double>(cfg.aim_full.algorithm)));
				cfg.aim_full.algorithm = ya::FullAimBridge::parse_algorithm(algo);
			}
			if (json_has(sub, "\"use_dynamic_fov\": true") ||
			    json_has(sub, "\"use_dynamic_fov\":true"))
				cfg.aim_full.use_dynamic_fov = true;
			auto &p0 = cfg.aim_full.profiles[0];
			p0.enabled = cfg.aim_enabled;
			p0.mc.enableMouseControl = cfg.aim_enabled;
			p0.mc.hotkeyVirtualKey =
			    static_cast<int>(json_number_field(sub, "hotkey_vk", p0.mc.hotkeyVirtualKey));
			p0.mc.controllerType = ya::FullAimBridge::parse_controller(
			    json_string_field(sub, "controller", "WindowsAPI"));
			p0.mc.pidPMin =
			    static_cast<float>(json_number_field(content, "pidPMin", p0.mc.pidPMin));
			p0.mc.pidPMax =
			    static_cast<float>(json_number_field(content, "pidPMax", p0.mc.pidPMax));
			p0.mc.pidD = static_cast<float>(json_number_field(content, "pidD", p0.mc.pidD));
			p0.mc.pidI = static_cast<float>(json_number_field(content, "pidI", p0.mc.pidI));
			p0.mc.deadZonePixels = static_cast<float>(
			    json_number_field(content, "deadZonePixels", p0.mc.deadZonePixels));
			p0.mc.maxPixelMove = static_cast<float>(
			    json_number_field(content, "maxPixelMove", p0.mc.maxPixelMove));
			p0.mc.targetYOffset = static_cast<float>(
			    json_number_field(content, "targetYOffset", p0.mc.targetYOffset));
			p0.mc.fovRadiusPixels = cfg.aim_full.fov_radius;
			p0.mc.algorithmType = cfg.aim_full.algorithm;
			if (json_has(sub, "\"continuous_aim\": true") ||
			    json_has(sub, "\"continuous_aim\":true"))
				p0.continuous_aim = true;
			p0.mc.makcuPort = json_string_field(content, "makcuPort", p0.mc.makcuPort);
			p0.mc.makcuBaudRate = static_cast<int>(
			    json_number_field(content, "makcuBaudRate", p0.mc.makcuBaudRate));
			p0.mc.logiDriverType = static_cast<int>(
			    json_number_field(content, "logiDriverType", p0.mc.logiDriverType));
#else
			cfg.aim.enabled = cfg.aim_enabled;
			cfg.aim.hotkey_vk =
			    static_cast<int>(json_number_field(sub, "hotkey_vk", cfg.aim.hotkey_vk));
			cfg.aim.fov_radius =
			    static_cast<int>(json_number_field(sub, "fov_radius", cfg.aim.fov_radius));
			cfg.aim.controller = json_string_field(sub, "controller", "WindowsAPI");
			cfg.aim.pid_p_min =
			    static_cast<float>(json_number_field(content, "pidPMin", cfg.aim.pid_p_min));
			cfg.aim.pid_p_max =
			    static_cast<float>(json_number_field(content, "pidPMax", cfg.aim.pid_p_max));
			cfg.aim.pid_d =
			    static_cast<float>(json_number_field(content, "pidD", cfg.aim.pid_d));
			cfg.aim.pid_i =
			    static_cast<float>(json_number_field(content, "pidI", cfg.aim.pid_i));
			cfg.aim.dead_zone_px = static_cast<float>(
			    json_number_field(content, "deadZonePixels", cfg.aim.dead_zone_px));
			cfg.aim.max_pixel_move = static_cast<float>(
			    json_number_field(content, "maxPixelMove", cfg.aim.max_pixel_move));
#endif
		}
	}

	std::cout << "[host] config: " << json_path
	          << " backend=" << ya::FrameSource::backend_name(cfg.capture_backend)
	          << " infer=" << (cfg.infer_enabled ? "on" : "off")
	          << " device=" << cfg.infer.device
	          << " aim=" << (cfg.aim_enabled ? "on" : "off") << "\n";
	std::cout << "[host] model: " << cfg.infer.model_path << "\n";
return cfg;
	}

	static ya::EngineLoop *g_engine_for_console = nullptr;

#ifdef _WIN32
	static BOOL WINAPI console_ctrl_handler(DWORD type)
	{
		if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT ||
		    type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT) {
			if (g_engine_for_console) {
				std::cout << "\n[host] console close — flushing user.json …\n";
				g_engine_for_console->save_user_config(nullptr);
				g_engine_for_console->stop();
			}
			return TRUE;
		}
		return FALSE;
	}
#endif

	int main(int argc, char **argv)
		{
			(void)argc;
			(void)argv;
	
		#ifdef _WIN32
			SetConsoleOutputCP(CP_UTF8);
			SetConsoleCP(CP_UTF8);
		#endif
	
			const fs::path base = exe_dir();
	#ifdef _WIN32
			// MUST run before any ORT/CUDA load so provider_bridge finds cudnn64_9.dll
			bootstrap_cuda_runtime_path(base);
	#endif
			const fs::path cfg_path = find_config(base);
			const fs::path web_root = find_web_root(base);
			const fs::path preview_dir = base;
	
			std::error_code ec;
			fs::create_directories(preview_dir, ec);
			fs::create_directories(base / "models", ec);
			fs::create_directories(base / "cuda", ec);
	
		std::cout << "============================================\n";
		#ifdef YA_WITH_AIM
			std::cout << "  YOLO Aim Standalone (OBS full parity)\n";
		#else
			std::cout << "  YOLO Aim Standalone Launcher (slim aim)\n";
		#endif
			std::cout << "============================================\n";
			std::cout << "[host] exe dir : " << base << "\n";
			std::cout << "[host] web root: " << web_root << "\n";
			std::cout << "[host] preview : /api/preview.bmp (memory, no disk write)\n";
	
			ya::EngineLoop engine;
			g_engine_for_console = &engine;
	#ifdef _WIN32
			SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
	#endif
ya::EngineConfig eng_cfg;
			eng_cfg.preview_dir.clear();
			eng_cfg.preview_interval_sec = 0.033;
			eng_cfg.infer.model_path = (base / "models" / "yolo.onnx").string();
		eng_cfg.infer.device = "cuda";
		eng_cfg.infer.model_version = 2;
		eng_cfg.infer.input_resolution = 320;
		eng_cfg.infer.confidence = 0.35f;
		eng_cfg.infer.nms = 0.45f;
		eng_cfg.infer_enabled = true;
		eng_cfg.aim_enabled = true;
		std::string load_err;
		if (!ya::ConfigStore::load_file(cfg_path.string(), eng_cfg, &load_err)) {
			std::cerr << "[host] ConfigStore load failed (" << load_err
			          << "), falling back to simple parser\n";
			eng_cfg = load_config_simple(cfg_path, preview_dir, base);
		} else {
			eng_cfg.infer.model_path =
			    resolve_model_path(base, eng_cfg.infer.model_path).string();
			eng_cfg.preview_dir.clear();
			if (eng_cfg.preview_interval_sec <= 0.0 || eng_cfg.preview_interval_sec > 1.0)
				eng_cfg.preview_interval_sec = 0.033;
#ifdef YA_WITH_AIM
			eng_cfg.aim_full.enabled = eng_cfg.aim_enabled;
			if (eng_cfg.aim_enabled && !eng_cfg.aim_full.profiles[0].enabled) {
				bool any = false;
				for (auto &pr : eng_cfg.aim_full.profiles)
					if (pr.enabled)
						any = true;
				if (!any) {
					eng_cfg.aim_full.profiles[0].enabled = true;
					eng_cfg.aim_full.profiles[0].mc.enableMouseControl = true;
				}
			}
#endif
			std::cout << "[host] config(nlohmann): " << cfg_path
			          << " backend=" << ya::FrameSource::backend_name(eng_cfg.capture_backend)
			          << " infer=" << (eng_cfg.infer_enabled ? "on" : "off")
			          << " device=" << eng_cfg.infer.device
			          << " aim=" << (eng_cfg.aim_enabled ? "on" : "off") << "\n";
			std::cout << "[host] model: " << eng_cfg.infer.model_path << "\n";
		}
// Always persist PUT /api/config + exit flush to config/user.json next to exe.
				eng_cfg.user_config_path = (base / "config" / "user.json").string();
				{
					std::error_code ec2;
					fs::create_directories(base / "config", ec2);
				}
				eng_cfg.preview_dir.clear(); // never write preview_last.bmp
				// Model picker roots (OBS model_path file dialog equivalent).
			eng_cfg.models_dir = (base / "models").string();
			eng_cfg.model_search_dirs.clear();
			eng_cfg.model_search_dirs.push_back((base / "models").string());
			{
				const fs::path extra[] = {
				    base / ".." / ".." / "NCNN",
				    base / ".." / "NCNN",
				    base / ".." / ".." / "data" / "models",
				    base / ".." / "data" / "models",
				    fs::current_path() / "models",
				};
				std::error_code ec;
				for (const auto &d : extra) {
					if (fs::is_directory(d, ec))
						eng_cfg.model_search_dirs.push_back(fs::weakly_canonical(d, ec).string());
				}
			}
			engine.set_config(eng_cfg);
			// Seed user.json on first run so next launch always has a last-session file.
			{
				std::string seed_err;
				if (!fs::exists(eng_cfg.user_config_path) ||
				    fs::file_size(eng_cfg.user_config_path, ec) == 0) {
					if (engine.save_user_config(&seed_err))
						std::cout << "[host] seeded user.json: " << eng_cfg.user_config_path
						          << "\n";
					else if (!seed_err.empty())
						std::cerr << "[host] seed user.json failed: " << seed_err << "\n";
				} else {
					std::cout << "[host] user config: " << eng_cfg.user_config_path << "\n";
				}
			}

	const std::string host = "127.0.0.1";
	const int port = 17890;
	ya::WebServer web(engine, web_root.string());
	if (!web.start(host, port)) {
		std::cerr << "[host] FATAL: web server failed to bind " << host << ":" << port << "\n";
		std::cerr << "[host] Is another yolo_host already running?\n";
#ifdef _WIN32
		system("pause");
#endif
		return 1;
	}

	std::cout << "[host] WEB OK  -> http://127.0.0.1:" << port << "/\n";
	std::cout << "[host] STATUS  -> http://127.0.0.1:" << port << "/api/status\n";
	std::cout << "[host] PREVIEW -> http://127.0.0.1:" << port << "/api/preview.bmp\n";
	std::cout << "[host] DETECTIONS -> http://127.0.0.1:" << port << "/api/detections\n";

	engine.start();
	std::this_thread::sleep_for(std::chrono::milliseconds(1200));
	auto snap = engine.snapshot();
	std::cout << "[host] engine running=" << snap.running << " capture_ok=" << snap.capture_ok
	          << " infer_ok=" << snap.infer_ok << " frames=" << engine.frame_count()
	          << " dets=" << snap.last_det_count << "\n";
	if (snap.last_error[0])
		std::cerr << "[host] note: " << snap.last_error << "\n";

		const std::string url = "http://127.0.0.1:" + std::to_string(port) + "/";
#ifdef _WIN32
		ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#endif

		std::cout << "\n[host] Browser should open. Preview via /api/preview.bmp (no disk BMP).\n";
#ifdef YA_WITH_AIM
	std::cout << "[host] Full aim: 5 profiles + 8 backends + 5 algorithms (OBS stack).\n";
	std::cout << "[host] Hold hotkey (default RMB=0x02) on slot0 to aim.\n";
#else
	std::cout << "[host] Aim: hold hotkey (default RMB=0x02) to move toward nearest FOV target.\n";
#endif
	std::cout << "[host] Press ENTER in this window to quit.\n\n";

	std::atomic<bool> quit{false};
	std::thread heartbeat([&] {
		while (!quit.load()) {
			std::this_thread::sleep_for(std::chrono::seconds(3));
			if (quit.load())
				break;
			auto s = engine.snapshot();
			std::cout << "[alive] running=" << s.running << " capture_ok=" << s.capture_ok
			          << " infer_ok=" << s.infer_ok << " fps=" << s.capture_fps
			          << " infer_ms=" << s.infer_ms << " dets=" << s.last_det_count
			          << " frames=" << engine.frame_count();
#ifdef YA_WITH_AIM
			auto as = engine.aim_status();
			std::cout << " aim_slot=" << as.active_slot
			          << " ctrl=" << ya::FullAimBridge::controller_name(
			                            static_cast<ControllerType>(as.controller_type));
#endif
			std::cout << std::endl;
		}
	});

	// Stay alive when stdin is a pipe/EOF (CI / background diagnostic).
		{
			std::string line;
			if (!std::getline(std::cin, line)) {
				std::cout << "[host] stdin EOF — keep running until process kill\n";
				while (!quit.load()) {
					std::this_thread::sleep_for(std::chrono::seconds(2));
				}
			}
		}
quit = true;
			if (heartbeat.joinable())
				heartbeat.join();

			// Flush config before teardown (also done inside stop(); double-safe).
			engine.save_user_config(nullptr);
			engine.stop();
			web.stop();
			g_engine_for_console = nullptr;
			std::cout << "[host] stopped. config saved. last preview: " << engine.preview_path()
			          << "\n";
			return 0;
		}
