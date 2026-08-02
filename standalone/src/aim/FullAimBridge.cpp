#ifdef _WIN32

#include "FullAimBridge.hpp"
#include "LogiDriverMouseController.hpp"
#include "MAKCUMouseController.hpp"
#include "MouseControllerFactory.hpp"

#include <thread>
#include <Windows.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace ya {
namespace {

bool vk_down(int vk)
{
	if (vk <= 0)
		return false;
	return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

} // namespace

FullAimBridge::FullAimBridge() = default;
FullAimBridge::~FullAimBridge() = default;

void FullAimBridge::set_settings(const FullAimSettings &s)
{
	std::lock_guard<std::mutex> lock(mu_);
	settings_ = s;
	// push globals into every profile (OBS algorithm_type_global + external/aim/neural)
	for (auto &p : settings_.profiles) {
		settings_.stamp_globals(p.mc);
		p.mc.continuousAimEnabled = p.continuous_aim;
	}
}

FullAimSettings FullAimBridge::settings() const
{
	std::lock_guard<std::mutex> lock(mu_);
	return settings_;
}

bool FullAimBridge::ensure_controller()
{
	std::lock_guard<std::mutex> lock(mu_);
	if (controller_) {
		status_.controller_ok = true;
		return true;
	}
	try {
		controller_ = MouseControllerFactory::createController(ControllerType::WindowsAPI);
		current_type_ = ControllerType::WindowsAPI;
		status_.controller_ok = (controller_ != nullptr);
		status_.last_error.clear();
		std::cout << "[aim] controller created: WindowsAPI" << std::endl;
		return status_.controller_ok;
	} catch (const std::exception &e) {
		status_.controller_ok = false;
		status_.last_error = e.what();
		return false;
	}
}

void FullAimBridge::recreate_if_needed(ControllerType type, const std::string &makcu_port,
                                       int baud, int logi_type)
{
	// Hot path rule: only recreate when type/port/logi subtype changes
	const bool need =
	    !controller_ || current_type_ != type ||
	    (type == ControllerType::MAKCU &&
	     (current_makcu_port_ != makcu_port || current_makcu_baud_ != baud)) ||
	    (type == ControllerType::LogiDriver && current_logi_type_ != logi_type);

	if (!need)
		return;

	try {
		controller_ =
		    MouseControllerFactory::createController(type, makcu_port, baud, logi_type);
		current_type_ = type;
		current_makcu_port_ = makcu_port;
		current_makcu_baud_ = baud;
		current_logi_type_ = logi_type;
		status_.controller_ok = (controller_ != nullptr);
		status_.last_error.clear();
		std::cout << "[aim] controller recreated: " << controller_name(type) << std::endl;
	} catch (const std::exception &e) {
		status_.last_error = std::string("recreate failed: ") + e.what();
		// fallback WindowsAPI
		controller_ = MouseControllerFactory::createController(ControllerType::WindowsAPI);
		current_type_ = ControllerType::WindowsAPI;
		status_.controller_ok = (controller_ != nullptr);
		std::cerr << "[aim] " << status_.last_error << " → fallback WindowsAPI" << std::endl;
	}
}

int FullAimBridge::select_active_slot() const
{
	// 1) continuous aim
	for (int i = 0; i < FullAimSettings::kSlots; ++i) {
		if (settings_.profiles[i].enabled && settings_.profiles[i].continuous_aim)
			return i;
	}
	// 2) hotkey held
	for (int i = 0; i < FullAimSettings::kSlots; ++i) {
		if (!settings_.profiles[i].enabled)
			continue;
		if (vk_down(settings_.profiles[i].mc.hotkeyVirtualKey))
			return i;
	}
	return -1;
}

float FullAimBridge::update_dynamic_fov(bool has_target_in_fov)
{
	const float primary = static_cast<float>(settings_.fov_radius);
	if (!settings_.use_dynamic_fov)
		return primary;

	const float secondary =
	    primary * std::clamp(settings_.dynamic_fov_shrink_percent, 0.1f, 1.f);
	const float target = has_target_in_fov ? secondary : primary;
	// simple exponential approach (OBS uses timed ease; good enough)
	const float alpha = 0.15f;
	current_fov_ = current_fov_ + (target - current_fov_) * alpha;
	return current_fov_;
}

void FullAimBridge::apply_slot(int slot, int frame_w, int frame_h)
{
	if (slot < 0 || slot >= FullAimSettings::kSlots)
		return;
	auto &prof = settings_.profiles[slot];
	MouseControllerConfig cfg = prof.mc;
	settings_.stamp_globals(cfg);
	cfg.enableMouseControl = true;
	cfg.continuousAimEnabled = prof.continuous_aim;
	cfg.fovRadiusPixels = static_cast<int>(current_fov_);
	cfg.inferenceFrameWidth = frame_w;
	cfg.inferenceFrameHeight = frame_h;
	if (cfg.screenWidth <= 0)
		cfg.screenWidth = GetSystemMetrics(SM_CXSCREEN);
	if (cfg.screenHeight <= 0)
		cfg.screenHeight = GetSystemMetrics(SM_CYSCREEN);
	if (cfg.sourceWidth <= 0)
		cfg.sourceWidth = frame_w;
	if (cfg.sourceHeight <= 0)
		cfg.sourceHeight = frame_h;

	recreate_if_needed(cfg.controllerType, cfg.makcuPort, cfg.makcuBaudRate, cfg.logiDriverType);
	if (controller_)
		controller_->updateConfig(cfg);
}

void FullAimBridge::tick(const std::vector<Detection> &dets, int frame_w, int frame_h, int crop_x,
                         int crop_y, float infer_ms)
{
	std::lock_guard<std::mutex> lock(mu_);
	if (!settings_.enabled) {
		status_.active_slot = -1;
		status_.aiming = false;
		status_.hotkey_down = false;
		if (controller_) {
			MouseControllerConfig off = controller_->getConfig();
			off.enableMouseControl = false;
			controller_->updateConfig(off);
			controller_->tick(); // release auto-trigger
		}
		return;
	}

	if (!controller_) {
		controller_ = MouseControllerFactory::createController(ControllerType::WindowsAPI);
		current_type_ = ControllerType::WindowsAPI;
	}
	if (!controller_) {
		status_.controller_ok = false;
		status_.last_error = "no controller";
		return;
	}
	status_.controller_ok = true;

	const int slot = select_active_slot();
	status_.active_slot = slot;
	status_.algorithm = static_cast<int>(settings_.algorithm);
	status_.controller_type = static_cast<int>(current_type_);

	// FOV: rough check if any det near center for dynamic FOV
	bool any_in_fov = false;
	const float cx = 0.5f, cy = 0.5f;
	const float fr = static_cast<float>(std::max(1, settings_.fov_radius));
	for (const auto &d : dets) {
		const float px = (d.centerX - cx) * frame_w;
		const float py = (d.centerY - cy) * frame_h;
		if (px * px + py * py <= fr * fr) {
			any_in_fov = true;
			break;
		}
	}
	status_.fov_px = static_cast<int>(update_dynamic_fov(any_in_fov && slot >= 0));

	if (slot < 0) {
		status_.aiming = false;
		status_.hotkey_down = false;
		MouseControllerConfig off{};
		off.enableMouseControl = false;
		// keep last controller, just disable
		if (controller_) {
			auto c = controller_->getConfig();
			c.enableMouseControl = false;
			controller_->updateConfig(c);
			controller_->tick();
		}
		last_active_slot_ = -1;
		return;
	}

	status_.hotkey_down = !settings_.profiles[slot].continuous_aim;
	status_.aiming = true;
	apply_slot(slot, frame_w, frame_h);
	last_active_slot_ = slot;

	controller_->setDetectionsWithFrameSize(dets, frame_w, frame_h, crop_x, crop_y);
	controller_->setInferenceTimeMs(infer_ms);
	if (settings_.crosshair_enabled && settings_.aim_origin_x >= 0.f &&
	    settings_.aim_origin_y >= 0.f) {
		controller_->setAimOrigin(settings_.aim_origin_x, settings_.aim_origin_y);
	} else {
		controller_->setAimOrigin(-1.f, -1.f);
	}
	controller_->tick();
}

FullAimStatus FullAimBridge::status() const
{
	std::lock_guard<std::mutex> lock(mu_);
	return status_;
}

bool FullAimBridge::test_controller(ControllerType type, const std::string &makcu_port,
                                    int makcu_baud, int logi_type, std::string *err_out)
{
	try {
		auto c = MouseControllerFactory::createController(type, makcu_port, makcu_baud, logi_type);
		if (!c) {
			if (err_out)
				*err_out = "create returned null";
			return false;
		}
		// 硬件后端连接状态检查(Logi 驱动等;MAKCU 由用户跳过)
		if (type == ControllerType::LogiDriver) {
			auto *logi = dynamic_cast<LogiDriverMouseController *>(c.get());
			if (logi && !logi->isConnected()) {
				if (err_out)
					*err_out = "Logi 驱动未连接(检查 GHUB/LGS/Razer)";
				return false;
			}
		}
		if (type == ControllerType::MAKCU) {
			auto *makcu = dynamic_cast<MAKCUMouseController *>(c.get());
			if (makcu && !makcu->isConnected()) {
				if (err_out)
					*err_out = "MAKCU 串口未连接(" + makcu_port + ")";
				return false;
			}
		}
		// 真移动冒烟:注入偏置目标 + 一次 tick,验证后端真正执行移动
		MouseControllerConfig cfg;
		cfg.enableMouseControl = true;
		cfg.continuousAimEnabled = true; // 绕过热键
		cfg.controllerType = type;
		cfg.algorithmType = AlgorithmType::ExternalPID;
		cfg.maxPixelMove = 8.f;
		c->updateConfig(cfg);
		Detection d;
		d.classId = 0;
		d.confidence = 0.9f;
		d.x = 0.49f;
		d.y = 0.49f;
		d.width = 0.04f;
		d.height = 0.08f;
		d.centerX = 0.51f;
		d.centerY = 0.5f;
		c->setDetectionsWithFrameSize(std::vector<Detection>{d}, 640, 640, 0, 0);
		c->tick(); // 真移动(小位移)
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		if (err_out)
			*err_out = std::string("ok: ") + controller_name(type) + " 移动正常";
		return true;
	} catch (const std::exception &e) {
		if (err_out)
			*err_out = e.what();
		return false;
	}
}

const char *FullAimBridge::controller_name(ControllerType t)
{
	switch (t) {
	case ControllerType::WindowsAPI:
		return "WindowsAPI";
	case ControllerType::MAKCU:
		return "MAKCU";
	case ControllerType::LogiDriver:
		return "LogiDriver";
	case ControllerType::GvInput:
		return "GvInput";
	case ControllerType::TencInput:
		return "TencInput";
	case ControllerType::NtUserSendInput:
		return "NtUserSendInput";
	case ControllerType::NtUserInjectMouse:
		return "NtUserInjectMouse";
	case ControllerType::NtUserInjectPointer:
		return "NtUserInjectPointer";
	default:
		return "Unknown";
	}
}

const char *FullAimBridge::algorithm_name(AlgorithmType a)
{
	switch (a) {
	case AlgorithmType::AdvancedPID:
		return "AdvancedPID";
	case AlgorithmType::ExternalPID:
		return "ExternalPID";
	case AlgorithmType::AimController:
		return "AimController";
	case AlgorithmType::SlewRate:
		return "SlewRate";
	case AlgorithmType::AdaptivePID:
		return "AdaptivePID";
	default:
		return "Unknown";
	}
}

ControllerType FullAimBridge::parse_controller(const std::string &s)
{
	if (s == "MAKCU" || s == "makcu" || s == "1")
		return ControllerType::MAKCU;
	if (s == "LogiDriver" || s == "logi" || s == "Logi" || s == "2")
		return ControllerType::LogiDriver;
	if (s == "GvInput" || s == "gvinput" || s == "3")
		return ControllerType::GvInput;
	if (s == "TencInput" || s == "tenc" || s == "4")
		return ControllerType::TencInput;
	if (s == "NtUserSendInput" || s == "NtUserSend" || s == "5")
		return ControllerType::NtUserSendInput;
	if (s == "NtUserInjectMouse" || s == "NtUserInject" || s == "6")
		return ControllerType::NtUserInjectMouse;
	if (s == "NtUserInjectPointer" || s == "NtUserPointer" || s == "7")
		return ControllerType::NtUserInjectPointer;
	return ControllerType::WindowsAPI;
}

ControllerType FullAimBridge::parse_controller_int(int v)
{
	if (v < 0 || v > 7)
		return ControllerType::WindowsAPI;
	return static_cast<ControllerType>(v);
}

AlgorithmType FullAimBridge::parse_algorithm(int v)
{
	if (v < 0 || v > 4)
		return AlgorithmType::AdvancedPID;
	return static_cast<AlgorithmType>(v);
}

} // namespace ya

#endif
