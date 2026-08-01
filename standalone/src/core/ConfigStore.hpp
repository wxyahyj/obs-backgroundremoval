#pragma once

#include "EngineLoop.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace ya {

// OBS-key-aligned config document (filter_properties.cpp names).
// GET /api/config returns this shape; PUT merges patch (flat OBS keys or nested).
class ConfigStore {
public:
	static nlohmann::json to_json(const EngineConfig &c);
	static bool merge_into(EngineConfig &c, const nlohmann::json &patch, std::string *err = nullptr);
	static bool load_file(const std::string &path, EngineConfig &c, std::string *err = nullptr);
	static bool save_file(const std::string &path, const EngineConfig &c, std::string *err = nullptr);

	// Apply flat OBS-style keys for one slot (enable_config_0, p_min_0, …)
	static void apply_obs_slot_keys(EngineConfig &c, const nlohmann::json &j, int slot);
	static void apply_obs_global_keys(EngineConfig &c, const nlohmann::json &j);

	// Build MouseControllerConfig for slot (with globals stamped).
	static MouseControllerConfig to_mouse_config(const EngineConfig &c, int slot);
};

} // namespace ya
