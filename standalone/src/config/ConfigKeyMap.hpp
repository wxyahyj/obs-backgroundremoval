#pragma once

// OBS 扁平键 → ConfigDocument 映射表(M2)。
// 覆盖 filter_properties.cpp 全部数据键:
//   - 槽键: {base}_{slot}(p_min_0 … ghost_noise_freq_4)
//   - 全局键: use_gpu / confidence_threshold / crosshair_* / external_* …
// 点键兼容: "mouse_config_2.p_min" 归一化为 p_min_2。

#include "ConfigModel.hpp"

#include <nlohmann/json.hpp>

#include <string>

namespace ya {
namespace config {

// 分发扁平 patch:槽键 → apply_obs_slot_keys,全局键 → apply_obs_global_keys。
bool apply_flat_obs_keys(ConfigDocument& d, const nlohmann::json& flat, std::string* err);

// 单槽应用(键形如 p_min_0;也接受 p_min.0)
void apply_obs_slot_keys(ConfigDocument& d, const nlohmann::json& j, int slot);

// 全局键应用
void apply_obs_global_keys(ConfigDocument& d, const nlohmann::json& j);

// 槽索引解析:"p_min_3"→3;"p_min.3"→3;失败返回 -1
int parse_slot_suffix(const std::string& key);

} // namespace config
} // namespace ya
