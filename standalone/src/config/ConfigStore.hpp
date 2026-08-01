#pragma once

// 配置存储 — M2 新配置层。
// 职责:加载/保存 user.json、嵌套或扁平 OBS 键合并、损坏自恢复。

#include "ConfigKeyMap.hpp"
#include "ConfigModel.hpp"

#include <string>

namespace ya {
namespace config {

class ConfigStore {
public:
    // 读文件;不存在返回 false(不报错,调用方决定是否用默认)。
    static bool load_file(const std::string& path, ConfigDocument& out, std::string* err);

    // 写文件(先写临时文件再 rename,避免写一半损坏)。
    static bool save_file(const std::string& path, const ConfigDocument& d, std::string* err);

    // 合并 patch:
    //   - 嵌套文档 {capture, infer, tracker, vision, aim} → 逐节合并(仅覆盖存在的键)
    //   - 扁平 OBS 键 patch(旧 webui data-key 形式)→ ConfigKeyMap 映射
    static bool merge(ConfigDocument& d, const nlohmann::json& patch, std::string* err);

    // 损坏恢复:解析失败 → 备份坏文件为 user.json.bak.<ts> → 返回默认文档。
    static bool load_or_default(const std::string& path, ConfigDocument& out, std::string* err);
};

} // namespace config
} // namespace ya
