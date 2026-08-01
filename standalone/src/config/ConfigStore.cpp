// 配置存储实现。

#include "ConfigStore.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace ya {
namespace config {

namespace {

bool has_nested_sections(const nlohmann::json& j)
{
    return (j.contains("capture") && j["capture"].is_object()) ||
           (j.contains("infer") && j["infer"].is_object()) ||
           (j.contains("tracker") && j["tracker"].is_object()) ||
           (j.contains("vision") && j["vision"].is_object()) ||
           (j.contains("aim") && j["aim"].is_object());
}

} // namespace

bool ConfigStore::load_file(const std::string& path, ConfigDocument& out, std::string* err)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err)
            *err = "cannot open: " + path;
        return false;
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        if (err)
            *err = std::string("parse failed: ") + e.what();
        return false;
    }
    if (!j.is_object()) {
        if (err)
            *err = "config root must be an object";
        return false;
    }
    document_from_json(out, j);
    if (err)
        err->clear();
    return true;
}

bool ConfigStore::save_file(const std::string& path, const ConfigDocument& d, std::string* err)
{
    const std::filesystem::path p(path);
    const std::filesystem::path tmp = p.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            if (err)
                *err = "cannot write: " + tmp.string();
            return false;
        }
        f << document_to_json(d).dump(2);
        f.flush();
    }
    std::error_code ec;
    std::filesystem::rename(tmp, p, ec);
    if (ec) {
        if (err)
            *err = "rename failed: " + ec.message();
        return false;
    }
    if (err)
        err->clear();
    return true;
}

bool ConfigStore::merge(ConfigDocument& d, const nlohmann::json& patch, std::string* err)
{
    if (!patch.is_object()) {
        if (err)
            *err = "patch must be an object";
        return false;
    }
    if (has_nested_sections(patch)) {
        document_from_json(d, patch); // 逐节合并,仅覆盖存在的键
    } else {
        apply_flat_obs_keys(d, patch, err); // 扁平 OBS 键(旧 webui 兼容)
    }
    return true;
}

bool ConfigStore::load_or_default(const std::string& path, ConfigDocument& out, std::string* err)
{
    std::string load_err;
    if (load_file(path, out, &load_err))
        return true;

    // 文件存在但解析失败 → 备份损坏文件
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        const auto ts = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        const std::string bak = path + ".bak." + std::to_string(ts);
        std::filesystem::rename(path, bak, ec);
        if (!ec)
            std::fprintf(stderr, "[config] corrupted config backed up to %s\n", bak.c_str());
    }

    out = ConfigDocument(); // 默认
    if (err)
        *err = load_err.empty() ? "using defaults" : load_err + " (using defaults)";
    return false;
}

} // namespace config
} // namespace ya
