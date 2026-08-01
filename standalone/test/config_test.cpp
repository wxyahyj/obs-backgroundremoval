// M2 配置层验证 — 序列化往返 / 扁平 OBS 键 / 合并 / 文件恢复。
// 构建:yolo_config_test 目标(独立于 yolo_host)。

#include "config/ConfigModel.hpp"
#include "config/ConfigStore.hpp"

#include <cstdio>
#include <string>

namespace {

int g_fail = 0;

#define CHECK(cond, msg)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, \
                         __FILE__, __LINE__);              \
            ++g_fail;                                      \
        }                                                  \
    } while (0)

} // namespace

int main()
{
    using namespace ya::config;

    // 1. 默认文档序列化往返
    ConfigDocument d;
    nlohmann::json j = document_to_json(d);
    ConfigDocument d2;
    document_from_json(d2, j);
    CHECK(d2.aim.slots[0].enabled, "slot0 enabled default");
    CHECK(d2.aim.slots[0].mc.hotkeyVirtualKey == 0x02, "slot0 hotkey RMB");
    CHECK(d2.aim.slots[1].mc.makcuPort == "COM5", "slot1 makcu default");
    CHECK(d2.infer.device == "cuda", "infer device default");
    CHECK(j.contains("aim") && j["aim"].contains("slots") &&
              j["aim"]["slots"].size() == 5,
          "json has 5 slots");

    // 2. 扁平 OBS 键 patch
    ConfigDocument d3;
    nlohmann::json flat = {
        {"p_min_0", 0.2f},
        {"p_max_0", 0.9f},
        {"crosshair_enabled", true},
        {"use_gpu", "cuda"},
        {"confidence_threshold", 0.6f},
        {"fov_radius", 200},
        {"makcu_port_2", "COM7"},
    };
    std::string err;
    CHECK(ConfigStore::merge(d3, flat, &err), "flat merge ok");
    CHECK(d3.aim.slots[0].mc.pidPMin == 0.2f, "p_min_0 applied");
    CHECK(d3.aim.slots[0].mc.pidPMax == 0.9f, "p_max_0 applied");
    CHECK(d3.aim.crosshair_enabled, "crosshair_enabled applied");
    CHECK(d3.infer.device == "cuda", "use_gpu applied");
    CHECK(d3.infer.confidence == 0.6f, "confidence applied");
    CHECK(d3.aim.fov_radius == 200, "fov_radius applied");
    CHECK(d3.aim.slots[2].mc.makcuPort == "COM7", "makcu_port_2 applied");

    // 3. 嵌套 patch 合并(仅覆盖存在的键)
    ConfigDocument d4;
    nlohmann::json nested = {
        {"infer", {{"confidence", 0.7f}, {"device", "dml"}}},
        {"aim", {{"fov_radius", 150}, {"enabled", false}}},
    };
    CHECK(ConfigStore::merge(d4, nested, &err), "nested merge ok");
    CHECK(d4.infer.confidence == 0.7f, "nested confidence");
    CHECK(d4.infer.device == "dml", "nested device");
    CHECK(d4.aim.fov_radius == 150, "nested fov");
    CHECK(!d4.aim.enabled, "nested aim enabled");
    CHECK(d4.aim.slots[0].enabled, "nested merge keeps slot0 default");

    // 4. 文件往返 + 损坏恢复
    const std::string path = "config_test_out.json";
    CHECK(ConfigStore::save_file(path, d, &err), "save ok");
    ConfigDocument d5;
    CHECK(ConfigStore::load_file(path, d5, &err), "load ok");
    CHECK(d5.aim.slots[0].mc.pidPMax == 0.6f, "roundtrip value");
    std::remove(path.c_str());

    // 5. 损坏文件 → load_or_default 备份 + 默认
    const std::string bad = "config_bad.json";
    {
        std::FILE* f = std::fopen(bad.c_str(), "w");
        std::fputs("{ not valid json !!!", f);
        std::fclose(f);
    }
    ConfigDocument d6;
    CHECK(!ConfigStore::load_or_default(bad, d6, &err), "bad file reported");
    CHECK(d6.aim.slots[0].enabled, "default after corruption");
    std::remove(bad.c_str());

    // 6. 点键 mouse_config_N.x
    ConfigDocument d7;
    nlohmann::json dot = {{"mouse_config_1.p_min", 0.33f}};
    CHECK(ConfigStore::merge(d7, dot, &err), "dot key merge ok");
    CHECK(d7.aim.slots[1].mc.pidPMin == 0.33f, "dot key applied");

    if (g_fail == 0) {
        std::fprintf(stderr, "config_test: ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "config_test: %d FAILED\n", g_fail);
    return 1;
}
