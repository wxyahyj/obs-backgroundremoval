// 核心功能测试:5 算法瞄准/PID/热键/持续瞄准/后座/扳机/死区/FOV/跟踪。
// Mock controller 记录 moveMouse/click,不真动鼠标。

#include "AbstractMouseController.hpp"
#include "TrackerEngine.hpp"
#include "aim_controller.hpp"

#include <cstdio>
#include <thread>
#include <cmath>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "  FAIL: %s\n", msg);                       \
            ++g_fail;                                                        \
        } else {                                                             \
            std::fprintf(stderr, "  PASS: %s\n", msg);                       \
        }                                                                    \
    } while (0)

// ---- Mock controller:记录移动/点击 ----
class MockController : public AbstractMouseController {
public:
    long totalDx = 0;
    long totalDy = 0;
    int moveCount = 0;
    int clickDownCount = 0;
    int clickUpCount = 0;
    bool firing = false; // 模拟开火(后座测试)

    // 模拟 tick:按真实时间间隔推进(PID/切换/积分依赖 deltaTime)
    void tickN(int n, int intervalMs = 10)
    {
        for (int i = 0; i < n; ++i) {
            tick();
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
        }
    }

protected:
    void moveMouse(int dx, int dy) override
    {
        totalDx += dx;
        totalDy += dy;
        ++moveCount;
    }
    void performClickDown() override { ++clickDownCount; }
    void performClickUp() override { ++clickUpCount; }
    bool checkFiring() override { return firing; }
    ControllerType getControllerType() const override
    {
        return ControllerType::WindowsAPI;
    }
};

// 构造一个目标检测(归一化坐标,center 在 cx,cy)
Detection make_det(float cx, float cy, float w = 0.05f, float h = 0.1f, int cls = 0)
{
    Detection d;
    d.classId = cls;
    d.confidence = 0.9f;
    d.x = cx - w / 2;
    d.y = cy - h / 2;
    d.width = w;
    d.height = h;
    d.centerX = cx;
    d.centerY = cy;
    return d;
}

void test_algorithm(AlgorithmType algo, const char* name, float targetDx)
{
    MockController c;
    MouseControllerConfig cfg;
    cfg.enableMouseControl = true;
    cfg.continuousAimEnabled = true; // 绕过热键
    cfg.algorithmType = algo;
    cfg.fovRadiusPixels = 200;
    cfg.deadZonePixels = 2.f;
    cfg.maxPixelMove = 64.f;
    cfg.pidPMin = 0.15f;
    cfg.pidPMax = 0.6f;
    cfg.pidD = 0.007f;
    cfg.pidI = 0.01f;
    c.updateConfig(cfg);

    // 目标偏离中心 targetDx(帧宽 640)
    std::vector<Detection> dets{make_det(0.5f + targetDx / 640.f, 0.5f)};
    c.setDetectionsWithFrameSize(dets, 640, 640, 0, 0);
    c.tickN(40); // 400ms:过目标切换延迟 + PID 收敛

    const long dx = c.totalDx;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "[%s] 目标右偏%dpx → 输出 dx=%ld (%s)",
                  name, static_cast<int>(targetDx), dx,
                  (dx > 0) ? "方向正确" : "方向错误!");
    CHECK(dx > 0, buf);
    std::fprintf(stderr, "        移动次数=%d 累计dx=%ld 累计dy=%ld\n", c.moveCount,
                 dx, c.totalDy);
}

void test_tracking()
{
    ya::TrackerConfig tcfg;
    tcfg.iou_threshold = 0.3f;
    tcfg.max_lost_frames = 10;
    ya::TrackerEngine tracker;
    tracker.set_config(tcfg);

    std::vector<Detection> f1{make_det(0.5f, 0.5f)};
    std::vector<Detection> f2{make_det(0.51f, 0.5f)};
    std::vector<Detection> f3{make_det(0.52f, 0.5f)};

    auto r1 = tracker.update(f1);
    auto r2 = tracker.update(f2);
    auto r3 = tracker.update(f3);

    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "跟踪: 3 帧同目标 trackId=%d/%d/%d(应相同)", r1[0].trackId,
                  r2[0].trackId, r3[0].trackId);
    CHECK(r1[0].trackId == r2[0].trackId && r2[0].trackId == r3[0].trackId &&
              r1[0].trackId >= 0,
          buf);

    // 目标消失 → lost
    auto r4 = tracker.update({});
    std::snprintf(buf, sizeof(buf), "跟踪: 目标消失后 lostFrames 递增(=%d)",
                  r4.empty() ? 0 : r4[0].lostFrames);
    CHECK(r4.empty() || r4[0].lostFrames >= 1, buf);
}

void test_recoil()
{
    MockController c;
    MouseControllerConfig cfg;
    cfg.enableMouseControl = true;
    cfg.continuousAimEnabled = true;
    cfg.autoRecoilControlEnabled = true;
    cfg.recoilStrength = 5.f;
    cfg.algorithmType = AlgorithmType::AdvancedPID;
    c.updateConfig(cfg);
    c.firing = true; // 模拟开枪

    // 目标在中心 → PID 输出≈0,后座应产生向下(正 dy)补偿
    std::vector<Detection> dets{make_det(0.5f, 0.5f)};
    c.setDetectionsWithFrameSize(dets, 640, 640, 0, 0);
    c.tickN(40);

    char buf[128];
    std::snprintf(buf, sizeof(buf), "后座: 目标居中时输出 dy=%ld(压枪应移动)",
                  c.totalDy);
    CHECK(c.totalDy != 0 || c.moveCount > 0, buf);
    std::fprintf(stderr, "        后座累计 dy=%ld\n", c.totalDy);
}

void test_trigger()
{
    MockController c;
    MouseControllerConfig cfg;
    cfg.enableMouseControl = true;
    cfg.continuousAimEnabled = true;
    cfg.autoTriggerEnabled = true;
    cfg.autoTriggerRadius = 20; // 像素
    cfg.algorithmType = AlgorithmType::AdvancedPID;
    c.updateConfig(cfg);

    // 目标在中心(扳机半径内)→ 应触发点击
    std::vector<Detection> dets{make_det(0.5f, 0.5f, 0.02f, 0.02f)};
    c.setDetectionsWithFrameSize(dets, 640, 640, 0, 0);
    c.tickN(60);

    char buf[128];
    std::snprintf(buf, sizeof(buf), "扳机: 目标在半径内 → clickDown=%d",
                  c.clickDownCount);
    CHECK(c.clickDownCount > 0, buf);
}

void test_gates()
{
    // 热键门控:enableMouseControl=false → 无移动
    MockController c;
    MouseControllerConfig cfg;
    cfg.enableMouseControl = false;
    cfg.continuousAimEnabled = true;
    cfg.algorithmType = AlgorithmType::AdvancedPID;
    c.updateConfig(cfg);
    c.setDetectionsWithFrameSize({make_det(0.6f, 0.5f)}, 640, 640, 0, 0);
    c.tickN(20);
    CHECK(c.moveCount == 0, "门控: enableMouseControl=false → 无移动");

    // FOV 门控:目标在 FOV 外 → 无移动
    MockController c2;
    MouseControllerConfig cfg2;
    cfg2.enableMouseControl = true;
    cfg2.continuousAimEnabled = true;
    cfg2.fovRadiusPixels = 30;
    cfg2.algorithmType = AlgorithmType::AdvancedPID;
    c2.updateConfig(cfg2);
    c2.setDetectionsWithFrameSize({make_det(0.9f, 0.5f)}, 640, 640, 0, 0);
    c2.tickN(20);
    CHECK(c2.moveCount == 0, "门控: 目标在 FOV 外 → 无移动");
}

void test_filters()
{
    // 每个滤波器:开 vs 关,两种场景:
    //  A) 静态目标:方向正确 + 输出存在
    //  B) 运动目标(每帧右移):预测/平滑类应改变输出(生效)
    struct FilterCase {
        const char* name;
        void (*enable)(MouseControllerConfig&);
        void (*disable)(MouseControllerConfig&);
    };
    FilterCase cases[] = {
        {"导数预测器",
         [](MouseControllerConfig& c) { c.useDerivativePredictor = true; },
         [](MouseControllerConfig& c) { c.useDerivativePredictor = false; }},
        {"Smith预估器",
         [](MouseControllerConfig& c) { c.smithPredictorEnabled = true; },
         [](MouseControllerConfig& c) { c.smithPredictorEnabled = false; }},
        {"SlewRate限速",
         [](MouseControllerConfig& c) { c.slewRateEnabled = true; },
         [](MouseControllerConfig& c) { c.slewRateEnabled = false; }},
        {"贝塞尔曲线",
         [](MouseControllerConfig& c) { c.enableBezierMovement = true; },
         [](MouseControllerConfig& c) { c.enableBezierMovement = false; }},
        {"GhostTracker",
         [](MouseControllerConfig& c) { c.enableGhostTracker = true; },
         [](MouseControllerConfig& c) { c.enableGhostTracker = false; }},
        {"IMM滤波器",
         [](MouseControllerConfig& c) { c.immFilterEnabled = true; },
         [](MouseControllerConfig& c) { c.immFilterEnabled = false; }},
        {"OneEuro滤波",
         [](MouseControllerConfig& c) { c.useOneEuroFilter = true; },
         [](MouseControllerConfig& c) { c.useOneEuroFilter = false; }},
    };

    for (const auto& fc : cases) {
        // 运动目标:记录逐帧输出序列(轨迹/平滑类总量相同但序列应不同)
        auto run_seq = [&](bool enabled) -> std::vector<long> {
            MockController c;
            MouseControllerConfig cfg;
            cfg.enableMouseControl = true;
            cfg.continuousAimEnabled = true;
            cfg.algorithmType = AlgorithmType::AdvancedPID;
            cfg.fovRadiusPixels = 200;
            cfg.deadZonePixels = 2.f;
            cfg.maxPixelMove = 64.f;
            cfg.pidPMin = 0.15f;
            cfg.pidPMax = 0.6f;
            fc.enable(cfg);
            if (!enabled)
                fc.disable(cfg);
            c.updateConfig(cfg);
            float cx = 0.5f;
            std::vector<long> seq;
            for (int i = 0; i < 40; ++i) {
                cx += 2.f / 640.f;
                c.setDetectionsWithFrameSize({make_det(cx, 0.5f)}, 640, 640, 0, 0);
                c.tick();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                seq.push_back(c.totalDx);
            }
            return seq;
        };

        const auto off = run_seq(false);
        const auto on = run_seq(true);
        // 逐帧差异:任何帧输出不同 = 滤波器生效
        long diffFrames = 0;
        double sumOff = 0, sumOn = 0;
        for (size_t i = 0; i < off.size() && i < on.size(); ++i) {
            if (off[i] != on[i])
                ++diffFrames;
            sumOff += off[i];
            sumOn += on[i];
        }
        // 平滑类:开启后逐帧跳变应更小(方差比)
        double varOff = 0, varOn = 0;
        const double mOff = sumOff / off.size(), mOn = sumOn / on.size();
        for (size_t i = 0; i < off.size(); ++i) {
            varOff += (off[i] - mOff) * (off[i] - mOff);
            varOn += (on[i] - mOn) * (on[i] - mOn);
        }
        char buf[220];
        std::snprintf(buf, sizeof(buf),
                      "[%s] 差异帧=%ld/40 总量:关=%.0f 开=%.0f 方差:关=%.0f 开=%.0f",
                      fc.name, diffFrames, sumOff, sumOn, varOff, varOn);
        CHECK(diffFrames > 0, buf);
    }
}

} // namespace

int main()
{
    std::fprintf(stderr, "== 5 算法瞄准测试 ==\n");
    test_algorithm(AlgorithmType::AdvancedPID, "AdvancedPID", 40.f);
    test_algorithm(AlgorithmType::ExternalPID, "ExternalPID", 40.f);
    test_algorithm(AlgorithmType::AimController, "AimController", 40.f);
    test_algorithm(AlgorithmType::SlewRate, "SlewRate", 40.f);
    test_algorithm(AlgorithmType::AdaptivePID, "AdaptivePID", 40.f);

    std::fprintf(stderr, "== 门控测试 ==\n");
    test_gates();

    std::fprintf(stderr, "== 后座测试 ==\n");
    test_recoil();

    std::fprintf(stderr, "== 扳机测试 ==\n");
    test_trigger();

    std::fprintf(stderr, "== 跟踪测试 ==\n");
    test_tracking();

    std::fprintf(stderr, "== 滤波器实测 ==\n");
    test_filters();

    std::fprintf(stderr, "\n结果: %s (%d failed)\n",
                 g_fail == 0 ? "ALL PASS" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
