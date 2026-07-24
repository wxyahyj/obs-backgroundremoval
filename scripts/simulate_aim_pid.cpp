// aim_controller PID 算法模拟测试
// 用途：评估 aim::AimController 相对于现有 AdvancedPID 的性能
// 编译：通过 build_x64_sim CMake 项目

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <string>

// 完整复制 aim_controller.hpp 的核心实现
namespace aim {

template <typename T>
constexpr T clamp(T value, T low, T high) noexcept {
    return value < low ? low : (value > high ? high : value);
}

class IncrementalPid {
public:
    void configure(double kp, double ki, double kd) noexcept {
        kp_ = kp; ki_ = ki; kd_ = kd;
    }
    void reset(double output = 0.0) noexcept {
        output_ = output; previous_output_ = output;
        previous_error_ = 0.0; previous_previous_error_ = 0.0;
    }
    void set_output_limits(double min_val, double max_val) noexcept {
        output_min_ = min_val; output_max_ = max_val;
        output_ = clamp(output_, output_min_, output_max_);
    }
    double update(double error, double scale = 1.0) noexcept {
        previous_output_ = output_;
        // 输入死区
        if (std::abs(error) < 0.3) error = 0.0;
        // 增量式 PID
        double delta =
            kp_ * (error - previous_error_) +
            ki_ * error +
            kd_ * (error - 2.0 * previous_error_ + previous_previous_error_);
        delta *= scale;
        output_ += delta;
        // 输出死区
        if (std::abs(output_) < 0.5) output_ *= 0.9;
        // 抗饱和限幅
        output_ = clamp(output_, output_min_, output_max_);
        previous_previous_error_ = previous_error_;
        previous_error_ = error;
        return output_;
    }
    double output() const noexcept { return output_; }
private:
    double output_ = 0.0, previous_output_ = 0.0;
    double previous_error_ = 0.0, previous_previous_error_ = 0.0;
    double kp_ = 0.0, ki_ = 0.0, kd_ = 0.0;
    double output_min_ = -1e18, output_max_ = 1e18;
};

} // namespace aim

// 优化版（移除输出死区，改进接近目标的行为）
class AimPidOptimized {
public:
    void configure(double kp, double ki, double kd) noexcept {
        kp_ = kp; ki_ = ki; kd_ = kd;
    }
    void reset(double output = 0.0) noexcept {
        output_ = output; previous_output_ = output;
        previous_error_ = 0.0; previous_previous_error_ = 0.0;
    }
    void set_output_limits(double min_val, double max_val) noexcept {
        output_min_ = min_val; output_max_ = max_val;
    }
    double update(double error, double scale = 1.0) noexcept {
        previous_output_ = output_;
        // 输入死区（保留）
        if (std::abs(error) < 0.3) error = 0.0;
        // 增量式 PID
        double delta =
            kp_ * (error - previous_error_) +
            ki_ * error +
            kd_ * (error - 2.0 * previous_error_ + previous_previous_error_);
        delta *= scale;
        output_ += delta;
        // 移除输出死区（原版 output *= 0.9 会阻碍小输出累积）
        // 抗饱和限幅
        output_ = (output_ < output_min_) ? output_min_ :
                  (output_ > output_max_) ? output_max_ : output_;
        previous_previous_error_ = previous_error_;
        previous_error_ = error;
        return output_;
    }
    double output() const noexcept { return output_; }
private:
    double output_ = 0.0, previous_output_ = 0.0;
    double previous_error_ = 0.0, previous_previous_error_ = 0.0;
    double kp_ = 0.0, ki_ = 0.0, kd_ = 0.0;
    double output_min_ = -1e18, output_max_ = 1e18;
};

struct SimResult {
    int steps_to_target;
    float final_error;
    float max_overshoot;
    int oscillation_count;
    float steady_state_error;
};

template<typename PidType>
SimResult simulate(PidType& pid, float target, int max_steps = 200, float dt = 1.0f) {
    SimResult result = {-1, 0, 0, 0, 0};
    float position = 0.0f;
    float initial_error = target - position;
    bool reached = false;
    float prev_error = initial_error;
    int sign_changes = 0;
    float sum_error_last_20 = 0;

    printf("  Step |  Error  | Output  | Position\n");
    for (int step = 0; step < max_steps; ++step) {
        float error = target - position;
        float output = (float)pid.update(error);
        position += output * dt;

        if (!reached && std::abs(error) < 2.0f) {
            reached = true;
            result.steps_to_target = step;
        }
        if (reached && (error > 0) != (prev_error > 0) && std::abs(prev_error) > 0.5f) {
            sign_changes++;
        }
        if (error * initial_error < 0 && std::abs(error) > result.max_overshoot) {
            result.max_overshoot = std::abs(error);
        }
        if (step >= max_steps - 20) sum_error_last_20 += error;
        prev_error = error;

        if (step < 30 || step % 20 == 0) {
            printf("  %4d | %7.2f | %7.2f | %7.2f\n", step, error, output, position);
        }
    }

    result.final_error = target - position;
    result.oscillation_count = sign_changes;
    result.steady_state_error = sum_error_last_20 / 20.0f;
    printf("  -> 到达:%d  最终误差:%.3f  过冲:%.3f  震荡:%d  稳态:%.3f\n",
           result.steps_to_target, result.final_error, result.max_overshoot,
           result.oscillation_count, result.steady_state_error);
    return result;
}

int main() {
    printf("========================================\n");
    printf("aim_controller PID 模拟测试\n");
    printf("========================================\n\n");

    const float target = 100.0f;
    const double kp = 0.6, ki = 0.01, kd = 0.007;

    printf("【测试1】aim::IncrementalPid 原版（kp=%.2f ki=%.3f kd=%.3f target=%.0f）\n", kp, ki, kd, target);
    aim::IncrementalPid p1;
    p1.configure(kp, ki, kd);
    p1.set_output_limits(-100, 100);
    p1.reset();
    SimResult r1 = simulate(p1, target);

    printf("\n【测试2】优化版（移除输出死区）\n");
    AimPidOptimized p2;
    p2.configure(kp, ki, kd);
    p2.set_output_limits(-100, 100);
    p2.reset();
    SimResult r2 = simulate(p2, target);

    printf("\n【测试3】大目标 target=500 - aim原版\n");
    aim::IncrementalPid p3;
    p3.configure(kp, ki, kd);
    p3.set_output_limits(-100, 100);
    p3.reset();
    SimResult r3 = simulate(p3, 500.0f);

    printf("\n【测试4】大目标 target=500 - 优化版\n");
    AimPidOptimized p4;
    p4.configure(kp, ki, kd);
    p4.set_output_limits(-100, 100);
    p4.reset();
    SimResult r4 = simulate(p4, 500.0f);

    printf("\n【测试5】小目标 target=10 - aim原版\n");
    aim::IncrementalPid p5;
    p5.configure(kp, ki, kd);
    p5.set_output_limits(-100, 100);
    p5.reset();
    SimResult r5 = simulate(p5, 10.0f);

    printf("\n【测试6】小目标 target=10 - 优化版\n");
    AimPidOptimized p6;
    p6.configure(kp, ki, kd);
    p6.set_output_limits(-100, 100);
    p6.reset();
    SimResult r6 = simulate(p6, 10.0f);

    printf("\n========================================\n");
    printf("汇总对比\n");
    printf("========================================\n");
    printf("%-22s | %-12s | %-12s\n", "测试", "aim原版", "优化版");
    printf("%-22s | %-12d | %-12d\n", "target=100 到达步数", r1.steps_to_target, r2.steps_to_target);
    printf("%-22s | %-12.3f | %-12.3f\n", "target=100 稳态误差", r1.steady_state_error, r2.steady_state_error);
    printf("%-22s | %-12.3f | %-12.3f\n", "target=100 过冲", r1.max_overshoot, r2.max_overshoot);
    printf("%-22s | %-12d | %-12d\n", "target=100 震荡次数", r1.oscillation_count, r2.oscillation_count);
    printf("%-22s | %-12d | %-12d\n", "target=500 到达步数", r3.steps_to_target, r4.steps_to_target);
    printf("%-22s | %-12.3f | %-12.3f\n", "target=500 稳态误差", r3.steady_state_error, r4.steady_state_error);
    printf("%-22s | %-12d | %-12d\n", "target=10 到达步数", r5.steps_to_target, r6.steps_to_target);
    printf("%-22s | %-12.3f | %-12.3f\n", "target=10 稳态误差", r5.steady_state_error, r6.steady_state_error);

    return 0;
}
