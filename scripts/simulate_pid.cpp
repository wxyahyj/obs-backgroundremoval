// PID 算法模拟测试程序
// 用途：模拟 IncrementalPid 在闭环控制系统中的行为，验证算法正确性
// 编译：g++ -std=c++17 -O2 simulate_pid.cpp -o simulate_pid.exe
// 运行：./simulate_pid.exe

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <string>

// 完整复制 mpid.cpp 的 IncrementalPid 实现，保证模拟与实际一致
class IncrementalPid {
public:
    void configure(float kp, float ki, float kd, float d_alpha = 0.2f, float input_alpha = 0.3f, float output_alpha = 0.4f) {
        kp_ = kp; ki_ = ki; kd_ = kd;
        d_alpha_ = d_alpha; input_alpha_ = input_alpha; output_alpha_ = output_alpha;
    }
    void reset(float output = 0.0f) {
        output_ = output; previous_output_ = output;
        previous_error_ = 0.0f; previous_previous_error_ = 0.0f; previous_d_term_ = 0.0f;
    }
    float update(float error) {
        float filtered_error = input_alpha_ * error + (1.0f - input_alpha_) * previous_error_;
        float p_term = kp_ * (filtered_error - previous_error_);
        float i_term = ki_ * filtered_error;
        float d_term = kd_ * (filtered_error - 2.0f * previous_error_ + previous_previous_error_);
        d_term = d_alpha_ * d_term + (1.0f - d_alpha_) * previous_d_term_;
        previous_d_term_ = d_term;
        float delta = p_term + i_term + d_term;
        const float max_delta = 50.0f;
        delta = std::clamp(delta, -max_delta, max_delta);
        output_ += delta;
        output_ = output_alpha_ * output_ + (1.0f - output_alpha_) * previous_output_;
        float decay_factor = 0.85f;
        if (std::abs(filtered_error) < std::abs(previous_error_) && (filtered_error > 0) == (previous_error_ > 0)) {
            decay_factor = 0.7f;
        }
        output_ *= decay_factor;
        const float max_output = 100.0f;
        output_ = std::clamp(output_, -max_output, max_output);
        previous_previous_error_ = previous_error_;
        previous_error_ = filtered_error;
        previous_output_ = output_;
        return output_;
    }
    float output() const { return output_; }
private:
    float output_ = 0.0f, previous_output_ = 0.0f;
    float previous_error_ = 0.0f, previous_previous_error_ = 0.0f, previous_d_term_ = 0.0f;
    float kp_ = 0.0f, ki_ = 0.0f, kd_ = 0.0f;
    float d_alpha_ = 0.2f, input_alpha_ = 0.3f, output_alpha_ = 0.4f;
};

// 优化版 PID：移除衰减、修正滤波顺序、增加抗饱和
class IncrementalPidOptimized {
public:
    void configure(float kp, float ki, float kd, float d_alpha = 0.2f, float input_alpha = 0.3f, float output_alpha = 0.4f, float max_output = 100.0f) {
        kp_ = kp; ki_ = ki; kd_ = kd;
        d_alpha_ = d_alpha; input_alpha_ = input_alpha; output_alpha_ = output_alpha;
        max_output_ = max_output;
    }
    void reset(float output = 0.0f) {
        output_ = output; previous_output_ = output;
        previous_error_ = 0.0f; previous_previous_error_ = 0.0f; previous_d_term_ = 0.0f;
        integral_ = 0.0f;
    }
    float update(float error) {
        // 1. 输入滤波
        float filtered_error = input_alpha_ * error + (1.0f - input_alpha_) * previous_error_;

        // 2. 增量式 PID
        float p_term = kp_ * (filtered_error - previous_error_);
        float i_term = ki_ * filtered_error;
        float d_term = kd_ * (filtered_error - 2.0f * previous_error_ + previous_previous_error_);

        // 3. D 项低通滤波
        d_term = d_alpha_ * d_term + (1.0f - d_alpha_) * previous_d_term_;
        previous_d_term_ = d_term;

        // 4. 抗积分饱和：仅当输出未饱和时累积积分
        if (output_ > -max_output_ && output_ < max_output_) {
            integral_ += i_term;
        } else if ((output_ >= max_output_ && i_term < 0) || (output_ <= -max_output_ && i_term > 0)) {
            integral_ += i_term;  // 饱和时仅允许反向积分
        }
        integral_ = std::clamp(integral_, -max_output_, max_output_);

        // 5. 增量计算
        float delta = p_term + i_term + d_term;
        const float max_delta = 50.0f;
        delta = std::clamp(delta, -max_delta, max_delta);

        // 6. 更新输出（无衰减）
        output_ += delta;

        // 7. 输出限幅（替代衰减）
        output_ = std::clamp(output_, -max_output_, max_output_);

        // 8. 输出滤波（使用上一帧最终输出作为参考）
        output_ = output_alpha_ * output_ + (1.0f - output_alpha_) * previous_output_;
        output_ = std::clamp(output_, -max_output_, max_output_);

        previous_previous_error_ = previous_error_;
        previous_error_ = filtered_error;
        previous_output_ = output_;
        return output_;
    }
    float output() const { return output_; }
private:
    float output_ = 0.0f, previous_output_ = 0.0f;
    float previous_error_ = 0.0f, previous_previous_error_ = 0.0f, previous_d_term_ = 0.0f;
    float kp_ = 0.0f, ki_ = 0.0f, kd_ = 0.0f;
    float d_alpha_ = 0.2f, input_alpha_ = 0.3f, output_alpha_ = 0.4f;
    float integral_ = 0.0f;
    float max_output_ = 100.0f;
};

// 模拟闭环系统
struct SimResult {
    int steps_to_target;       // 到达目标（误差<2）的步数，-1 表示未到达
    float final_error;         // 最终误差
    float max_overshoot;       // 最大过冲
    int oscillation_count;     // 在目标附近震荡次数（误差符号变化）
    float steady_state_error;  // 稳态误差（最后 20 步平均）
};

template<typename PidType>
SimResult simulate(PidType& pid, float target, int max_steps = 200, float dt = 1.0f) {
    SimResult result = {-1, 0, 0, 0, 0};
    float position = 0.0f;
    float initial_error = target - position;
    bool reached = false;
    int reach_step = -1;
    float prev_error = initial_error;
    int sign_changes = 0;
    float sum_error_last_20 = 0;

    printf("  Step |  Error  | Output  | Position\n");
    for (int step = 0; step < max_steps; ++step) {
        float error = target - position;
        float output = pid.update(error);
        position += output * dt;

        // 统计
        if (!reached && std::abs(error) < 2.0f) {
            reached = true;
            reach_step = step;
            result.steps_to_target = step;
        }
        if (reached && (error > 0) != (prev_error > 0) && std::abs(prev_error) > 0.5f) {
            sign_changes++;
        }
        if (error * initial_error < 0 && std::abs(error) > result.max_overshoot) {
            result.max_overshoot = std::abs(error);
        }
        if (step >= max_steps - 20) {
            sum_error_last_20 += error;
        }
        prev_error = error;

        // 每 10 步打印一次
        if (step < 30 || step % 20 == 0) {
            printf("  %4d | %7.2f | %7.2f | %7.2f\n", step, error, output, position);
        }
    }

    result.final_error = target - position;
    result.oscillation_count = sign_changes;
    result.steady_state_error = sum_error_last_20 / 20.0f;
    printf("  到达目标步数: %d, 最终误差: %.3f, 最大过冲: %.3f, 震荡次数: %d, 稳态误差: %.3f\n",
           result.steps_to_target, result.final_error, result.max_overshoot,
           result.oscillation_count, result.steady_state_error);
    return result;
}

int main() {
    printf("========================================\n");
    printf("PID 算法模拟测试\n");
    printf("========================================\n\n");

    // 测试场景：目标从 0 移动到 100（典型瞄准场景）
    const float target = 100.0f;

    // 参数配置（参考项目默认值）
    const float kp = 0.6f, ki = 0.01f, kd = 0.007f;

    printf("【测试1】原版 PID（kp=%.3f, ki=%.3f, kd=%.3f）\n", kp, ki, kd);
    printf("目标: %.1f\n", target);
    IncrementalPid original_pid;
    original_pid.configure(kp, ki, kd);
    original_pid.reset();
    SimResult r1 = simulate(original_pid, target);

    printf("\n【测试2】优化版 PID（相同参数，移除衰减+抗饱和）\n");
    printf("目标: %.1f\n", target);
    IncrementalPidOptimized opt_pid;
    opt_pid.configure(kp, ki, kd);
    opt_pid.reset();
    SimResult r2 = simulate(opt_pid, target);

    printf("\n【测试3】大目标距离（target=500）- 原版\n");
    IncrementalPid p3;
    p3.configure(kp, ki, kd);
    p3.reset();
    SimResult r3 = simulate(p3, 500.0f);

    printf("\n【测试4】大目标距离（target=500）- 优化版\n");
    IncrementalPidOptimized p4;
    p4.configure(kp, ki, kd);
    p4.reset();
    SimResult r4 = simulate(p4, 500.0f);

    printf("\n【测试5】小目标距离（target=10）- 原版\n");
    IncrementalPid p5;
    p5.configure(kp, ki, kd);
    p5.reset();
    SimResult r5 = simulate(p5, 10.0f);

    printf("\n【测试6】小目标距离（target=10）- 优化版\n");
    IncrementalPidOptimized p6;
    p6.configure(kp, ki, kd);
    p6.reset();
    SimResult r6 = simulate(p6, 10.0f);

    // 汇总
    printf("\n========================================\n");
    printf("汇总对比\n");
    printf("========================================\n");
    printf("%-20s | %-12s | %-12s\n", "测试", "原版", "优化版");
    printf("%-20s | %-12s | %-12s\n", "target=100 步数", std::to_string(r1.steps_to_target).c_str(), std::to_string(r2.steps_to_target).c_str());
    printf("%-20s | %-12.3f | %-12.3f\n", "target=100 稳态误差", r1.steady_state_error, r2.steady_state_error);
    printf("%-20s | %-12.3f | %-12.3f\n", "target=100 过冲", r1.max_overshoot, r2.max_overshoot);
    printf("%-20s | %-12d | %-12d\n", "target=100 震荡次数", r1.oscillation_count, r2.oscillation_count);
    printf("%-20s | %-12s | %-12s\n", "target=500 步数", std::to_string(r3.steps_to_target).c_str(), std::to_string(r4.steps_to_target).c_str());
    printf("%-20s | %-12.3f | %-12.3f\n", "target=500 稳态误差", r3.steady_state_error, r4.steady_state_error);
    printf("%-20s | %-12s | %-12s\n", "target=10 步数", std::to_string(r5.steps_to_target).c_str(), std::to_string(r6.steps_to_target).c_str());
    printf("%-20s | %-12.3f | %-12.3f\n", "target=10 稳态误差", r5.steady_state_error, r6.steady_state_error);

    return 0;
}
