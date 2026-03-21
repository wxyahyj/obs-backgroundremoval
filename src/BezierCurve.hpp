#ifndef BEZIER_CURVE_HPP
#define BEZIER_CURVE_HPP

#include <vector>
#include <cmath>
#include <random>

struct BezierPoint {
    float x;
    float y;
};

class BezierCurve {
public:
    BezierCurve() : randomGenerator(std::random_device{}()) {}
    
    // 生成二次贝塞尔曲线点
    // start: 起点
    // end: 终点
    // controlOffsetRatio: 控制点偏移比例（相对于线段长度的比例）
    // steps: 曲线步数
    std::vector<BezierPoint> generateQuadraticBezier(
        BezierPoint start, 
        BezierPoint end, 
        float controlOffsetRatio,
        int steps
    ) {
        std::vector<BezierPoint> points;
        points.reserve(steps + 1);
        
        // 计算中点
        float midX = (start.x + end.x) / 2.0f;
        float midY = (start.y + end.y) / 2.0f;
        
        // 计算控制点（在中点基础上偏移）
        // 偏移方向垂直于起点到终点的连线
        float dx = end.x - start.x;
        float dy = end.y - start.y;
        float length = std::sqrt(dx * dx + dy * dy);
        
        if (length < 0.001f) {
            // 起点和终点太近，直接返回
            for (int i = 0; i <= steps; i++) {
                points.push_back(start);
            }
            return points;
        }
        
        // 随机选择偏移方向（左或右）
        std::uniform_int_distribution<int> dirDist(0, 1);
        float direction = (dirDist(randomGenerator) == 0) ? 1.0f : -1.0f;
        
        // 计算垂直方向
        float perpX = -dy / length;
        float perpY = dx / length;
        
        // 控制点
        float controlX = midX + perpX * length * controlOffsetRatio * direction;
        float controlY = midY + perpY * length * controlOffsetRatio * direction;
        
        for (int i = 0; i <= steps; i++) {
            float t = static_cast<float>(i) / static_cast<float>(steps);
            BezierPoint point = quadraticBezier(start, {controlX, controlY}, end, t);
            points.push_back(point);
        }
        
        return points;
    }
    
    // 计算单步贝塞尔曲线增量
    // 返回从当前步到下一步的移动增量
    BezierPoint getStepDelta(
        BezierPoint start,
        BezierPoint end,
        float controlOffsetRatio,
        int currentStep,
        int totalSteps,
        float direction
    ) {
        float t1 = static_cast<float>(currentStep) / static_cast<float>(totalSteps);
        float t2 = static_cast<float>(currentStep + 1) / static_cast<float>(totalSteps);
        
        // 计算中点
        float midX = (start.x + end.x) / 2.0f;
        float midY = (start.y + end.y) / 2.0f;
        
        // 计算控制点
        float dx = end.x - start.x;
        float dy = end.y - start.y;
        float length = std::sqrt(dx * dx + dy * dy);
        
        if (length < 0.001f) {
            return {0.0f, 0.0f};
        }
        
        // 计算垂直方向
        float perpX = -dy / length;
        float perpY = dx / length;
        
        // 控制点
        float controlX = midX + perpX * length * controlOffsetRatio * direction;
        float controlY = midY + perpY * length * controlOffsetRatio * direction;
        
        BezierPoint p1 = quadraticBezier(start, {controlX, controlY}, end, t1);
        BezierPoint p2 = quadraticBezier(start, {controlX, controlY}, end, t2);
        
        return {p2.x - p1.x, p2.y - p1.y};
    }
    
    // 设置随机方向种子
    void setDirection(float dir) {
        cachedDirection = dir;
        hasCachedDirection = true;
    }
    
    void clearCachedDirection() {
        hasCachedDirection = false;
    }
    
private:
    std::mt19937 randomGenerator;
    float cachedDirection = 1.0f;
    bool hasCachedDirection = false;
    
    // 二次贝塞尔曲线公式
    // B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    BezierPoint quadraticBezier(BezierPoint p0, BezierPoint p1, BezierPoint p2, float t) {
        float oneMinusT = 1.0f - t;
        float x = oneMinusT * oneMinusT * p0.x + 
                  2.0f * oneMinusT * t * p1.x + 
                  t * t * p2.x;
        float y = oneMinusT * oneMinusT * p0.y + 
                  2.0f * oneMinusT * t * p1.y + 
                  t * t * p2.y;
        return {x, y};
    }
};

#endif
