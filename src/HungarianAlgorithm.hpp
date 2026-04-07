#ifndef HUNGARIAN_ALGORITHM_HPP
#define HUNGARIAN_ALGORITHM_HPP

#include <vector>
#include <opencv2/core.hpp>

class HungarianAlgorithm {
public:
    static std::vector<int> solve(const std::vector<std::vector<float>>& costMatrix);

    static float calculateIoUDistance(const cv::Rect2f& a, const cv::Rect2f& b);

    static float calculateEuclideanDistance(const cv::Point2f& a, const cv::Point2f& b);

    // 多指标融合距离计算
    static float calculateFusedDistance(
        const cv::Rect2f& detBox, const cv::Rect2f& trackBox,
        const cv::Point2f& detCenter, const cv::Point2f& trackCenter,
        float w_iou = 0.4f, float w_center = 0.3f,
        float w_aspect = 0.15f, float w_area = 0.15f);

    // 辅助函数
    static float calculateCenterDistance(const cv::Point2f& a, const cv::Point2f& b, float maxDist);
    static float calculateAspectDistance(const cv::Rect2f& a, const cv::Rect2f& b);
    static float calculateAreaDistance(const cv::Rect2f& a, const cv::Rect2f& b);
};

#endif
