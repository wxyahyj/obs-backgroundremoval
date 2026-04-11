#ifndef HUNGARIAN_ALGORITHM_HPP
#define HUNGARIAN_ALGORITHM_HPP

#include <vector>
#include <opencv2/core.hpp>

class HungarianAlgorithm {
public:
    static std::vector<int> solve(const std::vector<std::vector<float>>& costMatrix);

    static float calculateIoUDistance(const cv::Rect2f& a, const cv::Rect2f& b);

    static float calculateEuclideanDistance(const cv::Point2f& a, const cv::Point2f& b);
};

#endif
