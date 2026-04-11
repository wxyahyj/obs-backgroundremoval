#include "HungarianAlgorithm.hpp"
#include <limits>
#include <algorithm>
#include <cmath>

std::vector<int> HungarianAlgorithm::solve(const std::vector<std::vector<float>>& costMatrix) {
    if (costMatrix.empty() || costMatrix[0].empty()) {
        return {};
    }

    int n = static_cast<int>(costMatrix.size());
    int m = static_cast<int>(costMatrix[0].size());

    int size = std::max(n, m);

    std::vector<std::vector<float>> matrix(size, std::vector<float>(size, 0.0f));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = costMatrix[i][j];
        }
    }

    std::vector<float> uLabel(size + 1, 0.0f);
    std::vector<float> vLabel(size + 1, 0.0f);
    std::vector<int> matchV(size + 1, 0);
    std::vector<int> way(size + 1, 0);

    for (int i = 1; i <= size; ++i) {
        matchV[0] = i;
        int j0 = 0;
        std::vector<float> minv(size + 1, std::numeric_limits<float>::max());
        std::vector<bool> used(size + 1, false);

        do {
            used[j0] = true;
            int i0 = matchV[j0];
            float delta = std::numeric_limits<float>::max();
            int j1 = 0;

            for (int j = 1; j <= size; ++j) {
                if (!used[j]) {
                    float cur = matrix[i0 - 1][j - 1] - uLabel[i0] - vLabel[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for (int j = 0; j <= size; ++j) {
                if (used[j]) {
                    uLabel[matchV[j]] += delta;
                    vLabel[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (matchV[j0] != 0);

        do {
            int j1 = way[j0];
            matchV[j0] = matchV[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> result(n, -1);
    for (int j = 1; j <= size; ++j) {
        if (matchV[j] != 0 && matchV[j] - 1 < n && j - 1 < m) {
            result[matchV[j] - 1] = j - 1;
        }
    }

    return result;
}

float HungarianAlgorithm::calculateIoUDistance(const cv::Rect2f& a, const cv::Rect2f& b) {
    float intersectionArea = (a & b).area();
    float unionArea = a.area() + b.area() - intersectionArea;

    if (unionArea <= 0.0f) {
        return 1.0f;
    }

    float iou = intersectionArea / unionArea;
    return 1.0f - iou;
}

float HungarianAlgorithm::calculateEuclideanDistance(const cv::Point2f& a, const cv::Point2f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}
