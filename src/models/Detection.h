#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <opencv2/core.hpp>

struct Detection {
    int classId;
    std::string className;
    float confidence;

    float x;
    float y;
    float width;
    float height;

    float centerX;
    float centerY;

    int trackId = -1;

    cv::Rect getPixelBBox(int imageWidth, int imageHeight) const {
        return cv::Rect(
            static_cast<int>(x * imageWidth),
            static_cast<int>(y * imageHeight),
            static_cast<int>(width * imageWidth),
            static_cast<int>(height * imageHeight)
        );
    }

    cv::Point2f getCenterPixel(int imageWidth, int imageHeight) const {
        return cv::Point2f(
            centerX * imageWidth,
            centerY * imageHeight
        );
    }
};

#endif
