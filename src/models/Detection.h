#ifndef DETECTION_H
#define DETECTION_H

#include <string>
#include <opencv2/core.hpp>

struct Detection {
    // 类别信息
    int classId;                    // 类别 ID (0-79 for COCO)
    std::string className;          // 类别名称 (如 "person", "car")
    float confidence;               // 置信度 (0.0 - 1.0)
    
    // 边界框（归一化坐标 0.0-1.0）
    float x;                        // 左上角 x（相对于图像宽度）
    float y;                        // 左上角 y（相对于图像高度）
    float width;                    // 宽度（相对于图像宽度）
    float height;                   // 高度（相对于图像高度）
    
    // 中心点坐标（归一化）
    float centerX;                  // 中心点 x
    float centerY;                  // 中心点 y
    
    // 可选：跟踪 ID（预留给对象跟踪功能）
    int trackId = -1;
    
    // 转换为像素坐标
    cv::Rect getPixelBBox(int imageWidth, int imageHeight) const {
        return cv::Rect(
            static_cast<int>(x * imageWidth),
            static_cast<int>(y * imageHeight),
            static_cast<int>(width * imageWidth),
            static_cast<int>(height * imageHeight)
        );
    }
    
    // 转换为中心点格式（YOLO 原生格式）
    cv::Point2f getCenterPixel(int imageWidth, int imageHeight) const {
        return cv::Point2f(
            centerX * imageWidth,
            centerY * imageHeight
        );
    }
};

#endif // DETECTION_H