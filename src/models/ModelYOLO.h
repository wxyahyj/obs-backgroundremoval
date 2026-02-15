#ifndef MODEL_YOLO_H
#define MODEL_YOLO_H

#include "Model.h"
#include "Detection.h"
#include <vector>
#include <string>

class ModelYOLO : public ModelBCHW {
public:
    enum class Version {
        YOLOv5 = 0,
        YOLOv8 = 1,
        YOLOv11 = 2
    };

    explicit ModelYOLO(Version version);
    ~ModelYOLO() override;

    void loadModel(const std::string& modelPath);
    void preprocessInput(const cv::Mat& input, float* outputBuffer);

    std::vector<Detection> inference(const cv::Mat& input);

    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void setTargetClass(int classId);
    void loadClassNames(const std::string& namesFile);

    Version getVersion() const { return version_; }
    int getInputWidth() const { return inputWidth_; }
    int getInputHeight() const { return inputHeight_; }
    int getNumClasses() const { return numClasses_; }

private:
    std::vector<Detection> postprocessYOLOv5(
        const float* rawOutput,
        int numBoxes,
        int numClasses,
        const cv::Size& modelInputSize,
        const cv::Size& originalImageSize
    );

    std::vector<Detection> postprocessYOLOv8(
        const float* rawOutput,
        int numBoxes,
        int numClasses,
        const cv::Size& modelInputSize,
        const cv::Size& originalImageSize
    );

    std::vector<Detection> postprocessYOLOv11(
        const float* rawOutput,
        int numBoxes,
        int numClasses,
        const cv::Size& modelInputSize,
        const cv::Size& originalImageSize
    );

    std::vector<int> performNMS(
        const std::vector<cv::Rect2f>& boxes,
        const std::vector<float>& scores,
        float nmsThreshold
    );

    float calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b);

    void xywhToxyxy(float cx, float cy, float w, float h,
                    float& x1, float& y1, float& x2, float& y2);

    Version version_;
    float confidenceThreshold_;
    float nmsThreshold_;
    int targetClassId_;

    int inputWidth_;
    int inputHeight_;
    int numClasses_;

    std::vector<std::string> classNames_;
};

#endif
