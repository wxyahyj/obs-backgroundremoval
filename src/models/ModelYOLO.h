#ifndef MODEL_YOLO_H
#define MODEL_YOLO_H

#include "Model.h"
#include "Detection.h"
#include <vector>
#include <string>

class ModelYOLO : public Model {
public:
    enum class Version {
        YOLOv5 = 0,
        YOLOv8 = 1,
        YOLOv11 = 2
    };
    
    explicit ModelYOLO(Version version);
    ~ModelYOLO() override;
    
    // 基类方法实现
    void loadModel(const std::string& modelPath) override;
    void preprocessInput(const cv::Mat& input, float* outputBuffer) override;
    
    // YOLO 特定推理接口
    std::vector<Detection> inference(const cv::Mat& input);
    
    // 配置接口
    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void setTargetClass(int classId);
    void loadClassNames(const std::string& namesFile);
    
    // 访问器
    Version getVersion() const { return version_; }
    int getInputWidth() const { return inputWidth_; }
    int getInputHeight() const { return inputHeight_; }
    int getNumClasses() const { return numClasses_; }
    
private:
    // 后处理方法（版本特定）
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
    
    // NMS（非极大值抑制）实现
    std::vector<int> performNMS(
        const std::vector<cv::Rect2f>& boxes,
        const std::vector<float>& scores,
        float nmsThreshold
    );
    
    // IoU 计算
    float calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b);
    
    // 坐标转换辅助函数
    void xywhToxyxy(float cx, float cy, float w, float h, 
                    float& x1, float& y1, float& x2, float& y2);
    
    // 成员变量
    Version version_;
    float confidenceThreshold_;
    float nmsThreshold_;
    int targetClassId_;
    
    int inputWidth_;
    int inputHeight_;
    int numClasses_;
    
    std::vector<std::string> classNames_;
    
    // ONNX Runtime 相关（继承自 Model 基类）
    // Ort::Session* session_;
    // Ort::Env env_;
};

#endif // MODEL_YOLO_H