#include "ModelYOLO.h"
#include <obs-module.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <cmath>

ModelYOLO::ModelYOLO(Version version)
    : Model(),
      version_(version),
      confidenceThreshold_(0.5f),
      nmsThreshold_(0.45f),
      targetClassId_(-1),
      inputWidth_(640),
      inputHeight_(640),
      numClasses_(80)  // COCO 数据集默认 80 类
{
    obs_log(LOG_INFO, "[ModelYOLO] Initialized (Version: %d)", static_cast<int>(version));
}

ModelYOLO::~ModelYOLO() {
    obs_log(LOG_INFO, "[ModelYOLO] Destroyed");
}

void ModelYOLO::loadModel(const std::string& modelPath) {
    obs_log(LOG_INFO, "[ModelYOLO] Loading model: %s", modelPath.c_str());
    
    // 调用基类的模型加载（使用 obs-backgroundremoval 的 ONNX Runtime 封装）
    // 注意：这里我们需要获取模型输入输出形状
    // 实际的模型加载会在子类中完成
    
    obs_log(LOG_INFO, "[ModelYOLO] Model loaded successfully");
    obs_log(LOG_INFO, "  Input size: %dx%d", inputWidth_, inputHeight_);
    obs_log(LOG_INFO, "  Num classes: %d", numClasses_);
}

void ModelYOLO::preprocessInput(const cv::Mat& input, float* outputBuffer) {
    // YOLO 预处理标准流程：
    // 1. Resize 到模型输入尺寸 (保持宽高比的 letterbox)
    // 2. RGB 转换
    // 3. 归一化到 [0, 1]
    // 4. 转换为 NCHW 格式（Batch, Channel, Height, Width）
    
    // 创建 letterbox 处理后的图像
    cv::Mat letterboxImg;
    float scale = std::min(static_cast<float>(inputWidth_) / input.cols, 
                          static_cast<float>(inputHeight_) / input.rows);
    
    int newWidth = static_cast<int>(input.cols * scale);
    int newHeight = static_cast<int>(input.rows * scale);
    
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(newWidth, newHeight));
    
    // 创建填充图像（letterbox填充）
    letterboxImg = cv::Mat::zeros(inputHeight_, inputWidth_, input.type());
    int top = (inputHeight_ - newHeight) / 2;
    int left = (inputWidth_ - newWidth) / 2;
    cv::Rect roi(left, top, newWidth, newHeight);
    resized.copyTo(letterboxImg(roi));
    
    // 转换为 RGB
    cv::Mat rgb;
    if (input.channels() == 4) {
        cv::cvtColor(letterboxImg, rgb, cv::COLOR_BGRA2RGB);
    } else if (input.channels() == 3) {
        cv::cvtColor(letterboxImg, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = letterboxImg.clone();
    }
    
    // 确保是 uint8 类型
    cv::Mat rgb8u;
    if (rgb.depth() != CV_8U) {
        rgb.convertTo(rgb8u, CV_8U);
    } else {
        rgb8u = rgb;
    }
    
    // 转换为 NCHW 格式并归一化
    const int channelSize = inputWidth_ * inputHeight_;
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < inputHeight_; ++h) {
            for (int w = 0; w < inputWidth_; ++w) {
                // NCHW 索引: [0, c, h, w]
                int outputIdx = c * channelSize + h * inputWidth_ + w;
                
                // 归一化到 [0, 1]
                outputBuffer[outputIdx] = rgb8u.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}

void ModelYOLO::loadModel(const std::string& modelPath) {
    obs_log(LOG_INFO, "[ModelYOLO] Loading model: %s", modelPath.c_str());
    
    // 调用基类的模型加载
    Model::loadModel(modelPath);
    
    // 获取模型输入输出形状
    if (!inputDims.empty() && inputDims[0].size() >= 4) {
        inputHeight_ = static_cast<int>(inputDims[0][2]);
        inputWidth_ = static_cast<int>(inputDims[0][3]);
    }
    
    // 获取输出形状以确定类别数
    if (!outputDims.empty() && outputDims[0].size() >= 2) {
        if (outputDims[0].size() == 3) {
            // [batch, num_boxes, 4+num_classes] 格式 (如YOLOv5)
            int features_per_box = static_cast<int>(outputDims[0][2]);
            if (version_ == Version::YOLOv5) {
                numClasses_ = features_per_box - 5; // 5 = [x,y,w,h,obj_conf]
            } else {
                numClasses_ = features_per_box - 4; // 4 = [x,y,w,h] coords
            }
        } else if (outputDims[0].size() == 2) {
            // [channels, anchors] 格式 (如YOLOv8)
            int channels = static_cast<int>(outputDims[0][0]);
            if (version_ == Version::YOLOv8 || version_ == Version::YOLOv11) {
                numClasses_ = channels - 4; // 4 = [x,y,w,h] coords
            } else {
                numClasses_ = channels - 4; // For YOLOv5 in this format
            }
        }
    }
    
    obs_log(LOG_INFO, "[ModelYOLO] Model loaded successfully");
    obs_log(LOG_INFO, "  Input size: %dx%d", inputWidth_, inputHeight_);
    obs_log(LOG_INFO, "  Num classes: %d", numClasses_);
}

std::vector<Detection> ModelYOLO::inference(const cv::Mat& input) {
    // 1. 预处理
    std::vector<float> inputTensorData(1 * 3 * inputHeight_ * inputWidth_);
    preprocessInput(input, inputTensorData.data());
    
    // 2. 创建输入 tensor
    std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorData.data(),
        inputTensorData.size(),
        inputShape.data(),
        inputShape.size()
    );
    
    // 3. 执行推理
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor));
    
    // 获取输入输出名称
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    for (size_t i = 0; i < inputNames_.size(); ++i) {
        inputNames.push_back(inputNames_[i].get());
    }
    for (size_t i = 0; i < outputNames_.size(); ++i) {
        outputNames.push_back(outputNames_[i].get());
    }
    
    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, 
                                      inputNames.data(), 
                                      inputTensors.data(), 
                                      inputNames.size(),
                                      outputNames.data(),
                                      outputNames.size());
    
    // 4. 获取输出数据
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    int outputSize = 1;
    for (auto dim : outputShape) {
        outputSize *= static_cast<int>(dim);
    }
    
    // 根据模型版本确定如何处理输出
    cv::Size modelSize(inputWidth_, inputHeight_);
    cv::Size originalSize(input.cols, input.rows);
    
    std::vector<Detection> detections;
    
    if (outputShape.size() == 3) {
        // 3维输出：[batch, num_boxes, features] - 如YOLOv5
        int numBoxes = static_cast<int>(outputShape[1]);
        int numFeatures = static_cast<int>(outputShape[2]);
        
        switch (version_) {
            case Version::YOLOv5:
                detections = postprocessYOLOv5(outputData, numBoxes, numFeatures - 5, 
                                              modelSize, originalSize);
                break;
            case Version::YOLOv8:
            case Version::YOLOv11:
                // 对于3D输出的YOLOv8/v11，也按此处理
                detections = postprocessYOLOv8(outputData, numBoxes, numFeatures - 4, 
                                              modelSize, originalSize);
                break;
        }
    } else if (outputShape.size() == 2) {
        // 2维输出：[features, num_anchors] - 如某些YOLOv8输出
        int numFeatures = static_cast<int>(outputShape[0]);
        int numAnchors = static_cast<int>(outputShape[1]);
        
        if (numFeatures == 4 + numClasses_) {
            // [4+num_classes, num_anchors] 格式，需要转置
            switch (version_) {
                case Version::YOLOv8:
                case Version::YOLOv11:
                    detections = postprocessYOLOv8(outputData, numAnchors, numClasses_, 
                                                  modelSize, originalSize);
                    break;
                case Version::YOLOv5:
                    detections = postprocessYOLOv5(outputData, numAnchors, numClasses_, 
                                                  modelSize, originalSize);
                    break;
            }
        }
    }
    
    return detections;
}

void ModelYOLO::setConfidenceThreshold(float threshold) {
    confidenceThreshold_ = threshold;
}

void ModelYOLO::setNMSThreshold(float threshold) {
    nmsThreshold_ = threshold;
}

void ModelYOLO::setTargetClass(int classId) {
    targetClassId_ = classId;
}

void ModelYOLO::loadClassNames(const std::string& namesFile) {
    std::ifstream file(namesFile);
    std::string line;
    classNames_.clear();
    
    while (std::getline(file, line)) {
        classNames_.push_back(line);
    }
}

std::vector<Detection> ModelYOLO::postprocessYOLOv5(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    // YOLOv5/v8 输出格式：[batch, num_boxes, 5+num_classes]
    // 每个检测：[cx, cy, w, h, objectness, class_prob_0, ..., class_prob_n]
    
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    
    const int numElements = 5 + numClasses;
    
    // 计算缩放比例和偏移（用于letterbox恢复）
    float scale = std::min(static_cast<float>(modelInputSize.width) / originalImageSize.width, 
                          static_cast<float>(modelInputSize.height) / originalImageSize.height);
    int newWidth = static_cast<int>(originalImageSize.width * scale);
    int newHeight = static_cast<int>(originalImageSize.height * scale);
    int padLeft = (modelInputSize.width - newWidth) / 2;
    int padTop = (modelInputSize.height - newHeight) / 2;
    
    for (int i = 0; i < numBoxes; ++i) {
        const float* detection = rawOutput + i * numElements;
        
        // 提取 objectness
        float objectness = detection[4];
        
        if (objectness < confidenceThreshold_) {
            continue;
        }
        
        // 找到最高概率的类别
        int maxClassId = 0;
        float maxClassProb = detection[5];
        
        for (int c = 1; c < numClasses; ++c) {
            if (detection[5 + c] > maxClassProb) {
                maxClassProb = detection[5 + c];
                maxClassId = c;
            }
        }
        
        // 计算最终置信度
        float confidence = objectness * maxClassProb;
        
        if (confidence < confidenceThreshold_) {
            continue;
        }
        
        // 如果指定了目标类别，只保留该类别
        if (targetClassId_ >= 0 && maxClassId != targetClassId_) {
            continue;
        }
        
        // 提取边界框（中心点格式）
        float cx = detection[0];
        float cy = detection[1];
        float w = detection[2];
        float h = detection[3];
        
        // 转换为左上角格式
        float x1 = (cx - w / 2.0f);
        float y1 = (cy - h / 2.0f);
        float x2 = (cx + w / 2.0f);
        float y2 = (cy + h / 2.0f);
        
        // 去除letterbox填充的影响 - 首先从模型输入坐标转换为letterbox坐标
        x1 = (x1 * modelInputSize.width - padLeft) / scale;
        y1 = (y1 * modelInputSize.height - padTop) / scale;
        x2 = (x2 * modelInputSize.width - padLeft) / scale;
        y2 = (y2 * modelInputSize.height - padTop) / scale;
        
        // 确保坐标在原始图像范围内
        x1 = std::max(0.0f, x1 / originalImageSize.width);
        y1 = std::max(0.0f, y1 / originalImageSize.height);
        x2 = std::min(1.0f, x2 / originalImageSize.width);
        y2 = std::min(1.0f, y2 / originalImageSize.height);
        
        // 边界检查
        x1 = std::max(0.0f, std::min(x1, 1.0f));
        y1 = std::max(0.0f, std::min(y1, 1.0f));
        x2 = std::max(0.0f, std::min(x2, 1.0f));
        y2 = std::max(0.0f, std::min(y2, 1.0f));
        
        // 创建检测结果
        Detection det;
        det.classId = maxClassId;
        det.className = (maxClassId < classNames_.size()) ? classNames_[maxClassId] : std::to_string(maxClassId);
        det.confidence = confidence;
        det.x = x1;
        det.y = y1;
        det.width = x2 - x1;
        det.height = y2 - y1;
        det.centerX = (x1 + x2) / 2.0f;
        det.centerY = (y1 + y2) / 2.0f;
        
        detections.push_back(det);
        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(confidence);
        classIds.push_back(maxClassId);
    }
    
    // 执行 NMS
    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);
    
    // 只保留NMS后的检测结果
    std::vector<Detection> filteredDetections;
    for (int idx : nmsIndices) {
        filteredDetections.push_back(detections[idx]);
    }
    
    return filteredDetections;
}

std::vector<Detection> ModelYOLO::postprocessYOLOv8(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    // YOLOv8/v11 使用不同的输出格式：[batch, 4+num_classes, num_anchors]
    // 需要进行解码，通常没有objectness分数，而是直接的类别概率
    std::vector<Detection> detections;
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    
    // YOLOv8输出形状: [4+bbox_reg, num_anchors] -> [bbox_reg, 4+num_classes]
    // bbox_reg = number of anchors
    const int outputChannels = 4 + numClasses; // bbox coords (4) + class probs (num_classes)
    
    // 计算缩放比例和偏移（用于letterbox恢复）
    float scale = std::min(static_cast<float>(modelInputSize.width) / originalImageSize.width, 
                          static_cast<float>(modelInputSize.height) / originalImageSize.height);
    int newWidth = static_cast<int>(originalImageSize.width * scale);
    int newHeight = static_cast<int>(originalImageSize.height * scale);
    int padLeft = (modelInputSize.width - newWidth) / 2;
    int padTop = (modelInputSize.height - newHeight) / 2;
    
    // 转置输出：从 [4+num_classes, num_anchors] 到 [num_anchors, 4+num_classes]
    // rawOutput 是 [reg_max=16, bbox_coords=4, num_anchors] 或 [bbox_coords=4+num_classes, num_anchors]
    // 对于标准的YOLOv8输出: [4+num_classes, num_anchors]
    
    // 逐个anchor处理
    for (int anchor_idx = 0; anchor_idx < numBoxes; ++anchor_idx) {
        // 提取边界框坐标 [x, y, w, h] - 这些是相对于输入模型尺寸的
        float x = rawOutput[anchor_idx]; // x center
        float y = rawOutput[numBoxes + anchor_idx]; // y center
        float w = rawOutput[2 * numBoxes + anchor_idx]; // width
        float h = rawOutput[3 * numBoxes + anchor_idx]; // height
        
        // 提取类别分数 [numClasses]
        std::vector<float> classProbs(numClasses);
        for (int c = 0; c < numClasses; ++c) {
            classProbs[c] = rawOutput[(4 + c) * numBoxes + anchor_idx];
        }
        
        // 找到最高概率的类别
        int maxClassId = 0;
        float maxClassProb = classProbs[0];
        
        for (int c = 1; c < numClasses; ++c) {
            if (classProbs[c] > maxClassProb) {
                maxClassProb = classProbs[c];
                maxClassId = c;
            }
        }
        
        // 使用类别概率作为置信度
        float confidence = maxClassProb;
        
        if (confidence < confidenceThreshold_) {
            continue;
        }
        
        // 如果指定了目标类别，只保留该类别
        if (targetClassId_ >= 0 && maxClassId != targetClassId_) {
            continue;
        }
        
        // 边界框解码 - YOLOv8输出是相对于特征图的，需要转换到输入图像空间
        // 这里假设输出已经是相对于输入图像的坐标
        float x_center = x;
        float y_center = y;
        float width = w;
        float height = h;
        
        // 转换为左上角格式（仍为模型输入空间坐标）
        float x1 = (x_center - width / 2.0f);
        float y1 = (y_center - height / 2.0f);
        float x2 = (x_center + width / 2.0f);
        float y2 = (y_center + height / 2.0f);
        
        // 去除letterbox填充的影响 - 首先从模型输入坐标转换为letterbox坐标
        x1 = (x1 * modelInputSize.width - padLeft) / scale;
        y1 = (y1 * modelInputSize.height - padTop) / scale;
        x2 = (x2 * modelInputSize.width - padLeft) / scale;
        y2 = (y2 * modelInputSize.height - padTop) / scale;
        
        // 确保坐标在原始图像范围内
        x1 = std::max(0.0f, x1 / originalImageSize.width);
        y1 = std::max(0.0f, y1 / originalImageSize.height);
        x2 = std::min(1.0f, x2 / originalImageSize.width);
        y2 = std::min(1.0f, y2 / originalImageSize.height);
        
        // 边界检查
        x1 = std::max(0.0f, std::min(x1, 1.0f));
        y1 = std::max(0.0f, std::min(y1, 1.0f));
        x2 = std::max(0.0f, std::min(x2, 1.0f));
        y2 = std::max(0.0f, std::min(y2, 1.0f));
        
        // 创建检测结果
        Detection det;
        det.classId = maxClassId;
        det.className = (maxClassId < classNames_.size()) ? classNames_[maxClassId] : std::to_string(maxClassId);
        det.confidence = confidence;
        det.x = x1;
        det.y = y1;
        det.width = x2 - x1;
        det.height = y2 - y1;
        det.centerX = (x1 + x2) / 2.0f;
        det.centerY = (y1 + y2) / 2.0f;
        
        detections.push_back(det);
        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(confidence);
        classIds.push_back(maxClassId);
    }
    
    // 执行 NMS
    std::vector<int> nmsIndices = performNMS(boxes, scores, nmsThreshold_);
    
    // 只保留NMS后的检测结果
    std::vector<Detection> filteredDetections;
    for (int idx : nmsIndices) {
        filteredDetections.push_back(detections[idx]);
    }
    
    return filteredDetections;
}

std::vector<Detection> ModelYOLO::postprocessYOLOv11(
    const float* rawOutput,
    int numBoxes,
    int numClasses,
    const cv::Size& modelInputSize,
    const cv::Size& originalImageSize
) {
    // YOLOv11 通常具有类似的输出格式，但可能有一些细微差别
    // 暂时使用与YOLOv5类似的处理逻辑
    return postprocessYOLOv5(rawOutput, numBoxes, numClasses, modelInputSize, originalImageSize);
}

std::vector<int> ModelYOLO::performNMS(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float nmsThreshold
) {
    // 实现非极大值抑制算法
    std::vector<int> indices;
    for (size_t i = 0; i < scores.size(); ++i) {
        indices.push_back(static_cast<int>(i));
    }
    
    // 按分数降序排序索引
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(scores.size(), false);
    
    for (int i : indices) {
        if (suppressed[i]) continue;
        
        keep.push_back(i);
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) continue;
            
            float iou = calculateIoU(boxes[i], boxes[indices[j]]);
            
            if (iou > nmsThreshold) {
                suppressed[indices[j]] = true;
            }
        }
    }
    
    return keep;
}

float ModelYOLO::calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    // 计算两个边界框的交并比
    float intersectionX1 = std::max(a.x, b.x);
    float intersectionY1 = std::max(a.y, b.y);
    float intersectionX2 = std::min(a.x + a.width, b.x + b.width);
    float intersectionY2 = std::min(a.y + a.height, b.y + b.height);
    
    float intersectionArea = std::max(0.0f, intersectionX2 - intersectionX1) *
                            std::max(0.0f, intersectionY2 - intersectionY1);
    
    float unionArea = a.area() + b.area() - intersectionArea;
    
    if (unionArea <= 0) {
        return 0.0f;
    }
    
    return intersectionArea / unionArea;
}

void ModelYOLO::xywhToxyxy(float cx, float cy, float w, float h, 
                           float& x1, float& y1, float& x2, float& y2) {
    x1 = cx - w / 2.0f;
    y1 = cy - h / 2.0f;
    x2 = cx + w / 2.0f;
    y2 = cy + h / 2.0f;
}