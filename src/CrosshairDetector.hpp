#ifndef CROSSHAIR_DETECTOR_HPP
#define CROSSHAIR_DETECTOR_HPP

#ifdef _WIN32

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "models/Detection.h"

// 准星检测器配置
struct CrosshairDetectorConfig {
	bool enabled = false;

	// HSV颜色范围（吸管取色后自动设置，也可手动微调）
	int hMin = 0, hMax = 180;
	int sMin = 100, sMax = 255;
	int vMin = 100, vMax = 255;

	// 吸管取色容差
	int hTolerance = 10;   // H容差 (1-90)
	int sTolerance = 40;   // S容差 (1-128)
	int vTolerance = 40;   // V容差 (1-128)

	// 吸管取色状态（运行时临时状态，不序列化）
	bool pickingColor = false;   // 是否处于取色模式
	bool colorPicked = false;    // 是否已取色成功
	int pickedH = 0, pickedS = 0, pickedV = 0; // 取色结果

	// 形态学参数
	int morphKernelSize = 3;     // 核大小 (1-15)
	int erodeIterations = 1;     // 腐蚀迭代 (0-5)
	int dilateIterations = 2;    // 膨胀迭代 (0-10)

	// 子矩阵分位数过滤
	int gridRows = 8;            // 网格行数 (2-20)
	int gridCols = 8;            // 网格列数 (2-20)
	float quantileThreshold = 0.05f; // 分位数阈值 (0.0-1.0)

	// 模板匹配
	float matchThreshold = 0.6f; // 匹配阈值 (0.0-1.0)
	std::string templateImagePath; // 模板图片路径

	// 通用参数
	int minArea = 10;            // 最小面积 (像素)
	int maxArea = 5000;          // 最大面积 (像素)
	int detectEveryNFrames = 1;  // 检测帧间隔 (1-60)
	int searchRadius = 0;        // 搜索半径（像素，0=自动1/6帧宽，仅搜索准星附近区域）

	// 可视化
	bool colorIsolationView = false;  // 颜色隔离视图（黑底只显示匹配颜色+检测框）
	bool showDebugMask = false;       // 显示HSV掩码调试（半透明白色叠加）
};

class CrosshairDetector {
public:
	CrosshairDetector() = default;
	~CrosshairDetector() = default;

	// 主检测接口：输入BGR帧，输出归一化坐标的Detection列表
	std::vector<Detection> detect(const cv::Mat& bgrFrame,
	                               int frameWidth, int frameHeight,
	                               int cropX, int cropY,
	                               float fovCenterX, float fovCenterY,
	                               float fovRadiusNorm);

	// 吸管取色：从帧中指定归一化坐标位置采样HSV，自动计算范围
	bool pickColorFromFrame(const cv::Mat& bgrFrame,
	                        float normX, float normY,
	                        int frameWidth, int frameHeight,
	                        int cropX, int cropY);

	// 从帧中心取色（简化接口，默认取FOV中心）
	bool pickColorFromCenter(const cv::Mat& bgrFrame,
	                         int frameWidth, int frameHeight,
	                         int cropX, int cropY);

	// 更新配置
	void updateConfig(const CrosshairDetectorConfig& cfg);

	// 获取当前配置
	const CrosshairDetectorConfig& getConfig() const { return config_; }

	// 加载模板图像
	void loadTemplate(const std::string& path);

	// 获取调试掩码（用于渲染）
	const cv::Mat& getDebugMask() const { return debugMask_; }

	// 获取最近的HSV inRange掩码（颜色隔离视图用，未经形态学/分位数过滤）
	const cv::Mat& getLastHsvMask() const { return lastHsvMask_; }

	// 获取最近的HSV掩码ROI偏移（在bgrFrame中的位置）
	void getLastMaskROI(int& roiX, int& roiY, int& roiW, int& roiH) const {
		roiX = lastMaskRoiX_; roiY = lastMaskRoiY_;
		roiW = lastMaskRoiW_; roiH = lastMaskRoiH_;
	}

private:
	CrosshairDetectorConfig config_;
	cv::Mat templateImage_;  // 模板图像（灰度）
	cv::Mat debugMask_;      // 最近一次HSV掩码（调试用）
	cv::Mat lastHsvMask_;    // 最近一次inRange掩码（颜色隔离视图用）
	int lastMaskRoiX_ = 0, lastMaskRoiY_ = 0, lastMaskRoiW_ = 0, lastMaskRoiH_ = 0;
	std::string lastTemplatePath_; // 上次加载的模板路径（避免重复加载）
	int frameCounter_ = 0;

	// 子矩阵分位数过滤：将二值图划分为网格，低于阈值的子区域清零
	void filterBySubMatrixQuantile(cv::Mat& binaryMask, int rows, int cols, float threshold);

	// 模板匹配精确定位：在候选区域内做matchTemplate
	// 返回是否精确定位成功，如果成功则更新centerX/centerY和confidence
	bool refineByTemplateMatch(const cv::Mat& bgrFrame,
	                           const cv::Rect& candidateROI,
	                           float& outX, float& outY,
	                           float& outConfidence,
	                           int frameWidth, int frameHeight,
	                           int cropX, int cropY);

	// HSV范围clamp到合法区间
	static void clampHSV(int& hMin, int& hMax, int& sMin, int& sMax, int& vMin, int& vMax);
};

#endif // _WIN32
#endif // CROSSHAIR_DETECTOR_HPP
