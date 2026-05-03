#ifdef _WIN32

#include "CrosshairDetector.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <obs-module.h>
#include <plugin-support.h>

void CrosshairDetector::updateConfig(const CrosshairDetectorConfig& cfg)
{
	config_ = cfg;

	// 如果模板路径变了，重新加载
	if (!config_.templateImagePath.empty() &&
	    (templateImage_.empty() || config_.templateImagePath != lastTemplatePath_)) {
		loadTemplate(config_.templateImagePath);
	}
}

void CrosshairDetector::loadTemplate(const std::string& path)
{
	lastTemplatePath_ = path;
	if (path.empty()) {
		templateImage_.release();
		return;
	}

	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		obs_log(LOG_WARNING, "[CrosshairDetector] Failed to load template image: %s", path.c_str());
		templateImage_.release();
		return;
	}

	templateImage_ = img;
	obs_log(LOG_INFO, "[CrosshairDetector] Template loaded: %s (%dx%d)",
	        path.c_str(), img.cols, img.rows);
}

void CrosshairDetector::clampHSV(int& hMin, int& hMax, int& sMin, int& sMax, int& vMin, int& vMax)
{
	hMin = std::max(0, std::min(180, hMin));
	hMax = std::max(0, std::min(180, hMax));
	sMin = std::max(0, std::min(255, sMin));
	sMax = std::max(0, std::min(255, sMax));
	vMin = std::max(0, std::min(255, vMin));
	vMax = std::max(0, std::min(255, vMax));
}

bool CrosshairDetector::pickColorFromFrame(const cv::Mat& bgrFrame,
                                            float normX, float normY,
                                            int frameWidth, int frameHeight,
                                            int cropX, int cropY)
{
	if (bgrFrame.empty()) return false;

	// 直接用bgrFrame的实际尺寸计算像素坐标（不依赖frameWidth/frameHeight）
	int px = static_cast<int>(normX * bgrFrame.cols);
	int py = static_cast<int>(normY * bgrFrame.rows);
	px = std::max(0, std::min(bgrFrame.cols - 1, px));
	py = std::max(0, std::min(bgrFrame.rows - 1, py));

	// 采样区域：取3x3均值，更稳定
	int r = 1;
	int x0 = std::max(0, px - r), x1 = std::min(bgrFrame.cols, px + r + 1);
	int y0 = std::max(0, py - r), y1 = std::min(bgrFrame.rows, py + r + 1);

	cv::Mat patch;
	bgrFrame(cv::Rect(x0, y0, x1 - x0, y1 - y0)).copyTo(patch);
	cv::Mat hsvPatch;
	cv::cvtColor(patch, hsvPatch, cv::COLOR_BGR2HSV);

	// 计算区域均值
	cv::Scalar meanHSV = cv::mean(hsvPatch);

	int h = static_cast<int>(meanHSV[0]);
	int s = static_cast<int>(meanHSV[1]);
	int v = static_cast<int>(meanHSV[2]);

	// 按容差计算HSV范围
	config_.hMin = h - config_.hTolerance;
	config_.hMax = h + config_.hTolerance;
	config_.sMin = s - config_.sTolerance;
	config_.sMax = s + config_.sTolerance;
	config_.vMin = v - config_.vTolerance;
	config_.vMax = v + config_.vTolerance;

	clampHSV(config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);

	// 记录取色结果
	config_.pickedH = h;
	config_.pickedS = s;
	config_.pickedV = v;
	config_.colorPicked = true;
	config_.pickingColor = false;

	obs_log(LOG_INFO, "[CrosshairDetector] Color picked at (%.2f, %.2f): H=%d S=%d V=%d -> range H[%d-%d] S[%d-%d] V[%d-%d]",
	        normX, normY, h, s, v,
	        config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);

	return true;
}

bool CrosshairDetector::pickColorFromCenter(const cv::Mat& bgrFrame,
                                             int frameWidth, int frameHeight,
                                             int cropX, int cropY)
{
	// 取帧中心
	return pickColorFromFrame(bgrFrame, 0.5f, 0.5f, frameWidth, frameHeight, cropX, cropY);
}

std::vector<Detection> CrosshairDetector::detect(const cv::Mat& bgrFrame,
                                                  int frameWidth, int frameHeight,
                                                  int cropX, int cropY,
                                                  float fovCenterX, float fovCenterY,
                                                  float fovRadiusNorm)
{
	std::vector<Detection> results;

	if (bgrFrame.empty() || !config_.enabled) {
		return results;
	}

	// 帧间隔控制
	frameCounter_++;
	if (frameCounter_ % config_.detectEveryNFrames != 0) {
		return results;
	}

	// ========== 第0步：吸管取色 ==========
	if (config_.pickingColor) {
		pickColorFromCenter(bgrFrame, frameWidth, frameHeight, cropX, cropY);
		// 取色后本次不执行检测，下一帧再检测
		return results;
	}

	// ========== 第1步：ROI裁剪（FOV区域） ==========
	int imgW = bgrFrame.cols;
	int imgH = bgrFrame.rows;

	// 计算FOV圆形区域的包围盒
	int fovRadiusPx = static_cast<int>(fovRadiusNorm * imgW);
	int fovCenterXPx = static_cast<int>(fovCenterX * imgW);
	int fovCenterYPx = static_cast<int>(fovCenterY * imgH);

	// ROI范围，留一些余量
	int margin = 10;
	int roiX = std::max(0, fovCenterXPx - fovRadiusPx - margin);
	int roiY = std::max(0, fovCenterYPx - fovRadiusPx - margin);
	int roiW = std::min(imgW - roiX, 2 * (fovRadiusPx + margin));
	int roiH = std::min(imgH - roiY, 2 * (fovRadiusPx + margin));

	if (roiW <= 0 || roiH <= 0) return results;

	cv::Mat roiFrame = bgrFrame(cv::Rect(roiX, roiY, roiW, roiH));

	// ========== 第2步：HSV颜色分割 ==========
	cv::Mat hsvFrame;
	cv::cvtColor(roiFrame, hsvFrame, cv::COLOR_BGR2HSV);

	cv::Scalar lower(config_.hMin, config_.sMin, config_.vMin);
	cv::Scalar upper(config_.hMax, config_.sMax, config_.vMax);
	cv::Mat mask;
	cv::inRange(hsvFrame, lower, upper, mask);

	// 保存原始inRange掩码（颜色隔离视图用）
	lastHsvMask_ = mask.clone();
	lastMaskRoiX_ = roiX; lastMaskRoiY_ = roiY;
	lastMaskRoiW_ = roiW; lastMaskRoiH_ = roiH;

	// ========== 第3步：形态学过滤 ==========
	if (config_.erodeIterations > 0 && config_.morphKernelSize > 0) {
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
			cv::Size(config_.morphKernelSize, config_.morphKernelSize));
		cv::erode(mask, mask, kernel, cv::Point(-1, -1), config_.erodeIterations);
	}

	if (config_.dilateIterations > 0 && config_.morphKernelSize > 0) {
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
			cv::Size(config_.morphKernelSize, config_.morphKernelSize));
		cv::dilate(mask, mask, kernel, cv::Point(-1, -1), config_.dilateIterations);
	}

	// ========== 第4步：子矩阵分位数过滤 ==========
	filterBySubMatrixQuantile(mask, config_.gridRows, config_.gridCols, config_.quantileThreshold);

	// 保存调试掩码
	if (config_.showDebugMask) {
		debugMask_ = mask.clone();
	}

	// ========== 第5步：轮廓提取+候选区域 ==========
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area < config_.minArea || area > config_.maxArea) continue;

		cv::Rect bbox = cv::boundingRect(contour);

		// 候选区域中心（ROI坐标系）
		float cx = bbox.x + bbox.width * 0.5f;
		float cy = bbox.y + bbox.height * 0.5f;

		// 转换到全帧坐标系
		cx += roiX;
		cy += roiY;

		// FOV圆形过滤：排除FOV外的候选
		float dx = cx / imgW - fovCenterX;
		float dy = cy / imgH - fovCenterY;
		float dist = std::sqrt(dx * dx + dy * dy);
		if (dist > fovRadiusNorm * 1.1f) continue; // 留10%余量

		// 转换为归一化坐标
		float normCX = cx / imgW;
		float normCY = cy / imgH;
		float normW = static_cast<float>(bbox.width) / imgW;
		float normH = static_cast<float>(bbox.height) / imgH;
		float normX = (bbox.x + roiX) / static_cast<float>(imgW);
		float normY = (bbox.y + roiY) / static_cast<float>(imgH);

		// 默认confidence基于面积占比
		float confidence = static_cast<float>(area) / config_.maxArea;
		confidence = std::min(1.0f, std::max(0.1f, confidence));

		// ========== 第6步：模板匹配精确定位 ==========
		if (!templateImage_.empty()) {
			float refinedX = normCX, refinedY = normCY, refinedConf = confidence;
			bool matched = refineByTemplateMatch(bgrFrame,
			                                     cv::Rect(static_cast<int>(normX * imgW),
			                                              static_cast<int>(normY * imgH),
			                                              static_cast<int>(normW * imgW),
			                                              static_cast<int>(normH * imgH)),
			                                     refinedX, refinedY, refinedConf,
			                                     frameWidth, frameHeight, cropX, cropY);
			if (matched) {
				normCX = refinedX;
				normCY = refinedY;
				confidence = refinedConf;
			}
		}

		// 构建Detection结果
		Detection det;
		det.classId = -2;  // 准星检测专用classId
		det.className = "crosshair";
		det.confidence = confidence;
		det.x = normX;
		det.y = normY;
		det.width = normW;
		det.height = normH;
		det.centerX = normCX;
		det.centerY = normCY;
		det.trackId = -1;
		det.lostFrames = 0;

		results.push_back(det);
	}

	return results;
}

void CrosshairDetector::filterBySubMatrixQuantile(cv::Mat& binaryMask, int rows, int cols, float threshold)
{
	if (binaryMask.empty() || rows <= 0 || cols <= 0 || threshold <= 0.0f) return;

	int cellH = binaryMask.rows / rows;
	int cellW = binaryMask.cols / cols;

	if (cellH <= 0 || cellW <= 0) return;

	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			int y0 = r * cellH;
			int x0 = c * cellW;
			int y1 = (r == rows - 1) ? binaryMask.rows : (r + 1) * cellH;
			int x1 = (c == cols - 1) ? binaryMask.cols : (c + 1) * cellW;

			cv::Rect cellRect(x0, y0, x1 - x0, y1 - y0);
			if (cellRect.x + cellRect.width > binaryMask.cols ||
			    cellRect.y + cellRect.height > binaryMask.rows) continue;

			cv::Mat cell = binaryMask(cellRect);

			// 计算白像素比例
			int totalPixels = cell.rows * cell.cols;
			int whitePixels = cv::countNonZero(cell);
			float ratio = static_cast<float>(whitePixels) / totalPixels;

			// 低于分位数阈值，清零该子区域
			if (ratio < threshold) {
				cell.setTo(0);
			}
		}
	}
}

bool CrosshairDetector::refineByTemplateMatch(const cv::Mat& bgrFrame,
                                               const cv::Rect& candidateROI,
                                               float& outX, float& outY,
                                               float& outConfidence,
                                               int frameWidth, int frameHeight,
                                               int cropX, int cropY)
{
	if (templateImage_.empty()) return false;

	// 候选区域扩展，给模板匹配留搜索空间
	int expandPx = std::max(templateImage_.cols, templateImage_.rows);
	int x0 = std::max(0, candidateROI.x - expandPx);
	int y0 = std::max(0, candidateROI.y - expandPx);
	int x1 = std::min(bgrFrame.cols, candidateROI.x + candidateROI.width + expandPx);
	int y1 = std::min(bgrFrame.rows, candidateROI.y + candidateROI.height + expandPx);

	cv::Rect searchROI(x0, y0, x1 - x0, y1 - y0);

	// 搜索区域必须大于模板尺寸
	if (searchROI.width < templateImage_.cols || searchROI.height < templateImage_.rows) {
		return false;
	}

	// 转灰度
	cv::Mat searchGray;
	cv::cvtColor(bgrFrame(searchROI), searchGray, cv::COLOR_BGR2GRAY);

	// 模板匹配
	cv::Mat result;
	int matchMethod = cv::TM_CCOEFF_NORMED;
	cv::matchTemplate(searchGray, templateImage_, result, matchMethod);

	// 找最佳匹配位置
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	float bestConfidence = static_cast<float>(maxVal);
	if (bestConfidence < config_.matchThreshold) {
		return false; // 匹配置信度不够
	}

	// 计算精确定位中心（搜索ROI坐标系 → 全帧坐标系 → 归一化）
	float matchCX = searchROI.x + maxLoc.x + templateImage_.cols * 0.5f;
	float matchCY = searchROI.y + maxLoc.y + templateImage_.rows * 0.5f;

	outX = matchCX / bgrFrame.cols;
	outY = matchCY / bgrFrame.rows;
	outConfidence = bestConfidence;

	return true;
}

#endif // _WIN32
