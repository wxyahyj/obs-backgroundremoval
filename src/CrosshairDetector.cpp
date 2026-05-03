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

	// 采样策略：准星设计就是为了在背景上醒目，所以准星像素 = 与局部背景差异最大的像素
	// 第一步：采样9x9区域，计算局部均值（被背景主导，准星只占1-2像素）
	// 第二步：在中心3x3（准星实际所在位置）找与背景差异最大的像素
	int outerR = 4; // 9x9背景采样区
	int x0 = std::max(0, px - outerR), x1 = std::min(bgrFrame.cols, px + outerR + 1);
	int y0 = std::max(0, py - outerR), y1 = std::min(bgrFrame.rows, py + outerR + 1);

	cv::Mat patch;
	bgrFrame(cv::Rect(x0, y0, x1 - x0, y1 - y0)).copyTo(patch);

	// 计算整个9x9区域的BGR均值作为背景参考（准星仅1-2像素，对均值影响极小）
	cv::Scalar bgMean = cv::mean(patch);

	// 在中心3x3区域内，找与背景均值差异最大的像素
	int innerR = 1; // 3x3搜索区
	int localCX = px - x0; // 点击点在patch中的坐标
	int localCY = py - y0;

	int bestIdx = -1;
	double maxDist = -1;
	for (int dy = -innerR; dy <= innerR; ++dy) {
		for (int dx = -innerR; dx <= innerR; ++dx) {
			int yy = localCY + dy, xx = localCX + dx;
			if (yy < 0 || yy >= patch.rows || xx < 0 || xx >= patch.cols) continue;
			int i = yy * patch.cols + xx;
			double db = static_cast<double>(patch.data[i * 3])     - bgMean[0];
			double dg = static_cast<double>(patch.data[i * 3 + 1]) - bgMean[1];
			double dr = static_cast<double>(patch.data[i * 3 + 2]) - bgMean[2];
			double dist = db * db + dg * dg + dr * dr;
			if (dist > maxDist) {
				maxDist = dist;
				bestIdx = i;
			}
		}
	}

	// 如果3x3区域没找到有效像素，回退到整个patch中找
	if (bestIdx < 0) {
		for (int i = 0; i < patch.rows * patch.cols; ++i) {
			double db = static_cast<double>(patch.data[i * 3])     - bgMean[0];
			double dg = static_cast<double>(patch.data[i * 3 + 1]) - bgMean[1];
			double dr = static_cast<double>(patch.data[i * 3 + 2]) - bgMean[2];
			double dist = db * db + dg * dg + dr * dr;
			if (dist > maxDist) {
				maxDist = dist;
				bestIdx = i;
			}
		}
	}

	// 将选中的BGR像素转为HSV
	int bVal = patch.data[bestIdx * 3];
	int gVal = patch.data[bestIdx * 3 + 1];
	int rVal = patch.data[bestIdx * 3 + 2];
	cv::Mat bestBgr(1, 1, CV_8UC3, cv::Scalar(bVal, gVal, rVal));
	cv::Mat bestHsv;
	cv::cvtColor(bestBgr, bestHsv, cv::COLOR_BGR2HSV);
	int h = bestHsv.at<cv::Vec3b>(0, 0)[0];
	int s = bestHsv.at<cv::Vec3b>(0, 0)[1];
	int v = bestHsv.at<cv::Vec3b>(0, 0)[2];

	// 低对比度警告：如果最大对比度太小，说明没有明显的准星像素
	if (maxDist < 30.0 * 30.0) { // RGB欧氏距离 < 30
		obs_log(LOG_WARNING, "[CrosshairDetector] Low contrast at pick point (dist=%.1f). "
		        "准星颜色与背景太接近，自动取色可能不准确，建议手动输入RGB。", std::sqrt(maxDist));
	}

	// 按容差计算HSV范围
	config_.hMin = h - config_.hTolerance;
	config_.hMax = h + config_.hTolerance;
	config_.sMin = s - config_.sTolerance;
	config_.sMax = s + config_.sTolerance;
	config_.vMin = v - config_.vTolerance;
	config_.vMax = v + config_.vTolerance;

	// S和V下限：如果取到的准星本身就是低饱和度（如白色），不能强制sMin>=1
	// 只在准星颜色高饱和度时才抬高sMin，避免误匹配灰色噪点
	if (s >= 20) {
		config_.sMin = std::max(1, config_.sMin);
	}
	config_.vMin = std::max(1, config_.vMin);

	clampHSV(config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);

	// 记录取色结果
	config_.pickedH = h;
	config_.pickedS = s;
	config_.pickedV = v;
	config_.pickedB = bVal;
	config_.pickedG = gVal;
	config_.pickedR = rVal;
	config_.colorPicked = true;
	config_.pickingColor = false;

	obs_log(LOG_INFO, "[CrosshairDetector] Color picked at (%.2f, %.2f): RGB(%d,%d,%d) HSV(%d,%d,%d) contrast=%.0f bg(%.0f,%.0f,%.0f) -> range H[%d-%d] S[%d-%d] V[%d-%d]",
	        normX, normY, rVal, gVal, bVal, h, s, v, std::sqrt(maxDist),
	        bgMean[2], bgMean[1], bgMean[0],
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

void CrosshairDetector::applyManualRgb(int r, int g, int b)
{
	// RGB→HSV转换
	int h, s, v;
	rgbToHsv(r, g, b, h, s, v);

	// 按容差计算HSV范围
	config_.hMin = h - config_.hTolerance;
	config_.hMax = h + config_.hTolerance;
	config_.sMin = s - config_.sTolerance;
	config_.sMax = s + config_.sTolerance;
	config_.vMin = v - config_.vTolerance;
	config_.vMax = v + config_.vTolerance;

	config_.sMin = (s >= 20) ? std::max(1, config_.sMin) : config_.sMin;
	config_.vMin = std::max(1, config_.vMin);

	clampHSV(config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);

	// 记录
	config_.pickedH = h;
	config_.pickedS = s;
	config_.pickedV = v;
	config_.pickedR = r;
	config_.pickedG = g;
	config_.pickedB = b;
	config_.manualR = r;
	config_.manualG = g;
	config_.manualB = b;
	config_.colorPicked = true;

	obs_log(LOG_INFO, "[CrosshairDetector] Manual RGB(%d,%d,%d) -> HSV(%d,%d,%d) -> range H[%d-%d] S[%d-%d] V[%d-%d]",
	        r, g, b, h, s, v,
	        config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);
}

void CrosshairDetector::rgbToHsv(int r, int g, int b, int& h, int& s, int& v)
{
	// 用OpenCV的cvtColor做精确转换
	cv::Mat bgrPixel(1, 1, CV_8UC3, cv::Scalar(b, g, r));
	cv::Mat hsvPixel;
	cv::cvtColor(bgrPixel, hsvPixel, cv::COLOR_BGR2HSV);
	h = hsvPixel.at<cv::Vec3b>(0, 0)[0];
	s = hsvPixel.at<cv::Vec3b>(0, 0)[1];
	v = hsvPixel.at<cv::Vec3b>(0, 0)[2];
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
		return results;
	}

	// ========== 第1步：ROI裁剪 ==========
	int imgW = bgrFrame.cols;
	int imgH = bgrFrame.rows;

	// 确定搜索中心：优先用前一帧检测到的位置
	float searchCenterX = fovCenterX;
	float searchCenterY = fovCenterY;
	float searchRadius = fovRadiusNorm;

	if (hasLastDetection_) {
		searchCenterX = lastDetectedX_;
		searchCenterY = lastDetectedY_;
		searchRadius = std::min(fovRadiusNorm * 0.5f, 0.2f);
	}

	int searchRadiusPx = static_cast<int>(searchRadius * imgW);
	int searchCenterXPx = static_cast<int>(searchCenterX * imgW);
	int searchCenterYPx = static_cast<int>(searchCenterY * imgH);

	int margin = 10;
	int roiX = std::max(0, searchCenterXPx - searchRadiusPx - margin);
	int roiY = std::max(0, searchCenterYPx - searchRadiusPx - margin);
	int roiW = std::min(imgW - roiX, 2 * (searchRadiusPx + margin));
	int roiH = std::min(imgH - roiY, 2 * (searchRadiusPx + margin));

	if (roiW <= 0 || roiH <= 0) return results;

	cv::Mat roiFrame = bgrFrame(cv::Rect(roiX, roiY, roiW, roiH));

	// ========== 第2步：HSV颜色分割 ==========
	cv::Mat hsvFrame;
	cv::cvtColor(roiFrame, hsvFrame, cv::COLOR_BGR2HSV);

	cv::Scalar lower(config_.hMin, config_.sMin, config_.vMin);
	cv::Scalar upper(config_.hMax, config_.sMax, config_.vMax);
	cv::Mat mask;
	cv::inRange(hsvFrame, lower, upper, mask);

	lastHsvMask_ = mask.clone();
	lastMaskRoiX_ = roiX; lastMaskRoiY_ = roiY;
	lastMaskRoiW_ = roiW; lastMaskRoiH_ = roiH;

	int originalWhitePixels = cv::countNonZero(mask);

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
	if (config_.quantileThreshold > 0.0f) {
		filterBySubMatrixQuantile(mask, config_.gridRows, config_.gridCols, config_.quantileThreshold);
	}

	if (config_.showDebugMask) {
		debugMask_ = mask.clone();
	}

	// ========== 第5步：连通域分析（关键改进！）==========
	struct Candidate {
		float cx, cy;
		float area;
		cv::Rect bbox;
		float score;
	};

	std::vector<Candidate> candidates;

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// 计算理想中心位置（ROI坐标系）
	float idealCenterX = searchCenterX * imgW - roiX;
	float idealCenterY = searchCenterY * imgH - roiY;

	// 计算上一帧位置（ROI坐标系）
	float lastCenterX = hasLastDetection_ ? (lastDetectedX_ * imgW - roiX) : idealCenterX;
	float lastCenterY = hasLastDetection_ ? (lastDetectedY_ * imgH - roiY) : idealCenterY;

	for (size_t i = 0; i < contours.size(); i++) {
		Candidate cand;
		cand.area = static_cast<float>(cv::contourArea(contours[i]));

		// 面积过滤
		if (cand.area < config_.minArea || cand.area > config_.maxArea) {
			continue;
		}

		cv::Moments m = cv::moments(contours[i]);
		if (m.m00 <= 0) continue;
		cand.cx = static_cast<float>(m.m10 / m.m00);
		cand.cy = static_cast<float>(m.m01 / m.m00);
		cand.bbox = cv::boundingRect(contours[i]);

		// ========== 评分机制 ==========
		cand.score = 0.0f;

		// 1. 面积评分：准心通常是小面积，大面积是背景
		// 面积越小分数越高（但有最小限制）
		float areaScore = 0.0f;
		if (cand.area < 500) {
			areaScore = 100.0f;  // 小面积高分
		} else if (cand.area < 2000) {
			areaScore = 50.0f;   // 中等面积中等分
		} else {
			areaScore = 10.0f;   // 大面积低分
		}
		cand.score += areaScore;

		// 2. 距离理想中心的评分
		float distToIdeal = std::sqrt(std::pow(cand.cx - idealCenterX, 2) + std::pow(cand.cy - idealCenterY, 2));
		float distScore = std::max(0.0f, 100.0f - distToIdeal / 3.0f);
		cand.score += distScore;

		// 3. 如果有前一帧结果，距离前一帧位置的评分（权重最高）
		if (hasLastDetection_) {
			float distToLast = std::sqrt(std::pow(cand.cx - lastCenterX, 2) + std::pow(cand.cy - lastCenterY, 2));
			float lastPosScore = std::max(0.0f, 200.0f - distToLast / 2.0f);
			cand.score += lastPosScore;
		}

		candidates.push_back(cand);
	}

	if (candidates.empty()) {
		obs_log(LOG_INFO, "[CrosshairDetector] 检测失败: 没有找到符合条件的连通域 (原始mask有%d个白色像素)", originalWhitePixels);
		return results;
	}

	// ========== 第6步：选择最佳候选 ==========
	std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
		return a.score > b.score;
	});

	Candidate best = candidates[0];

	// ========== 第7步：形状过滤（可选） ==========
	if (config_.shapeFilterEnabled && config_.shapeType != CrosshairShapeType::Any) {
		int expand = 10;
		int shapeROIX = std::max(0, best.bbox.x - expand);
		int shapeROIY = std::max(0, best.bbox.y - expand);
		int shapeROIW = std::min(roiW - shapeROIX, best.bbox.width + 2 * expand);
		int shapeROIH = std::min(roiH - shapeROIY, best.bbox.height + 2 * expand);

		if (shapeROIW > 0 && shapeROIH > 0) {
			cv::Mat shapeMask = mask(cv::Rect(shapeROIX, shapeROIY, shapeROIW, shapeROIH));
			float fillRatio, aspectRatio;
			bool hasCrossPoint;
			if (!filterByShape(shapeMask, fillRatio, aspectRatio, hasCrossPoint)) {
				return results;
			}
		}
	}

	// ========== 第8步：构建检测结果 ==========
	float fullFrameCx = best.cx + roiX;
	float fullFrameCy = best.cy + roiY;

	float dx = fullFrameCx / imgW - fovCenterX;
	float dy = fullFrameCy / imgH - fovCenterY;
	float dist = std::sqrt(dx * dx + dy * dy);
	if (dist > fovRadiusNorm * 1.1f) return results;

	float normCX = fullFrameCx / imgW;
	float normCY = fullFrameCy / imgH;

	float confidence = std::min(1.0f, best.score / 400.0f);
	confidence = std::max(0.1f, confidence);

	Detection det;
	det.classId = -2;
	det.className = "crosshair";
	det.confidence = confidence;
	det.centerX = normCX;
	det.centerY = normCY;
	det.x = (best.bbox.x + roiX) / static_cast<float>(imgW);
	det.y = (best.bbox.y + roiY) / static_cast<float>(imgH);
	det.width = static_cast<float>(best.bbox.width) / imgW;
	det.height = static_cast<float>(best.bbox.height) / imgH;
	det.trackId = -1;
	det.lostFrames = 0;

	// ========== 第9步：模板匹配精确定位（可选） ==========
	if (!templateImage_.empty()) {
		float refinedX = normCX, refinedY = normCY, refinedConf = confidence;
		bool matched = refineByTemplateMatch(bgrFrame,
		                                     cv::Rect(static_cast<int>(det.x * imgW),
		                                              static_cast<int>(det.y * imgH),
		                                              static_cast<int>(det.width * imgW),
		                                              static_cast<int>(det.height * imgH)),
		                                     refinedX, refinedY, refinedConf,
		                                     frameWidth, frameHeight, cropX, cropY);
		if (matched) {
			det.centerX = refinedX;
			det.centerY = refinedY;
			det.confidence = refinedConf;
		}
	}

	results.push_back(det);

	lastDetectedX_ = det.centerX;
	lastDetectedY_ = det.centerY;
	hasLastDetection_ = true;

	return results;
}

void CrosshairDetector::resetTracking() {
	hasLastDetection_ = false;
	lastDetectedX_ = 0.5f;
	lastDetectedY_ = 0.5f;
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

			int totalPixels = cell.rows * cell.cols;
			int whitePixels = cv::countNonZero(cell);
			float ratio = static_cast<float>(whitePixels) / totalPixels;

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

	int expandPx = std::max(templateImage_.cols, templateImage_.rows);
	int x0 = std::max(0, candidateROI.x - expandPx);
	int y0 = std::max(0, candidateROI.y - expandPx);
	int x1 = std::min(bgrFrame.cols, candidateROI.x + candidateROI.width + expandPx);
	int y1 = std::min(bgrFrame.rows, candidateROI.y + candidateROI.height + expandPx);

	cv::Rect searchROI(x0, y0, x1 - x0, y1 - y0);

	if (searchROI.width < templateImage_.cols || searchROI.height < templateImage_.rows) {
		return false;
	}

	cv::Mat searchGray;
	cv::cvtColor(bgrFrame(searchROI), searchGray, cv::COLOR_BGR2GRAY);

	cv::Mat result;
	int matchMethod = cv::TM_CCOEFF_NORMED;
	cv::matchTemplate(searchGray, templateImage_, result, matchMethod);

	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	float bestConfidence = static_cast<float>(maxVal);
	if (bestConfidence < config_.matchThreshold) {
		return false;
	}

	float matchCX = searchROI.x + maxLoc.x + templateImage_.cols * 0.5f;
	float matchCY = searchROI.y + maxLoc.y + templateImage_.rows * 0.5f;

	outX = matchCX / bgrFrame.cols;
	outY = matchCY / bgrFrame.rows;
	outConfidence = bestConfidence;

	return true;
}

bool CrosshairDetector::filterByShape(const cv::Mat& mask, 
                                       float& outFillRatio, 
                                       float& outAspectRatio,
                                       bool& outHasCrossPoint)
{
	if (mask.empty()) return false;

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (contours.empty()) {
		obs_log(LOG_INFO, "[CrosshairDetector] 形状过滤失败: 无轮廓");
		return false;
	}

	int maxIdx = -1;
	double maxArea = 0;
	for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
		double area = cv::contourArea(contours[i]);
		if (area > maxArea) {
			maxArea = area;
			maxIdx = i;
		}
	}

	if (maxIdx < 0) {
		obs_log(LOG_INFO, "[CrosshairDetector] 形状过滤失败: 无有效轮廓");
		return false;
	}

	cv::Rect bbox = cv::boundingRect(contours[maxIdx]);
	double bboxArea = bbox.width * bbox.height;
	outFillRatio = static_cast<float>(maxArea / bboxArea);
	outAspectRatio = static_cast<float>(bbox.width) / std::max(1, bbox.height);
	outHasCrossPoint = detectCrossShape(mask);

	bool passed = true;
	std::string rejectReason;

	if (config_.shapeType == CrosshairShapeType::Cross) {
		if (outFillRatio > config_.maxFillRatio) {
			passed = false;
			rejectReason = "填充率过高(非十字形)";
		}
		if (outAspectRatio < config_.minAspectRatio || outAspectRatio > config_.maxAspectRatio) {
			passed = false;
			rejectReason = "纵横比不符合";
		}
		if (!outHasCrossPoint) {
			passed = false;
			rejectReason = "无十字交叉点";
		}
	}
	else if (config_.shapeType == CrosshairShapeType::Dot) {
		if (outFillRatio < config_.minFillRatio) {
			passed = false;
			rejectReason = "填充率过低(非点状)";
		}
		if (outAspectRatio < config_.minAspectRatio || outAspectRatio > config_.maxAspectRatio) {
			passed = false;
			rejectReason = "纵横比不符合";
		}
	}
	else if (config_.shapeType == CrosshairShapeType::TShape) {
		if (outFillRatio > config_.maxFillRatio) {
			passed = false;
			rejectReason = "填充率过高(非T字形)";
		}
		if (outAspectRatio < config_.minAspectRatio || outAspectRatio > config_.maxAspectRatio) {
			passed = false;
			rejectReason = "纵横比不符合";
		}
	}

	if (!passed) {
		obs_log(LOG_INFO, "[CrosshairDetector] 形状过滤失败: %s (填充率=%.2f, 纵横比=%.2f, 交叉点=%d)",
		        rejectReason.c_str(), outFillRatio, outAspectRatio, outHasCrossPoint ? 1 : 0);
	}

	return passed;
}

bool CrosshairDetector::detectCrossShape(const cv::Mat& mask)
{
	if (mask.empty()) return false;

	cv::Mat maskCopy = mask.clone();

	cv::Mat skel(mask.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;
	
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	
	bool done = false;
	int iterations = 0;
	const int maxIterations = 20;

	do {
		cv::erode(maskCopy, eroded, element);
		cv::dilate(eroded, temp, element);
		cv::subtract(maskCopy, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(maskCopy);
		
		done = (cv::countNonZero(eroded) == 0);
		iterations++;
	} while (!done && iterations < maxIterations);

	cv::Mat crossKernel = (cv::Mat_<uchar>(3, 3) << 
		0, 1, 0,
		1, 1, 1,
		0, 1, 0);
	
	cv::Mat hitResult;
	cv::morphologyEx(skel, hitResult, cv::MORPH_HITMISS, crossKernel);
	
	int crossPoints = cv::countNonZero(hitResult);
	
	return crossPoints > 0;
}

#endif // _WIN32
