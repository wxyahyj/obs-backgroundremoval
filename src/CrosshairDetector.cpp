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

	// 统计原始mask白色像素数（用于诊断）
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
	// 注意：准心线条极细（1-2像素），在网格子区域中占比极低，
	// 默认quantileThreshold=0跳过此步骤，避免误杀准心像素
	if (config_.quantileThreshold > 0.0f) {
		filterBySubMatrixQuantile(mask, config_.gridRows, config_.gridCols, config_.quantileThreshold);
	}

	// 保存调试掩码
	if (config_.showDebugMask) {
		debugMask_ = mask.clone();
	}

	// ========== 第5步：准心定位 ==========
	// 准心是细线条，dilate后仍可能分裂成多个小碎片，用轮廓面积过滤不可靠
	// 改用mask质心：直接计算所有白色像素的加权中心点，更稳定
	cv::Moments m = cv::moments(mask, true);
	if (m.m00 <= 0) {
		obs_log(LOG_INFO, "[CrosshairDetector] 检测失败: 形态学处理后mask无白色像素 (原始mask有%d个白色像素)", originalWhitePixels);
		return results; // mask中没有白色像素
	}

	// 质心坐标（ROI坐标系）
	float cx = static_cast<float>(m.m10 / m.m00);
	float cy = static_cast<float>(m.m01 / m.m00);

	// 白像素总面积（用于置信度）
	double totalArea = m.m00;

	// 可选面积过滤：太少可能是噪点，太多可能匹配了整个背景
	if (totalArea < config_.minArea || totalArea > config_.maxArea) {
		obs_log(LOG_INFO, "[CrosshairDetector] 检测失败: 面积过滤 totalArea=%.0f, minArea=%d, maxArea=%d (原始mask有%d个白色像素)",
		        totalArea, config_.minArea, config_.maxArea, originalWhitePixels);
		return results;
	}

	// ========== 第6步：轮廓形状过滤 ==========
	if (config_.shapeFilterEnabled && config_.shapeType != CrosshairShapeType::Any) {
		float fillRatio, aspectRatio;
		bool hasCrossPoint;
		if (!filterByShape(mask, fillRatio, aspectRatio, hasCrossPoint)) {
			// 形状过滤失败，日志已在filterByShape中输出
			return results;
		}
	}

	// 转换到全帧坐标系
	cx += roiX;
	cy += roiY;

	// FOV圆形过滤
	float dx = cx / imgW - fovCenterX;
	float dy = cy / imgH - fovCenterY;
	float dist = std::sqrt(dx * dx + dy * dy);
	if (dist > fovRadiusNorm * 1.1f) return results;

	// 转换为归一化坐标
	float normCX = cx / imgW;
	float normCY = cy / imgH;

	// 置信度：白色像素越多越可靠，但归一化到0-1
	float confidence = static_cast<float>(std::min(1.0, totalArea / config_.maxArea));
	confidence = std::max(0.1f, confidence);

	// 构建Detection结果
	Detection det;
	det.classId = -2;  // 准星检测专用classId
	det.className = "crosshair";
	det.confidence = confidence;
	det.centerX = normCX;
	det.centerY = normCY;

	// 计算包围盒（mask中白色像素的范围，ROI坐标系 → 归一化）
	std::vector<cv::Point> whitePixels;
	cv::findNonZero(mask, whitePixels);
	if (!whitePixels.empty()) {
		cv::Rect bbox = cv::boundingRect(whitePixels);
		det.x = (bbox.x + roiX) / static_cast<float>(imgW);
		det.y = (bbox.y + roiY) / static_cast<float>(imgH);
		det.width = static_cast<float>(bbox.width) / imgW;
		det.height = static_cast<float>(bbox.height) / imgH;
	} else {
		det.x = normCX;
		det.y = normCY;
		det.width = 0.01f;
		det.height = 0.01f;
	}

	det.trackId = -1;
	det.lostFrames = 0;

	// ========== 第6步：模板匹配精确定位（可选） ==========
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

bool CrosshairDetector::filterByShape(const cv::Mat& mask, 
                                       float& outFillRatio, 
                                       float& outAspectRatio,
                                       bool& outHasCrossPoint)
{
	if (mask.empty()) return false;

	// 找到所有轮廓
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (contours.empty()) {
		obs_log(LOG_INFO, "[CrosshairDetector] 形状过滤失败: 无轮廓");
		return false;
	}

	// 找到最大轮廓（准心主体）
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

	// 计算boundingRect
	cv::Rect bbox = cv::boundingRect(contours[maxIdx]);
	
	// 计算填充率 = 轮廓面积 / boundingRect面积
	double bboxArea = bbox.width * bbox.height;
	outFillRatio = static_cast<float>(maxArea / bboxArea);

	// 计算纵横比 = 宽 / 高
	outAspectRatio = static_cast<float>(bbox.width) / std::max(1, bbox.height);

	// 检测十字形特征
	outHasCrossPoint = detectCrossShape(mask);

	// 根据形状类型进行过滤
	bool passed = true;
	std::string rejectReason;

	if (config_.shapeType == CrosshairShapeType::Cross) {
		// 十字形准心：填充率低（线条细），纵横比接近1，有交叉点
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
		// 点状准心：填充率高（接近圆形），纵横比接近1，无交叉点
		if (outFillRatio < config_.minFillRatio) {
			passed = false;
			rejectReason = "填充率过低(非点状)";
		}
		if (outAspectRatio < config_.minAspectRatio || outAspectRatio > config_.maxAspectRatio) {
			passed = false;
			rejectReason = "纵横比不符合";
		}
		// 点状不需要交叉点，反而有交叉点可能不是点状
	}
	else if (config_.shapeType == CrosshairShapeType::TShape) {
		// T字形：填充率低，纵横比接近1，有交叉点但不完全对称
		if (outFillRatio > config_.maxFillRatio) {
			passed = false;
			rejectReason = "填充率过高(非T字形)";
		}
		if (outAspectRatio < config_.minAspectRatio || outAspectRatio > config_.maxAspectRatio) {
			passed = false;
			rejectReason = "纵横比不符合";
		}
	}
	// Any类型不做额外过滤

	if (!passed) {
		obs_log(LOG_INFO, "[CrosshairDetector] 形状过滤失败: %s (填充率=%.2f, 纵横比=%.2f, 交叉点=%d)",
		        rejectReason.c_str(), outFillRatio, outAspectRatio, outHasCrossPoint ? 1 : 0);
	}

	return passed;
}

bool CrosshairDetector::detectCrossShape(const cv::Mat& mask)
{
	// 十字形检测方法：在mask中心区域检测是否有交叉点
	// 方法：对mask进行细化（骨架化），然后检测交叉点

	if (mask.empty()) return false;

	// 复制mask，避免修改原始数据
	cv::Mat maskCopy = mask.clone();

	// 使用形态学骨架化
	cv::Mat skel(mask.size(), CV_8UC1, cv::Scalar(0));
	cv::Mat temp;
	cv::Mat eroded;
	
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	
	bool done = false;
	int iterations = 0;
	const int maxIterations = 20;  // 防止无限循环

	do {
		cv::erode(maskCopy, eroded, element);
		cv::dilate(eroded, temp, element);
		cv::subtract(maskCopy, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(maskCopy);
		
		done = (cv::countNonZero(eroded) == 0);
		iterations++;
	} while (!done && iterations < maxIterations);

	// 在骨架图中检测交叉点
	// 交叉点特征：周围有多个方向的骨架线
	cv::Mat crossKernel = (cv::Mat_<uchar>(3, 3) << 
		0, 1, 0,
		1, 1, 1,
		0, 1, 0);
	
	cv::Mat hitResult;
	cv::morphologyEx(skel, hitResult, cv::MORPH_HITMISS, crossKernel);
	
	int crossPoints = cv::countNonZero(hitResult);
	
	// 如果有交叉点，返回true
	return crossPoints > 0;
}

#endif // _WIN32
