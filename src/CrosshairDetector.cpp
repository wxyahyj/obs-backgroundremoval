#ifdef _WIN32

#include "CrosshairDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <climits>
#include <obs-module.h>
#include <plugin-support.h>

// ============================================================
// RGB -> HSV (OpenCV range: H 0-180, S/V 0-255)
// ============================================================
void CrosshairDetector::rgbToHsv(int r, int g, int b, int& h, int& s, int& v)
{
	float rf = r / 255.0f;
	float gf = g / 255.0f;
	float bf = b / 255.0f;

	float maxC = std::max({rf, gf, bf});
	float minC = std::min({rf, gf, bf});
	float delta = maxC - minC;

	v = static_cast<int>(maxC * 255.0f);

	if (maxC < 0.0001f) {
		s = 0; h = 0; return;
	}
	s = static_cast<int>((delta / maxC) * 255.0f);

	if (delta < 0.0001f) {
		h = 0; return;
	}

	float hf;
	if (maxC == rf) {
		hf = 60.0f * std::fmod((gf - bf) / delta, 6.0f);
	} else if (maxC == gf) {
		hf = 60.0f * ((bf - rf) / delta + 2.0f);
	} else {
		hf = 60.0f * ((rf - gf) / delta + 4.0f);
	}
	if (hf < 0.0f) hf += 360.0f;
	h = static_cast<int>(hf * 0.5f);
}

// ============================================================
// Match single pixel against HSV range
// ============================================================
bool CrosshairDetector::matchPixelHSV(uint8_t b, uint8_t g, uint8_t r,
                                       int hMin, int hMax,
                                       int sMin, int sMax,
                                       int vMin, int vMax) const
{
	int h, s, v;
	rgbToHsv(r, g, b, h, s, v);

	bool hMatch;
	if (hMin <= hMax) {
		hMatch = (h >= hMin && h <= hMax);
	} else {
		// Wrap-around case (e.g. hMin=170, hMax=10 for red)
		hMatch = (h >= hMin || h <= hMax);
	}

	return hMatch && (s >= sMin && s <= sMax) && (v >= vMin && v <= vMax);
}

// ============================================================
// Core: scan ROI for matching pixels, compute centroid
// ============================================================
bool CrosshairDetector::scanForColor(const cv::Mat& bgrROI,
                                      int hMin, int hMax,
                                      int sMin, int sMax,
                                      int vMin, int vMax,
                                      int minPixels,
                                      float& outX, float& outY) const
{
	if (bgrROI.empty()) return false;

	int roiW = bgrROI.cols;
	int roiH = bgrROI.rows;
	int roiX0 = 0, roiY0 = 0; // ROI offset within full frame (set by caller via ROI position)

	// Collect matched pixels
	double sumX = 0.0, sumY = 0.0;
	int matchCount = 0;

	for (int y = 0; y < roiH; ++y) {
		const uint8_t* row = bgrROI.ptr<uint8_t>(y);
		for (int x = 0; x < roiW; ++x) {
			uint8_t b = row[x * 3];
			uint8_t g = row[x * 3 + 1];
			uint8_t r = row[x * 3 + 2];

			if (matchPixelHSV(b, g, r, hMin, hMax, sMin, sMax, vMin, vMax)) {
				sumX += x;
				sumY += y;
				matchCount++;
			}
		}
	}

	if (matchCount < minPixels) return false;

	outX = static_cast<float>(sumX / matchCount);
	outY = static_cast<float>(sumY / matchCount);
	return true;
}

// ============================================================
// Detect: scan around frame center, return Detection list
// ============================================================
std::vector<Detection> CrosshairDetector::detect(const cv::Mat& bgrFrame,
                                                  int frameWidth, int frameHeight,
                                                  int cropX, int cropY,
                                                  float fovCenterX, float fovCenterY,
                                                  float fovRadiusNorm)
{
	std::vector<Detection> results;

	if (!config_.enabled || bgrFrame.empty()) return results;

	// Frame interval
	frameCounter_++;
	if (config_.detectEveryNFrames > 1 &&
	    frameCounter_ % config_.detectEveryNFrames != 0) {
		return results;
	}

	// Determine search radius
	int searchR = config_.searchRadius;
	if (searchR <= 0) {
		// Auto: 1/6 of frame width
		searchR = std::max(20, bgrFrame.cols / 6);
	}

	// Search center = frame center
	float centerX = bgrFrame.cols * 0.5f;
	float centerY = bgrFrame.rows * 0.5f;

	// ROI: square around center, clamped to frame bounds
	int x0 = std::max(0, static_cast<int>(centerX) - searchR);
	int x1 = std::min(bgrFrame.cols, static_cast<int>(centerX) + searchR);
	int y0 = std::max(0, static_cast<int>(centerY) - searchR);
	int y1 = std::min(bgrFrame.rows, static_cast<int>(centerY) + searchR);

	if (x1 <= x0 || y1 <= y0) return results;

	cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
	cv::Mat roiMat = bgrFrame(roi);

	float localX = -1.0f, localY = -1.0f;
	bool found = scanForColor(roiMat,
	                           config_.hMin, config_.hMax,
	                           config_.sMin, config_.sMax,
	                           config_.vMin, config_.vMax,
	                           config_.minPixels,
	                           localX, localY);

	if (!found) return results;

	// Convert local ROI coords to full-frame pixel coords
	float pixelX = localX + x0;
	float pixelY = localY + y0;

	// Convert to normalized coords [0,1] within bgrFrame
	float normX = pixelX / bgrFrame.cols;
	float normY = pixelY / bgrFrame.rows;

	// === ?????v2 ?????????????? + ??? + ??? ===
	if (config_.shapeFilterEnabled) {
		// ????? ROI???????? bounding box
		int bbMinX = INT_MAX, bbMinY = INT_MAX;
		int bbMaxX = INT_MIN, bbMaxY = INT_MIN;
		int matchCountForBb = 0;
		for (int y = 0; y < roiMat.rows; ++y) {
			const uint8_t* row = roiMat.ptr<uint8_t>(y);
			for (int x = 0; x < roiMat.cols; ++x) {
				uint8_t b = row[x * 3];
				uint8_t g = row[x * 3 + 1];
				uint8_t r = row[x * 3 + 2];
				if (matchPixelHSV(b, g, r, config_.hMin, config_.hMax,
				                 config_.sMin, config_.sMax,
				                 config_.vMin, config_.vMax)) {
					matchCountForBb++;
					if (x < bbMinX) bbMinX = x;
					if (x > bbMaxX) bbMaxX = x;
					if (y < bbMinY) bbMinY = y;
					if (y > bbMaxY) bbMaxY = y;
				}
			}
		}
		if (matchCountForBb <= 0) return results;
		int bbW = bbMaxX - bbMinX + 1;
		int bbH = bbMaxY - bbMinY + 1;
		float bbArea = static_cast<float>(bbW) * static_cast<float>(bbH);
		if (bbArea <= 0) return results;
		float fillRatio = matchCountForBb / bbArea;
		float aspectRatio = (bbH > 0) ? static_cast<float>(bbW) / static_cast<float>(bbH) : 0.0f;
		// fillRatio ??
		if (fillRatio < config_.minFillRatio || fillRatio > config_.maxFillRatio) {
			return results;
		}
		// aspectRatio ??
		if (aspectRatio < config_.minAspectRatio || aspectRatio > config_.maxAspectRatio) {
			return results;
		}
		// shapeType: 0=Any ??; 1=Cross ??(????1, fillRatio??); 2=Dot ??(fillRatio??); 3=TShape T??(?)
		switch (static_cast<CrosshairShapeType>(config_.shapeType)) {
			case CrosshairShapeType::Any:
				break;
			case CrosshairShapeType::Cross:
				// ?????????? 1???????-???
				if (aspectRatio < 0.5f || aspectRatio > 2.0f) return results;
				if (fillRatio > 0.5f) return results;
				break;
			case CrosshairShapeType::Dot:
				// ?????????? 1???????
				if (aspectRatio < 0.7f || aspectRatio > 1.4f) return results;
				if (fillRatio < 0.3f) return results;
				break;
			case CrosshairShapeType::TShape:
				// T ??: ???????????
				if (aspectRatio < 0.4f || aspectRatio > 1.5f) return results;
				if (fillRatio < 0.05f || fillRatio > 0.5f) return results;
				break;
		}
		// ???????? width/height??????????????????
		float bbWNorm = static_cast<float>(bbW) / bgrFrame.cols;
		float bbHNorm = static_cast<float>(bbH) / bgrFrame.rows;
		(void)normX;
		(void)normY;
		Detection det;
		det.x = static_cast<float>(bbMinX + x0) / bgrFrame.cols;
		det.y = static_cast<float>(bbMinY + y0) / bgrFrame.rows;
		det.width = bbWNorm;
		det.height = bbHNorm;
		det.centerX = det.x + det.width / 2.0f;
		det.centerY = det.y + det.height / 2.0f;
		det.confidence = fillRatio;
		det.classId = -1;
		results.push_back(det);
		return results;
	}

	Detection det;
	det.x = normX - 0.01f;
	det.y = normY - 0.01f;
	det.width = 0.02f;   // small placeholder
	det.height = 0.02f;
	det.centerX = normX;  // ????: ?????????????
	det.centerY = normY;
	det.confidence = 1.0f;
	det.classId = -1;    // special: crosshair detection

	results.push_back(det);

	return results;
}

// ============================================================
// Pick color from center (simplified: sample center pixel + background contrast)
// ============================================================
bool CrosshairDetector::pickColorFromCenter(const cv::Mat& bgrFrame,
                                             int frameWidth, int frameHeight,
                                             int cropX, int cropY)
{
	if (bgrFrame.empty()) return false;

	int cx = bgrFrame.cols / 2;
	int cy = bgrFrame.rows / 2;

	// Sample 9x9 region, compute background mean
	int outerR = 4;
	int x0 = std::max(0, cx - outerR);
	int x1 = std::min(bgrFrame.cols, cx + outerR + 1);
	int y0 = std::max(0, cy - outerR);
	int y1 = std::min(bgrFrame.rows, cy + outerR + 1);

	if (x1 <= x0 || y1 <= y0) return false;

	cv::Rect patchRect(x0, y0, x1 - x0, y1 - y0);
	cv::Mat patch = bgrFrame(patchRect).clone();
	cv::Scalar bgMean = cv::mean(patch);

	// Find pixel with max distance from background mean in 3x3 around center
	int localCX = cx - x0;
	int localCY = cy - y0;
	int innerR = 1;

	int bestIdx = -1;
	double maxDist = -1;
	for (int dy = -innerR; dy <= innerR; ++dy) {
		for (int dx = -innerR; dx <= innerR; ++dx) {
			int yy = localCY + dy, xx = localCX + dx;
			if (yy < 0 || yy >= patch.rows || xx < 0 || xx >= patch.cols) continue;
			int i = yy * patch.cols + xx;
			double db = (double)patch.data[i * 3]     - bgMean[0];
			double dg = (double)patch.data[i * 3 + 1] - bgMean[1];
			double dr = (double)patch.data[i * 3 + 2] - bgMean[2];
			double dist = db * db + dg * dg + dr * dr;
			if (dist > maxDist) { maxDist = dist; bestIdx = i; }
		}
	}

	if (bestIdx < 0) return false;

	int bVal = patch.data[bestIdx * 3];
	int gVal = patch.data[bestIdx * 3 + 1];
	int rVal = patch.data[bestIdx * 3 + 2];

	cv::Mat bestBgr(1, 1, CV_8UC3, cv::Scalar(bVal, gVal, rVal));
	cv::Mat bestHsv;
	cv::cvtColor(bestBgr, bestHsv, cv::COLOR_BGR2HSV);
	int h = bestHsv.at<cv::Vec3b>(0, 0)[0];
	int s = bestHsv.at<cv::Vec3b>(0, 0)[1];
	int v = bestHsv.at<cv::Vec3b>(0, 0)[2];

	config_.pickedH = h;
	config_.pickedS = s;
	config_.pickedV = v;
	config_.pickedR = rVal;
	config_.pickedG = gVal;
	config_.pickedB = bVal;

	// Set range with tolerances
	config_.hMin = std::max(0, h - config_.hTolerance);
	config_.hMax = std::min(180, h + config_.hTolerance);
	config_.sMin = std::max(0, s - config_.sTolerance);
	config_.sMax = std::min(255, s + config_.sTolerance);
	config_.vMin = std::max(0, v - config_.vTolerance);
	config_.vMax = std::min(255, v + config_.vTolerance);

	clampHSV(config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);

	config_.colorPicked = true;
	config_.pickingColor = false;

	obs_log(LOG_INFO, "[CrosshairDetector] Color picked: RGB(%d,%d,%d) HSV(%d,%d,%d) -> H[%d-%d] S[%d-%d] V[%d-%d]",
	        rVal, gVal, bVal, h, s, v,
	        config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);

	return true;
}

bool CrosshairDetector::pickColorFromFrame(const cv::Mat& bgrFrame,
                                            float normX, float normY,
                                            int frameWidth, int frameHeight,
                                            int cropX, int cropY)
{
	// Redirect to center pick for simplicity (v2 uses center-based detection)
	// If specific-position pick is needed, it can be added later
	return pickColorFromCenter(bgrFrame, frameWidth, frameHeight, cropX, cropY);
}

// ============================================================
// Manual RGB -> HSV range
// ============================================================
void CrosshairDetector::applyManualRgb(int r, int g, int b)
{
	int h, s, v;
	rgbToHsv(r, g, b, h, s, v);

	config_.manualR = r;
	config_.manualG = g;
	config_.manualB = b;

	config_.hMin = std::max(0, h - config_.hTolerance);
	config_.hMax = std::min(180, h + config_.hTolerance);
	config_.sMin = std::max(0, s - config_.sTolerance);
	config_.sMax = std::min(255, s + config_.sTolerance);
	config_.vMin = std::max(0, v - config_.vTolerance);
	config_.vMax = std::min(255, v + config_.vTolerance);

	clampHSV(config_.hMin, config_.hMax, config_.sMin, config_.sMax, config_.vMin, config_.vMax);

	config_.colorPicked = true;
}

// ============================================================
// Config & helpers
// ============================================================
void CrosshairDetector::updateConfig(const CrosshairDetectorConfig& cfg)
{
	config_ = cfg;
}

void CrosshairDetector::resetTracking()
{
	frameCounter_ = 0;
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

#endif // _WIN32
// [DEPRECATED v2] Stub - template loading removed