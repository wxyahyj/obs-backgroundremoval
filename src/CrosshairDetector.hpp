#ifndef CROSSHAIR_DETECTOR_HPP
#define CROSSHAIR_DETECTOR_HPP

#ifdef _WIN32

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include "models/Detection.h"

// Crosshair detector config - simplified v2: pure pixel-scan color matching
// [DEPRECATED v2] Shape type enum - kept for compile compatibility
enum class CrosshairShapeType {
	Any = 0,
	Cross = 1,
	Dot = 2,
	TShape = 3
};

struct CrosshairDetectorConfig {
	bool enabled = false;

	// HSV color range (auto-set after pick, also manually tunable)
	int hMin = 0, hMax = 180;
	int sMin = 100, sMax = 255;
	int vMin = 100, vMax = 255;

	// Pick tolerances
	int hTolerance = 10;   // H tolerance (1-90)
	int sTolerance = 40;   // S tolerance (1-128)
	int vTolerance = 40;   // V tolerance (1-128)

	// Pick state (runtime transient, not serialized)
	bool pickingColor = false;
	bool colorPicked = false;
	int pickedH = 0, pickedS = 0, pickedV = 0;
	int pickedR = 0, pickedG = 0, pickedB = 0;

	// Manual RGB input
	int manualR = 0, manualG = 0, manualB = 0;

	// Detection params
	int searchRadius = 0;       // 0=auto 1/6 frame width
	int minPixels = 3;          // minimum matched pixels for valid detection
	int detectEveryNFrames = 1; // detection frame interval
// [DEPRECATED v2] Morphology params - kept for compile compatibility, unused by v2
	int morphKernelSize = 3;
	int erodeIterations = 0;
	int dilateIterations = 3;

	// [DEPRECATED v2] Sub-matrix quantile filter
	int gridRows = 4;
	int gridCols = 4;
	float quantileThreshold = 0.0f;

	// [DEPRECATED v2] Template matching
	float matchThreshold = 0.6f;
	std::string templateImagePath;

	// [DEPRECATED v2] Area filters
	int minArea = 2;
	int maxArea = 5000;

	// [DEPRECATED v2] Shape filter
	bool shapeFilterEnabled = false;
	int shapeType = 0;  // CrosshairShapeType cast to int
	// CrosshairShapeType shapeType = CrosshairShapeType::Any;  // removed - enum gone
	float minFillRatio = 0.05f;
	float maxFillRatio = 0.8f;
	float minAspectRatio = 0.3f;
	float maxAspectRatio = 3.0f;

	// [DEPRECATED v2] Visualization
	bool colorIsolationView = false;
	bool showDebugMask = false;
};

class CrosshairDetector {
public:
	CrosshairDetector() = default;
	~CrosshairDetector() = default;

	// Main detect: input BGR frame, output Detection list with normalized coords
	std::vector<Detection> detect(const cv::Mat& bgrFrame,
	                              int frameWidth, int frameHeight,
	                              int cropX, int cropY,
	                              float fovCenterX, float fovCenterY,
	                              float fovRadiusNorm);

	// Pick color from frame center (simplified)
	bool pickColorFromCenter(const cv::Mat& bgrFrame,
	                         int frameWidth, int frameHeight,
	                         int cropX, int cropY);

	// Pick color from specified normalized position
	bool pickColorFromFrame(const cv::Mat& bgrFrame,
	                        float normX, float normY,
	                        int frameWidth, int frameHeight,
	                        int cropX, int cropY);

	// Apply manual RGB -> HSV range
	void applyManualRgb(int r, int g, int b);

	// Update config
	void updateConfig(const CrosshairDetectorConfig& cfg);

	// Get current config
	const CrosshairDetectorConfig& getConfig() const { return config_; }

	// Reset tracking state
	void resetTracking();


	// [DEPRECATED v2] Stubs for compile compatibility
	void loadTemplate(const std::string& path) {}  // inline stub
	const cv::Mat& getDebugMask() const { static cv::Mat empty; return empty; }
	const cv::Mat& getLastHsvMask() const { static cv::Mat empty; return empty; }
	void getLastMaskROI(int& roiX, int& roiY, int& roiW, int& roiH) const { roiX=roiY=roiW=roiH=0; }
private:
	CrosshairDetectorConfig config_;
	int frameCounter_ = 0;

	// RGB -> HSV conversion (OpenCV range: H 0-180, S/V 0-255)
	static void rgbToHsv(int r, int g, int b, int& h, int& s, int& v);

	// Clamp HSV to valid range
	static void clampHSV(int& hMin, int& hMax, int& sMin, int& sMax, int& vMin, int& vMax);

	// Match a single BGR pixel against HSV range
	bool matchPixelHSV(uint8_t b, uint8_t g, uint8_t r,
	                   int hMin, int hMax, int sMin, int sMax, int vMin, int vMax) const;

	// Core pixel-scan detection (replaces OpenCV inRange + morphology + contours + template)
	// Returns centroid in image pixel coordinates, or (-1,-1) if not found
	bool scanForColor(const cv::Mat& bgrROI,
	                  int hMin, int hMax, int sMin, int sMax, int vMin, int vMax,
	                  int minPixels,
	                  float& outX, float& outY) const;
};

#endif // _WIN32
#endif // CROSSHAIR_DETECTOR_HPP