#include "yolo_detector_filter.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <plugin-support.h>
#include "obs-utils/obs-utils.h"
#include "consts.h"

void renderDetectionBoxes(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	std::lock_guard<std::mutex> lock(filter->detectionsMutex);

	if (filter->detections.empty()) {
		return;
	}

	gs_effect_t *solid = filter->solidEffect;
	gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
	gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

	struct vec4 color;
	float r = ((filter->bboxColor >> 16) & 0xFF) / 255.0f;
	float g = ((filter->bboxColor >> 8) & 0xFF) / 255.0f;
	float b = (filter->bboxColor & 0xFF) / 255.0f;
	float a = ((filter->bboxColor >> 24) & 0xFF) / 255.0f;
	vec4_set(&color, r, g, b, a);

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);
	gs_effect_set_vec4(colorParam, &color);

	for (const auto& det : filter->detections) {
		float x = det.x * frameWidth;
		float y = det.y * frameHeight;
		float w = det.width * frameWidth;
		float h = det.height * frameHeight;

		// 使用 GS_LINESTRIP 绘制闭合矩形（5个顶点）
		gs_render_start(true);
		gs_vertex2f(x, y);         // 左上
		gs_vertex2f(x + w, y);     // 右上
		gs_vertex2f(x + w, y + h); // 右下
		gs_vertex2f(x, y + h);     // 左下
		gs_vertex2f(x, y);         // 回到左上，闭合
		gs_render_stop(GS_LINESTRIP);
	}

	gs_technique_end_pass(tech);
	gs_technique_end(tech);
}

void renderKalmanPredictions(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (!filter->useKalmanTracker || !filter->showKalmanPredictions) {
		return;
	}

	std::lock_guard<std::mutex> lock(filter->kalmanPredictionsMutex);

	if (filter->kalmanPredictions.empty()) {
		return;
	}

	gs_effect_t *solid = filter->solidEffect;
	gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
	gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

	// 使用青色绘制预测框
	struct vec4 color;
	float r = ((filter->kalmanPredictionColor >> 16) & 0xFF) / 255.0f;
	float g = ((filter->kalmanPredictionColor >> 8) & 0xFF) / 255.0f;
	float b = (filter->kalmanPredictionColor & 0xFF) / 255.0f;
	float a = ((filter->kalmanPredictionColor >> 24) & 0xFF) / 255.0f;
	vec4_set(&color, r, g, b, a);

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);
	gs_effect_set_vec4(colorParam, &color);

	for (const auto& pred : filter->kalmanPredictions) {
		float x = pred.x * frameWidth;
		float y = pred.y * frameHeight;
		float w = pred.width * frameWidth;
		float h = pred.height * frameHeight;

		// 使用虚线样式绘制预测框
		float dashLength = 8.0f;
		float gapLength = 4.0f;

		// 上边
		gs_render_start(true);
		for (float px = x; px < x + w; px += dashLength + gapLength) {
			float endX = std::min(px + dashLength, x + w);
			gs_vertex2f(px, y);
			gs_vertex2f(endX, y);
		}
		gs_render_stop(GS_LINES);

		// 下边
		gs_render_start(true);
		for (float px = x; px < x + w; px += dashLength + gapLength) {
			float endX = std::min(px + dashLength, x + w);
			gs_vertex2f(px, y + h);
			gs_vertex2f(endX, y + h);
		}
		gs_render_stop(GS_LINES);

		// 左边
		gs_render_start(true);
		for (float py = y; py < y + h; py += dashLength + gapLength) {
			float endY = std::min(py + dashLength, y + h);
			gs_vertex2f(x, py);
			gs_vertex2f(x, endY);
		}
		gs_render_stop(GS_LINES);

		// 右边
		gs_render_start(true);
		for (float py = y; py < y + h; py += dashLength + gapLength) {
			float endY = std::min(py + dashLength, y + h);
			gs_vertex2f(x + w, py);
			gs_vertex2f(x + w, endY);
		}
		gs_render_stop(GS_LINES);
	}

	gs_technique_end_pass(tech);
	gs_technique_end(tech);
}

void renderKalmanTrajectories(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (!filter->useKalmanTracker || !filter->showKalmanTrajectories) {
		return;
	}

	std::lock_guard<std::mutex> lock(filter->kalmanTrajectoriesMutex);

	if (filter->kalmanTrajectories.empty()) {
		return;
	}

	gs_effect_t *solid = filter->solidEffect;
	gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
	gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

	// 使用黄色绘制预测轨迹
	struct vec4 color;
	float r = ((filter->kalmanTrajectoryColor >> 16) & 0xFF) / 255.0f;
	float g = ((filter->kalmanTrajectoryColor >> 8) & 0xFF) / 255.0f;
	float b = (filter->kalmanTrajectoryColor & 0xFF) / 255.0f;
	float a = ((filter->kalmanTrajectoryColor >> 24) & 0xFF) / 255.0f;
	vec4_set(&color, r, g, b, a);

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);
	gs_effect_set_vec4(colorParam, &color);

	for (const auto& trajectory : filter->kalmanTrajectories) {
		if (trajectory.size() < 2) continue;

		gs_render_start(true);
		for (size_t i = 0; i < trajectory.size() - 1; ++i) {
			float x1 = trajectory[i].first * frameWidth;
			float y1 = trajectory[i].second * frameHeight;
			float x2 = trajectory[i + 1].first * frameWidth;
			float y2 = trajectory[i + 1].second * frameHeight;

			gs_vertex2f(x1, y1);
			gs_vertex2f(x2, y2);
		}
		gs_render_stop(GS_LINES);

		// 在预测点位置绘制小圆点
		for (size_t i = 1; i < trajectory.size(); ++i) {
			float cx = trajectory[i].first * frameWidth;
			float cy = trajectory[i].second * frameHeight;
			float radius = 3.0f;

			gs_render_start(true);
			for (int a = 0; a < 16; ++a) {
				float angle1 = (a / 16.0f) * 2.0f * 3.14159f;
				float angle2 = ((a + 1) / 16.0f) * 2.0f * 3.14159f;
				gs_vertex2f(cx + radius * cosf(angle1), cy + radius * sinf(angle1));
				gs_vertex2f(cx + radius * cosf(angle2), cy + radius * sinf(angle2));
			}
			gs_render_stop(GS_LINES);
		}
	}

	gs_technique_end_pass(tech);
	gs_technique_end(tech);
}

void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (!filter->showFOV) {
		return;
	}

	gs_effect_t *solid = filter->solidEffect;
	gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
	gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

	float centerX = frameWidth / 2.0f;
	float centerY = frameHeight / 2.0f;
	float radius = filter->useDynamicFOV ? filter->currentFovRadius : static_cast<float>(filter->fovRadius);
	float crossLineLength = static_cast<float>(filter->fovCrossLineScale);

	struct vec4 color;
	float r = ((filter->fovColor >> 16) & 0xFF) / 255.0f;
	float g = ((filter->fovColor >> 8) & 0xFF) / 255.0f;
	float b = (filter->fovColor & 0xFF) / 255.0f;
	float a = ((filter->fovColor >> 24) & 0xFF) / 255.0f;
	vec4_set(&color, r, g, b, a);

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);
	gs_effect_set_vec4(colorParam, &color);

	// 渲染十字线
	if (filter->showFOVCross) {
		gs_render_start(true);
		gs_vertex2f(centerX - crossLineLength, centerY);
		gs_vertex2f(centerX + crossLineLength, centerY);
		gs_vertex2f(centerX, centerY - crossLineLength);
		gs_vertex2f(centerX, centerY + crossLineLength);
		gs_render_stop(GS_LINES);
	}

	// 渲染圆圈
	if (filter->showFOVCircle) {
		const int circleSegments = 64;
		gs_render_start(true);
		for (int i = 0; i <= circleSegments; ++i) {
			float angle = 2.0f * 3.1415926f * static_cast<float>(i) / static_cast<float>(circleSegments);
			float x = centerX + radius * cosf(angle);
			float y = centerY + radius * sinf(angle);
			gs_vertex2f(x, y);
		}
		gs_render_stop(GS_LINESTRIP);
	}

	gs_technique_end_pass(tech);
	gs_technique_end(tech);
}

void renderRegion(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (!filter->useRegion) {
		return;
	}

	gs_effect_t *solid = filter->solidEffect;
	gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
	gs_eparam_t *colorParam = gs_effect_get_param_by_name(solid, "color");

	// 计算区域边界（归一化坐标转像素坐标）
	float x = static_cast<float>(filter->regionX);
	float y = static_cast<float>(filter->regionY);
	float w = static_cast<float>(filter->regionWidth);
	float h = static_cast<float>(filter->regionHeight);

	// 使用黄色虚线
	struct vec4 color;
	vec4_set(&color, 1.0f, 1.0f, 0.0f, 1.0f); // 黄色

	gs_technique_begin(tech);
	gs_technique_begin_pass(tech, 0);
	gs_effect_set_vec4(colorParam, &color);

	// 绘制虚线矩形边框
	float dashLength = 10.0f; // 虚线段长度
	float gapLength = 5.0f;   // 间隔长度

	// 上边
	gs_render_start(true);
	for (float px = x; px < x + w; px += dashLength + gapLength) {
		float endX = std::min(px + dashLength, x + w);
		gs_vertex2f(px, y);
		gs_vertex2f(endX, y);
	}
	gs_render_stop(GS_LINES);

	// 下边
	gs_render_start(true);
	for (float px = x; px < x + w; px += dashLength + gapLength) {
		float endX = std::min(px + dashLength, x + w);
		gs_vertex2f(px, y + h);
		gs_vertex2f(endX, y + h);
	}
	gs_render_stop(GS_LINES);

	// 左边
	gs_render_start(true);
	for (float py = y; py < y + h; py += dashLength + gapLength) {
		float endY = std::min(py + dashLength, y + h);
		gs_vertex2f(x, py);
		gs_vertex2f(x, endY);
	}
	gs_render_stop(GS_LINES);

	// 右边
	gs_render_start(true);
	for (float py = y; py < y + h; py += dashLength + gapLength) {
		float endY = std::min(py + dashLength, y + h);
		gs_vertex2f(x + w, py);
		gs_vertex2f(x + w, endY);
	}
	gs_render_stop(GS_LINES);

	gs_technique_end_pass(tech);
	gs_technique_end(tech);
}

static void renderLabelsWithOpenCV(cv::Mat &image, yolo_detector_filter *filter)
{
	std::vector<Detection> detectionsCopy;
	{
		std::lock_guard<std::mutex> lock(filter->detectionsMutex);
		if (filter->detections.empty()) {
			return;
		}
		detectionsCopy = filter->detections;
	}

	int frameWidth = image.cols;
	int frameHeight = image.rows;
	int fontFace = cv::FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.5;
	int thickness = 2;
	int baseline = 0;

	for (const auto& det : detectionsCopy) {
		int x = static_cast<int>(det.x * frameWidth);
		int y = static_cast<int>(det.y * frameHeight);
		int w = static_cast<int>(det.width * frameWidth);
		int h = static_cast<int>(det.height * frameHeight);

		// 构建标签文本：类别ID(0-10) + 置信度(浮点数)
		char labelText[64];
		snprintf(labelText, sizeof(labelText), "%d: %.2f", det.classId, det.confidence);

		// 获取文本大小
		cv::Size textSize = cv::getTextSize(labelText, fontFace, fontScale, thickness, &baseline);

		// 绘制标签背景
		cv::Point textOrg(x, y - 5);
		cv::rectangle(image, 
			cv::Point(textOrg.x, textOrg.y - textSize.height - 5),
			cv::Point(textOrg.x + textSize.width + 10, textOrg.y + baseline),
			cv::Scalar(0, 0, 0, 200),
			-1);

		// 绘制文本
		cv::putText(image, labelText, 
			cv::Point(textOrg.x + 5, textOrg.y),
			fontFace, fontScale, 
			cv::Scalar(0, 255, 0, 255), 
			thickness);
	}
}

void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight)
{
	if (filter->coordinateOutputPath.empty()) {
		return;
	}

	std::lock_guard<std::mutex> lock(filter->detectionsMutex);

	try {
		std::ofstream file(filter->coordinateOutputPath);
		if (!file.is_open()) {
			obs_log(LOG_ERROR, "[YOLO Filter] Failed to open coordinate file: %s", 
					filter->coordinateOutputPath.c_str());
			return;
		}

		auto now = std::chrono::system_clock::now();
		auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
			now.time_since_epoch()
		).count();

		file << "{\n";
		file << "  \"timestamp\": " << timestamp << ",\n";
		file << "  \"frame_width\": " << frameWidth << ",\n";
		file << "  \"frame_height\": " << frameHeight << ",\n";
		file << "  \"detections\": [\n";

		for (size_t i = 0; i < filter->detections.size(); ++i) {
			const auto& det = filter->detections[i];

			file << "    {\n";
			file << "      \"class_id\": " << det.classId << ",\n";
			file << "      \"class_name\": \"" << det.className << "\",\n";
			file << "      \"confidence\": " << det.confidence << ",\n";
			file << "      \"bbox\": {\n";
			file << "        \"x\": " << (det.x * frameWidth) << ",\n";
			file << "        \"y\": " << (det.y * frameHeight) << ",\n";
			file << "        \"width\": " << (det.width * frameWidth) << ",\n";
			file << "        \"height\": " << (det.height * frameHeight) << "\n";
			file << "      },\n";
			file << "      \"center\": {\n";
			file << "        \"x\": " << (det.centerX * frameWidth) << ",\n";
			file << "        \"y\": " << (det.centerY * frameHeight) << "\n";
			file << "      },\n";
			file << "      \"track_id\": " << det.trackId << "\n";
			file << "    }";

			if (i < filter->detections.size() - 1) {
				file << ",";
			}
			file << "\n";
		}

		file << "  ]\n";
		file << "}\n";

		file.close();

	} catch (const std::exception& e) {
		obs_log(LOG_ERROR, "[YOLO Filter] Error exporting coordinates: %s", e.what());
	}
}

