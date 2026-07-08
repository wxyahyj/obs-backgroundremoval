#include "yolo_detector_filter.h"
#include "yolo-detector-filter.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
#include "MouseController.hpp"
#include "ConfigManager.hpp"
#endif

#include <opencv2/imgproc.hpp>
#include <sstream>
#include <algorithm>

#include <plugin-support.h>
#include "consts.h"

#ifdef _WIN32

static yolo_detector_filter *g_floatingWindowFilter = nullptr;

#include "consts.h"

LRESULT CALLBACK FloatingWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	yolo_detector_filter *filter = g_floatingWindowFilter;

	switch (msg) {
	case WM_CREATE: {
		CREATESTRUCT *cs = (CREATESTRUCT *)lParam;
		SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
		break;
	}
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);
		if (filter) {
			std::lock_guard<std::mutex> lock(filter->floatingWindowMutex);
			if (!filter->floatingWindowFrame.empty()) {
				// 实现双缓冲
				HDC memDC = CreateCompatibleDC(hdc);
				HBITMAP memBitmap = CreateCompatibleBitmap(hdc, filter->floatingWindowFrame.cols, filter->floatingWindowFrame.rows);
				HBITMAP oldBitmap = (HBITMAP)SelectObject(memDC, memBitmap);
				
				// 在内存DC中绘制
				BITMAPINFO bmi = {};
				bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bmi.bmiHeader.biWidth = filter->floatingWindowFrame.cols;
				bmi.bmiHeader.biHeight = -filter->floatingWindowFrame.rows;
				bmi.bmiHeader.biPlanes = 1;
				bmi.bmiHeader.biBitCount = 32;
				bmi.bmiHeader.biCompression = BI_RGB;
				SetDIBitsToDevice(memDC, 0, 0, filter->floatingWindowFrame.cols, filter->floatingWindowFrame.rows,
					0, 0, 0, filter->floatingWindowFrame.rows, filter->floatingWindowFrame.data, &bmi, DIB_RGB_COLORS);
				
				// 一次性复制到窗口DC
				BitBlt(hdc, 0, 0, filter->floatingWindowFrame.cols, filter->floatingWindowFrame.rows,
					memDC, 0, 0, SRCCOPY);
				
				// 清理资源
				SelectObject(memDC, oldBitmap);
				DeleteObject(memBitmap);
				DeleteDC(memDC);
			}
		}
		EndPaint(hwnd, &ps);
		break;
	}
	case WM_LBUTTONDOWN: {
		if (filter) {
			filter->floatingWindowDragging = true;
			POINT pt;
			GetCursorPos(&pt);
			RECT rect;
			GetWindowRect(hwnd, &rect);
			filter->floatingWindowDragOffset.x = pt.x - rect.left;
			filter->floatingWindowDragOffset.y = pt.y - rect.top;
			SetCapture(hwnd);
		}
		break;
	}
	case WM_LBUTTONUP: {
		if (filter) {
			filter->floatingWindowDragging = false;
			ReleaseCapture();
		}
		break;
	}
	case WM_MOUSEMOVE: {
		if (filter && filter->floatingWindowDragging) {
			POINT pt;
			GetCursorPos(&pt);
			SetWindowPos(hwnd, NULL, pt.x - filter->floatingWindowDragOffset.x, pt.y - filter->floatingWindowDragOffset.y,
				0, 0, SWP_NOSIZE | SWP_NOZORDER);
		}
		break;
	}
	case WM_CLOSE: {
			if (filter) {
				filter->showFloatingWindow = false;
				destroyFloatingWindow(filter);
				// 保存悬浮窗关闭状态
				obs_data_t *settings = obs_source_get_settings(filter->source);
				if (settings) {
					obs_data_set_bool(settings, "show_floating_window", false);
					obs_data_release(settings);
				}
			}
			break;
		}
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

void createFloatingWindow(yolo_detector_filter *filter)
{
	if (filter->floatingWindowHandle) {
		return;
	}

	g_floatingWindowFilter = filter;

	WNDCLASS wc = {};
	wc.lpfnWndProc = FloatingWindowProc;
	wc.hInstance = GetModuleHandle(NULL);
	wc.lpszClassName = L"YOLODetectorFloatingWindow";
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClass(&wc);

	int x = GetSystemMetrics(SM_CXSCREEN) / 2 - filter->floatingWindowWidth / 2;
	int y = GetSystemMetrics(SM_CYSCREEN) / 2 - filter->floatingWindowHeight / 2;

	filter->floatingWindowHandle = CreateWindowEx(
		WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
		L"YOLODetectorFloatingWindow",
		L"YOLO Detector",
		WS_POPUP | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		x, y,
		filter->floatingWindowWidth, filter->floatingWindowHeight,
		NULL, NULL, GetModuleHandle(NULL), filter
	);

	filter->floatingWindowX = x;
	filter->floatingWindowY = y;
	filter->floatingWindowDragging = false;

	obs_log(LOG_INFO, "[YOLO Detector] Floating window created");
}

void destroyFloatingWindow(yolo_detector_filter *filter)
{
	if (filter->floatingWindowHandle) {
		DestroyWindow(filter->floatingWindowHandle);
		filter->floatingWindowHandle = nullptr;
		g_floatingWindowFilter = nullptr;
		obs_log(LOG_INFO, "[YOLO Detector] Floating window destroyed");
	}
}

void updateFloatingWindowFrame(yolo_detector_filter *filter, const cv::Mat &frame)
{
	std::lock_guard<std::mutex> lock(filter->floatingWindowMutex);
	frame.copyTo(filter->floatingWindowFrame);

#ifdef _WIN32
	int frameWidth = filter->floatingWindowFrame.cols;
	int frameHeight = filter->floatingWindowFrame.rows;

	// 检测框和trackId已在主渲染中绘制，这里不再重复绘制

	// 在底部绘制统计信息面板（透明背景）
	{
		int frameWidth = filter->floatingWindowFrame.cols;
		int frameHeight = filter->floatingWindowFrame.rows;
		int panelHeight = 50;
		int panelY = frameHeight - panelHeight;
		
		// 绘制统计信息（无背景，透明显示）
		int textY = panelY + 15;
		int lineHeight = 15;
		double fontScale = 0.4;
		int thickness = 1;
		cv::Scalar textColor(255, 255, 255);
		cv::Scalar shadowColor(0, 0, 0);
		
		// 第一行：FPS和推理时间
		char buf[256];
		snprintf(buf, sizeof(buf), "FPS: %.1f | Inference: %.1fms | Detections: %zu",
			filter->currentFps,
			filter->avgInferenceTimeMs,
			filter->detections.size());
		// 绘制阴影提高可读性
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(11, textY + 1), cv::FONT_HERSHEY_SIMPLEX, fontScale, shadowColor, thickness + 1);
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(10, textY), cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
		
		// 第二行：帧统计
		textY += lineHeight;
		snprintf(buf, sizeof(buf), "Submitted: %llu | Inferred: %llu | Consumed: %llu",
			(unsigned long long)filter->framesSubmitted.load(),
			(unsigned long long)filter->framesInferred.load(),
			(unsigned long long)filter->framesConsumed.load());
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(11, textY + 1), cv::FONT_HERSHEY_SIMPLEX, fontScale, shadowColor, thickness + 1);
		cv::putText(filter->floatingWindowFrame, buf, 
			cv::Point(10, textY), cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
		
		// 第三行：丢帧统计（如果有）
		int dropped = filter->framesDropped.load();
		if (dropped > 0) {
			textY += lineHeight;
			snprintf(buf, sizeof(buf), "Dropped: %d", dropped);
			cv::putText(filter->floatingWindowFrame, buf, 
				cv::Point(11, textY + 1), cv::FONT_HERSHEY_SIMPLEX, fontScale, shadowColor, thickness + 1);
			cv::putText(filter->floatingWindowFrame, buf, 
				cv::Point(10, textY), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 100, 255), thickness);
		}
	}

#endif
}

// ============================================================
// PID 调试面板 - 增强版绘制函数
// 6 区域布局: 目标速度 | 误差 | P/I/D分项 | 总输出 | 积分+占比 | 控制模式
// ============================================================

// 绘制单条曲线（复用工具函数）
static void drawSingleCurve(cv::Mat &canvas,
                             int baseY, int topMargin, int bottomMargin,
                             int width,
                             const std::deque<PidDebugData> &history,
                             std::function<float(const PidDebugData&)> getValue,
                             float maxValue,
                             const cv::Scalar &color,
                             int thickness = 2)
{
    if (history.empty() || maxValue < 1e-6f) return;

    int usableHeight = bottomMargin - topMargin - 6;
    float scale = usableHeight / maxValue;

    std::vector<cv::Point> points;
    int x = 0;
    int step = (history.size() > 0) ? std::max(1, width / static_cast<int>(history.size())) : 1;

    for (const auto &data : history) {
        float value = getValue(data);
        int y = baseY - static_cast<int>(value * scale);
        y = std::clamp(y, topMargin + 3, bottomMargin - 3);
        points.push_back(cv::Point(x, y));
        x += step;
        if (x >= width) break;
    }

    if (points.size() > 1) {
        cv::polylines(canvas, points, false, color, thickness);
    }
}

// 绘制区域标题
static void drawRegionLabel(cv::Mat &canvas, int regionIndex, int regionHeight, const char *text)
{
    int y = regionIndex * regionHeight + 14;
    cv::putText(canvas, text, cv::Point(5, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(200, 200, 220), 1);
}

// 绘制区域分隔线（细线）
static void drawRegionSeparator(cv::Mat &canvas, int y, int width)
{
    cv::line(canvas, cv::Point(0, y), cv::Point(width, y), cv::Scalar(55, 55, 65), 1);
}

// 绘制零线（中线）
static void drawZeroLine(cv::Mat &canvas, int y, int width)
{
    cv::line(canvas, cv::Point(0, y), cv::Point(width, y), cv::Scalar(70, 70, 80), 1);
}

// ============================================================
// 核心: 积分仪表盘绘制（圆弧 + 颜色编码）
// ============================================================
static void drawIntegralGauge(cv::Mat &canvas, int cx, int cy, int radius,
                               float ratioX, float ratioY)
{
    // 取两轴中较大的积分占用率
    float ratio = std::max(ratioX, ratioY);

    // 背景圆环（暗灰）
    cv::ellipse(canvas, cv::Point(cx, cy), cv::Size(radius, radius),
                0, 180, 360, cv::Scalar(45, 45, 55), 4);

    // 根据比例选择颜色
    cv::Scalar gaugeColor;
    if (ratio > 0.85f) {
        // 危险区：红色闪烁效果用高亮红
        gaugeColor = cv::Scalar(20, 20, 240);   // BGR 红色
    } else if (ratio > 0.65f) {
        // 警告区：橙黄色
        gaugeColor = cv::Scalar(20, 180, 230);  // BGR 黄色
    } else {
        // 安全区：绿色
        gaugeColor = cv::Scalar(40, 210, 80);   // BGR 绿色
    }

    // 绘制进度弧线（从左到右的半圆，180°~0°）
    float angleEnd = 180.0f - ratio * 180.0f;
    if (ratio > 0.001f) {
        cv::ellipse(canvas, cv::Point(cx, cy), cv::Size(radius, radius),
                    0, angleEnd, 180, gaugeColor, 4);
    }

    // 中心百分比文字
    char buf[16];
    snprintf(buf, sizeof(buf), "%.0f%%", ratio * 100.0f);
    int fontBase = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, nullptr).height;

    // 文字阴影
    cv::putText(canvas, buf, cv::Point(cx + 1, cy + fontBase / 3 + 1),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(15, 15, 15), 1);
    cv::putText(canvas, buf, cv::Point(cx, cy + fontBase / 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, gaugeColor, 1);
}

// ============================================================
// 核心: P/I/D 占比条形图绘制
// ============================================================
static void drawPidBarChart(cv::Mat &canvas, int barX, int barY, int barWidth, int barHeight,
                            float pVal, float iVal, float dVal)
{
    float total = pVal + iVal + dVal;
    if (total < 0.01f) total = 1.0f;

    int pPx = static_cast<int>(barWidth * pVal / total);
    int iPxBg = pPx;
    int iPx = static_cast<int>(barWidth * iVal / total);
    int dPx = barWidth - pPx - iPx;

    // 背景
    cv::rectangle(canvas, cv::Rect(barX, barY, barWidth, barHeight),
                  cv::Scalar(38, 38, 48), -1);

    // P 段（红色）
    if (pPx > 0) {
        cv::rectangle(canvas, cv::Rect(barX, barY, pPx, barHeight),
                      cv::Scalar(90, 50, 55), -1);  // 暗红
    }
    // I 段（绿色）
    if (iPx > 0) {
        cv::rectangle(canvas, cv::Rect(barX + iPxBg, barY, iPx, barHeight),
                      cv::Scalar(45, 95, 55), -1);  // 暗绿
    }
    // D 段（蓝色）
    if (dPx > 0) {
        cv::rectangle(canvas, cv::Rect(barX + iPxBg + iPx, barY, dPx, barHeight),
                      cv::Scalar(50, 55, 95), -1);  // 暗蓝
    }

    // 边框
    cv::rectangle(canvas, cv::Rect(barX, barY, barWidth, barHeight),
                  cv::Scalar(100, 100, 110), 1);

    // 百分比标注
    char buf[64];
    int textY = barY + barHeight - 3;
    int pPct = static_cast<int>(pVal / total * 100.0f);
    int iPct = static_cast<int>(iVal / total * 100.0f);
    int dPct = static_cast<int>(dVal / total * 100.0f);

    // 在各段中心显示百分比（如果段足够宽）
    if (pPx > 22) {
        snprintf(buf, sizeof(buf), "P%d%%", pPct);
        cv::putText(canvas, buf, cv::Point(barX + pPx/2 - 12, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(200, 120, 130), 1);
    }
    if (iPx > 22) {
        snprintf(buf, sizeof(buf), "I%d%%", iPct);
        cv::putText(canvas, buf, cv::Point(barX + iPxBg + iPx/2 - 12, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(120, 220, 140), 1);
    }
    if (dPx > 22) {
        snprintf(buf, sizeof(buf), "D%d%%", dPct);
        cv::putText(canvas, buf, cv::Point(barX + iPxBg + iPx + dPx/2 - 12, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, cv::Scalar(130, 140, 220), 1);
    }
}

// ============================================================
// 控制 mode → 字符串映射
// ============================================================
static const char* controlModeToString(int mode)
{
    switch (mode) {
        case 0: return "IDLE";
        case 1: return "TRACKING";
        case 2: return "LOCKED";
        case 3: return "I-SAT!";
        case 4: return "OSCILLATE";
        case 5: return "PREDICT";
        default: return "UNKNOWN";
    }
}

static cv::Scalar controlModeColor(int mode)
{
    switch (mode) {
        case 0: return cv::Scalar(120, 120, 130);      // IDLE: 灰色
        case 1: return cv::Scalar(255, 255, 255);       // TRACKING: 白色
        case 2: return cv::Scalar(60, 220, 110);        // LOCKED: 绿色
        case 3: return cv::Scalar(20, 20, 250);         // I-SAT: 红色
        case 4: return cv::Scalar(20, 200, 240);        // OSCILLATE: 黄色
        case 5: return cv::Scalar(200, 160, 255);       // PREDICT: 粉紫
        default: return cv::Scalar(150, 150, 150);
    }
}

// ============================================================
// 主绘制入口: drawPidDebugGraph (增强版 6 区域布局)
// ============================================================
static void drawPidDebugGraph(yolo_detector_filter *filter, cv::Mat &canvas)
{
    if (!filter || filter->pidHistory.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(filter->pidHistoryMutex);

    int width = canvas.cols;
    int height = canvas.rows;
    const int NUM_REGIONS = 6;
    int regionH = height / NUM_REGIONS;

    // === 背景 ===
    canvas.setTo(cv::Scalar(25, 26, 33));

    // === 分隔线 ===
    for (int i = 1; i < NUM_REGIONS; ++i) {
        drawRegionSeparator(canvas, i * regionH, width);
    }

    // === 计算最大值用于自适应缩放 ===
    float maxVel = 500.0f;
    float maxErr = 100.0f;
    float maxComp = 50.0f;   // P/I/D 分项最大值
    float maxOut = 50.0f;
    float maxKp = 1.0f;

    for (const auto &data : filter->pidHistory) {
        maxVel = std::max(maxVel, std::max(std::abs(data.targetVelocityX), std::abs(data.targetVelocityY)));
        maxErr = std::max(maxErr, std::max(std::abs(data.errorX), std::abs(data.errorY)));
        maxOut = std::max(maxOut, std::max(std::abs(data.outputX), std::abs(data.outputY)));

        float pMag = std::max(std::abs(data.pTermX), std::abs(data.pTermY));
        float iMag = std::max(std::abs(data.iTermX), std::abs(data.iTermY));
        float dMag = std::max(std::abs(data.dTermX), std::abs(data.dTermY));
        maxComp = std::max(maxComp, std::max(pMag, std::max(iMag, dMag)));

        maxKp = std::max(maxKp, std::abs(data.currentKp));
    }

    // 保证非零下界
    maxVel = std::max(maxVel, 1.0f);
    maxErr = std::max(maxErr, 1.0f);
    maxComp = std::max(maxComp, 1.0f);
    maxOut = std::max(maxOut, 1.0f);
    maxKp = std::max(maxKp, 0.01f);

    // =====================================================================
    // 区域 0: 目标速度 Target Velocity X/Y
    // =====================================================================
    int r0Top = 0 * regionH;
    int r0Bot = 1 * regionH;
    int r0Mid = r0Top + regionH / 2;
    drawRegionLabel(canvas, 0, regionH, "Target Velocity");
    drawZeroLine(canvas, r0Mid, width);

    drawSingleCurve(canvas, r0Mid, r0Top, r0Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.targetVelocityX; },
                    maxVel, cv::Scalar(0, 0, 220), 2);  // X: 红

    drawSingleCurve(canvas, r0Mid, r0Top, r0Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.targetVelocityY; },
                    maxVel, cv::Scalar(0, 200, 0), 2);    // Y: 绿

    // =====================================================================
    // 区域 1: 误差 Error X/Y
    // =====================================================================
    int r1Top = 1 * regionH;
    int r1Bot = 2 * regionH;
    int r1Mid = r1Top + regionH / 2;
    drawRegionLabel(canvas, 1, regionH, "Error X/Y");
    drawZeroLine(canvas, r1Mid, width);

    drawSingleCurve(canvas, r1Mid, r1Top, r1Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.errorX; },
                    maxErr, cv::Scalar(0, 0, 220), 2);

    drawSingleCurve(canvas, r1Mid, r1Top, r1Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.errorY; },
                    maxErr, cv::Scalar(0, 200, 0), 2);

    // =====================================================================
    // 区域 2: P/I/D 分项输出（堆叠曲线图）★核心新增★
    // =====================================================================
    int r2Top = 2 * regionH;
    int r2Bot = 3 * regionH;
    int r2Mid = r2Top + regionH / 2;
    drawRegionLabel(canvas, 2, regionH, "PID Components (P=Red I=Green D=Blue)");
    drawZeroLine(canvas, r2Mid, width);

    // P项曲线（红色系）
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.pTermX; },
                    maxComp, cv::Scalar(100, 50, 55), 2);
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.pTermY; },
                    maxComp, cv::Scalar(140, 75, 80), 2);

    // I项曲线（绿色系）
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.iTermX; },
                    maxComp, cv::Scalar(50, 120, 55), 2);
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.iTermY; },
                    maxComp, cv::Scalar(70, 155, 75), 2);

    // D项曲线（蓝色系）
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.dTermX; },
                    maxComp, cv::Scalar(55, 60, 130), 2);
    drawSingleCurve(canvas, r2Mid, r2Top, r2Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.dTermY; },
                    maxComp, cv::Scalar(75, 80, 160), 2);

    // 图例
    cv::putText(canvas, "P", cv::Point(width - 18, r2Top + 13),
                cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(170, 90, 95), 1);
    cv::putText(canvas, "I", cv::Point(width - 36, r2Top + 13),
                cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(90, 190, 105), 1);
    cv::putText(canvas, "D", cv::Point(width - 54, r2Top + 13),
                cv::FONT_HERSHEY_SIMPLEX, 0.34, cv::Scalar(95, 100, 200), 1);

    // =====================================================================
    // 区域 3: 总输出 Output X/Y
    // =====================================================================
    int r3Top = 3 * regionH;
    int r3Bot = 4 * regionH;
    int r3Mid = r3Top + regionH / 2;
    drawRegionLabel(canvas, 3, regionH, "Total Output X/Y");
    drawZeroLine(canvas, r3Mid, width);

    drawSingleCurve(canvas, r3Mid, r3Top, r3Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.outputX; },
                    maxOut, cv::Scalar(0, 0, 220), 2);

    drawSingleCurve(canvas, r3Mid, r3Top, r3Bot, width, filter->pidHistory,
                    [](const PidDebugData &d) { return d.outputY; },
                    maxOut, cv::Scalar(0, 200, 0), 2);

    // =====================================================================
    // 区域 4: 积分仪表盘 + P/I/D 占比条 ★核心新增★
    // =====================================================================
    int r4Top = 4 * regionH;
    drawRegionLabel(canvas, 4, regionH, "Integral Gauge & PID Ratio");

    if (!filter->pidHistory.empty()) {
        const auto &latest = filter->pidHistory.back();

        // --- 左侧: 积分仪表盘 ---
        int gaugeCX = 48;
        int gaugeCY = r4Top + regionH / 2;
        int gaugeR = std::min(regionH / 2 - 8, 28);
        drawIntegralGauge(canvas, gaugeCX, gaugeCY, gaugeR,
                          latest.integralRatioX, latest.integralRatioY);

        // 仪表盘下方标签
        char iBuf[24];
        snprintf(iBuf, sizeof(iBuf), "I-X:%.0f%% I-Y:%.0f%%",
                 latest.integralRatioX * 100.0f, latest.integralRatioY * 100.0f);
        cv::putText(canvas, iBuf, cv::Point(8, r4Top + regionH - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.30, cv::Scalar(160, 165, 175), 1);

        // --- 右侧: P/I/D 占比条 ---
        int barX = 115;
        int barY = r4Top + regionH / 2 - 9;
        int barW = width - barX - 8;
        int barH = 18;

        float pAbs = std::abs(latest.pTermX) + std::abs(latest.pTermY);
        float iAbs = std::abs(latest.iTermX) + std::abs(latest.iTermY);
        float dAbs = std::abs(latest.dTermX) + std::abs(latest.dTermY);

        drawPidBarChart(canvas, barX, barY, barW, barH, pAbs, iAbs, dAbs);

        // 占比条下方文字
        char ratioBuf[48];
        snprintf(ratioBuf, sizeof(ratioBuf), "P:%.1f I:%.1f D:%.1f",
                 latest.pTermX + latest.pTermY,
                 latest.iTermX + latest.iTermY,
                 latest.dTermX + latest.dTermY);
        cv::putText(canvas, ratioBuf, cv::Point(barX, r4Top + regionH - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.30, cv::Scalar(145, 150, 160), 1);
    }

    // =====================================================================
    // 区域 5: 控制模式 + 实时状态栏 ★核心新增★
    // =====================================================================
    int r5Top = 5 * regionH;
    int r5Bot = height;
    drawRegionLabel(canvas, 5, regionH, "Control Mode & Status");

    if (!filter->pidHistory.empty()) {
        const auto &latest = filter->pidHistory.back();

        // --- 第一行: 控制模式指示器 ---
        int modeY = r5Top + 22;
        const char *modeStr = controlModeToString(latest.controlMode);
        cv::Scalar modeCol = controlModeColor(latest.controlMode);

        // 模式名称（带背景框突出显示）
        int tw = cv::getTextSize(modeStr, cv::FONT_HERSHEY_SIMPLEX, 0.44, 1, nullptr).width;
        cv::rectangle(canvas, cv::Rect(5, modeY - 14, tw + 10, 18),
                      cv::Scalar(35, 36, 43), -1);
        cv::rectangle(canvas, cv::Rect(5, modeY - 14, tw + 10, 18),
                      modeCol, 1);
        cv::putText(canvas, modeStr, cv::Point(10, modeY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.44, modeCol, 1);

        // 射击状态指示器
        int firingX = tw + 20;
        if (latest.isFiring) {
            cv::putText(canvas, "[FIRE]", cv::Point(firingX, modeY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(20, 20, 240), 1);
        } else {
            cv::putText(canvas, "[IDLE]", cv::Point(firingX, modeY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(100, 100, 110), 1);
        }

        // 算法类型
        const char *algoNames[] = {"AdvPID", "StdPID", "Chris"};
        // algorithmType 存储在 PidDebugData 中但此处未直接使用，显示默认值
        cv::putText(canvas, algoNames[0],
                    cv::Point(firingX + 52, modeY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(140, 140, 160), 1);

        // --- 第二行: 关键数值状态栏 ---
        int statY = r5Top + 43;
        char statBuf[160];
        snprintf(statBuf, sizeof(statBuf),
                 "eX:%.1f eY:%.1f | outX:%.1f outY:%.1f | Kp:%.3f Ki:%.4f Kd:%.4f",
                 latest.errorX, latest.errorY,
                 latest.outputX, latest.outputY,
                 latest.currentKp, latest.currentKi, latest.currentKd);
        cv::putText(canvas, statBuf, cv::Point(5, statY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.33, cv::Scalar(185, 190, 200), 1);

        // --- 第三行: 积分数值详情 ---
        int detailY = r5Top + 62;
        char detailBuf[140];
        snprintf(detailBuf, sizeof(detailBuf),
                 "iAbsX:%.1f iAbsY:%.1f | iRatioX:%.0f%% iRatioY:%.0f%%",
                 latest.integralAbsX, latest.integralAbsY,
                 latest.integralRatioX * 100.0f, latest.integralRatioY * 100.0f);
        cv::putText(canvas, detailBuf, cv::Point(5, detailY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.31, cv::Scalar(135, 140, 155), 1);
    }
}

void renderFloatingWindow(yolo_detector_filter *filter)
{
	if (!filter->floatingWindowHandle || filter->floatingWindowFrame.empty()) {
		return;
	}
	InvalidateRect(filter->floatingWindowHandle, NULL, FALSE);
}

void setupPidDataCallback(yolo_detector_filter *filter)
{
	if (!filter || !filter->mouseController) {
		return;
	}
	
	filter->mouseController->setPidDataCallback([filter](const PidDebugData &data) {
		std::lock_guard<std::mutex> lock(filter->pidHistoryMutex);

		PidDebugData point;
		point.errorX = data.errorX;
		point.errorY = data.errorY;
		point.outputX = data.outputX;
		point.outputY = data.outputY;
		point.targetX = data.targetX;
		point.targetY = data.targetY;
		point.targetVelocityX = data.targetVelocityX;
		point.targetVelocityY = data.targetVelocityY;
		point.currentKp = data.currentKp;
		point.currentKi = data.currentKi;
		point.currentKd = data.currentKd;

		// 新增字段
		point.pTermX = data.pTermX;
		point.pTermY = data.pTermY;
		point.iTermX = data.iTermX;
		point.iTermY = data.iTermY;
		point.dTermX = data.dTermX;
		point.dTermY = data.dTermY;
		point.integralAbsX = data.integralAbsX;
		point.integralAbsY = data.integralAbsY;
		point.integralRatioX = data.integralRatioX;
		point.integralRatioY = data.integralRatioY;
		point.controlMode = data.controlMode;
		point.isFiring = data.isFiring;

		point.timestamp = std::chrono::steady_clock::now();
		
		filter->pidHistory.push_back(point);
		
		while (filter->pidHistory.size() > filter->PID_HISTORY_SIZE) {
			filter->pidHistory.pop_front();
		}
	});
}

static yolo_detector_filter *g_pidDebugWindowFilter = nullptr;

static LRESULT CALLBACK PidDebugWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	yolo_detector_filter *filter = g_pidDebugWindowFilter;

	switch (msg) {
	case WM_PAINT: {
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hwnd, &ps);
		
		if (filter) {
			std::lock_guard<std::mutex> lock(filter->pidDebugWindowMutex);
			if (!filter->pidDebugWindowFrame.empty()) {
				BITMAPINFO bmi = {};
				bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bmi.bmiHeader.biWidth = filter->pidDebugWindowFrame.cols;
				bmi.bmiHeader.biHeight = -filter->pidDebugWindowFrame.rows;
				bmi.bmiHeader.biPlanes = 1;
				bmi.bmiHeader.biBitCount = 24;
				bmi.bmiHeader.biCompression = BI_RGB;
				SetDIBitsToDevice(hdc, 0, 0, filter->pidDebugWindowFrame.cols, filter->pidDebugWindowFrame.rows,
					0, 0, 0, filter->pidDebugWindowFrame.rows, filter->pidDebugWindowFrame.data, &bmi, DIB_RGB_COLORS);
			}
		}
		
		EndPaint(hwnd, &ps);
		break;
	}
	case WM_LBUTTONDOWN: {
		if (filter) {
			filter->pidDebugWindowDragging = true;
			POINT pt;
			GetCursorPos(&pt);
			filter->pidDebugWindowDragOffset.x = pt.x - filter->pidDebugWindowX;
			filter->pidDebugWindowDragOffset.y = pt.y - filter->pidDebugWindowY;
			SetCapture(hwnd);
		}
		break;
	}
	case WM_LBUTTONUP: {
		if (filter) {
			filter->pidDebugWindowDragging = false;
			ReleaseCapture();
		}
		break;
	}
	case WM_MOUSEMOVE: {
		if (filter && filter->pidDebugWindowDragging) {
			POINT pt;
			GetCursorPos(&pt);
			SetWindowPos(hwnd, NULL, pt.x - filter->pidDebugWindowDragOffset.x, pt.y - filter->pidDebugWindowDragOffset.y,
				0, 0, SWP_NOSIZE | SWP_NOZORDER);
			filter->pidDebugWindowX = pt.x - filter->pidDebugWindowDragOffset.x;
			filter->pidDebugWindowY = pt.y - filter->pidDebugWindowDragOffset.y;
		}
		break;
	}
	case WM_CLOSE: {
		if (filter) {
			filter->showPidDebugWindow = false;
			destroyPidDebugWindow(filter);
			obs_data_t *settings = obs_source_get_settings(filter->source);
			if (settings) {
				obs_data_set_bool(settings, "show_pid_debug_window", false);
				obs_data_release(settings);
			}
		}
		break;
	}
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

void createPidDebugWindow(yolo_detector_filter *filter)
{
	if (filter->pidDebugWindowHandle) {
		return;
	}

	g_pidDebugWindowFilter = filter;

	WNDCLASS wc = {};
	wc.lpfnWndProc = PidDebugWindowProc;
	wc.hInstance = GetModuleHandle(NULL);
	wc.lpszClassName = L"YOLODetectorPidDebugWindow";
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

	RegisterClass(&wc);

	int x = filter->pidDebugWindowX;
	int y = filter->pidDebugWindowY;
	if (x == 0 && y == 0) {
		x = GetSystemMetrics(SM_CXSCREEN) / 2 - filter->pidDebugWindowWidth / 2;
		y = GetSystemMetrics(SM_CYSCREEN) / 2 - filter->pidDebugWindowHeight / 2 + 100;
	}

	filter->pidDebugWindowHandle = CreateWindowEx(
		WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
		L"YOLODetectorPidDebugWindow",
		L"PID Debug",
		WS_POPUP | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
		x, y,
		filter->pidDebugWindowWidth, filter->pidDebugWindowHeight,
		NULL, NULL, GetModuleHandle(NULL), filter
	);

	filter->pidDebugWindowX = x;
	filter->pidDebugWindowY = y;
	filter->pidDebugWindowDragging = false;

	obs_log(LOG_INFO, "[YOLO Detector] PID debug window created");
}

void destroyPidDebugWindow(yolo_detector_filter *filter)
{
	if (filter->pidDebugWindowHandle) {
		DestroyWindow(filter->pidDebugWindowHandle);
		filter->pidDebugWindowHandle = nullptr;
		g_pidDebugWindowFilter = nullptr;
		obs_log(LOG_INFO, "[YOLO Detector] PID debug window destroyed");
	}
}

void updatePidDebugWindow(yolo_detector_filter *filter)
{
	if (!filter->pidDebugWindowHandle) {
		return;
	}

	{
		std::lock_guard<std::mutex> lock(filter->pidDebugWindowMutex);
		filter->pidDebugWindowFrame.create(filter->pidDebugWindowHeight, filter->pidDebugWindowWidth, CV_8UC3);
		drawPidDebugGraph(filter, filter->pidDebugWindowFrame);
	}

	InvalidateRect(filter->pidDebugWindowHandle, NULL, FALSE);
}

// 线程池工作函数

#endif // _WIN32
