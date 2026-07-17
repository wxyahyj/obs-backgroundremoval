#ifndef FILTER_FLOATING_H
#define FILTER_FLOATING_H

struct yolo_detector_filter;

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <opencv2/core.hpp>

LRESULT CALLBACK FloatingWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
void createFloatingWindow(yolo_detector_filter *filter);
void destroyFloatingWindow(yolo_detector_filter *filter);
void updateFloatingWindowFrame(yolo_detector_filter *filter, const cv::Mat &frame);
void renderFloatingWindow(yolo_detector_filter *filter);
void setupPidDataCallback(yolo_detector_filter *filter);
void createPidDebugWindow(yolo_detector_filter *filter);
void destroyPidDebugWindow(yolo_detector_filter *filter);
void updatePidDebugWindow(yolo_detector_filter *filter);
#endif

#endif
