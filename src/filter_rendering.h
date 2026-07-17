#ifndef FILTER_RENDERING_H
#define FILTER_RENDERING_H

#include <cstdint>

struct yolo_detector_filter;

void renderDetectionBoxes(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderKalmanPredictions(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderKalmanTrajectories(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderFOV(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void renderRegion(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);
void exportCoordinatesToFile(yolo_detector_filter *filter, uint32_t frameWidth, uint32_t frameHeight);

#endif
