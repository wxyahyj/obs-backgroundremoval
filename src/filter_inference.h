#ifndef FILTER_INFERENCE_H
#define FILTER_INFERENCE_H

struct yolo_detector_filter;

void inferenceThreadWorker(yolo_detector_filter *filter);
void threadPoolWorker(yolo_detector_filter *filter);

#endif
