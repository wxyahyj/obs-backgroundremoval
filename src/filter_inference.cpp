#include "yolo_detector_filter.h"
#include "HungarianAlgorithm.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#ifdef HAVE_CUDA
#include <d3d11.h>
#endif
#endif

#include <opencv2/imgproc.hpp>
#include <algorithm>

#include <plugin-support.h>
#include "obs-utils/obs-utils.h"
#include "consts.h"


void inferenceThreadWorker(yolo_detector_filter *filter)
{
	obs_log(LOG_INFO, "[YOLO Detector] Async inference thread started (4-buffer mode)");

	// 提高线程优先级以减少延迟
	#ifdef _WIN32
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	#endif

	int inferenceFrameCounter = 0;
	// 从UI配置读取推理间隔，0表示每帧都推理
	int inferenceInterval = filter->inferenceIntervalFrames <= 0 ? 1 : filter->inferenceIntervalFrames;

	while (filter->inferenceRunning) {
		if (!filter->isInferencing) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}

		// 帧间隔控制：不是每帧都推理
		inferenceFrameCounter++;
		if (inferenceFrameCounter < inferenceInterval) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		inferenceFrameCounter = 0;

		// 无锁获取待推理帧
		int readIdx = -1;
		int startIdx = filter->inputReadIdx.load(std::memory_order_acquire);
		
		for (int i = 0; i < filter->BUFFER_COUNT; i++) {
			int checkIdx = (startIdx + i) % filter->BUFFER_COUNT;
			uint8_t expected = 1;  // 期望状态为"有数据待推理"
			
			if (filter->bufferState[checkIdx].compare_exchange_strong(
				expected, 2, std::memory_order_acq_rel)) {
				readIdx = checkIdx;
				filter->inputReadIdx.store(checkIdx, std::memory_order_release);
				break;
			}
		}
		
		if (readIdx == -1) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}

		// 读取帧数据（克隆以避免数据竞争）
		// 加锁保护 inputFrames 的读取，防止分辨率变化时的竞态条件
		cv::Mat frame;
		int fullWidth, fullHeight;
		int cropX, cropY;
		int cropWidth, cropHeight;
		{
			std::lock_guard<std::mutex> lock(filter->inputFramesMutex);
			frame = filter->inputFrames[readIdx].clone();
			fullWidth = filter->inputFrameWidths[readIdx];
			fullHeight = filter->inputFrameHeights[readIdx];
			cropX = filter->inputCropX[readIdx];
			cropY = filter->inputCropY[readIdx];
			cropWidth = filter->inputCropWidth[readIdx];
			cropHeight = filter->inputCropHeight[readIdx];
		}

		// 标记输入缓冲区为空闲（已读取完毕）
		filter->bufferState[readIdx].store(0, std::memory_order_release);
		
		// 安全检查：确保帧数据有效
		if (frame.empty() || fullWidth <= 0 || fullHeight <= 0) {
			continue;
		}
		
		// 安全检查：确保裁剪区域有效
		if (cropWidth <= 0 || cropHeight <= 0) {
			cropWidth = fullWidth;
			cropHeight = fullHeight;
			cropX = 0;
			cropY = 0;
		}
		
		// 安全检查：确保裁剪区域不超出边界
		if (cropX < 0) cropX = 0;
		if (cropY < 0) cropY = 0;
		if (cropX + cropWidth > fullWidth) cropWidth = fullWidth - cropX;
		if (cropY + cropHeight > fullHeight) cropHeight = fullHeight - cropY;
		if (cropWidth <= 0 || cropHeight <= 0) {
			continue;
		}

		auto startTime = std::chrono::high_resolution_clock::now();
		auto inferenceStartTime = startTime;

		// 如果需要裁切，提取裁切区域
		cv::Mat inferenceFrame;
		if (cropX > 0 || cropY > 0 || cropWidth < fullWidth || cropHeight < fullHeight) {
			inferenceFrame = frame(cv::Rect(cropX, cropY, cropWidth, cropHeight)).clone();
		} else {
			inferenceFrame = frame;
			cropX = 0;
			cropY = 0;
			cropWidth = fullWidth;
			cropHeight = fullHeight;
		}

		// 执行推理
		std::vector<Detection> newDetections;
		{
			std::lock_guard<std::mutex> lock(filter->yoloModelMutex);
			if (filter->yoloModel) {
#ifdef _WIN32
#if defined(HAVE_CUDA) || defined(HAVE_ONNXRUNTIME_DML_EP)
				if (filter->useGpuTextureInference && filter->cachedD3D11Texture &&
				    filter->gpuTextureWidth > 0 && filter->gpuTextureHeight > 0) {
					bool gpuInferenceSuccess = false;
#ifdef HAVE_CUDA
					if (filter->yoloModel->isGpuTextureSupported() && !gpuInferenceSuccess) {
						try {
							newDetections = filter->yoloModel->inferenceFromTexture(
								filter->cachedD3D11Texture,
								filter->gpuTextureWidth,
								filter->gpuTextureHeight,
								fullWidth, fullHeight
							);
							gpuInferenceSuccess = true;
						} catch (const std::exception& e) {
							obs_log(LOG_WARNING, "[YOLO Filter] GPU texture inference failed: %s, falling back to CPU", e.what());
							gpuInferenceSuccess = false;
						} catch (...) {
							obs_log(LOG_WARNING, "[YOLO Filter] GPU texture inference unknown error, falling back to CPU");
							gpuInferenceSuccess = false;
						}
					}
#endif
#ifdef HAVE_ONNXRUNTIME_DML_EP
					if (filter->yoloModel->isDmlTextureSupported() && !gpuInferenceSuccess) {
						try {
							newDetections = filter->yoloModel->inferenceFromTextureDml(
								filter->cachedD3D11Texture,
								filter->gpuTextureWidth,
								filter->gpuTextureHeight,
								fullWidth, fullHeight
							);
							gpuInferenceSuccess = true;
						} catch (const std::exception& e) {
							obs_log(LOG_WARNING, "[YOLO Filter] DML texture inference failed: %s, falling back to CPU", e.what());
							gpuInferenceSuccess = false;
						} catch (...) {
							obs_log(LOG_WARNING, "[YOLO Filter] DML texture inference unknown error, falling back to CPU");
							gpuInferenceSuccess = false;
						}
					}
#endif
					if (!gpuInferenceSuccess) {
						newDetections = filter->yoloModel->inference(inferenceFrame);
					}
				} else {
					newDetections = filter->yoloModel->inference(inferenceFrame);
				}
#else
				newDetections = filter->yoloModel->inference(inferenceFrame);
#endif
#else
				newDetections = filter->yoloModel->inference(inferenceFrame);
#endif
			}
		}

		// 记录推理时间
		auto endTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
			endTime - inferenceStartTime).count();

		// 坐标转换（如果有裁切区域且有检测结果）
		if (!newDetections.empty() && cropWidth > 0 && cropHeight > 0 && 
			(cropX > 0 || cropY > 0 || cropWidth < fullWidth || cropHeight < fullHeight)) {
			for (auto& det : newDetections) {
				float pixelX = det.x * cropWidth + cropX;
				float pixelY = det.y * cropHeight + cropY;
				float pixelW = det.width * cropWidth;
				float pixelH = det.height * cropHeight;
				float pixelCenterX = det.centerX * cropWidth + cropX;
				float pixelCenterY = det.centerY * cropHeight + cropY;
				
				det.x = pixelX / fullWidth;
				det.y = pixelY / fullHeight;
				det.width = pixelW / fullWidth;
				det.height = pixelH / fullHeight;
				det.centerX = pixelCenterX / fullWidth;
				det.centerY = pixelCenterY / fullHeight;
			}
		}

		// 处理目标追踪（如果有检测结果或已有追踪目标）
		if (!newDetections.empty() || !filter->trackedTargets.empty()) {
			std::vector<Detection> trackedDetections;
			{
				std::lock_guard<std::mutex> trackLock(filter->trackedTargetsMutex);
				
				// 使用 KalmanFilter 追踪
				if (filter->useKalmanTracker) {
					std::vector<KalmanDetail::DetectionObject> kalmanDets;
					for (const auto& det : newDetections) {
						KalmanDetail::DetectionObject kdet;
						kdet.bbox.x = det.x;
						kdet.bbox.y = det.y;
						kdet.bbox.width = det.width;
						kdet.bbox.height = det.height;
						kdet.label = det.classId;
						kdet.prob = det.confidence;
						kdet.track_id = -1;
						kalmanDets.push_back(kdet);
					}
					
					// 获取预测位置（在 predict 内部已经调用了每个 track 的 predict）
					std::vector<KalmanDetail::DetectionObject> kalmanTracked = filter->kalmanTracker.predict(kalmanDets);
					
					// 保存预测位置用于渲染
					{
						std::lock_guard<std::mutex> predLock(filter->kalmanPredictionsMutex);
						filter->kalmanPredictions.clear();
						std::vector<KalmanDetail::DetectionObject> predictions = filter->kalmanTracker.getPredictions();
						for (const auto& pred : predictions) {
							yolo_detector_filter::KalmanPrediction kp;
							kp.x = pred.bbox.x;
							kp.y = pred.bbox.y;
							kp.width = pred.bbox.width;
							kp.height = pred.bbox.height;
							kp.trackId = pred.track_id;
							filter->kalmanPredictions.push_back(kp);
						}
					}
					
					// 保存多帧预测轨迹用于渲染
					{
						std::lock_guard<std::mutex> trajLock(filter->kalmanTrajectoriesMutex);
						filter->kalmanTrajectories = filter->kalmanTracker.getMultiFramePredictions(filter->kalmanPredictionFrames);
					}
					
					for (const auto& kt : kalmanTracked) {
						Detection det;
						det.x = kt.bbox.x;
						det.y = kt.bbox.y;
						det.width = kt.bbox.width;
						det.height = kt.bbox.height;
						det.centerX = det.x + det.width / 2.0f;
						det.centerY = det.y + det.height / 2.0f;
						det.classId = kt.label;
						det.confidence = kt.prob;
						det.trackId = kt.track_id;
						det.lostFrames = 0;
						trackedDetections.push_back(det);
					}
					
					filter->trackedTargets = trackedDetections;
				} else {
					// 原有追踪逻辑
					std::vector<Detection>& trackedTargets = filter->trackedTargets;
				
				if (trackedTargets.empty()) {
					for (auto& det : newDetections) {
						det.trackId = filter->nextTrackId++;
						det.lostFrames = 0;
						trackedDetections.push_back(det);
					}
				} else {
					int n = static_cast<int>(newDetections.size());
					int m = static_cast<int>(trackedTargets.size());
					
					std::vector<std::vector<float>> costMatrix(n, std::vector<float>(m, 1.0f));
					
					for (int i = 0; i < n; ++i) {
						cv::Rect2f detBox(
							newDetections[i].x,
							newDetections[i].y,
							newDetections[i].width,
							newDetections[i].height
						);
						cv::Point2f detCenter(newDetections[i].centerX, newDetections[i].centerY);
						
						for (int j = 0; j < m; ++j) {
							cv::Rect2f trackBox(
								trackedTargets[j].x,
								trackedTargets[j].y,
								trackedTargets[j].width,
								trackedTargets[j].height
							);
							cv::Point2f trackCenter(trackedTargets[j].centerX, trackedTargets[j].centerY);
							
							costMatrix[i][j] = HungarianAlgorithm::calculateFusedDistance(
								detBox, trackBox, detCenter, trackCenter,
								filter->trackingWeightIou,
								filter->trackingWeightCenter,
								filter->trackingWeightAspect,
								filter->trackingWeightArea);
						}
					}
					
					std::vector<int> assignment = HungarianAlgorithm::solve(costMatrix);
					
					std::vector<bool> detectionMatched(n, false);
					std::vector<bool> trackMatched(m, false);
					
					for (int i = 0; i < n; ++i) {
						int j = assignment[i];
						if (j >= 0 && j < m && costMatrix[i][j] < (1.0f - filter->iouThreshold)) {
							newDetections[i].trackId = trackedTargets[j].trackId;
							newDetections[i].lostFrames = 0;
							trackedDetections.push_back(newDetections[i]);
							detectionMatched[i] = true;
							trackMatched[j] = true;
						}
					}
					
					for (int i = 0; i < n; ++i) {
						if (!detectionMatched[i]) {
							newDetections[i].trackId = filter->nextTrackId++;
							newDetections[i].lostFrames = 0;
							trackedDetections.push_back(newDetections[i]);
						}
					}
					
					for (int j = 0; j < m; ++j) {
						if (!trackMatched[j]) {
							trackedTargets[j].lostFrames++;
							if (trackedTargets[j].lostFrames <= filter->maxLostFrames) {
								trackedDetections.push_back(trackedTargets[j]);
							} else {
								// 目标丢失超过阈值，添加到重识别缓冲区
								std::lock_guard<std::mutex> lostLock(filter->lostTargetsMutex);
								LostTarget lost;
								lost.trackId = trackedTargets[j].trackId;
								lost.x = trackedTargets[j].x;
								lost.y = trackedTargets[j].y;
								lost.width = trackedTargets[j].width;
								lost.height = trackedTargets[j].height;
								lost.centerX = trackedTargets[j].centerX;
								lost.centerY = trackedTargets[j].centerY;
								lost.lostFrames = 0;
								lost.lostTime = std::chrono::steady_clock::now();
								
								// 检查是否已存在相同trackId，更新而非添加
								bool found = false;
								for (auto& existing : filter->lostTargets) {
									if (existing.trackId == lost.trackId) {
										existing = lost;
										found = true;
										break;
									}
								}
								if (!found) {
									filter->lostTargets.push_back(lost);
								}
							}
						}
					}
					
					// 尝试重识别丢失目标
					{
						std::lock_guard<std::mutex> lostLock(filter->lostTargetsMutex);
						auto now = std::chrono::steady_clock::now();
						for (auto it = filter->lostTargets.begin(); it != filter->lostTargets.end(); ) {
							// 检查重识别时间是否超时
							auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->lostTime).count();
							if (elapsed > filter->maxReidentifyFrames * 33) {  // 约33ms每帧
								it = filter->lostTargets.erase(it);
								continue;
							}
							
							// 尝试与未匹配的检测进行重识别
							for (int i = 0; i < n; ++i) {
								if (detectionMatched[i]) continue;
								
								float dx = newDetections[i].centerX - it->centerX;
								float dy = newDetections[i].centerY - it->centerY;
								float centerDist = std::sqrt(dx * dx + dy * dy);
								
								// 中心点距离很近，认为是同一目标
								if (centerDist < filter->reidentifyCenterThreshold) {
									newDetections[i].trackId = it->trackId;
									newDetections[i].lostFrames = 0;
									trackedDetections.push_back(newDetections[i]);
									detectionMatched[i] = true;
									it = filter->lostTargets.erase(it);
									break;
								}
							}
							
							if (it != filter->lostTargets.end()) {
								++it;
							}
						}
					}
				}
				}  // end of else (原有追踪逻辑)
				
				filter->trackedTargets = std::move(trackedDetections);
			}
		}

		// 写入共享指针（替代四缓冲区）
		{
			auto result = std::make_shared<yolo_detector_filter::InferenceResult>();
			{
				std::lock_guard<std::mutex> trackLock(filter->trackedTargetsMutex);
				result->detections = filter->trackedTargets;
				result->trackedTargets = filter->trackedTargets;
			}
			result->frameWidth = fullWidth;
			result->frameHeight = fullHeight;
			result->cropX = cropX;
			result->cropY = cropY;
			result->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now().time_since_epoch()).count();
			
			std::lock_guard<std::mutex> resultLock(filter->inferenceResultMutex_);
			filter->inferenceResultPtr_ = result;
		}

		// 更新统计信息
		filter->inferenceCount++;
		filter->avgInferenceTimeMs = (filter->avgInferenceTimeMs * (filter->inferenceCount - 1) + duration) / filter->inferenceCount;
		filter->framesInferred.fetch_add(1, std::memory_order_relaxed);

		// 更新最后有结果的时刻
		filter->lastResultTimestamp.store(std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now().time_since_epoch()).count(), std::memory_order_relaxed);

		// 导出坐标（如果有检测结果）
		if (filter->exportCoordinates && !newDetections.empty()) {
			exportCoordinatesToFile(filter, fullWidth, fullHeight);
		}
	}

	obs_log(LOG_INFO, "[YOLO Detector] Async inference thread stopped");
}

void threadPoolWorker(yolo_detector_filter *filter)
{
	while (filter->threadPoolRunning) {
		std::function<void()> task;
		{
			std::unique_lock<std::mutex> lock(filter->taskQueueMutex);
			filter->taskCondition.wait(lock, [filter] { return !filter->threadPoolRunning || !filter->taskQueue.empty(); });
			if (!filter->threadPoolRunning && filter->taskQueue.empty()) {
				return;
			}
			task = std::move(filter->taskQueue.front());
			filter->taskQueue.pop();
		}
		task();
	}
}

// 提交任务到线程池
template<typename F>
void submitTask(yolo_detector_filter *filter, F &&task)
{
	std::unique_lock<std::mutex> lock(filter->taskQueueMutex);
	filter->taskQueue.push(std::function<void()>(std::forward<F>(task)));
	lock.unlock();
	filter->taskCondition.notify_one();
}

// 从内存池中获取图像缓冲区
cv::Mat getImageBuffer(yolo_detector_filter *filter, int rows, int cols, int type)
{
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	yolo_detector_filter::ImageBufferKey key{rows, cols, type};
	auto it = filter->imageBufferPool.find(key);
	
	if (it != filter->imageBufferPool.end() && !it->second.empty()) {
		cv::Mat buffer = std::move(it->second.back());
		it->second.pop_back();
		return buffer;
	}
	
	// 如果没有合适的缓冲区，创建一个新的
	return cv::Mat(rows, cols, type);
}

// 释放图像缓冲区到内存池
void releaseImageBuffer(yolo_detector_filter *filter, cv::Mat &&buffer)
{
	if (buffer.empty()) {
		return;
	}
	
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	yolo_detector_filter::ImageBufferKey key{buffer.rows, buffer.cols, buffer.type()};
	auto it = filter->imageBufferPool.find(key);
	
	if (it != filter->imageBufferPool.end()) {
		if (it->second.size() < filter->MAX_BUFFER_POOL_SIZE) {
			it->second.push_back(std::move(buffer));
		}
	} else {
		std::vector<cv::Mat> buffers;
		buffers.reserve(5);
		buffers.push_back(std::move(buffer));
		filter->imageBufferPool[key] = std::move(buffers);
	}
}

// 从内存池中获取检测结果缓冲区
std::vector<Detection> getDetectionBuffer(yolo_detector_filter *filter)
{
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	if (!filter->detectionBufferPool.empty()) {
		std::vector<Detection> buffer = std::move(filter->detectionBufferPool.back());
		filter->detectionBufferPool.pop_back();
		buffer.clear(); // 清空缓冲区内容
		return buffer;
	}
	
	// 如果没有可用的缓冲区，创建一个新的
	return std::vector<Detection>();
}

// 释放检测结果缓冲区到内存池
void releaseDetectionBuffer(yolo_detector_filter *filter, std::vector<Detection> &&buffer)
{
	std::lock_guard<std::mutex> lock(filter->bufferPoolMutex);
	
	// 确保内存池不超过最大大小
	if (filter->detectionBufferPool.size() < filter->MAX_BUFFER_POOL_SIZE) {
		// 清空缓冲区内容，保留容量
		buffer.clear();
		filter->detectionBufferPool.push_back(std::move(buffer));
	}
}

