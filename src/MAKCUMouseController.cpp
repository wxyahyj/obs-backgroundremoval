#ifdef _WIN32

#include "MAKCUMouseController.hpp"
#include "DerivativePredictor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <obs-module.h>
#include "plugin-support.h"

MAKCUMouseController::MAKCUMouseController()
    : cachedScreenWidth(0)
    , cachedScreenHeight(0)
    , hSerial(INVALID_HANDLE_VALUE)
    , serialConnected(false)
    , portName("COM5")
    , baudRate(4000000)
    , isMoving(false)
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , filteredDeltaErrorX(0.0f)
    , filteredDeltaErrorY(0.0f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
    , integralX(0.0f)
    , integralY(0.0f)
    , stdIntegralX(0.0f)
    , stdIntegralY(0.0f)
    , stdIntegralGainX(0.0f)
    , stdIntegralGainY(0.0f)
    , stdLastErrorX(0.0f)
    , stdLastErrorY(0.0f)
    , lastRecoilTime(std::chrono::steady_clock::now())
    , isFiring(false)
    , lastTickTime(std::chrono::steady_clock::now())
    , deltaTime(0.016f)
    , hotkeyPressStartTime(std::chrono::steady_clock::now())
    , yUnlockActive(false)
    , lastAutoTriggerTime(std::chrono::steady_clock::now())
    , autoTriggerFireStartTime(std::chrono::steady_clock::now())
    , autoTriggerDelayStartTime(std::chrono::steady_clock::now())
    , autoTriggerHolding(false)
    , autoTriggerWaitingForDelay(false)
    , currentFireDuration(50)
    , randomGenerator(std::random_device{}())
    , currentTargetTrackId(-1)
    , targetLockStartTime(std::chrono::steady_clock::now())
    , currentTargetDistance(0.0f)
    , kalmanFilterInitialized(false)
{
    cachedScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    cachedScreenHeight = GetSystemMetrics(SM_CYSCREEN);
    connectSerial();

    if (serialConnected) {
        move(0, 0);
    }
}

MAKCUMouseController::MAKCUMouseController(const std::string& port, int baud)
    : cachedScreenWidth(0)
    , cachedScreenHeight(0)
    , hSerial(INVALID_HANDLE_VALUE)
    , serialConnected(false)
    , portName(port)
    , baudRate(baud)
    , isMoving(false)
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , filteredDeltaErrorX(0.0f)
    , filteredDeltaErrorY(0.0f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
    , integralX(0.0f)
    , integralY(0.0f)
    , lastRecoilTime(std::chrono::steady_clock::now())
    , isFiring(false)
    , lastTickTime(std::chrono::steady_clock::now())
    , deltaTime(0.016f)
    , hotkeyPressStartTime(std::chrono::steady_clock::now())
    , yUnlockActive(false)
    , lastAutoTriggerTime(std::chrono::steady_clock::now())
    , autoTriggerFireStartTime(std::chrono::steady_clock::now())
    , autoTriggerDelayStartTime(std::chrono::steady_clock::now())
    , autoTriggerHolding(false)
    , autoTriggerWaitingForDelay(false)
    , currentFireDuration(50)
    , randomGenerator(std::random_device{}())
    , currentTargetTrackId(-1)
    , targetLockStartTime(std::chrono::steady_clock::now())
    , currentTargetDistance(0.0f)
{
    cachedScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    cachedScreenHeight = GetSystemMetrics(SM_CYSCREEN);
    connectSerial();
    
    if (serialConnected) {
        move(0, 0);
    }
}

MAKCUMouseController::~MAKCUMouseController()
{
    disconnectSerial();
}

bool MAKCUMouseController::connectSerial()
{
    if (serialConnected) {
        return true;
    }

    std::wstring wPortName(portName.begin(), portName.end());
    hSerial = CreateFileW(
        wPortName.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hSerial == INVALID_HANDLE_VALUE) {
        return false;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

    if (!GetCommState(hSerial, &dcbSerialParams)) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    dcbSerialParams.BaudRate = baudRate;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;

    if (!SetCommState(hSerial, &dcbSerialParams)) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;

    if (!SetCommTimeouts(hSerial, &timeouts)) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    serialConnected = true;
    return true;
}

void MAKCUMouseController::disconnectSerial()
{
    if (serialConnected && hSerial != INVALID_HANDLE_VALUE) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        serialConnected = false;
    }
}

bool MAKCUMouseController::sendSerialCommand(const std::string& command)
{
    if (!serialConnected || hSerial == INVALID_HANDLE_VALUE) {
        return false;
    }

    DWORD bytesWritten;
    std::string cmd = command + "\r\n";
    bool success = WriteFile(hSerial, cmd.c_str(), static_cast<DWORD>(cmd.length()), &bytesWritten, NULL);
    if (success && bytesWritten == static_cast<DWORD>(cmd.length())) {
        // 读取设备响应（可选）
        char buffer[256];
        DWORD bytesRead;
        DWORD events;
        if (WaitCommEvent(hSerial, &events, NULL)) {
            if (events & EV_RXCHAR) {
                ReadFile(hSerial, buffer, sizeof(buffer) - 1, &bytesRead, NULL);
            }
        }
        
        return true;
    } else {
        return false;
    }
}

void MAKCUMouseController::move(int dx, int dy)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.move(%d,%d)", dx, dy);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::moveTo(int x, int y)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.moveTo(%d,%d)", x, y);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::click(bool left)
{
    sendSerialCommand(left ? "km.left(1)" : "km.left(0)");
}

void MAKCUMouseController::clickDown()
{
    obs_log(LOG_INFO, "[MAKCU] clickDown: sending km.left(1)");
    bool success = sendSerialCommand("km.left(1)");
    obs_log(LOG_INFO, "[MAKCU] clickDown result: %d, serialConnected=%d", success, serialConnected);
}

void MAKCUMouseController::clickUp()
{
    obs_log(LOG_INFO, "[MAKCU] clickUp: sending km.left(0)");
    bool success = sendSerialCommand("km.left(0)");
    obs_log(LOG_INFO, "[MAKCU] clickUp result: %d, serialConnected=%d", success, serialConnected);
}

void MAKCUMouseController::wheel(int delta)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.wheel(%d)", delta);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::updateConfig(const MouseControllerConfig& newConfig)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    bool configChanged = (config.enableMouseControl != newConfig.enableMouseControl ||
                          config.autoTriggerEnabled != newConfig.autoTriggerEnabled ||
                          config.autoTriggerFireDuration != newConfig.autoTriggerFireDuration ||
                          config.autoTriggerInterval != newConfig.autoTriggerInterval ||
                          config.targetSwitchDelayMs != newConfig.targetSwitchDelayMs ||
                          config.targetSwitchTolerance != newConfig.targetSwitchTolerance);
    
    bool portChanged = (newConfig.makcuPort != portName);
    bool baudChanged = (newConfig.makcuBaudRate != baudRate);
    
    config = newConfig;
    
    if (configChanged) {
        obs_log(LOG_INFO, "[MAKCU] Config updated: enableMouseControl=%d, autoTriggerEnabled=%d, fireDuration=%dms, interval=%dms, targetSwitchDelay=%dms, targetSwitchTolerance=%.2f",
                newConfig.enableMouseControl, newConfig.autoTriggerEnabled, newConfig.autoTriggerFireDuration, newConfig.autoTriggerInterval,
                newConfig.targetSwitchDelayMs, newConfig.targetSwitchTolerance);
    }
    
    if (portChanged || baudChanged) {
        portName = newConfig.makcuPort;
        baudRate = newConfig.makcuBaudRate;
        
        disconnectSerial();
        
        connectSerial();
    }
}

void MAKCUMouseController::setDetections(const std::vector<Detection>& detections)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
}

void MAKCUMouseController::setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
    config.inferenceFrameWidth = frameWidth;
    config.inferenceFrameHeight = frameHeight;
    config.cropOffsetX = cropX;
    config.cropOffsetY = cropY;
}

void MAKCUMouseController::tick()
{
    std::lock_guard<std::mutex> lock(mutex);

    if (!config.enableMouseControl) {
        if (autoTriggerHolding) {
            releaseAutoTrigger();
        }
        autoTriggerWaitingForDelay = false;
        isMoving = false;
        return;
    }

    bool hotkeyPressed = (GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000) != 0;

    // 判断是否应该瞄准
    bool shouldAim = config.continuousAimEnabled || hotkeyPressed;

    // 如果不应该瞄准，停止所有操作
    if (!shouldAim) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        yUnlockActive = false;
        releaseAutoTrigger();
        return;
    }

    // 更新热键状态和Y轴解锁（仅在非持续自瞄模式下）
    if (!config.continuousAimEnabled) {
        static bool wasHotkeyPressed = false;
        if (!wasHotkeyPressed && hotkeyPressed) {
            hotkeyPressStartTime = std::chrono::steady_clock::now();
            yUnlockActive = false;
        }
        wasHotkeyPressed = hotkeyPressed;

        if (config.yUnlockEnabled) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - hotkeyPressStartTime).count();
            if (elapsed >= config.yUnlockDelayMs) {
                yUnlockActive = true;
            }
        } else {
            yUnlockActive = false;
        }
    } else {
        // 持续自瞄模式下，禁用Y轴解锁
        yUnlockActive = false;
    }

    // 计算时间步长
    auto now = std::chrono::steady_clock::now();
    deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTickTime).count() / 1000.0f;
    deltaTime = std::max(0.001f, std::min(deltaTime, 0.05f));
    lastTickTime = now;

    // 更新目标跟踪器
    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth :
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight :
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    targetTracker.update(currentDetections, deltaTime, frameWidth, frameHeight);

    // 使用增强的目标选择
    Detection* target = selectTarget();
    if (!target) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        if (autoTriggerHolding) {
            auto now = std::chrono::steady_clock::now();
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        }
        return;
    }

    float fovCenterX = frameWidth / 2.0f;
    float fovCenterY = frameHeight / 2.0f;

    float targetPixelX = target->centerX * frameWidth;
    float targetPixelY = target->centerY * frameHeight - config.targetYOffset;

    float errorX = targetPixelX - fovCenterX + config.screenOffsetX;
    float errorY = targetPixelY - fovCenterY + config.screenOffsetY;

    float distanceSquared = errorX * errorX + errorY * errorY;
    float deadZoneSquared = config.deadZonePixels * config.deadZonePixels;
    
    if (distanceSquared < deadZoneSquared) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        if (autoTriggerHolding) {
            auto now = std::chrono::steady_clock::now();
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        }
        return;
    }

    float distance = std::sqrt(distanceSquared);

    if (config.autoTriggerEnabled) {
        auto now = std::chrono::steady_clock::now();

        if (autoTriggerHolding) {
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        } else {
            if (distance < config.autoTriggerRadius) {
                if (!autoTriggerWaitingForDelay) {
                    autoTriggerWaitingForDelay = true;
                    autoTriggerDelayStartTime = now;
                }
                
                auto delayElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerDelayStartTime).count();
                int totalDelay = config.autoTriggerFireDelay + getRandomDelay();
                
                if (delayElapsed >= totalDelay) {
                    auto cooldownElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAutoTriggerTime).count();
                    if (cooldownElapsed >= config.autoTriggerInterval) {
                        performAutoClick();
                    }
                }
            } else {
                autoTriggerWaitingForDelay = false;
            }
        }
    }

    isMoving = true;

    float moveX, moveY;

    if (config.algorithmType == AlgorithmType::StandardPID) {
        // 标准PID算法
        moveX = calculateStandardPID(errorX, stdIntegralX, stdIntegralGainX, stdLastErrorX, deltaTime);
        moveY = calculateStandardPID(errorY, stdIntegralY, stdIntegralGainY, stdLastErrorY, deltaTime);
        
        // 详细日志输出
        static int stdLogCounter = 0;
        if (++stdLogCounter >= 30) {
            stdLogCounter = 0;
            blog(LOG_INFO, "[MAKCU标准PID] errorX=%.1f errorY=%.1f | moveX=%.1f moveY=%.1f | stdKp=%.2f stdKd=%.3f",
                 errorX, errorY, moveX, moveY, config.stdKp, config.stdKd);
        }
        
        // 重置高级PID状态变量，避免算法切换时状态不一致
        pidPreviousErrorX = 0.0f;
        pidPreviousErrorY = 0.0f;
        previousErrorX = 0.0f;
        previousErrorY = 0.0f;
        filteredDeltaErrorX = 0.0f;
        filteredDeltaErrorY = 0.0f;
        integralX = 0.0f;
        integralY = 0.0f;
    } else {
        // 高级PID算法（默认）
        float predictedX = 0.0f, predictedY = 0.0f;

        if (config.useKalmanFilter) {
            // 使用卡尔曼滤波器
            if (!kalmanFilterInitialized) {
                kalmanFilter.init(targetPixelX, targetPixelY);
                kalmanFilter.setProcessNoise(config.kalmanProcessNoise);
                kalmanFilter.setMeasurementNoise(config.kalmanMeasurementNoise);
                kalmanFilter.setConfidenceScale(config.kalmanConfidenceScale);
                kalmanFilterInitialized = true;
            }

            // 更新卡尔曼滤波器参数
            kalmanFilter.setProcessNoise(config.kalmanProcessNoise);
            kalmanFilter.setMeasurementNoise(config.kalmanMeasurementNoise);
            kalmanFilter.setConfidenceScale(config.kalmanConfidenceScale);

            // 预测步骤
            kalmanFilter.predict(deltaTime);

            // 更新步骤
            kalmanFilter.update(targetPixelX, targetPixelY, target->confidence);

            // 获取预测偏移量（直接返回相对偏移，可用于PID误差融合）
            kalmanFilter.getPredictionOffset(deltaTime, targetPixelX, targetPixelY, predictedX, predictedY);
        } else {
            // 使用原有的DerivativePredictor
            predictor.update(errorX, errorY, deltaTime);
            predictor.predict(deltaTime, predictedX, predictedY);
        }

        // 误差融合
        float fusedErrorX = errorX + config.predictionWeightX * predictedX;
        float fusedErrorY = errorY + config.predictionWeightY * predictedY;

        // 计算动态P增益，应用P-Gain Ramp
        float dynamicP = calculateDynamicP(distance) * getCurrentPGain();

        // 计算微分项
        float deltaErrorX = fusedErrorX - pidPreviousErrorX;
        float deltaErrorY = fusedErrorY - pidPreviousErrorY;

        // 微分滤波
        float alpha = config.derivativeFilterAlpha;
        filteredDeltaErrorX = alpha * deltaErrorX + (1.0f - alpha) * filteredDeltaErrorX;
        filteredDeltaErrorY = alpha * deltaErrorY + (1.0f - alpha) * filteredDeltaErrorY;

        // 积分项
        float integralTermX = calculateIntegral(fusedErrorX, integralX, deltaTime);
        float integralTermY = calculateIntegral(fusedErrorY, integralY, deltaTime);

        // PID输出（直接使用pidD，移除了自适应D）
        float pidOutputX = dynamicP * fusedErrorX + config.pidD * filteredDeltaErrorX + integralTermX;
        float pidOutputY = dynamicP * fusedErrorY + config.pidD * filteredDeltaErrorY + integralTermY;

        float baselineX = fusedErrorX * config.baselineCompensation;
        float baselineY = fusedErrorY * config.baselineCompensation;

        float moveX = pidOutputX + baselineX;
        float moveY = pidOutputY + baselineY;

        // 详细日志输出
        static int logCounter = 0;
        if (++logCounter >= 30) {  // 每30帧输出一次
            logCounter = 0;
            blog(LOG_INFO, "[MAKCU高级PID] errorX=%.1f errorY=%.1f | fusedX=%.1f fusedY=%.1f | dynamicP=%.3f",
                 errorX, errorY, fusedErrorX, fusedErrorY, dynamicP);
            blog(LOG_INFO, "[MAKCU高级PID] deltaErrX=%.1f deltaErrY=%.1f | filteredDeltaX=%.1f filteredDeltaY=%.1f",
                 deltaErrorX, deltaErrorY, filteredDeltaErrorX, filteredDeltaErrorY);
            blog(LOG_INFO, "[MAKCU高级PID] integralX=%.1f integralY=%.1f | pidOutX=%.1f pidOutY=%.1f | moveX=%.1f moveY=%.1f",
                 integralTermX, integralTermY, pidOutputX, pidOutputY, moveX, moveY);
        }

        // 更新高级PID状态
        pidPreviousErrorX = fusedErrorX;
        pidPreviousErrorY = fusedErrorY;
        previousErrorX = errorX;
        previousErrorY = errorY;
        
        // 重置标准PID状态变量，避免算法切换时状态不一致
        stdIntegralX = 0.0f;
        stdIntegralY = 0.0f;
        stdIntegralGainX = 0.0f;
        stdIntegralGainY = 0.0f;
        stdLastErrorX = errorX;
        stdLastErrorY = errorY;
    }
    
    // 检测射击状态（鼠标左键按下）
    bool isFiring = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    
    // 按下左键时降低Y轴PID增益，避免与压枪对抗（压枪功能独立，不需要自瞄触发）
    if (isFiring && config.autoRecoilControlEnabled) {
        moveY *= config.recoilPidGainScale;  // 使用可配置的增益系数
    }
    
    // 限制最大移动量
    float moveDistSquared = moveX * moveX + moveY * moveY;
    float maxMoveSquared = config.maxPixelMove * config.maxPixelMove;
    if (moveDistSquared > maxMoveSquared && moveDistSquared > 0.0f) {
        float scale = config.maxPixelMove / std::sqrt(moveDistSquared);
        moveX *= scale;
        moveY *= scale;
    }
    
    if (yUnlockActive) {
        moveY = 0.0f;
    }

    // 自动压枪逻辑：独立功能，只要开启压枪且按下左键就压枪，不需要目标识别
    if (config.autoRecoilControlEnabled && isFiring) {
        // 计算每帧应该压枪的量（基于压枪速度和强度）
        // recoilSpeed是毫秒间隔，recoilStrength是该间隔内的压枪量
        float recoilPerMs = config.recoilStrength / static_cast<float>(config.recoilSpeed);
        float recoilThisFrame = recoilPerMs * deltaTime * 1000.0f;  // deltaTime转毫秒
        
        // 累积到moveY
        moveY += recoilThisFrame;
    }

    // 平滑处理
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;

    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;

    if (serialConnected) {
        move(static_cast<int>(finalMoveX), static_cast<int>(finalMoveY));
    } else {
        connectSerial();
    }
}

Detection* MAKCUMouseController::selectTarget()
{
    if (currentDetections.empty()) {
        currentTargetTrackId = -1;
        currentTargetDistance = 0.0f;
        targetTracker.unlock();
        return nullptr;
    }

    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth :
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight :
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    int fovCenterX = frameWidth / 2;
    int fovCenterY = frameHeight / 2;
    float fovRadius = static_cast<float>(config.fovRadiusPixels);

    // 使用增强的目标跟踪器获取最佳目标
    TrackedTarget* trackedTarget = targetTracker.getBestTarget(frameWidth, frameHeight, fovCenterX, fovCenterY, fovRadius);

    if (!trackedTarget) {
        currentTargetTrackId = -1;
        currentTargetDistance = 0.0f;
        return nullptr;
    }

    // 更新当前跟踪ID（使用persistentId）
    currentTargetTrackId = trackedTarget->persistentId;

    // 在原始检测中查找对应的Detection
    for (auto& det : currentDetections) {
        // 使用IOU匹配找到对应的检测
        if (trackedTarget->getIOU(det) > 0.5f) {
            // 计算距离
            float pixelX = det.centerX * frameWidth;
            float pixelY = det.centerY * frameHeight;
            float dx = pixelX - fovCenterX;
            float dy = pixelY - fovCenterY;
            currentTargetDistance = std::sqrt(dx * dx + dy * dy);
            return &det;
        }
    }

    // 如果找不到精确匹配，返回最接近的检测
    float minDistance = std::numeric_limits<float>::max();
    Detection* closestDet = nullptr;

    for (auto& det : currentDetections) {
        float dist = trackedTarget->getDistanceToDetection(det, frameWidth, frameHeight);
        if (dist < minDistance) {
            minDistance = dist;
            closestDet = &det;
        }
    }

    if (closestDet) {
        float pixelX = closestDet->centerX * frameWidth;
        float pixelY = closestDet->centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        currentTargetDistance = std::sqrt(dx * dx + dy * dy);
    }

    return closestDet;
}

POINT MAKCUMouseController::convertToScreenCoordinates(const Detection& det)
{
    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth : 
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight : 
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    float screenPixelX = det.centerX * frameWidth + config.screenOffsetX;
    float screenPixelY = det.centerY * frameHeight - config.targetYOffset + config.screenOffsetY;

    static bool loggedOnce = false;
    if (!loggedOnce) {
        obs_log(LOG_INFO, "[MAKCU] 坐标转换调试信息:");
        obs_log(LOG_INFO, "[MAKCU]   屏幕尺寸: %dx%d", cachedScreenWidth, cachedScreenHeight);
        obs_log(LOG_INFO, "[MAKCU]   推理帧尺寸: %dx%d", frameWidth, frameHeight);
        obs_log(LOG_INFO, "[MAKCU]   检测中心(归一化): %.4f, %.4f", det.centerX, det.centerY);
        obs_log(LOG_INFO, "[MAKCU]   屏幕偏移: %d, %d", config.screenOffsetX, config.screenOffsetY);
        obs_log(LOG_INFO, "[MAKCU]   最终屏幕坐标: %.1f, %.1f", screenPixelX, screenPixelY);
        loggedOnce = true;
    }

    POINT result;
    result.x = static_cast<LONG>(screenPixelX);
    result.y = static_cast<LONG>(screenPixelY);

    LONG maxX = static_cast<LONG>(cachedScreenWidth - 1);
    LONG maxY = static_cast<LONG>(cachedScreenHeight - 1);
    
    result.x = std::max(0L, std::min(result.x, maxX));
    result.y = std::max(0L, std::min(result.y, maxY));

    return result;
}

float MAKCUMouseController::calculateDynamicP(float distance)
{
    float normalizedDistance = distance / static_cast<float>(config.fovRadiusPixels);
    normalizedDistance = std::max(0.0f, std::min(1.0f, normalizedDistance));
    float distancePower = std::pow(normalizedDistance, config.pidPSlope);
    float p = config.pidPMin + (config.pidPMax - config.pidPMin) * distancePower;
    return std::max(config.pidPMin, std::min(config.pidPMax, p));
}

float MAKCUMouseController::calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor)
{
    UNUSED_PARAMETER(distance);
    UNUSED_PARAMETER(deltaError);
    UNUSED_PARAMETER(error);
    adaptiveFactor = 1.0f;
    return config.pidD;
}

float MAKCUMouseController::calculateIntegral(float error, float& integral, float deltaTime)
{
    if (std::abs(error) < config.integralDeadZone) {
        return 0.0f;
    }
    
    if (std::abs(error) > config.integralSeparationThreshold) {
        return 0.0f;
    }
    
    integral += error * deltaTime;
    integral = std::max(-config.integralLimit, std::min(integral, config.integralLimit));
    
    return integral;
}

float MAKCUMouseController::getCurrentPGain()
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetLockStartTime).count() / 1000.0f;
    
    float rampFactor = std::min(elapsed / config.pGainRampDuration, 1.0f);
    float currentScale = config.pGainRampInitialScale + (1.0f - config.pGainRampInitialScale) * rampFactor;
    
    return currentScale;
}

// 标准PID计算函数
float MAKCUMouseController::calculateStandardPID(float error, float& integral, float& integralGain, float& lastError, float deltaTime)
{
    // 死区处理
    if (std::abs(error) < config.stdDeadZone) {
        error = 0.0f;
    }

    // 比例计算
    float kp = config.stdKp * error;

    // 积分计算配合积分分离
    if (adjustStandardIntegral(error, lastError, integralGain))
    {
        // 积分累加
        integral += error;

        // 积分限幅
        integral = std::clamp(integral, -config.stdIntegralLimit, config.stdIntegralLimit);
    }
    else
    {
        // 积分清零
        integral = 0;
    }

    // 积分计算 + 死区处理
    float ki = (std::abs(integral) > config.stdIntegralDeadzone) ? config.stdKi * integral : 0;

    // 微分计算
    float kd = config.stdKd * (error - lastError);

    // 总输出计算
    float output = kp + ki + kd;

    // 总输出限幅
    output = std::clamp(output, -config.stdOutputLimit, config.stdOutputLimit);

    // 状态更新
    lastError = error;

    // 返回输出
    return output;
}

bool MAKCUMouseController::adjustStandardIntegral(float error, float lastError, float& integralGain)
{
    float errorDerivative = std::abs(error - lastError);

    if (std::abs(error) < config.stdIntegralThreshold) {
        float adaptRate = config.stdIntegralRate * (1.0f - errorDerivative / (config.stdIntegralThreshold * 2.0f));
        adaptRate = (adaptRate < 0.0f) ? 0.0f : ((adaptRate > config.stdIntegralRate) ? config.stdIntegralRate : adaptRate);
        integralGain = (integralGain + adaptRate > 1.0f) ? 1.0f : integralGain + adaptRate;
    }
    else {
        float decay = 0.1f + 0.9f * std::tanh(std::abs(error) / (config.stdIntegralThreshold * 2.0f));
        integralGain *= (1.0f - 0.05f * decay);
    }
    integralGain = (integralGain < 0.0f) ? 0.0f : ((integralGain > 1.0f) ? 1.0f : integralGain);
    return integralGain > 0.01f;  // 只有当积分增益大于0.01时才进行积分
}

void MAKCUMouseController::resetPidState()
{
    pidPreviousErrorX = 0.0f;
    pidPreviousErrorY = 0.0f;
    filteredDeltaErrorX = 0.0f;
    filteredDeltaErrorY = 0.0f;
    integralX = 0.0f;
    integralY = 0.0f;
    predictor.reset();
    // 重置卡尔曼滤波器
    kalmanFilter.reset();
    kalmanFilterInitialized = false;
    // 重置标准PID状态
    stdIntegralX = 0.0f;
    stdIntegralY = 0.0f;
    stdIntegralGainX = 0.0f;
    stdIntegralGainY = 0.0f;
    stdLastErrorX = 0.0f;
    stdLastErrorY = 0.0f;
}

void MAKCUMouseController::resetMotionState()
{
    currentVelocityX = 0.0f;
    currentVelocityY = 0.0f;
    currentAccelerationX = 0.0f;
    currentAccelerationY = 0.0f;
    previousMoveX = 0.0f;
    previousMoveY = 0.0f;
}

int MAKCUMouseController::getRandomDelay()
{
    if (!config.autoTriggerDelayRandomEnabled) {
        return 0;
    }
    if (config.autoTriggerDelayRandomMin >= config.autoTriggerDelayRandomMax) {
        return config.autoTriggerDelayRandomMin;
    }
    std::uniform_int_distribution<int> dist(config.autoTriggerDelayRandomMin, config.autoTriggerDelayRandomMax);
    return dist(randomGenerator);
}

int MAKCUMouseController::getRandomDuration()
{
    if (!config.autoTriggerDurationRandomEnabled) {
        return 0;
    }
    if (config.autoTriggerDurationRandomMin >= config.autoTriggerDurationRandomMax) {
        return config.autoTriggerDurationRandomMin;
    }
    std::uniform_int_distribution<int> dist(config.autoTriggerDurationRandomMin, config.autoTriggerDurationRandomMax);
    return dist(randomGenerator);
}

void MAKCUMouseController::performAutoClick()
{
    clickDown();
    autoTriggerHolding = true;
    isFiring = true;
    autoTriggerFireStartTime = std::chrono::steady_clock::now();
    currentFireDuration = config.autoTriggerFireDuration + getRandomDuration();
}

void MAKCUMouseController::releaseAutoTrigger()
{
    obs_log(LOG_INFO, "[MAKCU-AutoTrigger] releaseAutoTrigger called, holding=%d", autoTriggerHolding);
    if (autoTriggerHolding) {
        obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Sending km.left(0) to release");
        clickUp();
        autoTriggerHolding = false;
    }
    isFiring = false;
    autoTriggerWaitingForDelay = false;
}

bool MAKCUMouseController::isConnected()
{
    return serialConnected;
}

bool MAKCUMouseController::testCommunication()
{
    if (!serialConnected || hSerial == INVALID_HANDLE_VALUE) {
        return false;
    }

    std::string testCommand = "km.echo(1)";
    bool success = sendSerialCommand(testCommand);
    
    return success;
}

void MAKCUMouseController::setCurrentWeapon(const std::string& weaponName)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentWeapon_ = weaponName;
}

std::string MAKCUMouseController::getCurrentWeapon() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return currentWeapon_;
}

bool MAKCUMouseController::getKalmanPrediction(float& predX, float& predY) const
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!kalmanFilterInitialized || !config.useKalmanFilter) {
        return false;
    }
    // 获取预测的绝对坐标位置
    kalmanFilter.getPrediction(deltaTime, predX, predY);
    return true;
}

#endif
