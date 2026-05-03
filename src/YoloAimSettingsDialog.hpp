#pragma once

#ifdef ENABLE_QT

#include <QDialog>
#include <QMainWindow>
#include <QComboBox>
#include <QTabWidget>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QSlider>
#include <QGroupBox>
#include <QScrollArea>
#include <obs-frontend-api.h>
#include <obs-module.h>

class YoloAimSettingsDialog : public QDialog
{
    Q_OBJECT

public:
    static YoloAimSettingsDialog* instance();
    static void showSettingsDialog();
    
    ~YoloAimSettingsDialog();

private slots:
    void onSourceChanged(int index);
    void onConfigChanged(int index);
    void onPageChanged(int index);
    void onSaveClicked();
    void onResetClicked();
    void onApplyClicked();

private:
    YoloAimSettingsDialog(QWidget *parent = nullptr);
    YoloAimSettingsDialog(const YoloAimSettingsDialog&) = delete;
    YoloAimSettingsDialog& operator=(const YoloAimSettingsDialog&) = delete;
    
    void setupUI();
    void setupModelDetectionPage();
    void setupVisualPage();
    void setupMouseControlPage();
    void setupTrackingPredictorPage();
    void setupMotionSimPage();
    
    void loadSettings();
    void saveSettings();
    void applySettings();
    void refreshSourceList();
    void refreshConfigUI();
    void updateVisibility();
    
    void attachFilterToSource(const QString& sourceName);
    void detachFilterFromSource(const QString& sourceName);
    
    static YoloAimSettingsDialog* dialogInstance;
    
    QWidget* createConfigWidget(int configIndex);
    QGroupBox* createAdvancedPIDGroup(int configIndex);
    QGroupBox* createAutoTriggerGroup(int configIndex);
    QGroupBox* createRecoilGroup(int configIndex);
    QGroupBox* createPredictorGroup(int configIndex);
    QGroupBox* createBezierGroup(int configIndex);
    void initConfigWidgetStruct(int configIndex);
    void initAllConfigWidgetStructs();
    void switchConfigVisibility();
    
    QComboBox* m_sourceCombo;
    QTabWidget* m_tabWidget;
    QComboBox* m_configSelect;
    
    QCheckBox* m_showDetectionResultsCheck;
    QCheckBox* m_showFOVCheck;
    QCheckBox* m_showFOVCircleCheck;
    QCheckBox* m_showFOVCrossCheck;
    QSpinBox* m_fovRadiusSpin;
    QSpinBox* m_fovCrossLineScaleSpin;
    QSpinBox* m_fovCrossLineThicknessSpin;
    QSpinBox* m_fovCircleThicknessSpin;
    
    // 模型配置页面
    QLineEdit* m_modelPathEdit;
    QPushButton* m_modelPathBtn;
    QComboBox* m_modelVersionCombo;
    QComboBox* m_useGPUCombo;
    QCheckBox* m_useGPUTextureCheck;
    QComboBox* m_inputResolutionCombo;
    QSpinBox* m_numThreadsSpin;
    
    // 检测配置页面
    QDoubleSpinBox* m_confidenceThresholdSpin;
    QDoubleSpinBox* m_nmsThresholdSpin;
    QComboBox* m_targetClassCombo;
    QLineEdit* m_targetClassesTextEdit;
    QSpinBox* m_inferenceIntervalSpin;
    
    // 视觉页面 - 区域检测
    QCheckBox* m_useRegionCheck;
    QSpinBox* m_regionXSpin;
    QSpinBox* m_regionYSpin;
    QSpinBox* m_regionWidthSpin;
    QSpinBox* m_regionHeightSpin;
    
    // 视觉页面 - 渲染配置
    QSpinBox* m_bboxLineWidthSpin;
    QPushButton* m_bboxColorBtn;
    QDoubleSpinBox* m_labelFontScaleSpin;
    
    // 视觉页面 - 动态FOV
    QCheckBox* m_useDynamicFOVCheck;
    QCheckBox* m_showFOV2Check;
    QSpinBox* m_fovRadius2Spin;
    QPushButton* m_fovColorBtn;
    QPushButton* m_fovColor2Btn;
    QSpinBox* m_dynamicFovShrinkSpin;
    QSpinBox* m_dynamicFovTransitionSpin;
    
    // 检测框平滑
    QCheckBox* m_detectionSmoothingCheck;
    QDoubleSpinBox* m_detectionSmoothingAlphaSpin;
    
    // KalmanFilter 追踪设置
    QCheckBox* m_useKalmanTrackerCheck;
    QSpinBox* m_kalmanGenerateThresholdSpin;
    QSpinBox* m_kalmanTerminateCountSpin;
    
    // MotionSimulator 人类行为模拟器设置
    QCheckBox* m_enableMotionSimulatorCheck;
    QCheckBox* m_motionSimRandomPosCheck;
    QCheckBox* m_motionSimOvershootCheck;
    QCheckBox* m_motionSimMicroOvershootCheck;
    QCheckBox* m_motionSimInertiaCheck;
    QCheckBox* m_motionSimLeftBtnAdaptiveCheck;
    QCheckBox* m_motionSimSprayModeCheck;
    QCheckBox* m_motionSimTapPauseCheck;
    QCheckBox* m_motionSimRetryCheck;
    QSpinBox* m_motionSimMaxRetrySpin;
    QSpinBox* m_motionSimDelayMsSpin;
    QDoubleSpinBox* m_motionSimDirectProbSpin;
    QDoubleSpinBox* m_motionSimOvershootProbSpin;
    QDoubleSpinBox* m_motionSimMicroOvshootProbSpin;
    
    // 神经网络轨迹生成器设置
    QCheckBox* m_enableNeuralPathCheck;
    QSpinBox* m_neuralPathPointsSpin;
    QDoubleSpinBox* m_neuralMouseStepSizeSpin;
    QSpinBox* m_neuralTargetRadiusSpin;
    QSpinBox* m_neuralConsumePerFrameSpin;
    
    // 高级配置
    QCheckBox* m_exportCoordinatesCheck;
    QLineEdit* m_coordinateOutputPathEdit;
    QPushButton* m_coordinateOutputPathBtn;
    
    // 推理控制
    QPushButton* m_toggleInferenceBtn;
    QLabel* m_inferenceStatusLabel;
    
    int m_currentConfig;
    QString m_currentSource;
    
    // 每套配置的容器widget，用于切换可见性
    QWidget* m_mouseConfigContainers[5];
    QWidget* m_trackingConfigContainers[5];
    
    struct ConfigWidgets {
        QCheckBox* enabledCheck;
        QComboBox* hotkeyCombo;
        QComboBox* controllerTypeCombo;
        
        QDoubleSpinBox* pMinSpin;
        QDoubleSpinBox* pMaxSpin;
        QDoubleSpinBox* pSlopeSpin;
        QDoubleSpinBox* dSpin;
        QDoubleSpinBox* iSpin;
        QDoubleSpinBox* derivativeFilterAlphaSpin;
        
        QDoubleSpinBox* advTargetThresholdSpin;
        QDoubleSpinBox* advMinCoefficientSpin;
        QDoubleSpinBox* advMaxCoefficientSpin;
        QDoubleSpinBox* advTransitionSharpnessSpin;
        QDoubleSpinBox* advTransitionMidpointSpin;
        QDoubleSpinBox* advOutputSmoothingSpin;
        QDoubleSpinBox* advSpeedFactorSpin;
        
        QCheckBox* useOneEuroFilterCheck;
        QDoubleSpinBox* oneEuroMinCutoffSpin;
        QDoubleSpinBox* oneEuroBetaSpin;
        QDoubleSpinBox* oneEuroDCutoffSpin;
        
        QDoubleSpinBox* aimSmoothingXSpin;
        QDoubleSpinBox* aimSmoothingYSpin;
        QDoubleSpinBox* targetYOffsetSpin;
        QDoubleSpinBox* maxPixelMoveSpin;
        QDoubleSpinBox* deadZonePixelsSpin;
        
        QSpinBox* screenOffsetXSpin;
        QSpinBox* screenOffsetYSpin;
        QSpinBox* screenWidthSpin;
        QSpinBox* screenHeightSpin;
        
        QCheckBox* enableYAxisUnlockCheck;
        QSpinBox* yAxisUnlockDelaySpin;
        
        QGroupBox* autoTriggerGroup;
        QSpinBox* triggerRadiusSpin;
        QSpinBox* triggerCooldownSpin;
        QSpinBox* triggerFireDelaySpin;
        QSpinBox* triggerFireDurationSpin;
        QSpinBox* triggerIntervalSpin;
        QCheckBox* enableTriggerDelayRandomCheck;
        QSpinBox* triggerDelayRandomMinSpin;
        QSpinBox* triggerDelayRandomMaxSpin;
        QCheckBox* enableTriggerDurationRandomCheck;
        QSpinBox* triggerDurationRandomMinSpin;
        QSpinBox* triggerDurationRandomMaxSpin;
        QSpinBox* triggerMoveCompensationSpin;
        
        QGroupBox* recoilGroup;
        QDoubleSpinBox* recoilStrengthSpin;
        QSpinBox* recoilSpeedSpin;
        QDoubleSpinBox* recoilPidGainScaleSpin;
        
        QDoubleSpinBox* integralLimitSpin;
        QDoubleSpinBox* integralSeparationThresholdSpin;
        QDoubleSpinBox* integralDeadZoneSpin;
        QDoubleSpinBox* integralRateSpin;
        QDoubleSpinBox* pGainRampInitialScaleSpin;
        QDoubleSpinBox* pGainRampDurationSpin;
        
        QGroupBox* predictorGroup;
        QDoubleSpinBox* predictionWeightXSpin;
        QDoubleSpinBox* predictionWeightYSpin;
        QDoubleSpinBox* maxPredictionTimeSpin;
        
        QGroupBox* bezierGroup;
        QDoubleSpinBox* bezierCurvatureSpin;
        QDoubleSpinBox* bezierRandomnessSpin;
    };
    
    ConfigWidgets m_configWidgets[5];
};

#endif
