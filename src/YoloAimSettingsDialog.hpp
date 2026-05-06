#pragma once

#ifdef ENABLE_QT

#include <QDialog>
#include <QMainWindow>
#include <obs-frontend-api.h>
#include <obs-module.h>

#include "Fluent/FluentDialog.h"
#include "Fluent/FluentButton.h"
#include "Fluent/FluentCheckBox.h"
#include "Fluent/FluentComboBox.h"
#include "Fluent/FluentSpinBox.h"
#include "Fluent/FluentLineEdit.h"
#include "Fluent/FluentSlider.h"
#include "Fluent/FluentLabel.h"
#include "Fluent/FluentTabWidget.h"
#include "Fluent/FluentScrollArea.h"
#include "Fluent/FluentTheme.h"
#include "Fluent/FluentGroupBox.h"
#include "Fluent/FluentWidget.h"

class YoloAimSettingsDialog : public Fluent::FluentDialog
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
    
    Fluent::FluentComboBox* m_sourceCombo;
    Fluent::FluentTabWidget* m_tabWidget;
    Fluent::FluentComboBox* m_configSelect;
    
    Fluent::FluentCheckBox* m_showDetectionResultsCheck;
    Fluent::FluentCheckBox* m_showFOVCheck;
    Fluent::FluentCheckBox* m_showFOVCircleCheck;
    Fluent::FluentCheckBox* m_showFOVCrossCheck;
    Fluent::FluentSpinBox* m_fovRadiusSpin;
    Fluent::FluentSpinBox* m_fovCrossLineScaleSpin;
    Fluent::FluentSpinBox* m_fovCrossLineThicknessSpin;
    Fluent::FluentSpinBox* m_fovCircleThicknessSpin;
    
    // 模型配置页面
    Fluent::FluentLineEdit* m_modelPathEdit;
    Fluent::FluentButton* m_modelPathBtn;
    Fluent::FluentComboBox* m_modelVersionCombo;
    Fluent::FluentComboBox* m_useGPUCombo;
    Fluent::FluentCheckBox* m_useGPUTextureCheck;
    Fluent::FluentComboBox* m_inputResolutionCombo;
    Fluent::FluentSpinBox* m_numThreadsSpin;
    
    // 检测配置页面
    Fluent::FluentDoubleSpinBox* m_confidenceThresholdSpin;
    Fluent::FluentDoubleSpinBox* m_nmsThresholdSpin;
    Fluent::FluentComboBox* m_targetClassCombo;
    Fluent::FluentLineEdit* m_targetClassesTextEdit;
    Fluent::FluentSpinBox* m_inferenceIntervalSpin;
    
    // 视觉页面 - 区域检测
    Fluent::FluentCheckBox* m_useRegionCheck;
    Fluent::FluentSpinBox* m_regionXSpin;
    Fluent::FluentSpinBox* m_regionYSpin;
    Fluent::FluentSpinBox* m_regionWidthSpin;
    Fluent::FluentSpinBox* m_regionHeightSpin;
    
    // 视觉页面 - 渲染配置
    Fluent::FluentSpinBox* m_bboxLineWidthSpin;
    Fluent::FluentButton* m_bboxColorBtn;
    Fluent::FluentDoubleSpinBox* m_labelFontScaleSpin;
    
    // 视觉页面 - 动态FOV
    Fluent::FluentCheckBox* m_useDynamicFOVCheck;
    Fluent::FluentCheckBox* m_showFOV2Check;
    Fluent::FluentSpinBox* m_fovRadius2Spin;
    Fluent::FluentButton* m_fovColorBtn;
    Fluent::FluentButton* m_fovColor2Btn;
    Fluent::FluentSpinBox* m_dynamicFovShrinkSpin;
    Fluent::FluentSpinBox* m_dynamicFovTransitionSpin;
    
    // 检测框平滑
    Fluent::FluentCheckBox* m_detectionSmoothingCheck;
    Fluent::FluentDoubleSpinBox* m_detectionSmoothingAlphaSpin;
    
    // KalmanFilter 追踪设置
    Fluent::FluentCheckBox* m_useKalmanTrackerCheck;
    Fluent::FluentSpinBox* m_kalmanGenerateThresholdSpin;
    Fluent::FluentSpinBox* m_kalmanTerminateCountSpin;
    
    // 神经网络轨迹生成器设置
    Fluent::FluentCheckBox* m_enableNeuralPathCheck;
    Fluent::FluentSpinBox* m_neuralPathPointsSpin;
    Fluent::FluentDoubleSpinBox* m_neuralMouseStepSizeSpin;
    Fluent::FluentSpinBox* m_neuralTargetRadiusSpin;
    Fluent::FluentSpinBox* m_neuralConsumePerFrameSpin;
    
    // 高级配置
    Fluent::FluentCheckBox* m_exportCoordinatesCheck;
    Fluent::FluentLineEdit* m_coordinateOutputPathEdit;
    Fluent::FluentButton* m_coordinateOutputPathBtn;
    
    // 推理控制
    Fluent::FluentButton* m_toggleInferenceBtn;
    Fluent::FluentLabel* m_inferenceStatusLabel;
    
    int m_currentConfig;
    QString m_currentSource;
    
    // 每套配置的容器widget，用于切换可见性
    QWidget* m_mouseConfigContainers[5];
    QWidget* m_trackingConfigContainers[5];
    
    struct ConfigWidgets {
        Fluent::FluentCheckBox* enabledCheck;
        Fluent::FluentComboBox* hotkeyCombo;
        Fluent::FluentComboBox* controllerTypeCombo;
        
        Fluent::FluentDoubleSpinBox* pMinSpin;
        Fluent::FluentDoubleSpinBox* pMaxSpin;
        Fluent::FluentDoubleSpinBox* pSlopeSpin;
        Fluent::FluentDoubleSpinBox* dSpin;
        Fluent::FluentDoubleSpinBox* iSpin;
        Fluent::FluentDoubleSpinBox* derivativeFilterAlphaSpin;
        
        Fluent::FluentDoubleSpinBox* advTargetThresholdSpin;
        Fluent::FluentDoubleSpinBox* advMinCoefficientSpin;
        Fluent::FluentDoubleSpinBox* advMaxCoefficientSpin;
        Fluent::FluentDoubleSpinBox* advTransitionSharpnessSpin;
        Fluent::FluentDoubleSpinBox* advTransitionMidpointSpin;
        Fluent::FluentDoubleSpinBox* advOutputSmoothingSpin;
        Fluent::FluentDoubleSpinBox* advSpeedFactorSpin;
        
        Fluent::FluentCheckBox* useOneEuroFilterCheck;
        Fluent::FluentDoubleSpinBox* oneEuroMinCutoffSpin;
        Fluent::FluentDoubleSpinBox* oneEuroBetaSpin;
        Fluent::FluentDoubleSpinBox* oneEuroDCutoffSpin;
        
        Fluent::FluentDoubleSpinBox* aimSmoothingXSpin;
        Fluent::FluentDoubleSpinBox* aimSmoothingYSpin;
        Fluent::FluentDoubleSpinBox* targetYOffsetSpin;
        Fluent::FluentDoubleSpinBox* maxPixelMoveSpin;
        Fluent::FluentDoubleSpinBox* deadZonePixelsSpin;
        
        Fluent::FluentSpinBox* screenOffsetXSpin;
        Fluent::FluentSpinBox* screenOffsetYSpin;
        Fluent::FluentSpinBox* screenWidthSpin;
        Fluent::FluentSpinBox* screenHeightSpin;
        
        Fluent::FluentCheckBox* enableYAxisUnlockCheck;
        Fluent::FluentSpinBox* yAxisUnlockDelaySpin;
        
        QGroupBox* autoTriggerGroup;
        Fluent::FluentSpinBox* triggerRadiusSpin;
        Fluent::FluentSpinBox* triggerCooldownSpin;
        Fluent::FluentSpinBox* triggerFireDelaySpin;
        Fluent::FluentSpinBox* triggerFireDurationSpin;
        Fluent::FluentSpinBox* triggerIntervalSpin;
        Fluent::FluentCheckBox* enableTriggerDelayRandomCheck;
        Fluent::FluentSpinBox* triggerDelayRandomMinSpin;
        Fluent::FluentSpinBox* triggerDelayRandomMaxSpin;
        Fluent::FluentCheckBox* enableTriggerDurationRandomCheck;
        Fluent::FluentSpinBox* triggerDurationRandomMinSpin;
        Fluent::FluentSpinBox* triggerDurationRandomMaxSpin;
        Fluent::FluentSpinBox* triggerMoveCompensationSpin;
        
        QGroupBox* recoilGroup;
        Fluent::FluentDoubleSpinBox* recoilStrengthSpin;
        Fluent::FluentSpinBox* recoilSpeedSpin;
        Fluent::FluentDoubleSpinBox* recoilPidGainScaleSpin;
        
        Fluent::FluentDoubleSpinBox* integralLimitSpin;
        Fluent::FluentDoubleSpinBox* integralRateSpin;
        Fluent::FluentDoubleSpinBox* pGainRampInitialScaleSpin;
        Fluent::FluentDoubleSpinBox* pGainRampDurationSpin;
        
        QGroupBox* predictorGroup;
        Fluent::FluentDoubleSpinBox* predictionWeightXSpin;
        Fluent::FluentDoubleSpinBox* predictionWeightYSpin;
        Fluent::FluentDoubleSpinBox* maxPredictionTimeSpin;
        
        QGroupBox* bezierGroup;
        Fluent::FluentDoubleSpinBox* bezierCurvatureSpin;
        Fluent::FluentDoubleSpinBox* bezierRandomnessSpin;
    };
    
    ConfigWidgets m_configWidgets[5];
};

#endif
