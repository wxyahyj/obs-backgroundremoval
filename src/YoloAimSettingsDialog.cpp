#ifdef ENABLE_QT

#include "YoloAimSettingsDialog.hpp"
#include "yolo-detector-filter.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QFile>
#include <QDir>
#include <QFrame>
#include <obs.hpp>

extern "C" {
void YoloAimSettingsDialog_Show()
{
    YoloAimSettingsDialog::showSettingsDialog();
}
}

YoloAimSettingsDialog* YoloAimSettingsDialog::dialogInstance = nullptr;

YoloAimSettingsDialog* YoloAimSettingsDialog::instance()
{
    if (!dialogInstance) {
        void* obsWindow = obs_frontend_get_main_window();
        QMainWindow* mainWindow = obsWindow ? reinterpret_cast<QMainWindow*>(obsWindow) : nullptr;
        dialogInstance = new YoloAimSettingsDialog(mainWindow);
    }
    return dialogInstance;
}

void YoloAimSettingsDialog::showSettingsDialog()
{
    YoloAimSettingsDialog* dialog = instance();
    dialog->refreshSourceList();
    dialog->loadSettings();
    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

YoloAimSettingsDialog::YoloAimSettingsDialog(QWidget *parent)
    : Fluent::FluentDialog(parent)
    , m_currentConfig(0)
    , m_showDetectionResultsCheck(nullptr)
    , m_showFOVCheck(nullptr)
    , m_showFOVCircleCheck(nullptr)
    , m_showFOVCrossCheck(nullptr)
    , m_fovRadiusSpin(nullptr)
    , m_fovCrossLineScaleSpin(nullptr)
    , m_fovCrossLineThicknessSpin(nullptr)
    , m_fovCircleThicknessSpin(nullptr)
    , m_modelPathEdit(nullptr)
    , m_modelPathBtn(nullptr)
    , m_modelVersionCombo(nullptr)
    , m_useGPUCombo(nullptr)
    , m_useGPUTextureCheck(nullptr)
    , m_inputResolutionCombo(nullptr)
    , m_numThreadsSpin(nullptr)
    , m_confidenceThresholdSpin(nullptr)
    , m_nmsThresholdSpin(nullptr)
    , m_targetClassCombo(nullptr)
    , m_targetClassesTextEdit(nullptr)
    , m_inferenceIntervalSpin(nullptr)
    , m_useRegionCheck(nullptr)
    , m_regionXSpin(nullptr)
    , m_regionYSpin(nullptr)
    , m_regionWidthSpin(nullptr)
    , m_regionHeightSpin(nullptr)
    , m_bboxLineWidthSpin(nullptr)
    , m_bboxColorBtn(nullptr)
    , m_labelFontScaleSpin(nullptr)
    , m_useDynamicFOVCheck(nullptr)
    , m_showFOV2Check(nullptr)
    , m_fovRadius2Spin(nullptr)
    , m_fovColorBtn(nullptr)
    , m_fovColor2Btn(nullptr)
    , m_dynamicFovShrinkSpin(nullptr)
    , m_dynamicFovTransitionSpin(nullptr)
    , m_detectionSmoothingCheck(nullptr)
    , m_detectionSmoothingAlphaSpin(nullptr)
    , m_exportCoordinatesCheck(nullptr)
    , m_coordinateOutputPathEdit(nullptr)
    , m_coordinateOutputPathBtn(nullptr)
    , m_toggleInferenceBtn(nullptr)
    , m_inferenceStatusLabel(nullptr)
{
    initAllConfigWidgetStructs();
    
    Fluent::ThemeManager::instance().setDarkMode();
    
    setWindowTitle(QStringLiteral("小鱼设置"));
    setMinimumSize(1000, 600);
    resize(1200, 700);
    
    setupUI();
    refreshSourceList();
}

YoloAimSettingsDialog::~YoloAimSettingsDialog()
{
    dialogInstance = nullptr;
}

void YoloAimSettingsDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(16, 16, 16, 16);
    mainLayout->setSpacing(12);
    
    QHBoxLayout* topLayout = new QHBoxLayout();
    topLayout->setSpacing(12);
    
    QLabel* sourceLabel = new Fluent::FluentLabel(QStringLiteral("视频源:"), this);
    m_sourceCombo = new Fluent::FluentComboBox(this);
    m_sourceCombo->setMinimumWidth(200);
    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &YoloAimSettingsDialog::onSourceChanged);
    
    topLayout->addWidget(sourceLabel);
    topLayout->addWidget(m_sourceCombo);
    topLayout->addSpacing(24);
    
    m_toggleInferenceBtn = new Fluent::FluentButton(QStringLiteral("▶ 开始推理"), this);
    m_toggleInferenceBtn->setCheckable(true);
    connect(m_toggleInferenceBtn, &QPushButton::clicked, this, [this]() {
        bool isInferencing = m_toggleInferenceBtn->isChecked();
        
        if (!m_currentSource.isEmpty()) {
            obs_source_t* source = obs_get_source_by_name(m_currentSource.toUtf8().constData());
            if (!source) {
                QMessageBox::warning(this, QStringLiteral("错误"), QStringLiteral("未找到视频源: ") + m_currentSource);
                m_toggleInferenceBtn->setChecked(false);
                return;
            }
            
            obs_source_t* filter = obs_source_get_filter_by_name(source, "visual-assist-hidden");
            if (!filter) {
                QMessageBox::warning(this, QStringLiteral("错误"), QStringLiteral("未找到视觉辅助过滤器，请确保已添加过滤器到视频源"));
                obs_source_release(source);
                m_toggleInferenceBtn->setChecked(false);
                return;
            }
            
            obs_data_t* settings = obs_source_get_settings(filter);
            if (settings) {
                obs_data_set_bool(settings, "is_inferencing", isInferencing);
                obs_source_update(filter, settings);
                obs_data_release(settings);
            }
            obs_source_release(filter);
            obs_source_release(source);
            
            m_toggleInferenceBtn->setText(isInferencing ? QStringLiteral("⏹ 停止推理") : QStringLiteral("▶ 开始推理"));
            m_inferenceStatusLabel->setText(isInferencing ? QStringLiteral("状态: 推理运行中") : QStringLiteral("状态: 已停止"));
        } else {
            QMessageBox::warning(this, QStringLiteral("错误"), QStringLiteral("请先选择视频源"));
            m_toggleInferenceBtn->setChecked(false);
        }
    });
    topLayout->addWidget(m_toggleInferenceBtn);
    
    m_inferenceStatusLabel = new Fluent::FluentLabel(QStringLiteral("状态: 已停止"), this);
    topLayout->addWidget(m_inferenceStatusLabel);
    
    topLayout->addSpacing(36);
    
    QLabel* configLabel = new Fluent::FluentLabel(QStringLiteral("配置:"), this);
    topLayout->addWidget(configLabel);
    m_configSelect = new Fluent::FluentComboBox(this);
    for (int i = 0; i < 5; i++) {
        m_configSelect->addItem(QStringLiteral("配置 %1").arg(i + 1));
    }
    m_configSelect->setMinimumWidth(120);
    connect(m_configSelect, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &YoloAimSettingsDialog::onConfigChanged);
    topLayout->addWidget(m_configSelect);
    
    topLayout->addStretch();
    
    mainLayout->addLayout(topLayout);
    
    m_tabWidget = new Fluent::FluentTabWidget(this);
    
    setupModelDetectionPage();
    setupVisualPage();
    setupMouseControlPage();
    setupTrackingPredictorPage();
    setupMotionSimPage();
    
    mainLayout->addWidget(m_tabWidget, 1);
    
    QDialogButtonBox* buttonBox = new QDialogButtonBox(
        QDialogButtonBox::Save | QDialogButtonBox::Apply | QDialogButtonBox::Reset | QDialogButtonBox::Close,
        this
    );
    
    connect(buttonBox->button(QDialogButtonBox::Save), &QPushButton::clicked, this, &YoloAimSettingsDialog::onSaveClicked);
    connect(buttonBox->button(QDialogButtonBox::Apply), &QPushButton::clicked, this, &YoloAimSettingsDialog::onApplyClicked);
    connect(buttonBox->button(QDialogButtonBox::Reset), &QPushButton::clicked, this, &YoloAimSettingsDialog::onResetClicked);
    connect(buttonBox->button(QDialogButtonBox::Close), &QPushButton::clicked, this, &QDialog::close);
    
    mainLayout->addWidget(buttonBox);
}

void YoloAimSettingsDialog::setupModelDetectionPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(12, 12, 12, 12);
    pageLayout->setSpacing(16);
    
    QScrollArea* scrollArea = new Fluent::FluentScrollArea(page);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget* scrollContent = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(scrollContent);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(16);
    
    QGroupBox* modelGroup = new Fluent::FluentGroupBox(QStringLiteral("📦 模型设置"), scrollContent);
    QGridLayout* modelLayout = new QGridLayout(modelGroup);
    modelLayout->setContentsMargins(16, 24, 16, 16);
    modelLayout->setHorizontalSpacing(16);
    modelLayout->setVerticalSpacing(12);
    int row = 0;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("模型路径:"), scrollContent), row, 0);
    m_modelPathEdit = new Fluent::FluentLineEdit(scrollContent);
    m_modelPathBtn = new Fluent::FluentButton(QStringLiteral("浏览..."), scrollContent);
    QHBoxLayout* modelPathLayout = new QHBoxLayout();
    modelPathLayout->addWidget(m_modelPathEdit);
    modelPathLayout->addWidget(m_modelPathBtn);
    modelLayout->addLayout(modelPathLayout, row, 1);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("模型版本:"), scrollContent), row, 0);
    m_modelVersionCombo = new Fluent::FluentComboBox(scrollContent);
    m_modelVersionCombo->addItem(QStringLiteral("YOLOv5"), 0);
    m_modelVersionCombo->addItem(QStringLiteral("YOLOv8"), 1);
    m_modelVersionCombo->addItem(QStringLiteral("YOLOv11"), 2);
    modelLayout->addWidget(m_modelVersionCombo, row, 1);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("计算设备:"), scrollContent), row, 0);
    m_useGPUCombo = new Fluent::FluentComboBox(scrollContent);
    m_useGPUCombo->addItem(QStringLiteral("CPU"), QStringLiteral("cpu"));
    m_useGPUCombo->addItem(QStringLiteral("CUDA"), QStringLiteral("cuda"));
    m_useGPUCombo->addItem(QStringLiteral("DirectML"), QStringLiteral("dml"));
    m_useGPUCombo->addItem(QStringLiteral("CoreML"), QStringLiteral("coreml"));
    modelLayout->addWidget(m_useGPUCombo, row, 1);
    row++;
    
    m_useGPUTextureCheck = new Fluent::FluentCheckBox(QStringLiteral("启用GPU纹理推理(实验性)"), scrollContent);
    modelLayout->addWidget(m_useGPUTextureCheck, row, 0, 1, 2);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("输入分辨率:"), scrollContent), row, 0);
    m_inputResolutionCombo = new Fluent::FluentComboBox(scrollContent);
    m_inputResolutionCombo->addItem(QStringLiteral("640x640"), 640);
    m_inputResolutionCombo->addItem(QStringLiteral("320x320"), 320);
    m_inputResolutionCombo->addItem(QStringLiteral("1280x1280"), 1280);
    modelLayout->addWidget(m_inputResolutionCombo, row, 1);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("推理线程数:"), scrollContent), row, 0);
    m_numThreadsSpin = new Fluent::FluentSpinBox(scrollContent);
    m_numThreadsSpin->setRange(1, 16);
    m_numThreadsSpin->setValue(4);
    modelLayout->addWidget(m_numThreadsSpin, row, 1);
    
    layout->addWidget(modelGroup);
    
    QGroupBox* detectGroup = new Fluent::FluentGroupBox(QStringLiteral("🎯 检测设置"), scrollContent);
    QGridLayout* detectLayout = new QGridLayout(detectGroup);
    detectLayout->setContentsMargins(16, 24, 16, 16);
    detectLayout->setHorizontalSpacing(16);
    detectLayout->setVerticalSpacing(12);
    row = 0;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("置信度阈值:"), scrollContent), row, 0);
    m_confidenceThresholdSpin = new Fluent::FluentDoubleSpinBox(scrollContent);
    m_confidenceThresholdSpin->setRange(0.01, 1.0);
    m_confidenceThresholdSpin->setSingleStep(0.01);
    m_confidenceThresholdSpin->setValue(0.5);
    detectLayout->addWidget(m_confidenceThresholdSpin, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("NMS阈值:"), scrollContent), row, 0);
    m_nmsThresholdSpin = new Fluent::FluentDoubleSpinBox(scrollContent);
    m_nmsThresholdSpin->setRange(0.01, 1.0);
    m_nmsThresholdSpin->setSingleStep(0.01);
    m_nmsThresholdSpin->setValue(0.45);
    detectLayout->addWidget(m_nmsThresholdSpin, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("目标类别:"), scrollContent), row, 0);
    m_targetClassCombo = new Fluent::FluentComboBox(scrollContent);
    m_targetClassCombo->addItem(QStringLiteral("全部"), -1);
    m_targetClassCombo->addItem(QStringLiteral("人物"), 0);
    m_targetClassCombo->addItem(QStringLiteral("车辆"), -1);
    detectLayout->addWidget(m_targetClassCombo, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("自定义类别:"), scrollContent), row, 0);
    m_targetClassesTextEdit = new Fluent::FluentLineEdit(scrollContent);
    m_targetClassesTextEdit->setPlaceholderText(QStringLiteral("多个类别用逗号分隔，如: 0,1,2"));
    detectLayout->addWidget(m_targetClassesTextEdit, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("推理间隔(帧):"), scrollContent), row, 0);
    m_inferenceIntervalSpin = new Fluent::FluentSpinBox(scrollContent);
    m_inferenceIntervalSpin->setRange(0, 10);
    m_inferenceIntervalSpin->setValue(0);
    detectLayout->addWidget(m_inferenceIntervalSpin, row, 1);
    
    layout->addWidget(detectGroup);
    
    QGroupBox* regionGroup = new Fluent::FluentGroupBox(QStringLiteral("📐 区域检测"), scrollContent);
    QGridLayout* regionLayout = new QGridLayout(regionGroup);
    regionLayout->setContentsMargins(16, 24, 16, 16);
    regionLayout->setHorizontalSpacing(16);
    regionLayout->setVerticalSpacing(12);
    row = 0;
    
    m_useRegionCheck = new Fluent::FluentCheckBox(QStringLiteral("启用区域检测"), scrollContent);
    regionLayout->addWidget(m_useRegionCheck, row, 0, 1, 2);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域X:"), scrollContent), row, 0);
    m_regionXSpin = new Fluent::FluentSpinBox(scrollContent);
    m_regionXSpin->setRange(0, 3840);
    regionLayout->addWidget(m_regionXSpin, row, 1);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域Y:"), scrollContent), row, 0);
    m_regionYSpin = new Fluent::FluentSpinBox(scrollContent);
    m_regionYSpin->setRange(0, 2160);
    regionLayout->addWidget(m_regionYSpin, row, 1);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域宽度:"), scrollContent), row, 0);
    m_regionWidthSpin = new Fluent::FluentSpinBox(scrollContent);
    m_regionWidthSpin->setRange(1, 3840);
    m_regionWidthSpin->setValue(640);
    regionLayout->addWidget(m_regionWidthSpin, row, 1);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域高度:"), scrollContent), row, 0);
    m_regionHeightSpin = new Fluent::FluentSpinBox(scrollContent);
    m_regionHeightSpin->setRange(1, 2160);
    m_regionHeightSpin->setValue(480);
    regionLayout->addWidget(m_regionHeightSpin, row, 1);
    
    layout->addWidget(regionGroup);
    layout->addStretch();
    
    scrollArea->setWidget(scrollContent);
    pageLayout->addWidget(scrollArea);
    
    m_tabWidget->addTab(page, QStringLiteral("📦 模型检测"));
}

void YoloAimSettingsDialog::setupVisualPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(16);
    
    QScrollArea* scrollArea = new Fluent::FluentScrollArea(page);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget* scrollContent = new QWidget();
    QVBoxLayout* scrollLayout = new QVBoxLayout(scrollContent);
    scrollLayout->setContentsMargins(8, 8, 8, 8);
    scrollLayout->setSpacing(16);
    
    QGroupBox* displayGroup = new Fluent::FluentGroupBox(QStringLiteral("👁 显示设置"), scrollContent);
    QGridLayout* displayLayout = new QGridLayout(displayGroup);
    displayLayout->setContentsMargins(16, 24, 16, 16);
    displayLayout->setHorizontalSpacing(16);
    displayLayout->setVerticalSpacing(12);
    int row = 0;
    
    m_showDetectionResultsCheck = new Fluent::FluentCheckBox(QStringLiteral("显示检测结果"), scrollContent);
    displayLayout->addWidget(m_showDetectionResultsCheck, row, 0, 1, 2);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("边框线宽:"), scrollContent), row, 0);
    m_bboxLineWidthSpin = new Fluent::FluentSpinBox(scrollContent);
    m_bboxLineWidthSpin->setRange(1, 5);
    m_bboxLineWidthSpin->setValue(2);
    displayLayout->addWidget(m_bboxLineWidthSpin, row, 1);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("标签字体大小:"), scrollContent), row, 0);
    m_labelFontScaleSpin = new Fluent::FluentDoubleSpinBox(scrollContent);
    m_labelFontScaleSpin->setRange(0.2, 1.0);
    m_labelFontScaleSpin->setSingleStep(0.05);
    m_labelFontScaleSpin->setValue(0.5);
    displayLayout->addWidget(m_labelFontScaleSpin, row, 1);
    row++;
    
    m_detectionSmoothingCheck = new Fluent::FluentCheckBox(QStringLiteral("启用检测框平滑"), scrollContent);
    displayLayout->addWidget(m_detectionSmoothingCheck, row, 0, 1, 2);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("平滑系数:"), scrollContent), row, 0);
    m_detectionSmoothingAlphaSpin = new Fluent::FluentDoubleSpinBox(scrollContent);
    m_detectionSmoothingAlphaSpin->setRange(0.01, 1.0);
    m_detectionSmoothingAlphaSpin->setSingleStep(0.01);
    m_detectionSmoothingAlphaSpin->setValue(0.3);
    displayLayout->addWidget(m_detectionSmoothingAlphaSpin, row, 1);
    
    scrollLayout->addWidget(displayGroup);
    
    QGroupBox* fovGroup = new Fluent::FluentGroupBox(QStringLiteral("🎯 FOV设置"), scrollContent);
    QGridLayout* fovLayout = new QGridLayout(fovGroup);
    fovLayout->setContentsMargins(16, 24, 16, 16);
    fovLayout->setHorizontalSpacing(16);
    fovLayout->setVerticalSpacing(12);
    row = 0;
    
    m_showFOVCheck = new Fluent::FluentCheckBox(QStringLiteral("显示FOV"), scrollContent);
    fovLayout->addWidget(m_showFOVCheck, row, 0, 1, 2);
    row++;
    
    m_showFOVCircleCheck = new Fluent::FluentCheckBox(QStringLiteral("显示FOV圆圈"), scrollContent);
    fovLayout->addWidget(m_showFOVCircleCheck, row, 0);
    
    m_showFOVCrossCheck = new Fluent::FluentCheckBox(QStringLiteral("显示FOV十字"), scrollContent);
    fovLayout->addWidget(m_showFOVCrossCheck, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("FOV半径:"), scrollContent), row, 0);
    m_fovRadiusSpin = new Fluent::FluentSpinBox(scrollContent);
    m_fovRadiusSpin->setRange(10, 500);
    fovLayout->addWidget(m_fovRadiusSpin, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("十字线长度:"), scrollContent), row, 0);
    m_fovCrossLineScaleSpin = new Fluent::FluentSpinBox(scrollContent);
    m_fovCrossLineScaleSpin->setRange(5, 300);
    fovLayout->addWidget(m_fovCrossLineScaleSpin, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("十字线粗细:"), scrollContent), row, 0);
    m_fovCrossLineThicknessSpin = new Fluent::FluentSpinBox(scrollContent);
    m_fovCrossLineThicknessSpin->setRange(1, 10);
    fovLayout->addWidget(m_fovCrossLineThicknessSpin, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("圆圈粗细:"), scrollContent), row, 0);
    m_fovCircleThicknessSpin = new Fluent::FluentSpinBox(scrollContent);
    m_fovCircleThicknessSpin->setRange(1, 10);
    fovLayout->addWidget(m_fovCircleThicknessSpin, row, 1);
    
    scrollLayout->addWidget(fovGroup);
    
    QGroupBox* dynamicFovGroup = new Fluent::FluentGroupBox(QStringLiteral("⚡ 动态FOV"), scrollContent);
    QGridLayout* dynamicFovLayout = new QGridLayout(dynamicFovGroup);
    dynamicFovLayout->setContentsMargins(16, 24, 16, 16);
    dynamicFovLayout->setHorizontalSpacing(16);
    dynamicFovLayout->setVerticalSpacing(12);
    row = 0;
    
    m_useDynamicFOVCheck = new Fluent::FluentCheckBox(QStringLiteral("启用动态FOV"), scrollContent);
    dynamicFovLayout->addWidget(m_useDynamicFOVCheck, row, 0, 1, 2);
    row++;
    
    m_showFOV2Check = new Fluent::FluentCheckBox(QStringLiteral("显示第二个FOV"), scrollContent);
    dynamicFovLayout->addWidget(m_showFOV2Check, row, 0, 1, 2);
    row++;
    
    dynamicFovLayout->addWidget(new QLabel(QStringLiteral("第二FOV半径:"), scrollContent), row, 0);
    m_fovRadius2Spin = new Fluent::FluentSpinBox(scrollContent);
    m_fovRadius2Spin->setRange(1, 200);
    dynamicFovLayout->addWidget(m_fovRadius2Spin, row, 1);
    row++;
    
    dynamicFovLayout->addWidget(new QLabel(QStringLiteral("缩放百分比:"), scrollContent), row, 0);
    m_dynamicFovShrinkSpin = new Fluent::FluentSpinBox(scrollContent);
    m_dynamicFovShrinkSpin->setRange(10, 100);
    m_dynamicFovShrinkSpin->setValue(50);
    dynamicFovLayout->addWidget(m_dynamicFovShrinkSpin, row, 1);
    row++;
    
    dynamicFovLayout->addWidget(new QLabel(QStringLiteral("过渡时间(ms):"), scrollContent), row, 0);
    m_dynamicFovTransitionSpin = new Fluent::FluentSpinBox(scrollContent);
    m_dynamicFovTransitionSpin->setRange(0, 1000);
    m_dynamicFovTransitionSpin->setSingleStep(10);
    dynamicFovLayout->addWidget(m_dynamicFovTransitionSpin, row, 1);
    
    scrollLayout->addWidget(dynamicFovGroup);
    
    QGroupBox* advancedGroup = new Fluent::FluentGroupBox(QStringLiteral("🔧 高级设置"), scrollContent);
    QGridLayout* advancedLayout = new QGridLayout(advancedGroup);
    advancedLayout->setContentsMargins(16, 24, 16, 16);
    advancedLayout->setHorizontalSpacing(16);
    advancedLayout->setVerticalSpacing(12);
    row = 0;
    
    m_exportCoordinatesCheck = new Fluent::FluentCheckBox(QStringLiteral("导出坐标"), scrollContent);
    advancedLayout->addWidget(m_exportCoordinatesCheck, row, 0, 1, 2);
    row++;
    
    advancedLayout->addWidget(new QLabel(QStringLiteral("输出路径:"), scrollContent), row, 0);
    m_coordinateOutputPathEdit = new Fluent::FluentLineEdit(scrollContent);
    m_coordinateOutputPathBtn = new Fluent::FluentButton(QStringLiteral("浏览..."), scrollContent);
    QHBoxLayout* coordPathLayout = new QHBoxLayout();
    coordPathLayout->addWidget(m_coordinateOutputPathEdit);
    coordPathLayout->addWidget(m_coordinateOutputPathBtn);
    advancedLayout->addLayout(coordPathLayout, row, 1);
    
    scrollLayout->addWidget(advancedGroup);
    scrollLayout->addStretch();
    
    scrollArea->setWidget(scrollContent);
    layout->addWidget(scrollArea);
    
    m_tabWidget->addTab(page, QStringLiteral("👁 视觉"));
}

void YoloAimSettingsDialog::setupMouseControlPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(16);
    
    QScrollArea* scrollArea = new Fluent::FluentScrollArea(page);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget* scrollContent = new QWidget();
    QVBoxLayout* scrollLayout = new QVBoxLayout(scrollContent);
    scrollLayout->setContentsMargins(8, 8, 8, 8);
    scrollLayout->setSpacing(16);
    
    // 创建5套配置，每套包含：基础+高级PID+扳机+后坐力
    for (int i = 0; i < 5; i++) {
        QWidget* container = new QWidget(scrollContent);
        QVBoxLayout* containerLayout = new QVBoxLayout(container);
        containerLayout->setContentsMargins(0, 0, 0, 0);
        containerLayout->setSpacing(12);
        
        // 基础参数
        QWidget* basicWidget = createConfigWidget(i);
        containerLayout->addWidget(basicWidget);
        
        // 高级PID参数
        QGroupBox* pidGroup = createAdvancedPIDGroup(i);
        containerLayout->addWidget(pidGroup);
        
        // 扳机
        m_configWidgets[i].autoTriggerGroup = createAutoTriggerGroup(i);
        containerLayout->addWidget(m_configWidgets[i].autoTriggerGroup);
        
        // 后坐力
        m_configWidgets[i].recoilGroup = createRecoilGroup(i);
        containerLayout->addWidget(m_configWidgets[i].recoilGroup);
        
        containerLayout->addStretch();
        
        m_mouseConfigContainers[i] = container;
        container->setVisible(i == 0);
        scrollLayout->addWidget(container);
    }
    
    scrollLayout->addStretch();
    scrollArea->setWidget(scrollContent);
    layout->addWidget(scrollArea);
    
    m_tabWidget->addTab(page, QStringLiteral("🖱️ 鼠标控制"));
}

void YoloAimSettingsDialog::setupTrackingPredictorPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(16);
    
    QScrollArea* scrollArea = new Fluent::FluentScrollArea(page);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget* scrollContent = new QWidget();
    QVBoxLayout* scrollLayout = new QVBoxLayout(scrollContent);
    scrollLayout->setContentsMargins(8, 8, 8, 8);
    scrollLayout->setSpacing(16);
    
    // Kalman追踪（全局）
    QGroupBox* kalmanGroup = new Fluent::FluentGroupBox(QStringLiteral("📍 卡尔曼追踪"), scrollContent);
    QFormLayout* kalmanLayout = new QFormLayout(kalmanGroup);
    kalmanLayout->setContentsMargins(16, 24, 16, 16);
    kalmanLayout->setSpacing(12);
    kalmanLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    
    m_useKalmanTrackerCheck = new Fluent::FluentCheckBox(QStringLiteral("启用卡尔曼追踪"), scrollContent);
    m_useKalmanTrackerCheck->setToolTip(QStringLiteral("启用卡尔曼滤波器进行目标追踪，提供更稳定的目标ID和预测能力"));
    kalmanLayout->addRow(m_useKalmanTrackerCheck);
    
    m_kalmanGenerateThresholdSpin = new Fluent::FluentSpinBox(scrollContent);
    m_kalmanGenerateThresholdSpin->setRange(1, 10);
    m_kalmanGenerateThresholdSpin->setValue(2);
    m_kalmanGenerateThresholdSpin->setToolTip(QStringLiteral("目标需要连续检测到的帧数才能被确认追踪"));
    kalmanLayout->addRow(QStringLiteral("追踪确认阈值:"), m_kalmanGenerateThresholdSpin);
    
    m_kalmanTerminateCountSpin = new Fluent::FluentSpinBox(scrollContent);
    m_kalmanTerminateCountSpin->setRange(1, 10);
    m_kalmanTerminateCountSpin->setValue(5);
    m_kalmanTerminateCountSpin->setToolTip(QStringLiteral("目标丢失多少帧后停止追踪"));
    kalmanLayout->addRow(QStringLiteral("追踪丢失阈值:"), m_kalmanTerminateCountSpin);
    
    scrollLayout->addWidget(kalmanGroup);
    
    // 每套配置：预测器+贝塞尔
    for (int i = 0; i < 5; i++) {
        QWidget* container = new QWidget(scrollContent);
        QVBoxLayout* containerLayout = new QVBoxLayout(container);
        containerLayout->setContentsMargins(0, 0, 0, 0);
        
        m_configWidgets[i].predictorGroup = createPredictorGroup(i);
        containerLayout->addWidget(m_configWidgets[i].predictorGroup);
        
        m_configWidgets[i].bezierGroup = createBezierGroup(i);
        containerLayout->addWidget(m_configWidgets[i].bezierGroup);
        
        containerLayout->addStretch();
        
        m_trackingConfigContainers[i] = container;
        container->setVisible(i == 0);
        scrollLayout->addWidget(container);
    }
    
    scrollLayout->addStretch();
    scrollArea->setWidget(scrollContent);
    layout->addWidget(scrollArea);
    
    m_tabWidget->addTab(page, QStringLiteral("📍 追踪预测"));
}

void YoloAimSettingsDialog::setupMotionSimPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(16);
    
    QScrollArea* scrollArea = new Fluent::FluentScrollArea(page);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget* scrollContent = new QWidget();
    QVBoxLayout* scrollLayout = new QVBoxLayout(scrollContent);
    scrollLayout->setContentsMargins(8, 8, 8, 8);
    scrollLayout->setSpacing(16);
    
    // NeuralPath
    m_enableNeuralPathCheck = new Fluent::FluentCheckBox(QStringLiteral("🧠 启用神经网络轨迹"), scrollContent);
    m_enableNeuralPathCheck->setToolTip(QStringLiteral("启用神经网络轨迹生成器，生成更自然的鼠标移动轨迹"));
    scrollLayout->addWidget(m_enableNeuralPathCheck);
    
    Fluent::FluentLabel* descLabel = new Fluent::FluentLabel(QStringLiteral(
        "神经网络轨迹生成器使用预训练模型生成类人鼠标移动轨迹。\n"
        "相比贝塞尔曲线，神经网络生成的轨迹更自然、更难被检测。"
    ), scrollContent);
    descLabel->setWordWrap(true);
    scrollLayout->addWidget(descLabel);
    
    QGroupBox* neuralParamsGroup = new Fluent::FluentGroupBox(QStringLiteral("📊 神经轨迹参数"), scrollContent);
    QFormLayout* neuralParamsLayout = new QFormLayout(neuralParamsGroup);
    neuralParamsLayout->setContentsMargins(16, 24, 16, 16);
    neuralParamsLayout->setSpacing(12);
    neuralParamsLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    m_neuralPathPointsSpin = new Fluent::FluentSpinBox(scrollContent);
    m_neuralPathPointsSpin->setRange(10, 100);
    m_neuralPathPointsSpin->setValue(25);
    m_neuralPathPointsSpin->setSuffix(QStringLiteral(" 点"));
    neuralParamsLayout->addRow(QStringLiteral("轨迹点数量:"), m_neuralPathPointsSpin);
    m_neuralMouseStepSizeSpin = new Fluent::FluentDoubleSpinBox(scrollContent);
    m_neuralMouseStepSizeSpin->setRange(1.0, 20.0);
    m_neuralMouseStepSizeSpin->setDecimals(1);
    m_neuralMouseStepSizeSpin->setSingleStep(0.5);
    m_neuralMouseStepSizeSpin->setValue(4.0);
    neuralParamsLayout->addRow(QStringLiteral("鼠标步长:"), m_neuralMouseStepSizeSpin);
    m_neuralTargetRadiusSpin = new Fluent::FluentSpinBox(scrollContent);
    m_neuralTargetRadiusSpin->setRange(1, 50);
    m_neuralTargetRadiusSpin->setValue(8);
    m_neuralTargetRadiusSpin->setSuffix(QStringLiteral(" px"));
    neuralParamsLayout->addRow(QStringLiteral("目标半径:"), m_neuralTargetRadiusSpin);
    m_neuralConsumePerFrameSpin = new Fluent::FluentSpinBox(scrollContent);
    m_neuralConsumePerFrameSpin->setRange(1, 5);
    m_neuralConsumePerFrameSpin->setValue(2);
    neuralParamsLayout->addRow(QStringLiteral("每帧消费点数:"), m_neuralConsumePerFrameSpin);
    scrollLayout->addWidget(neuralParamsGroup);
    
    Fluent::FluentLabel* noteLabel = new Fluent::FluentLabel(QStringLiteral(
        "⚠️ 注意：神经网络轨迹与运动仿真模式互斥，启用其中一个会自动禁用另一个。"
    ), scrollContent);
    noteLabel->setWordWrap(true);
    scrollLayout->addWidget(noteLabel);
    
    scrollLayout->addStretch();
    scrollArea->setWidget(scrollContent);
    layout->addWidget(scrollArea);
    
    m_tabWidget->addTab(page, QStringLiteral("🎮 运动模拟"));
}

QWidget* YoloAimSettingsDialog::createConfigWidget(int configIndex)
{
    QWidget* widget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(12);
    
    QGroupBox* basicGroup = new Fluent::FluentGroupBox(QStringLiteral("⚙️ 基础参数"), widget);
    QGridLayout* basicLayout = new QGridLayout(basicGroup);
    basicLayout->setContentsMargins(16, 24, 16, 16);
    basicLayout->setHorizontalSpacing(16);
    basicLayout->setVerticalSpacing(12);
    
    int row = 0;
    
    m_configWidgets[configIndex].enabledCheck = new Fluent::FluentCheckBox(QStringLiteral("启用"), widget);
    basicLayout->addWidget(m_configWidgets[configIndex].enabledCheck, row, 0, 1, 2);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("热键:"), widget), row, 0);
    m_configWidgets[configIndex].hotkeyCombo = new Fluent::FluentComboBox(widget);
    m_configWidgets[configIndex].hotkeyCombo->addItem(QStringLiteral("鼠标侧键1"), 0x05);
    m_configWidgets[configIndex].hotkeyCombo->addItem(QStringLiteral("鼠标侧键2"), 0x06);
    m_configWidgets[configIndex].hotkeyCombo->addItem(QStringLiteral("中键"), 0x04);
    basicLayout->addWidget(m_configWidgets[configIndex].hotkeyCombo, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("控制器:"), widget), row, 2);
    m_configWidgets[configIndex].controllerTypeCombo = new Fluent::FluentComboBox(widget);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("高级PID"), 0);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("标准PID"), 1);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("ChrisPID"), 2);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("DynamicPID"), 3);
    basicLayout->addWidget(m_configWidgets[configIndex].controllerTypeCombo, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("X平滑:"), widget), row, 0);
    m_configWidgets[configIndex].aimSmoothingXSpin = new Fluent::FluentDoubleSpinBox(widget);
    m_configWidgets[configIndex].aimSmoothingXSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].aimSmoothingXSpin->setDecimals(2);
    basicLayout->addWidget(m_configWidgets[configIndex].aimSmoothingXSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("Y平滑:"), widget), row, 2);
    m_configWidgets[configIndex].aimSmoothingYSpin = new Fluent::FluentDoubleSpinBox(widget);
    m_configWidgets[configIndex].aimSmoothingYSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].aimSmoothingYSpin->setDecimals(2);
    basicLayout->addWidget(m_configWidgets[configIndex].aimSmoothingYSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("Y偏移(%):"), widget), row, 0);
    m_configWidgets[configIndex].targetYOffsetSpin = new Fluent::FluentDoubleSpinBox(widget);
    m_configWidgets[configIndex].targetYOffsetSpin->setRange(-50.0, 50.0);
    basicLayout->addWidget(m_configWidgets[configIndex].targetYOffsetSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("最大移动:"), widget), row, 2);
    m_configWidgets[configIndex].maxPixelMoveSpin = new Fluent::FluentDoubleSpinBox(widget);
    m_configWidgets[configIndex].maxPixelMoveSpin->setRange(0.0, 500.0);
    basicLayout->addWidget(m_configWidgets[configIndex].maxPixelMoveSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("死区:"), widget), row, 0);
    m_configWidgets[configIndex].deadZonePixelsSpin = new Fluent::FluentDoubleSpinBox(widget);
    m_configWidgets[configIndex].deadZonePixelsSpin->setRange(0.0, 50.0);
    m_configWidgets[configIndex].deadZonePixelsSpin->setDecimals(1);
    basicLayout->addWidget(m_configWidgets[configIndex].deadZonePixelsSpin, row, 1);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕偏移X:"), widget), row, 0);
    m_configWidgets[configIndex].screenOffsetXSpin = new Fluent::FluentSpinBox(widget);
    m_configWidgets[configIndex].screenOffsetXSpin->setRange(0, 3840);
    basicLayout->addWidget(m_configWidgets[configIndex].screenOffsetXSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕偏移Y:"), widget), row, 2);
    m_configWidgets[configIndex].screenOffsetYSpin = new Fluent::FluentSpinBox(widget);
    m_configWidgets[configIndex].screenOffsetYSpin->setRange(0, 2160);
    basicLayout->addWidget(m_configWidgets[configIndex].screenOffsetYSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕宽度:"), widget), row, 0);
    m_configWidgets[configIndex].screenWidthSpin = new Fluent::FluentSpinBox(widget);
    m_configWidgets[configIndex].screenWidthSpin->setRange(640, 3840);
    m_configWidgets[configIndex].screenWidthSpin->setValue(1920);
    basicLayout->addWidget(m_configWidgets[configIndex].screenWidthSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕高度:"), widget), row, 2);
    m_configWidgets[configIndex].screenHeightSpin = new Fluent::FluentSpinBox(widget);
    m_configWidgets[configIndex].screenHeightSpin->setRange(480, 2160);
    m_configWidgets[configIndex].screenHeightSpin->setValue(1080);
    basicLayout->addWidget(m_configWidgets[configIndex].screenHeightSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("Y轴解锁:"), widget), row, 0);
    m_configWidgets[configIndex].enableYAxisUnlockCheck = new Fluent::FluentCheckBox(widget);
    basicLayout->addWidget(m_configWidgets[configIndex].enableYAxisUnlockCheck, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("解锁延迟(ms):"), widget), row, 2);
    m_configWidgets[configIndex].yAxisUnlockDelaySpin = new Fluent::FluentSpinBox(widget);
    m_configWidgets[configIndex].yAxisUnlockDelaySpin->setRange(100, 2000);
    basicLayout->addWidget(m_configWidgets[configIndex].yAxisUnlockDelaySpin, row, 3);
    
    layout->addWidget(basicGroup);
    layout->addStretch();
    
    return widget;
}

QGroupBox* YoloAimSettingsDialog::createAutoTriggerGroup(int configIndex)
{
    QGroupBox* group = new Fluent::FluentGroupBox(QStringLiteral("🎯 自动扳机"), this);
    group->setCheckable(true);
    group->setChecked(false);
    
    QGridLayout* layout = new QGridLayout(group);
    layout->setContentsMargins(16, 24, 16, 16);
    layout->setHorizontalSpacing(16);
    layout->setVerticalSpacing(12);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("触发半径:"), this), row, 0);
    m_configWidgets[configIndex].triggerRadiusSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerRadiusSpin->setRange(1, 50);
    layout->addWidget(m_configWidgets[configIndex].triggerRadiusSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("冷却时间(ms):"), this), row, 2);
    m_configWidgets[configIndex].triggerCooldownSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerCooldownSpin->setRange(50, 1000);
    layout->addWidget(m_configWidgets[configIndex].triggerCooldownSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("开火延迟(ms):"), this), row, 0);
    m_configWidgets[configIndex].triggerFireDelaySpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerFireDelaySpin->setRange(0, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerFireDelaySpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("开火时长(ms):"), this), row, 2);
    m_configWidgets[configIndex].triggerFireDurationSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerFireDurationSpin->setRange(10, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerFireDurationSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("间隔(ms):"), this), row, 0);
    m_configWidgets[configIndex].triggerIntervalSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerIntervalSpin->setRange(10, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerIntervalSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("移动补偿:"), this), row, 2);
    m_configWidgets[configIndex].triggerMoveCompensationSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerMoveCompensationSpin->setRange(0, 100);
    layout->addWidget(m_configWidgets[configIndex].triggerMoveCompensationSpin, row, 3);
    row++;
    
    // 随机延迟
    m_configWidgets[configIndex].enableTriggerDelayRandomCheck = new Fluent::FluentCheckBox(QStringLiteral("启用随机延迟"), this);
    layout->addWidget(m_configWidgets[configIndex].enableTriggerDelayRandomCheck, row, 0, 1, 4);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("延迟最小(ms):"), this), row, 0);
    m_configWidgets[configIndex].triggerDelayRandomMinSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerDelayRandomMinSpin->setRange(0, 1000);
    layout->addWidget(m_configWidgets[configIndex].triggerDelayRandomMinSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("延迟最大(ms):"), this), row, 2);
    m_configWidgets[configIndex].triggerDelayRandomMaxSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerDelayRandomMaxSpin->setRange(0, 1000);
    layout->addWidget(m_configWidgets[configIndex].triggerDelayRandomMaxSpin, row, 3);
    row++;
    
    // 随机时长
    m_configWidgets[configIndex].enableTriggerDurationRandomCheck = new Fluent::FluentCheckBox(QStringLiteral("启用随机时长"), this);
    layout->addWidget(m_configWidgets[configIndex].enableTriggerDurationRandomCheck, row, 0, 1, 4);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("时长最小(ms):"), this), row, 0);
    m_configWidgets[configIndex].triggerDurationRandomMinSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerDurationRandomMinSpin->setRange(0, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerDurationRandomMinSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("时长最大(ms):"), this), row, 2);
    m_configWidgets[configIndex].triggerDurationRandomMaxSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].triggerDurationRandomMaxSpin->setRange(0, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerDurationRandomMaxSpin, row, 3);
    
    return group;
}

QGroupBox* YoloAimSettingsDialog::createRecoilGroup(int configIndex)
{
    QGroupBox* group = new Fluent::FluentGroupBox(QStringLiteral("💥 后坐力控制"), this);
    group->setCheckable(true);
    group->setChecked(false);
    
    QGridLayout* layout = new QGridLayout(group);
    layout->setContentsMargins(16, 24, 16, 16);
    layout->setHorizontalSpacing(16);
    layout->setVerticalSpacing(12);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("强度:"), this), row, 0);
    m_configWidgets[configIndex].recoilStrengthSpin = new Fluent::FluentDoubleSpinBox(this);
    m_configWidgets[configIndex].recoilStrengthSpin->setRange(0.0, 50.0);
    layout->addWidget(m_configWidgets[configIndex].recoilStrengthSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("速度:"), this), row, 2);
    m_configWidgets[configIndex].recoilSpeedSpin = new Fluent::FluentSpinBox(this);
    m_configWidgets[configIndex].recoilSpeedSpin->setRange(1, 100);
    layout->addWidget(m_configWidgets[configIndex].recoilSpeedSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("PID增益比例:"), this), row, 0);
    m_configWidgets[configIndex].recoilPidGainScaleSpin = new Fluent::FluentDoubleSpinBox(this);
    m_configWidgets[configIndex].recoilPidGainScaleSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].recoilPidGainScaleSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].recoilPidGainScaleSpin, row, 1);
    
    return group;
}

QGroupBox* YoloAimSettingsDialog::createPredictorGroup(int configIndex)
{
    QGroupBox* group = new Fluent::FluentGroupBox(QStringLiteral("🔮 预测器"), this);
    group->setCheckable(true);
    group->setChecked(true);
    
    QGridLayout* layout = new QGridLayout(group);
    layout->setContentsMargins(16, 24, 16, 16);
    layout->setHorizontalSpacing(16);
    layout->setVerticalSpacing(12);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("X预测权重:"), this), row, 0);
    m_configWidgets[configIndex].predictionWeightXSpin = new Fluent::FluentDoubleSpinBox(this);
    m_configWidgets[configIndex].predictionWeightXSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].predictionWeightXSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].predictionWeightXSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("Y预测权重:"), this), row, 2);
    m_configWidgets[configIndex].predictionWeightYSpin = new Fluent::FluentDoubleSpinBox(this);
    m_configWidgets[configIndex].predictionWeightYSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].predictionWeightYSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].predictionWeightYSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("最大预测时间(s):"), this), row, 0);
    m_configWidgets[configIndex].maxPredictionTimeSpin = new Fluent::FluentDoubleSpinBox(this);
    m_configWidgets[configIndex].maxPredictionTimeSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].maxPredictionTimeSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].maxPredictionTimeSpin, row, 1);
    
    return group;
}

QGroupBox* YoloAimSettingsDialog::createBezierGroup(int configIndex)
{
    QGroupBox* group = new Fluent::FluentGroupBox(QStringLiteral("🌊 贝塞尔曲线"), this);
    group->setCheckable(true);
    group->setChecked(false);
    
    QGridLayout* layout = new QGridLayout(group);
    layout->setContentsMargins(16, 24, 16, 16);
    layout->setHorizontalSpacing(16);
    layout->setVerticalSpacing(12);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("曲率:"), this), row, 0);
    m_configWidgets[configIndex].bezierCurvatureSpin = new Fluent::FluentDoubleSpinBox(this);
    m_configWidgets[configIndex].bezierCurvatureSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].bezierCurvatureSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].bezierCurvatureSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("随机性:"), this), row, 2);
    m_configWidgets[configIndex].bezierRandomnessSpin = new Fluent::FluentDoubleSpinBox(this);
    m_configWidgets[configIndex].bezierRandomnessSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].bezierRandomnessSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].bezierRandomnessSpin, row, 3);
    
    return group;
}

QGroupBox* YoloAimSettingsDialog::createAdvancedPIDGroup(int configIndex)
{
    QGroupBox* group = new Fluent::FluentGroupBox(QStringLiteral("🎛️ 高级PID参数"), this);
    QVBoxLayout* mainLayout = new QVBoxLayout(group);
    mainLayout->setContentsMargins(12, 12, 12, 12);
    mainLayout->setSpacing(16);

    // 基础PID参数
    QGroupBox* basicPidGroup = new Fluent::FluentGroupBox(QStringLiteral("📊 基础PID"), group);
    QGridLayout* basicPidLayout = new QGridLayout(basicPidGroup);
    basicPidLayout->setContentsMargins(16, 24, 16, 16);
    basicPidLayout->setHorizontalSpacing(16);
    basicPidLayout->setVerticalSpacing(12);
    int row = 0;

    basicPidLayout->addWidget(new QLabel(QStringLiteral("P最小:"), group), row, 0);
    m_configWidgets[configIndex].pMinSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].pMinSpin->setRange(0.0, 2.0);
    m_configWidgets[configIndex].pMinSpin->setDecimals(3);
    m_configWidgets[configIndex].pMinSpin->setSingleStep(0.01);
    basicPidLayout->addWidget(m_configWidgets[configIndex].pMinSpin, row, 1);

    basicPidLayout->addWidget(new QLabel(QStringLiteral("P最大:"), group), row, 2);
    m_configWidgets[configIndex].pMaxSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].pMaxSpin->setRange(0.0, 5.0);
    m_configWidgets[configIndex].pMaxSpin->setDecimals(3);
    m_configWidgets[configIndex].pMaxSpin->setSingleStep(0.01);
    basicPidLayout->addWidget(m_configWidgets[configIndex].pMaxSpin, row, 3);
    row++;

    basicPidLayout->addWidget(new QLabel(QStringLiteral("P斜率:"), group), row, 0);
    m_configWidgets[configIndex].pSlopeSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].pSlopeSpin->setRange(0.0, 10.0);
    m_configWidgets[configIndex].pSlopeSpin->setDecimals(3);
    m_configWidgets[configIndex].pSlopeSpin->setSingleStep(0.01);
    basicPidLayout->addWidget(m_configWidgets[configIndex].pSlopeSpin, row, 1);

    basicPidLayout->addWidget(new QLabel(QStringLiteral("D:"), group), row, 2);
    m_configWidgets[configIndex].dSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].dSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].dSpin->setDecimals(4);
    m_configWidgets[configIndex].dSpin->setSingleStep(0.001);
    basicPidLayout->addWidget(m_configWidgets[configIndex].dSpin, row, 3);
    row++;

    basicPidLayout->addWidget(new QLabel(QStringLiteral("I:"), group), row, 0);
    m_configWidgets[configIndex].iSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].iSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].iSpin->setDecimals(4);
    m_configWidgets[configIndex].iSpin->setSingleStep(0.001);
    basicPidLayout->addWidget(m_configWidgets[configIndex].iSpin, row, 1);

    basicPidLayout->addWidget(new QLabel(QStringLiteral("D滤波α:"), group), row, 2);
    m_configWidgets[configIndex].derivativeFilterAlphaSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].derivativeFilterAlphaSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].derivativeFilterAlphaSpin->setDecimals(3);
    m_configWidgets[configIndex].derivativeFilterAlphaSpin->setSingleStep(0.01);
    basicPidLayout->addWidget(m_configWidgets[configIndex].derivativeFilterAlphaSpin, row, 3);

    mainLayout->addWidget(basicPidGroup);

    // 高级PID系数
    QGroupBox* advPidGroup = new Fluent::FluentGroupBox(QStringLiteral("⚡ 高级PID系数"), group);
    QGridLayout* advPidLayout = new QGridLayout(advPidGroup);
    advPidLayout->setContentsMargins(16, 24, 16, 16);
    advPidLayout->setHorizontalSpacing(16);
    advPidLayout->setVerticalSpacing(12);
    row = 0;

    advPidLayout->addWidget(new QLabel(QStringLiteral("目标阈值:"), group), row, 0);
    m_configWidgets[configIndex].advTargetThresholdSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].advTargetThresholdSpin->setRange(0.0, 200.0);
    m_configWidgets[configIndex].advTargetThresholdSpin->setDecimals(1);
    advPidLayout->addWidget(m_configWidgets[configIndex].advTargetThresholdSpin, row, 1);

    advPidLayout->addWidget(new QLabel(QStringLiteral("最小系数:"), group), row, 2);
    m_configWidgets[configIndex].advMinCoefficientSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].advMinCoefficientSpin->setRange(0.0, 10.0);
    m_configWidgets[configIndex].advMinCoefficientSpin->setDecimals(2);
    advPidLayout->addWidget(m_configWidgets[configIndex].advMinCoefficientSpin, row, 3);
    row++;

    advPidLayout->addWidget(new QLabel(QStringLiteral("最大系数:"), group), row, 0);
    m_configWidgets[configIndex].advMaxCoefficientSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].advMaxCoefficientSpin->setRange(0.0, 10.0);
    m_configWidgets[configIndex].advMaxCoefficientSpin->setDecimals(2);
    advPidLayout->addWidget(m_configWidgets[configIndex].advMaxCoefficientSpin, row, 1);

    advPidLayout->addWidget(new QLabel(QStringLiteral("过渡锐度:"), group), row, 2);
    m_configWidgets[configIndex].advTransitionSharpnessSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].advTransitionSharpnessSpin->setRange(0.0, 20.0);
    m_configWidgets[configIndex].advTransitionSharpnessSpin->setDecimals(1);
    advPidLayout->addWidget(m_configWidgets[configIndex].advTransitionSharpnessSpin, row, 3);
    row++;

    advPidLayout->addWidget(new QLabel(QStringLiteral("过渡中点:"), group), row, 0);
    m_configWidgets[configIndex].advTransitionMidpointSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].advTransitionMidpointSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].advTransitionMidpointSpin->setDecimals(2);
    advPidLayout->addWidget(m_configWidgets[configIndex].advTransitionMidpointSpin, row, 1);

    advPidLayout->addWidget(new QLabel(QStringLiteral("输出平滑:"), group), row, 2);
    m_configWidgets[configIndex].advOutputSmoothingSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].advOutputSmoothingSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].advOutputSmoothingSpin->setDecimals(2);
    advPidLayout->addWidget(m_configWidgets[configIndex].advOutputSmoothingSpin, row, 3);
    row++;

    advPidLayout->addWidget(new QLabel(QStringLiteral("速度因子:"), group), row, 0);
    m_configWidgets[configIndex].advSpeedFactorSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].advSpeedFactorSpin->setRange(0.0, 5.0);
    m_configWidgets[configIndex].advSpeedFactorSpin->setDecimals(2);
    advPidLayout->addWidget(m_configWidgets[configIndex].advSpeedFactorSpin, row, 1);

    mainLayout->addWidget(advPidGroup);

    // OneEuro滤波器
    QGroupBox* oneEuroGroup = new Fluent::FluentGroupBox(QStringLiteral("🔬 OneEuro滤波器"), group);
    QGridLayout* oneEuroLayout = new QGridLayout(oneEuroGroup);
    oneEuroLayout->setContentsMargins(16, 24, 16, 16);
    oneEuroLayout->setHorizontalSpacing(16);
    oneEuroLayout->setVerticalSpacing(12);
    row = 0;

    m_configWidgets[configIndex].useOneEuroFilterCheck = new Fluent::FluentCheckBox(QStringLiteral("启用OneEuro滤波"), group);
    oneEuroLayout->addWidget(m_configWidgets[configIndex].useOneEuroFilterCheck, row, 0, 1, 4);
    row++;

    oneEuroLayout->addWidget(new QLabel(QStringLiteral("最小截止:"), group), row, 0);
    m_configWidgets[configIndex].oneEuroMinCutoffSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].oneEuroMinCutoffSpin->setRange(0.0, 10.0);
    m_configWidgets[configIndex].oneEuroMinCutoffSpin->setDecimals(2);
    oneEuroLayout->addWidget(m_configWidgets[configIndex].oneEuroMinCutoffSpin, row, 1);

    oneEuroLayout->addWidget(new QLabel(QStringLiteral("Beta:"), group), row, 2);
    m_configWidgets[configIndex].oneEuroBetaSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].oneEuroBetaSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].oneEuroBetaSpin->setDecimals(2);
    oneEuroLayout->addWidget(m_configWidgets[configIndex].oneEuroBetaSpin, row, 3);
    row++;

    oneEuroLayout->addWidget(new QLabel(QStringLiteral("D截止:"), group), row, 0);
    m_configWidgets[configIndex].oneEuroDCutoffSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].oneEuroDCutoffSpin->setRange(0.0, 10.0);
    m_configWidgets[configIndex].oneEuroDCutoffSpin->setDecimals(2);
    oneEuroLayout->addWidget(m_configWidgets[configIndex].oneEuroDCutoffSpin, row, 1);

    mainLayout->addWidget(oneEuroGroup);

    // 积分控制
    QGroupBox* integralGroup = new Fluent::FluentGroupBox(QStringLiteral("📐 积分控制"), group);
    QGridLayout* integralLayout = new QGridLayout(integralGroup);
    integralLayout->setContentsMargins(16, 24, 16, 16);
    integralLayout->setHorizontalSpacing(16);
    integralLayout->setVerticalSpacing(12);
    row = 0;

    integralLayout->addWidget(new QLabel(QStringLiteral("积分限幅:"), group), row, 0);
    m_configWidgets[configIndex].integralLimitSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].integralLimitSpin->setRange(0.0, 500.0);
    m_configWidgets[configIndex].integralLimitSpin->setDecimals(1);
    integralLayout->addWidget(m_configWidgets[configIndex].integralLimitSpin, row, 1);

    integralLayout->addWidget(new QLabel(QStringLiteral("积分速率:"), group), row, 2);
    m_configWidgets[configIndex].integralRateSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].integralRateSpin->setRange(0.0, 2.0);
    m_configWidgets[configIndex].integralRateSpin->setDecimals(2);
    m_configWidgets[configIndex].integralRateSpin->setSingleStep(0.1);
    integralLayout->addWidget(m_configWidgets[configIndex].integralRateSpin, row, 3);

    mainLayout->addWidget(integralGroup);

    // P增益爬坡
    QGroupBox* rampGroup = new Fluent::FluentGroupBox(QStringLiteral("📈 P增益爬坡"), group);
    QGridLayout* rampLayout = new QGridLayout(rampGroup);
    rampLayout->setContentsMargins(16, 24, 16, 16);
    rampLayout->setHorizontalSpacing(16);
    rampLayout->setVerticalSpacing(12);
    row = 0;

    rampLayout->addWidget(new QLabel(QStringLiteral("初始缩放:"), group), row, 0);
    m_configWidgets[configIndex].pGainRampInitialScaleSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].pGainRampInitialScaleSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].pGainRampInitialScaleSpin->setDecimals(2);
    rampLayout->addWidget(m_configWidgets[configIndex].pGainRampInitialScaleSpin, row, 1);

    rampLayout->addWidget(new QLabel(QStringLiteral("爬坡时长(s):"), group), row, 2);
    m_configWidgets[configIndex].pGainRampDurationSpin = new Fluent::FluentDoubleSpinBox(group);
    m_configWidgets[configIndex].pGainRampDurationSpin->setRange(0.0, 5.0);
    m_configWidgets[configIndex].pGainRampDurationSpin->setDecimals(2);
    rampLayout->addWidget(m_configWidgets[configIndex].pGainRampDurationSpin, row, 3);

    mainLayout->addWidget(rampGroup);

    return group;
}

void YoloAimSettingsDialog::initConfigWidgetStruct(int configIndex)
{
    auto& w = m_configWidgets[configIndex];
    w.enabledCheck = nullptr;
    w.hotkeyCombo = nullptr;
    w.controllerTypeCombo = nullptr;
    w.pMinSpin = nullptr;
    w.pMaxSpin = nullptr;
    w.pSlopeSpin = nullptr;
    w.dSpin = nullptr;
    w.iSpin = nullptr;
    w.derivativeFilterAlphaSpin = nullptr;
    w.advTargetThresholdSpin = nullptr;
    w.advMinCoefficientSpin = nullptr;
    w.advMaxCoefficientSpin = nullptr;
    w.advTransitionSharpnessSpin = nullptr;
    w.advTransitionMidpointSpin = nullptr;
    w.advOutputSmoothingSpin = nullptr;
    w.advSpeedFactorSpin = nullptr;
    w.useOneEuroFilterCheck = nullptr;
    w.oneEuroMinCutoffSpin = nullptr;
    w.oneEuroBetaSpin = nullptr;
    w.oneEuroDCutoffSpin = nullptr;
    w.aimSmoothingXSpin = nullptr;
    w.aimSmoothingYSpin = nullptr;
    w.targetYOffsetSpin = nullptr;
    w.maxPixelMoveSpin = nullptr;
    w.deadZonePixelsSpin = nullptr;
    w.screenOffsetXSpin = nullptr;
    w.screenOffsetYSpin = nullptr;
    w.screenWidthSpin = nullptr;
    w.screenHeightSpin = nullptr;
    w.enableYAxisUnlockCheck = nullptr;
    w.yAxisUnlockDelaySpin = nullptr;
    w.autoTriggerGroup = nullptr;
    w.triggerRadiusSpin = nullptr;
    w.triggerCooldownSpin = nullptr;
    w.triggerFireDelaySpin = nullptr;
    w.triggerFireDurationSpin = nullptr;
    w.triggerIntervalSpin = nullptr;
    w.enableTriggerDelayRandomCheck = nullptr;
    w.triggerDelayRandomMinSpin = nullptr;
    w.triggerDelayRandomMaxSpin = nullptr;
    w.enableTriggerDurationRandomCheck = nullptr;
    w.triggerDurationRandomMinSpin = nullptr;
    w.triggerDurationRandomMaxSpin = nullptr;
    w.triggerMoveCompensationSpin = nullptr;
    w.recoilGroup = nullptr;
    w.recoilStrengthSpin = nullptr;
    w.recoilSpeedSpin = nullptr;
    w.recoilPidGainScaleSpin = nullptr;
    w.integralLimitSpin = nullptr;
    w.integralRateSpin = nullptr;
    w.pGainRampInitialScaleSpin = nullptr;
    w.pGainRampDurationSpin = nullptr;
    w.predictorGroup = nullptr;
    w.predictionWeightXSpin = nullptr;
    w.predictionWeightYSpin = nullptr;
    w.maxPredictionTimeSpin = nullptr;
    w.bezierGroup = nullptr;
    w.bezierCurvatureSpin = nullptr;
    w.bezierRandomnessSpin = nullptr;
}

void YoloAimSettingsDialog::initAllConfigWidgetStructs()
{
    for (int i = 0; i < 5; i++) {
        initConfigWidgetStruct(i);
        m_mouseConfigContainers[i] = nullptr;
        m_trackingConfigContainers[i] = nullptr;
    }
}

void YoloAimSettingsDialog::switchConfigVisibility()
{
    for (int i = 0; i < 5; i++) {
        if (m_mouseConfigContainers[i])
            m_mouseConfigContainers[i]->setVisible(i == m_currentConfig);
        if (m_trackingConfigContainers[i])
            m_trackingConfigContainers[i]->setVisible(i == m_currentConfig);
    }
}

void YoloAimSettingsDialog::refreshSourceList()
{
    m_sourceCombo->clear();
    m_sourceCombo->addItem(QStringLiteral("(无)"), QString());
    
    obs_enum_sources([](void* data, obs_source_t* source) {
        uint32_t flags = obs_source_get_output_flags(source);
        if (flags & OBS_SOURCE_VIDEO) {
            QComboBox* combo = static_cast<QComboBox*>(data);
            QString name = QString::fromUtf8(obs_source_get_name(source));
            combo->addItem(name, name);
        }
        return true;
    }, m_sourceCombo);
}

void YoloAimSettingsDialog::onSourceChanged(int index)
{
    QString newSource = m_sourceCombo->currentData().toString();
    
    if (!m_currentSource.isEmpty()) {
        detachFilterFromSource(m_currentSource);
    }
    
    if (!newSource.isEmpty()) {
        attachFilterToSource(newSource);
    }
    
    m_currentSource = newSource;
    loadSettings();
}

void YoloAimSettingsDialog::onConfigChanged(int index)
{
    // 切换前先保存当前配置，避免丢失未保存的修改
    if (index != m_currentConfig) {
        saveSettings();
    }
    m_currentConfig = index;
    switchConfigVisibility();
    loadSettings();
}

void YoloAimSettingsDialog::onPageChanged(int index)
{
    updateVisibility();
}

void YoloAimSettingsDialog::onSaveClicked()
{
    saveSettings();
    applySettings();
    accept();
}

void YoloAimSettingsDialog::onResetClicked()
{
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        QStringLiteral("重置设置"),
        QStringLiteral("确定要重置所有设置吗？"),
        QMessageBox::Yes | QMessageBox::No
    );
    
    if (reply == QMessageBox::Yes) {
        loadSettings();
    }
}

void YoloAimSettingsDialog::onApplyClicked()
{
    saveSettings();
    applySettings();
}

void YoloAimSettingsDialog::loadSettings()
{
    if (m_currentSource.isEmpty()) return;
    
    obs_source_t* source = obs_get_source_by_name(m_currentSource.toUtf8().constData());
    if (!source) return;
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "visual-assist-hidden");
    if (!filter) {
        obs_source_release(source);
        return;
    }
    
    obs_data_t* settings = obs_source_get_settings(filter);
    if (!settings) {
        obs_source_release(filter);
        obs_source_release(source);
        return;
    }
    
    int idx = m_currentConfig;
    auto& w = m_configWidgets[idx];
    
    if (w.enabledCheck) w.enabledCheck->setChecked(obs_data_get_bool(settings, QString("enable_config_%1").arg(idx).toUtf8().constData()));
    if (w.hotkeyCombo) w.hotkeyCombo->setCurrentIndex(obs_data_get_int(settings, QString("hotkey_%1").arg(idx).toUtf8().constData()));
    if (w.controllerTypeCombo) w.controllerTypeCombo->setCurrentIndex(obs_data_get_int(settings, QString("controller_type_%1").arg(idx).toUtf8().constData()));
    
    if (w.pMinSpin) w.pMinSpin->setValue(obs_data_get_double(settings, QString("p_min_%1").arg(idx).toUtf8().constData()));
    if (w.pMaxSpin) w.pMaxSpin->setValue(obs_data_get_double(settings, QString("p_max_%1").arg(idx).toUtf8().constData()));
    if (w.pSlopeSpin) w.pSlopeSpin->setValue(obs_data_get_double(settings, QString("p_slope_%1").arg(idx).toUtf8().constData()));
    if (w.dSpin) w.dSpin->setValue(obs_data_get_double(settings, QString("d_%1").arg(idx).toUtf8().constData()));
    if (w.iSpin) w.iSpin->setValue(obs_data_get_double(settings, QString("i_%1").arg(idx).toUtf8().constData()));
    if (w.derivativeFilterAlphaSpin) w.derivativeFilterAlphaSpin->setValue(obs_data_get_double(settings, QString("derivative_filter_alpha_%1").arg(idx).toUtf8().constData()));
    
    if (w.advTargetThresholdSpin) w.advTargetThresholdSpin->setValue(obs_data_get_double(settings, QString("adv_target_threshold_%1").arg(idx).toUtf8().constData()));
    if (w.advMinCoefficientSpin) w.advMinCoefficientSpin->setValue(obs_data_get_double(settings, QString("adv_min_coefficient_%1").arg(idx).toUtf8().constData()));
    if (w.advMaxCoefficientSpin) w.advMaxCoefficientSpin->setValue(obs_data_get_double(settings, QString("adv_max_coefficient_%1").arg(idx).toUtf8().constData()));
    if (w.advTransitionSharpnessSpin) w.advTransitionSharpnessSpin->setValue(obs_data_get_double(settings, QString("adv_transition_sharpness_%1").arg(idx).toUtf8().constData()));
    if (w.advTransitionMidpointSpin) w.advTransitionMidpointSpin->setValue(obs_data_get_double(settings, QString("adv_transition_midpoint_%1").arg(idx).toUtf8().constData()));
    if (w.advOutputSmoothingSpin) w.advOutputSmoothingSpin->setValue(obs_data_get_double(settings, QString("adv_output_smoothing_%1").arg(idx).toUtf8().constData()));
    if (w.advSpeedFactorSpin) w.advSpeedFactorSpin->setValue(obs_data_get_double(settings, QString("adv_speed_factor_%1").arg(idx).toUtf8().constData()));
    
    if (w.useOneEuroFilterCheck) w.useOneEuroFilterCheck->setChecked(obs_data_get_bool(settings, QString("use_one_euro_filter_%1").arg(idx).toUtf8().constData()));
    if (w.oneEuroMinCutoffSpin) w.oneEuroMinCutoffSpin->setValue(obs_data_get_double(settings, QString("one_euro_min_cutoff_%1").arg(idx).toUtf8().constData()));
    if (w.oneEuroBetaSpin) w.oneEuroBetaSpin->setValue(obs_data_get_double(settings, QString("one_euro_beta_%1").arg(idx).toUtf8().constData()));
    if (w.oneEuroDCutoffSpin) w.oneEuroDCutoffSpin->setValue(obs_data_get_double(settings, QString("one_euro_d_cutoff_%1").arg(idx).toUtf8().constData()));
    
    // 积分控制
    if (w.integralLimitSpin) w.integralLimitSpin->setValue(obs_data_get_double(settings, QString("integral_limit_%1").arg(idx).toUtf8().constData()));
    if (w.integralRateSpin) w.integralRateSpin->setValue(obs_data_get_double(settings, QString("integral_rate_%1").arg(idx).toUtf8().constData()));
    
    // P增益爬坡
    if (w.pGainRampInitialScaleSpin) w.pGainRampInitialScaleSpin->setValue(obs_data_get_double(settings, QString("p_gain_ramp_initial_scale_%1").arg(idx).toUtf8().constData()));
    if (w.pGainRampDurationSpin) w.pGainRampDurationSpin->setValue(obs_data_get_double(settings, QString("p_gain_ramp_duration_%1").arg(idx).toUtf8().constData()));
    
    if (w.aimSmoothingXSpin) w.aimSmoothingXSpin->setValue(obs_data_get_double(settings, QString("aim_smoothing_x_%1").arg(idx).toUtf8().constData()));
    if (w.aimSmoothingYSpin) w.aimSmoothingYSpin->setValue(obs_data_get_double(settings, QString("aim_smoothing_y_%1").arg(idx).toUtf8().constData()));
    if (w.targetYOffsetSpin) w.targetYOffsetSpin->setValue(obs_data_get_double(settings, QString("target_y_offset_%1").arg(idx).toUtf8().constData()));
    if (w.maxPixelMoveSpin) w.maxPixelMoveSpin->setValue(obs_data_get_double(settings, QString("max_pixel_move_%1").arg(idx).toUtf8().constData()));
    if (w.deadZonePixelsSpin) w.deadZonePixelsSpin->setValue(obs_data_get_double(settings, QString("dead_zone_pixels_%1").arg(idx).toUtf8().constData()));
    
    if (w.screenOffsetXSpin) w.screenOffsetXSpin->setValue(obs_data_get_int(settings, QString("screen_offset_x_%1").arg(idx).toUtf8().constData()));
    if (w.screenOffsetYSpin) w.screenOffsetYSpin->setValue(obs_data_get_int(settings, QString("screen_offset_y_%1").arg(idx).toUtf8().constData()));
    if (w.screenWidthSpin) w.screenWidthSpin->setValue(obs_data_get_int(settings, QString("screen_width_%1").arg(idx).toUtf8().constData()));
    if (w.screenHeightSpin) w.screenHeightSpin->setValue(obs_data_get_int(settings, QString("screen_height_%1").arg(idx).toUtf8().constData()));
    
    if (w.enableYAxisUnlockCheck) w.enableYAxisUnlockCheck->setChecked(obs_data_get_bool(settings, QString("enable_y_axis_unlock_%1").arg(idx).toUtf8().constData()));
    if (w.yAxisUnlockDelaySpin) w.yAxisUnlockDelaySpin->setValue(obs_data_get_int(settings, QString("y_axis_unlock_delay_%1").arg(idx).toUtf8().constData()));
    
    if (w.autoTriggerGroup) w.autoTriggerGroup->setChecked(obs_data_get_bool(settings, QString("auto_trigger_group_%1").arg(idx).toUtf8().constData()));
    if (w.triggerRadiusSpin) w.triggerRadiusSpin->setValue(obs_data_get_int(settings, QString("trigger_radius_%1").arg(idx).toUtf8().constData()));
    if (w.triggerCooldownSpin) w.triggerCooldownSpin->setValue(obs_data_get_int(settings, QString("trigger_cooldown_%1").arg(idx).toUtf8().constData()));
    if (w.triggerFireDelaySpin) w.triggerFireDelaySpin->setValue(obs_data_get_int(settings, QString("trigger_fire_delay_%1").arg(idx).toUtf8().constData()));
    if (w.triggerFireDurationSpin) w.triggerFireDurationSpin->setValue(obs_data_get_int(settings, QString("trigger_fire_duration_%1").arg(idx).toUtf8().constData()));
    if (w.triggerIntervalSpin) w.triggerIntervalSpin->setValue(obs_data_get_int(settings, QString("trigger_interval_%1").arg(idx).toUtf8().constData()));
    if (w.triggerMoveCompensationSpin) w.triggerMoveCompensationSpin->setValue(obs_data_get_int(settings, QString("trigger_move_compensation_%1").arg(idx).toUtf8().constData()));
    
    // 随机触发延迟
    if (w.enableTriggerDelayRandomCheck) w.enableTriggerDelayRandomCheck->setChecked(obs_data_get_bool(settings, QString("enable_trigger_delay_random_%1").arg(idx).toUtf8().constData()));
    if (w.triggerDelayRandomMinSpin) w.triggerDelayRandomMinSpin->setValue(obs_data_get_int(settings, QString("trigger_delay_random_min_%1").arg(idx).toUtf8().constData()));
    if (w.triggerDelayRandomMaxSpin) w.triggerDelayRandomMaxSpin->setValue(obs_data_get_int(settings, QString("trigger_delay_random_max_%1").arg(idx).toUtf8().constData()));
    
    // 随机触发持续时间
    if (w.enableTriggerDurationRandomCheck) w.enableTriggerDurationRandomCheck->setChecked(obs_data_get_bool(settings, QString("enable_trigger_duration_random_%1").arg(idx).toUtf8().constData()));
    if (w.triggerDurationRandomMinSpin) w.triggerDurationRandomMinSpin->setValue(obs_data_get_int(settings, QString("trigger_duration_random_min_%1").arg(idx).toUtf8().constData()));
    if (w.triggerDurationRandomMaxSpin) w.triggerDurationRandomMaxSpin->setValue(obs_data_get_int(settings, QString("trigger_duration_random_max_%1").arg(idx).toUtf8().constData()));
    
    if (w.recoilGroup) w.recoilGroup->setChecked(obs_data_get_bool(settings, QString("recoil_group_%1").arg(idx).toUtf8().constData()));
    if (w.recoilStrengthSpin) w.recoilStrengthSpin->setValue(obs_data_get_double(settings, QString("recoil_strength_%1").arg(idx).toUtf8().constData()));
    if (w.recoilSpeedSpin) w.recoilSpeedSpin->setValue(obs_data_get_int(settings, QString("recoil_speed_%1").arg(idx).toUtf8().constData()));
    if (w.recoilPidGainScaleSpin) w.recoilPidGainScaleSpin->setValue(obs_data_get_double(settings, QString("recoil_pid_gain_scale_%1").arg(idx).toUtf8().constData()));
    
    if (w.predictorGroup) w.predictorGroup->setChecked(obs_data_get_bool(settings, QString("derivative_predictor_group_%1").arg(idx).toUtf8().constData()));
    if (w.predictionWeightXSpin) w.predictionWeightXSpin->setValue(obs_data_get_double(settings, QString("prediction_weight_x_%1").arg(idx).toUtf8().constData()));
    if (w.predictionWeightYSpin) w.predictionWeightYSpin->setValue(obs_data_get_double(settings, QString("prediction_weight_y_%1").arg(idx).toUtf8().constData()));
    if (w.maxPredictionTimeSpin) w.maxPredictionTimeSpin->setValue(obs_data_get_double(settings, QString("max_prediction_time_%1").arg(idx).toUtf8().constData()));
    
    if (w.bezierGroup) w.bezierGroup->setChecked(obs_data_get_bool(settings, QString("bezier_movement_group_%1").arg(idx).toUtf8().constData()));
    if (w.bezierCurvatureSpin) w.bezierCurvatureSpin->setValue(obs_data_get_double(settings, QString("bezier_curvature_%1").arg(idx).toUtf8().constData()));
    if (w.bezierRandomnessSpin) w.bezierRandomnessSpin->setValue(obs_data_get_double(settings, QString("bezier_randomness_%1").arg(idx).toUtf8().constData()));
    
    if (m_showDetectionResultsCheck) m_showDetectionResultsCheck->setChecked(obs_data_get_bool(settings, "show_detection_results"));
    if (m_showFOVCheck) m_showFOVCheck->setChecked(obs_data_get_bool(settings, "show_fov"));
    if (m_showFOVCircleCheck) m_showFOVCircleCheck->setChecked(obs_data_get_bool(settings, "show_fov_circle"));
    if (m_showFOVCrossCheck) m_showFOVCrossCheck->setChecked(obs_data_get_bool(settings, "show_fov_cross"));
    if (m_fovRadiusSpin) m_fovRadiusSpin->setValue(obs_data_get_int(settings, "fov_radius"));
    if (m_fovCrossLineScaleSpin) m_fovCrossLineScaleSpin->setValue(obs_data_get_int(settings, "fov_cross_line_scale"));
    if (m_fovCrossLineThicknessSpin) m_fovCrossLineThicknessSpin->setValue(obs_data_get_int(settings, "fov_cross_line_thickness"));
    if (m_fovCircleThicknessSpin) m_fovCircleThicknessSpin->setValue(obs_data_get_int(settings, "fov_circle_thickness"));
    
    // 推理状态
    bool isInferencing = obs_data_get_bool(settings, "is_inferencing");
    if (m_toggleInferenceBtn) {
        m_toggleInferenceBtn->setChecked(isInferencing);
        m_toggleInferenceBtn->setText(isInferencing ? QStringLiteral("⏹ 停止推理") : QStringLiteral("▶ 开始推理"));
    }
    if (m_inferenceStatusLabel) {
        m_inferenceStatusLabel->setText(isInferencing ? QStringLiteral("状态: 推理运行中") : QStringLiteral("状态: 已停止"));
    }
    
    // 模型配置
    if (m_modelPathEdit) m_modelPathEdit->setText(QString::fromUtf8(obs_data_get_string(settings, "model_path")));
    if (m_modelVersionCombo) m_modelVersionCombo->setCurrentIndex(obs_data_get_int(settings, "model_version"));
    if (m_useGPUCombo) {
        QString gpuSetting = QString::fromUtf8(obs_data_get_string(settings, "use_gpu"));
        int gpuIndex = m_useGPUCombo->findData(gpuSetting);
        if (gpuIndex >= 0) m_useGPUCombo->setCurrentIndex(gpuIndex);
    }
    if (m_useGPUTextureCheck) m_useGPUTextureCheck->setChecked(obs_data_get_bool(settings, "use_gpu_texture_inference"));
    if (m_inputResolutionCombo) {
        int res = obs_data_get_int(settings, "input_resolution");
        int resIndex = m_inputResolutionCombo->findData(res);
        if (resIndex >= 0) m_inputResolutionCombo->setCurrentIndex(resIndex);
    }
    if (m_numThreadsSpin) m_numThreadsSpin->setValue(obs_data_get_int(settings, "num_threads"));
    
    // 检测配置
    if (m_confidenceThresholdSpin) m_confidenceThresholdSpin->setValue(obs_data_get_double(settings, "confidence_threshold"));
    if (m_nmsThresholdSpin) m_nmsThresholdSpin->setValue(obs_data_get_double(settings, "nms_threshold"));
    if (m_targetClassCombo) m_targetClassCombo->setCurrentIndex(obs_data_get_int(settings, "target_class"));
    if (m_targetClassesTextEdit) m_targetClassesTextEdit->setText(QString::fromUtf8(obs_data_get_string(settings, "target_classes_text")));
    if (m_inferenceIntervalSpin) m_inferenceIntervalSpin->setValue(obs_data_get_int(settings, "inference_interval_frames"));
    
    // 区域检测
    if (m_useRegionCheck) m_useRegionCheck->setChecked(obs_data_get_bool(settings, "use_region"));
    if (m_regionXSpin) m_regionXSpin->setValue(obs_data_get_int(settings, "region_x"));
    if (m_regionYSpin) m_regionYSpin->setValue(obs_data_get_int(settings, "region_y"));
    if (m_regionWidthSpin) m_regionWidthSpin->setValue(obs_data_get_int(settings, "region_width"));
    if (m_regionHeightSpin) m_regionHeightSpin->setValue(obs_data_get_int(settings, "region_height"));
    
    // 渲染配置
    if (m_bboxLineWidthSpin) m_bboxLineWidthSpin->setValue(obs_data_get_int(settings, "bbox_line_width"));
    if (m_labelFontScaleSpin) m_labelFontScaleSpin->setValue(obs_data_get_double(settings, "label_font_scale"));
    
    // 动态FOV
    if (m_useDynamicFOVCheck) m_useDynamicFOVCheck->setChecked(obs_data_get_bool(settings, "use_dynamic_fov"));
    if (m_showFOV2Check) m_showFOV2Check->setChecked(obs_data_get_bool(settings, "show_fov2"));
    if (m_fovRadius2Spin) m_fovRadius2Spin->setValue(obs_data_get_int(settings, "fov_radius2"));
    if (m_dynamicFovShrinkSpin) m_dynamicFovShrinkSpin->setValue(obs_data_get_int(settings, "dynamic_fov_shrink_percent"));
    if (m_dynamicFovTransitionSpin) m_dynamicFovTransitionSpin->setValue(obs_data_get_int(settings, "dynamic_fov_transition_time"));
    
    // 检测框平滑
    if (m_detectionSmoothingCheck) m_detectionSmoothingCheck->setChecked(obs_data_get_bool(settings, "detection_smoothing_enabled"));
    if (m_detectionSmoothingAlphaSpin) m_detectionSmoothingAlphaSpin->setValue(obs_data_get_double(settings, "detection_smoothing_alpha"));
    
    // KalmanFilter 追踪设置
    if (m_useKalmanTrackerCheck) m_useKalmanTrackerCheck->setChecked(obs_data_get_bool(settings, "use_kalman_tracker"));
    if (m_kalmanGenerateThresholdSpin) m_kalmanGenerateThresholdSpin->setValue(obs_data_get_int(settings, "kalman_generate_threshold"));
    if (m_kalmanTerminateCountSpin) m_kalmanTerminateCountSpin->setValue(obs_data_get_int(settings, "kalman_terminate_count"));
    
    // 神经网络轨迹生成器设置
    if (m_enableNeuralPathCheck) m_enableNeuralPathCheck->setChecked(obs_data_get_bool(settings, "enable_neural_path"));
    if (m_neuralPathPointsSpin) m_neuralPathPointsSpin->setValue(obs_data_get_int(settings, "neural_path_points"));
    if (m_neuralMouseStepSizeSpin) m_neuralMouseStepSizeSpin->setValue(obs_data_get_double(settings, "neural_mouse_step_size"));
    if (m_neuralTargetRadiusSpin) m_neuralTargetRadiusSpin->setValue(obs_data_get_int(settings, "neural_target_radius"));
    if (m_neuralConsumePerFrameSpin) m_neuralConsumePerFrameSpin->setValue(obs_data_get_int(settings, "neural_consume_per_frame"));
    
    // 高级配置
    if (m_exportCoordinatesCheck) m_exportCoordinatesCheck->setChecked(obs_data_get_bool(settings, "export_coordinates"));
    if (m_coordinateOutputPathEdit) m_coordinateOutputPathEdit->setText(QString::fromUtf8(obs_data_get_string(settings, "coordinate_output_path")));
    
    obs_data_release(settings);
    obs_source_release(filter);
    obs_source_release(source);
}

void YoloAimSettingsDialog::saveSettings()
{
    if (m_currentSource.isEmpty()) return;
    
    obs_source_t* source = obs_get_source_by_name(m_currentSource.toUtf8().constData());
    if (!source) return;
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "visual-assist-hidden");
    if (!filter) {
        obs_source_release(source);
        return;
    }
    
    obs_data_t* settings = obs_source_get_settings(filter);
    if (!settings) {
        obs_source_release(filter);
        obs_source_release(source);
        return;
    }
    
    int idx = m_currentConfig;
    auto& w = m_configWidgets[idx];
    
    if (w.enabledCheck) obs_data_set_bool(settings, QString("enable_config_%1").arg(idx).toUtf8().constData(), w.enabledCheck->isChecked());
    if (w.hotkeyCombo) obs_data_set_int(settings, QString("hotkey_%1").arg(idx).toUtf8().constData(), w.hotkeyCombo->currentIndex());
    if (w.controllerTypeCombo) obs_data_set_int(settings, QString("controller_type_%1").arg(idx).toUtf8().constData(), w.controllerTypeCombo->currentIndex());
    
    if (w.pMinSpin) obs_data_set_double(settings, QString("p_min_%1").arg(idx).toUtf8().constData(), w.pMinSpin->value());
    if (w.pMaxSpin) obs_data_set_double(settings, QString("p_max_%1").arg(idx).toUtf8().constData(), w.pMaxSpin->value());
    if (w.pSlopeSpin) obs_data_set_double(settings, QString("p_slope_%1").arg(idx).toUtf8().constData(), w.pSlopeSpin->value());
    if (w.dSpin) obs_data_set_double(settings, QString("d_%1").arg(idx).toUtf8().constData(), w.dSpin->value());
    if (w.iSpin) obs_data_set_double(settings, QString("i_%1").arg(idx).toUtf8().constData(), w.iSpin->value());
    if (w.derivativeFilterAlphaSpin) obs_data_set_double(settings, QString("derivative_filter_alpha_%1").arg(idx).toUtf8().constData(), w.derivativeFilterAlphaSpin->value());
    
    if (w.advTargetThresholdSpin) obs_data_set_double(settings, QString("adv_target_threshold_%1").arg(idx).toUtf8().constData(), w.advTargetThresholdSpin->value());
    if (w.advMinCoefficientSpin) obs_data_set_double(settings, QString("adv_min_coefficient_%1").arg(idx).toUtf8().constData(), w.advMinCoefficientSpin->value());
    if (w.advMaxCoefficientSpin) obs_data_set_double(settings, QString("adv_max_coefficient_%1").arg(idx).toUtf8().constData(), w.advMaxCoefficientSpin->value());
    if (w.advTransitionSharpnessSpin) obs_data_set_double(settings, QString("adv_transition_sharpness_%1").arg(idx).toUtf8().constData(), w.advTransitionSharpnessSpin->value());
    if (w.advTransitionMidpointSpin) obs_data_set_double(settings, QString("adv_transition_midpoint_%1").arg(idx).toUtf8().constData(), w.advTransitionMidpointSpin->value());
    if (w.advOutputSmoothingSpin) obs_data_set_double(settings, QString("adv_output_smoothing_%1").arg(idx).toUtf8().constData(), w.advOutputSmoothingSpin->value());
    if (w.advSpeedFactorSpin) obs_data_set_double(settings, QString("adv_speed_factor_%1").arg(idx).toUtf8().constData(), w.advSpeedFactorSpin->value());
    
    if (w.useOneEuroFilterCheck) obs_data_set_bool(settings, QString("use_one_euro_filter_%1").arg(idx).toUtf8().constData(), w.useOneEuroFilterCheck->isChecked());
    if (w.oneEuroMinCutoffSpin) obs_data_set_double(settings, QString("one_euro_min_cutoff_%1").arg(idx).toUtf8().constData(), w.oneEuroMinCutoffSpin->value());
    if (w.oneEuroBetaSpin) obs_data_set_double(settings, QString("one_euro_beta_%1").arg(idx).toUtf8().constData(), w.oneEuroBetaSpin->value());
    if (w.oneEuroDCutoffSpin) obs_data_set_double(settings, QString("one_euro_d_cutoff_%1").arg(idx).toUtf8().constData(), w.oneEuroDCutoffSpin->value());
    
    // 积分控制
    if (w.integralLimitSpin) obs_data_set_double(settings, QString("integral_limit_%1").arg(idx).toUtf8().constData(), w.integralLimitSpin->value());
    if (w.integralRateSpin) obs_data_set_double(settings, QString("integral_rate_%1").arg(idx).toUtf8().constData(), w.integralRateSpin->value());
    
    // P增益爬坡
    if (w.pGainRampInitialScaleSpin) obs_data_set_double(settings, QString("p_gain_ramp_initial_scale_%1").arg(idx).toUtf8().constData(), w.pGainRampInitialScaleSpin->value());
    if (w.pGainRampDurationSpin) obs_data_set_double(settings, QString("p_gain_ramp_duration_%1").arg(idx).toUtf8().constData(), w.pGainRampDurationSpin->value());
    
    if (w.aimSmoothingXSpin) obs_data_set_double(settings, QString("aim_smoothing_x_%1").arg(idx).toUtf8().constData(), w.aimSmoothingXSpin->value());
    if (w.aimSmoothingYSpin) obs_data_set_double(settings, QString("aim_smoothing_y_%1").arg(idx).toUtf8().constData(), w.aimSmoothingYSpin->value());
    if (w.targetYOffsetSpin) obs_data_set_double(settings, QString("target_y_offset_%1").arg(idx).toUtf8().constData(), w.targetYOffsetSpin->value());
    if (w.maxPixelMoveSpin) obs_data_set_double(settings, QString("max_pixel_move_%1").arg(idx).toUtf8().constData(), w.maxPixelMoveSpin->value());
    if (w.deadZonePixelsSpin) obs_data_set_double(settings, QString("dead_zone_pixels_%1").arg(idx).toUtf8().constData(), w.deadZonePixelsSpin->value());
    
    if (w.screenOffsetXSpin) obs_data_set_int(settings, QString("screen_offset_x_%1").arg(idx).toUtf8().constData(), w.screenOffsetXSpin->value());
    if (w.screenOffsetYSpin) obs_data_set_int(settings, QString("screen_offset_y_%1").arg(idx).toUtf8().constData(), w.screenOffsetYSpin->value());
    if (w.screenWidthSpin) obs_data_set_int(settings, QString("screen_width_%1").arg(idx).toUtf8().constData(), w.screenWidthSpin->value());
    if (w.screenHeightSpin) obs_data_set_int(settings, QString("screen_height_%1").arg(idx).toUtf8().constData(), w.screenHeightSpin->value());
    
    if (w.enableYAxisUnlockCheck) obs_data_set_bool(settings, QString("enable_y_axis_unlock_%1").arg(idx).toUtf8().constData(), w.enableYAxisUnlockCheck->isChecked());
    if (w.yAxisUnlockDelaySpin) obs_data_set_int(settings, QString("y_axis_unlock_delay_%1").arg(idx).toUtf8().constData(), w.yAxisUnlockDelaySpin->value());
    
    if (w.autoTriggerGroup) obs_data_set_bool(settings, QString("auto_trigger_group_%1").arg(idx).toUtf8().constData(), w.autoTriggerGroup->isChecked());
    if (w.triggerRadiusSpin) obs_data_set_int(settings, QString("trigger_radius_%1").arg(idx).toUtf8().constData(), w.triggerRadiusSpin->value());
    if (w.triggerCooldownSpin) obs_data_set_int(settings, QString("trigger_cooldown_%1").arg(idx).toUtf8().constData(), w.triggerCooldownSpin->value());
    if (w.triggerFireDelaySpin) obs_data_set_int(settings, QString("trigger_fire_delay_%1").arg(idx).toUtf8().constData(), w.triggerFireDelaySpin->value());
    if (w.triggerFireDurationSpin) obs_data_set_int(settings, QString("trigger_fire_duration_%1").arg(idx).toUtf8().constData(), w.triggerFireDurationSpin->value());
    if (w.triggerIntervalSpin) obs_data_set_int(settings, QString("trigger_interval_%1").arg(idx).toUtf8().constData(), w.triggerIntervalSpin->value());
    if (w.triggerMoveCompensationSpin) obs_data_set_int(settings, QString("trigger_move_compensation_%1").arg(idx).toUtf8().constData(), w.triggerMoveCompensationSpin->value());
    
    // 随机触发延迟
    if (w.enableTriggerDelayRandomCheck) obs_data_set_bool(settings, QString("enable_trigger_delay_random_%1").arg(idx).toUtf8().constData(), w.enableTriggerDelayRandomCheck->isChecked());
    if (w.triggerDelayRandomMinSpin) obs_data_set_int(settings, QString("trigger_delay_random_min_%1").arg(idx).toUtf8().constData(), w.triggerDelayRandomMinSpin->value());
    if (w.triggerDelayRandomMaxSpin) obs_data_set_int(settings, QString("trigger_delay_random_max_%1").arg(idx).toUtf8().constData(), w.triggerDelayRandomMaxSpin->value());
    
    // 随机触发持续时间
    if (w.enableTriggerDurationRandomCheck) obs_data_set_bool(settings, QString("enable_trigger_duration_random_%1").arg(idx).toUtf8().constData(), w.enableTriggerDurationRandomCheck->isChecked());
    if (w.triggerDurationRandomMinSpin) obs_data_set_int(settings, QString("trigger_duration_random_min_%1").arg(idx).toUtf8().constData(), w.triggerDurationRandomMinSpin->value());
    if (w.triggerDurationRandomMaxSpin) obs_data_set_int(settings, QString("trigger_duration_random_max_%1").arg(idx).toUtf8().constData(), w.triggerDurationRandomMaxSpin->value());
    
    if (w.recoilGroup) obs_data_set_bool(settings, QString("recoil_group_%1").arg(idx).toUtf8().constData(), w.recoilGroup->isChecked());
    if (w.recoilStrengthSpin) obs_data_set_double(settings, QString("recoil_strength_%1").arg(idx).toUtf8().constData(), w.recoilStrengthSpin->value());
    if (w.recoilSpeedSpin) obs_data_set_int(settings, QString("recoil_speed_%1").arg(idx).toUtf8().constData(), w.recoilSpeedSpin->value());
    if (w.recoilPidGainScaleSpin) obs_data_set_double(settings, QString("recoil_pid_gain_scale_%1").arg(idx).toUtf8().constData(), w.recoilPidGainScaleSpin->value());
    
    if (w.predictorGroup) obs_data_set_bool(settings, QString("derivative_predictor_group_%1").arg(idx).toUtf8().constData(), w.predictorGroup->isChecked());
    if (w.predictionWeightXSpin) obs_data_set_double(settings, QString("prediction_weight_x_%1").arg(idx).toUtf8().constData(), w.predictionWeightXSpin->value());
    if (w.predictionWeightYSpin) obs_data_set_double(settings, QString("prediction_weight_y_%1").arg(idx).toUtf8().constData(), w.predictionWeightYSpin->value());
    if (w.maxPredictionTimeSpin) obs_data_set_double(settings, QString("max_prediction_time_%1").arg(idx).toUtf8().constData(), w.maxPredictionTimeSpin->value());
    
    if (w.bezierGroup) obs_data_set_bool(settings, QString("bezier_movement_group_%1").arg(idx).toUtf8().constData(), w.bezierGroup->isChecked());
    if (w.bezierCurvatureSpin) obs_data_set_double(settings, QString("bezier_curvature_%1").arg(idx).toUtf8().constData(), w.bezierCurvatureSpin->value());
    if (w.bezierRandomnessSpin) obs_data_set_double(settings, QString("bezier_randomness_%1").arg(idx).toUtf8().constData(), w.bezierRandomnessSpin->value());
    
    if (m_showDetectionResultsCheck) obs_data_set_bool(settings, "show_detection_results", m_showDetectionResultsCheck->isChecked());
    if (m_showFOVCheck) obs_data_set_bool(settings, "show_fov", m_showFOVCheck->isChecked());
    if (m_showFOVCircleCheck) obs_data_set_bool(settings, "show_fov_circle", m_showFOVCircleCheck->isChecked());
    if (m_showFOVCrossCheck) obs_data_set_bool(settings, "show_fov_cross", m_showFOVCrossCheck->isChecked());
    if (m_fovRadiusSpin) obs_data_set_int(settings, "fov_radius", m_fovRadiusSpin->value());
    if (m_fovCrossLineScaleSpin) obs_data_set_int(settings, "fov_cross_line_scale", m_fovCrossLineScaleSpin->value());
    if (m_fovCrossLineThicknessSpin) obs_data_set_int(settings, "fov_cross_line_thickness", m_fovCrossLineThicknessSpin->value());
    if (m_fovCircleThicknessSpin) obs_data_set_int(settings, "fov_circle_thickness", m_fovCircleThicknessSpin->value());
    
    // 推理状态
    if (m_toggleInferenceBtn) obs_data_set_bool(settings, "is_inferencing", m_toggleInferenceBtn->isChecked());
    
    // 模型配置
    if (m_modelPathEdit) obs_data_set_string(settings, "model_path", m_modelPathEdit->text().toUtf8().constData());
    if (m_modelVersionCombo) obs_data_set_int(settings, "model_version", m_modelVersionCombo->currentIndex());
    if (m_useGPUCombo) obs_data_set_string(settings, "use_gpu", m_useGPUCombo->currentData().toString().toUtf8().constData());
    if (m_useGPUTextureCheck) obs_data_set_bool(settings, "use_gpu_texture_inference", m_useGPUTextureCheck->isChecked());
    if (m_inputResolutionCombo) obs_data_set_int(settings, "input_resolution", m_inputResolutionCombo->currentData().toInt());
    if (m_numThreadsSpin) obs_data_set_int(settings, "num_threads", m_numThreadsSpin->value());
    
    // 检测配置
    if (m_confidenceThresholdSpin) obs_data_set_double(settings, "confidence_threshold", m_confidenceThresholdSpin->value());
    if (m_nmsThresholdSpin) obs_data_set_double(settings, "nms_threshold", m_nmsThresholdSpin->value());
    if (m_targetClassCombo) obs_data_set_int(settings, "target_class", m_targetClassCombo->currentIndex());
    if (m_targetClassesTextEdit) obs_data_set_string(settings, "target_classes_text", m_targetClassesTextEdit->text().toUtf8().constData());
    if (m_inferenceIntervalSpin) obs_data_set_int(settings, "inference_interval_frames", m_inferenceIntervalSpin->value());
    
    // 区域检测
    if (m_useRegionCheck) obs_data_set_bool(settings, "use_region", m_useRegionCheck->isChecked());
    if (m_regionXSpin) obs_data_set_int(settings, "region_x", m_regionXSpin->value());
    if (m_regionYSpin) obs_data_set_int(settings, "region_y", m_regionYSpin->value());
    if (m_regionWidthSpin) obs_data_set_int(settings, "region_width", m_regionWidthSpin->value());
    if (m_regionHeightSpin) obs_data_set_int(settings, "region_height", m_regionHeightSpin->value());
    
    // 渲染配置
    if (m_bboxLineWidthSpin) obs_data_set_int(settings, "bbox_line_width", m_bboxLineWidthSpin->value());
    if (m_labelFontScaleSpin) obs_data_set_double(settings, "label_font_scale", m_labelFontScaleSpin->value());
    
    // 动态FOV
    if (m_useDynamicFOVCheck) obs_data_set_bool(settings, "use_dynamic_fov", m_useDynamicFOVCheck->isChecked());
    if (m_showFOV2Check) obs_data_set_bool(settings, "show_fov2", m_showFOV2Check->isChecked());
    if (m_fovRadius2Spin) obs_data_set_int(settings, "fov_radius2", m_fovRadius2Spin->value());
    if (m_dynamicFovShrinkSpin) obs_data_set_int(settings, "dynamic_fov_shrink_percent", m_dynamicFovShrinkSpin->value());
    if (m_dynamicFovTransitionSpin) obs_data_set_int(settings, "dynamic_fov_transition_time", m_dynamicFovTransitionSpin->value());
    
    // 检测框平滑
    if (m_detectionSmoothingCheck) obs_data_set_bool(settings, "detection_smoothing_enabled", m_detectionSmoothingCheck->isChecked());
    if (m_detectionSmoothingAlphaSpin) obs_data_set_double(settings, "detection_smoothing_alpha", m_detectionSmoothingAlphaSpin->value());
    
    // KalmanFilter 追踪设置
    if (m_useKalmanTrackerCheck) obs_data_set_bool(settings, "use_kalman_tracker", m_useKalmanTrackerCheck->isChecked());
    if (m_kalmanGenerateThresholdSpin) obs_data_set_int(settings, "kalman_generate_threshold", m_kalmanGenerateThresholdSpin->value());
    if (m_kalmanTerminateCountSpin) obs_data_set_int(settings, "kalman_terminate_count", m_kalmanTerminateCountSpin->value());
    
    // 神经网络轨迹生成器设置
    if (m_enableNeuralPathCheck) obs_data_set_bool(settings, "enable_neural_path", m_enableNeuralPathCheck->isChecked());
    if (m_neuralPathPointsSpin) obs_data_set_int(settings, "neural_path_points", m_neuralPathPointsSpin->value());
    if (m_neuralMouseStepSizeSpin) obs_data_set_double(settings, "neural_mouse_step_size", m_neuralMouseStepSizeSpin->value());
    if (m_neuralTargetRadiusSpin) obs_data_set_int(settings, "neural_target_radius", m_neuralTargetRadiusSpin->value());
    if (m_neuralConsumePerFrameSpin) obs_data_set_int(settings, "neural_consume_per_frame", m_neuralConsumePerFrameSpin->value());
    
    // 高级配置
    if (m_exportCoordinatesCheck) obs_data_set_bool(settings, "export_coordinates", m_exportCoordinatesCheck->isChecked());
    if (m_coordinateOutputPathEdit) obs_data_set_string(settings, "coordinate_output_path", m_coordinateOutputPathEdit->text().toUtf8().constData());
    
    obs_source_update(filter, settings);
    
    obs_data_release(settings);
    obs_source_release(filter);
    obs_source_release(source);
}

void YoloAimSettingsDialog::applySettings()
{
    saveSettings();
}

void YoloAimSettingsDialog::refreshConfigUI()
{
    loadSettings();
}

void YoloAimSettingsDialog::updateVisibility()
{
}

void YoloAimSettingsDialog::attachFilterToSource(const QString& sourceName)
{
    obs_source_t* source = obs_get_source_by_name(sourceName.toUtf8().constData());
    if (!source) return;
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "visual-assist-hidden");
    if (!filter) {
        obs_data_t* settings = obs_data_create();
        filter = obs_source_create("yolo-detector-filter", "visual-assist-hidden", settings, nullptr);
        if (filter) {
            obs_source_filter_add(source, filter);
        }
        obs_data_release(settings);
    }
    
    if (filter) obs_source_release(filter);
    obs_source_release(source);
}

void YoloAimSettingsDialog::detachFilterFromSource(const QString& sourceName)
{
    obs_source_t* source = obs_get_source_by_name(sourceName.toUtf8().constData());
    if (!source) return;
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "visual-assist-hidden");
    if (filter) {
        obs_source_filter_remove(source, filter);
        obs_source_release(filter);
    }
    
    obs_source_release(source);
}

#endif
