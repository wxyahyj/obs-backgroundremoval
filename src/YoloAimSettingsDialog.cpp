#ifdef ENABLE_QT

#include "YoloAimSettingsDialog.hpp"
#include "OBSQTDisplay.hpp"
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
    : QDialog(parent)
    , m_currentConfig(0)
    , m_previewDisplay(nullptr)
    , m_previewPlaceholder(nullptr)
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
    
    setWindowTitle(QStringLiteral("🐟 小鱼"));
    setMinimumSize(1000, 600);
    resize(1200, 700);
    
    setStyleSheet(R"(
        QDialog {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0a0a12, stop:1 #1a1025);
            border: 2px solid #8b5cf6;
            border-radius: 10px;
        }
        
        QLabel {
            color: #e0e0e0;
            font-size: 13px;
        }
        
        QGroupBox {
            background-color: rgba(139, 92, 246, 0.1);
            border: 1px solid #8b5cf6;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 12px;
            font-weight: bold;
            color: #a78bfa;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px;
            color: #a78bfa;
        }
        
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a1025, stop:1 #0a0a12);
            border: 2px solid #8b5cf6;
            border-radius: 8px;
            padding: 10px 25px;
            color: #c4b5fd;
            font-weight: bold;
            font-size: 13px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background: #8b5cf6;
            color: #0a0a12;
        }
        
        QPushButton:pressed {
            background: #7c3aed;
        }
        
        QSpinBox, QDoubleSpinBox {
            background-color: #1a1025;
            border: 1px solid #8b5cf6;
            border-radius: 5px;
            padding: 6px 10px;
            color: #c4b5fd;
            font-size: 13px;
            min-width: 80px;
        }
        
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid #a78bfa;
        }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background-color: #1a1025;
            border: none;
            width: 20px;
        }
        
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            width: 10px;
            height: 10px;
        }
        
        QComboBox {
            background-color: #1a1025;
            border: 1px solid #8b5cf6;
            border-radius: 5px;
            padding: 8px 12px;
            color: #c4b5fd;
            font-size: 13px;
            min-width: 120px;
        }
        
        QComboBox:hover {
            border: 2px solid #a78bfa;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        
        QComboBox::down-arrow {
            width: 12px;
            height: 12px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #1a1025;
            border: 1px solid #8b5cf6;
            color: #c4b5fd;
            selection-background-color: #8b5cf6;
            selection-color: #0a0a12;
        }
        
        QCheckBox {
            color: #e0e0e0;
            spacing: 8px;
            font-size: 13px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 2px solid #8b5cf6;
            background-color: #1a1025;
        }
        
        QCheckBox::indicator:checked {
            background-color: #8b5cf6;
            border-color: #8b5cf6;
        }
        
        QCheckBox::indicator:hover {
            border: 2px solid #a78bfa;
        }
        
        QTabWidget::pane {
            border: 1px solid #8b5cf6;
            border-radius: 8px;
            background-color: rgba(139, 92, 246, 0.05);
        }
        
        QTabBar::tab {
            background: #1a1025;
            border: 1px solid #333;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 10px 20px;
            color: #888888;
            font-size: 13px;
            margin-right: 2px;
        }
        
        QTabBar::tab:hover {
            color: #a78bfa;
            border-color: #8b5cf6;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #8b5cf6, stop:1 #7c3aed);
            color: #0a0a12;
            font-weight: bold;
        }
        
        QScrollArea {
            background-color: transparent;
            border: none;
        }
        
        QScrollBar:vertical {
            background: #0a0a12;
            width: 10px;
            border-radius: 5px;
        }
        
        QScrollBar::handle:vertical {
            background: #8b5cf6;
            border-radius: 5px;
            min-height: 30px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #a78bfa;
        }
        
        QSplitter::handle {
            background: #8b5cf6;
        }
        
        QDialogButtonBox QPushButton {
            min-width: 90px;
        }
    )");
    
    setupUI();
    refreshSourceList();
    
    if (m_previewDisplay) {
        m_previewDisplay->hide();
    }
    if (m_previewPlaceholder) {
        m_previewPlaceholder->show();
    }
}

YoloAimSettingsDialog::~YoloAimSettingsDialog()
{
    dialogInstance = nullptr;
}

void YoloAimSettingsDialog::setupUI()
{
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    
    QHBoxLayout* topLayout = new QHBoxLayout();
    
    QLabel* sourceLabel = new QLabel(QStringLiteral("视频源:"), this);
    m_sourceCombo = new QComboBox(this);
    m_sourceCombo->setMinimumWidth(200);
    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &YoloAimSettingsDialog::onSourceChanged);
    
    topLayout->addWidget(sourceLabel);
    topLayout->addWidget(m_sourceCombo);
    topLayout->addSpacing(20);
    
    m_toggleInferenceBtn = new QPushButton(QStringLiteral("▶ 开始推理"), this);
    m_toggleInferenceBtn->setCheckable(true);
    m_toggleInferenceBtn->setStyleSheet(R"(
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a1025, stop:1 #0a0a12);
            border: 2px solid #22c55e;
            border-radius: 8px;
            padding: 8px 20px;
            color: #22c55e;
            font-weight: bold;
            font-size: 13px;
        }
        QPushButton:hover {
            background: #22c55e;
            color: #0a0a12;
        }
        QPushButton:checked {
            background: #ef4444;
            border-color: #ef4444;
            color: white;
        }
    )");
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
    
    m_inferenceStatusLabel = new QLabel(QStringLiteral("状态: 已停止"), this);
    m_inferenceStatusLabel->setStyleSheet("color: #a78bfa; font-size: 13px; padding: 5px 10px;");
    topLayout->addWidget(m_inferenceStatusLabel);
    
    topLayout->addStretch();
    
    mainLayout->addLayout(topLayout);
    
    QSplitter* splitter = new QSplitter(Qt::Horizontal, this);
    
    QWidget* previewContainer = new QWidget(this);
    QVBoxLayout* previewLayout = new QVBoxLayout(previewContainer);
    previewLayout->setContentsMargins(0, 0, 0, 0);
    
    QLabel* previewLabel = new QLabel(QStringLiteral("📹 视频预览"), this);
    previewLabel->setAlignment(Qt::AlignCenter);
    previewLabel->setStyleSheet("color: #a78bfa; font-size: 14px; font-weight: bold; padding: 5px;");
    previewLayout->addWidget(previewLabel);
    
    m_previewDisplay = new OBSQTDisplay(this);
    m_previewDisplay->setMinimumSize(320, 180);
    m_previewDisplay->SetBackgroundColor(0x0d0d15);
    previewLayout->addWidget(m_previewDisplay, 1);
    
    m_previewPlaceholder = new QLabel(QStringLiteral("请选择视频源"), this);
    m_previewPlaceholder->setAlignment(Qt::AlignCenter);
    m_previewPlaceholder->setStyleSheet("background-color: #0a0a12; color: #a78bfa; font-size: 16px; border: 2px solid #8b5cf6; border-radius: 10px;");
    m_previewPlaceholder->setMinimumSize(320, 180);
    m_previewPlaceholder->hide();
    previewLayout->addWidget(m_previewPlaceholder, 1);
    
    splitter->addWidget(previewContainer);
    
    QWidget* settingsContainer = new QWidget(this);
    QVBoxLayout* settingsLayout = new QVBoxLayout(settingsContainer);
    settingsLayout->setContentsMargins(0, 0, 0, 0);
    
    m_tabWidget = new QTabWidget(this);
    
    setupModelPage();
    setupDetectionPage();
    setupVisualPage();
    setupBasicPage();
    setupAdvancedPIDPage();
    setupTriggerPage();
    setupTrackingPage();
    setupPredictorPage();
    setupBezierPage();
    setupMotionSimulatorPage();
    setupNeuralPathPage();
    
    settingsLayout->addWidget(m_tabWidget);
    
    QHBoxLayout* configLayout = new QHBoxLayout();
    QLabel* configLabel = new QLabel(QStringLiteral("配置:"), this);
    m_configSelect = new QComboBox(this);
    for (int i = 0; i < 5; i++) {
        m_configSelect->addItem(QStringLiteral("配置 %1").arg(i + 1));
    }
    connect(m_configSelect, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &YoloAimSettingsDialog::onConfigChanged);
    
    configLayout->addWidget(configLabel);
    configLayout->addWidget(m_configSelect);
    configLayout->addStretch();
    
    settingsLayout->addLayout(configLayout);
    
    splitter->addWidget(settingsContainer);
    
    splitter->setSizes({400, 500});
    
    mainLayout->addWidget(splitter, 1);
    
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

void YoloAimSettingsDialog::setupModelPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* modelGroup = new QGroupBox(QStringLiteral("📦 模型设置"), page);
    QGridLayout* modelLayout = new QGridLayout(modelGroup);
    int row = 0;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("模型路径:"), page), row, 0);
    m_modelPathEdit = new QLineEdit(page);
    m_modelPathBtn = new QPushButton(QStringLiteral("浏览..."), page);
    QHBoxLayout* modelPathLayout = new QHBoxLayout();
    modelPathLayout->addWidget(m_modelPathEdit);
    modelPathLayout->addWidget(m_modelPathBtn);
    modelLayout->addLayout(modelPathLayout, row, 1);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("模型版本:"), page), row, 0);
    m_modelVersionCombo = new QComboBox(page);
    m_modelVersionCombo->addItem(QStringLiteral("YOLOv5"), 0);
    m_modelVersionCombo->addItem(QStringLiteral("YOLOv8"), 1);
    m_modelVersionCombo->addItem(QStringLiteral("YOLOv11"), 2);
    modelLayout->addWidget(m_modelVersionCombo, row, 1);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("计算设备:"), page), row, 0);
    m_useGPUCombo = new QComboBox(page);
    m_useGPUCombo->addItem(QStringLiteral("CPU"), QStringLiteral("cpu"));
    m_useGPUCombo->addItem(QStringLiteral("CUDA"), QStringLiteral("cuda"));
    m_useGPUCombo->addItem(QStringLiteral("DirectML"), QStringLiteral("dml"));
    m_useGPUCombo->addItem(QStringLiteral("CoreML"), QStringLiteral("coreml"));
    modelLayout->addWidget(m_useGPUCombo, row, 1);
    row++;
    
    m_useGPUTextureCheck = new QCheckBox(QStringLiteral("启用GPU纹理推理(实验性)"), page);
    modelLayout->addWidget(m_useGPUTextureCheck, row, 0, 1, 2);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("输入分辨率:"), page), row, 0);
    m_inputResolutionCombo = new QComboBox(page);
    m_inputResolutionCombo->addItem(QStringLiteral("640x640"), 640);
    m_inputResolutionCombo->addItem(QStringLiteral("320x320"), 320);
    m_inputResolutionCombo->addItem(QStringLiteral("1280x1280"), 1280);
    modelLayout->addWidget(m_inputResolutionCombo, row, 1);
    row++;
    
    modelLayout->addWidget(new QLabel(QStringLiteral("推理线程数:"), page), row, 0);
    m_numThreadsSpin = new QSpinBox(page);
    m_numThreadsSpin->setRange(1, 16);
    m_numThreadsSpin->setValue(4);
    modelLayout->addWidget(m_numThreadsSpin, row, 1);
    
    layout->addWidget(modelGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("📦 模型"));
}

void YoloAimSettingsDialog::setupDetectionPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* detectGroup = new QGroupBox(QStringLiteral("🎯 检测设置"), page);
    QGridLayout* detectLayout = new QGridLayout(detectGroup);
    int row = 0;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("置信度阈值:"), page), row, 0);
    m_confidenceThresholdSpin = new QDoubleSpinBox(page);
    m_confidenceThresholdSpin->setRange(0.01, 1.0);
    m_confidenceThresholdSpin->setSingleStep(0.01);
    m_confidenceThresholdSpin->setValue(0.5);
    detectLayout->addWidget(m_confidenceThresholdSpin, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("NMS阈值:"), page), row, 0);
    m_nmsThresholdSpin = new QDoubleSpinBox(page);
    m_nmsThresholdSpin->setRange(0.01, 1.0);
    m_nmsThresholdSpin->setSingleStep(0.01);
    m_nmsThresholdSpin->setValue(0.45);
    detectLayout->addWidget(m_nmsThresholdSpin, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("目标类别:"), page), row, 0);
    m_targetClassCombo = new QComboBox(page);
    m_targetClassCombo->addItem(QStringLiteral("全部"), -1);
    m_targetClassCombo->addItem(QStringLiteral("人物"), 0);
    m_targetClassCombo->addItem(QStringLiteral("车辆"), -1);
    detectLayout->addWidget(m_targetClassCombo, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("自定义类别:"), page), row, 0);
    m_targetClassesTextEdit = new QLineEdit(page);
    m_targetClassesTextEdit->setPlaceholderText(QStringLiteral("多个类别用逗号分隔，如: 0,1,2"));
    detectLayout->addWidget(m_targetClassesTextEdit, row, 1);
    row++;
    
    detectLayout->addWidget(new QLabel(QStringLiteral("推理间隔(帧):"), page), row, 0);
    m_inferenceIntervalSpin = new QSpinBox(page);
    m_inferenceIntervalSpin->setRange(0, 10);
    m_inferenceIntervalSpin->setValue(0);
    detectLayout->addWidget(m_inferenceIntervalSpin, row, 1);
    
    layout->addWidget(detectGroup);
    
    QGroupBox* regionGroup = new QGroupBox(QStringLiteral("📐 区域检测"), page);
    QGridLayout* regionLayout = new QGridLayout(regionGroup);
    row = 0;
    
    m_useRegionCheck = new QCheckBox(QStringLiteral("启用区域检测"), page);
    regionLayout->addWidget(m_useRegionCheck, row, 0, 1, 2);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域X:"), page), row, 0);
    m_regionXSpin = new QSpinBox(page);
    m_regionXSpin->setRange(0, 3840);
    regionLayout->addWidget(m_regionXSpin, row, 1);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域Y:"), page), row, 0);
    m_regionYSpin = new QSpinBox(page);
    m_regionYSpin->setRange(0, 2160);
    regionLayout->addWidget(m_regionYSpin, row, 1);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域宽度:"), page), row, 0);
    m_regionWidthSpin = new QSpinBox(page);
    m_regionWidthSpin->setRange(1, 3840);
    m_regionWidthSpin->setValue(640);
    regionLayout->addWidget(m_regionWidthSpin, row, 1);
    row++;
    
    regionLayout->addWidget(new QLabel(QStringLiteral("区域高度:"), page), row, 0);
    m_regionHeightSpin = new QSpinBox(page);
    m_regionHeightSpin->setRange(1, 2160);
    m_regionHeightSpin->setValue(480);
    regionLayout->addWidget(m_regionHeightSpin, row, 1);
    
    layout->addWidget(regionGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("🎯 检测"));
}

void YoloAimSettingsDialog::setupVisualPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QScrollArea* scrollArea = new QScrollArea(page);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);
    
    QWidget* scrollContent = new QWidget();
    QVBoxLayout* scrollLayout = new QVBoxLayout(scrollContent);
    
    QGroupBox* displayGroup = new QGroupBox(QStringLiteral("👁 显示设置"), scrollContent);
    QGridLayout* displayLayout = new QGridLayout(displayGroup);
    int row = 0;
    
    m_showDetectionResultsCheck = new QCheckBox(QStringLiteral("显示检测结果"), scrollContent);
    displayLayout->addWidget(m_showDetectionResultsCheck, row, 0, 1, 2);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("边框线宽:"), scrollContent), row, 0);
    m_bboxLineWidthSpin = new QSpinBox(scrollContent);
    m_bboxLineWidthSpin->setRange(1, 5);
    m_bboxLineWidthSpin->setValue(2);
    displayLayout->addWidget(m_bboxLineWidthSpin, row, 1);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("标签字体大小:"), scrollContent), row, 0);
    m_labelFontScaleSpin = new QDoubleSpinBox(scrollContent);
    m_labelFontScaleSpin->setRange(0.2, 1.0);
    m_labelFontScaleSpin->setSingleStep(0.05);
    m_labelFontScaleSpin->setValue(0.5);
    displayLayout->addWidget(m_labelFontScaleSpin, row, 1);
    row++;
    
    m_detectionSmoothingCheck = new QCheckBox(QStringLiteral("启用检测框平滑"), scrollContent);
    displayLayout->addWidget(m_detectionSmoothingCheck, row, 0, 1, 2);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("平滑系数:"), scrollContent), row, 0);
    m_detectionSmoothingAlphaSpin = new QDoubleSpinBox(scrollContent);
    m_detectionSmoothingAlphaSpin->setRange(0.01, 1.0);
    m_detectionSmoothingAlphaSpin->setSingleStep(0.01);
    m_detectionSmoothingAlphaSpin->setValue(0.3);
    displayLayout->addWidget(m_detectionSmoothingAlphaSpin, row, 1);
    
    scrollLayout->addWidget(displayGroup);
    
    QGroupBox* fovGroup = new QGroupBox(QStringLiteral("🎯 FOV设置"), scrollContent);
    QGridLayout* fovLayout = new QGridLayout(fovGroup);
    row = 0;
    
    m_showFOVCheck = new QCheckBox(QStringLiteral("显示FOV"), scrollContent);
    fovLayout->addWidget(m_showFOVCheck, row, 0, 1, 2);
    row++;
    
    m_showFOVCircleCheck = new QCheckBox(QStringLiteral("显示FOV圆圈"), scrollContent);
    fovLayout->addWidget(m_showFOVCircleCheck, row, 0);
    
    m_showFOVCrossCheck = new QCheckBox(QStringLiteral("显示FOV十字"), scrollContent);
    fovLayout->addWidget(m_showFOVCrossCheck, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("FOV半径:"), scrollContent), row, 0);
    m_fovRadiusSpin = new QSpinBox(scrollContent);
    m_fovRadiusSpin->setRange(10, 500);
    fovLayout->addWidget(m_fovRadiusSpin, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("十字线长度:"), scrollContent), row, 0);
    m_fovCrossLineScaleSpin = new QSpinBox(scrollContent);
    m_fovCrossLineScaleSpin->setRange(5, 300);
    fovLayout->addWidget(m_fovCrossLineScaleSpin, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("十字线粗细:"), scrollContent), row, 0);
    m_fovCrossLineThicknessSpin = new QSpinBox(scrollContent);
    m_fovCrossLineThicknessSpin->setRange(1, 10);
    fovLayout->addWidget(m_fovCrossLineThicknessSpin, row, 1);
    row++;
    
    fovLayout->addWidget(new QLabel(QStringLiteral("圆圈粗细:"), scrollContent), row, 0);
    m_fovCircleThicknessSpin = new QSpinBox(scrollContent);
    m_fovCircleThicknessSpin->setRange(1, 10);
    fovLayout->addWidget(m_fovCircleThicknessSpin, row, 1);
    
    scrollLayout->addWidget(fovGroup);
    
    QGroupBox* dynamicFovGroup = new QGroupBox(QStringLiteral("⚡ 动态FOV"), scrollContent);
    QGridLayout* dynamicFovLayout = new QGridLayout(dynamicFovGroup);
    row = 0;
    
    m_useDynamicFOVCheck = new QCheckBox(QStringLiteral("启用动态FOV"), scrollContent);
    dynamicFovLayout->addWidget(m_useDynamicFOVCheck, row, 0, 1, 2);
    row++;
    
    m_showFOV2Check = new QCheckBox(QStringLiteral("显示第二个FOV"), scrollContent);
    dynamicFovLayout->addWidget(m_showFOV2Check, row, 0, 1, 2);
    row++;
    
    dynamicFovLayout->addWidget(new QLabel(QStringLiteral("第二FOV半径:"), scrollContent), row, 0);
    m_fovRadius2Spin = new QSpinBox(scrollContent);
    m_fovRadius2Spin->setRange(1, 200);
    dynamicFovLayout->addWidget(m_fovRadius2Spin, row, 1);
    row++;
    
    dynamicFovLayout->addWidget(new QLabel(QStringLiteral("缩放百分比:"), scrollContent), row, 0);
    m_dynamicFovShrinkSpin = new QSpinBox(scrollContent);
    m_dynamicFovShrinkSpin->setRange(10, 100);
    m_dynamicFovShrinkSpin->setValue(50);
    dynamicFovLayout->addWidget(m_dynamicFovShrinkSpin, row, 1);
    row++;
    
    dynamicFovLayout->addWidget(new QLabel(QStringLiteral("过渡时间(ms):"), scrollContent), row, 0);
    m_dynamicFovTransitionSpin = new QSpinBox(scrollContent);
    m_dynamicFovTransitionSpin->setRange(0, 1000);
    m_dynamicFovTransitionSpin->setSingleStep(10);
    dynamicFovLayout->addWidget(m_dynamicFovTransitionSpin, row, 1);
    
    scrollLayout->addWidget(dynamicFovGroup);
    
    QGroupBox* advancedGroup = new QGroupBox(QStringLiteral("🔧 高级设置"), scrollContent);
    QGridLayout* advancedLayout = new QGridLayout(advancedGroup);
    row = 0;
    
    m_exportCoordinatesCheck = new QCheckBox(QStringLiteral("导出坐标"), scrollContent);
    advancedLayout->addWidget(m_exportCoordinatesCheck, row, 0, 1, 2);
    row++;
    
    advancedLayout->addWidget(new QLabel(QStringLiteral("输出路径:"), scrollContent), row, 0);
    m_coordinateOutputPathEdit = new QLineEdit(scrollContent);
    m_coordinateOutputPathBtn = new QPushButton(QStringLiteral("浏览..."), scrollContent);
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

void YoloAimSettingsDialog::setupBasicPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    for (int i = 0; i < 5; i++) {
        QWidget* configWidget = createConfigWidget(i);
        configWidget->setVisible(i == 0);
        layout->addWidget(configWidget);
    }
    
    m_tabWidget->addTab(page, QStringLiteral("⚙️ 基础"));
}

void YoloAimSettingsDialog::initConfigWidgetStruct(int configIndex)
{
    if (configIndex < 0 || configIndex >= 5) return;
    
    m_configWidgets[configIndex].enabledCheck = nullptr;
    m_configWidgets[configIndex].hotkeyCombo = nullptr;
    m_configWidgets[configIndex].controllerTypeCombo = nullptr;
    m_configWidgets[configIndex].pMinSpin = nullptr;
    m_configWidgets[configIndex].pMaxSpin = nullptr;
    m_configWidgets[configIndex].pSlopeSpin = nullptr;
    m_configWidgets[configIndex].dSpin = nullptr;
    m_configWidgets[configIndex].iSpin = nullptr;
    m_configWidgets[configIndex].derivativeFilterAlphaSpin = nullptr;
    m_configWidgets[configIndex].advTargetThresholdSpin = nullptr;
    m_configWidgets[configIndex].advMinCoefficientSpin = nullptr;
    m_configWidgets[configIndex].advMaxCoefficientSpin = nullptr;
    m_configWidgets[configIndex].advTransitionSharpnessSpin = nullptr;
    m_configWidgets[configIndex].advTransitionMidpointSpin = nullptr;
    m_configWidgets[configIndex].advOutputSmoothingSpin = nullptr;
    m_configWidgets[configIndex].advSpeedFactorSpin = nullptr;
    m_configWidgets[configIndex].useOneEuroFilterCheck = nullptr;
    m_configWidgets[configIndex].oneEuroMinCutoffSpin = nullptr;
    m_configWidgets[configIndex].oneEuroBetaSpin = nullptr;
    m_configWidgets[configIndex].oneEuroDCutoffSpin = nullptr;
    m_configWidgets[configIndex].aimSmoothingXSpin = nullptr;
    m_configWidgets[configIndex].aimSmoothingYSpin = nullptr;
    m_configWidgets[configIndex].targetYOffsetSpin = nullptr;
    m_configWidgets[configIndex].maxPixelMoveSpin = nullptr;
    m_configWidgets[configIndex].deadZonePixelsSpin = nullptr;
    m_configWidgets[configIndex].screenOffsetXSpin = nullptr;
    m_configWidgets[configIndex].screenOffsetYSpin = nullptr;
    m_configWidgets[configIndex].screenWidthSpin = nullptr;
    m_configWidgets[configIndex].screenHeightSpin = nullptr;
    m_configWidgets[configIndex].enableYAxisUnlockCheck = nullptr;
    m_configWidgets[configIndex].yAxisUnlockDelaySpin = nullptr;
    m_configWidgets[configIndex].autoTriggerGroup = nullptr;
    m_configWidgets[configIndex].triggerRadiusSpin = nullptr;
    m_configWidgets[configIndex].triggerCooldownSpin = nullptr;
    m_configWidgets[configIndex].triggerFireDelaySpin = nullptr;
    m_configWidgets[configIndex].triggerFireDurationSpin = nullptr;
    m_configWidgets[configIndex].triggerIntervalSpin = nullptr;
    m_configWidgets[configIndex].enableTriggerDelayRandomCheck = nullptr;
    m_configWidgets[configIndex].triggerDelayRandomMinSpin = nullptr;
    m_configWidgets[configIndex].triggerDelayRandomMaxSpin = nullptr;
    m_configWidgets[configIndex].enableTriggerDurationRandomCheck = nullptr;
    m_configWidgets[configIndex].triggerDurationRandomMinSpin = nullptr;
    m_configWidgets[configIndex].triggerDurationRandomMaxSpin = nullptr;
    m_configWidgets[configIndex].triggerMoveCompensationSpin = nullptr;
    m_configWidgets[configIndex].recoilGroup = nullptr;
    m_configWidgets[configIndex].recoilStrengthSpin = nullptr;
    m_configWidgets[configIndex].recoilSpeedSpin = nullptr;
    m_configWidgets[configIndex].recoilPidGainScaleSpin = nullptr;
    m_configWidgets[configIndex].integralLimitSpin = nullptr;
    m_configWidgets[configIndex].integralSeparationThresholdSpin = nullptr;
    m_configWidgets[configIndex].integralDeadZoneSpin = nullptr;
    m_configWidgets[configIndex].integralRateSpin = nullptr;
    m_configWidgets[configIndex].pGainRampInitialScaleSpin = nullptr;
    m_configWidgets[configIndex].pGainRampDurationSpin = nullptr;
    m_configWidgets[configIndex].predictorGroup = nullptr;
    m_configWidgets[configIndex].predictionWeightXSpin = nullptr;
    m_configWidgets[configIndex].predictionWeightYSpin = nullptr;
    m_configWidgets[configIndex].maxPredictionTimeSpin = nullptr;
    m_configWidgets[configIndex].bezierGroup = nullptr;
    m_configWidgets[configIndex].bezierCurvatureSpin = nullptr;
    m_configWidgets[configIndex].bezierRandomnessSpin = nullptr;
}

void YoloAimSettingsDialog::initAllConfigWidgetStructs()
{
    for (int i = 0; i < 5; i++) {
        initConfigWidgetStruct(i);
    }
}

void YoloAimSettingsDialog::setupAdvancedPIDPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* pidGroup = new QGroupBox(QStringLiteral("⚡ PID参数"), page);
    QGridLayout* pidLayout = new QGridLayout(pidGroup);
    
    int row = 0;
    
    pidLayout->addWidget(new QLabel(QStringLiteral("P最小:"), page), row, 0);
    m_configWidgets[0].pMinSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].pMinSpin->setRange(0.0, 2.0);
    m_configWidgets[0].pMinSpin->setDecimals(3);
    m_configWidgets[0].pMinSpin->setSingleStep(0.001);
    pidLayout->addWidget(m_configWidgets[0].pMinSpin, row, 1);
    
    pidLayout->addWidget(new QLabel(QStringLiteral("P最大:"), page), row, 2);
    m_configWidgets[0].pMaxSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].pMaxSpin->setRange(0.0, 2.0);
    m_configWidgets[0].pMaxSpin->setDecimals(3);
    m_configWidgets[0].pMaxSpin->setSingleStep(0.001);
    pidLayout->addWidget(m_configWidgets[0].pMaxSpin, row, 3);
    row++;
    
    pidLayout->addWidget(new QLabel(QStringLiteral("P斜率:"), page), row, 0);
    m_configWidgets[0].pSlopeSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].pSlopeSpin->setRange(0.0, 5.0);
    m_configWidgets[0].pSlopeSpin->setDecimals(2);
    pidLayout->addWidget(m_configWidgets[0].pSlopeSpin, row, 1);
    
    pidLayout->addWidget(new QLabel(QStringLiteral("D:"), page), row, 2);
    m_configWidgets[0].dSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].dSpin->setRange(0.0, 0.1);
    m_configWidgets[0].dSpin->setDecimals(4);
    m_configWidgets[0].dSpin->setSingleStep(0.0001);
    pidLayout->addWidget(m_configWidgets[0].dSpin, row, 3);
    row++;
    
    pidLayout->addWidget(new QLabel(QStringLiteral("I:"), page), row, 0);
    m_configWidgets[0].iSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].iSpin->setRange(0.0, 0.5);
    m_configWidgets[0].iSpin->setDecimals(4);
    m_configWidgets[0].iSpin->setSingleStep(0.0001);
    pidLayout->addWidget(m_configWidgets[0].iSpin, row, 1);
    
    pidLayout->addWidget(new QLabel(QStringLiteral("微分滤波:"), page), row, 2);
    m_configWidgets[0].derivativeFilterAlphaSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].derivativeFilterAlphaSpin->setRange(0.0, 1.0);
    m_configWidgets[0].derivativeFilterAlphaSpin->setDecimals(2);
    pidLayout->addWidget(m_configWidgets[0].derivativeFilterAlphaSpin, row, 3);
    row++;
    
    layout->addWidget(pidGroup);
    
    QGroupBox* advGroup = new QGroupBox(QStringLiteral("🔧 高级参数"), page);
    QGridLayout* advLayout = new QGridLayout(advGroup);
    
    row = 0;
    advLayout->addWidget(new QLabel(QStringLiteral("目标阈值:"), page), row, 0);
    m_configWidgets[0].advTargetThresholdSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].advTargetThresholdSpin->setRange(0.0, 100.0);
    advLayout->addWidget(m_configWidgets[0].advTargetThresholdSpin, row, 1);
    
    advLayout->addWidget(new QLabel(QStringLiteral("最小系数:"), page), row, 2);
    m_configWidgets[0].advMinCoefficientSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].advMinCoefficientSpin->setRange(0.0, 5.0);
    advLayout->addWidget(m_configWidgets[0].advMinCoefficientSpin, row, 3);
    row++;
    
    advLayout->addWidget(new QLabel(QStringLiteral("最大系数:"), page), row, 0);
    m_configWidgets[0].advMaxCoefficientSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].advMaxCoefficientSpin->setRange(0.0, 5.0);
    advLayout->addWidget(m_configWidgets[0].advMaxCoefficientSpin, row, 1);
    
    advLayout->addWidget(new QLabel(QStringLiteral("过渡锐度:"), page), row, 2);
    m_configWidgets[0].advTransitionSharpnessSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].advTransitionSharpnessSpin->setRange(0.0, 20.0);
    advLayout->addWidget(m_configWidgets[0].advTransitionSharpnessSpin, row, 3);
    row++;
    
    advLayout->addWidget(new QLabel(QStringLiteral("过渡中点:"), page), row, 0);
    m_configWidgets[0].advTransitionMidpointSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].advTransitionMidpointSpin->setRange(0.0, 1.0);
    m_configWidgets[0].advTransitionMidpointSpin->setDecimals(2);
    advLayout->addWidget(m_configWidgets[0].advTransitionMidpointSpin, row, 1);
    
    advLayout->addWidget(new QLabel(QStringLiteral("速度因子:"), page), row, 2);
    m_configWidgets[0].advSpeedFactorSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].advSpeedFactorSpin->setRange(0.0, 5.0);
    advLayout->addWidget(m_configWidgets[0].advSpeedFactorSpin, row, 3);
    row++;
    
    layout->addWidget(advGroup);
    
    QGroupBox* oneEuroGroup = new QGroupBox(QStringLiteral("💫 One Euro Filter"), page);
    QHBoxLayout* oneEuroLayout = new QHBoxLayout(oneEuroGroup);
    
    m_configWidgets[0].useOneEuroFilterCheck = new QCheckBox(QStringLiteral("启用"), page);
    oneEuroLayout->addWidget(m_configWidgets[0].useOneEuroFilterCheck);
    
    oneEuroLayout->addWidget(new QLabel(QStringLiteral("最小截止:"), page));
    m_configWidgets[0].oneEuroMinCutoffSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].oneEuroMinCutoffSpin->setRange(0.0, 10.0);
    oneEuroLayout->addWidget(m_configWidgets[0].oneEuroMinCutoffSpin);
    
    oneEuroLayout->addWidget(new QLabel(QStringLiteral("Beta:"), page));
    m_configWidgets[0].oneEuroBetaSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].oneEuroBetaSpin->setRange(0.0, 10.0);
    oneEuroLayout->addWidget(m_configWidgets[0].oneEuroBetaSpin);
    
    oneEuroLayout->addWidget(new QLabel(QStringLiteral("D截止:"), page));
    m_configWidgets[0].oneEuroDCutoffSpin = new QDoubleSpinBox(page);
    m_configWidgets[0].oneEuroDCutoffSpin->setRange(0.0, 10.0);
    oneEuroLayout->addWidget(m_configWidgets[0].oneEuroDCutoffSpin);
    
    oneEuroLayout->addStretch();
    
    layout->addWidget(oneEuroGroup);
    
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("⚡ 高级PID"));
}

void YoloAimSettingsDialog::setupTriggerPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    m_configWidgets[0].autoTriggerGroup = createAutoTriggerGroup(0);
    layout->addWidget(m_configWidgets[0].autoTriggerGroup);
    
    m_configWidgets[0].recoilGroup = createRecoilGroup(0);
    layout->addWidget(m_configWidgets[0].recoilGroup);
    
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("🎯 扳机"));
}

void YoloAimSettingsDialog::setupTrackingPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* kalmanGroup = new QGroupBox(QStringLiteral("🎯 卡尔曼追踪"), page);
    QFormLayout* kalmanLayout = new QFormLayout(kalmanGroup);
    
    m_useKalmanTrackerCheck = new QCheckBox(QStringLiteral("启用卡尔曼追踪"), page);
    m_useKalmanTrackerCheck->setToolTip(QStringLiteral("启用卡尔曼滤波器进行目标追踪，提供更稳定的目标ID和预测能力"));
    kalmanLayout->addRow(m_useKalmanTrackerCheck);
    
    m_kalmanGenerateThresholdSpin = new QSpinBox(page);
    m_kalmanGenerateThresholdSpin->setRange(1, 10);
    m_kalmanGenerateThresholdSpin->setValue(2);
    m_kalmanGenerateThresholdSpin->setToolTip(QStringLiteral("目标需要连续检测到的帧数才能被确认追踪"));
    kalmanLayout->addRow(QStringLiteral("追踪确认阈值:"), m_kalmanGenerateThresholdSpin);
    
    m_kalmanTerminateCountSpin = new QSpinBox(page);
    m_kalmanTerminateCountSpin->setRange(1, 10);
    m_kalmanTerminateCountSpin->setValue(5);
    m_kalmanTerminateCountSpin->setToolTip(QStringLiteral("目标丢失多少帧后停止追踪"));
    kalmanLayout->addRow(QStringLiteral("追踪丢失阈值:"), m_kalmanTerminateCountSpin);
    
    layout->addWidget(kalmanGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("📍 追踪"));
}

void YoloAimSettingsDialog::setupPredictorPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    m_configWidgets[0].predictorGroup = createPredictorGroup(0);
    layout->addWidget(m_configWidgets[0].predictorGroup);
    
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("🔮 预测"));
}

void YoloAimSettingsDialog::setupBezierPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    m_configWidgets[0].bezierGroup = createBezierGroup(0);
    layout->addWidget(m_configWidgets[0].bezierGroup);
    
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("🌊 贝塞尔"));
}

void YoloAimSettingsDialog::setupMotionSimulatorPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    // 启用开关
    m_enableMotionSimulatorCheck = new QCheckBox(QStringLiteral("🎮 启用人类行为模拟"), page);
    m_enableMotionSimulatorCheck->setToolTip(QStringLiteral("启用人类行为模拟器，使鼠标移动更自然"));
    layout->addWidget(m_enableMotionSimulatorCheck);
    
    // 功能开关组
    QGroupBox* featureGroup = new QGroupBox(QStringLiteral("🔧 功能开关"), page);
    QGridLayout* featureLayout = new QGridLayout(featureGroup);
    
    int row = 0;
    m_motionSimRandomPosCheck = new QCheckBox(QStringLiteral("随机落点"), page);
    m_motionSimRandomPosCheck->setToolTip(QStringLiteral("在目标范围内随机选择落点"));
    featureLayout->addWidget(m_motionSimRandomPosCheck, row, 0);
    
    m_motionSimOvershootCheck = new QCheckBox(QStringLiteral("过冲"), page);
    m_motionSimOvershootCheck->setToolTip(QStringLiteral("模拟人类移动时的过冲行为"));
    featureLayout->addWidget(m_motionSimOvershootCheck, row, 1);
    row++;
    
    m_motionSimMicroOvershootCheck = new QCheckBox(QStringLiteral("微过冲"), page);
    m_motionSimMicroOvershootCheck->setToolTip(QStringLiteral("模拟人类移动时的微小过冲"));
    featureLayout->addWidget(m_motionSimMicroOvershootCheck, row, 0);
    
    m_motionSimInertiaCheck = new QCheckBox(QStringLiteral("惯性停止"), page);
    m_motionSimInertiaCheck->setToolTip(QStringLiteral("模拟人类移动时的惯性停止效果"));
    featureLayout->addWidget(m_motionSimInertiaCheck, row, 1);
    row++;
    
    m_motionSimLeftBtnAdaptiveCheck = new QCheckBox(QStringLiteral("左键自适应"), page);
    m_motionSimLeftBtnAdaptiveCheck->setToolTip(QStringLiteral("根据左键状态调整移动行为"));
    featureLayout->addWidget(m_motionSimLeftBtnAdaptiveCheck, row, 0);
    
    m_motionSimSprayModeCheck = new QCheckBox(QStringLiteral("连射模式"), page);
    m_motionSimSprayModeCheck->setToolTip(QStringLiteral("启用连射模式下的特殊行为"));
    featureLayout->addWidget(m_motionSimSprayModeCheck, row, 1);
    row++;
    
    m_motionSimTapPauseCheck = new QCheckBox(QStringLiteral("点击暂停"), page);
    m_motionSimTapPauseCheck->setToolTip(QStringLiteral("点击时暂停移动"));
    featureLayout->addWidget(m_motionSimTapPauseCheck, row, 0);
    
    m_motionSimRetryCheck = new QCheckBox(QStringLiteral("重试"), page);
    m_motionSimRetryCheck->setToolTip(QStringLiteral("未命中时自动重试"));
    featureLayout->addWidget(m_motionSimRetryCheck, row, 1);
    
    layout->addWidget(featureGroup);
    
    // 参数设置组
    QGroupBox* paramsGroup = new QGroupBox(QStringLiteral("📊 参数设置"), page);
    QFormLayout* paramsLayout = new QFormLayout(paramsGroup);
    
    m_motionSimMaxRetrySpin = new QSpinBox(page);
    m_motionSimMaxRetrySpin->setRange(0, 5);
    m_motionSimMaxRetrySpin->setValue(2);
    m_motionSimMaxRetrySpin->setToolTip(QStringLiteral("未命中时的最大重试次数"));
    paramsLayout->addRow(QStringLiteral("最大重试次数:"), m_motionSimMaxRetrySpin);
    
    m_motionSimDelayMsSpin = new QSpinBox(page);
    m_motionSimDelayMsSpin->setRange(0, 500);
    m_motionSimDelayMsSpin->setValue(80);
    m_motionSimDelayMsSpin->setSuffix(QStringLiteral(" ms"));
    m_motionSimDelayMsSpin->setToolTip(QStringLiteral("目标延迟时间（毫秒）"));
    paramsLayout->addRow(QStringLiteral("目标延迟:"), m_motionSimDelayMsSpin);
    
    m_motionSimDirectProbSpin = new QDoubleSpinBox(page);
    m_motionSimDirectProbSpin->setRange(0.0, 1.0);
    m_motionSimDirectProbSpin->setDecimals(2);
    m_motionSimDirectProbSpin->setSingleStep(0.01);
    m_motionSimDirectProbSpin->setValue(0.85);
    m_motionSimDirectProbSpin->setToolTip(QStringLiteral("直接移动到目标的概率"));
    paramsLayout->addRow(QStringLiteral("直线移动概率:"), m_motionSimDirectProbSpin);
    
    m_motionSimOvershootProbSpin = new QDoubleSpinBox(page);
    m_motionSimOvershootProbSpin->setRange(0.0, 1.0);
    m_motionSimOvershootProbSpin->setDecimals(2);
    m_motionSimOvershootProbSpin->setSingleStep(0.01);
    m_motionSimOvershootProbSpin->setValue(0.10);
    m_motionSimOvershootProbSpin->setToolTip(QStringLiteral("过冲移动的概率"));
    paramsLayout->addRow(QStringLiteral("过冲概率:"), m_motionSimOvershootProbSpin);
    
    m_motionSimMicroOvshootProbSpin = new QDoubleSpinBox(page);
    m_motionSimMicroOvshootProbSpin->setRange(0.0, 1.0);
    m_motionSimMicroOvshootProbSpin->setDecimals(2);
    m_motionSimMicroOvshootProbSpin->setSingleStep(0.01);
    m_motionSimMicroOvshootProbSpin->setValue(0.05);
    m_motionSimMicroOvshootProbSpin->setToolTip(QStringLiteral("微过冲移动的概率"));
    paramsLayout->addRow(QStringLiteral("微过冲概率:"), m_motionSimMicroOvshootProbSpin);
    
    layout->addWidget(paramsGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("🎮 运动仿真"));
}

void YoloAimSettingsDialog::setupNeuralPathPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    // 启用开关
    m_enableNeuralPathCheck = new QCheckBox(QStringLiteral("🧠 启用神经网络轨迹"), page);
    m_enableNeuralPathCheck->setToolTip(QStringLiteral("启用神经网络轨迹生成器，生成更自然的鼠标移动轨迹"));
    layout->addWidget(m_enableNeuralPathCheck);
    
    // 说明文字
    QLabel* descLabel = new QLabel(QStringLiteral(
        "神经网络轨迹生成器使用预训练模型生成类人鼠标移动轨迹。\n"
        "相比贝塞尔曲线，神经网络生成的轨迹更自然、更难被检测。"
    ), page);
    descLabel->setWordWrap(true);
    descLabel->setStyleSheet(QStringLiteral("color: #a78bfa; padding: 10px; background: rgba(139, 92, 246, 0.1); border-radius: 5px;"));
    layout->addWidget(descLabel);
    
    // 参数设置组
    QGroupBox* paramsGroup = new QGroupBox(QStringLiteral("📊 参数设置"), page);
    QFormLayout* paramsLayout = new QFormLayout(paramsGroup);
    
    m_neuralPathPointsSpin = new QSpinBox(page);
    m_neuralPathPointsSpin->setRange(10, 100);
    m_neuralPathPointsSpin->setValue(25);
    m_neuralPathPointsSpin->setSuffix(QStringLiteral(" 点"));
    m_neuralPathPointsSpin->setToolTip(QStringLiteral("轨迹点数量，越多越平滑但移动越慢"));
    paramsLayout->addRow(QStringLiteral("轨迹点数量:"), m_neuralPathPointsSpin);
    
    m_neuralMouseStepSizeSpin = new QDoubleSpinBox(page);
    m_neuralMouseStepSizeSpin->setRange(1.0, 20.0);
    m_neuralMouseStepSizeSpin->setDecimals(1);
    m_neuralMouseStepSizeSpin->setSingleStep(0.5);
    m_neuralMouseStepSizeSpin->setValue(4.0);
    m_neuralMouseStepSizeSpin->setToolTip(QStringLiteral("每次移动的步长大小"));
    paramsLayout->addRow(QStringLiteral("鼠标步长:"), m_neuralMouseStepSizeSpin);
    
    m_neuralTargetRadiusSpin = new QSpinBox(page);
    m_neuralTargetRadiusSpin->setRange(1, 50);
    m_neuralTargetRadiusSpin->setValue(8);
    m_neuralTargetRadiusSpin->setSuffix(QStringLiteral(" px"));
    m_neuralTargetRadiusSpin->setToolTip(QStringLiteral("到达目标的判定半径"));
    paramsLayout->addRow(QStringLiteral("目标半径:"), m_neuralTargetRadiusSpin);

    m_neuralConsumePerFrameSpin = new QSpinBox(page);
    m_neuralConsumePerFrameSpin->setRange(1, 5);
    m_neuralConsumePerFrameSpin->setValue(2);
    m_neuralConsumePerFrameSpin->setToolTip(QStringLiteral("每帧消费的路径点数量，越大移动越快但曲线越粗略（1=拟人,2=平衡,3+=快速）"));
    paramsLayout->addRow(QStringLiteral("每帧消费点数:"), m_neuralConsumePerFrameSpin);
    
    layout->addWidget(paramsGroup);
    
    // 注意事项
    QLabel* noteLabel = new QLabel(QStringLiteral(
        "⚠️ 注意：神经网络轨迹与运动仿真模式互斥，启用其中一个会自动禁用另一个。"
    ), page);
    noteLabel->setWordWrap(true);
    noteLabel->setStyleSheet(QStringLiteral("color: #f59e0b; padding: 10px; background: rgba(245, 158, 11, 0.1); border-radius: 5px;"));
    layout->addWidget(noteLabel);
    
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("🧠 神经轨迹"));
}

QWidget* YoloAimSettingsDialog::createConfigWidget(int configIndex)
{
    QWidget* widget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(widget);
    
    QGroupBox* basicGroup = new QGroupBox(QStringLiteral("⚙️ 基础参数"), widget);
    QGridLayout* basicLayout = new QGridLayout(basicGroup);
    
    int row = 0;
    
    m_configWidgets[configIndex].enabledCheck = new QCheckBox(QStringLiteral("启用"), widget);
    basicLayout->addWidget(m_configWidgets[configIndex].enabledCheck, row, 0, 1, 2);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("热键:"), widget), row, 0);
    m_configWidgets[configIndex].hotkeyCombo = new QComboBox(widget);
    m_configWidgets[configIndex].hotkeyCombo->addItem(QStringLiteral("鼠标侧键1"), 0x05);
    m_configWidgets[configIndex].hotkeyCombo->addItem(QStringLiteral("鼠标侧键2"), 0x06);
    m_configWidgets[configIndex].hotkeyCombo->addItem(QStringLiteral("中键"), 0x04);
    basicLayout->addWidget(m_configWidgets[configIndex].hotkeyCombo, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("控制器:"), widget), row, 2);
    m_configWidgets[configIndex].controllerTypeCombo = new QComboBox(widget);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("高级PID"), 0);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("标准PID"), 1);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("ChrisPID"), 2);
    m_configWidgets[configIndex].controllerTypeCombo->addItem(QStringLiteral("DynamicPID"), 3);
    basicLayout->addWidget(m_configWidgets[configIndex].controllerTypeCombo, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("X平滑:"), widget), row, 0);
    m_configWidgets[configIndex].aimSmoothingXSpin = new QDoubleSpinBox(widget);
    m_configWidgets[configIndex].aimSmoothingXSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].aimSmoothingXSpin->setDecimals(2);
    basicLayout->addWidget(m_configWidgets[configIndex].aimSmoothingXSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("Y平滑:"), widget), row, 2);
    m_configWidgets[configIndex].aimSmoothingYSpin = new QDoubleSpinBox(widget);
    m_configWidgets[configIndex].aimSmoothingYSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].aimSmoothingYSpin->setDecimals(2);
    basicLayout->addWidget(m_configWidgets[configIndex].aimSmoothingYSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("Y偏移(%):"), widget), row, 0);
    m_configWidgets[configIndex].targetYOffsetSpin = new QDoubleSpinBox(widget);
    m_configWidgets[configIndex].targetYOffsetSpin->setRange(-50.0, 50.0);
    basicLayout->addWidget(m_configWidgets[configIndex].targetYOffsetSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("最大移动:"), widget), row, 2);
    m_configWidgets[configIndex].maxPixelMoveSpin = new QDoubleSpinBox(widget);
    m_configWidgets[configIndex].maxPixelMoveSpin->setRange(0.0, 500.0);
    basicLayout->addWidget(m_configWidgets[configIndex].maxPixelMoveSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("死区:"), widget), row, 0);
    m_configWidgets[configIndex].deadZonePixelsSpin = new QDoubleSpinBox(widget);
    m_configWidgets[configIndex].deadZonePixelsSpin->setRange(0.0, 50.0);
    m_configWidgets[configIndex].deadZonePixelsSpin->setDecimals(1);
    basicLayout->addWidget(m_configWidgets[configIndex].deadZonePixelsSpin, row, 1);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕偏移X:"), widget), row, 0);
    m_configWidgets[configIndex].screenOffsetXSpin = new QSpinBox(widget);
    m_configWidgets[configIndex].screenOffsetXSpin->setRange(0, 3840);
    basicLayout->addWidget(m_configWidgets[configIndex].screenOffsetXSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕偏移Y:"), widget), row, 2);
    m_configWidgets[configIndex].screenOffsetYSpin = new QSpinBox(widget);
    m_configWidgets[configIndex].screenOffsetYSpin->setRange(0, 2160);
    basicLayout->addWidget(m_configWidgets[configIndex].screenOffsetYSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕宽度:"), widget), row, 0);
    m_configWidgets[configIndex].screenWidthSpin = new QSpinBox(widget);
    m_configWidgets[configIndex].screenWidthSpin->setRange(640, 3840);
    m_configWidgets[configIndex].screenWidthSpin->setValue(1920);
    basicLayout->addWidget(m_configWidgets[configIndex].screenWidthSpin, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("屏幕高度:"), widget), row, 2);
    m_configWidgets[configIndex].screenHeightSpin = new QSpinBox(widget);
    m_configWidgets[configIndex].screenHeightSpin->setRange(480, 2160);
    m_configWidgets[configIndex].screenHeightSpin->setValue(1080);
    basicLayout->addWidget(m_configWidgets[configIndex].screenHeightSpin, row, 3);
    row++;
    
    basicLayout->addWidget(new QLabel(QStringLiteral("Y轴解锁:"), widget), row, 0);
    m_configWidgets[configIndex].enableYAxisUnlockCheck = new QCheckBox(widget);
    basicLayout->addWidget(m_configWidgets[configIndex].enableYAxisUnlockCheck, row, 1);
    
    basicLayout->addWidget(new QLabel(QStringLiteral("解锁延迟(ms):"), widget), row, 2);
    m_configWidgets[configIndex].yAxisUnlockDelaySpin = new QSpinBox(widget);
    m_configWidgets[configIndex].yAxisUnlockDelaySpin->setRange(100, 2000);
    basicLayout->addWidget(m_configWidgets[configIndex].yAxisUnlockDelaySpin, row, 3);
    
    layout->addWidget(basicGroup);
    layout->addStretch();
    
    return widget;
}

QGroupBox* YoloAimSettingsDialog::createAutoTriggerGroup(int configIndex)
{
    QGroupBox* group = new QGroupBox(QStringLiteral("🎯 自动扳机"), this);
    group->setCheckable(true);
    group->setChecked(false);
    
    QGridLayout* layout = new QGridLayout(group);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("触发半径:"), this), row, 0);
    m_configWidgets[configIndex].triggerRadiusSpin = new QSpinBox(this);
    m_configWidgets[configIndex].triggerRadiusSpin->setRange(1, 50);
    layout->addWidget(m_configWidgets[configIndex].triggerRadiusSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("冷却时间(ms):"), this), row, 2);
    m_configWidgets[configIndex].triggerCooldownSpin = new QSpinBox(this);
    m_configWidgets[configIndex].triggerCooldownSpin->setRange(50, 1000);
    layout->addWidget(m_configWidgets[configIndex].triggerCooldownSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("开火延迟(ms):"), this), row, 0);
    m_configWidgets[configIndex].triggerFireDelaySpin = new QSpinBox(this);
    m_configWidgets[configIndex].triggerFireDelaySpin->setRange(0, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerFireDelaySpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("开火时长(ms):"), this), row, 2);
    m_configWidgets[configIndex].triggerFireDurationSpin = new QSpinBox(this);
    m_configWidgets[configIndex].triggerFireDurationSpin->setRange(10, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerFireDurationSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("间隔(ms):"), this), row, 0);
    m_configWidgets[configIndex].triggerIntervalSpin = new QSpinBox(this);
    m_configWidgets[configIndex].triggerIntervalSpin->setRange(10, 500);
    layout->addWidget(m_configWidgets[configIndex].triggerIntervalSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("移动补偿:"), this), row, 2);
    m_configWidgets[configIndex].triggerMoveCompensationSpin = new QSpinBox(this);
    m_configWidgets[configIndex].triggerMoveCompensationSpin->setRange(0, 100);
    layout->addWidget(m_configWidgets[configIndex].triggerMoveCompensationSpin, row, 3);
    
    return group;
}

QGroupBox* YoloAimSettingsDialog::createRecoilGroup(int configIndex)
{
    QGroupBox* group = new QGroupBox(QStringLiteral("💥 后坐力控制"), this);
    group->setCheckable(true);
    group->setChecked(false);
    
    QGridLayout* layout = new QGridLayout(group);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("强度:"), this), row, 0);
    m_configWidgets[configIndex].recoilStrengthSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].recoilStrengthSpin->setRange(0.0, 50.0);
    layout->addWidget(m_configWidgets[configIndex].recoilStrengthSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("速度:"), this), row, 2);
    m_configWidgets[configIndex].recoilSpeedSpin = new QSpinBox(this);
    m_configWidgets[configIndex].recoilSpeedSpin->setRange(1, 100);
    layout->addWidget(m_configWidgets[configIndex].recoilSpeedSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("PID增益比例:"), this), row, 0);
    m_configWidgets[configIndex].recoilPidGainScaleSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].recoilPidGainScaleSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].recoilPidGainScaleSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].recoilPidGainScaleSpin, row, 1);
    
    return group;
}

QGroupBox* YoloAimSettingsDialog::createPredictorGroup(int configIndex)
{
    QGroupBox* group = new QGroupBox(QStringLiteral("🔮 预测器"), this);
    group->setCheckable(true);
    group->setChecked(true);
    
    QGridLayout* layout = new QGridLayout(group);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("X预测权重:"), this), row, 0);
    m_configWidgets[configIndex].predictionWeightXSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].predictionWeightXSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].predictionWeightXSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].predictionWeightXSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("Y预测权重:"), this), row, 2);
    m_configWidgets[configIndex].predictionWeightYSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].predictionWeightYSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].predictionWeightYSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].predictionWeightYSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("最大预测时间(s):"), this), row, 0);
    m_configWidgets[configIndex].maxPredictionTimeSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].maxPredictionTimeSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].maxPredictionTimeSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].maxPredictionTimeSpin, row, 1);
    
    return group;
}

QGroupBox* YoloAimSettingsDialog::createBezierGroup(int configIndex)
{
    QGroupBox* group = new QGroupBox(QStringLiteral("🌊 贝塞尔曲线"), this);
    group->setCheckable(true);
    group->setChecked(false);
    
    QGridLayout* layout = new QGridLayout(group);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("曲率:"), this), row, 0);
    m_configWidgets[configIndex].bezierCurvatureSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].bezierCurvatureSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].bezierCurvatureSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].bezierCurvatureSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("随机性:"), this), row, 2);
    m_configWidgets[configIndex].bezierRandomnessSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].bezierRandomnessSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].bezierRandomnessSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].bezierRandomnessSpin, row, 3);
    
    return group;
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
        
        obs_source_t* source = obs_get_source_by_name(newSource.toUtf8().constData());
        if (source) {
            m_previewDisplay->SetSource(source);
            m_previewDisplay->show();
            m_previewPlaceholder->hide();
            obs_source_release(source);
        }
    } else {
        m_previewDisplay->SetSource(nullptr);
        m_previewDisplay->hide();
        m_previewPlaceholder->show();
    }
    
    m_currentSource = newSource;
    loadSettings();
}

void YoloAimSettingsDialog::onConfigChanged(int index)
{
    m_currentConfig = index;
    refreshConfigUI();
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
    if (w.advSpeedFactorSpin) w.advSpeedFactorSpin->setValue(obs_data_get_double(settings, QString("adv_speed_factor_%1").arg(idx).toUtf8().constData()));
    
    if (w.useOneEuroFilterCheck) w.useOneEuroFilterCheck->setChecked(obs_data_get_bool(settings, QString("use_one_euro_filter_%1").arg(idx).toUtf8().constData()));
    if (w.oneEuroMinCutoffSpin) w.oneEuroMinCutoffSpin->setValue(obs_data_get_double(settings, QString("one_euro_min_cutoff_%1").arg(idx).toUtf8().constData()));
    if (w.oneEuroBetaSpin) w.oneEuroBetaSpin->setValue(obs_data_get_double(settings, QString("one_euro_beta_%1").arg(idx).toUtf8().constData()));
    if (w.oneEuroDCutoffSpin) w.oneEuroDCutoffSpin->setValue(obs_data_get_double(settings, QString("one_euro_d_cutoff_%1").arg(idx).toUtf8().constData()));
    
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
    
    // MotionSimulator 人类行为模拟器设置
    if (m_enableMotionSimulatorCheck) m_enableMotionSimulatorCheck->setChecked(obs_data_get_bool(settings, "enable_motion_simulator"));
    if (m_motionSimRandomPosCheck) m_motionSimRandomPosCheck->setChecked(obs_data_get_bool(settings, "motion_sim_random_pos"));
    if (m_motionSimOvershootCheck) m_motionSimOvershootCheck->setChecked(obs_data_get_bool(settings, "motion_sim_overshoot"));
    if (m_motionSimMicroOvershootCheck) m_motionSimMicroOvershootCheck->setChecked(obs_data_get_bool(settings, "motion_sim_micro_overshoot"));
    if (m_motionSimInertiaCheck) m_motionSimInertiaCheck->setChecked(obs_data_get_bool(settings, "motion_sim_inertia"));
    if (m_motionSimLeftBtnAdaptiveCheck) m_motionSimLeftBtnAdaptiveCheck->setChecked(obs_data_get_bool(settings, "motion_sim_left_btn_adaptive"));
    if (m_motionSimSprayModeCheck) m_motionSimSprayModeCheck->setChecked(obs_data_get_bool(settings, "motion_sim_spray_mode"));
    if (m_motionSimTapPauseCheck) m_motionSimTapPauseCheck->setChecked(obs_data_get_bool(settings, "motion_sim_tap_pause"));
    if (m_motionSimRetryCheck) m_motionSimRetryCheck->setChecked(obs_data_get_bool(settings, "motion_sim_retry"));
    if (m_motionSimMaxRetrySpin) m_motionSimMaxRetrySpin->setValue(obs_data_get_int(settings, "motion_sim_max_retry"));
    if (m_motionSimDelayMsSpin) m_motionSimDelayMsSpin->setValue(obs_data_get_int(settings, "motion_sim_delay_ms"));
    if (m_motionSimDirectProbSpin) m_motionSimDirectProbSpin->setValue(obs_data_get_double(settings, "motion_sim_direct_prob"));
    if (m_motionSimOvershootProbSpin) m_motionSimOvershootProbSpin->setValue(obs_data_get_double(settings, "motion_sim_overshoot_prob"));
    if (m_motionSimMicroOvshootProbSpin) m_motionSimMicroOvshootProbSpin->setValue(obs_data_get_double(settings, "motion_sim_micro_ovshoot_prob"));
    
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
    if (w.advSpeedFactorSpin) obs_data_set_double(settings, QString("adv_speed_factor_%1").arg(idx).toUtf8().constData(), w.advSpeedFactorSpin->value());
    
    if (w.useOneEuroFilterCheck) obs_data_set_bool(settings, QString("use_one_euro_filter_%1").arg(idx).toUtf8().constData(), w.useOneEuroFilterCheck->isChecked());
    if (w.oneEuroMinCutoffSpin) obs_data_set_double(settings, QString("one_euro_min_cutoff_%1").arg(idx).toUtf8().constData(), w.oneEuroMinCutoffSpin->value());
    if (w.oneEuroBetaSpin) obs_data_set_double(settings, QString("one_euro_beta_%1").arg(idx).toUtf8().constData(), w.oneEuroBetaSpin->value());
    if (w.oneEuroDCutoffSpin) obs_data_set_double(settings, QString("one_euro_d_cutoff_%1").arg(idx).toUtf8().constData(), w.oneEuroDCutoffSpin->value());
    
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
    
    // MotionSimulator 人类行为模拟器设置
    if (m_enableMotionSimulatorCheck) obs_data_set_bool(settings, "enable_motion_simulator", m_enableMotionSimulatorCheck->isChecked());
    if (m_motionSimRandomPosCheck) obs_data_set_bool(settings, "motion_sim_random_pos", m_motionSimRandomPosCheck->isChecked());
    if (m_motionSimOvershootCheck) obs_data_set_bool(settings, "motion_sim_overshoot", m_motionSimOvershootCheck->isChecked());
    if (m_motionSimMicroOvershootCheck) obs_data_set_bool(settings, "motion_sim_micro_overshoot", m_motionSimMicroOvershootCheck->isChecked());
    if (m_motionSimInertiaCheck) obs_data_set_bool(settings, "motion_sim_inertia", m_motionSimInertiaCheck->isChecked());
    if (m_motionSimLeftBtnAdaptiveCheck) obs_data_set_bool(settings, "motion_sim_left_btn_adaptive", m_motionSimLeftBtnAdaptiveCheck->isChecked());
    if (m_motionSimSprayModeCheck) obs_data_set_bool(settings, "motion_sim_spray_mode", m_motionSimSprayModeCheck->isChecked());
    if (m_motionSimTapPauseCheck) obs_data_set_bool(settings, "motion_sim_tap_pause", m_motionSimTapPauseCheck->isChecked());
    if (m_motionSimRetryCheck) obs_data_set_bool(settings, "motion_sim_retry", m_motionSimRetryCheck->isChecked());
    if (m_motionSimMaxRetrySpin) obs_data_set_int(settings, "motion_sim_max_retry", m_motionSimMaxRetrySpin->value());
    if (m_motionSimDelayMsSpin) obs_data_set_int(settings, "motion_sim_delay_ms", m_motionSimDelayMsSpin->value());
    if (m_motionSimDirectProbSpin) obs_data_set_double(settings, "motion_sim_direct_prob", m_motionSimDirectProbSpin->value());
    if (m_motionSimOvershootProbSpin) obs_data_set_double(settings, "motion_sim_overshoot_prob", m_motionSimOvershootProbSpin->value());
    if (m_motionSimMicroOvshootProbSpin) obs_data_set_double(settings, "motion_sim_micro_ovshoot_prob", m_motionSimMicroOvshootProbSpin->value());
    
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
