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
{
    initAllConfigWidgetStructs();
    
    setWindowTitle(QStringLiteral("🎮 YOLO自瞄系统"));
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
    setupVisualPage();
    setupBasicPage();
    setupAdvancedPIDPage();
    setupTriggerPage();
    setupTrackingPage();
    setupPredictorPage();
    setupBezierPage();
    
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
    QFormLayout* modelLayout = new QFormLayout(modelGroup);
    
    layout->addWidget(modelGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("📦 模型"));
}

void YoloAimSettingsDialog::setupVisualPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* displayGroup = new QGroupBox(QStringLiteral("👁 显示设置"), page);
    QGridLayout* displayLayout = new QGridLayout(displayGroup);
    
    int row = 0;
    
    m_showDetectionResultsCheck = new QCheckBox(QStringLiteral("显示检测结果"), page);
    displayLayout->addWidget(m_showDetectionResultsCheck, row, 0, 1, 2);
    row++;
    
    m_showFOVCheck = new QCheckBox(QStringLiteral("显示FOV"), page);
    displayLayout->addWidget(m_showFOVCheck, row, 0, 1, 2);
    row++;
    
    m_showFOVCircleCheck = new QCheckBox(QStringLiteral("显示FOV圆圈"), page);
    displayLayout->addWidget(m_showFOVCircleCheck, row, 0);
    
    m_showFOVCrossCheck = new QCheckBox(QStringLiteral("显示FOV十字"), page);
    displayLayout->addWidget(m_showFOVCrossCheck, row, 1);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("FOV半径:"), page), row, 0);
    m_fovRadiusSpin = new QSpinBox(page);
    m_fovRadiusSpin->setRange(10, 500);
    displayLayout->addWidget(m_fovRadiusSpin, row, 1);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("十字线长度:"), page), row, 0);
    m_fovCrossLineScaleSpin = new QSpinBox(page);
    m_fovCrossLineScaleSpin->setRange(5, 200);
    displayLayout->addWidget(m_fovCrossLineScaleSpin, row, 1);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("十字线粗细:"), page), row, 0);
    m_fovCrossLineThicknessSpin = new QSpinBox(page);
    m_fovCrossLineThicknessSpin->setRange(1, 10);
    displayLayout->addWidget(m_fovCrossLineThicknessSpin, row, 1);
    row++;
    
    displayLayout->addWidget(new QLabel(QStringLiteral("圆圈粗细:"), page), row, 0);
    m_fovCircleThicknessSpin = new QSpinBox(page);
    m_fovCircleThicknessSpin->setRange(1, 10);
    displayLayout->addWidget(m_fovCircleThicknessSpin, row, 1);
    
    layout->addWidget(displayGroup);
    layout->addStretch();
    
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
    
    QGroupBox* trackingGroup = new QGroupBox(QStringLiteral("📍 追踪设置"), page);
    QFormLayout* trackingLayout = new QFormLayout(trackingGroup);
    
    layout->addWidget(trackingGroup);
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
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "yolo-aim-filter-hidden");
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
    
    obs_data_release(settings);
    obs_source_release(filter);
    obs_source_release(source);
}

void YoloAimSettingsDialog::saveSettings()
{
    if (m_currentSource.isEmpty()) return;
    
    obs_source_t* source = obs_get_source_by_name(m_currentSource.toUtf8().constData());
    if (!source) return;
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "yolo-aim-filter-hidden");
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
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "yolo-aim-filter-hidden");
    if (!filter) {
        obs_data_t* settings = obs_data_create();
        filter = obs_source_create("yolo-detector-filter", "yolo-aim-filter-hidden", settings, nullptr);
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
    
    obs_source_t* filter = obs_source_get_filter_by_name(source, "yolo-aim-filter-hidden");
    if (filter) {
        obs_source_filter_remove(source, filter);
        obs_source_release(filter);
    }
    
    obs_source_release(source);
}

#endif
