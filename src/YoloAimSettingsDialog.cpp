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
    setWindowTitle(QStringLiteral("YOLO自瞄设置"));
    setMinimumSize(1000, 600);
    resize(1200, 700);
    
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
    
    QLabel* previewLabel = new QLabel(QStringLiteral("视频预览"), this);
    previewLabel->setAlignment(Qt::AlignCenter);
    previewLayout->addWidget(previewLabel);
    
    m_previewDisplay = new OBSQTDisplay(this);
    m_previewDisplay->setMinimumSize(320, 180);
    m_previewDisplay->SetBackgroundColor(0x1A1A1A);
    previewLayout->addWidget(m_previewDisplay, 1);
    
    m_previewPlaceholder = new QLabel(QStringLiteral("请选择视频源"), this);
    m_previewPlaceholder->setAlignment(Qt::AlignCenter);
    m_previewPlaceholder->setStyleSheet("background-color: #1A1A1A; color: #888888; font-size: 16px;");
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
    
    QGroupBox* modelGroup = new QGroupBox(QStringLiteral("模型设置"), page);
    QFormLayout* modelLayout = new QFormLayout(modelGroup);
    
    layout->addWidget(modelGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("模型"));
}

void YoloAimSettingsDialog::setupVisualPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* displayGroup = new QGroupBox(QStringLiteral("显示设置"), page);
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
    
    m_tabWidget->addTab(page, QStringLiteral("视觉"));
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
    
    m_tabWidget->addTab(page, QStringLiteral("基础设置"));
}

void YoloAimSettingsDialog::setupAdvancedPIDPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* pidGroup = new QGroupBox(QStringLiteral("高级PID参数"), page);
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
    
    QGroupBox* advGroup = new QGroupBox(QStringLiteral("高级参数"), page);
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
    
    QGroupBox* oneEuroGroup = new QGroupBox(QStringLiteral("One Euro Filter"), page);
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
    
    m_tabWidget->addTab(page, QStringLiteral("高级PID"));
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
    
    m_tabWidget->addTab(page, QStringLiteral("扳机"));
}

void YoloAimSettingsDialog::setupTrackingPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    QGroupBox* trackingGroup = new QGroupBox(QStringLiteral("追踪设置"), page);
    QFormLayout* trackingLayout = new QFormLayout(trackingGroup);
    
    layout->addWidget(trackingGroup);
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("追踪"));
}

void YoloAimSettingsDialog::setupPredictorPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    m_configWidgets[0].predictorGroup = createPredictorGroup(0);
    layout->addWidget(m_configWidgets[0].predictorGroup);
    
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("预测"));
}

void YoloAimSettingsDialog::setupBezierPage()
{
    QWidget* page = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(page);
    
    m_configWidgets[0].bezierGroup = createBezierGroup(0);
    layout->addWidget(m_configWidgets[0].bezierGroup);
    
    layout->addStretch();
    
    m_tabWidget->addTab(page, QStringLiteral("贝塞尔"));
}

QWidget* YoloAimSettingsDialog::createConfigWidget(int configIndex)
{
    QWidget* widget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(widget);
    
    QGroupBox* basicGroup = new QGroupBox(QStringLiteral("基础参数"), widget);
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
    QGroupBox* group = new QGroupBox(QStringLiteral("自动扳机"), this);
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
    QGroupBox* group = new QGroupBox(QStringLiteral("后坐力控制"), this);
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
    QGroupBox* group = new QGroupBox(QStringLiteral("预测器"), this);
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
    QGroupBox* group = new QGroupBox(QStringLiteral("贝塞尔曲线移动"), this);
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
    
    m_configWidgets[idx].enabledCheck->setChecked(obs_data_get_bool(settings, QString("enable_config_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].hotkeyCombo->setCurrentIndex(obs_data_get_int(settings, QString("hotkey_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].controllerTypeCombo->setCurrentIndex(obs_data_get_int(settings, QString("controller_type_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].pMinSpin->setValue(obs_data_get_double(settings, QString("p_min_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].pMaxSpin->setValue(obs_data_get_double(settings, QString("p_max_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].pSlopeSpin->setValue(obs_data_get_double(settings, QString("p_slope_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].dSpin->setValue(obs_data_get_double(settings, QString("d_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].iSpin->setValue(obs_data_get_double(settings, QString("i_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].derivativeFilterAlphaSpin->setValue(obs_data_get_double(settings, QString("derivative_filter_alpha_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].advTargetThresholdSpin->setValue(obs_data_get_double(settings, QString("adv_target_threshold_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].advMinCoefficientSpin->setValue(obs_data_get_double(settings, QString("adv_min_coefficient_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].advMaxCoefficientSpin->setValue(obs_data_get_double(settings, QString("adv_max_coefficient_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].advTransitionSharpnessSpin->setValue(obs_data_get_double(settings, QString("adv_transition_sharpness_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].advTransitionMidpointSpin->setValue(obs_data_get_double(settings, QString("adv_transition_midpoint_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].advSpeedFactorSpin->setValue(obs_data_get_double(settings, QString("adv_speed_factor_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].useOneEuroFilterCheck->setChecked(obs_data_get_bool(settings, QString("use_one_euro_filter_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].oneEuroMinCutoffSpin->setValue(obs_data_get_double(settings, QString("one_euro_min_cutoff_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].oneEuroBetaSpin->setValue(obs_data_get_double(settings, QString("one_euro_beta_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].oneEuroDCutoffSpin->setValue(obs_data_get_double(settings, QString("one_euro_d_cutoff_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].aimSmoothingXSpin->setValue(obs_data_get_double(settings, QString("aim_smoothing_x_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].aimSmoothingYSpin->setValue(obs_data_get_double(settings, QString("aim_smoothing_y_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].targetYOffsetSpin->setValue(obs_data_get_double(settings, QString("target_y_offset_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].maxPixelMoveSpin->setValue(obs_data_get_double(settings, QString("max_pixel_move_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].deadZonePixelsSpin->setValue(obs_data_get_double(settings, QString("dead_zone_pixels_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].screenOffsetXSpin->setValue(obs_data_get_int(settings, QString("screen_offset_x_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].screenOffsetYSpin->setValue(obs_data_get_int(settings, QString("screen_offset_y_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].screenWidthSpin->setValue(obs_data_get_int(settings, QString("screen_width_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].screenHeightSpin->setValue(obs_data_get_int(settings, QString("screen_height_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].enableYAxisUnlockCheck->setChecked(obs_data_get_bool(settings, QString("enable_y_axis_unlock_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].yAxisUnlockDelaySpin->setValue(obs_data_get_int(settings, QString("y_axis_unlock_delay_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].autoTriggerGroup->setChecked(obs_data_get_bool(settings, QString("auto_trigger_group_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].triggerRadiusSpin->setValue(obs_data_get_int(settings, QString("trigger_radius_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].triggerCooldownSpin->setValue(obs_data_get_int(settings, QString("trigger_cooldown_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].triggerFireDelaySpin->setValue(obs_data_get_int(settings, QString("trigger_fire_delay_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].triggerFireDurationSpin->setValue(obs_data_get_int(settings, QString("trigger_fire_duration_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].triggerIntervalSpin->setValue(obs_data_get_int(settings, QString("trigger_interval_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].triggerMoveCompensationSpin->setValue(obs_data_get_int(settings, QString("trigger_move_compensation_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].recoilGroup->setChecked(obs_data_get_bool(settings, QString("recoil_group_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].recoilStrengthSpin->setValue(obs_data_get_double(settings, QString("recoil_strength_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].recoilSpeedSpin->setValue(obs_data_get_int(settings, QString("recoil_speed_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].recoilPidGainScaleSpin->setValue(obs_data_get_double(settings, QString("recoil_pid_gain_scale_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].predictorGroup->setChecked(obs_data_get_bool(settings, QString("derivative_predictor_group_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].predictionWeightXSpin->setValue(obs_data_get_double(settings, QString("prediction_weight_x_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].predictionWeightYSpin->setValue(obs_data_get_double(settings, QString("prediction_weight_y_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].maxPredictionTimeSpin->setValue(obs_data_get_double(settings, QString("max_prediction_time_%1").arg(idx).toUtf8().constData()));
    
    m_configWidgets[idx].bezierGroup->setChecked(obs_data_get_bool(settings, QString("bezier_movement_group_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].bezierCurvatureSpin->setValue(obs_data_get_double(settings, QString("bezier_curvature_%1").arg(idx).toUtf8().constData()));
    m_configWidgets[idx].bezierRandomnessSpin->setValue(obs_data_get_double(settings, QString("bezier_randomness_%1").arg(idx).toUtf8().constData()));
    
    m_showDetectionResultsCheck->setChecked(obs_data_get_bool(settings, "show_detection_results"));
    m_showFOVCheck->setChecked(obs_data_get_bool(settings, "show_fov"));
    m_showFOVCircleCheck->setChecked(obs_data_get_bool(settings, "show_fov_circle"));
    m_showFOVCrossCheck->setChecked(obs_data_get_bool(settings, "show_fov_cross"));
    m_fovRadiusSpin->setValue(obs_data_get_int(settings, "fov_radius"));
    m_fovCrossLineScaleSpin->setValue(obs_data_get_int(settings, "fov_cross_line_scale"));
    m_fovCrossLineThicknessSpin->setValue(obs_data_get_int(settings, "fov_cross_line_thickness"));
    m_fovCircleThicknessSpin->setValue(obs_data_get_int(settings, "fov_circle_thickness"));
    
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
    
    obs_data_set_bool(settings, QString("enable_config_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].enabledCheck->isChecked());
    obs_data_set_int(settings, QString("hotkey_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].hotkeyCombo->currentIndex());
    obs_data_set_int(settings, QString("controller_type_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].controllerTypeCombo->currentIndex());
    
    obs_data_set_double(settings, QString("p_min_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].pMinSpin->value());
    obs_data_set_double(settings, QString("p_max_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].pMaxSpin->value());
    obs_data_set_double(settings, QString("p_slope_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].pSlopeSpin->value());
    obs_data_set_double(settings, QString("d_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].dSpin->value());
    obs_data_set_double(settings, QString("i_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].iSpin->value());
    obs_data_set_double(settings, QString("derivative_filter_alpha_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].derivativeFilterAlphaSpin->value());
    
    obs_data_set_double(settings, QString("adv_target_threshold_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].advTargetThresholdSpin->value());
    obs_data_set_double(settings, QString("adv_min_coefficient_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].advMinCoefficientSpin->value());
    obs_data_set_double(settings, QString("adv_max_coefficient_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].advMaxCoefficientSpin->value());
    obs_data_set_double(settings, QString("adv_transition_sharpness_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].advTransitionSharpnessSpin->value());
    obs_data_set_double(settings, QString("adv_transition_midpoint_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].advTransitionMidpointSpin->value());
    obs_data_set_double(settings, QString("adv_speed_factor_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].advSpeedFactorSpin->value());
    
    obs_data_set_bool(settings, QString("use_one_euro_filter_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].useOneEuroFilterCheck->isChecked());
    obs_data_set_double(settings, QString("one_euro_min_cutoff_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].oneEuroMinCutoffSpin->value());
    obs_data_set_double(settings, QString("one_euro_beta_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].oneEuroBetaSpin->value());
    obs_data_set_double(settings, QString("one_euro_d_cutoff_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].oneEuroDCutoffSpin->value());
    
    obs_data_set_double(settings, QString("aim_smoothing_x_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].aimSmoothingXSpin->value());
    obs_data_set_double(settings, QString("aim_smoothing_y_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].aimSmoothingYSpin->value());
    obs_data_set_double(settings, QString("target_y_offset_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].targetYOffsetSpin->value());
    obs_data_set_double(settings, QString("max_pixel_move_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].maxPixelMoveSpin->value());
    obs_data_set_double(settings, QString("dead_zone_pixels_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].deadZonePixelsSpin->value());
    
    obs_data_set_int(settings, QString("screen_offset_x_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].screenOffsetXSpin->value());
    obs_data_set_int(settings, QString("screen_offset_y_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].screenOffsetYSpin->value());
    obs_data_set_int(settings, QString("screen_width_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].screenWidthSpin->value());
    obs_data_set_int(settings, QString("screen_height_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].screenHeightSpin->value());
    
    obs_data_set_bool(settings, QString("enable_y_axis_unlock_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].enableYAxisUnlockCheck->isChecked());
    obs_data_set_int(settings, QString("y_axis_unlock_delay_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].yAxisUnlockDelaySpin->value());
    
    obs_data_set_bool(settings, QString("auto_trigger_group_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].autoTriggerGroup->isChecked());
    obs_data_set_int(settings, QString("trigger_radius_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].triggerRadiusSpin->value());
    obs_data_set_int(settings, QString("trigger_cooldown_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].triggerCooldownSpin->value());
    obs_data_set_int(settings, QString("trigger_fire_delay_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].triggerFireDelaySpin->value());
    obs_data_set_int(settings, QString("trigger_fire_duration_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].triggerFireDurationSpin->value());
    obs_data_set_int(settings, QString("trigger_interval_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].triggerIntervalSpin->value());
    obs_data_set_int(settings, QString("trigger_move_compensation_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].triggerMoveCompensationSpin->value());
    
    obs_data_set_bool(settings, QString("recoil_group_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].recoilGroup->isChecked());
    obs_data_set_double(settings, QString("recoil_strength_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].recoilStrengthSpin->value());
    obs_data_set_int(settings, QString("recoil_speed_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].recoilSpeedSpin->value());
    obs_data_set_double(settings, QString("recoil_pid_gain_scale_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].recoilPidGainScaleSpin->value());
    
    obs_data_set_bool(settings, QString("derivative_predictor_group_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].predictorGroup->isChecked());
    obs_data_set_double(settings, QString("prediction_weight_x_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].predictionWeightXSpin->value());
    obs_data_set_double(settings, QString("prediction_weight_y_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].predictionWeightYSpin->value());
    obs_data_set_double(settings, QString("max_prediction_time_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].maxPredictionTimeSpin->value());
    
    obs_data_set_bool(settings, QString("bezier_movement_group_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].bezierGroup->isChecked());
    obs_data_set_double(settings, QString("bezier_curvature_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].bezierCurvatureSpin->value());
    obs_data_set_double(settings, QString("bezier_randomness_%1").arg(idx).toUtf8().constData(), m_configWidgets[idx].bezierRandomnessSpin->value());
    
    obs_data_set_bool(settings, "show_detection_results", m_showDetectionResultsCheck->isChecked());
    obs_data_set_bool(settings, "show_fov", m_showFOVCheck->isChecked());
    obs_data_set_bool(settings, "show_fov_circle", m_showFOVCircleCheck->isChecked());
    obs_data_set_bool(settings, "show_fov_cross", m_showFOVCrossCheck->isChecked());
    obs_data_set_int(settings, "fov_radius", m_fovRadiusSpin->value());
    obs_data_set_int(settings, "fov_cross_line_scale", m_fovCrossLineScaleSpin->value());
    obs_data_set_int(settings, "fov_cross_line_thickness", m_fovCrossLineThicknessSpin->value());
    obs_data_set_int(settings, "fov_circle_thickness", m_fovCircleThicknessSpin->value());
    
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
