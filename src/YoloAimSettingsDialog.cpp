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
{
    setWindowTitle(QStringLiteral("YOLO自瞄设置"));
    setMinimumSize(800, 600);
    resize(900, 700);
    
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
    
    m_tabWidget = new QTabWidget(this);
    
    setupModelPage();
    setupVisualPage();
    setupBasicPage();
    setupAdvancedPIDPage();
    setupTriggerPage();
    setupTrackingPage();
    setupPredictorPage();
    setupBezierPage();
    
    mainLayout->addWidget(m_tabWidget);
    
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
    
    mainLayout->addLayout(configLayout);
    
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
    
    QGroupBox* fovGroup = new QGroupBox(QStringLiteral("FOV设置"), page);
    QFormLayout* fovLayout = new QFormLayout(fovGroup);
    
    layout->addWidget(fovGroup);
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
    
    QGroupBox* colorAimGroup = createColorAimGroup(0);
    layout->addWidget(colorAimGroup);
    
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

QGroupBox* YoloAimSettingsDialog::createColorAimGroup(int configIndex)
{
    QGroupBox* group = new QGroupBox(QStringLiteral("找色自瞄设置"), this);
    QGridLayout* layout = new QGridLayout(group);
    
    int row = 0;
    
    layout->addWidget(new QLabel(QStringLiteral("检测模式:"), this), row, 0);
    m_configWidgets[configIndex].detectionModeCombo = new QComboBox(this);
    m_configWidgets[configIndex].detectionModeCombo->addItem(QStringLiteral("YOLO检测"), 0);
    m_configWidgets[configIndex].detectionModeCombo->addItem(QStringLiteral("找色自瞄"), 1);
    m_configWidgets[configIndex].detectionModeCombo->addItem(QStringLiteral("混合模式"), 2);
    layout->addWidget(m_configWidgets[configIndex].detectionModeCombo, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("预设颜色:"), this), row, 2);
    m_configWidgets[configIndex].presetColorCombo = new QComboBox(this);
    m_configWidgets[configIndex].presetColorCombo->addItem(QStringLiteral("自定义"), 0);
    m_configWidgets[configIndex].presetColorCombo->addItem(QStringLiteral("红色"), 1);
    m_configWidgets[configIndex].presetColorCombo->addItem(QStringLiteral("绿色"), 2);
    m_configWidgets[configIndex].presetColorCombo->addItem(QStringLiteral("蓝色"), 3);
    m_configWidgets[configIndex].presetColorCombo->addItem(QStringLiteral("黄色"), 4);
    m_configWidgets[configIndex].presetColorCombo->addItem(QStringLiteral("青色"), 5);
    m_configWidgets[configIndex].presetColorCombo->addItem(QStringLiteral("品红"), 6);
    layout->addWidget(m_configWidgets[configIndex].presetColorCombo, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("H下限:"), this), row, 0);
    m_configWidgets[configIndex].colorHMinSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorHMinSpin->setRange(0, 180);
    layout->addWidget(m_configWidgets[configIndex].colorHMinSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("H上限:"), this), row, 2);
    m_configWidgets[configIndex].colorHMaxSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorHMaxSpin->setRange(0, 180);
    layout->addWidget(m_configWidgets[configIndex].colorHMaxSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("S下限:"), this), row, 0);
    m_configWidgets[configIndex].colorSMinSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorSMinSpin->setRange(0, 255);
    layout->addWidget(m_configWidgets[configIndex].colorSMinSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("S上限:"), this), row, 2);
    m_configWidgets[configIndex].colorSMaxSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorSMaxSpin->setRange(0, 255);
    layout->addWidget(m_configWidgets[configIndex].colorSMaxSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("V下限:"), this), row, 0);
    m_configWidgets[configIndex].colorVMinSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorVMinSpin->setRange(0, 255);
    layout->addWidget(m_configWidgets[configIndex].colorVMinSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("V上限:"), this), row, 2);
    m_configWidgets[configIndex].colorVMaxSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorVMaxSpin->setRange(0, 255);
    layout->addWidget(m_configWidgets[configIndex].colorVMaxSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("形态学核:"), this), row, 0);
    m_configWidgets[configIndex].morphKernelSizeSpin = new QSpinBox(this);
    m_configWidgets[configIndex].morphKernelSizeSpin->setRange(1, 15);
    layout->addWidget(m_configWidgets[configIndex].morphKernelSizeSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("迭代次数:"), this), row, 2);
    m_configWidgets[configIndex].morphIterationsSpin = new QSpinBox(this);
    m_configWidgets[configIndex].morphIterationsSpin->setRange(1, 10);
    layout->addWidget(m_configWidgets[configIndex].morphIterationsSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("子矩阵大小:"), this), row, 0);
    m_configWidgets[configIndex].subMatrixSizeSpin = new QSpinBox(this);
    m_configWidgets[configIndex].subMatrixSizeSpin->setRange(8, 128);
    layout->addWidget(m_configWidgets[configIndex].subMatrixSizeSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("分位数阈值:"), this), row, 2);
    m_configWidgets[configIndex].quantileThresholdSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].quantileThresholdSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].quantileThresholdSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].quantileThresholdSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("模板阈值:"), this), row, 0);
    m_configWidgets[configIndex].templateThresholdSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].templateThresholdSpin->setRange(0.0, 1.0);
    m_configWidgets[configIndex].templateThresholdSpin->setDecimals(2);
    layout->addWidget(m_configWidgets[configIndex].templateThresholdSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("最小面积:"), this), row, 2);
    m_configWidgets[configIndex].minDetectionAreaSpin = new QDoubleSpinBox(this);
    m_configWidgets[configIndex].minDetectionAreaSpin->setRange(1.0, 500.0);
    layout->addWidget(m_configWidgets[configIndex].minDetectionAreaSpin, row, 3);
    row++;
    
    layout->addWidget(new QLabel(QStringLiteral("FOV宽度:"), this), row, 0);
    m_configWidgets[configIndex].colorFovWidthSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorFovWidthSpin->setRange(100, 800);
    layout->addWidget(m_configWidgets[configIndex].colorFovWidthSpin, row, 1);
    
    layout->addWidget(new QLabel(QStringLiteral("FOV高度:"), this), row, 2);
    m_configWidgets[configIndex].colorFovHeightSpin = new QSpinBox(this);
    m_configWidgets[configIndex].colorFovHeightSpin->setRange(100, 800);
    layout->addWidget(m_configWidgets[configIndex].colorFovHeightSpin, row, 3);
    
    return group;
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
    }
    
    m_currentSource = newSource;
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
}

void YoloAimSettingsDialog::saveSettings()
{
}

void YoloAimSettingsDialog::applySettings()
{
}

void YoloAimSettingsDialog::refreshConfigUI()
{
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
