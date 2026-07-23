#ifndef MOUSE_CONTROLLER_INTERFACE_HPP
#define MOUSE_CONTROLLER_INTERFACE_HPP

#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include "models/Detection.h"

enum class ControllerType {
    WindowsAPI,
    MAKCU,
    LogiDriver,
    GvInput,
    TencInput,
    NtUserSendInput,
    NtUserInjectMouse,
    NtUserInjectPointer
};


enum class AlgorithmType {
    AdvancedPID,      // 0: 高级PID（动态P增益+卡尔曼滤波）
    ExternalPID,      // 1: 专业PID（外部库逆向重构）
    AimController,    // 2: aim 控制器（增量式PID+运动预测+柏林噪声）
    SlewRate,         // 3: SlewRate（限速平滑趋近）
    AdaptivePID       // 4: 自适应PID（位置式+自适应积分增益+积分死区）
};

// PID闁轰胶澧楀畵渚€宕堕悙鍓佹闁告垼濮ら弳鐔虹尵鐠囪尙鈧?
struct PidDebugData {
    // === 闁糕晞娅ｉ、鍛村极閻楀牆绁?===
    float errorX = 0;
    float errorY = 0;
    float outputX = 0;
    float outputY = 0;
    float targetX = 0;
    float targetY = 0;
    float targetVelocityX = 0;
    float targetVelocityY = 0;
    float currentKp = 0;
    float currentKi = 0;
    float currentKd = 0;

    // === P/I/D 闁告帒妫濋妴宥嗘綇閹惧啿姣?===
    float pTermX = 0;
    float pTermY = 0;
    float iTermX = 0;
    float iTermY = 0;
    float dTermX = 0;
    float dTermY = 0;

    // === 缂佸鍨伴崹搴ㄥ磻閵夈儲鍊ｉ柣妯垮煐閳?===
    float integralAbsX = 0;
    float integralAbsY = 0;
    float integralLimitX = 1.0f;
    float integralLimitY = 1.0f;
    float integralRatioX = 0;
    float integralRatioY = 0;

    // === 闁硅矇鍐ㄧ厬婵☆垪鈧磭纭€闁煎浜滄慨鈺冩嫚婵犲啯鐒?===
    int controlMode = 0;   // 0=IDLE 1=TRACKING 2=LOCKED 3=I_SATURATION 4=OSCILLATING 5=PREDICTING
    int algorithmType = 0; // 0=AdvancedPID 1=ExternalPID 5=NeuralPath 6=AimController

    // === 濡増绻傞ˇ鑽ゆ嫚婵犲啯鐒藉ǎ鍥ｅ墲娴?===
    bool isFiring = false;
    float smoothingFactorX = 0;
    float smoothingFactorY = 0;

    // === AimController 濞戞挻鎸鹃弫銈囨嫬閸愵厾妲搁悗娑欘殕椤?===
    float aimPredictedX = 0;   // 閺夆晜鍔曟慨鈺傦紣閸曨剛銈撮柣?X 閺夌偞娼欐禍鍝ョ矓?
    float aimPredictedY = 0;   // 閺夆晜鍔曟慨鈺傦紣閸曨剛銈撮柣?Y 閺夌偞娼欐禍鍝ョ矓?
    float aimFusedX = 0;       // 闁捐绉撮幃搴ㄥ触鎼达絾鐣?X 閺夌偟顥愰銈咁啅椤曞棛绀勯柛妯煎枎椤?濡澘瀚粊?闁哄洩灏欓崵搴ㄦ晬?
    float aimFusedY = 0;       // 闁捐绉撮幃搴ㄥ触鎼达絾鐣?Y 閺夌偟顥愰銈咁啅?
    float aimCurveLen = 0;     // 闁捐绉撮幃搴ｆ嫚椤栨碍鈻曢柛姘灴閸ｆ椽姊归崹顔碱唺

    // === 闁哄啫鐖煎Λ鍧楀箣缁涘湱绀勯柣顫妺缁剟宕㈤崱妤€钑夐悹浣规緲缂嶅秹鏁?===
    std::chrono::steady_clock::time_point timestamp;
};
using PidDataCallback = std::function<void(const PidDebugData&)>;

struct MouseControllerConfig {
    bool enableMouseControl = false;
    int hotkeyVirtualKey = 0;
    int fovRadiusPixels = 100;
    float sourceCanvasPosX = 0.0f;
    float sourceCanvasPosY = 0.0f;
    float sourceCanvasScaleX = 1.0f;
    float sourceCanvasScaleY = 1.0f;
    int sourceWidth = 1920;
    int sourceHeight = 1080;
    int inferenceFrameWidth = 0;
    int inferenceFrameHeight = 0;
    int cropOffsetX = 0;
    int cropOffsetY = 0;
    int screenOffsetX = 0;
    int screenOffsetY = 0;
    int screenWidth = 0;
    int screenHeight = 0;
    
    // 缂佺姵顨嗙涵鍫曟焻婢跺顏?
    AlgorithmType algorithmType = AlgorithmType::AdvancedPID;
    
    // 濡ゅ倹顭囨鍢滻D闁告瑥鍊归弳鐔兼晬閸垻缈辩紒鐘亾闁绘鐗炵槐?
    float pidPMin = 0.15f;
    float pidPMax = 0.6f;
    float pidPSlope = 1.0f;
    float pidD = 0.007f;
    float pidI = 0.01f;
    float maxPixelMove = 128.0f;
    float deadZonePixels = 5.0f;
    float targetYOffset = 0.0f;
    float derivativeFilterAlpha = 0.2f;
    // 濡ゅ倹顭囨鍢滻D闁告瑯鍨甸惃鐔煎矗閸屾稒娈堕柨娑樼墕椤曨喗鎯旈弬鍓х懇濞戞挻鐡岻D闁汇劌鍩edict闁告粌顔慳te闁?
    float adaptivePGainRate = 0.03f;      // 闁煎浜埀顒€鍊哥花鐫靛褏鍋熷▔顓㈠矗濡搫顕ч柣婊冩祫缁辨瑧鈧數鎳撶花鍙夌▔閹捐尙鐟筆ID闁汇劌鍩edict闁?
    float dTermScale = 0.3f;              // D濡炪倕婀辩紓澶愬绩閹勭閻庢稒鍔х槐娆戔偓鐢垫嚀缁ㄥ弶绋夐幘鑼懝PID闁汇劌澧攁te闁?
    // 婵炲鐓夌槐浼存嚊椤忓牃鍋撻崒姘卞畨濠⒀呭仧濞夘參濡存担绋垮耿閻忓繑姊瑰ù鏍ь煥閵堝棗鐨鹃柕鍡曟祰閻戯箓宕ｅΟ娲绘⒕婵?30闁稿秴绻掔粈?闁靛棔娴囬拏瀣⒔閹邦剛鐣?atan2)濠殿喖顑囩划鎾诲触椤栨粍鏆忛柨娑樿嫰閸庢碍绋夐幘鑼懝PID濞戞挴鍋撻柡宥囨焿閸ゆ粓宕濋妸銊х閻?
    
    ControllerType controllerType = ControllerType::WindowsAPI;
    std::string makcuPort;
    int makcuBaudRate = 115200;
    int logiDriverType = 0;  // LogiDriver閻庢稒鍔楃悮顐﹀垂鐎ｅ墎绐?=闁煎浜滄慨? 1=GHUB, 2=LGS, 3=Razer
    int yUnlockDelayMs = 300;
    bool yUnlockEnabled = false;
    bool autoTriggerEnabled = false;
    int autoTriggerRadius = 5;
    int autoTriggerCooldownMs = 200;
    int autoTriggerFireDelay = 0;
    int autoTriggerFireDuration = 50;
    int autoTriggerInterval = 50;
    bool autoTriggerDelayRandomEnabled = false;
    int autoTriggerDelayRandomMin = 0;
    int autoTriggerDelayRandomMax = 0;
    bool autoTriggerDurationRandomEnabled = false;
    int autoTriggerDurationRandomMin = 0;
    int autoTriggerDurationRandomMax = 0;
    int autoTriggerMoveCompensation = 0;
    int targetSwitchDelayMs = 500;
    float targetSwitchTolerance = 0.15f;
    float integralLimit = 100.0f;
    float integralRate = 1.0f;             // 缂佸鍨伴崹搴ㄦ焻閻斿搫鑺崇紒顖滅帛閺?
    float pGainRampInitialScale = 0.6f;
    float pGainRampDuration = 0.5f;
    // DerivativePredictor配置
    bool useDerivativePredictor = true;
    float predictionWeightX = 0.5f;
    float predictionWeightY = 0.1f;
    float maxPredictionTime = 0.1f;

    // Smith预估器（纯滞后补偿控制）
    bool smithPredictorEnabled = false;
    float smithModelGain = 1.0f;
    float smithModelTau = 0.02f;
    bool smithAutoTau = true;

    // IMM交互多模型滤波器（陈金广《目标跟踪系统中的滤波方法》）
    bool immFilterEnabled = false;
    float immProcessNoisePos = 0.1f;
    float immProcessNoiseVel = 0.5f;
    float immProcessNoiseAcc = 1.0f;
    float immProcessNoiseTurn = 0.1f;
    float immMeasurementNoiseX = 1.0f;
    float immMeasurementNoiseY = 1.0f;
    int immActiveModels = 3;

    // OneEuro 误差滤波（压检测/关联抖动，快移时自动提高截止频率）
    bool useOneEuroFilter = false;
    float oneEuroMinCutoff = 1.0f;
    float oneEuroBeta = 0.007f;
    float oneEuroDCutoff = 1.0f;

    // SlewRate控制器（限速平滑趋近）
    bool slewRateEnabled = false;
    float slewRateOutputGain = 0.25f;
    float slewRateResponseSmoothing = 0.0008f;
    float slewRateApproachDamping = 5.0f;
    float slewRateUpdateIntervalMs = 5.0f;
    float slewRateNormalizationScale = 5.0f;

    // 自适应PID控制器（位置式+自适应积分增益+积分死区+双重抗饱和）
    float adaptivePidKp = 1.0f;                    // 比例增益
    float adaptivePidKi = 0.1f;                    // 积分增益
    float adaptivePidKd = 0.05f;                   // 微分增益
    float adaptivePidDeadZone = 0.3f;              // 输入死区阈值
    float adaptivePidIntegralLimit = 100.0f;       // 积分限幅
    float adaptivePidIntegralDeadzone = 1.0f;      // 积分死区阈值
    float adaptivePidIntegralGainThreshold = 50.0f; // 积分自适应阈值
    float adaptivePidIntegralGainRate = 0.015f;    // 积分自适应速率
    float adaptivePidOutputLimit = 10.0f;          // 输出限幅

    // 连续瞄准与弹道控制
    bool continuousAimEnabled = false;
    bool autoRecoilControlEnabled = false;
    float recoilStrength = 5.0f;
    int recoilSpeed = 16;
    float recoilPidGainScale = 0.3f;
    
    // 閻犳劖绻傞、锝囦焊閺冣偓濞插摜鐥捄銊愨晠宕濋妸銉ユ闁?
    bool enableBezierMovement = false;
    float bezierCurvature = 0.3f;
    float bezierRandomness = 0.2f;
    
    // GhostTracker闁哄洩灏欓崵搴㈡姜閵娿劍濮夐柛娆忓€归弳?
    bool enableGhostTracker = false;        // 闁哄嫷鍨伴幆渚€宕ラ婊勬殢闁哄洩灏欓崵搴㈡姜閵娿劍濮?
    float ghostCurvature = 0.5f;            // 闁哄洩灏欓崵搴☆嚕閸濆嫬顔?(0.0 - 1.0)
    float ghostNoiseIntensity = 12.0f;      // 闁哄牃鍋撳鍫嗗啯鐝ㄥ瑙勬緲閸庢氨妲?
    float ghostVerticalSnapRatio = 3.0f;    // 闁搞劌鍊诲ú鍧楀触閹间焦顎嶆慨锝嗘煣缁躲儵姊奸崼婵冨亾?
    float ghostNoiseFreq = 0.8f;            // Perlin闁革綆浜滈敍鎰紣閹寸姴鑺?
    
    // 濠㈣埖鐗犻崕纰D闁告瑥鍊归弳鐔兼晬閸ь湳d_x64.lib闁?
    float externalKpX = 1.5f;                  // X閺夌偛鐡ㄩ惁顔界瑹鐎ｎ剟鍏囬柡?
    float externalKiX = 0.0f;                  // X閺夌偛顕ⅶ闁告帒妫涢柈鎾极?
    float externalKdX = 1.5f;                  // X閺夌偞娼欐禍鏇㈠礆閸℃瑩鍏囬柡?
    float externalKpY = 1.5f;                  // Y閺夌偛鐡ㄩ惁顔界瑹鐎ｎ剟鍏囬柡?
    float externalKiY = 0.0f;                  // Y閺夌偛顕ⅶ闁告帒妫涢柈鎾极?
    float externalKdY = 1.5f;                  // Y閺夌偞娼欐禍鏇㈠礆閸℃瑩鍏囬柡?
    float externalPredictX = 1.0f;             // X閺夌偞鎸抽。鈺伱圭€ｎ亜妫橀柡?
    float externalPredictY = 1.0f;             // Y閺夌偞鎸抽。鈺伱圭€ｎ亜妫橀柡?
    float externalRateX = 0.3f;                // X閺夌偞鎸抽崳浼村冀妞嬪骸鑺?
    float externalRateY = 0.3f;                // Y閺夌偞鎸抽崳浼村冀妞嬪骸鑺?
    float externalKiMode = 1.0f;               // 缂佸鍨伴崹搴∥熼垾宕囩
    float externalKpLimit = 9900.0f;           // P濡炪倕缍婂娲嵁?
    float externalKiLimit = 9900.0f;           // I濡炪倕缍婂娲嵁?
    float externalKdLimit = 9900.0f;           // D濡炪倕缍婂娲嵁?
    float externalOutputLimit = 0.0f;          // 閺夊牊鎸搁崵顓㈡⒔閹邦剛鐣介柨?=濞戞挸绉瑰娲嵁閸滃啰绀?
    float externalKiRate = 0.05f;              // 缂佸鍨伴崹搴ㄦ焻閻斿搫鑺?
    float externalKiDeadband = 0.5f;           // 缂佸鍨伴崹搴☆潰鐠囨彃闅?

    // aim 闁硅矇鍐ㄧ厬闁革絻鍔屽顒勫极鐢喚绀勫褏鍋ら崳鍝勵嚕瀵ゆ换D+閺夆晜鍔曟慨鈺傦紣閸曨剛銈?闁哄苯绻戦悘鍕闯椤忓嫸绱ｉ柨娑樿嫰閻ｎ剟寮€电顣奸柨?
    float aimKp = 0.6f;                        // 婵絾鏌х欢銉︽櫠閻愬灚鎶?
    float aimKi = 0.01f;                       // 缂佸鍨伴崹搴㈡櫠閻愬灚鎶?
    float aimKd = 0.007f;                      // 鐎甸偊鍠栭崹搴㈡櫠閻愬灚鎶?
    bool  aimNoiseEnabled = false;             // 闁哄嫷鍨伴幆渚€宕ラ婊勬殢濞存粏娅ｇ悮顐﹀礌閺嶃劌顫嶉柛鏃戠厜缁辨瑩寮昏箛鏃備簻闁革綆浜滈敍鎰版晬?
    float aimNoiseAmplitude = 2.0f;            // 闁革綆浜滈敍鎰扮嵁閸涱厼顔婇柨娑樼墕閸庢氨妲愰悪鍛濞?aimNoiseEnabled=true 闁哄啫澧庨弫鎾诲极閸剛绀?
    float aimPredictionWeightX = 0.3f;         // X閺夌偞鎸抽。鈺伱圭€ｎ偅缍€闂?
    float aimPredictionWeightY = 0.1f;         // Y閺夌偞鎸抽。鈺伱圭€ｎ偅缍€闂?
    float aimRampTime = 0.3f;                  // 婵炴挻鍔曢崣鍡涘籍閸洘锛熼柨娑樼墢椤鏁嶇仦鑲╃煠 aimInitScale 闁?1.0闁?
    float aimInitScale = 0.6f;                 // 闁告帗绻傞～鎰綇閹惧啿姣夌紓鍌楁櫆閺備線鏁?-1闁挎稑鏈粭搴ㄥ礂閵夈劍宕抽柣鎰缁?
    float aimOutputMax = 128.0f;               // 闁哄牃鍋撳鍫嗗棛缈婚柛鎴濇惈缁犳瑦鎯?

    // 濠㈣埖纰嶇€垫岸寮介崶顏嗏偓娲触閸絾瀚归棅顏嶄簼濞煎牓鏌?
    float trackingWeightIou = 0.4f;       // IoU閻犵儤绻勯‖鍥级閸愵喖娅?
    float trackingWeightCenter = 0.3f;    // 濞戞搩鍘肩缓楣冩倷绾懐鐛╃紒鍌滅帛濞煎牓鏌?
    float trackingWeightAspect = 0.15f;   // 閻庨€涚矙閻濐喖袙閺冨洨鐛╃紒鍌滅帛濞煎牓鏌?
    float trackingWeightArea = 0.15f;     // 闂傚牄鍨昏ⅶ閻犵儤绻勯‖鍥级閸愵喖娅?
    
    // 缂佷胶鍋熺划锛勭磾閹寸姷鎹曢弶鐐姀閹舵鎮介悢绋跨亣闁革絻鍔戦崢銈囩磾?
    bool enableNeuralPath = false;             // 闁哄嫷鍨伴幆渚€宕ラ婊勬殢缂佷胶鍋熺划锛勭磾閹寸姷鎹曢弶鐐姀閹?
    int neuralPathPoints = 25;                 // 閺夌偑鍔忛幎妤呮倷鐟欏嫭娈堕梺?
    double neuralMouseStepSize = 8.0;          // 濮捬呭У閻栵絽顫㈤妷鈺傛瘣
    int neuralTargetRadius = 8;                // 闁烩晩鍠楅悥锝夊础婵犲倻绐為柨娑樼墕閸╁本娼忛幆褍鐏查悗瑙勭啲缁?
    int neuralConsumePerFrame = 2;             // 婵絽绻愰幎姘槈閸絽鐎悹渚灠缁剁偤鎮欑憴鍕闁挎稑鐗嗘慨鐐烘焻閻旂鈷旈悶娑樼焿缁?
    bool enableNeuralPathDebug = false;        // 闁哄嫷鍨伴幆浣规綇閹惧啿姣夌紒浣哄仧缁紕绱旈幋鐘垫崟閻犲鍟抽惁顖炲籍閵夈儳绠?
    
    // 闁哄啫鐖煎Λ鍧楁儎缁嬪灝褰犵紒澶庮嚙婵晠鏌婂鍥╂瀭闁挎稑鐗嗛幎姘舵偝閸ヮ€宕戦崠锛勭
    bool enableTimeBasedMovement = true;       // 闁哄嫷鍨伴幆渚€宕ラ婊勬殢闁哄啫鐖煎Λ鍧楁儎缁嬪灝褰犵紒澶庮嚙婵晠鏁嶉崼婵囧闁绘粌娲╄棢闁稿灏呯槐?
    float targetFrameRate = 60.0f;             // 闁烩晩鍠楅悥锝囨暜瑜忓濂稿春閸濆嫬娅欓柨娑樼墢閺併倖绂嶆惔銈庡悁缂佺姵顨嗗鍌炴⒒閺夋垶绀堥悗娑欏姧缁?

};

class MouseControllerInterface {
public:
    virtual ~MouseControllerInterface() = default;

    virtual void updateConfig(const MouseControllerConfig& config) = 0;
    
    virtual MouseControllerConfig getConfig() const = 0;

    virtual void setDetections(const std::vector<Detection>& detections) = 0;

    virtual void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) = 0;
    virtual void setDetectionsWithFrameSize(std::vector<Detection>&& detections, int frameWidth, int frameHeight, int cropX, int cropY) {
        setDetectionsWithFrameSize(static_cast<const std::vector<Detection>&>(detections), frameWidth, frameHeight, cropX, cropY);
    }

    virtual void tick() = 0;
    
    virtual void setCurrentWeapon(const std::string& weaponName) = 0;
    virtual std::string getCurrentWeapon() const = 0;
    virtual ControllerType getControllerType() const = 0;
    virtual void setInferenceTimeMs(float ms) = 0;

    // 閻犱礁澧介悿鍝朓D闁轰胶澧楀畵渚€宕堕悙鍓佹闁告垼濮ら弳鐔兼晬閸垺鏆忓ù婊冩唉閻ㄧ喓鎷犻弴鐐茶閻熸瑥妫楃€垫煡鏁?
    virtual void setPidDataCallback(PidDataCallback callback) = 0;

    // 閻犱礁澧介悿鍡涙儔閸曨偄娅欓悹褔顥撻崑锝夋晬閸繂娅欓柡鍕枍缂嶅懐绱旈鍡欑闁挎稑鏈ù娑欑閿濆甯涢悹浣靛€楀▓鎴︽偨婵犳碍妗ㄥ☉鎿冨幖缁?
    // x/y 濞戞挸鎼崕姘辨閻樺弶缍忛柡宥呮祫缁辨繄鎷嬮崣銉ㄧ-1閻炴稏鍔庨妵姘媴鐠恒劍鏆忛柣銏狀煼濞肩増绋夐鐐靛
    virtual void setAimOrigin(float x, float y) = 0;
};

#endif
