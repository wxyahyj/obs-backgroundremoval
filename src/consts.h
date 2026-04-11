#ifndef CONSTS_H
#define CONSTS_H

const char *const MODEL_SINET = "models/SINet_Softmax_simple.with_runtime_opt.ort";
const char *const MODEL_MEDIAPIPE = "models/mediapipe.with_runtime_opt.ort";
const char *const MODEL_SELFIE = "models/selfie_segmentation.with_runtime_opt.ort";
const char *const MODEL_SELFIE_MULTICLASS = "models/selfie_multiclass_256x256.with_runtime_opt.ort";
const char *const MODEL_RVM = "models/rvm_mobilenetv3_fp32.with_runtime_opt.ort";
const char *const MODEL_PPHUMANSEG = "models/pphumanseg_fp32.with_runtime_opt.ort";
const char *const MODEL_ENHANCE_TBEFN = "models/tbefn_fp32.with_runtime_opt.ort";
const char *const MODEL_ENHANCE_URETINEX = "models/uretinex_net_180x320.with_runtime_opt.ort";
const char *const MODEL_ENHANCE_SGLLIE = "models/semantic_guided_llie_180x324.with_runtime_opt.ort";
const char *const MODEL_ENHANCE_ZERODCE = "models/zero_dce_180x320.with_runtime_opt.ort";
const char *const MODEL_DEPTH_TCMONODEPTH = "models/tcmonodepth_tcsmallnet_192x320.with_runtime_opt.ort";
const char *const MODEL_RMBG = "models/bria_rmbg_1_4_qint8.with_runtime_opt.ort";

const char *const USEGPU_CPU = "cpu";
const char *const USEGPU_CUDA = "cuda";
const char *const USEGPU_ROCM = "rocm";
const char *const USEGPU_MIGRAPHX = "migraphx";
const char *const USEGPU_TENSORRT = "tensorrt";
const char *const USEGPU_COREML = "coreml";
const char *const USEGPU_DML = "dml";

const char *const EFFECT_PATH = "effects/mask_alpha_filter.effect";
const char *const KAWASE_BLUR_EFFECT_PATH = "effects/kawase_blur.effect";
const char *const BLEND_EFFECT_PATH = "effects/blend_images.effect";

const char *const PLUGIN_INFO_TEMPLATE =
	"<a href=\"https://github.com/royshil/obs-backgroundremoval/\">Background Removal</a> (%1) by "
	"<a href=\"https://github.com/royshil\">Roy Shilkrot</a> ❤️ "
	"<a href=\"https://github.com/sponsors/royshil\">Support & Follow</a>";
const char *const PLUGIN_INFO_TEMPLATE_UPDATE_AVAILABLE =
	"<center><a href=\"https://github.com/royshil/obs-backgroundremoval/releases\">🚀 Update available! (%1)</a></center>";

// ========================================
// 真正的编译期常量（不可变的）
// ========================================

// 最大配置数量（固定 5 个配置槽）
constexpr int MAX_CONFIGS = 5;

// 匈牙利算法最大距离阈值（固定算法参数）
constexpr float HUNGARIAN_MAX_DISTANCE = 100.0f;

// 目标丢失最大帧数（固定逻辑）
constexpr int MAX_LOST_FRAMES = 30;

// 最小检测置信度（硬阈值，低于此值直接过滤）
constexpr float MIN_CONFIDENCE_THRESHOLD = 0.3f;

#endif /* CONSTS_H */
