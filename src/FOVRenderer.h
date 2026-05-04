#pragma once

#include <cstdint>
#include <graphics/graphics.h>

class FOVRenderer {
public:
    struct Config {
        bool showFOV = false;
        int fovRadius = 100;
        uint32_t fovColor = 0x00FF00FF;
        int fovCrossLineScale = 10;
        int fovCrossLineThickness = 2;
        int fovCircleThickness = 2;
        bool showFOVCircle = true;
        bool showFOVCross = true;
        
        bool showFOV2 = false;
        int fovRadius2 = 150;
        uint32_t fovColor2 = 0x0000FFFF;
        
        bool useDynamicFOV = false;
        float currentFovRadius = 100.0f;
        
        bool useRegion = false;
        int regionX = 0;
        int regionY = 0;
        int regionWidth = 640;
        int regionHeight = 480;
    };

    explicit FOVRenderer(gs_effect_t* solidEffect);
    ~FOVRenderer() = default;

    FOVRenderer(const FOVRenderer&) = delete;
    FOVRenderer& operator=(const FOVRenderer&) = delete;
    FOVRenderer(FOVRenderer&&) noexcept = default;
    FOVRenderer& operator=(FOVRenderer&&) noexcept = default;

    void render(uint32_t frameWidth, uint32_t frameHeight);
    void updateConfig(const Config& config);
    Config getConfig() const { return config_; }

    void setSolidEffect(gs_effect_t* solidEffect) { solidEffect_ = solidEffect; }

private:
    void renderFOV(uint32_t frameWidth, uint32_t frameHeight);
    void renderFOV2(uint32_t frameWidth, uint32_t frameHeight);
    void renderRegion(uint32_t frameWidth, uint32_t frameHeight);

    Config config_;
    gs_effect_t* solidEffect_;
};
