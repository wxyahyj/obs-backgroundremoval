#include "FOVRenderer.h"
#include <graphics/graphics.h>
#include <graphics/matrix4.h>
#include <cmath>
#include <algorithm>

FOVRenderer::FOVRenderer(gs_effect_t* solidEffect) 
    : solidEffect_(solidEffect) {
}

void FOVRenderer::render(uint32_t frameWidth, uint32_t frameHeight) {
    if (!solidEffect_) {
        return;
    }
    
    renderFOV(frameWidth, frameHeight);
    renderFOV2(frameWidth, frameHeight);
    renderRegion(frameWidth, frameHeight);
}

void FOVRenderer::updateConfig(const Config& config) {
    config_ = config;
}

void FOVRenderer::renderFOV(uint32_t frameWidth, uint32_t frameHeight) {
    if (!config_.showFOV) {
        return;
    }

    gs_technique_t *tech = gs_effect_get_technique(solidEffect_, "Solid");
    gs_eparam_t *colorParam = gs_effect_get_param_by_name(solidEffect_, "color");

    float centerX = frameWidth / 2.0f;
    float centerY = frameHeight / 2.0f;
    float radius = config_.useDynamicFOV ? config_.currentFovRadius : static_cast<float>(config_.fovRadius);
    float crossLineLength = static_cast<float>(config_.fovCrossLineScale);

    struct vec4 color;
    float r = ((config_.fovColor >> 16) & 0xFF) / 255.0f;
    float g = ((config_.fovColor >> 8) & 0xFF) / 255.0f;
    float b = (config_.fovColor & 0xFF) / 255.0f;
    float a = ((config_.fovColor >> 24) & 0xFF) / 255.0f;
    vec4_set(&color, r, g, b, a);

    gs_technique_begin(tech);
    gs_technique_begin_pass(tech, 0);
    gs_effect_set_vec4(colorParam, &color);

    if (config_.showFOVCross) {
        gs_render_start(true);
        gs_vertex2f(centerX - crossLineLength, centerY);
        gs_vertex2f(centerX + crossLineLength, centerY);
        gs_vertex2f(centerX, centerY - crossLineLength);
        gs_vertex2f(centerX, centerY + crossLineLength);
        gs_render_stop(GS_LINES);
    }

    if (config_.showFOVCircle) {
        const int circleSegments = 64;
        gs_render_start(true);
        for (int i = 0; i <= circleSegments; ++i) {
            float angle = 2.0f * 3.1415926f * static_cast<float>(i) / static_cast<float>(circleSegments);
            float x = centerX + radius * cosf(angle);
            float y = centerY + radius * sinf(angle);
            gs_vertex2f(x, y);
        }
        gs_render_stop(GS_LINESTRIP);
    }

    gs_technique_end_pass(tech);
    gs_technique_end(tech);
}

void FOVRenderer::renderFOV2(uint32_t frameWidth, uint32_t frameHeight) {
    if (!config_.showFOV2) {
        return;
    }

    gs_technique_t *tech = gs_effect_get_technique(solidEffect_, "Solid");
    gs_eparam_t *colorParam = gs_effect_get_param_by_name(solidEffect_, "color");

    float centerX = frameWidth / 2.0f;
    float centerY = frameHeight / 2.0f;
    float radius = static_cast<float>(config_.fovRadius2);

    struct vec4 color;
    float r = ((config_.fovColor2 >> 16) & 0xFF) / 255.0f;
    float g = ((config_.fovColor2 >> 8) & 0xFF) / 255.0f;
    float b = (config_.fovColor2 & 0xFF) / 255.0f;
    float a = ((config_.fovColor2 >> 24) & 0xFF) / 255.0f;
    vec4_set(&color, r, g, b, a);

    gs_technique_begin(tech);
    gs_technique_begin_pass(tech, 0);
    gs_effect_set_vec4(colorParam, &color);

    const int circleSegments = 64;
    gs_render_start(true);
    for (int i = 0; i <= circleSegments; ++i) {
        float angle = 2.0f * 3.1415926f * static_cast<float>(i) / static_cast<float>(circleSegments);
        float x = centerX + radius * cosf(angle);
        float y = centerY + radius * sinf(angle);
        gs_vertex2f(x, y);
    }
    gs_render_stop(GS_LINESTRIP);

    gs_technique_end_pass(tech);
    gs_technique_end(tech);
}

void FOVRenderer::renderRegion(uint32_t frameWidth, uint32_t frameHeight) {
    if (!config_.useRegion) {
        return;
    }

    gs_technique_t *tech = gs_effect_get_technique(solidEffect_, "Solid");
    gs_eparam_t *colorParam = gs_effect_get_param_by_name(solidEffect_, "color");

    float x = static_cast<float>(config_.regionX);
    float y = static_cast<float>(config_.regionY);
    float w = static_cast<float>(config_.regionWidth);
    float h = static_cast<float>(config_.regionHeight);

    struct vec4 color;
    vec4_set(&color, 1.0f, 1.0f, 0.0f, 1.0f);

    gs_technique_begin(tech);
    gs_technique_begin_pass(tech, 0);
    gs_effect_set_vec4(colorParam, &color);

    float dashLength = 10.0f;
    float gapLength = 5.0f;

    gs_render_start(true);
    for (float px = x; px < x + w; px += dashLength + gapLength) {
        float endX = std::min(px + dashLength, x + w);
        gs_vertex2f(px, y);
        gs_vertex2f(endX, y);
    }
    gs_render_stop(GS_LINES);

    gs_render_start(true);
    for (float px = x; px < x + w; px += dashLength + gapLength) {
        float endX = std::min(px + dashLength, x + w);
        gs_vertex2f(px, y + h);
        gs_vertex2f(endX, y + h);
    }
    gs_render_stop(GS_LINES);

    gs_render_start(true);
    for (float py = y; py < y + h; py += dashLength + gapLength) {
        float endY = std::min(py + dashLength, y + h);
        gs_vertex2f(x, py);
        gs_vertex2f(x, endY);
    }
    gs_render_stop(GS_LINES);

    gs_render_start(true);
    for (float py = y; py < y + h; py += dashLength + gapLength) {
        float endY = std::min(py + dashLength, y + h);
        gs_vertex2f(x + w, py);
        gs_vertex2f(x + w, endY);
    }
    gs_render_stop(GS_LINES);

    gs_technique_end_pass(tech);
    gs_technique_end(tech);
}
