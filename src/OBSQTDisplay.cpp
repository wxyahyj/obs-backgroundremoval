#ifdef ENABLE_QT

#include "OBSQTDisplay.hpp"
#include <QWindow>
#include <QResizeEvent>
#include <obs-module.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

OBSQTDisplay::OBSQTDisplay(QWidget *parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_PaintOnScreen);
    setAttribute(Qt::WA_StaticContents);
    setAttribute(Qt::WA_NoSystemBackground);
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_DontCreateNativeAncestors);
    setAttribute(Qt::WA_NativeWindow);

    setMinimumSize(320, 180);
}

OBSQTDisplay::~OBSQTDisplay()
{
    destroying = true;
    DestroyDisplay();
    if (source) {
        obs_source_dec_showing(source);
        obs_source_release(source);
        source = nullptr;
    }
}

void OBSQTDisplay::SetSource(obs_source_t *newSource)
{
    if (source) {
        obs_source_dec_showing(source);
        obs_source_release(source);
    }

    source = newSource;

    if (source) {
        obs_source_addref(source);
        obs_source_inc_showing(source);
    }
}

void OBSQTDisplay::SetBackgroundColor(uint32_t color)
{
    backgroundColor = color;
    if (display) {
        obs_display_set_background_color(display, backgroundColor);
    }
}

void OBSQTDisplay::CreateDisplay()
{
    if (display || destroying)
        return;

    if (!windowHandle() || !windowHandle()->isExposed())
        return;

    QSize size = sizeHint();
    gs_init_data info = {};
    info.cx = size.width();
    info.cy = size.height();
    info.format = GS_BGRA;
    info.zsformat = GS_ZS_NONE;

#if defined(_WIN32)
    info.window.hwnd = (HWND)winId();
#elif defined(__APPLE__)
    info.window.view = (void*)winId();
#else
    info.window.id = winId();
    info.window.display = nullptr;
#endif

    display = obs_display_create(&info, backgroundColor);
    if (display) {
        obs_display_add_draw_callback(display, DrawCallback, this);
        emit DisplayCreated(this);
    }
}

void OBSQTDisplay::DestroyDisplay()
{
    if (display) {
        obs_display_remove_draw_callback(display, DrawCallback, this);
        obs_display_destroy(display);
        display = nullptr;
        emit DisplayDestroyed();
    }
}

void OBSQTDisplay::DrawCallback(void *data, uint32_t cx, uint32_t cy)
{
    OBSQTDisplay *displayWidget = static_cast<OBSQTDisplay *>(data);
    obs_source_t *source = displayWidget->source;

    if (source) {
        gs_blend_state_push();
        gs_reset_blend_state();

        obs_source_video_render(source);

        gs_blend_state_pop();
    }
}

void OBSQTDisplay::paintEvent(QPaintEvent *event)
{
    CreateDisplay();
    QWidget::paintEvent(event);
}

void OBSQTDisplay::moveEvent(QMoveEvent *event)
{
    QWidget::moveEvent(event);
    OnMove();
}

void OBSQTDisplay::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    OnSize();
}

void OBSQTDisplay::OnMove()
{
    if (display) {
        obs_display_resize(display, width(), height());
    }
}

void OBSQTDisplay::OnSize()
{
    if (display) {
        obs_display_resize(display, width(), height());
    }
}

#ifdef _WIN32
bool OBSQTDisplay::nativeEvent(const QByteArray &eventType, void *message, qintptr *result)
{
    MSG *msg = static_cast<MSG *>(message);

    switch (msg->message) {
    case WM_DISPLAYCHANGE:
        if (display)
            obs_display_resize(display, width(), height());
        break;
    }

    return QWidget::nativeEvent(eventType, message, result);
}
#else
bool OBSQTDisplay::nativeEvent(const QByteArray &eventType, void *message, qintptr *result)
{
    return QWidget::nativeEvent(eventType, message, result);
}
#endif

#endif
