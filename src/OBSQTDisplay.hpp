#pragma once

#ifdef ENABLE_QT

#include <QWidget>
#include <obs.h>

class OBSQTDisplay : public QWidget
{
    Q_OBJECT

public:
    explicit OBSQTDisplay(QWidget *parent = nullptr);
    ~OBSQTDisplay();

    obs_display_t *GetDisplay() const { return display; }
    void SetSource(obs_source_t *source);
    void SetBackgroundColor(uint32_t color);

signals:
    void DisplayCreated(OBSQTDisplay *display);
    void DisplayDestroyed();

protected:
    virtual void paintEvent(QPaintEvent *event) override;
    virtual void moveEvent(QMoveEvent *event) override;
    virtual void resizeEvent(QResizeEvent *event) override;
    virtual bool nativeEvent(const QByteArray &eventType, void *message, qintptr *result) override;

private:
    void CreateDisplay();
    void DestroyDisplay();
    void OnMove();
    void OnSize();

    obs_display_t *display = nullptr;
    obs_source_t *source = nullptr;
    uint32_t backgroundColor = 0x4C4C4C;
    bool destroying = false;

    static void DrawCallback(void *data, uint32_t cx, uint32_t cy);
};

#endif
