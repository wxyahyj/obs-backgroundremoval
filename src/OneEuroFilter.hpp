#ifndef ONE_EURO_FILTER_HPP
#define ONE_EURO_FILTER_HPP

#ifdef _WIN32

#include <cmath>
#include <algorithm>

class OneEuroFilter {
public:
    OneEuroFilter();

    float filter(float value, float dt);
    void reset();

    void setMinCutoff(float v) { minCutoff = v; }
    void setBeta(float v) { beta = v; }
    void setDCutoff(float v) { dCutoff = v; }

    float getX() const { return xPrev; }
    float getDx() const { return dxPrev; }

private:
    float computeAlpha(float cutoff, float dt);

    float minCutoff;
    float beta;
    float dCutoff;
    float xPrev;
    float dxPrev;
    bool initialized;
};

#endif // _WIN32
#endif // ONE_EURO_FILTER_HPP
