#include "UtilsAngles.h"


// Floating-point modulo
// The result (the remainder) has same sign as the divisor.
// Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1   fmod(-3,4)= -3
template<typename T>
T UtilsAngles::Mod(T x, T y) {
    static_assert(!std::numeric_limits<T>::is_exact, "Mod: floating-point type expected");

    if (0. == y)
        return x;

    double m = x - y * floor(x / y);

    // handle boundary cases resulted from floating-point cut off:

    if (y > 0)              // modulo range: [0..y)
    {
        if (m >= y)           // Mod(-1e-16             , 360.    ): m= 360.
            return 0;

        if (m < 0) {
            if (y + m == y)
                return 0; // just in case...
            else
                return y + m; // Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
        }
    } else                    // modulo range: (y..0]
    {
        if (m <= y)           // Mod(1e-16              , -360.   ): m= -360.
            return 0;

        if (m > 0) {
            if (y + m == y)
                return 0; // just in case...
            else
                return y + m; // Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
        }
    }

    return m;
}

// wrap [rad] angle to [-PI..PI)
double UtilsAngles::WrapPosNegPI(double fAng) {
    return Mod(fAng + UtilsAngles::PI, UtilsAngles::TWO_PI) - UtilsAngles::PI;
}

// wrap [rad] angle to [0..TWO_PI)
double UtilsAngles::WrapTwoPI(double fAng) {
    return Mod(fAng, UtilsAngles::TWO_PI);
}

// wrap [deg] angle to [-180..180)
double UtilsAngles::WrapPosNeg180(double fAng) {
    return Mod(fAng + 180., 360.) - 180.;
}

// wrap [deg] angle to [0..360)
double UtilsAngles::Wrap360(double fAng) {
    return Mod(fAng, 360.);
}