//
// Created by mihai on 15/03/23.
//

#ifndef C_ML_UTILSANGLES_H
#define C_ML_UTILSANGLES_H

#include <limits>
#include <cmath>


namespace UtilsAngles {
    static const double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348;
    static const double TWO_PI = 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696;

    template<typename T>
    T Mod(T x, T y);

    // wrap [rad] angle to [-PI..PI)
    double WrapPosNegPI(double fAng);

    // wrap [rad] angle to [0..TWO_PI)
    double WrapTwoPI(double fAng);

    // wrap [deg] angle to [-180..180)
    double WrapPosNeg180(double fAng);

    // wrap [deg] angle to [0..360)
    double Wrap360(double fAng);
}

#endif //C_ML_UTILSANGLES_H
