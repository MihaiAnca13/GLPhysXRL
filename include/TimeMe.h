//
// Created by mihai on 30/04/23.
//

#ifndef C_ML_TIMEME_H
#define C_ML_TIMEME_H

#include <chrono>
#include <iostream>
#include <string>
#include <utility>


class TimeMe {

public:
    explicit TimeMe(std::string name) : name(std::move(name)) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~TimeMe() {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << name + " took " << elapsed.count() << " milliseconds\n";
    }

private:
    string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

#endif //C_ML_TIMEME_H
