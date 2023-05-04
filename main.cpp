#include "Modes.h"
#include <iostream>

using std::cout, std::endl;


int main(int argc, char** argv) {
    // check if first argument is train, test or play
    if (argc < 2) {
        std::cout << "Please specify train, test or play" << std::endl;
        return 0;
    }

    std::string mode = argv[1];
    if (mode == "train") {
        if (argc < 3) {
            std::cout << "Please specify run path" << std::endl;
            return 0;
        }

        train(argv[2]);
    } else if (mode == "test") {
        if (argc < 3) {
            std::cout << "Please specify load path" << std::endl;
            return 0;
        }

        test(argv[2]);
    } else if (mode == "play") {
        play();
    } else {
        std::cout << "Please specify train, test or play" << std::endl;
    }


    return 0;
}