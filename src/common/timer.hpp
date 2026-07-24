#pragma once
#include <chrono>

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;

    Timer() : t0(clock::now()) {}

    double elapsed_sec() const {
        return std::chrono::duration<double>(
            clock::now() - t0
        ).count();
    }
};
