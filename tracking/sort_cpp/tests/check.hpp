#pragma once

#include <cmath>
#include <cstdio>

// Tiny test helper: prints PASS/FAIL for each check and counts failures.
inline int g_failures = 0;

inline void check(bool condition, const char* name) {
    std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) g_failures++;
}

inline bool near(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

inline int summary() {
    if (g_failures == 0) {
        std::printf("\nAll checks passed.\n");
        return 0;
    }
    std::printf("\n%d check(s) failed.\n", g_failures);
    return 1;
}
