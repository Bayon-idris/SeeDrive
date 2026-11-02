#include "check.hpp"
#include "iou.hpp"

int main() {
    const Box a{0, 0, 10, 10};

    check(near(iou(a, a), 1.0), "identical boxes -> 1");
    check(near(iou(a, Box{50, 50, 60, 60}), 0.0), "far apart boxes -> 0");
    check(near(iou(a, Box{10, 0, 20, 10}), 0.0), "boxes touching on one edge -> 0");

    // B overlaps half of A: intersection = 50, union = 100 + 100 - 50 = 150
    check(near(iou(a, Box{5, 0, 15, 10}), 50.0 / 150.0), "half overlap -> 1/3");

    // Small box fully inside: intersection = 25, union = 100
    check(near(iou(a, Box{0, 0, 5, 5}), 0.25), "small box inside -> 0.25");

    check(near(iou(a, Box{0, 0, 10, 10}), iou(Box{0, 0, 10, 10}, a)), "symmetric");

    return summary();
}
