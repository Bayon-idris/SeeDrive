#include <cstdio>

#include "check.hpp"
#include "kalman_box_tracker.hpp"

double center_x(const Box& b) { return (b.x1 + b.x2) / 2.0; }

int main() {
    // A 40x30 box that moves 5 px to the right at every frame
    const double speed = 5.0;
    auto box_at = [&](int frame) {
        const double x = 100.0 + speed * frame;
        return Box{x, 200.0, x + 40.0, 230.0};
    };

    KalmanBoxTracker trk(box_at(0), 1);

    // 1) Feed 10 detections: the filter learns the velocity
    for (int f = 1; f <= 10; ++f) {
        trk.predict();
        trk.update(box_at(f));
    }
    check(trk.hits == 10 && trk.hit_streak == 10, "10 updates counted");

    // 2) No more detections: predictions must keep moving ~5 px per frame
    double previous = center_x(trk.state_box());
    bool steady = true;
    for (int f = 11; f <= 15; ++f) {
        const double now = center_x(trk.predict());
        std::printf("  frame %d: predicted center x = %.2f (true = %.2f)\n",
                    f, now, center_x(box_at(f)));
        if (!near(now - previous, speed, 1.0)) steady = false;
        previous = now;
    }
    check(steady, "keeps moving ~5 px/frame without detections");
    check(near(center_x(trk.state_box()), center_x(box_at(15)), 3.0),
          "prediction close to the true position after 5 frames");

    // 3) Counters after missing frames
    check(trk.time_since_update == 5, "time_since_update = 5");
    check(trk.hit_streak == 0, "hit_streak reset after a missed frame");

    // 4) Box size is preserved
    const Box b = trk.state_box();
    check(near(b.x2 - b.x1, 40.0, 1.0) && near(b.y2 - b.y1, 30.0, 1.0), "box size ~40x30");

    return summary();
}
