#include <cstdio>
#include <set>
#include <vector>

#include "check.hpp"
#include "sort.hpp"

// Returns the id of the output box closest to x (or -1)
int id_near(const std::vector<TrackOutput>& out, double x) {
    for (const auto& t : out) {
        if (t.box.x1 - 20 < x && x < t.box.x2 + 20) return t.id;
    }
    return -1;
}

int main() {
    Sort tracker(/*max_age=*/5, /*min_hits=*/3, /*iou=*/0.3, /*det_threshold=*/0.5);

    // Car A moves right 3 px/frame, car B is parked far away.
    // A low-confidence detection (0.2) appears too: it must be ignored.
    bool a_stable = true, b_stable = true;
    int first_a = -1, first_b = -1;

    for (int f = 1; f <= 10; ++f) {
        const double ax = 100.0 + 3.0 * f;
        std::vector<Detection> dets = {
            {{ax, 100, ax + 50, 140}, 0.9},
            {{600, 300, 660, 350}, 0.8},
            {{300, 400, 330, 420}, 0.2},
        };
        const auto out = tracker.update(dets);

        const int ia = id_near(out, ax + 25), ib = id_near(out, 630);
        if (f == 1) { first_a = ia; first_b = ib; }
        if (ia != first_a) a_stable = false;
        if (ib != first_b) b_stable = false;
    }
    std::printf("  ids: A=%d, B=%d\n", first_a, first_b);
    check(first_a == 1 && first_b == 2, "A gets id 1, B gets id 2");
    check(a_stable, "A keeps the same id while moving");
    check(b_stable, "B keeps the same id");
    check(tracker.num_tracks() == 2, "low-confidence detection ignored (2 tracks)");

    // Car A disappears. After more than max_age (5) frames, it must be deleted.
    for (int f = 11; f <= 17; ++f) {
        tracker.update({{{600, 300, 660, 350}, 0.8}});
    }
    check(tracker.num_tracks() == 1, "A deleted after max_age frames without detection");

    // A new car C appears: it must get id 3, shown after min_hits (3) frames.
    int id_c = -1;
    for (int f = 18; f <= 22; ++f) {
        const auto out = tracker.update({{{600, 300, 660, 350}, 0.8},
                                         {{200, 50, 240, 80}, 0.9}});
        const int ic = id_near(out, 220);
        std::printf("  frame %d: C %s\n", f, ic == -1 ? "not shown yet" : "shown");
        if (ic != -1) id_c = ic;
    }
    check(id_c == 3, "new car C gets id 3");

    // Empty frame: must not crash
    tracker.update({});
    check(true, "empty frame handled");

    return summary();
}
