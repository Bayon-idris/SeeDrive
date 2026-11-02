#pragma once

#include <vector>

#include "iou.hpp"
#include "kalman_box_tracker.hpp"

struct Detection {
    Box box;
    double score;
};

struct TrackOutput {
    int id;
    Box box;
};

class Sort {
public:
    // max_age       : frames a track is kept without detection (30 = 1 s at 25 fps)
    // min_hits      : consecutive detections needed before a track is shown
    // iou_threshold : minimum overlap to associate a detection with a track
    // det_threshold : detections below this confidence are ignored
    explicit Sort(int max_age = 30, int min_hits = 3,
                  double iou_threshold = 0.3, double det_threshold = 0.5);

    // Call once per frame (even with no detection). Returns the confirmed tracks.
    std::vector<TrackOutput> update(const std::vector<Detection>& detections);

    int num_tracks() const { return static_cast<int>(trackers_.size()); }

private:
    int max_age_;
    int min_hits_;
    double iou_threshold_;
    double det_threshold_;

    std::vector<KalmanBoxTracker> trackers_;
    int frame_count_ = 0;
    int next_id_ = 1;
};
