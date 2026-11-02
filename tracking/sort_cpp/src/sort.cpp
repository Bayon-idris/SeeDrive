#include "sort.hpp"

#include <cmath>

#include "matching.hpp"

namespace {
bool is_valid(const Box& b) {
    return std::isfinite(b.x1) && std::isfinite(b.y1) &&
           std::isfinite(b.x2) && std::isfinite(b.y2);
}
}  // namespace

Sort::Sort(int max_age, int min_hits, double iou_threshold, double det_threshold)
    : max_age_(max_age),
      min_hits_(min_hits),
      iou_threshold_(iou_threshold),
      det_threshold_(det_threshold) {}

std::vector<TrackOutput> Sort::update(const std::vector<Detection>& detections) {
    frame_count_ += 1;

    // 1) Keep only confident detections
    std::vector<Box> dets;
    for (const auto& d : detections) {
        if (d.score >= det_threshold_) {
            dets.push_back(d.box);
        }
    }

    // 2) PREDICT where every track should be now (drop broken ones)
    std::vector<Box> predicted;
    std::vector<KalmanBoxTracker> alive;
    for (auto& trk : trackers_) {
        const Box p = trk.predict();
        if (is_valid(p)) {
            predicted.push_back(p);
            alive.push_back(trk);
        }
    }
    trackers_ = std::move(alive);

    // 3 + 4) IoU between detections and predictions, then MATCHING
    const MatchResult m = greedy_match(dets, predicted, iou_threshold_);

    // 5) Matched tracks are corrected with their detection
    for (const auto& [d, t] : m.matches) {
        trackers_[t].update(dets[d]);
    }

    // 6) Unmatched detections become new tracks
    for (int d : m.unmatched_detections) {
        trackers_.emplace_back(dets[d], next_id_++);
    }

    // 7) OUTPUT: tracks seen this frame and confirmed (or at the very start of the video)
    std::vector<TrackOutput> output;
    for (const auto& trk : trackers_) {
        const bool seen_now = trk.time_since_update == 0;
        const bool confirmed = trk.hit_streak >= min_hits_ || frame_count_ <= min_hits_;
        if (seen_now && confirmed) {
            output.push_back({trk.id, trk.state_box()});
        }
    }

    // 8) DELETE tracks lost for too long
    std::vector<KalmanBoxTracker> kept;
    for (const auto& trk : trackers_) {
        if (trk.time_since_update <= max_age_) {
            kept.push_back(trk);
        }
    }
    trackers_ = std::move(kept);

    return output;
}
