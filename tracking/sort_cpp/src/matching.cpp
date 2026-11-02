#include "matching.hpp"

#include <algorithm>
#include <tuple>

MatchResult greedy_match(const std::vector<Box>& detections,
                         const std::vector<Box>& tracks,
                         double iou_threshold) {
    const int n_det = static_cast<int>(detections.size());
    const int n_trk = static_cast<int>(tracks.size());

    // 1) All candidate pairs with enough overlap
    std::vector<std::tuple<double, int, int>> pairs;  // (iou, det, trk)
    for (int d = 0; d < n_det; ++d) {
        for (int t = 0; t < n_trk; ++t) {
            const double v = iou(detections[d], tracks[t]);
            if (v >= iou_threshold) {
                pairs.emplace_back(v, d, t);
            }
        }
    }

    // 2) Best IoU first
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    // 3) Take a pair only if both sides are still free
    std::vector<bool> det_used(n_det, false);
    std::vector<bool> trk_used(n_trk, false);
    MatchResult result;

    for (const auto& [v, d, t] : pairs) {
        if (det_used[d] || trk_used[t]) {
            continue;
        }
        det_used[d] = true;
        trk_used[t] = true;
        result.matches.emplace_back(d, t);
    }

    for (int d = 0; d < n_det; ++d) {
        if (!det_used[d]) result.unmatched_detections.push_back(d);
    }
    for (int t = 0; t < n_trk; ++t) {
        if (!trk_used[t]) result.unmatched_tracks.push_back(t);
    }
    return result;
}
