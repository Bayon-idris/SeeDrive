#pragma once

#include <utility>
#include <vector>

#include "iou.hpp"

struct MatchResult {
    std::vector<std::pair<int, int>> matches;  // (detection index, track index)
    std::vector<int> unmatched_detections;
    std::vector<int> unmatched_tracks;
};

// Greedy matching: sort all (detection, track) pairs by IoU, best first,
// and keep a pair if neither side is already taken and IoU >= iou_threshold.
// Simple version; can be replaced later by the Hungarian algorithm.
MatchResult greedy_match(const std::vector<Box>& detections,
                         const std::vector<Box>& tracks,
                         double iou_threshold);
