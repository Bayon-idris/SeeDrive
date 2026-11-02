#pragma once

// A bounding box: top-left corner (x1, y1) and bottom-right corner (x2, y2), in pixels.
struct Box {
    double x1;
    double y1;
    double x2;
    double y2;
};

// Intersection over Union: overlap between two boxes, from 0 (no overlap) to 1 (identical).
double iou(const Box& a, const Box& b);
