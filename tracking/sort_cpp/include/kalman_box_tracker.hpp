#pragma once

#include <Eigen/Dense>

#include "iou.hpp"

// Follows ONE object with a Kalman filter.
//
// State (7 values):        x = [cx, cy, s, r, vcx, vcy, vs]
//     cx, cy : box center
//     s      : box area (w * h)
//     r      : aspect ratio (w / h), assumed constant
//     vcx, vcy, vs : velocities of cx, cy and s
//
// Measurement (4 values):  z = [cx, cy, s, r]   (what the detector gives us)
class KalmanBoxTracker {
public:
    KalmanBoxTracker(const Box& box, int id);

    // Moves the state one frame forward and returns the predicted box.
    Box predict();

    // Corrects the state with a detection associated to this track.
    void update(const Box& box);

    // Current estimated box.
    Box state_box() const;

    int id;
    int hits = 0;               // total number of associated detections
    int hit_streak = 0;         // consecutive frames with an associated detection
    int age = 0;                // frames since creation
    int time_since_update = 0;  // frames since the last associated detection

private:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec4 = Eigen::Matrix<double, 4, 1>;
    using Mat7 = Eigen::Matrix<double, 7, 7>;
    using Mat4 = Eigen::Matrix<double, 4, 4>;
    using Mat47 = Eigen::Matrix<double, 4, 7>;

    Vec7 x_;   // state
    Mat7 P_;   // state uncertainty
    Mat7 F_;   // motion model
    Mat7 Q_;   // motion noise
    Mat47 H_;  // state -> measurement
    Mat4 R_;   // measurement noise

    static Vec4 box_to_z(const Box& box);
};
