#include "kalman_box_tracker.hpp"

#include <cmath>

KalmanBoxTracker::KalmanBoxTracker(const Box& box, int id) : id(id) {
    // F: identity + "position moves by its velocity"
    F_ = Mat7::Identity();
    F_(0, 4) = 1.0;  // cx += vcx
    F_(1, 5) = 1.0;  // cy += vcy
    F_(2, 6) = 1.0;  // s  += vs

    // H: keeps the first 4 values of the state
    H_ = Mat47::Zero();
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;
    H_(3, 3) = 1.0;

    // Initial values from the original SORT paper/code
    R_ = Vec4(1, 1, 10, 10).asDiagonal();
    P_ = (Vec7() << 10, 10, 10, 10, 10000, 10000, 10000).finished().asDiagonal();
    Q_ = (Vec7() << 1, 1, 1, 1, 0.01, 0.01, 0.0001).finished().asDiagonal();

    // Initial state: the detection, with zero velocity
    x_ = Vec7::Zero();
    x_.head<4>() = box_to_z(box);
}

KalmanBoxTracker::Vec4 KalmanBoxTracker::box_to_z(const Box& box) {
    const double w = box.x2 - box.x1;
    const double h = box.y2 - box.y1;
    const double cx = box.x1 + w / 2.0;
    const double cy = box.y1 + h / 2.0;
    const double s = w * h;
    const double r = (h > 0.0) ? w / h : 0.0;
    return Vec4(cx, cy, s, r);
}

Box KalmanBoxTracker::state_box() const {
    const double s = x_(2);
    const double r = x_(3);
    const double w = (s * r > 0.0) ? std::sqrt(s * r) : 0.0;
    const double h = (w > 0.0) ? s / w : 0.0;
    const double cx = x_(0);
    const double cy = x_(1);
    return Box{cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0};
}

Box KalmanBoxTracker::predict() {
    // The area can't become negative
    if (x_(2) + x_(6) <= 0.0) {
        x_(6) = 0.0;
    }

    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;

    age += 1;
    if (time_since_update > 0) {
        hit_streak = 0;  // missed at least one frame: the streak is broken
    }
    time_since_update += 1;

    return state_box();
}

void KalmanBoxTracker::update(const Box& box) {
    time_since_update = 0;
    hits += 1;
    hit_streak += 1;

    const Vec4 z = box_to_z(box);
    const Vec4 y = z - H_ * x_;                           // gap between measure and prediction
    const Mat4 S = H_ * P_ * H_.transpose() + R_;
    const Eigen::Matrix<double, 7, 4> K =
        P_ * H_.transpose() * S.inverse();                // Kalman gain

    x_ = x_ + K * y;
    P_ = (Mat7::Identity() - K * H_) * P_;
}
