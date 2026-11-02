// Python bridge for the C++ SORT tracker (pybind11).
//
// In Python:
//     import numpy as np
//     import sort_tracker
//
//     tracker = sort_tracker.Sort(max_age=30, min_hits=3, iou_threshold=0.3, det_threshold=0.5)
//     dets = np.array([[x1, y1, x2, y2, score], ...])    # one row per YOLO box
//     tracks = tracker.update(dets)                      # one row per track: [x1, y1, x2, y2, id]

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <vector>

#include "sort.hpp"

namespace py = pybind11;

namespace {

// numpy array (N x 5) -> std::vector<Detection>
std::vector<Detection> to_detections(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& dets) {
    std::vector<Detection> out;

    if (dets.size() == 0) {
        return out;  // empty frame: no detection
    }
    if (dets.ndim() != 2 || dets.shape(1) < 5) {
        throw std::invalid_argument("detections must have shape (N, 5): x1, y1, x2, y2, score");
    }

    auto d = dets.unchecked<2>();  // fast read-only access, no bounds checks
    out.reserve(static_cast<size_t>(d.shape(0)));
    for (py::ssize_t i = 0; i < d.shape(0); ++i) {
        out.push_back({Box{d(i, 0), d(i, 1), d(i, 2), d(i, 3)}, d(i, 4)});
    }
    return out;
}

// std::vector<TrackOutput> -> numpy array (M x 5)
py::array_t<double> to_numpy(const std::vector<TrackOutput>& tracks) {
    py::array_t<double> result({static_cast<py::ssize_t>(tracks.size()), py::ssize_t{5}});
    auto r = result.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(tracks.size()); ++i) {
        const TrackOutput& t = tracks[static_cast<size_t>(i)];
        r(i, 0) = t.box.x1;
        r(i, 1) = t.box.y1;
        r(i, 2) = t.box.x2;
        r(i, 3) = t.box.y2;
        r(i, 4) = static_cast<double>(t.id);
    }
    return result;
}

}  // namespace

// "sort_tracker" = the name used in Python: import sort_tracker
PYBIND11_MODULE(sort_tracker, m) {
    m.doc() = "SORT multi-object tracker implemented in C++";

    py::class_<Sort>(m, "Sort")
        .def(py::init<int, int, double, double>(),
             py::arg("max_age") = 30,
             py::arg("min_hits") = 3,
             py::arg("iou_threshold") = 0.3,
             py::arg("det_threshold") = 0.5)

        .def("update",
             [](Sort& self, const py::array_t<double, py::array::c_style | py::array::forcecast>& dets) {
                 return to_numpy(self.update(to_detections(dets)));
             },
             py::arg("detections"),
             "Call once per frame. detections: (N, 5) [x1, y1, x2, y2, score]. "
             "Returns (M, 5) [x1, y1, x2, y2, id].")

        .def_property_readonly("num_tracks", &Sort::num_tracks,
                               "Number of tracks currently kept in memory");
}