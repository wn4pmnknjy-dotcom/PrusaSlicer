#pragma once

#include <optional>
#include "Slic3r/Biz/Arrange/Bed.hpp"
#include "Slic3r/Biz/Algorithms/Scaling.hpp"

namespace Slic3r::Biz::Arrange {

enum class Strategy
{
    Overfit,
    Gravity,
    // Raster packing where every translation is a whole number of printer pixels (MSLA).
    PixelGrid
};

enum class GeometryHandling
{
    Convex,
    Arbitrary,
};

struct Settings
{
    Strategy strategy{Strategy::Overfit};
    double scaled_offset{0.0};
    double unscaled_bed_offset{0.0};
    bool allow_rotations{false};
    GeometryHandling fixed_geometry{GeometryHandling::Convex};
    GeometryHandling movable_geometry{GeometryHandling::Convex};
    std::optional<PivotPoint> bed_pivot_point;
    std::optional<Domain::BedSegments> bed_segments;
    int scaled_simplification_tolerance{Biz::Algorithms::Scaling::scaled(.2)};
    std::optional<Domain::Vec2d> auxiliary_travel_anchor;

    // Strategy::PixelGrid only.
    // Size of one printer pixel in mm, item positions are snapped to multiples of it.
    double unscaled_pixel_width{0.05};
    double unscaled_pixel_height{0.05};
    // Coarse search step in mm, the final position is refined pixel by pixel.
    double unscaled_pixel_search_step{0.5};
    // Corner of the pixel grid in bed coordinates (mm), (0, 0) if not set.
    std::optional<Domain::Vec2d> pixel_grid_origin;
    // Rotate items to their minimum area bounding box before packing.
    bool pixel_grid_min_footprint{true};
};
} // namespace Slic3r::Biz::Arrange
