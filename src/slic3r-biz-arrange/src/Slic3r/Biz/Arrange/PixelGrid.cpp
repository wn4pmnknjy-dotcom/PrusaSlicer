#include "Slic3r/Biz/Arrange/PixelGrid.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <numeric>
#include "Slic3r/Assert.hpp"
#include "Slic3r/Biz/Algorithms/Scaling.hpp"
#include "Slic3r/Biz/Arrange/MinAreaBoundingBox.hpp"

namespace Slic3r::Biz::Arrange {

using Algorithms::Scaling::scaled;
using Domain::BoundingBox2crd;
using Domain::coord_t;
using Domain::Polygon;
using Domain::Polygons;
using Domain::Vec2crd;

namespace {
constexpr int bits_per_word{64};

coord_t floor_div(const coord_t a, const coord_t b)
{
    const coord_t q{a / b};
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

Vec2crd floor_div(const Vec2crd& a, const Vec2crd& b)
{
    return {floor_div(a.x(), b.x()), floor_div(a.y(), b.y())};
}

// The x range covered by a convex polygon within the horizontal slab [y_lo, y_hi].
std::optional<std::pair<double, double>> slab_x_range(const Polygon& poly, const double y_lo, const double y_hi)
{
    double x_min{std::numeric_limits<double>::max()};
    double x_max{std::numeric_limits<double>::lowest()};
    const std::size_t n{poly.points.size()};
    for (std::size_t i{}; i < n; ++i) {
        const Vec2crd& a{poly.points[i]};
        const Vec2crd& b{poly.points[(i + 1) % n]};
        const double ay{double(a.y())}, by{double(b.y())};
        const double lo{std::max(std::min(ay, by), y_lo)};
        const double hi{std::min(std::max(ay, by), y_hi)};
        if (lo > hi) {
            continue;
        }
        if (ay == by) {
            x_min = std::min({x_min, double(a.x()), double(b.x())});
            x_max = std::max({x_max, double(a.x()), double(b.x())});
            continue;
        }
        for (const double y : {lo, hi}) {
            const double t{(y - ay) / (by - ay)};
            const double x{a.x() + t * (b.x() - a.x())};
            x_min = std::min(x_min, x);
            x_max = std::max(x_max, x);
        }
    }
    if (x_min > x_max) {
        return std::nullopt;
    }
    return std::make_pair(x_min, x_max);
}

std::vector<double> candidate_rotations(const ArrangeItem& item, const Settings& settings)
{
    std::vector<double> result;
    if (settings.pixel_grid_min_footprint) {
        const double min_bb_rotation{
            MinAreaBoundigBox{item.movable_shape().convex_hull(), MinAreaBoundigBox::pcConvex}.angle_to_X()
        };
        for (int quarter{}; quarter < 4; ++quarter) {
            result.push_back(min_bb_rotation + quarter * std::numbers::pi / 2.);
        }
    } else {
        result.push_back(0.);
    }
    if (settings.allow_rotations) {
        const std::vector<double>& allowed{item.allowed_rotations()};
        result.insert(result.end(), allowed.begin(), allowed.end());
    }
    return result;
}

std::vector<int> search_positions(const int last, const int stride)
{
    std::vector<int> result;
    for (int i{}; i <= last; i += stride) {
        result.push_back(i);
    }
    if (!result.empty() && result.back() != last) {
        result.push_back(last);
    }
    return result;
}

// Finds the lowest, then leftmost free spot on a coarse stride and then slides
// the mask pixel by pixel down and left until it touches something.
std::optional<Vec2crd> find_lowest_left(const PixelMask& occupancy, const PixelMask& mask, const int stride)
{
    const int last_x{occupancy.width() - mask.width()};
    const int last_y{occupancy.height() - mask.height()};
    if (last_x < 0 || last_y < 0) {
        return std::nullopt;
    }
    const std::vector<int> xs{search_positions(last_x, stride)};
    for (const int y0 : search_positions(last_y, stride)) {
        for (const int x0 : xs) {
            if (!occupancy.fits(mask, x0, y0)) {
                continue;
            }
            int x{x0}, y{y0};
            for (bool moved{true}; moved;) {
                moved = false;
                while (y > 0 && occupancy.fits(mask, x, y - 1)) {
                    --y;
                    moved = true;
                }
                while (x > 0 && occupancy.fits(mask, x - 1, y)) {
                    --x;
                    moved = true;
                }
            }
            return Vec2crd{x, y};
        }
    }
    return std::nullopt;
}

BoundingBox2crd packed_extents(const std::vector<ArrangeItem>& items)
{
    BoundingBox2crd result{items.front().fixed_shape().bounding_box()};
    for (const ArrangeItem& item : items) {
        const BoundingBox2crd& bb{item.fixed_shape().bounding_box()};
        result.min = result.min.cwiseMin(bb.min);
        result.max = result.max.cwiseMax(bb.max);
    }
    return result;
}

// Moves the whole pile by whole pixels so that it sits at the requested pivot of the bed.
void align_pile_to_pixels(std::vector<ArrangeItem>& items, const BoundingBox2crd& usable, const PivotPoint pivot, const Vec2crd& pitch)
{
    if (items.empty()) {
        return;
    }
    const BoundingBox2crd pile{packed_extents(items)};
    const Vec2crd free_space{(usable.max - usable.min) - (pile.max - pile.min)};
    Vec2crd target{usable.min};
    switch (pivot) {
    case PivotPoint::Center: target += free_space / 2; break;
    case PivotPoint::BottomLeft: break;
    case PivotPoint::BottomRight: target.x() += free_space.x(); break;
    case PivotPoint::TopLeft: target.y() += free_space.y(); break;
    case PivotPoint::TopRight: target += free_space; break;
    }
    Vec2crd shift{target - pile.min};
    shift = floor_div(shift, pitch).cwiseProduct(pitch);
    for (ArrangeItem& item : items) {
        item.set_translation(item.get_translation() + shift);
    }
}
} // namespace

PixelMask::PixelMask(const Vec2crd& origin, const Vec2crd& pitch, const int width, const int height)
    : m_origin{origin}
    , m_pitch{pitch}
    , m_width{std::max(width, 0)}
    , m_height{std::max(height, 0)}
    , m_words{(m_width + bits_per_word - 1) / bits_per_word}
    , m_bits(std::size_t(m_words) * m_height, 0)
{
    ASSERT(pitch.x() > 0 && pitch.y() > 0);
}

PixelMask PixelMask::rasterize_convex(const Polygons& convex_polygons, const Vec2crd& pitch, const Vec2crd& grid_origin)
{
    Vec2crd lo{Vec2crd::Constant(std::numeric_limits<coord_t>::max())};
    Vec2crd hi{Vec2crd::Constant(std::numeric_limits<coord_t>::lowest())};
    for (const Polygon& poly : convex_polygons) {
        for (const Vec2crd& p : poly.points) {
            lo = lo.cwiseMin(p);
            hi = hi.cwiseMax(p);
        }
    }
    if (lo.x() > hi.x()) {
        return PixelMask{grid_origin, pitch, 0, 0};
    }

    const Vec2crd cell_lo{floor_div(lo - grid_origin, pitch)};
    const Vec2crd cell_hi{floor_div(hi - grid_origin, pitch)};
    PixelMask mask{
        grid_origin + cell_lo.cwiseProduct(pitch),
        pitch,
        int(cell_hi.x() - cell_lo.x() + 1),
        int(cell_hi.y() - cell_lo.y() + 1)
    };

    for (const Polygon& poly : convex_polygons) {
        if (poly.points.empty()) {
            continue;
        }
        coord_t poly_lo{poly.points.front().y()}, poly_hi{poly_lo};
        for (const Vec2crd& p : poly.points) {
            poly_lo = std::min(poly_lo, p.y());
            poly_hi = std::max(poly_hi, p.y());
        }
        const int row_lo{int(floor_div(poly_lo - mask.m_origin.y(), pitch.y()))};
        const int row_hi{int(floor_div(poly_hi - mask.m_origin.y(), pitch.y()))};
        for (int row{row_lo}; row <= row_hi; ++row) {
            const double y_lo{double(mask.m_origin.y() + row * pitch.y())};
            const auto range{slab_x_range(poly, y_lo, y_lo + double(pitch.y()))};
            if (!range) {
                continue;
            }
            const int x_from{int(std::floor((range->first - mask.m_origin.x()) / pitch.x()))};
            const int x_to{int(std::floor((range->second - mask.m_origin.x()) / pitch.x()))};
            mask.set_span(row, x_from, x_to);
        }
    }
    return mask;
}

bool PixelMask::get(const int x, const int y) const
{
    if (x < 0 || y < 0 || x >= m_width || y >= m_height) {
        return false;
    }
    return (word(y, x / bits_per_word) >> (x % bits_per_word)) & 1u;
}

void PixelMask::set(const int x, const int y)
{
    set_span(y, x, x);
}

void PixelMask::set_span(const int y, int x_from, int x_to)
{
    if (y < 0 || y >= m_height) {
        return;
    }
    x_from = std::max(x_from, 0);
    x_to   = std::min(x_to, m_width - 1);
    for (int x{x_from}; x <= x_to; ++x) {
        m_bits[std::size_t(y) * m_words + x / bits_per_word] |= std::uint64_t{1} << (x % bits_per_word);
    }
}

std::size_t PixelMask::count() const
{
    return std::accumulate(m_bits.begin(), m_bits.end(), std::size_t{0}, [](std::size_t s, std::uint64_t w) {
        return s + std::popcount(w);
    });
}

bool PixelMask::fits(const PixelMask& mask, const int x, const int y) const
{
    if (x < 0 || y < 0 || x + mask.m_width > m_width || y + mask.m_height > m_height) {
        return false;
    }
    const int word_offset{x / bits_per_word};
    const int shift{x % bits_per_word};
    for (int row{}; row < mask.m_height; ++row) {
        for (int w{}; w < mask.m_words; ++w) {
            const std::uint64_t bits{mask.word(row, w)};
            if (bits == 0) {
                continue;
            }
            const int target{word_offset + w};
            if ((word(y + row, target) & (bits << shift)) != 0) {
                return false;
            }
            if (shift != 0 && target + 1 < m_words
                && (word(y + row, target + 1) & (bits >> (bits_per_word - shift))) != 0) {
                return false;
            }
        }
    }
    return true;
}

void PixelMask::stamp(const PixelMask& mask, const int x, const int y)
{
    for (int row{}; row < mask.m_height; ++row) {
        for (int col{}; col < mask.m_width; ++col) {
            if (mask.get(col, row)) {
                set(x + col, y + row);
            }
        }
    }
}

std::optional<ArrangeResult> arrange_on_pixel_grid(
    const BoundingBox2crd& bed,
    std::vector<ArrangeItem>& items,
    const std::vector<ArrangeItem>& fixed_items,
    const Settings& settings,
    const StopCondition& stop_condition
)
{
    const Vec2crd pitch{
        std::max<coord_t>(scaled(settings.unscaled_pixel_width), 1),
        std::max<coord_t>(scaled(settings.unscaled_pixel_height), 1)
    };
    const double largest_pixel{std::max(settings.unscaled_pixel_width, settings.unscaled_pixel_height)};
    const int stride{std::max(1, int(std::lround(settings.unscaled_pixel_search_step / largest_pixel)))};

    // The pixel grid is anchored at pixel_grid_origin (the corner of the printer's
    // display), the bed contour itself is usually shrunk by the spacing and safety offsets.
    const Vec2crd anchor{settings.pixel_grid_origin ? scaled(*settings.pixel_grid_origin) : Vec2crd{0, 0}};
    const Vec2crd first_cell{floor_div(bed.min - anchor, pitch)};
    const Vec2crd last_cell{floor_div(bed.max - anchor, pitch)};
    const Vec2crd grid_origin{anchor + first_cell.cwiseProduct(pitch)};
    PixelMask occupancy{
        grid_origin,
        pitch,
        int(last_cell.x() - first_cell.x() + 1),
        int(last_cell.y() - first_cell.y() + 1)
    };
    // Block the cells which are not completely inside the bed.
    for (int y{}; y < occupancy.height(); ++y) {
        const coord_t y0{grid_origin.y() + y * pitch.y()};
        if (y0 < bed.min.y() || y0 + pitch.y() > bed.max.y()) {
            occupancy.set_span(y, 0, occupancy.width() - 1);
            continue;
        }
        for (int x{}; x < occupancy.width(); ++x) {
            const coord_t x0{grid_origin.x() + x * pitch.x()};
            if (x0 < bed.min.x() || x0 + pitch.x() > bed.max.x()) {
                occupancy.set(x, y);
            }
        }
    }
    const auto cell_of{[&](const PixelMask& mask) {
        return Vec2crd{(mask.origin() - grid_origin).cwiseQuotient(pitch)};
    }};

    for (const ArrangeItem& fixed : fixed_items) {
        const PixelMask mask{PixelMask::rasterize_convex(fixed.fixed_shape().transformed_outline(), pitch, grid_origin)};
        const Vec2crd cell{cell_of(mask)};
        occupancy.stamp(mask, int(cell.x()), int(cell.y()));
    }

    std::vector<std::size_t> order(items.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return items[a].fixed_shape().area_unscaled() > items[b].fixed_shape().area_unscaled();
    });

    struct Spot
    {
        std::pair<coord_t, coord_t> key;
        double rotation;
        Vec2crd cell;
        PixelMask fixed_mask;
        Vec2crd translation;
    };

    ArrangeResult result;
    for (const std::size_t idx : order) {
        if (stop_condition()) {
            return std::nullopt;
        }
        ArrangeItem& item{items[idx]};
        const double base_rotation{item.get_rotation()};
        const Vec2crd base_translation{item.get_translation()};

        std::optional<Spot> best;
        for (const double rotation : candidate_rotations(item, settings)) {
            item.set_rotation(base_rotation + rotation);
            item.set_translation(base_translation);
            const PixelMask movable{
                PixelMask::rasterize_convex(item.movable_shape().transformed_outline(), pitch, grid_origin)
            };
            const std::optional<Vec2crd> cell{find_lowest_left(occupancy, movable, stride)};
            if (!cell) {
                continue;
            }
            // Prefer the spot that keeps the pile lowest, then narrowest.
            const std::pair<coord_t, coord_t> key{cell->y() + movable.height(), cell->x() + movable.width()};
            if (best && best->key <= key) {
                continue;
            }
            const Vec2crd translation{base_translation + grid_origin + cell->cwiseProduct(pitch) - movable.origin()};
            PixelMask fixed_mask{PixelMask::rasterize_convex(item.fixed_shape().transformed_outline(), pitch, grid_origin)};
            best = Spot{key, rotation, *cell + cell_of(fixed_mask) - cell_of(movable), std::move(fixed_mask), translation};
        }

        if (!best) {
            item.set_rotation(base_rotation);
            item.set_translation(base_translation);
            result.not_packed.push_back(item);
            continue;
        }
        occupancy.stamp(best->fixed_mask, int(best->cell.x()), int(best->cell.y()));
        item.set_rotation(base_rotation + best->rotation);
        item.set_translation(best->translation);
        result.packed.push_back(item);
    }

    if (fixed_items.empty()) {
        align_pile_to_pixels(result.packed, bed, settings.bed_pivot_point.value_or(PivotPoint::Center), pitch);
    }
    return result;
}

} // namespace Slic3r::Biz::Arrange
