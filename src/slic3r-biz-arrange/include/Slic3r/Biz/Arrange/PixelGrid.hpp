#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "Slic3r/Biz/Arrange/Arrange.hpp"
#include "Slic3r/Domain/Polygon.hpp"

namespace Slic3r::Biz::Arrange {

// Binary raster aligned to a pixel grid. Cell (0, 0) has its lower left corner at `origin`
// (scaled coordinates) and every cell is `pitch.x()` scaled units wide and `pitch.y()` tall.
class PixelMask
{
public:
    PixelMask() = default;
    PixelMask(const Domain::Vec2crd& origin, const Domain::Vec2crd& pitch, int width, int height);

    // Mark every cell whose square touches one of the convex polygons, so the
    // raster always covers the real outline.
    static PixelMask rasterize_convex(
        const Domain::Polygons& convex_polygons,
        const Domain::Vec2crd& pitch,
        const Domain::Vec2crd& grid_origin
    );

    int width() const { return m_width; }
    int height() const { return m_height; }
    const Domain::Vec2crd& origin() const { return m_origin; }
    const Domain::Vec2crd& pitch() const { return m_pitch; }

    bool get(int x, int y) const;
    void set(int x, int y);
    void set_span(int y, int x_from, int x_to);
    std::size_t count() const;

    // True if no set cell of `mask` lands on a set cell of this raster when the
    // mask's cell (0, 0) is put at cell (x, y). Cells outside of this raster count as set.
    bool fits(const PixelMask& mask, int x, int y) const;
    void stamp(const PixelMask& mask, int x, int y);

private:
    std::uint64_t word(int y, int w) const { return m_bits[std::size_t(y) * m_words + w]; }

    Domain::Vec2crd m_origin{0, 0};
    Domain::Vec2crd m_pitch{1, 1};
    int m_width{0};
    int m_height{0};
    int m_words{0};
    std::vector<std::uint64_t> m_bits;
};

// Packs the items onto a rectangular bed so that every translation is a whole
// number of pixels. Items are put at their minimum footprint rotation (when
// enabled in settings) and pushed towards the lower left corner.
std::optional<ArrangeResult> arrange_on_pixel_grid(
    const Domain::BoundingBox2crd& bed,
    std::vector<ArrangeItem>& items,
    const std::vector<ArrangeItem>& fixed_items,
    const Settings& settings,
    const StopCondition& stop_condition
);

} // namespace Slic3r::Biz::Arrange
