#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numbers>
#include <random>
#include "Slic3r/Biz/Algorithms/BoundingBox.hpp"
#include "Slic3r/Biz/Algorithms/ClipperUtils.hpp"
#include "Slic3r/Biz/Algorithms/Scaling.hpp"
#include "Slic3r/Biz/Arrange/Arrange.hpp"
#include "Slic3r/Biz/Arrange/PixelGrid.hpp"

using Slic3r::Biz::Algorithms::BoundingBox::to_polygon;
using Slic3r::Biz::Algorithms::ClipperUtils::intersection;
using Slic3r::Biz::Algorithms::ClipperUtils::offset;
using Slic3r::Biz::Algorithms::Scaling::scaled;
using Slic3r::Biz::Arrange::arrange;
using Slic3r::Biz::Arrange::ArrangeItem;
using Slic3r::Biz::Arrange::ArrangeResult;
using Slic3r::Biz::Arrange::GeometryHandling;
using Slic3r::Biz::Arrange::InputShape;
using Slic3r::Biz::Arrange::PixelMask;
using Slic3r::Biz::Arrange::Settings;
using Slic3r::Biz::Arrange::Strategy;
using Slic3r::Biz::Arrange::to_arrange_items;
using Slic3r::Domain::BoundingBox2crd;
using Slic3r::Domain::coord_t;
using Slic3r::Domain::ElementRef;
using Slic3r::Domain::ExPolygon;
using Slic3r::Domain::Points;
using Slic3r::Domain::Polygon;
using Slic3r::Domain::Polygons;
using Slic3r::Domain::Vec2crd;
using Slic3r::Domain::Vec2d;

namespace {
const auto never_stop{[]() { return false; }};

InputShape
rectangle(const double w, const double h, const Vec2d& at = {0., 0.}, const double rotation = 0.)
{
    ExPolygon poly{
        scaled(Vec2d{0., 0.}),
        scaled(Vec2d{w, 0.}),
        scaled(Vec2d{w, h}),
        scaled(Vec2d{0., h}),
    };
    poly.rotate(rotation);
    poly.translate(scaled(at));
    return {ElementRef{}, {poly}};
}

// An L shaped outline, its notch can host smaller items.
InputShape l_shape(const double size, const double arm)
{
    return {
        ElementRef{},
        {ExPolygon{
            scaled(Vec2d{0., 0.}),
            scaled(Vec2d{size, 0.}),
            scaled(Vec2d{size, arm}),
            scaled(Vec2d{arm, arm}),
            scaled(Vec2d{arm, size}),
            scaled(Vec2d{0., size}),
        }}
    };
}

Settings pixel_settings(const double pitch)
{
    Settings settings;
    settings.strategy              = Strategy::PixelGrid;
    settings.unscaled_pixel_width  = pitch;
    settings.unscaled_pixel_height = pitch;
    return settings;
}

Points bed_points(const BoundingBox2crd& bb)
{
    return to_polygon(bb).points;
}

bool is_collision_free(const std::vector<ArrangeItem>& items)
{
    for (std::size_t i{}; i < items.size(); ++i) {
        for (std::size_t j{i + 1}; j < items.size(); ++j) {
            const Polygons a{offset(
                items[i].fixed_shape().transformed_outline(),
                -float(Slic3r::Domain::SCALED_EPSILON)
            )};
            const Polygons b{offset(
                items[j].fixed_shape().transformed_outline(),
                -float(Slic3r::Domain::SCALED_EPSILON)
            )};
            if (!intersection(a, b).empty()) {
                return false;
            }
        }
    }
    return true;
}

bool is_on_pixel_grid(const Vec2crd& translation, const Vec2crd& pitch)
{
    return translation.x() % pitch.x() == 0 && translation.y() % pitch.y() == 0;
}

bool is_on_pixel_grid(const Vec2crd& translation, const coord_t pitch)
{
    return is_on_pixel_grid(translation, Vec2crd{pitch, pitch});
}
} // namespace

TEST_CASE("Convex rasterization covers the whole polygon", "[Arrange][PixelGrid]")
{
    const coord_t pitch{scaled(0.5)};
    const Vec2crd pitch2{pitch, pitch};
    const Polygon
        triangle{scaled(Vec2d{0.2, 0.3}), scaled(Vec2d{9.7, 1.1}), scaled(Vec2d{4.0, 8.9})};
    const PixelMask mask{PixelMask::rasterize_convex({triangle}, pitch2, Vec2crd{0, 0})};

    CHECK(mask.origin() == Vec2crd{0, 0});
    // Every vertex and the centroid must land on a set cell.
    for (const Vec2crd& p : triangle.points) {
        CHECK(mask.get(
            int((p.x() - mask.origin().x()) / pitch),
            int((p.y() - mask.origin().y()) / pitch)
        ));
    }
    const double real_area{std::abs(triangle.area()) / double(pitch) / double(pitch)};
    CHECK(double(mask.count()) >= real_area);
    CHECK(double(mask.count()) < real_area * 1.6);
}

TEST_CASE("Pixel mask fit test respects occupied cells and borders", "[Arrange][PixelGrid]")
{
    PixelMask grid{Vec2crd{0, 0}, Vec2crd{1, 1}, 130, 10};
    PixelMask block{Vec2crd{0, 0}, Vec2crd{1, 1}, 3, 2};
    block.set_span(0, 0, 2);
    block.set_span(1, 0, 2);

    CHECK(grid.fits(block, 0, 0));
    CHECK(grid.fits(block, 127, 8));
    CHECK_FALSE(grid.fits(block, 128, 0));
    CHECK_FALSE(grid.fits(block, 0, 9));

    // Crossing a 64 bit word boundary.
    grid.set(64, 5);
    CHECK_FALSE(grid.fits(block, 62, 4));
    CHECK_FALSE(grid.fits(block, 64, 5));
    CHECK(grid.fits(block, 65, 5));
    CHECK(grid.fits(block, 61, 5));
}

TEST_CASE("Pixel grid arrange snaps every translation to whole pixels", "[Arrange][PixelGrid]")
{
    for (const double pitch : {0.05, 0.5}) {
        Settings settings{pixel_settings(pitch)};
        settings.scaled_offset = scaled(1.0);
        std::vector<InputShape> shapes;
        std::mt19937 gen{42};
        std::uniform_real_distribution<double> size{3., 25.};
        for (int i{}; i < 15; ++i) {
            shapes.push_back(rectangle(size(gen), size(gen), {13.37, -4.2}, 0.3 * i));
        }
        std::vector<ArrangeItem> items{to_arrange_items(shapes, settings)};
        const BoundingBox2crd bed{scaled(Vec2d{0., 0.}), scaled(Vec2d{160., 100.})};
        const std::optional<ArrangeResult> result{
            arrange(bed_points(bed), items, {}, settings, never_stop)
        };
        REQUIRE(result);
        CHECK(result->not_packed.empty());
        CHECK(is_collision_free(result->packed));
        for (const ArrangeItem& item : result->packed) {
            CHECK(is_on_pixel_grid(item.get_translation(), scaled(pitch)));
            CHECK(bed.contains(item.fixed_shape().bounding_box()));
        }
    }
}

TEST_CASE("Pixel grid arrange turns items to their minimum footprint", "[Arrange][PixelGrid]")
{
    const Settings settings{pixel_settings(0.1)};
    std::vector<ArrangeItem> items{
        to_arrange_items({rectangle(40., 5., {}, std::numbers::pi / 6.)}, settings)
    };
    const BoundingBox2crd bed{scaled(Vec2d{0., 0.}), scaled(Vec2d{100., 100.})};
    const std::optional<ArrangeResult> result{
        arrange(bed_points(bed), items, {}, settings, never_stop)
    };
    REQUIRE(result);
    REQUIRE(result->packed.size() == 1);

    const Vec2crd sizes{Slic3r::Biz::Algorithms::BoundingBox::sizes(
        result->packed.front().fixed_shape().bounding_box()
    )};
    const double area{
        double(sizes.x()) * double(sizes.y()) / double(scaled(1.)) / double(scaled(1.))
    };
    CHECK_THAT(area, Catch::Matchers::WithinRel(200., 0.01));
}

TEST_CASE("Pixel grid arrange fills the notch of concave items", "[Arrange][PixelGrid]")
{
    Settings settings{pixel_settings(0.5)};
    settings.fixed_geometry           = GeometryHandling::Arbitrary;
    settings.movable_geometry         = GeometryHandling::Arbitrary;
    settings.pixel_grid_min_footprint = false;

    std::vector<ArrangeItem> items{
        to_arrange_items({l_shape(60., 10.), rectangle(30., 30.)}, settings)
    };
    const BoundingBox2crd bed{scaled(Vec2d{0., 0.}), scaled(Vec2d{70., 70.})};
    const std::optional<ArrangeResult> result{
        arrange(bed_points(bed), items, {}, settings, never_stop)
    };
    REQUIRE(result);
    CHECK(result->packed.size() == 2);
    CHECK(is_collision_free(result->packed));
}

TEST_CASE("Pixel grid arrange reports items that do not fit", "[Arrange][PixelGrid]")
{
    const Settings settings{pixel_settings(0.5)};
    std::vector<InputShape> shapes(5, rectangle(45., 45.));
    std::vector<ArrangeItem> items{to_arrange_items(shapes, settings)};
    const BoundingBox2crd bed{scaled(Vec2d{0., 0.}), scaled(Vec2d{100., 100.})};
    const std::optional<ArrangeResult> result{
        arrange(bed_points(bed), items, {}, settings, never_stop)
    };
    REQUIRE(result);
    CHECK(result->packed.size() == 4);
    CHECK(result->not_packed.size() == 1);
}

TEST_CASE("Pixel grid arrange keeps away from fixed items", "[Arrange][PixelGrid]")
{
    Settings settings{pixel_settings(0.5)};
    const std::vector<ArrangeItem> fixed{
        to_arrange_items({rectangle(50., 50., {0., 0.})}, settings)
    };
    std::vector<ArrangeItem> items{
        to_arrange_items({rectangle(40., 40.), rectangle(40., 40.)}, settings)
    };
    const BoundingBox2crd bed{scaled(Vec2d{0., 0.}), scaled(Vec2d{100., 100.})};
    const std::optional<ArrangeResult> result{
        arrange(bed_points(bed), items, fixed, settings, never_stop)
    };
    REQUIRE(result);
    REQUIRE(result->packed.size() == 2);

    std::vector<ArrangeItem> all{result->packed};
    all.push_back(fixed.front());
    CHECK(is_collision_free(all));
}

TEST_CASE("Pixel grid arrange works with a shifted grid origin and bed", "[Arrange][PixelGrid]")
{
    Settings settings{pixel_settings(0.05)};
    settings.pixel_grid_origin = Vec2d{0.02, 0.03};
    std::vector<ArrangeItem> items{
        to_arrange_items({rectangle(10., 10.), rectangle(20., 7.)}, settings)
    };
    const BoundingBox2crd bed{scaled(Vec2d{0.91, 1.37}), scaled(Vec2d{50., 50.})};
    const std::optional<ArrangeResult> result{
        arrange(bed_points(bed), items, {}, settings, never_stop)
    };
    REQUIRE(result);
    REQUIRE(result->packed.size() == 2);
    CHECK(is_collision_free(result->packed));
    for (const ArrangeItem& item : result->packed) {
        CHECK(is_on_pixel_grid(item.get_translation(), scaled(0.05)));
        CHECK(bed.contains(item.fixed_shape().bounding_box()));
    }
}

TEST_CASE("Pixel grid arrange supports non square pixels", "[Arrange][PixelGrid]")
{
    Settings settings{pixel_settings(0.05)};
    settings.unscaled_pixel_width = 0.035;
    std::vector<InputShape> shapes(6, rectangle(12.3, 8.7, {1.111, 2.222}));
    std::vector<ArrangeItem> items{to_arrange_items(shapes, settings)};
    const BoundingBox2crd bed{scaled(Vec2d{0., 0.}), scaled(Vec2d{80., 60.})};
    const std::optional<ArrangeResult> result{
        arrange(bed_points(bed), items, {}, settings, never_stop)
    };
    REQUIRE(result);
    CHECK(result->packed.size() == 6);
    CHECK(is_collision_free(result->packed));
    for (const ArrangeItem& item : result->packed) {
        CHECK(is_on_pixel_grid(item.get_translation(), Vec2crd{scaled(0.035), scaled(0.05)}));
    }
}
